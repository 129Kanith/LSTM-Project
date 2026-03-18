import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import config
from inference import threatInference, ATTACK_LABELS, FEATURE_COLS

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="SOC Command Center",
    layout="wide",
    page_icon="🛡️"
)

# ------------------------------------------------
# DARK SOC THEME & CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>
/* Base Dark Theme */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}
/* Top Padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
/* Custom Metric Cards */
div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
div[data-testid="stMetricValue"] {
    font-size: 32px;
    font-weight: bold;
    color: #00E5FF;
}
div[data-testid="stMetricDelta"] {
    font-size: 16px;
}
/* Log box */
.stCodeBlock {
    border: 1px solid #333;
    background-color: #1A1C23 !important;
}
header[data-testid="stHeader"] {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.title("SOC Command Center")
st.caption("AI-Driven Real-Time Network Intrusion Monitoring")

# ------------------------------------------------
# LOAD ENGINE
# ------------------------------------------------
@st.cache_resource
def load_engine():
    return threatInference()

engine = load_engine()

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload Network Log Dataset or Connect Stream", type=["csv"])

if uploaded_file:
    df_stream = pd.read_csv(uploaded_file)

    protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}

    if "protocol_type" in df_stream.columns and df_stream["protocol_type"].dtype == object:
        df_stream["protocol_type"] = df_stream["protocol_type"].map(protocol_map)

    # Ensure no NaN after mapping
    df_stream["protocol_type"] = df_stream["protocol_type"].fillna(0)

    # Bridge distinct dataset features
    if "same_srv_rate" in df_stream.columns and "logged_in" not in df_stream.columns:
        df_stream["logged_in"] = df_stream["same_srv_rate"].apply(lambda x: 1 if x > 0.5 else 0)
    
    if "diff_srv_rate" in df_stream.columns and "serror_rate" not in df_stream.columns:
        df_stream["serror_rate"] = df_stream["diff_srv_rate"]
        
    if "diff_srv_rate" in df_stream.columns and "srv_serror_rate" not in df_stream.columns:
        df_stream["srv_serror_rate"] = df_stream["diff_srv_rate"]

    feature_cols = [
        "duration", "protocol_type", "src_bytes", "dst_bytes", 
        "failed_logins", "logged_in", "count", "srv_count", 
        "serror_rate", "srv_serror_rate"
    ]
    
    for col in feature_cols:
        if col not in df_stream.columns:
            df_stream[col] = 0

    df_stream.fillna(0, inplace=True)

    # ------------------------------------------------
    # SOC LAYOUT PLACEHOLDERS
    # ------------------------------------------------
    
    # Top KPI Row
    st.markdown("### Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    packets_metric = kpi1.empty()
    attacks_metric = kpi2.empty()
    rate_metric = kpi3.empty()
    status_metric = kpi4.empty()
    
    st.markdown("---")
    
    # Middle Row: Charts
    col_trend, col_dist = st.columns([2, 1])
    with col_trend:
        st.markdown("#### Event Volume & Threat Timeline")
        attack_timeline = st.empty()
    with col_dist:
        st.markdown("#### Threat Distribution")
        distribution_chart = st.empty()

    # Bottom Row: Traffic & Logs
    st.markdown("---")
    bottom1, bottom2 = st.columns([2, 1])
    with bottom1:
        st.markdown("#### Live Network Traffic (Bandwidth)")
        traffic_chart = st.empty()
    with bottom2:
        st.markdown("#### Live Incident Feed")
        log_box = st.empty()
        alert_box = st.empty()

    # ------------------------------------------------
    # MONITORING VARIABLES
    # ------------------------------------------------
    buffer = []
    logs = []
    attack_history = []
    traffic_history = []

    packet_counter = 0
    attack_counter = 0

    attack_distribution = {
        "Normal": 0, "DDoS": 0, "Brute Force": 0,
        "Probe": 0, "R2L": 0, "U2R": 0
    }

    # ------------------------------------------------
    # START MONITORING LOGIC
    # ------------------------------------------------
    if st.button("Start Live Monitoring", type="primary"):
        
        for i in range(len(df_stream)):
            row = df_stream.iloc[[i]] # Get the row as a DataFrame to keep shape and names
            # Map required columns explicitly without dropping names
            feature_df = row[feature_cols].copy()
            features_scaled = engine.scaler.transform(feature_df)
            buffer.append(features_scaled[0])
            packet_counter += 1

            # Simulated traffic load
            if "src_bytes" in row.columns and "dst_bytes" in row.columns:
                traffic = int(row["src_bytes"].iloc[0]) + int(row["dst_bytes"].iloc[0])
            else:
                traffic = np.random.randint(100, 500)
            traffic_history.append(traffic)

            if len(buffer) >= config.TIME_STEPS:
                # Use engine for prediction
                # Create a temporary df for sequence processing if needed, 
                # but since app.py does real-time, we keep manual sequence builder here or 
                # add a 'predict_next' to engine. 
                # Let's keep it simple: use engine.model directly but with engine data standards.
                
                sequence = np.array(buffer[-config.TIME_STEPS:])
                sequence = sequence.reshape(1, config.TIME_STEPS, -1)

                prediction = engine.model.predict(sequence, verbose=0)[0]
                
                # Top Class logic
                top_idx = int(np.argmax(prediction))
                top_prob = float(prediction[top_idx])
                predicted_label = ATTACK_LABELS[top_idx]
                
                # Raw probability string for dist
                dist_str = " | ".join([f"{ATTACK_LABELS[idx]}:{prob:.22f}" for idx, prob in enumerate(prediction)])
                
                # Decision Thresholding
                if top_prob > 0.45 and top_idx != 0:
                    alert_level = "Confirmed Threat"
                    chart_label = predicted_label
                else:
                    alert_level = "Normal"
                    predicted_label = "Normal"
                    chart_label = "Normal"

                attack_history.append({"packet": packet_counter, "class": top_idx if alert_level != "Normal" else 0, "label": chart_label})
                attack_distribution[chart_label] += 1

                if alert_level == "Confirmed Threat":
                    attack_counter += 1
                    logs.append(f"[ALERT] 🚨 Threat: {predicted_label} | Confidence {top_prob:.2f} | Dist: [{dist_str}]")
                    alert_box.error(f"🚨 **Confirmed Threat:** {predicted_label} | Confidence: {top_prob:.2f}")
                else:
                    logs.append(f"[INFO] Pkt {packet_counter} ✅ Normal | Confidence {top_prob:.2f} | Dist: [{dist_str}]")
                    alert_box.success("✅ **Normal Traffic**")

                # Update KPIs
                attack_rate = (attack_counter / packet_counter) * 100
                packets_metric.metric("Packets Processed", f"{packet_counter:,}", delta=f"+1", delta_color="normal")
                attacks_metric.metric("Total Threats", f"{attack_counter:,}", delta=f"+1" if alert_level != "Normal" else "0", delta_color="inverse")
                rate_metric.metric("Threat Rate", f"{attack_rate:.1f}%")
                status_metric.metric("SOC Status", "Monitoring 🟢")

                # Update Charts (Throttled to update every 5 packets for performance)
                if len(attack_history) > 0 and len(attack_history) % 5 == 0:
                    
                    # 1. Timeline Chart (Plotly)
                    recent_attacks = pd.DataFrame(attack_history[-150:])
                    fig_timeline = px.scatter(
                        recent_attacks, x="packet", y="label", 
                        color="class",
                        color_continuous_scale="Reds",
                        template="plotly_dark",
                        height=300
                    )
                    fig_timeline.update_layout(
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis_title="Packet Window",
                        yaxis_title="",
                        coloraxis_showscale=False
                    )
                    attack_timeline.plotly_chart(fig_timeline, key=f"timeline_{packet_counter}")

                    # 2. Distribution Chart (Plotly)
                    dist_df = pd.DataFrame({
                        "Attack Type": list(attack_distribution.keys()),
                        "Count": list(attack_distribution.values())
                    })
                    # Exclude Normal traffic to focus on threats
                    threat_df = dist_df[dist_df["Attack Type"] != "Normal"]
                    fig_dist = px.bar(
                        threat_df, x="Attack Type", y="Count",
                        color="Attack Type",
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        height=300
                    )
                    fig_dist.update_layout(
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False
                    )
                    distribution_chart.plotly_chart(fig_dist, key=f"dist_{packet_counter}")

                    # 3. Traffic Area Chart (Plotly)
                    recent_traffic = traffic_history[-100:]
                    fig_traffic = px.area(
                        x=list(range(len(recent_traffic))), y=recent_traffic,
                        template="plotly_dark", 
                        color_discrete_sequence=["#00E5FF"],
                        height=250
                    )
                    fig_traffic.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Recent Packets", 
                        yaxis_title="Bytes"
                    )
                    traffic_chart.plotly_chart(fig_traffic, key=f"traffic_{packet_counter}")

                # Update Log Stream
                log_box.code("\n".join(logs[-12:]))

            time.sleep(0.05) # Sped up slightly for smoother SOC feel

        status_metric.metric("SOC Status", "Completed ⚪")