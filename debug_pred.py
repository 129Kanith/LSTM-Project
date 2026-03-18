import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import config

print("Loading artifacts...")
model = load_model(f"models/lstm_{config.MODEL_VERSION}.h5")
scaler = joblib.load("models/scaler_v1.pkl")

df = pd.read_csv(r"K:\LSTM\data\10000.csv")

protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}

if "protocol_type" in df.columns and df["protocol_type"].dtype == object:
    df["protocol_type"] = df["protocol_type"].map(protocol_map)

df["protocol_type"] = df["protocol_type"].fillna(0)

# Bridge distinct dataset features
if "same_srv_rate" in df.columns and "logged_in" not in df.columns:
    df["logged_in"] = df["same_srv_rate"].apply(lambda x: 1 if x > 0.5 else 0)

if "diff_srv_rate" in df.columns and "serror_rate" not in df.columns:
    df["serror_rate"] = df["diff_srv_rate"]
    
if "diff_srv_rate" in df.columns and "srv_serror_rate" not in df.columns:
    df["srv_serror_rate"] = df["diff_srv_rate"]

feature_cols = [
    "duration", "protocol_type", "src_bytes", "dst_bytes", 
    "failed_logins", "logged_in", "count", "srv_count", 
    "serror_rate", "srv_serror_rate"
]

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0
            
df.fillna(0, inplace=True)

attack_labels = {
    0: "Normal",
    1: "DDoS",
    2: "Brute Force",
    3: "Probe",
    4: "R2L",
    5: "U2R"
}

print(f"Total rows: {len(df)}")
print("Processing data scaling...")
feature_df = df[feature_cols].copy()
features_scaled = scaler.transform(feature_df)

print("Building sequences...")
sequences = []
# Create sequences manually for batched predict
for i in range(len(features_scaled)):
    if i < config.TIME_STEPS - 1:
        # Buffer early packets identically as real-time stream would
        seq = np.zeros((config.TIME_STEPS, features_scaled.shape[1]))
        seq[config.TIME_STEPS - 1 - i:] = features_scaled[0:i + 1]
    else:
        seq = features_scaled[i - config.TIME_STEPS + 1 : i + 1]
    sequences.append(seq)
    
X_test = np.array(sequences)
print(f"Sequence shape: {X_test.shape}")

print("Running batch predictions...")
predictions = model.predict(X_test, batch_size=64)

attack_counter = 0
dist = {label: 0 for label in attack_labels.values()}

for i, pred in enumerate(predictions):
    top_idx = int(np.argmax(pred))
    top_prob = float(pred[top_idx])
    
    if top_prob > 0.45 and top_idx != 0:
        attack_counter += 1
        predicted_label = attack_labels[top_idx]
        dist[predicted_label] += 1
    else:
        dist["Normal"] += 1

print(f"Total threats detected in test: {attack_counter}")
print("Distribution details:")
for k, v in dist.items():
    print(f"  {k}: {v}")
