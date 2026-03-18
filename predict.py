import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from core import sequence_builder
import config


def predict_attack(csv_path, model_path, scaler_path):

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ---------------------------------------------------
    # Encode categorical features identically to training
    # ---------------------------------------------------
    protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}

    if "protocol_type" in df.columns:
        if df["protocol_type"].dtype == object:
            df["protocol_type"] = df["protocol_type"].map(protocol_map)

    # Ensure no NaN after mapping
    df["protocol_type"] = df["protocol_type"].fillna(0)

    # ---------------------------------------------------
    # Ensure required features exist
    # ---------------------------------------------------
    feature_cols = [
        "duration",
        "protocol_type",
        "src_bytes",
        "dst_bytes",
        "failed_logins",
        "logged_in",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate"
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].copy()

    # Fill NaN values
    X = X.fillna(0)

    # ---------------------------------------------------
    # Scale features
    # ---------------------------------------------------
    X_scaled = scaler.transform(X)

    # ---------------------------------------------------
    # Build LSTM sequences
    # ---------------------------------------------------
    X_seq, _ = sequence_builder.create_sequences(
        X_scaled,
        np.zeros(len(X_scaled)),
        config.TIME_STEPS
    )

    # ---------------------------------------------------
    # Run inference
    # ---------------------------------------------------
    print("Performing inference...")
    predictions = model.predict(X_seq, verbose=0)

    class_map = {
        0: "Normal",
        1: "DDoS",
        2: "Brute Force",
        3: "Probe",
        4: "R2L",
        5: "U2R"
    }

    results = []

    print("\nFirst 10 Predictions:\n")

    for i, probs in enumerate(predictions):

        # sort probabilities
        top_idx = np.argsort(probs)[::-1][:3]
        top_probs = probs[top_idx]

        top1_idx = top_idx[0]
        top1_prob = top_probs[0]

        # ---------------------------------------------------
        # Exact Predict Logic matches user requirement
        # always outputs: predicted attack name, confidence score, prob distribution
        # ---------------------------------------------------
        predicted_attack_name = class_map.get(top1_idx)
        confidence_score = float(top1_prob)
        
        # Build probability distribution string across all classes
        prob_distribution = {class_map[idx]: float(prob) for idx, prob in enumerate(probs)}

        result_obj = {
            "predicted_attack": predicted_attack_name,
            "confidence": confidence_score,
            "distribution": prob_distribution
        }
        
        results.append(result_obj)

        # ---------------------------------------------------
        # Debug Output
        # ---------------------------------------------------
        if i < 10:
            dist_str = " | ".join([f"{k}:{v:.2f}" for k, v in prob_distribution.items()])
            print(f"Sequence {i+1}: {predicted_attack_name} | Confidence: {confidence_score:.2f} | Dist: [{dist_str}]")

    return results


if __name__ == "__main__":

    DATA_PATH = r"K:\LSTM\data\offline_dataset.csv"

    MODEL_PATH = f"models/lstm_{config.MODEL_VERSION}.h5"

    SCALER_PATH = f"models/scaler_v1.pkl"

    predict_attack(DATA_PATH, MODEL_PATH, SCALER_PATH)