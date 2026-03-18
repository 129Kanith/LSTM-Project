import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from core import sequence_builder
import config

# Define standardized attack labels
ATTACK_LABELS = {
    0: "Normal",
    1: "DDoS",
    2: "Brute Force",
    3: "Probe",
    4: "R2L",
    5: "U2R"
}

# Standard feature columns expected by the model
FEATURE_COLS = [
    "duration", "protocol_type", "src_bytes", "dst_bytes",
    "failed_logins", "logged_in", "count", "srv_count",
    "serror_rate", "srv_serror_rate"
]

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

class threatInference:
    def __init__(self, model_path=None, scaler_path=None):
        if model_path is None:
            model_path = f"models/lstm_{config.MODEL_VERSION}.h5"
        if scaler_path is None:
            scaler_path = "models/scaler_v1.pkl"
            
        print(f"Loading model: {model_path}")
        self.model = load_model(model_path)
        
        print(f"Loading scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

    def preprocess_row(self, row_dict):
        """Prep a single dictionary input for the model."""
        df = pd.DataFrame([row_dict])
        
        # 1. Map protocol
        if "protocol_type" in df.columns:
            val = df.at[0, "protocol_type"]
            if isinstance(val, str):
                df["protocol_type"] = df["protocol_type"].str.lower().map(PROTOCOL_MAP)
        
        # 2. Ensure all features exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0
        
        # 3. Clean and scale
        X = df[FEATURE_COLS].copy().fillna(0)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict_single(self, features_dict):
        """Predict for a single packet by creating a static sequence."""
        X_scaled = self.preprocess_row(features_dict)
        
        # Build sequence of length config.TIME_STEPS
        X_seq = np.tile(X_scaled, (config.TIME_STEPS, 1))
        X_seq = np.expand_dims(X_seq, axis=0)

        prediction = self.model.predict(X_seq, verbose=0)[0]
        top_idx = int(np.argmax(prediction))
        
        return {
            "prediction": ATTACK_LABELS.get(top_idx, "Unknown"),
            "confidence": float(prediction[top_idx]),
            "distribution": {ATTACK_LABELS[i]: float(p) for i, p in enumerate(prediction)}
        }

    def predict_dataframe(self, df):
        """Predict for an entire dataframe using sliding windows."""
        # Preprocess features
        working_df = df.copy()
        if "protocol_type" in working_df.columns and working_df["protocol_type"].dtype == object:
            working_df["protocol_type"] = working_df["protocol_type"].str.lower().map(PROTOCOL_MAP)
            
        for col in FEATURE_COLS:
            if col not in working_df.columns:
                working_df[col] = 0
                
        X = working_df[FEATURE_COLS].copy().fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Build sequences
        X_seq, _ = sequence_builder.create_sequences(X_scaled, np.zeros(len(X_scaled)), config.TIME_STEPS)
        
        # Inference
        probs = self.model.predict(X_seq, verbose=0)
        indices = np.argmax(probs, axis=1)
        
        results = [ATTACK_LABELS.get(idx) for idx in indices]
        
        # Pad results
        padding = ["Pending Sequence..."] * (len(df) - len(results))
        return padding + results

if __name__ == "__main__":
    # Quick test
    engine = threatInference()
    test_data = {
        "duration": 0, "protocol_type": "tcp", "src_bytes": 100, "dst_bytes": 100,
        "failed_logins": 5, "logged_in": 0, "count": 10, "srv_count": 5,
        "serror_rate": 0, "srv_serror_rate": 0
    }
    print(engine.predict_single(test_data))
