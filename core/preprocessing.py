import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

REQUIRED_COLUMNS = [
    "duration",
    "protocol_type",
    "src_bytes",
    "dst_bytes",
    "failed_logins",
    "logged_in",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "attack_type",
]

def preprocess_training_data(path, scaler_path):
    df = pd.read_csv(path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Validate schema
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
            
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Encode categorical features
    protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
    if "protocol_type" in df.columns:
        if df["protocol_type"].dtype == object:
            df["protocol_type"] = df["protocol_type"].map(protocol_map)
            
    # Ensure no NaN after mapping
    df["protocol_type"] = df["protocol_type"].fillna(0)
    
    X = df.drop("attack_type", axis=1)
    y = df["attack_type"].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, scaler_path)
    
    return X_scaled, y