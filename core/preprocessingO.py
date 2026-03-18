import pandas as pd
import csv
import os

# FIXED PATH
DATA_PATH = r"K:\LSTM\data\offline_dataset.csv"

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
    "attack_type"
]

def detect_delimiter(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
    return delimiter

def check_dataset(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    print("\nDetecting delimiter...")
    delimiter = detect_delimiter(path)
    print(f"Detected delimiter: '{delimiter}'\n")

    print("Loading dataset...\n")

    df = pd.read_csv(
        path,
        delimiter=delimiter,
        on_bad_lines="skip",
        engine="python"
    )

    print("Dataset Loaded Successfully\n")

    print("Shape:", df.shape, "\n")

    print("Columns:")
    print(df.columns.tolist(), "\n")

    df.columns = df.columns.str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        print("❌ Missing Required Columns:", missing)
    else:
        print("✅ All required columns present\n")

    print("Missing Values:\n", df.isnull().sum(), "\n")
    print("Duplicate Rows:", df.duplicated().sum(), "\n")

    if "attack_type" in df.columns:
        print("Attack Distribution:\n", df["attack_type"].value_counts(), "\n")

    print("First 5 rows:\n", df.head())

if __name__ == "__main__":
    check_dataset(DATA_PATH)