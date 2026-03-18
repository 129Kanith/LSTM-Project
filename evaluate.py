import numpy as np
import pandas as pd
import joblib
import config
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from core.sequence_builder import create_sequences
from inference import threatInference, ATTACK_LABELS, FEATURE_COLS, PROTOCOL_MAP

# -------------------------
# 1. Load Engine
# -------------------------
engine = threatInference()

# -------------------------
# 2. Load Dataset
# -------------------------
print("Loading offline dataset for verification...")
df = pd.read_csv(r"K:\LSTM\data\offline_dataset.csv")

# Ensure feature consistency and encode protocol
df["protocol_type"] = df["protocol_type"].str.lower().map(PROTOCOL_MAP)

for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0

X = df[FEATURE_COLS].copy().fillna(0)
y = df["attack_type"].values

# -------------------------
# 3. Scale & Sequence
# -------------------------
print("Preprocessing and building sequences...")
X_scaled = engine.scaler.transform(X)
X_seq, y_seq = create_sequences(X_scaled, y, config.TIME_STEPS)

# -------------------------
# 4. Predict
# -------------------------
print(f"Running inference on {len(X_seq)} points...")
predictions = engine.model.predict(X_seq, verbose=1)
pred_classes = np.argmax(predictions, axis=1)

# -------------------------
# 5. Evaluation Metrics
# -------------------------
accuracy = accuracy_score(y_seq, pred_classes)
matrix = confusion_matrix(y_seq, pred_classes)

# Use actual class names in the report
unique_y = np.unique(y_seq)
target_names = [ATTACK_LABELS[i] for i in sorted(ATTACK_LABELS.keys()) if i in unique_y]

report = classification_report(y_seq, pred_classes, target_names=target_names)

print("\n" + "="*30)
print("     MODEL EVALUATION REPORT")
print("="*30)
print(f"Overall Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
print(matrix)
print("\nDetailed Classification Report:")
print(report)
print("="*30)