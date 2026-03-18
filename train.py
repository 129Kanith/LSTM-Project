import logging
import numpy as np
import tensorflow as tf
import config
from core import preprocessing, sequence_builder, model_builder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from core.metrics import evaluate_model

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Training started.")

# -------------------------------------------------
# 1. Load & Preprocess Dataset
# -------------------------------------------------
X_scaled, y = preprocessing.preprocess_training_data(
    r"K:\LSTM\data\offline_dataset.csv",
    r"K:\LSTM\models\scaler_v1.pkl"
)

logging.info(f"Dataset loaded with {len(X_scaled)} samples.")

# -------------------------------------------------
# Dataset sanity check
# -------------------------------------------------
unique, counts = np.unique(y, return_counts=True)
distribution = dict(zip(unique, counts))

print("\nDataset Class Distribution:")
print(distribution)

logging.info(f"Dataset class distribution: {distribution}")

# -------------------------------------------------
# 2. Create Temporal Sequences
# -------------------------------------------------
X_seq, y_seq = sequence_builder.create_sequences(
    X_scaled,
    y,
    config.TIME_STEPS
)

logging.info(f"Sequence dataset created with shape: {X_seq.shape}")

# -------------------------------------------------
# 3. Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    random_state=42,
    stratify=y_seq
)

logging.info("Dataset split into training and testing.")

# -------------------------------------------------
# 4. Compute Class Weights
# -------------------------------------------------
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights:")
print(class_weights)

logging.info(f"Class weights calculated: {class_weights}")

# -------------------------------------------------
# 5. Build Model
# -------------------------------------------------
model = model_builder.build_model(
    config.TIME_STEPS,
    X_seq.shape[2],
    len(classes),
    config.LEARNING_RATE
)

model.summary()

logging.info("Model architecture built.")

# -------------------------------------------------
# 6. Training Callbacks
# -------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=f"models/lstm_best_{config.MODEL_VERSION}.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# -------------------------------------------------
# 7. Train Model
# -------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    shuffle=True,
    verbose=1
)

logging.info("Model training completed.")

# -------------------------------------------------
# 8. Evaluate Model
# -------------------------------------------------
report, matrix = evaluate_model(model, X_test, y_test)

print("\n===== MODEL EVALUATION =====")
print(report)

print("\nConfusion Matrix:")
print(matrix)

logging.info("Model evaluation completed.")

# -------------------------------------------------
# 9. Save Final Model
# -------------------------------------------------
final_model_path = f"models/lstm_{config.MODEL_VERSION}.h5"

model.save(final_model_path)

logging.info(f"Final model saved at {final_model_path}")

print("\nModel trained and saved successfully.")