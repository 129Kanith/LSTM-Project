from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam


def build_model(time_steps, feature_count, num_classes, lr):

    model = Sequential()

    # Input Layer explicitly defined
    model.add(Input(shape=(time_steps, feature_count)))

    # First LSTM layer
    model.add(
        LSTM(
            128,
            return_sequences=True
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.2)) # Slightly reduced dropout for better learning of patterns

    # Second LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Third LSTM layer (feature compression)
    model.add(LSTM(32))
    model.add(BatchNormalization()) # Added batch norm here
    model.add(Dropout(0.2))

    # Dense feature learning
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    
    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model