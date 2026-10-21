
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SHAPE = (40, 87)
NUM_CLASSES = 3

def build_model():
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE, name="mfcc_input"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="mfcc_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()

    output_path = MODEL_DIR / "mfcc_mlp.keras"
    model.save(output_path)

    print("\nModel created successfully.")
    print("Saved to:", output_path)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Parameters:", model.count_params())
