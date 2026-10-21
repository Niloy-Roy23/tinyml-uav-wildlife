
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
)
MODEL_PATH = ROOT / "models" / "mfcc_mlp.keras"
BEST_PATH = ROOT / "models" / "mfcc_mlp_best.keras"
NORM_PATH = ROOT / "models" / "mfcc_normalization.npz"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30


def main():
    tf.keras.utils.set_random_seed(SEED)

    # Load the MFCC dataset.
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        splits = data["splits"]

    train_mask = splits == "train"
    val_mask = splits == "validation"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training or validation split is empty.")

    # Calculate normalization parameters from training data only.
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)

    # Save parameters for consistent inference on the Raspberry Pi.
    np.savez(
        NORM_PATH,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    print("Normalization parameters saved:", NORM_PATH)
    print("Normalization shape:", mean.shape)

    # Apply the same normalization to training and validation data.
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Load the existing model architecture.
    model = keras.models.load_model(MODEL_PATH)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the final model and report the best validation epoch.
    model.save(MODEL_PATH)

    best_epoch = (
        int(np.argmax(history.history["val_accuracy"])) + 1
    )
    best_val_accuracy = float(
        max(history.history["val_accuracy"])
    )

    print("\nTraining complete.")
    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("Best model:", BEST_PATH)
    print("Final model:", MODEL_PATH)
    print("Normalization file:", NORM_PATH)


if __name__ == "__main__":
    main()
