
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
MODEL_PATH = ROOT / "models" / "mfcc_mlp_best.keras"

CLASS_NAMES = ["bird", "insect", "background"]


def main():
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    splits = data["splits"]

    train_mask = splits == "train"
    test_mask = splits == "test"

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    mean = X_train.mean(axis=0, keepdims=True)
    std = np.maximum(X_train.std(axis=0, keepdims=True), 1e-6)
    X_test = (X_test - mean) / std

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("Model:", MODEL_PATH.name)
    print("Test windows:", len(y_test))
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))


if __name__ == "__main__":
    main()
