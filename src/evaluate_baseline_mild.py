from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "windowed_dataset.npz"
MODEL_PATH = ROOT / "models" / "baseline_cnn_mild.keras"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

with np.load(DATA_PATH) as data:
    X = data["X"]
    y = data["y"]
    splits = data["splits"]

print("Available splits:", np.unique(splits, return_counts=True))

test_mask = splits == "test"
if not np.any(test_mask):
    raise ValueError(
        "No split named 'test' was found. Check the split names above."
    )

X_test = X[test_mask][..., np.newaxis]
y_test = y[test_mask]

model = tf.keras.models.load_model(MODEL_PATH)
probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

class_names = ["bird", "insect", "background"]

print(f"\nTest samples: {len(y_test)}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=class_names,
    digits=4,
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted):")
print("Class order:", class_names)
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
