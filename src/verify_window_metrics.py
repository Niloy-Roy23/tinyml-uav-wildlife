from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/mfcc/mfcc_dataset.npz"

MODELS = {
    "Baseline MLP INT8": (
        "models/mfcc_mlp_int8.tflite",
        "models/mfcc_normalization.npz",
    ),
    "MLP v2 INT8": (
        "models/mfcc_mlp_v2_int8.tflite",
        "models/mfcc_normalization_v2.npz",
    ),
}

with np.load(DATA) as d:
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    splits = d["splits"]

mask = splits == "test"
X_test, y_test = X[mask], y[mask]

print("TEST WINDOWS:", len(y_test))
print("TEST CLASS COUNTS:", dict(zip(*np.unique(y_test, return_counts=True))))

for name, (model_path, norm_path) in MODELS.items():
    with np.load(ROOT / norm_path) as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)

    X_norm = (X_test - mean) / std

    interpreter = tf.lite.Interpreter(
        model_path=str(ROOT / model_path)
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    scale, zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    predictions = []
    for sample in X_norm:
        q = np.clip(
            np.round(sample / scale + zero), -128, 127
        ).astype(np.int8)

        interpreter.set_tensor(inp["index"], q[None, ...])
        interpreter.invoke()

        raw = interpreter.get_tensor(out["index"])
        scores = (raw.astype(np.float32) - out_zero) * out_scale
        predictions.append(int(np.argmax(scores[0])))

    predictions = np.asarray(predictions)

    print(f"\n{name}")
    print(f"Accuracy:   {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro-F1:   {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"Macro-recall: {recall_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print("Class recalls [bird, insect, background]:",
          np.round(recall_score(y_test, predictions, labels=[0, 1, 2],
                                average=None, zero_division=0), 4))
