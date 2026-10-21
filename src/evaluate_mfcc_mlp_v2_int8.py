from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
NORM_PATH = ROOT / "models" / "mfcc_normalization_v2.npz"
MODEL_PATH = ROOT / "models" / "mfcc_mlp_v2_int8.tflite"
OUT_DIR = ROOT / "reports" / "mfcc_mlp_evaluation"
CLASS_NAMES = ["bird", "insect", "background"]


def main():
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        splits = data["splits"]

    with np.load(NORM_PATH) as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)

    test_mask = splits == "test"
    X_test = (X[test_mask] - mean) / std
    y_test = y[test_mask]

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise ValueError("Expected INT8 input and output tensors.")

    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    if in_scale <= 0 or out_scale <= 0:
        raise ValueError("Invalid quantization parameters.")

    predictions = []
    start = time.perf_counter()

    for sample in X_test:
        q = np.round(sample / in_scale + in_zero)
        q = np.clip(q, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], q[np.newaxis, ...])
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])
        scores = (raw.astype(np.float32) - out_zero) * out_scale
        predictions.append(int(np.argmax(scores[0])))

    elapsed = time.perf_counter() - start
    predictions = np.asarray(predictions)

    report = classification_report(
        y_test, predictions, labels=[0, 1, 2],
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(OUT_DIR / "mlp_v2_int8_test_metrics.csv")
    pd.DataFrame(
        cm,
        index=["true_bird", "true_insect", "true_background"],
        columns=["pred_bird", "pred_insect", "pred_background"],
    ).to_csv(OUT_DIR / "mlp_v2_int8_test_confusion.csv")

    print("Model:", MODEL_PATH.name)
    print("Test windows:", len(y_test))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro-F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(f"Total inference time: {elapsed:.4f} seconds")
    print(f"Average inference time: {elapsed / len(y_test) * 1000:.4f} ms/window")
    print("\nClass metrics:")
    print(pd.DataFrame(report).T.round(4).to_string())
    print("\nConfusion matrix (true rows, predicted columns):")
    print(cm)
    print("\nReports saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
