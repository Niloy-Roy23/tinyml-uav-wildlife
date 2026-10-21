from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
NORM_PATH = ROOT / "models" / "mfcc_cnn1d_normalization.npz"
MODEL_PATH = ROOT / "models" / "mfcc_cnn1d_int8.tflite"
REPORT_DIR = ROOT / "reports" / "cnn1d_evaluation"
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
    X_test = ((X[test_mask] - mean) / std).transpose(0, 2, 1)
    y_test = y[test_mask]

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise ValueError("Expected INT8 input and output tensors.")

    input_scale, input_zero_point = input_info["quantization"]
    output_scale, output_zero_point = output_info["quantization"]

    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("Invalid quantization parameters.")

    predictions = []
    start = time.perf_counter()

    for sample in X_test:
        quantized = np.round(sample / input_scale + input_zero_point)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_info["index"], quantized[np.newaxis, ...])
        interpreter.invoke()

        output = interpreter.get_tensor(output_info["index"])
        scores = (
            output.astype(np.float32) - output_zero_point
        ) * output_scale
        predictions.append(int(np.argmax(scores[0])))

    elapsed = time.perf_counter() - start
    predictions = np.asarray(predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(REPORT_DIR / "test_int8_class_metrics.csv")
    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(REPORT_DIR / "test_int8_confusion_matrix.csv")

    print("Model:", MODEL_PATH.name)
    print("Test windows:", len(y_test))
    print(f"INT8 test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"INT8 macro-F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(f"Total inference time: {elapsed:.4f} seconds")
    print(f"Average inference time: {elapsed / len(y_test) * 1000:.4f} ms/window")
    print("\nClassification report:")
    print(pd.DataFrame(report).T.round(4).to_string())
    print("\nConfusion matrix (true rows, predicted columns):")
    print(cm)
    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
