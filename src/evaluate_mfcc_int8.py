
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
NORM_PATH = ROOT / "models" / "mfcc_normalization.npz"
MODEL_PATH = ROOT / "models" / "mfcc_mlp_int8.tflite"

CLASS_NAMES = ["bird", "insect", "background"]


def main():
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        splits = data["splits"]

    with np.load(NORM_PATH) as norm:
        mean = norm["mean"]
        std = norm["std"]

    test_mask = splits == "test"
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Apply the exact normalization used during training.
    X_test = (X_test - mean) / std

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_PATH)
    )
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_info["quantization"]
    output_scale, output_zero_point = output_info["quantization"]

    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("Invalid model quantization parameters.")

    predictions = []
    start_time = time.perf_counter()

    for sample in X_test:
        # Quantize the normalized MFCC input to int8.
        quantized = np.round(
            sample / input_scale + input_zero_point
        )
        quantized = np.clip(quantized, -128, 127).astype(np.int8)

        interpreter.set_tensor(
            input_info["index"],
            quantized[np.newaxis, ...],
        )
        interpreter.invoke()

        output = interpreter.get_tensor(output_info["index"])
        scores = (
            output.astype(np.float32) - output_zero_point
        ) * output_scale

        predictions.append(int(np.argmax(scores[0])))

    elapsed = time.perf_counter() - start_time
    predictions = np.asarray(predictions)

    print("Model:", MODEL_PATH.name)
    print("Test windows:", len(y_test))
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Total inference time: {elapsed:.4f} seconds")
    print(f"Average inference time: {elapsed / len(y_test) * 1000:.4f} ms/window")

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
