
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
Interpreter = __import__('tensorflow').lite.Interpreter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "windowed_dataset.npz"
MODELS = {
    "E6": PROJECT_DIR / "models" / "rotor_background_cnn_best_int8.tflite",
    "E7": PROJECT_DIR / "models" / "rotor_combined_cnn_best_int8.tflite",
}
CLASS_NAMES = ["bird", "insect", "background"]


def evaluate_model(name, model_path, X_test, y_test, filenames):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    input_scale = input_info["quantization_parameters"]["scales"][0]
    input_zero = input_info["quantization_parameters"]["zero_points"][0]
    output_scale = output_info["quantization_parameters"]["scales"][0]
    output_zero = output_info["quantization_parameters"]["zero_points"][0]

    predictions = []
    inference_seconds = 0.0

    for feature in X_test:
        quantized = np.clip(
            np.rint(feature / input_scale + input_zero),
            -128,
            127,
        ).astype(np.int8).reshape(input_info["shape"])

        interpreter.set_tensor(input_info["index"], quantized)

        start = time.perf_counter()
        interpreter.invoke()
        inference_seconds += time.perf_counter() - start

        raw = interpreter.get_tensor(output_info["index"])[0]
        scores = (raw.astype(np.float32) - output_zero) * output_scale
        predictions.append(int(np.argmax(scores)))

    predictions = np.asarray(predictions)

    # Recording-level prediction: majority vote across its windows.
    recording_truth = []
    recording_predictions = []

    for filename in np.unique(filenames):
        indices = np.where(filenames == filename)[0]
        true_labels = np.unique(y_test[indices])

        if len(true_labels) != 1:
            raise ValueError(
                f"Recording {filename} has inconsistent true labels."
            )

        recording_truth.append(int(true_labels[0]))
        counts = np.bincount(predictions[indices], minlength=3)
        recording_predictions.append(int(np.argmax(counts)))

    recording_truth = np.asarray(recording_truth)
    recording_predictions = np.asarray(recording_predictions)

    print(f"\n{'=' * 55}")
    print(f"{name}: {model_path.name}")
    print(f"Model size: {model_path.stat().st_size:,} bytes")
    print(f"Window accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Window recall (bird, insect, background):",
          np.round(recall_score(
              y_test, predictions, labels=[0, 1, 2],
              average=None, zero_division=0
          ), 4))
    print("Window confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    print(f"Recording accuracy: "
          f"{accuracy_score(recording_truth, recording_predictions):.4f}")
    print("Recording recall (bird, insect, background):",
          np.round(recall_score(
              recording_truth, recording_predictions, labels=[0, 1, 2],
              average=None, zero_division=0
          ), 4))
    print("Recording confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(
        recording_truth, recording_predictions, labels=[0, 1, 2]
    ))
    print(f"Total inference time for {len(X_test)} windows: "
          f"{inference_seconds:.4f} seconds")
    print(f"Average inference time: "
          f"{inference_seconds / len(X_test) * 1000:.4f} ms/window")

    return predictions


def main():
    for path in [DATASET_PATH, *MODELS.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    with np.load(DATASET_PATH) as data:
        X = data["X"]
        y = data["y"]
        splits = data["splits"]
        filenames = data["filenames"]

    test_mask = splits == "test"
    X_test = X[test_mask].astype(np.float32)[..., np.newaxis]
    y_test = y[test_mask]
    test_filenames = filenames[test_mask]

    print("Held-out test windows:", len(X_test))
    print("Unique test recordings:", len(np.unique(test_filenames)))
    print("Classes:", CLASS_NAMES)

    results = {}
    for name, path in MODELS.items():
        results[name] = evaluate_model(
            name, path, X_test, y_test, test_filenames
        )

    print(f"\n{'=' * 55}")
    print("E6/E7 prediction agreement:",
          f"{np.mean(results['E6'] == results['E7']) * 100:.2f}%")
    print("Evaluation complete. No models or dataset files were changed.")


if __name__ == "__main__":
    main()
