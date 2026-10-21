import os
import time
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

AUDIO_DIR = "data/raw/ESC-50-master/audio"
METADATA_PATH = "data/metadata/dataset_split.csv"
OUTPUT_DIR = "reports/int8_robustness"

MODELS = {
    "E6_rotor_background_INT8": "models/rotor_background_cnn_best_int8.tflite",
    "E7_rotor_combined_INT8": "models/rotor_combined_cnn_best_int8.tflite",
}

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512
SEED = 42
LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CLASS_NAMES = ["bird", "insect", "background"]
CONDITIONS = [("clean", None), ("rotor_0db", 0.0), ("rotor_minus10db", -10.0)]


def add_rotor_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-12:
        return signal.copy()

    t = np.arange(len(signal), dtype=np.float32) / SAMPLE_RATE
    fundamental = rng.uniform(80.0, 180.0)
    rotor = np.zeros(len(signal), dtype=np.float32)

    for harmonic in range(1, 6):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        rotor += np.sin(2.0 * np.pi * fundamental * harmonic * t + phase) / harmonic

    modulation_frequency = rng.uniform(15.0, 45.0)
    modulation = 0.55 + 0.45 * np.sin(2.0 * np.pi * modulation_frequency * t)
    rotor *= modulation
    rotor += rng.normal(0.0, 0.12, len(signal)).astype(np.float32)
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    rotor *= np.sqrt(target_noise_power / (np.mean(rotor ** 2) + 1e-12))
    return (signal + rotor).astype(np.float32)


def extract_feature(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=MEL_HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def predict_int8(interpreter, input_detail, output_detail, feature):
    scale, zero_point = input_detail["quantization"]
    if scale <= 0:
        raise ValueError("Invalid input quantisation scale.")

    quantized = np.clip(
        np.rint(feature / scale + zero_point), -128, 127
    ).astype(np.int8)
    model_input = quantized[np.newaxis, ..., np.newaxis]
    interpreter.set_tensor(input_detail["index"], model_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]
    return int(np.argmax(output))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy().reset_index(drop=True)

    if test_metadata.empty:
        raise ValueError("No test recordings found.")

    interpreters = {}
    for name, path in MODELS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        interpreters[name] = load_interpreter(path)
        _, inp, out = interpreters[name]
        if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
            raise TypeError(f"{name} is not fully INT8 at input/output.")
        print(f"{name}: input={inp['shape']}, output={out['shape']}, "
              f"input_quant={inp['quantization']}, output_quant={out['quantization']}")

    # Build one shared set of windows/features so both models receive exactly
    # the same clean and noisy inputs. This preserves the existing noise recipe.
    rng = np.random.default_rng(SEED)
    y_true = []
    features_by_condition = {condition: [] for condition, _ in CONDITIONS}

    for row in test_metadata.itertuples(index=False):
        audio_path = os.path.join(AUDIO_DIR, row.filename)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate {sr}: {row.filename}")

        recording_features = {condition: [] for condition, _ in CONDITIONS}
        for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP):
            window = audio[start:start + WINDOW_SIZE].astype(np.float32)
            for condition, snr_db in CONDITIONS:
                tested_audio = window if snr_db is None else add_rotor_noise(window, snr_db, rng)
                recording_features[condition].append(extract_feature(tested_audio))

        y_true.append(LABEL_MAP[row.group])
        for condition, _ in CONDITIONS:
            features_by_condition[condition].append(recording_features[condition])
        print(f"Prepared {row.filename}")

    y_true = np.asarray(y_true)
    summary_rows, class_rows, matrix_rows, timing_rows = [], [], [], []

    for model_name, (interpreter, input_detail, output_detail) in interpreters.items():
        for condition, _ in CONDITIONS:
            recording_predictions = []
            total_windows = 0
            inference_seconds = 0.0

            for recording_features in features_by_condition[condition]:
                window_predictions = []
                for feature in recording_features:
                    start_time = time.perf_counter()
                    pred = predict_int8(interpreter, input_detail, output_detail, feature)
                    inference_seconds += time.perf_counter() - start_time
                    total_windows += 1
                    window_predictions.append(pred)

                counts = np.bincount(window_predictions, minlength=3)
                recording_predictions.append(int(np.argmax(counts)))

            y_pred = np.asarray(recording_predictions)
            report = classification_report(
                y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                output_dict=True, zero_division=0
            )
            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

            summary_rows.append({
                "model": model_name, "condition": condition,
                "recordings": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            })
            for class_name in CLASS_NAMES:
                class_rows.append({
                    "model": model_name, "condition": condition, "class": class_name,
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1_score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                })
            for ti, true_name in enumerate(CLASS_NAMES):
                for pi, pred_name in enumerate(CLASS_NAMES):
                    matrix_rows.append({
                        "model": model_name, "condition": condition,
                        "true_class": true_name, "predicted_class": pred_name,
                        "count": int(matrix[ti, pi]),
                    })
            timing_rows.append({
                "model": model_name, "condition": condition,
                "windows": total_windows,
                "inference_seconds": inference_seconds,
                "mean_inference_ms_per_window": 1000 * inference_seconds / max(total_windows, 1),
                "model_file_bytes": os.path.getsize(MODELS[model_name]),
            })
            print(f"Completed {model_name} / {condition}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(os.path.join(OUTPUT_DIR, "class_metrics.csv"), index=False)
    pd.DataFrame(matrix_rows).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrices.csv"), index=False)
    pd.DataFrame(timing_rows).to_csv(os.path.join(OUTPUT_DIR, "inference_timing.csv"), index=False)

    print("\nRecording-level summary")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nReports saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
