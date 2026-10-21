from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data/raw/ESC-50-master/audio"
METADATA_PATH = ROOT / "data/metadata/dataset_split.csv"
OUTPUT_DIR = ROOT / "reports/mfcc_mlp_evaluation"

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MFCC = 40
N_FFT = 2048
MFCC_HOP_LENGTH = 512
SEED = 42

CLASS_NAMES = ["bird", "insect", "background"]
LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CONDITIONS = [("clean", None), ("rotor_0db", 0.0), ("rotor_minus10db", -10.0)]

MODELS = {
    "baseline_mlp_int8": {
        "model": ROOT / "models/mfcc_mlp_int8.tflite",
        "norm": ROOT / "models/mfcc_normalization.npz",
    },
    "mlp_v2_int8": {
        "model": ROOT / "models/mfcc_mlp_v2_int8.tflite",
        "norm": ROOT / "models/mfcc_normalization_v2.npz",
    },
}


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


def extract_mfcc(audio):
    return librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=MFCC_HOP_LENGTH,
    ).astype(np.float32)


def predict_int8(interpreter, feature, mean, std):
    normalized = (feature - mean.reshape(40, 87)) / std.reshape(40, 87)

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    if in_scale <= 0 or out_scale <= 0:
        raise ValueError("Invalid INT8 quantization parameters.")

    q = np.round(normalized / in_scale + in_zero)
    q = np.clip(q, -128, 127).astype(np.int8)
    interpreter.set_tensor(inp["index"], q[np.newaxis, ...])
    interpreter.invoke()

    raw = interpreter.get_tensor(out["index"])
    scores = (raw.astype(np.float32) - out_zero) * out_scale
    return int(np.argmax(scores[0]))


def main():
    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy()
    if test_metadata.empty:
        raise ValueError("No test recordings found.")

    model_objects = {}
    for name, paths in MODELS.items():
        with np.load(paths["norm"]) as norm:
            mean = norm["mean"].astype(np.float32)
            std = norm["std"].astype(np.float32)

        interpreter = tf.lite.Interpreter(model_path=str(paths["model"]))
        interpreter.allocate_tensors()

        if interpreter.get_input_details()[0]["dtype"] != np.int8:
            raise ValueError(f"{name}: input is not INT8.")
        if interpreter.get_output_details()[0]["dtype"] != np.int8:
            raise ValueError(f"{name}: output is not INT8.")

        model_objects[name] = {
            "interpreter": interpreter,
            "mean": mean,
            "std": std,
        }

    keys = [(model, condition) for model in MODELS for condition, _ in CONDITIONS]
    all_predictions = {key: [] for key in keys}
    y_true = []
    rng = np.random.default_rng(SEED)

    for row in test_metadata.itertuples(index=False):
        audio_path = AUDIO_DIR / row.filename
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(audio) < WINDOW_SIZE:
            audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

        per_recording = {key: [] for key in keys}

        for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP):
            window = audio[start:start + WINDOW_SIZE].astype(np.float32)

            for condition, snr_db in CONDITIONS:
                tested_audio = window if snr_db is None else add_rotor_noise(window, snr_db, rng)
                feature = extract_mfcc(tested_audio)

                # Both models receive the same audio and MFCC feature.
                for model_name, obj in model_objects.items():
                    pred = predict_int8(
                        obj["interpreter"], feature, obj["mean"], obj["std"]
                    )
                    per_recording[(model_name, condition)].append(pred)

        y_true.append(LABEL_MAP[row.group])

        for key, predictions in per_recording.items():
            counts = np.bincount(predictions, minlength=3)
            all_predictions[key].append(int(np.argmax(counts)))

        print("Evaluated:", row.filename)

    y_true = np.asarray(y_true, dtype=np.int64)
    summary_rows, class_rows, matrix_rows = [], [], []

    for (model_name, condition), values in all_predictions.items():
        y_pred = np.asarray(values, dtype=np.int64)
        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2],
            target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        summary_rows.append({
            "model": model_name,
            "condition": condition,
            "recordings": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        })

        for class_name in CLASS_NAMES:
            class_rows.append({
                "model": model_name,
                "condition": condition,
                "class": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1_score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            })

        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                matrix_rows.append({
                    "model": model_name,
                    "condition": condition,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    classes = pd.DataFrame(class_rows)
    matrices = pd.DataFrame(matrix_rows)

    summary.to_csv(OUTPUT_DIR / "mlp_int8_rotor_recording_summary.csv", index=False)
    classes.to_csv(OUTPUT_DIR / "mlp_int8_rotor_class_metrics.csv", index=False)
    matrices.to_csv(OUTPUT_DIR / "mlp_int8_rotor_confusion_matrices.csv", index=False)

    print("\nRecording-level summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClass-wise recall:")
    print(classes.pivot(index=["model", "condition"], columns="class", values="recall")
          .round(4).to_string())
    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
