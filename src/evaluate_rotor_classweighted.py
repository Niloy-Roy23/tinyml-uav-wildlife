
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

AUDIO_DIR = "data/raw/ESC-50-master/audio"
METADATA_PATH = "data/metadata/dataset_split.csv"
OUTPUT_DIR = "reports/rotor_classweighted"

MODELS = {
    "rotor_classweighted": "models/rotor_classweighted_cnn_best.keras",
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


def add_rotor_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)

    if signal_power < 1e-12:
        return signal.copy()

    t = np.arange(len(signal), dtype=np.float32) / SAMPLE_RATE
    fundamental = rng.uniform(80.0, 180.0)
    rotor = np.zeros(len(signal), dtype=np.float32)

    for harmonic in range(1, 6):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        rotor += (
            np.sin(2.0 * np.pi * fundamental * harmonic * t + phase)
            / harmonic
        )

    modulation_frequency = rng.uniform(15.0, 45.0)
    modulation = 0.55 + 0.45 * np.sin(
        2.0 * np.pi * modulation_frequency * t
    )
    rotor *= modulation
    rotor += rng.normal(0.0, 0.12, len(signal)).astype(np.float32)
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    rotor *= np.sqrt(target_noise_power / (np.mean(rotor ** 2) + 1e-12))

    return (signal + rotor).astype(np.float32)


def extract_feature(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def evaluate_model(model, test_metadata):
    rng = np.random.default_rng(SEED)

    conditions = {
        "clean": ("clean", None),
        "rotor_0db": ("rotor", 0.0),
        "rotor_minus10db": ("rotor", -10.0),
    }

    y_true = []
    predictions = {name: [] for name in conditions}

    for row in test_metadata.itertuples(index=False):
        audio_path = os.path.join(AUDIO_DIR, row.filename)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate: {row.filename}")

        recording_preds = {name: [] for name in conditions}

        for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP):
            window = audio[start:start + WINDOW_SIZE].astype(np.float32)

            for condition, (noise_type, snr_db) in conditions.items():
                if noise_type == "rotor":
                    tested_audio = add_rotor_noise(window, snr_db, rng)
                else:
                    tested_audio = window

                feature = extract_feature(tested_audio)
                model_input = feature[np.newaxis, ..., np.newaxis]
                scores = model.predict(model_input, verbose=0)

                recording_preds[condition].append(
                    int(np.argmax(scores[0]))
                )

        y_true.append(LABEL_MAP[row.group])

        for condition in conditions:
            counts = np.bincount(recording_preds[condition], minlength=3)
            predictions[condition].append(int(np.argmax(counts)))

        print(f"Evaluated {row.filename}")

    return np.array(y_true), {
        name: np.array(values) for name, values in predictions.items()
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy()

    if test_metadata.empty:
        raise ValueError("No test recordings found.")

    all_reports = []
    all_matrices = []
    summary = []

    for model_name, model_path in MODELS.items():
        print(f"\nLoading model: {model_name}")
        model = tf.keras.models.load_model(model_path)
        y_true, predictions = evaluate_model(model, test_metadata)

        for condition, y_pred in predictions.items():
            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )

            for class_name in CLASS_NAMES:
                all_reports.append({
                    "model": model_name,
                    "condition": condition,
                    "class": class_name,
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1_score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                })

            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            for true_index, true_name in enumerate(CLASS_NAMES):
                for pred_index, pred_name in enumerate(CLASS_NAMES):
                    all_matrices.append({
                        "model": model_name,
                        "condition": condition,
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": int(matrix[true_index, pred_index]),
                    })

            summary.append({
                "model": model_name,
                "condition": condition,
                "recordings": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
            })

    pd.DataFrame(all_reports).to_csv(
        os.path.join(OUTPUT_DIR, "class_metrics.csv"), index=False
    )
    pd.DataFrame(all_matrices).to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrices.csv"), index=False
    )
    pd.DataFrame(summary).to_csv(
        os.path.join(OUTPUT_DIR, "summary.csv"), index=False
    )

    print("\nSummary")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nReports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
