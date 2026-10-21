
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

AUDIO_DIR = "data/raw/ESC-50-master/audio"
METADATA_PATH = "data/metadata/dataset_split.csv"
OUTPUT_DIR = "reports/rotor_four_models"

MODELS = {
    "existing_augmentation": "models/augmentation_cnn.keras",
    "rotor_augmentation": "models/rotor_augmentation_cnn_best.keras",
    "rotor_balanced": "models/rotor_balanced_cnn_best.keras",
    "rotor_consistent": "models/rotor_consistent_cnn_best.keras",
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


def generate_rotor_noise(length, rng):
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    fundamental = rng.uniform(80.0, 180.0)
    rotor = np.zeros(length, dtype=np.float32)

    for harmonic in range(1, 6):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        rotor += np.sin(
            2.0 * np.pi * fundamental * harmonic * t + phase
        ) / harmonic

    modulation_frequency = rng.uniform(15.0, 45.0)
    modulation = 0.55 + 0.45 * np.sin(
        2.0 * np.pi * modulation_frequency * t
    )
    rotor *= modulation
    rotor += rng.normal(0.0, 0.12, length).astype(np.float32)
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)
    return rotor.astype(np.float32)


def add_rotor_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-12:
        return signal.copy()

    rotor = generate_rotor_noise(len(signal), rng)
    rotor_power = np.mean(rotor ** 2)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rotor *= np.sqrt(target_noise_power / (rotor_power + 1e-12))
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


def make_test_examples(test_metadata):
    """Generate each test feature once for fair comparison."""
    conditions = {
        "clean": None,
        "rotor_0db": 0.0,
        "rotor_minus10db": -10.0,
    }

    rng = np.random.default_rng(SEED)
    examples = {name: [] for name in conditions}
    y_true = []

    for row in test_metadata.itertuples(index=False):
        path = os.path.join(AUDIO_DIR, row.filename)
        audio, sr = librosa.load(path, sr=None, mono=True)

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate for {row.filename}: {sr}"
            )

        recording_features = {name: [] for name in conditions}

        for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP):
            window = audio[start:start + WINDOW_SIZE].astype(np.float32)

            for condition, snr_db in conditions.items():
                if snr_db is None:
                    tested_audio = window
                else:
                    tested_audio = add_rotor_noise(window, snr_db, rng)

                feature = extract_feature(tested_audio)
                recording_features[condition].append(
                    feature[..., np.newaxis]
                )

        y_true.append(LABEL_MAP[row.group])

        for condition in conditions:
            examples[condition].append(recording_features[condition])

        print(f"Prepared {row.filename}")

    return np.asarray(y_true), examples


def predict_recordings(model, recordings):
    predictions = []

    for windows in recordings:
        batch = np.asarray(windows, dtype=np.float32)
        scores = model.predict(batch, verbose=0)
        window_predictions = np.argmax(scores, axis=1)
        counts = np.bincount(window_predictions, minlength=3)
        predictions.append(int(np.argmax(counts)))

    return np.asarray(predictions)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy()

    if test_metadata.empty:
        raise ValueError("No test recordings found.")

    print("Preparing held-out test examples once...")
    y_true, examples = make_test_examples(test_metadata)

    summaries = []
    class_rows = []
    matrix_rows = []

    for model_name, model_path in MODELS.items():
        print(f"\nLoading model: {model_name}")
        model = tf.keras.models.load_model(model_path)

        for condition, recordings in examples.items():
            y_pred = predict_recordings(model, recordings)

            report = classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )

            summaries.append({
                "model": model_name,
                "condition": condition,
                "recordings": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                ),
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

            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            for i, true_name in enumerate(CLASS_NAMES):
                for j, predicted_name in enumerate(CLASS_NAMES):
                    matrix_rows.append({
                        "model": model_name,
                        "condition": condition,
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[i, j]),
                    })

    pd.DataFrame(summaries).to_csv(
        os.path.join(OUTPUT_DIR, "summary.csv"), index=False
    )
    pd.DataFrame(class_rows).to_csv(
        os.path.join(OUTPUT_DIR, "class_metrics.csv"), index=False
    )
    pd.DataFrame(matrix_rows).to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrices.csv"), index=False
    )

    print("\nFour-model comparison")
    print(pd.DataFrame(summaries).round(4).to_string(index=False))
    print(f"\nReports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
