
import os

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

AUDIO_DIR = "data/raw/ESC-50-master/audio"
METADATA_PATH = "data/metadata/dataset_split.csv"
MODEL_PATH = "models/augmentation_cnn.keras"
OUTPUT_PATH = "reports/week2_noise_evaluation.csv"

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512
SEED = 42

LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}


def add_noise_at_snr(signal, snr_db, rng):
    """Add Gaussian noise at the requested signal-to-noise ratio."""
    signal_power = np.mean(signal ** 2)

    if signal_power < 1e-12:
        return signal.copy()

    noise = rng.normal(size=signal.shape).astype(np.float32)
    noise_power = np.mean(noise ** 2)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / max(noise_power, 1e-12))

    return signal + noise


def extract_feature(audio):
    """Use the same log-mel settings as the Week 1 pipeline."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def main():
    os.makedirs("reports", exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy()

    if test_metadata.empty:
        raise ValueError("No test recordings found in the metadata.")

    model = tf.keras.models.load_model(MODEL_PATH)
    rng = np.random.default_rng(SEED)

    conditions = [
        ("clean", None),
        ("snr_0db", 0.0),
        ("snr_minus10db", -10.0),
    ]

    true_labels = []
    predictions = {name: [] for name, _ in conditions}

    for row in test_metadata.itertuples(index=False):
        audio_path = os.path.join(AUDIO_DIR, row.filename)
        audio, sample_rate = librosa.load(
            audio_path, sr=None, mono=True
        )

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate for {row.filename}: "
                f"{sample_rate}"
            )

        if row.group not in LABEL_MAP:
            raise ValueError(f"Unknown class: {row.group}")

        recording_predictions = {name: [] for name, _ in conditions}

        for start in range(
            0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP
        ):
            window = audio[start:start + WINDOW_SIZE].astype(
                np.float32
            )

            for condition, snr_db in conditions:
                if snr_db is None:
                    noisy_window = window
                else:
                    noisy_window = add_noise_at_snr(
                        window, snr_db, rng
                    )

                feature = extract_feature(noisy_window)
                model_input = feature[np.newaxis, ..., np.newaxis]
                scores = model.predict(model_input, verbose=0)

                recording_predictions[condition].append(
                    int(np.argmax(scores[0]))
                )

        if not recording_predictions["clean"]:
            raise ValueError(
                f"No complete one-second windows in {row.filename}"
            )

        true_labels.append(LABEL_MAP[row.group])

        # Majority vote across windows gives one prediction per recording.
        for condition, _ in conditions:
            counts = np.bincount(
                recording_predictions[condition], minlength=3
            )
            predictions[condition].append(int(np.argmax(counts)))

        print(f"Evaluated {row.filename}")

    results = []

    for condition, _ in conditions:
        results.append({
            "condition": condition,
            "recordings": len(true_labels),
            "accuracy": accuracy_score(
                true_labels, predictions[condition]
            ),
            "macro_f1": f1_score(
                true_labels,
                predictions[condition],
                average="macro",
                zero_division=0,
            ),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nWeek 2 noise evaluation")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
