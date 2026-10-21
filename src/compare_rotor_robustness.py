import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

AUDIO_DIR = "data/raw/ESC-50-master/audio"
METADATA_PATH = "data/metadata/dataset_split.csv"

MODELS = {
    "existing_augmentation": "models/augmentation_cnn.keras",
    "rotor_augmentation": "models/rotor_augmentation_cnn_best.keras",
}

OUTPUT_PATH = "reports/rotor_robustness_comparison.csv"

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512
SEED = 42

LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CLASS_NAMES = ["bird", "insect", "background"]

def add_noise_at_snr(signal, snr_db, rng):
    """Add Gaussian noise at a specified signal-to-noise ratio."""
    signal_power = np.mean(signal ** 2)

    if signal_power < 1e-12:
        return signal.copy()

    noise = rng.normal(size=signal.shape).astype(np.float32)
    noise_power = np.mean(noise ** 2)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / max(noise_power, 1e-12))

    return signal + noise


def generate_rotor_noise(length, sample_rate, rng):
    """Generate synthetic periodic, harmonic rotor-like noise."""
    t = np.arange(length, dtype=np.float32) / sample_rate
    fundamental = rng.uniform(80.0, 180.0)

    rotor = np.zeros(length, dtype=np.float32)

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

    rotor += rng.normal(0.0, 0.12, length).astype(np.float32)
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)

    return rotor.astype(np.float32)


def add_rotor_noise_at_snr(signal, snr_db, rng):
    """Mix synthetic rotor noise at a specified SNR."""
    signal_power = np.mean(signal ** 2)

    if signal_power < 1e-12:
        return signal.copy()

    rotor = generate_rotor_noise(len(signal), SAMPLE_RATE, rng)
    rotor_power = np.mean(rotor ** 2)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    rotor *= np.sqrt(target_noise_power / max(rotor_power, 1e-12))

    return (signal + rotor).astype(np.float32)


def extract_feature(audio):
    """Use the project's existing log-mel feature settings."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
    )

    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def evaluate_model(model, test_metadata, conditions):
    rng = np.random.default_rng(SEED)
    true_labels = []
    predictions = {name: [] for name, _, _ in conditions}

    for row in test_metadata.itertuples(index=False):
        audio_path = os.path.join(AUDIO_DIR, row.filename)
        audio, sr = librosa.load(
            audio_path, sr=None, mono=True
        )

        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate: {sr}")

        recording_predictions = {
            name: [] for name, _, _ in conditions
        }

        for start in range(
            0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP
        ):
            window = audio[start:start + WINDOW_SIZE].astype(
                np.float32
            )

            for condition, noise_type, snr_db in conditions:
                if noise_type == "gaussian":
                    tested_audio = add_noise_at_snr(
                        window, snr_db, rng
                    )
                elif noise_type == "rotor":
                    tested_audio = add_rotor_noise_at_snr(
                        window, snr_db, rng
                    )
                else:
                    tested_audio = window

                feature = extract_feature(tested_audio)
                model_input = feature[np.newaxis, ..., np.newaxis]
                scores = model.predict(model_input, verbose=0)

                recording_predictions[condition].append(
                    int(np.argmax(scores[0]))
                )

        true_labels.append(LABEL_MAP[row.group])

        for condition, _, _ in conditions:
            counts = np.bincount(
                recording_predictions[condition], minlength=3
            )
            predictions[condition].append(int(np.argmax(counts)))

        print(f"Evaluated {row.filename}")

    return true_labels, predictions

def main():
    os.makedirs("reports", exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy()

    if test_metadata.empty:
        raise ValueError("No test recordings found.")

    conditions = [
        ("clean", "clean", None),
        ("gaussian_0db", "gaussian", 0.0),
        ("gaussian_minus10db", "gaussian", -10.0),
        ("rotor_0db", "rotor", 0.0),
        ("rotor_minus10db", "rotor", -10.0),
    ]

    all_results = []

    for model_name, model_path in MODELS.items():
        print(f"\nLoading model: {model_name}")
        model = tf.keras.models.load_model(model_path)

        true_labels, predictions = evaluate_model(
            model, test_metadata, conditions
        )

        for condition, _, _ in conditions:
            y_pred = predictions[condition]

            all_results.append({
                "model": model_name,
                "condition": condition,
                "recordings": len(true_labels),
                "accuracy": accuracy_score(true_labels, y_pred),
                "macro_f1": f1_score(
                    true_labels,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            })

    results = pd.DataFrame(all_results)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nRobustness comparison")
    print(results.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
