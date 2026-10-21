
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)

METADATA_PATH = "data/metadata/dataset_split.csv"
AUDIO_DIR = "data/raw/ESC-50-master/audio"

MODEL_PATH = "models/rotor_consistent_cnn_balanced.keras"
BEST_MODEL_PATH = "models/rotor_consistent_cnn_balanced_best.keras"
HISTORY_PATH = "reports/rotor_consistent_balanced_history.csv"

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
HOP_SIZE = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512

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


def mix_rotor_noise(audio, snr_db, rng):
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-12:
        return audio.copy()

    rotor = generate_rotor_noise(len(audio), rng)
    rotor_power = np.mean(rotor ** 2)
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rotor *= np.sqrt(target_noise_power / (rotor_power + 1e-12))

    mixed = audio + rotor
    peak = np.max(np.abs(mixed)) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def audio_to_log_mel(window):
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_recordings():
    df = pd.read_csv(METADATA_PATH)
    recordings = {"train": [], "validation": []}

    for _, row in df.iterrows():
        split = row["split"]
        if split not in recordings:
            continue

        path = os.path.join(AUDIO_DIR, row["filename"])
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        recordings[split].append({
            "filename": row["filename"],
            "label": LABEL_MAP[row["group"]],
            "audio": audio.astype(np.float32),
        })

    return recordings


def make_training_examples(recordings):
    rng = np.random.default_rng(SEED)
    features, labels = [], []

    for recording in recordings:
        audio = recording["audio"]
        label = recording["label"]

        for start in range(0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE):
            window = audio[start:start + WINDOW_SIZE]

            # Include the clean example for every class.
            features.append(audio_to_log_mel(window))
            labels.append(label)

            # Give every class the same number of noisy copies
            # and the same range of SNR levels.
            for snr_db in (-10.0, -5.0, 0.0, 5.0):
                noisy = mix_rotor_noise(window, snr_db, rng)
                features.append(audio_to_log_mel(noisy))
                labels.append(label)

        print(f"Prepared training recording: {recording['filename']}")

    X = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def make_clean_examples(recordings):
    features, labels = [], []

    for recording in recordings:
        audio = recording["audio"]

        for start in range(0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE):
            window = audio[start:start + WINDOW_SIZE]
            features.append(audio_to_log_mel(window))
            labels.append(recording["label"])

    X = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(40, 87, 1)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Loading recordings...")
    recordings = load_recordings()

    print("Preparing consistent rotor-augmented training examples...")
    X_train, y_train = make_training_examples(recordings["train"])
    X_val, y_val = make_clean_examples(recordings["validation"])

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Training windows by class:")

    for label, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_train == label).sum()}")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(HISTORY_PATH),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        class_weight={0: 0.75, 1: 1.50, 2: 1.00},
        verbose=1,
    )

    model.save(MODEL_PATH)
    print("\nTraining complete.")
    print("Model:", MODEL_PATH)
    print("Best checkpoint:", BEST_MODEL_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()
