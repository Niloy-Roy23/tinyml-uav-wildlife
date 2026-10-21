import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)

# Dataset paths
METADATA_PATH = "data/metadata/dataset_split.csv"
AUDIO_DIR = "data/raw/ESC-50-master/audio"

# New output files: existing models will not be overwritten
MODEL_PATH = "models/e4_insect_focused_augmentation_and_balanced_sampling.keras"
BEST_MODEL_PATH = "models/e4_insect_focused_augmentation_and_balanced_sampling_best.keras"
HISTORY_PATH = "reports/e4_insect_focused_augmentation_and_balanced_sampling_history.csv"
# Audio and feature settings (match existing preprocessing)
SAMPLE_RATE = 44100
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 512

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SIZE = int(SAMPLE_RATE * HOP_SECONDS)

LABEL_MAP = {
    "bird": 0,
    "insect": 1,
    "background": 2
}
# -----------------------------
# Synthetic rotor-like noise
# -----------------------------
def generate_rotor_noise(length, sample_rate, rng):
    """Generate synthetic, periodically modulated rotor-like interference."""
    t = np.arange(length, dtype=np.float32) / sample_rate

    # Vary the fundamental frequency between recordings
    fundamental = rng.uniform(80.0, 180.0)

    # Harmonics approximate tonal components from rotating propellers
    rotor = np.zeros(length, dtype=np.float32)
    for harmonic in range(1, 6):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        rotor += (
            (1.0 / harmonic)
            * np.sin(
                2.0 * np.pi * fundamental * harmonic * t + phase
            )
        )

    # Periodic amplitude modulation creates a rough buzzing character
    modulation_frequency = rng.uniform(15.0, 45.0)
    modulation = 0.55 + 0.45 * np.sin(
        2.0 * np.pi * modulation_frequency * t
    )
    rotor *= modulation

    # Add a little broadband component
    rotor += rng.normal(0.0, 0.12, length).astype(np.float32)

    # Normalize
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)
    return rotor.astype(np.float32)


def mix_at_random_snr(audio, rotor_noise, rng, snr_db=None):
    """Mix rotor noise at a supplied or randomly selected SNR."""
    if snr_db is None:
        snr_db = float(
            rng.choice([5.0, 0.0, -5.0, -10.0],
                       p=[0.2, 0.3, 0.3, 0.2])
        )

    signal_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    noise_rms = np.sqrt(np.mean(rotor_noise ** 2) + 1e-8)
    desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    mixed = audio + rotor_noise * (desired_noise_rms / noise_rms)

    peak = np.max(np.abs(mixed)) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


# -----------------------------
# Audio to log-mel features
# -----------------------------
def audio_to_log_mel(window):
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def recording_to_windows(audio):
    windows = []

    for start in range(
        0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE
    ):
        window = audio[start:start + WINDOW_SIZE]
        windows.append(audio_to_log_mel(window))

    return windows
# -----------------------------
# Load recordings by split
# -----------------------------
def load_recordings():
    df = pd.read_csv(METADATA_PATH)

    recordings = {"train": [], "validation": [], "test": []}

    for _, row in df.iterrows():
        split = row["split"]
        if split not in recordings:
            continue

        audio_path = os.path.join(AUDIO_DIR, row["filename"])
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate: {sr}")

        recordings[split].append({
            "filename": row["filename"],
            "label": LABEL_MAP[row["group"]],
            "audio": audio.astype(np.float32)
        })

    return recordings


def make_training_examples(recordings):
    """Create clean windows and class-specific noisy training copies."""
    rng = np.random.default_rng(SEED)
    features, labels = [], []

    insect_snr_levels = [-10.0, -5.0, 0.0, 5.0]
    insect_snr_probabilities = [0.4, 0.3, 0.2, 0.1]

    for recording in recordings:
        audio = recording["audio"]
        label = recording["label"]

        for start in range(
            0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE
        ):
            clean_window = audio[start:start + WINDOW_SIZE]

            features.append(audio_to_log_mel(clean_window))
            labels.append(label)

            if label == LABEL_MAP["insect"]:
                # Always create two independent noisy insect copies.
                for _ in range(2):
                    rotor = generate_rotor_noise(
                        len(clean_window), SAMPLE_RATE, rng
                    )
                    snr_db = float(
                        rng.choice(
                            insect_snr_levels,
                            p=insect_snr_probabilities
                        )
                    )
                    noisy_window = mix_at_random_snr(
                        clean_window, rotor, rng, snr_db=snr_db
                    )
                    features.append(audio_to_log_mel(noisy_window))
                    labels.append(label)

            elif rng.random() < 0.8:
                # Preserve the existing augmentation policy for
                # bird and background training windows.
                rotor = generate_rotor_noise(
                    len(clean_window), SAMPLE_RATE, rng
                )
                noisy_window = mix_at_random_snr(
                    clean_window, rotor, rng
                )
                features.append(audio_to_log_mel(noisy_window))
                labels.append(label)

    X = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def make_clean_examples(recordings):
    features, labels = [], []

    for recording in recordings:
        windows = recording_to_windows(recording["audio"])

        for feature in windows:
            features.append(feature)
            labels.append(recording["label"])

    X = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)

    return X, y

# -----------------------------
# Build CNN
# -----------------------------
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
        layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -----------------------------
# Train
# -----------------------------
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Loading recordings...")
    recordings = load_recordings()

    print("Preparing augmented training examples...")
    X_train, y_train = make_training_examples(recordings["train"])

    print("Preparing clean validation examples...")
    X_val, y_val = make_clean_examples(recordings["validation"])

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(HISTORY_PATH)
    ]

    # Balanced sampling: draw the same number of examples from
    # each class in every epoch. Sampling is with replacement.
    rng = np.random.default_rng(SEED)
    classes = np.array([0, 1, 2])
    class_indices = {
        c: np.flatnonzero(y_train == c) for c in classes
    }

    for c, indices in class_indices.items():
        if len(indices) == 0:
            raise ValueError(f"No training examples for class {c}")

    samples_per_class = max(len(indices) for indices in class_indices.values())
    epoch_size = samples_per_class * len(classes)

    print("Training examples by class before balancing:")
    for c, indices in class_indices.items():
        print(f"  class {c}: {len(indices)}")

    print("Balanced samples per class per epoch:", samples_per_class)
    print("Total balanced samples per epoch:", epoch_size)

    # Build balanced index batches for each epoch.
    # Sampling with replacement oversamples smaller classes.
    def balanced_epoch_indices():
        selected = []
        for c in classes:
            selected.extend(
                rng.choice(
                    class_indices[c],
                    size=samples_per_class,
                    replace=True
                ).tolist()
            )
        rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)

    class BalancedSequence(keras.utils.Sequence):
        def __init__(self, X, y, batch_size, **kwargs):
            super().__init__(**kwargs)
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.indices = balanced_epoch_indices()

        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))

        def __getitem__(self, index):
            batch_indices = self.indices[
                index * self.batch_size:(index + 1) * self.batch_size
            ]
            return self.X[batch_indices], self.y[batch_indices]

        def on_epoch_end(self):
            self.indices = balanced_epoch_indices()

    train_sequence = BalancedSequence(
        X_train, y_train, batch_size=32
    )

    history = model.fit(
        train_sequence,
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    model.save(MODEL_PATH)
    print("\nTraining complete.")
    print("Final model:", MODEL_PATH)
    print("Best checkpoint:", BEST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()
