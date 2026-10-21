import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, recall_score

SEED = 42
SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512

METADATA_PATH = Path("data/metadata/dataset_split.csv")
AUDIO_DIR = Path("data/raw/ESC-50-master/audio")
OUTPUT_DIR = Path("reports/rotor_consistent_five_fold")

LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CLASS_NAMES = ["bird", "insect", "background"]
CONDITIONS = [
    ("clean", None),
    ("rotor_plus5db", 5.0),
    ("rotor_0db", 0.0),
    ("rotor_minus5db", -5.0),
    ("rotor_minus10db", -10.0),
]


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


def load_all_recordings():
    df = pd.read_csv(METADATA_PATH)
    recordings = []

    for _, row in df.iterrows():
        if row["group"] not in LABEL_MAP:
            continue

        path = AUDIO_DIR / row["filename"]
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        recordings.append({
            "filename": row["filename"],
            "fold": int(row["fold"]),
            "src_file": str(row["src_file"]),
            "label": LABEL_MAP[row["group"]],
            "audio": audio.astype(np.float32),
        })

    return recordings


def windows(audio):
    for start in range(
        0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP
    ):
        yield audio[start:start + WINDOW_SIZE]


def make_training_examples(recordings, rng):
    features, labels = [], []

    for recording in recordings:
        for window in windows(recording["audio"]):
            features.append(audio_to_log_mel(window))
            labels.append(recording["label"])

            for snr_db in (-10.0, -5.0, 0.0, 5.0):
                noisy = mix_rotor_noise(window, snr_db, rng)
                features.append(audio_to_log_mel(noisy))
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


def prepare_test_features(recordings, snr_db, seed):
    rng = np.random.default_rng(seed)
    features, owners = [], []

    for index, recording in enumerate(recordings):
        for window in windows(recording["audio"]):
            audio = (
                window if snr_db is None
                else mix_rotor_noise(window, snr_db, rng)
            )
            features.append(audio_to_log_mel(audio))
            owners.append(index)

    X = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    return X, np.asarray(owners, dtype=np.int64)


def evaluate_recordings(model, recordings, snr_db, seed):
    X, owners = prepare_test_features(recordings, snr_db, seed)
    probabilities = model.predict(X, batch_size=32, verbose=0)
    window_predictions = probabilities.argmax(axis=1)

    y_true, y_pred = [], []

    for index, recording in enumerate(recordings):
        predictions = window_predictions[owners == index]
        counts = np.bincount(predictions, minlength=3)
        y_pred.append(int(counts.argmax()))
        y_true.append(recording["label"])

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        probabilities,
        owners,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)

    recordings = load_all_recordings()
    fold_numbers = sorted({r["fold"] for r in recordings})

    if fold_numbers != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected folds 1-5, found {fold_numbers}")

    summary_rows = []
    prediction_rows = []

    for test_fold in fold_numbers:
        # Match the existing pipeline's three-train / one-validation /
        # one-test arrangement, rotating each fold through test.
        remaining_folds = [f for f in fold_numbers if f != test_fold]
        val_fold = remaining_folds[-1]
        train_folds = [f for f in remaining_folds if f != val_fold]

        train_records = [r for r in recordings if r["fold"] in train_folds]
        val_records = [r for r in recordings if r["fold"] == val_fold]
        test_records = [r for r in recordings if r["fold"] == test_fold]

        print(
            f"\nFOLD {test_fold}: train={train_folds}, "
            f"validation={val_fold}, test={test_fold}"
        )
        print(
            f"Recordings: train={len(train_records)}, "
            f"validation={len(val_records)}, test={len(test_records)}"
        )

        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + test_fold)

        rng = np.random.default_rng(SEED + test_fold)
        X_train, y_train = make_training_examples(train_records, rng)

        X_val_list, y_val_list = [], []
        for record in val_records:
            for window in windows(record["audio"]):
                X_val_list.append(audio_to_log_mel(window))
                y_val_list.append(record["label"])

        X_val = np.asarray(X_val_list, dtype=np.float32)[..., np.newaxis]
        y_val = np.asarray(y_val_list, dtype=np.int64)

        model = build_model()
        checkpoint = OUTPUT_DIR / "models" / f"fold_{test_fold}_best.keras"
        history_path = OUTPUT_DIR / f"fold_{test_fold}_history.csv"

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                str(checkpoint),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.CSVLogger(str(history_path)),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=2,
        )

        # Evaluate the held-out fold only after training is complete.
        for condition_index, (condition, snr_db) in enumerate(CONDITIONS):
            y_true, y_pred, _, _ = evaluate_recordings(
                model,
                test_records,
                snr_db,
                seed=SEED + test_fold * 1000 + condition_index,
            )

            insect_mask = y_true == LABEL_MAP["insect"]
            insect_recall = (
                float(np.mean(y_pred[insect_mask] == LABEL_MAP["insect"]))
                if insect_mask.any() else float("nan")
            )

            summary_rows.append({
                "test_fold": test_fold,
                "train_folds": ",".join(map(str, train_folds)),
                "validation_fold": val_fold,
                "condition": condition,
                "recordings": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=[0, 1, 2],
                    average="macro", zero_division=0
                ),
                "insect_recall": insect_recall,
            })

            for record, actual, predicted in zip(
                test_records, y_true, y_pred
            ):
                prediction_rows.append({
                    "test_fold": test_fold,
                    "condition": condition,
                    "filename": record["filename"],
                    "src_file": record["src_file"],
                    "actual_class": CLASS_NAMES[int(actual)],
                    "predicted_class": CLASS_NAMES[int(predicted)],
                    "correct": bool(actual == predicted),
                })

        model.save(OUTPUT_DIR / "models" / f"fold_{test_fold}_final.keras")

    summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows)

    summary.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "recording_predictions.csv", index=False)

    aggregate = (
        summary.groupby("condition")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            insect_recall_mean=("insect_recall", "mean"),
            insect_recall_std=("insect_recall", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)

    print("\n=== Five-fold mean and standard deviation ===")
    print(aggregate.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
