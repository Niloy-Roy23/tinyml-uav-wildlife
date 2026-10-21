import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512

METADATA_PATH = Path("data/metadata/source_grouped_folds.csv")
AUDIO_DIR = Path("data/raw/ESC-50-master/audio")
OUTPUT_DIR = Path("reports/rotor_source_grouped_mild")
MODEL_DIR = OUTPUT_DIR / "models"

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


def load_recordings():
    df = pd.read_csv(METADATA_PATH)
    required = {"filename", "group", "src_file", "cv_fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    recordings = []
    for row in df.itertuples(index=False):
        if row.group not in LABEL_MAP:
            continue

        path = AUDIO_DIR / row.filename
        if not path.is_file():
            raise FileNotFoundError(path)

        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        recordings.append({
            "filename": row.filename,
            "src_file": str(row.src_file),
            "fold": int(row.cv_fold),
            "label": LABEL_MAP[row.group],
            "audio": audio.astype(np.float32),
        })

    # Enforce source independence before any training begins.
    source_folds = {}
    for r in recordings:
        source_folds.setdefault(r["src_file"], set()).add(r["fold"])
    crossing = [s for s, folds in source_folds.items() if len(folds) != 1]
    if crossing:
        raise ValueError(f"Source IDs cross folds: {crossing}")

    return recordings


def windows(audio):
    for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP):
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

    return (
        np.asarray(features, dtype=np.float32)[..., np.newaxis],
        np.asarray(labels, dtype=np.int64),
    )


def make_clean_examples(recordings):
    features, labels, owners = [], [], []
    for index, recording in enumerate(recordings):
        for window in windows(recording["audio"]):
            features.append(audio_to_log_mel(window))
            labels.append(recording["label"])
            owners.append(index)

    return (
        np.asarray(features, dtype=np.float32)[..., np.newaxis],
        np.asarray(labels, dtype=np.int64),
        np.asarray(owners, dtype=np.int64),
    )


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


def predict_recordings(model, recordings, snr_db, seed):
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
    owners = np.asarray(owners, dtype=np.int64)
    window_predictions = model.predict(X, batch_size=32, verbose=0).argmax(axis=1)

    predictions = []
    for index in range(len(recordings)):
        counts = np.bincount(
            window_predictions[owners == index], minlength=3
        )
        predictions.append(int(counts.argmax()))

    actual = np.asarray([r["label"] for r in recordings], dtype=np.int64)
    return actual, np.asarray(predictions, dtype=np.int64)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    recordings = load_recordings()
    folds = sorted({r["fold"] for r in recordings})
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected cv_fold values 1-5; found {folds}")

    summary_rows, prediction_rows = [], []

    for test_fold in folds:
        remaining = [f for f in folds if f != test_fold]
        # Rotate validation fold, without using the test fold.
        val_fold = remaining[(test_fold - 1) % len(remaining)]
        train_folds = [f for f in remaining if f != val_fold]

        train_records = [r for r in recordings if r["fold"] in train_folds]
        val_records = [r for r in recordings if r["fold"] == val_fold]
        test_records = [r for r in recordings if r["fold"] == test_fold]

        train_sources = {r["src_file"] for r in train_records}
        val_sources = {r["src_file"] for r in val_records}
        test_sources = {r["src_file"] for r in test_records}
        assert not train_sources & val_sources
        assert not train_sources & test_sources
        assert not val_sources & test_sources

        print(
            f"\nTEST FOLD {test_fold}: train={train_folds}, "
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
        X_val, y_val, _ = make_clean_examples(val_records)

        # Mild class weighting: preserve other classes while prioritizing insects.
        class_weight = {0: 0.875, 1: 1.25, 2: 1.0}
        print("Mild class weights:", class_weight)

        model = build_model()
        checkpoint = MODEL_DIR / f"fold_{test_fold}_best.keras"
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
            class_weight=class_weight,
            verbose=2,
        )

        # Load the best validation checkpoint for held-out testing.
        model = keras.models.load_model(checkpoint)

        for condition_index, (condition, snr_db) in enumerate(CONDITIONS):
            y_true, y_pred = predict_recordings(
                model,
                test_records,
                snr_db,
                SEED + test_fold * 1000 + condition_index,
            )

            insect_mask = y_true == LABEL_MAP["insect"]
            summary_rows.append({
                "test_fold": test_fold,
                "validation_fold": val_fold,
                "train_folds": ",".join(map(str, train_folds)),
                "condition": condition,
                "recordings": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=[0, 1, 2],
                    average="macro", zero_division=0,
                ),
                "insect_recall": float(
                    np.mean(y_pred[insect_mask] == LABEL_MAP["insect"])
                ),
            })

            for record, actual, predicted in zip(test_records, y_true, y_pred):
                prediction_rows.append({
                    "test_fold": test_fold,
                    "condition": condition,
                    "filename": record["filename"],
                    "src_file": record["src_file"],
                    "actual_class": CLASS_NAMES[int(actual)],
                    "predicted_class": CLASS_NAMES[int(predicted)],
                    "correct": bool(actual == predicted),
                })

        del X_train, y_train, X_val, y_val
        print(f"Finished test fold {test_fold}")

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

    print("\n=== Source-grouped five-fold results ===")
    print(aggregate.to_string(index=False))
    print(f"\nReports and models saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
