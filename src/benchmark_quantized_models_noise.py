from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

AUDIO_DIR = Path("data/raw/ESC-50-master/audio")
METADATA_PATH = Path("data/metadata/dataset_split.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("reports/quantized_models_noise_benchmark")

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
SEED = 42

LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CLASS_NAMES = ["bird", "insect", "background"]

# Each condition uses a separate reproducible noise stream.
CONDITIONS = [
    ("clean", None),
    ("rotor_plus5db", 5.0),
    ("rotor_0db", 0.0),
    ("rotor_minus5db", -5.0),
    ("rotor_minus10db", -10.0),
]

NORMALIZATION_FILES = {
    "mfcc_mlp": "models/mfcc_normalization.npz",
    "mfcc_mlp_v2": "models/mfcc_normalization_v2.npz",
    "mfcc_cnn1d": "models/mfcc_cnn1d_normalization.npz",
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
        rotor += np.sin(
            2.0 * np.pi * fundamental * harmonic * t + phase
        ) / harmonic

    modulation_frequency = rng.uniform(15.0, 45.0)
    modulation = 0.55 + 0.45 * np.sin(
        2.0 * np.pi * modulation_frequency * t
    )
    rotor *= modulation
    rotor += rng.normal(0.0, 0.12, len(signal)).astype(np.float32)
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    rotor *= np.sqrt(
        target_noise_power / (np.mean(rotor ** 2) + 1e-12)
    )
    return (signal + rotor).astype(np.float32)


def extract_logmel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_mfcc(audio):
    return librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).astype(np.float32)


def load_test_recordings():
    metadata = pd.read_csv(METADATA_PATH)
    test = metadata[metadata["split"] == "test"].copy()
    test = test.reset_index(drop=True)

    if test.empty:
        raise ValueError("No test recordings found.")

    recordings = []
    for row in test.itertuples(index=False):
        path = AUDIO_DIR / row.filename
        if not path.is_file():
            raise FileNotFoundError(path)

        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate: {path}")

        recordings.append({
            "filename": row.filename,
            "label": LABEL_MAP[row.group],
            "audio": audio.astype(np.float32),
        })

    print(f"Loaded {len(recordings)} test recordings.")
    print("Class counts:", test["group"].value_counts().to_dict())
    return recordings


def discover_models():
    search_dirs = [MODEL_DIR, MODEL_DIR / "quantized_int8"]
    paths = sorted(
        p
        for folder in search_dirs
        if folder.is_dir()
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".keras", ".h5", ".tflite"}
    )
    if not paths:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
    return paths


def load_normalization(model_name):
    # Resolve the base family for checkpoints and TFLite exports.
    if model_name.startswith("mfcc_mlp_v2"):
        key = "mfcc_mlp_v2"
    elif model_name.startswith("mfcc_cnn1d"):
        key = "mfcc_cnn1d"
    elif model_name.startswith("mfcc_mlp"):
        key = "mfcc_mlp"
    else:
        return None

    path = Path(NORMALIZATION_FILES[key])
    if not path.is_file():
        raise FileNotFoundError(
            f"Required normalization file missing for {model_name}: {path}"
        )

    with np.load(path) as data:
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)

    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def load_model_bundle(path):
    name = path.stem
    is_mfcc = name.startswith("mfcc_")
    normalization = load_normalization(name) if is_mfcc else None

    if path.suffix.lower() == ".tflite":
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        bundle = {
            "name": name,
            "path": str(path),
            "format": "tflite",
            "family": "mfcc" if is_mfcc else "logmel",
            "normalization": normalization,
            "interpreter": interpreter,
            "input_detail": interpreter.get_input_details()[0],
            "output_detail": interpreter.get_output_details()[0],
        }
    else:
        model = tf.keras.models.load_model(str(path), compile=False)
        bundle = {
            "name": name,
            "path": str(path),
            "format": "keras",
            "family": "mfcc" if is_mfcc else "logmel",
            "normalization": normalization,
            "model": model,
        }

    return bundle


def prepare_condition_features(recordings, snr_db, condition_index):
    """Extract each condition once so all models receive identical audio."""
    rng = np.random.default_rng(SEED + condition_index * 1000)

    logmel_rows = []
    mfcc_rows = []
    recording_indices = []

    for rec_index, rec in enumerate(recordings):
        audio = rec["audio"]
        window_count = 0

        for start in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP):
            window = audio[start:start + WINDOW_SIZE].astype(np.float32)

            if snr_db is not None:
                window = add_rotor_noise(window, snr_db, rng)

            logmel_rows.append(extract_logmel(window))
            mfcc_rows.append(extract_mfcc(window))
            recording_indices.append(rec_index)
            window_count += 1

        if window_count == 0:
            raise ValueError(f"No full one-second windows for {rec['filename']}")

    return (
        np.asarray(logmel_rows, dtype=np.float32),
        np.asarray(mfcc_rows, dtype=np.float32),
        np.asarray(recording_indices, dtype=np.int32),
    )


def prepare_model_input(bundle, logmel, mfcc):
    if bundle["family"] == "logmel":
        x = logmel[..., np.newaxis]
    else:
        x = mfcc.copy()
        mean, std = bundle["normalization"]
        x = (x - mean) / std

        if "cnn1d" in bundle["name"]:
            x = np.transpose(x, (0, 2, 1))

    return np.asarray(x, dtype=np.float32)


def predict_bundle(bundle, logmel, mfcc):
    x = prepare_model_input(bundle, logmel, mfcc)

    if bundle["format"] == "keras":
        scores = bundle["model"].predict(x, batch_size=64, verbose=0)
    else:
        interpreter = bundle["interpreter"]
        inp = bundle["input_detail"]
        out = bundle["output_detail"]
        input_index = inp["index"]
        output_index = out["index"]

        input_dtype = inp["dtype"]
        input_scale, input_zero = inp["quantization"]

        predictions = []
        for sample in x:
            sample = sample[np.newaxis, ...]

            if np.issubdtype(input_dtype, np.integer):
                if input_scale <= 0:
                    raise ValueError(
                        f"Invalid input quantization for {bundle['name']}"
                    )
                limits = np.iinfo(input_dtype)
                sample = np.rint(sample / input_scale + input_zero)
                sample = np.clip(sample, limits.min, limits.max)
                sample = sample.astype(input_dtype)
            else:
                sample = sample.astype(input_dtype)

            interpreter.set_tensor(input_index, sample)
            interpreter.invoke()
            result = interpreter.get_tensor(output_index)[0]

            # Dequantize integer outputs for consistent score handling.
            if np.issubdtype(out["dtype"], np.integer):
                scale, zero = out["quantization"]
                if scale > 0:
                    result = (result.astype(np.float32) - zero) * scale

            predictions.append(result)

        scores = np.asarray(predictions)

    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError(
            f"{bundle['name']} returned unexpected output shape {scores.shape}"
        )

    return np.argmax(scores, axis=1)


def recording_level_predictions(window_preds, recording_indices, n_recordings):
    result = []
    for rec_index in range(n_recordings):
        values = window_preds[recording_indices == rec_index]
        if len(values) == 0:
            raise ValueError(f"No predictions for recording index {rec_index}")
        counts = np.bincount(values, minlength=3)
        result.append(int(np.argmax(counts)))
    return np.asarray(result, dtype=np.int32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recordings = load_test_recordings()
    paths = discover_models()

    print(f"\nDiscovered {len(paths)} model files.")
    print("Loading models...")
    bundles = []
    load_errors = []

    for path in paths:
        try:
            bundle = load_model_bundle(path)
            bundles.append(bundle)
            print(f"  Loaded: {path.name}")
        except Exception as exc:
            load_errors.append({
                "model_file": path.name,
                "stage": "load",
                "error": repr(exc),
            })
            print(f"  LOAD FAILED: {path.name}: {exc}")

    if not bundles:
        raise RuntimeError("No models loaded successfully.")

    y_true = np.asarray([r["label"] for r in recordings], dtype=np.int32)
    filenames = [r["filename"] for r in recordings]
    summary_rows = []
    class_rows = []
    confusion_rows = []
    prediction_rows = []

    start_time = time.time()

    for condition_index, (condition, snr_db) in enumerate(CONDITIONS):
        print(f"\n{'=' * 65}")
        print(f"Condition: {condition} | SNR: {snr_db}")
        print("Preparing shared log-mel and MFCC features...")

        logmel, mfcc, recording_indices = prepare_condition_features(
            recordings, snr_db, condition_index
        )
        print(f"Feature windows: {len(recording_indices)}")

        for model_index, bundle in enumerate(bundles, start=1):
            try:
                window_preds = predict_bundle(bundle, logmel, mfcc)
                y_pred = recording_level_predictions(
                    window_preds, recording_indices, len(recordings)
                )

                summary_rows.append({
                    "model_file": Path(bundle["path"]).name,
                    "model_name": bundle["name"],
                    "format": bundle["format"],
                    "feature_family": bundle["family"],
                    "condition": condition,
                    "snr_db": snr_db,
                    "recordings": len(y_true),
                    "accuracy": accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(
                        y_true, y_pred, labels=[0, 1, 2],
                        average="macro", zero_division=0,
                    ),
                })

                report = classification_report(
                    y_true, y_pred,
                    labels=[0, 1, 2],
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0,
                )

                for class_name in CLASS_NAMES:
                    item = report[class_name]
                    class_rows.append({
                        "model_file": Path(bundle["path"]).name,
                        "condition": condition,
                        "class": class_name,
                        "precision": item["precision"],
                        "recall": item["recall"],
                        "f1": item["f1-score"],
                        "support": item["support"],
                    })

                matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                for actual_index, actual_name in enumerate(CLASS_NAMES):
                    for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                        confusion_rows.append({
                            "model_file": Path(bundle["path"]).name,
                            "condition": condition,
                            "actual_class": actual_name,
                            "predicted_class": predicted_name,
                            "count": int(matrix[actual_index, predicted_index]),
                        })

                for filename, actual, predicted in zip(filenames, y_true, y_pred):
                    prediction_rows.append({
                        "model_file": Path(bundle["path"]).name,
                        "condition": condition,
                        "filename": filename,
                        "actual_class": CLASS_NAMES[int(actual)],
                        "predicted_class": CLASS_NAMES[int(predicted)],
                        "correct": bool(actual == predicted),
                    })

                model_accuracy = accuracy_score(y_true, y_pred)
                model_macro_f1 = f1_score(
                    y_true, y_pred, labels=[0, 1, 2],
                    average="macro", zero_division=0,
                )
                print(
                    f"[{model_index:02d}/{len(bundles):02d}] "
                    f"{bundle['name']}: "
                    f"accuracy={model_accuracy:.3f}, "
                    f"macro-F1={model_macro_f1:.3f}"
                )

            except Exception as exc:
                load_errors.append({
                    "model_file": Path(bundle["path"]).name,
                    "stage": f"inference:{condition}",
                    "error": repr(exc),
                })
                print(f"  INFERENCE FAILED: {bundle['name']}: {exc}")

        # Release condition-specific feature arrays before the next condition.
        del logmel, mfcc, recording_indices

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUTPUT_DIR / "class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "recording_predictions.csv", index=False
    )
    pd.DataFrame(load_errors, columns=["model_file", "stage", "error"]).to_csv(
        OUTPUT_DIR / "errors.csv", index=False
    )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.sort_values(
            ["condition", "macro_f1", "accuracy"],
            ascending=[True, False, False],
        ).to_csv(OUTPUT_DIR / "summary_ranked.csv", index=False)

    manifest = {
        "models_discovered": len(paths),
        "models_loaded": len(bundles),
        "test_recordings": len(recordings),
        "conditions": [
            {"name": name, "snr_db": snr} for name, snr in CONDITIONS
        ],
        "class_order": CLASS_NAMES,
        "seed": SEED,
        "window_seconds": WINDOW_SIZE / SAMPLE_RATE,
        "window_hop_seconds": WINDOW_HOP / SAMPLE_RATE,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 65}")
    print("BENCHMARK FINISHED")
    print(f"Discovered models: {len(paths)}")
    print(f"Loaded models: {len(bundles)}")
    print(f"Errors: {len(load_errors)}")
    print(f"Results folder: {OUTPUT_DIR}")
    print(f"Elapsed seconds: {manifest['elapsed_seconds']}")

    if summary_rows:
        print("\nTop models by macro-F1 in each condition:")
        summary_df = pd.DataFrame(summary_rows)
        for condition, group in summary_df.groupby("condition", sort=False):
            print(f"\n{condition}")
            print(
                group.sort_values(
                    ["macro_f1", "accuracy"], ascending=False
                )[["model_name", "accuracy", "macro_f1"]]
                .head(5).to_string(index=False)
            )


if __name__ == "__main__":
    main()
