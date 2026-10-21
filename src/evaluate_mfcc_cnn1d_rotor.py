from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "raw" / "ESC-50-master" / "audio"
METADATA_PATH = ROOT / "data" / "metadata" / "dataset_split.csv"
NORM_PATH = ROOT / "models" / "mfcc_cnn1d_normalization.npz"
FLOAT_MODEL_PATH = ROOT / "models" / "mfcc_cnn1d_best.keras"
INT8_MODEL_PATH = ROOT / "models" / "mfcc_cnn1d_int8.tflite"
OUTPUT_DIR = ROOT / "reports" / "cnn1d_evaluation"

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MFCC = 40
N_FFT = 2048
MFCC_HOP_LENGTH = 512
SEED = 42

CLASS_NAMES = ["bird", "insect", "background"]
LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CONDITIONS = [
    ("clean", None),
    ("rotor_0db", 0.0),
    ("rotor_minus10db", -10.0),
]


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


def extract_mfcc(audio):
    return librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=MFCC_HOP_LENGTH,
    ).astype(np.float32)


def float_predict(model, feature, mean, std):
    # CNN1D normalization has shape (1, 40, 1).
    # Remove the singleton dimensions to normalize a (40, 87) feature.
    mean_2d = np.asarray(mean).reshape(40, 1)
    std_2d = np.asarray(std).reshape(40, 1)
    normalized = (feature - mean_2d) / std_2d
    x = normalized.transpose(1, 0).astype(np.float32)
    scores = model.predict(x[np.newaxis, ...], verbose=0)
    return int(np.argmax(scores[0]))


def int8_predict(interpreter, feature, mean, std):
    mean_2d = np.asarray(mean).reshape(40, 1)
    std_2d = np.asarray(std).reshape(40, 1)
    normalized = (feature - mean_2d) / std_2d
    x = normalized.transpose(1, 0).astype(np.float32)

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    scale, zero_point = input_info["quantization"]
    out_scale, out_zero_point = output_info["quantization"]

    if scale <= 0 or out_scale <= 0:
        raise ValueError("Invalid INT8 quantization parameters.")

    q = np.round(x / scale + zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_info["index"], q[np.newaxis, ...])
    interpreter.invoke()
    output = interpreter.get_tensor(output_info["index"])
    scores = (output.astype(np.float32) - out_zero_point) * out_scale
    return int(np.argmax(scores[0]))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    test_metadata = metadata[metadata["split"] == "test"].copy()
    if test_metadata.empty:
        raise ValueError("No test recordings found.")

    with np.load(NORM_PATH) as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)

    float_model = tf.keras.models.load_model(FLOAT_MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=str(INT8_MODEL_PATH))
    interpreter.allocate_tensors()

    if interpreter.get_input_details()[0]["dtype"] != np.int8:
        raise ValueError("Expected INT8 input model.")
    if interpreter.get_output_details()[0]["dtype"] != np.int8:
        raise ValueError("Expected INT8 output model.")

    # Same noisy windows are used for both models.
    # A fixed seed makes this experiment reproducible.
    rng = np.random.default_rng(SEED)
    y_true = []
    all_predictions = {
        (model_name, condition): []
        for model_name in ["float", "int8"]
        for condition, _ in CONDITIONS
    }

    for row in test_metadata.itertuples(index=False):
        audio_path = AUDIO_DIR / row.filename
        audio, _ = librosa.load(
            audio_path, sr=SAMPLE_RATE, mono=True
        )
        if len(audio) < WINDOW_SIZE:
            audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

        recording_predictions = {
            (model_name, condition): []
            for model_name in ["float", "int8"]
            for condition, _ in CONDITIONS
        }

        for start in range(
            0, len(audio) - WINDOW_SIZE + 1, WINDOW_HOP
        ):
            window = audio[start:start + WINDOW_SIZE].astype(np.float32)

            # Generate each condition once, then feed identical audio
            # to the float and INT8 models.
            for condition, snr_db in CONDITIONS:
                tested_audio = (
                    window if snr_db is None
                    else add_rotor_noise(window, snr_db, rng)
                )
                feature = extract_mfcc(tested_audio)

                pred_float = float_predict(
                    float_model, feature, mean, std
                )
                pred_int8 = int8_predict(
                    interpreter, feature, mean, std
                )

                recording_predictions[("float", condition)].append(
                    pred_float
                )
                recording_predictions[("int8", condition)].append(
                    pred_int8
                )

        y_true.append(LABEL_MAP[row.group])

        for key, window_predictions in recording_predictions.items():
            counts = np.bincount(window_predictions, minlength=3)
            all_predictions[key].append(int(np.argmax(counts)))

        print("Evaluated:", row.filename)

    y_true = np.asarray(y_true, dtype=np.int64)
    summaries = []
    class_rows = []
    matrix_rows = []

    for (model_name, condition), values in all_predictions.items():
        y_pred = np.asarray(values, dtype=np.int64)
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

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

        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                matrix_rows.append({
                    "model": model_name,
                    "condition": condition,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "rotor_recording_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        OUTPUT_DIR / "rotor_recording_class_metrics.csv", index=False
    )
    pd.DataFrame(matrix_rows).to_csv(
        OUTPUT_DIR / "rotor_recording_confusion_matrices.csv",
        index=False,
    )

    print("\nRecording-level summary (72 recordings):")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClass-wise recall:")
    print(
        pd.DataFrame(class_rows)
        .pivot(index=["model", "condition"], columns="class", values="recall")
        .round(4)
        .to_string()
    )
    print("\nReports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
