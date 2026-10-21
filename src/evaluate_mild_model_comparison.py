from datetime import datetime
from pathlib import Path
import json

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
CLASS_NAMES = ["bird", "insect", "background"]
LABELS = [0, 1, 2]

LOGMEL_DATA = ROOT / "data/processed/windowed_dataset.npz"
MFCC_DATA = ROOT / "data/processed/mfcc/mfcc_dataset.npz"

# Evaluate saved best checkpoints where available.
MODELS = [
    {
        "name": "baseline_cnn_mild",
        "type": "logmel",
        "model": ROOT / "models/baseline_cnn_mild.keras",
    },
    {
        "name": "improved_cnn_mild",
        "type": "logmel",
        "model": ROOT / "models/improved_cnn_mild.keras",
    },
    {
        "name": "rotor_augmentation_cnn_mild",
        "type": "logmel",
        "model": ROOT / "models/rotor_augmentation_cnn_mild_best.keras",
    },
    {
        "name": "rotor_consistent_cnn_mild",
        "type": "logmel",
        "model": ROOT / "models/rotor_consistent_cnn_mild_best.keras",
    },
    {
        "name": "mfcc_cnn1d_mild",
        "type": "mfcc_cnn1d",
        "model": ROOT / "models/mfcc_cnn1d_mild_best.keras",
        "norm": ROOT / "models/mfcc_cnn1d_mild_normalization.npz",
    },
    {
        "name": "mfcc_mlp_v2_mild",
        "type": "mfcc_mlp",
        "model": ROOT / "models/mfcc_mlp_v2_mild_best.keras",
        "norm": ROOT / "models/mfcc_normalization_v2_mild.npz",
    },
    {
        "name": "mfcc_mlp_mild",
        "type": "mfcc_mlp",
        "model": ROOT / "models/mfcc_mlp_mild_best.keras",
        "norm": ROOT / "models/mfcc_normalization_mild.npz",
    },
]


def load_split(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        splits = data["splits"].astype(str)

    train_mask = splits == "train"
    test_mask = splits == "test"

    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError(f"Expected train and test splits in {path}")

    return X, y, train_mask, test_mask


def adapt_input_shape(X, model):
    """Match array layout to the saved model's declared input shape."""
    shape = model.input_shape

    if isinstance(shape, list):
        raise ValueError("This evaluator expects a model with one input.")

    expected = tuple(shape[1:])

    if X.ndim == 3 and len(expected) == 1:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 3 and len(expected) == 2:
        # MFCC CNN1D expects (time steps, coefficients).
        if expected == (X.shape[2], X.shape[1]):
            X = X.transpose(0, 2, 1)
    elif X.ndim == 3 and len(expected) == 3:
        if expected[-1] == 1:
            X = X[..., np.newaxis]

    actual = tuple(X.shape[1:])
    if len(expected) != len(actual):
        raise ValueError(
            f"Input rank mismatch: model expects {expected}, data has {actual}"
        )

    for want, got in zip(expected, actual):
        if want is not None and want != got:
            raise ValueError(
                f"Input shape mismatch: model expects {expected}, data has {actual}"
            )

    return X


def prepare_features(config, logmel, mfcc):
    if config["type"] == "logmel":
        X, y, _, test_mask = logmel
        X_test = X[test_mask][..., np.newaxis]
        y_test = y[test_mask]
    else:
        X, y, train_mask, test_mask = mfcc
        X_train = X[train_mask]
        X_test_raw = X[test_mask]
        y_test = y[test_mask]

        with np.load(config["norm"]) as norm:
            mean = norm["mean"].astype(np.float32)
            std = norm["std"].astype(np.float32)

        # Use the exact normalization saved by the corresponding training run.
        X_test = (X_test_raw - mean) / std

        if config["type"] == "mfcc_cnn1d":
            X_test = X_test.transpose(0, 2, 1)

    return X_test.astype(np.float32), y_test


def main():
    # Load both datasets once. All model evaluations use held-out test windows.
    logmel = load_split(LOGMEL_DATA)
    mfcc = load_split(MFCC_DATA)

    missing = []
    for config in MODELS:
        if not config["model"].exists():
            missing.append(str(config["model"]))
        if "norm" in config and not config["norm"].exists():
            missing.append(str(config["norm"]))

    if missing:
        raise FileNotFoundError(
            "Required model or normalization file(s) missing:\n"
            + "\n".join(missing)
        )

    # Timestamped directory avoids overwriting previous reports.
    report_dir = ROOT / "reports" / (
        "mild_model_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    report_dir.mkdir(parents=True, exist_ok=False)

    summary = []
    class_rows = []

    for config in MODELS:
        name = config["name"]
        print(f"\n{'=' * 65}")
        print("Evaluating:", name)

        X_test, y_test = prepare_features(config, logmel, mfcc)

        model = tf.keras.models.load_model(
            config["model"], compile=False
        )
        X_test = adapt_input_shape(X_test, model)

        print("Model input shape:", model.input_shape)
        print("Evaluation input shape:", X_test.shape)

        scores = model.predict(X_test, verbose=0)

        if scores.ndim != 2 or scores.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                f"{name}: expected three class scores, got {scores.shape}"
            )

        y_pred = np.argmax(scores, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test, y_pred, labels=LABELS, average="macro", zero_division=0
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)

        summary.append({
            "model": name,
            "feature_type": config["type"],
            "test_windows": len(y_test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "bird_recall": report["bird"]["recall"],
            "insect_recall": report["insect"]["recall"],
            "background_recall": report["background"]["recall"],
        })

        for class_name in CLASS_NAMES:
            class_rows.append({
                "model": name,
                "class": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1_score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            })

        pd.DataFrame(report).T.to_csv(
            report_dir / f"{name}_classification_report.csv"
        )
        pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in CLASS_NAMES],
            columns=[f"pred_{c}" for c in CLASS_NAMES],
        ).to_csv(report_dir / f"{name}_confusion_matrix.csv")

        print(f"Test windows: {len(y_test)}")
        print(f"Accuracy:     {accuracy:.4f}")
        print(f"Macro-F1:     {macro_f1:.4f}")
        print("Per-class results:")
        print(pd.DataFrame(report).T.round(4).to_string())
        print("Confusion matrix (rows=true, columns=predicted):")
        print(cm)

        del model

    summary_df = pd.DataFrame(summary).sort_values(
        "macro_f1", ascending=False
    )
    summary_df.to_csv(report_dir / "model_comparison_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(
        report_dir / "model_comparison_per_class.csv", index=False
    )

    print(f"\n{'=' * 65}")
    print("FINAL COMPARISON — sorted by macro-F1")
    print(summary_df.round(4).to_string(index=False))
    print("\nReports saved in:")
    print(report_dir)


if __name__ == "__main__":
    main()
