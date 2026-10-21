from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Reuse the existing benchmark's exact preprocessing and noise functions.
spec = importlib.util.spec_from_file_location(
    "benchmark",
    "src/benchmark_all_models_noise.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

OUTPUT_DIR = Path("reports/rotor_consistent_three_models")

MODEL_PATHS = {
    "original": Path("models/rotor_consistent_cnn_best.keras"),
    "class_weighted": Path("models/rotor_consistent_cnn_balanced_best.keras"),
    "undersampled": Path("models/rotor_consistent_cnn_undersampled_best.keras"),
}

def main():
    for label, path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label} model: {path}\n"
                "Check the model filename before continuing."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading held-out test recordings...")
    recordings = benchmark.load_test_recordings()
    y_true = np.asarray([r["label"] for r in recordings], dtype=np.int32)
    filenames = [r["filename"] for r in recordings]

    print("\nLoading the two models...")
    bundles = {}
    for label, path in MODEL_PATHS.items():
        bundles[label] = benchmark.load_model_bundle(path)
        print(f"Loaded {label}: {path}")

    summary_rows = []
    class_rows = []
    confusion_rows = []
    prediction_rows = []
    start_time = time.time()

    for condition_index, (condition, snr_db) in enumerate(
        benchmark.CONDITIONS
    ):
        print(f"\n{'=' * 60}")
        print(f"Condition: {condition} (SNR: {snr_db})")
        print("Extracting shared test features...")

        logmel, mfcc, recording_indices = (
            benchmark.prepare_condition_features(
                recordings, snr_db, condition_index
            )
        )

        # Both models receive the exact same feature arrays.
        for label, bundle in bundles.items():
            window_preds = benchmark.predict_bundle(
                bundle, logmel, mfcc
            )
            y_pred = benchmark.recording_level_predictions(
                window_preds, recording_indices, len(recordings)
            )

            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(
                y_true, y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )

            summary_rows.append({
                "model": label,
                "model_file": MODEL_PATHS[label].name,
                "condition": condition,
                "snr_db": snr_db,
                "recordings": len(y_true),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            })

            report = classification_report(
                y_true, y_pred,
                labels=[0, 1, 2],
                target_names=benchmark.CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )

            for class_name in benchmark.CLASS_NAMES:
                item = report[class_name]
                class_rows.append({
                    "model": label,
                    "condition": condition,
                    "class": class_name,
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1-score"],
                    "support": item["support"],
                })

            matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            for i, actual_name in enumerate(benchmark.CLASS_NAMES):
                for j, predicted_name in enumerate(benchmark.CLASS_NAMES):
                    confusion_rows.append({
                        "model": label,
                        "condition": condition,
                        "actual_class": actual_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[i, j]),
                    })

            for filename, actual, predicted in zip(
                filenames, y_true, y_pred
            ):
                prediction_rows.append({
                    "model": label,
                    "condition": condition,
                    "filename": filename,
                    "actual_class": benchmark.CLASS_NAMES[int(actual)],
                    "predicted_class": benchmark.CLASS_NAMES[int(predicted)],
                    "correct": bool(actual == predicted),
                })

            print(
                f"{label:8s} | accuracy={accuracy:.3f} | "
                f"macro-F1={macro_f1:.3f} | "
                f"insect recall={report['insect']['recall']:.3f}"
            )

        del logmel, mfcc, recording_indices

    summary = pd.DataFrame(summary_rows)
    classes = pd.DataFrame(class_rows)

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    classes.to_csv(OUTPUT_DIR / "class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        OUTPUT_DIR / "recording_predictions.csv", index=False
    )

    # Direct balanced-minus-original differences for each condition.
    comparison_rows = []
    for condition, group in summary.groupby("condition", sort=False):
        original = group[group["model"] == "original"].iloc[0]
        balanced = group[group["model"] == "balanced"].iloc[0]

        original_insect = classes[
            (classes["model"] == "original")
            & (classes["condition"] == condition)
            & (classes["class"] == "insect")
        ].iloc[0]

        balanced_insect = classes[
            (classes["model"] == "balanced")
            & (classes["condition"] == condition)
            & (classes["class"] == "insect")
        ].iloc[0]

        comparison_rows.append({
            "condition": condition,
            "original_accuracy": original["accuracy"],
            "balanced_accuracy": balanced["accuracy"],
            "accuracy_change_pp": (
                balanced["accuracy"] - original["accuracy"]
            ) * 100,
            "original_macro_f1": original["macro_f1"],
            "balanced_macro_f1": balanced["macro_f1"],
            "macro_f1_change": balanced["macro_f1"] - original["macro_f1"],
            "original_insect_recall": original_insect["recall"],
            "balanced_insect_recall": balanced_insect["recall"],
            "insect_recall_change_pp": (
                balanced_insect["recall"] - original_insect["recall"]
            ) * 100,
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    manifest = {
        "models": {
            label: str(path) for label, path in MODEL_PATHS.items()
        },
        "test_recordings": len(recordings),
        "conditions": [
            {"name": name, "snr_db": snr}
            for name, snr in benchmark.CONDITIONS
        ],
        "class_order": benchmark.CLASS_NAMES,
        "seed": benchmark.SEED,
        "window_seconds": benchmark.WINDOW_SIZE / benchmark.SAMPLE_RATE,
        "window_hop_seconds": benchmark.WINDOW_HOP / benchmark.SAMPLE_RATE,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print(f"Results folder: {OUTPUT_DIR}")
    print("\nBalanced minus original:")
    print(comparison.round(4).to_string(index=False))
    print("\nImportant: positive insect_recall_change_pp means "
          "the balanced model detected more insect recordings.")

if __name__ == "__main__":
    main()
