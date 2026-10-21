from pathlib import Path
import csv
import traceback

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = MODEL_DIR / "quantized_int8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGMEL_DATA = ROOT / "data/processed/windowed_dataset.npz"
MFCC_DATA = ROOT / "data/processed/mfcc/mfcc_dataset.npz"

NORMALIZATION = {
    "mfcc_mlp": MODEL_DIR / "mfcc_normalization.npz",
    "mfcc_mlp_v2": MODEL_DIR / "mfcc_normalization_v2.npz",
    "mfcc_cnn1d": MODEL_DIR / "mfcc_cnn1d_normalization.npz",
}

SAMPLES = 300
SEED = 42


def family_for(name):
    if name.startswith("mfcc_mlp_v2"):
        return "mfcc_mlp_v2"
    if name.startswith("mfcc_cnn1d"):
        return "mfcc_cnn1d"
    if name.startswith("mfcc_mlp"):
        return "mfcc_mlp"
    return "logmel"


def load_calibration_data(family):
    path = MFCC_DATA if family.startswith("mfcc_") else LOGMEL_DATA

    with np.load(path) as data:
        x = data["X"].astype(np.float32)
        splits = data["splits"].astype(str)

    x = x[splits == "train"]
    if len(x) == 0:
        raise ValueError(f"No training samples in {path}")

    if family.startswith("mfcc_"):
        norm_path = NORMALIZATION[family]
        with np.load(norm_path) as norm:
            mean = norm["mean"].astype(np.float32)
            std = norm["std"].astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        x = (x - mean) / std

        if family == "mfcc_cnn1d":
            x = np.transpose(x, (0, 2, 1))
    else:
        x = x[..., np.newaxis]

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(x), size=min(SAMPLES, len(x)), replace=False)
    return x[indices].astype(np.float32), len(x)


def convert_one(model_path):
    name = model_path.stem
    family = family_for(name)
    output_path = OUT_DIR / f"{name}_int8.tflite"

    if output_path.exists():
        return {
            "model": name, "status": "SKIPPED_EXISTS",
            "output": str(output_path), "bytes": output_path.stat().st_size,
            "detail": "Existing output preserved",
        }

    calibration, training_count = load_calibration_data(family)
    model = tf.keras.models.load_model(model_path, compile=False)

    def representative_dataset():
        for sample in calibration:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    content = converter.convert()

    # Verify the in-memory export before writing it.
    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise RuntimeError(
            f"Not full INT8: input={inp['dtype']}, output={out['dtype']}"
        )

    if tuple(inp["shape"][1:]) != tuple(model.input_shape[1:]):
        raise RuntimeError(
            f"Input shape mismatch: TFLite={inp['shape']}, "
            f"Keras={model.input_shape}"
        )

    # Write only after successful conversion and verification.
    output_path.write_bytes(content)

    return {
        "model": name, "status": "SUCCESS",
        "output": str(output_path), "bytes": len(content),
        "detail": (
            f"family={family}; training={training_count}; "
            f"calibration={len(calibration)}; "
            f"input={inp['shape'].tolist()}; output={out['shape'].tolist()}"
        ),
    }


def main():
    print("TensorFlow:", tf.__version__)
    print("Output directory:", OUT_DIR)
    print("Calibration: training split only; seed:", SEED)

    models = sorted(MODEL_DIR.glob("*.keras"))
    results = []

    for model_path in models:
        print(f"\n--- {model_path.name} ---")
        try:
            result = convert_one(model_path)
            print(result["status"], result["detail"])
        except Exception as exc:
            result = {
                "model": model_path.stem,
                "status": "FAILED",
                "output": "",
                "bytes": 0,
                "detail": f"{type(exc).__name__}: {exc}",
            }
            print("FAILED:", result["detail"])
            traceback.print_exc(limit=2)
        results.append(result)

    report_path = OUT_DIR / "conversion_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "status", "output", "bytes", "detail"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=== CONVERSION SUMMARY ===")
    for status in ("SUCCESS", "SKIPPED_EXISTS", "FAILED"):
        print(f"{status}: {sum(r['status'] == status for r in results)}")
    print("Report:", report_path)


if __name__ == "__main__":
    main()
