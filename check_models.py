from pathlib import Path
import os
import csv
import json

ROOT = Path.cwd()
OUTPUT = ROOT / "model_inventory.csv"

MODEL_EXTENSIONS = {
    ".tflite", ".lite", ".keras", ".h5", ".hdf5",
    ".onnx", ".pt", ".pth", ".pb", ".safetensors"
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules"
}

def inspect_tflite(path):
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()

        tensors = interpreter.get_tensor_details()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()

        counts = {}
        quantized_count = 0

        for tensor in tensors:
            dtype = str(tensor["dtype"]).replace("<class '", "").replace("'>", "")
            counts[dtype] = counts.get(dtype, 0) + 1

            scales = tensor.get("quantization_parameters", {}).get("scales", [])
            if len(scales) > 0:
                quantized_count += 1

        dtypes = set(counts)

        if "int8" in dtypes:
            classification = "INT8 tensors present (check inputs/outputs)"
        elif "uint8" in dtypes:
            classification = "UINT8 tensors present"
        elif dtypes == {"float32"}:
            classification = "FLOAT32 tensors only"
        elif "float32" in dtypes:
            classification = "MIXED / includes FLOAT32"
        else:
            classification = "OTHER / MIXED TYPES"

        return {
            "classification": classification,
            "tensor_dtype_counts": counts,
            "quantized_tensor_count": quantized_count,
            "inputs": [
                {"dtype": str(x["dtype"]), "shape": x["shape"].tolist()}
                for x in inputs
            ],
            "outputs": [
                {"dtype": str(x["dtype"]), "shape": x["shape"].tolist()}
                for x in outputs
            ],
        }

    except Exception as e:
        return {"classification": "INSPECTION FAILED", "error": str(e)}

def inspect_keras(path):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path, compile=False)

        counts = {}
        for weight in model.weights:
            dtype = str(weight.dtype)
            counts[dtype] = counts.get(dtype, 0) + 1

        return {
            "classification": (
                "FLOAT32 weights"
                if set(counts).issubset({"float32"})
                else "MIXED / OTHER WEIGHTS"
            ),
            "weight_dtype_counts": counts,
            "parameters": model.count_params(),
        }

    except Exception as e:
        return {"classification": "INSPECTION FAILED", "error": str(e)}

def main():
    models = []

    for current, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        folder = Path(current)

        # TensorFlow SavedModel directories
        if (folder / "saved_model.pb").exists():
            models.append(folder)
            dirs[:] = []

        for filename in files:
            path = folder / filename
            if path.suffix.lower() in MODEL_EXTENSIONS:
                models.append(path)

    models = sorted(set(models))

    if not models:
        print("No model files found in:", ROOT)
        return

    rows = []

    for path in models:
        size = (
            sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            if path.is_dir()
            else path.stat().st_size
        )

        suffix = path.suffix.lower()

        if suffix in {".tflite", ".lite"}:
            details = inspect_tflite(path)
        elif suffix in {".keras", ".h5", ".hdf5"}:
            details = inspect_keras(path)
        else:
            details = {
                "classification": "FORMAT FOUND; DTYPE NOT INSPECTED"
            }

        relative_path = str(path.relative_to(ROOT))

        print("\n" + "=" * 60)
        print("Model:", relative_path)
        print("Format:", "SavedModel" if path.is_dir() else suffix)
        print(f"Size: {size:,} bytes ({size / 1024:.1f} KB)")
        print("Classification:", details["classification"])
        print(json.dumps(details, indent=2, default=str))

        rows.append({
            "model_path": relative_path,
            "format": "SavedModel" if path.is_dir() else suffix,
            "size_bytes": size,
            "classification": details["classification"],
            "details": json.dumps(details, default=str),
        })

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nModels found:", len(rows))
    print("Report saved to:", OUTPUT)

if __name__ == "__main__":
    main()
