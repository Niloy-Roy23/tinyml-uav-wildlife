from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
NORM_PATH = ROOT / "models" / "mfcc_normalization_v2.npz"
MODEL_PATH = ROOT / "models" / "mfcc_mlp_v2_best.keras"
OUTPUT_PATH = ROOT / "models" / "mfcc_mlp_v2_int8.tflite"

def main():
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        splits = data["splits"]

    with np.load(NORM_PATH) as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)

    # Representative calibration uses training samples only.
    X_train = X[splits == "train"]
    if len(X_train) == 0:
        raise ValueError("No training samples found.")
    X_train = (X_train - mean) / std

    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(X_train), size=min(300, len(X_train)), replace=False
    )
    representative_data = X_train[indices]

    model = tf.keras.models.load_model(MODEL_PATH)

    def representative_dataset():
        for sample in representative_data:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    OUTPUT_PATH.write_bytes(tflite_model)

    print("MLP v2 full-INT8 conversion successful.")
    print("Output:", OUTPUT_PATH)
    print("Model size:", OUTPUT_PATH.stat().st_size, "bytes")

    interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_PATH))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    print("Input shape:", input_info["shape"])
    print("Input dtype:", input_info["dtype"])
    print("Input quantization:", input_info["quantization"])
    print("Output shape:", output_info["shape"])
    print("Output dtype:", output_info["dtype"])
    print("Output quantization:", output_info["quantization"])

    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise RuntimeError("Input and output are not both INT8.")

if __name__ == "__main__":
    main()
