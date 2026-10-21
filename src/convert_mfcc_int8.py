
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "mfcc" / "mfcc_dataset.npz"
NORM_PATH = ROOT / "models" / "mfcc_normalization.npz"
MODEL_PATH = ROOT / "models" / "mfcc_mlp_best.keras"
OUTPUT_PATH = ROOT / "models" / "mfcc_mlp_int8.tflite"

REPRESENTATIVE_SAMPLES = 300
SEED = 42


def main():
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        splits = data["splits"]

    with np.load(NORM_PATH) as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)

    # Use training data only for representative calibration.
    X_train = X[splits == "train"]

    if len(X_train) == 0:
        raise ValueError("No training samples found.")

    # Match the preprocessing used to train the model.
    X_train = (X_train - mean) / std

    rng = np.random.default_rng(SEED)
    count = min(REPRESENTATIVE_SAMPLES, len(X_train))
    indices = rng.choice(len(X_train), size=count, replace=False)
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

    print("Corrected INT8 conversion successful.")
    print("Output:", OUTPUT_PATH)
    print("Model size:", OUTPUT_PATH.stat().st_size, "bytes")

    interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_PATH))
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    print("Input dtype:", input_info["dtype"])
    print("Input quantization:", input_info["quantization"])
    print("Output dtype:", output_info["dtype"])
    print("Output quantization:", output_info["quantization"])


if __name__ == "__main__":
    main()
