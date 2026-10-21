
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "rotor_combined_cnn_best.keras"
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "windowed_dataset.npz"
OUTPUT_PATH = PROJECT_DIR / "models" / "rotor_combined_cnn_best_int8.tflite"

print("TensorFlow version:", tf.__version__)
print("Loading E7 model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Loading representative training data:", DATASET_PATH)

with np.load(DATASET_PATH) as data:
    X = data["X"]
    splits = data["splits"]

# Use training data only. Do not calibrate with validation or test windows.
X_train = X[splits == "train"]

if len(X_train) == 0:
    raise ValueError("No training windows found.")

# Model input shape is (batch, 40, 87, 1).
X_train = X_train[..., np.newaxis].astype(np.float32)

# Use a reproducible sample of 300 training windows for calibration.
rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(X_train),
    size=min(300, len(X_train)),
    replace=False,
)
representative_data = X_train[sample_indices]

def representative_dataset():
    for sample in representative_data:
        yield [sample[np.newaxis, ...]]

print("Training windows available:", len(X_train))
print("Calibration windows:", len(representative_data))
print("Converting to full INT8...")

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

print("Saved:", OUTPUT_PATH)
print("File size:", f"{len(tflite_model):,} bytes")

# Verify that the converted model accepts and returns INT8 tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

print("Input shape:", input_info["shape"])
print("Input dtype:", input_info["dtype"])
print("Output shape:", output_info["shape"])
print("Output dtype:", output_info["dtype"])

if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
    raise RuntimeError("Conversion did not produce INT8 input and output.")

print("INT8 conversion and verification successful.")
