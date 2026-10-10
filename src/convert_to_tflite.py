from pathlib import Path
import tensorflow as tf

project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "models" / "baseline_cnn.keras"
output_path = project_dir / "models" / "baseline_cnn.tflite"

print(f"Loading model: {model_path}")
model = tf.keras.models.load_model(model_path)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

output_path.write_bytes(tflite_model)
print(f"Saved model: {output_path}")
print(f"File size: {len(tflite_model):,} bytes")
