from pathlib import Path
from datetime import datetime

import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("baseline_cnn_mild", "models/baseline_cnn_mild.keras"),
    ("improved_cnn_mild", "models/improved_cnn_mild.keras"),
    ("rotor_augmentation_cnn_mild", "models/rotor_augmentation_cnn_mild_best.keras"),
    ("rotor_consistent_cnn_mild", "models/rotor_consistent_cnn_mild_best.keras"),
    ("mfcc_cnn1d_mild", "models/mfcc_cnn1d_mild_best.keras"),
    ("mfcc_mlp_v2_mild", "models/mfcc_mlp_v2_mild_best.keras"),
    ("mfcc_mlp_mild", "models/mfcc_mlp_mild_best.keras"),
]

out = ROOT / "reports" / (
    "model_technical_inventory_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
out.mkdir(parents=True, exist_ok=False)

rows = []
detail_lines = []

for name, relative_path in MODELS:
    path = ROOT / relative_path

    if not path.exists():
        print("SKIPPED — missing:", path)
        continue

    print("\nInspecting:", name)
    model = tf.keras.models.load_model(path, compile=False)

    trainable = sum(
        int(tf.keras.backend.count_params(w))
        for w in model.trainable_weights
    )
    non_trainable = sum(
        int(tf.keras.backend.count_params(w))
        for w in model.non_trainable_weights
    )
    file_mb = path.stat().st_size / (1024 * 1024)

    rows.append({
        "model": name,
        "file": relative_path,
        "file_size_MB": round(file_mb, 4),
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "total_parameters": model.count_params(),
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "layer_count": len(model.layers),
    })

    detail_lines.extend([
        "\n" + "=" * 70 + "\n",
        name + "\n",
        "File: " + relative_path + "\n",
        "File size: " + str(round(file_mb, 4)) + " MB\n",
        "Input shape: " + str(model.input_shape) + "\n",
        "Output shape: " + str(model.output_shape) + "\n\n",
    ])

    model.summary(
        print_fn=lambda line: detail_lines.append(line + "\n")
    )

    detail_lines.append("\nLayer details:\n")
    for layer in model.layers:
        try:
            output_shape = str(layer.output.shape)
        except (AttributeError, ValueError):
            output_shape = "unavailable"

        detail_lines.append(
            f"{layer.name}: {layer.__class__.__name__}; "
            f"output_shape={output_shape}; "
            f"parameters={layer.count_params():,}\n"
        )

    del model

if not rows:
    raise RuntimeError("No model files were found.")

summary = pd.DataFrame(rows)
summary.to_csv(out / "model_inventory.csv", index=False)

(out / "model_architecture_details.txt").write_text(
    "".join(detail_lines), encoding="utf-8"
)

print("\nMODEL INVENTORY")
print(summary.to_string(index=False))
print("\nDetailed architecture report:")
print(out / "model_architecture_details.txt")
print("\nInventory CSV:")
print(out / "model_inventory.csv")
