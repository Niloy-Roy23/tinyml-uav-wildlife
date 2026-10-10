
import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Load dataset
# -----------------------------
data = np.load("data/processed/windowed_dataset.npz")

X = data["X"]
y = data["y"]
splits = data["splits"]


# -----------------------------
# Separate train/validation/test
# -----------------------------
X_train = X[splits == "train"]
y_train = y[splits == "train"]

X_val = X[splits == "validation"]
y_val = y[splits == "validation"]

# Keep the test set separate for final evaluation
X_test = X[splits == "test"]
y_test = y[splits == "test"]


# -----------------------------
# Add channel dimension
# -----------------------------
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)


# -----------------------------
# Calculate class weights
# -----------------------------
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("Class weights:", class_weights)


# -----------------------------
# Build CNN model
# -----------------------------
model = keras.Sequential([
    layers.Input(shape=(40, 87, 1)),

    layers.Conv2D(
        16, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    layers.GlobalAveragePooling2D(),

    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax")
])


# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Prepare output directories
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# -----------------------------
# Training callbacks
# -----------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        filepath="models/weighted_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        "reports/weighted_training_history.csv"
    )
]


# -----------------------------
# Train class-weighted model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)


# -----------------------------
# Save final model and history
# -----------------------------
model.save("models/weighted_cnn_final.keras")

with open(
    "reports/weighted_training_history.json", "w"
) as f:
    json.dump(
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        f,
        indent=2
    )

print("\nTraining complete.")
print("Best checkpoint: models/weighted_cnn.keras")
print("Final model: models/weighted_cnn_final.keras")
print("CSV history: reports/weighted_training_history.csv")
print("JSON history: reports/weighted_training_history.json")
