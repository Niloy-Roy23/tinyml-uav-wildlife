
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
keras.utils.set_random_seed(SEED)


# -----------------------------
# Spectrogram augmentation
# -----------------------------
def augment_spectrogram(spectrogram):
    # Randomly mask a small frequency band
    freq_width = tf.random.uniform(
        shape=[], minval=0, maxval=5, dtype=tf.int32
    )
    freq_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=40 - freq_width + 1,
        dtype=tf.int32
    )

    freq_mask = tf.logical_and(
        tf.range(40) >= freq_start,
        tf.range(40) < freq_start + freq_width
    )

    spectrogram = tf.where(
        freq_mask[:, tf.newaxis, tf.newaxis],
        tf.zeros_like(spectrogram),
        spectrogram
    )

    # Randomly mask a short time region
    time_width = tf.random.uniform(
        shape=[], minval=0, maxval=9, dtype=tf.int32
    )
    time_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=87 - time_width + 1,
        dtype=tf.int32
    )

    time_mask = tf.logical_and(
        tf.range(87) >= time_start,
        tf.range(87) < time_start + time_width
    )

    spectrogram = tf.where(
        time_mask[tf.newaxis, :, tf.newaxis],
        tf.zeros_like(spectrogram),
        spectrogram
    )

    return spectrogram


# -----------------------------
# Load dataset
# -----------------------------
data = np.load("data/processed/windowed_dataset.npz")

X = data["X"]
y = data["y"]
splits = data["splits"]


# -----------------------------
# Separate data splits
# -----------------------------
X_train = X[splits == "train"]
y_train = y[splits == "train"]

X_val = X[splits == "validation"]
y_val = y[splits == "validation"]

# Retained for later final evaluation.
# Do not use the test set during training.
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
# Build CNN
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
# Prepare datasets
# -----------------------------
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_ds = train_ds.shuffle(
    buffer_size=len(X_train),
    seed=SEED,
    reshuffle_each_iteration=True
)

# Augmentation applies only to training samples
train_ds = train_ds.map(
    lambda x, label: (augment_spectrogram(x), label),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# Validation data is not augmented
val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)


# -----------------------------
# Prepare output directories
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        filepath="models/augmentation_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        "reports/augmentation_training_history.csv"
    )
]


# -----------------------------
# Train augmentation model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)


# -----------------------------
# Save final model and history
# -----------------------------
model.save("models/augmentation_cnn_final.keras")

with open(
    "reports/augmentation_training_history.json", "w"
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
print("Best checkpoint: models/augmentation_cnn.keras")
print("Final model: models/augmentation_cnn_final.keras")
print("CSV history: reports/augmentation_training_history.csv")
print("JSON history: reports/augmentation_training_history.json")
