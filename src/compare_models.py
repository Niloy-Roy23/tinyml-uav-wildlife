import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, classification_report

data = np.load("data/processed/windowed_dataset.npz")
X = data["X"][data["splits"] == "test"][..., np.newaxis]
y = data["y"][data["splits"] == "test"]

models = {
    "Baseline": "models/baseline_cnn.keras",
    "Early stopping": "models/improved_cnn.keras",
    "Class weighted": "models/weighted_cnn.keras",
    "Augmentation": "models/augmentation_cnn.keras",
}

for name, path in models.items():
    model = keras.models.load_model(path)
    predictions = np.argmax(model.predict(X, verbose=0), axis=1)

    print(f"\n{name} model")
    print("=" * 40)
    print(f"Test accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"Macro F1:      {f1_score(y, predictions, average='macro'):.4f}")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1, 2],
        target_names=["bird", "insect", "background"],
        zero_division=0
    ))
