import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -----------------------------
# Load dataset
# -----------------------------
data = np.load("data/processed/windowed_dataset.npz")

X = data["X"]
y = data["y"]
splits = data["splits"]

# Test data only
X_test = X[splits == "test"]
y_test = y[splits == "test"]

# Add channel dimension
X_test = X_test[..., np.newaxis]

print("Test data:", X_test.shape)


# -----------------------------
# Load trained model
# -----------------------------
model = keras.models.load_model("models/baseline_cnn.keras")


# -----------------------------
# Make predictions
# -----------------------------
probabilities = model.predict(X_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)


# -----------------------------
# Calculate metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print()
print("Baseline Test Results")
print("---------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print()
print("Confusion Matrix")
print("----------------")
print(confusion_matrix(y_test, y_pred))

print()
print("Classification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["bird", "insect", "background"]
    )
)