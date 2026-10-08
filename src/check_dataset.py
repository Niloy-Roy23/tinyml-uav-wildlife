import numpy as np

# Load dataset
data = np.load("data/processed/windowed_dataset.npz")

X = data["X"]
y = data["y"]
splits = data["splits"]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("\nUnique labels:", np.unique(y))
print("Unique splits:", np.unique(splits))

print("\nSamples by split:")
for split in ["train", "validation", "test"]:
    print(split, ":", np.sum(splits == split))

print("\nSamples by class:")
for label in [0, 1, 2]:
    print("Class", label, ":", np.sum(y == label))

print("\nFeature range:")
print("Minimum:", X.min())
print("Maximum:", X.max())

print("\nDataset verification complete.")