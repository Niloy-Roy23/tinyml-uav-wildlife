import numpy as np
from collections import Counter

data = np.load("data/processed/windowed_dataset.npz")
y = data["y"]
splits = data["splits"]

names = {0: "bird", 1: "insect", 2: "background"}

for split in ["train", "validation", "test"]:
    mask = splits == split
    counts = Counter(y[mask].tolist())

    print(f"\n{split.upper()} SET")
    for label, name in names.items():
        print(f"{name:12}: {counts.get(label, 0)} windows")
