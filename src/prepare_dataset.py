import os
import numpy as np
import pandas as pd
import librosa

# Paths
metadata_path = "data/metadata/dataset_split.csv"
audio_dir = "data/raw/ESC-50-master/audio"
output_path = "data/processed/windowed_dataset.npz"

# Load metadata
df = pd.read_csv(metadata_path)

# Class labels
label_map = {
    "bird": 0,
    "insect": 1,
    "background": 2
}

# Audio / feature settings
sample_rate_expected = 44100
window_seconds = 1.0
hop_seconds = 0.5
n_mels = 40
n_fft = 2048
hop_length = 512

window_size = int(sample_rate_expected * window_seconds)
hop_size = int(sample_rate_expected * hop_seconds)

# Storage
features = []
labels = []
splits = []
filenames = []
window_indices = []

print("Processing recordings...")

for _, row in df.iterrows():

    audio_path = os.path.join(audio_dir, row["filename"])

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if sample_rate != sample_rate_expected:
        raise ValueError(
            f"Unexpected sample rate in {row['filename']}: "
            f"{sample_rate} Hz"
        )

    # Create overlapping 1-second windows
    window_number = 0

    for start in range(0, len(audio) - window_size + 1, hop_size):

        window = audio[start:start + window_size]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=window,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        features.append(log_mel.astype(np.float32))
        labels.append(label_map[row["group"]])
        splits.append(row["split"])
        filenames.append(row["filename"])
        window_indices.append(window_number)

        window_number += 1

print("Finished processing recordings.")

# Convert to NumPy arrays
X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.int64)
splits = np.array(splits)
filenames = np.array(filenames)
window_indices = np.array(window_indices)

# Save everything
os.makedirs("data/processed", exist_ok=True)

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    splits=splits,
    filenames=filenames,
    window_indices=window_indices
)

print()
print("Dataset created successfully.")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)

print()
print("Windows by split:")
for split in ["train", "validation", "test"]:
    print(split, ":", np.sum(splits == split))

print()
print("Windows by class:")
for class_name, label in label_map.items():
    print(class_name, ":", np.sum(y == label))

print()
print("Saved to:", output_path)