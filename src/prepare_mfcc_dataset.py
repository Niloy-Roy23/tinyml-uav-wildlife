
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESC50 = ROOT / "data" / "raw" / "ESC-50-master"
OUT = ROOT / "data" / "processed" / "mfcc"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44100
WINDOW_SAMPLES = SAMPLE_RATE       # 1 second
HOP_SAMPLES = SAMPLE_RATE // 2     # 0.5 seconds
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

CLASS_MAP = {
    "chirping_birds": 0,
    "crow": 0,
    "hen": 0,
    "rooster": 0,
    "insects": 1,
    "crickets": 1,
    "rain": 2,
    "wind": 2,
    "sea_waves": 2,
}

SPLIT_MAP = {
    1: "train",
    2: "train",
    3: "train",
    4: "validation",
    5: "test",
}


def main():
    metadata = pd.read_csv(ESC50 / "meta" / "esc50.csv")
    metadata = metadata[metadata["category"].isin(CLASS_MAP)].copy()

    features = []
    labels = []
    splits = []
    filenames = []
    window_indices = []

    for row in metadata.itertuples(index=False):
        audio_path = ESC50 / "audio" / row.filename

        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        if len(audio) < WINDOW_SAMPLES:
            audio = np.pad(
                audio,
                (0, WINDOW_SAMPLES - len(audio)),
            )

        label = CLASS_MAP[row.category]
        split = SPLIT_MAP[int(row.fold)]

        count = 0
        for start in range(
            0,
            len(audio) - WINDOW_SAMPLES + 1,
            HOP_SAMPLES,
        ):
            window = audio[start:start + WINDOW_SAMPLES]

            mfcc = librosa.feature.mfcc(
                y=window,
                sr=SAMPLE_RATE,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            ).astype(np.float32)

            features.append(mfcc)
            labels.append(label)
            splits.append(split)
            filenames.append(row.filename)
            window_indices.append(count)
            count += 1

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    splits = np.asarray(splits)
    filenames = np.asarray(filenames)
    window_indices = np.asarray(window_indices, dtype=np.int64)

    output_path = OUT / "mfcc_dataset.npz"
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        splits=splits,
        filenames=filenames,
        window_indices=window_indices,
    )

    print("MFCC dataset created successfully.")
    print("Saved to:", output_path)
    print("Features:", X.shape)
    print("Labels:", y.shape)
    print(
        "Windows by split:",
        {s: int(np.sum(splits == s))
         for s in ["train", "validation", "test"]},
    )
    print(
        "Windows by class:",
        {name: int(np.sum(y == label))
         for name, label in [
             ("bird", 0),
             ("insect", 1),
             ("background", 2),
         ]},
    )


if __name__ == "__main__":
    main()
