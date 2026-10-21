
import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

AUDIO_DIR = "data/raw/ESC-50-master/audio"
METADATA_PATH = "data/metadata/dataset_split.csv"
OUTPUT_DIR = "reports/e3_rotor_augmented_comparison"

MODELS = {
    "e3_original": "models/rotor_classweighted_cnn_best.keras",
    "e3_rotor_augmented": "models/e3_rotor_augmented_cnn_best.keras",
}

SAMPLE_RATE = 44100
WINDOW_SIZE = SAMPLE_RATE
WINDOW_HOP = SAMPLE_RATE // 2
N_MELS = 40
N_FFT = 2048
MEL_HOP_LENGTH = 512
SEED = 42

LABEL_MAP = {"bird": 0, "insect": 1, "background": 2}
CLASS_NAMES = ["bird", "insect", "background"]
CONDITIONS = ["clean", "rotor_0db", "rotor_minus10db"]


def add_rotor_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    if signal_power < 1e-12:
        return signal.copy()

    t = np.arange(len(signal), dtype=np.float32) / SAMPLE_RATE
    fundamental = rng.uniform(80.0, 180.0)
    rotor = np.zeros(len(signal), dtype=np.float32)

    for harmonic in range(1, 6):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        rotor += np.sin(
            2.0 * np.pi * fundamental * harmonic * t + phase
        ) / harmonic

    modulation_frequency = rng.uniform(15.0, 45.0)
    modulation = 0.55 + 0.45 * np.sin(
        2.0 * np.pi * modulation_frequency * t
    )
    rotor *= modulation
    rotor += rng.normal(0.0, 0.12, len(signal)).astype(np.float32)
    rotor /= np.sqrt(np.mean(rotor ** 2) + 1e-8)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    rotor *= np.sqrt(target_noise_power / (np.mean(rotor ** 2) + 1e-12))
    return (signal + rotor).astype(np.float32)


def extract_feature(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def prepare_test_features(test_metadata):
    """Generate each test window/condition once for both models."""
    rng = np.random.default
