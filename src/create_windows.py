import librosa
import numpy as np

# One ESC-50 recording
audio_path = "data/raw/ESC-50-master/audio/1-100038-A-14.wav"

# Load audio
audio, sample_rate = librosa.load(audio_path, sr=None)

# 1-second window
window_size = sample_rate
window = audio[:window_size]

# Create mel spectrogram
mel = librosa.feature.melspectrogram(
    y=window,
    sr=sample_rate,
    n_mels=40,
    n_fft=2048,
    hop_length=512
)

# Convert to log scale
log_mel = librosa.power_to_db(mel, ref=np.max)

print("Sample rate:", sample_rate)
print("Audio samples:", len(window))
print("Mel spectrogram shape:", mel.shape)
print("Log-mel spectrogram shape:", log_mel.shape)
print("Minimum value:", log_mel.min())
print("Maximum value:", log_mel.max())