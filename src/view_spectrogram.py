import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Audio file
audio_path = "data/raw/ESC-50-master/audio/1-100038-A-14.wav"

# Load audio
audio, sample_rate = librosa.load(audio_path, sr=None)

# Take the first 1-second window
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

# Plot
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    log_mel,
    sr=sample_rate,
    hop_length=512,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram - Chirping Birds")
plt.xlabel("Time (seconds)")
plt.ylabel("Mel Frequency")

plt.tight_layout()
plt.show()