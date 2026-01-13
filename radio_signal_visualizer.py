import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# -----------------------------
# 1. Signal Parameters
# -----------------------------
sampling_rate = 1000  # Hz
duration = 2.0        # seconds
time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Frequencies of simulated radio signals
freq1 = 50   # Hz
freq2 = 120  # Hz

# -----------------------------
# 2. Signal Generation
# -----------------------------
signal_1 = np.sin(2 * np.pi * freq1 * time)
signal_2 = 0.6 * np.sin(2 * np.pi * freq2 * time)

# Combined clean signal
clean_signal = signal_1 + signal_2

# Add white Gaussian noise
noise = np.random.normal(0, 0.5, clean_signal.shape)
noisy_signal = clean_signal + noise

# -----------------------------
# 3. Short-Time Fourier Transform (STFT)
# -----------------------------
frequencies, times, Zxx = stft(
    noisy_signal,
    fs=sampling_rate,
    nperseg=256
)

# -----------------------------
# 4. Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
plt.title("Spectrogram of Simulated Radio Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Signal Intensity")
plt.tight_layout()
plt.show()
