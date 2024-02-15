import numpy as np
import matplotlib.pyplot as plt

# Define parameters
sampling_freq = 1000  # Sampling frequency in Hz
signal_length = 1024  # Length of the random signal

# Generate random signal
random_signal = np.random.randn(signal_length)

# Compute power spectral density (PSD) using Fourier transform
frequencies = np.fft.fftfreq(signal_length, 1 / sampling_freq)
psd = np.abs(np.fft.fft(random_signal)) ** 2

# Plot the power spectral density
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:signal_length // 2], psd[:signal_length // 2])
plt.title('Power Spectral Density (PSD) of Random Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency')
plt.grid(True)
plt.show()