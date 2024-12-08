import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import fftconvolve
from math import pi

# Add the "ablage" directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ablage_dir = os.path.join(current_dir, "../ablage")
sys.path.insert(0, ablage_dir)

from add import add

# Function to load and preprocess audio files
def load_audio(filepath, downsample_factor=1):
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:  # Convert stereo to mono if necessary
        data = data.mean(axis=1)
    if downsample_factor > 1:
        data = data[::downsample_factor]
        sample_rate = sample_rate // downsample_factor
    return data, sample_rate

# Function to normalize signals
def normalize_signal(signal):
    return signal / np.max(np.abs(signal))

# Function to apply FFT and return magnitude and frequency components
def apply_fft(signal, sample_rate):
    n = len(signal)
    freq = fftfreq(n, d=1/sample_rate)
    fft_signal = fft(signal)
    return fft_signal, freq

# Function to apply IFFT to recover the signal
def apply_ifft(fft_signal):
    return np.real(ifft(fft_signal))

# Function to apply a low-pass filter in the frequency domain
def low_pass_filter_fft(fft_signal, freq, cutoff):
    filtered_signal = np.copy(fft_signal)
    filtered_signal[np.abs(freq) > cutoff] = 0
    return filtered_signal

# Function to perform fast convolution using FFT
def fast_convolution(signal, kernel):
    return fftconvolve(signal, kernel, mode='same')

# Parameters
downsample_factor = 10  # Reduce the sampling rate for easier processing
cutoff_freq = 4000  # Low-pass filter cutoff frequency in Hz
modulation_freq1 = 2000  # Modulation frequency for audio 1
modulation_freq2 = 4000  # Modulation frequency for audio 2

# Load audio files
soundfiles_dir = os.path.join(current_dir, "../soundfiles")
audio1_path = os.path.join(soundfiles_dir, "Avengers.wav")
audio2_path = os.path.join(soundfiles_dir, "ozapft.wav")

audio1, sample_rate1 = load_audio(audio1_path, downsample_factor)
audio2, sample_rate2 = load_audio(audio2_path, downsample_factor)

assert sample_rate1 == sample_rate2, "Sample rates of the audio files must match!"
sample_rate = sample_rate1

# Normalize signals
audio1 = normalize_signal(audio1)
audio2 = normalize_signal(audio2)

# Truncate signals to the same length
min_len = min(len(audio1), len(audio2))
audio1 = audio1[:min_len]
audio2 = audio2[:min_len]
time = np.arange(min_len) / sample_rate

# FFT of the original signals
fft_audio1, freq = apply_fft(audio1, sample_rate)
fft_audio2, _ = apply_fft(audio2, sample_rate)

# Apply low-pass filter in the frequency domain
filtered_fft_audio1 = low_pass_filter_fft(fft_audio1, freq, cutoff_freq)
filtered_fft_audio2 = low_pass_filter_fft(fft_audio2, freq, cutoff_freq)

# IFFT to recover filtered signals
filtered_audio1 = apply_ifft(filtered_fft_audio1)
filtered_audio2 = apply_ifft(filtered_fft_audio2)

# Modulate signals in the frequency domain
modulation1 = np.exp(2j * pi * modulation_freq1 * time)
modulation2 = np.exp(2j * pi * modulation_freq2 * time)
modulated_fft_audio1 = filtered_fft_audio1 * modulation1
modulated_fft_audio2 = filtered_fft_audio2 * modulation2

# Combine modulated signals
combined_fft_audio = modulated_fft_audio1 + modulated_fft_audio2

# Demodulate signals in the frequency domain
demodulated_fft_audio1 = combined_fft_audio * np.conj(modulation1)
demodulated_fft_audio2 = combined_fft_audio * np.conj(modulation2)

# Use fast convolution for signal reconstruction
recovered_audio1 = fast_convolution(apply_ifft(demodulated_fft_audio1), np.ones(256) / 256)
recovered_audio2 = fast_convolution(apply_ifft(demodulated_fft_audio2), np.ones(256) / 256)

# Normalize recovered signals
recovered_audio1 = normalize_signal(recovered_audio1)
recovered_audio2 = normalize_signal(recovered_audio2)

# Save recovered audio signals
output_dir = os.path.join(current_dir, "../results")
os.makedirs(output_dir, exist_ok=True)
wavfile.write(os.path.join(output_dir, "Recovered_Audio_1_FastConvolution.wav"), sample_rate, (recovered_audio1 * 32767).astype(np.int16))
wavfile.write(os.path.join(output_dir, "Recovered_Audio_2_FastConvolution.wav"), sample_rate, (recovered_audio2 * 32767).astype(np.int16))

# Visualization
plt.figure(figsize=(10, 6))

# Plot Original Signals
plt.figure()
plt.title("Original Signals")
plt.plot(time, audio1, label="Original Audio 1")
plt.plot(time, audio2, label="Original Audio 2", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot Filtered Signals
plt.figure()
plt.title("Filtered Signals")
plt.plot(time, filtered_audio1, label="Filtered Audio 1")
plt.plot(time, filtered_audio2, label="Filtered Audio 2", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot Modulated Signals
plt.figure()
plt.title("Modulated Signals")
plt.plot(time, np.real(apply_ifft(modulated_fft_audio1)), label="Modulated Audio 1")
plt.plot(time, np.real(apply_ifft(modulated_fft_audio2)), label="Modulated Audio 2", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot Combined Signal
plt.figure()
plt.title("Combined Signal")
plt.plot(time, np.real(apply_ifft(combined_fft_audio)), label="Combined Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot Recovered Signals
plt.figure()
plt.title("Recovered Signals (Fast Convolution)")
plt.plot(time, recovered_audio1, label="Recovered Audio 1")
plt.plot(time, recovered_audio2, label="Recovered Audio 2", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Plot Comparison Original vs Recovered
plt.figure()
plt.title("Comparison: Original vs Recovered Audio 1")
plt.plot(time, audio1, label="Original Audio 1", linestyle="--")
plt.plot(time, recovered_audio1, label="Recovered Audio 1", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Comparison: Original vs Recovered Audio 2")
plt.plot(time, audio2, label="Original Audio 2", linestyle="--")
plt.plot(time, recovered_audio2, label="Recovered Audio 2", alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
