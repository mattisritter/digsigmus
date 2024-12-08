import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import fftconvolve, firwin
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

# Function to apply a Hamming filter
def apply_hamming_filter(signal, sample_rate, cutoff, numtaps=101):
    nyquist = sample_rate / 2
    if not (0 < cutoff < nyquist):
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")
    taps = firwin(numtaps, cutoff / nyquist, window="hamming")
    return fftconvolve(signal, taps, mode="same")

# Parameters
downsample_factor = 10  # Reduce the sampling rate for easier processing
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

# Apply Hamming filter
cutoff_freq = sample_rate / 4  # Low-pass filter cutoff frequency
filtered_audio1 = apply_hamming_filter(audio1, sample_rate, cutoff_freq)
filtered_audio2 = apply_hamming_filter(audio2, sample_rate, cutoff_freq)

# Modulate signals in the time domain
modulation1 = np.exp(2j * pi * modulation_freq1 * time)
modulation2 = np.exp(2j * pi * modulation_freq2 * time)
modulated_audio1 = fft(filtered_audio1) * modulation1
modulated_audio2 = fft(filtered_audio2) * modulation2

# Combine modulated signals
combined_audio = modulated_audio1 + modulated_audio2

# Demodulate signals in the time domain
demodulated_audio1 = combined_audio * np.conj(modulation1)
demodulated_audio2 = combined_audio * np.conj(modulation2)

# Apply Hamming filter for reconstruction
recovered_audio1 = apply_hamming_filter(apply_ifft(demodulated_audio1), sample_rate, cutoff_freq)
recovered_audio2 = apply_hamming_filter(apply_ifft(demodulated_audio2), sample_rate, cutoff_freq)

# Normalize recovered signals
recovered_audio1 = normalize_signal(recovered_audio1)
recovered_audio2 = normalize_signal(recovered_audio2)

# Save recovered audio signals
output_dir = os.path.join(current_dir, "../results")
os.makedirs(output_dir, exist_ok=True)
wavfile.write(os.path.join(output_dir, "Recovered_Audio_1_Hamming.wav"), sample_rate, (recovered_audio1 * 32767).astype(np.int16))
wavfile.write(os.path.join(output_dir, "Recovered_Audio_2_Hamming.wav"), sample_rate, (recovered_audio2 * 32767).astype(np.int16))

# Visualization
plt.figure(figsize=(18, 12))

# Original signals
plt.subplot(3, 2, 1)
plt.plot(time, audio1, label="Original Audio 1")
plt.plot(time, audio2, label="Original Audio 2", alpha=0.7)
plt.title("Original Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Filtered signals
plt.subplot(3, 2, 2)
plt.plot(time, filtered_audio1, label="Filtered Audio 1")
plt.plot(time, filtered_audio2, label="Filtered Audio 2", alpha=0.7)
plt.title("Filtered Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Modulated signals
plt.subplot(3, 2, 3)
plt.plot(time, np.real(apply_ifft(modulated_audio1)), label="Modulated Audio 1")
plt.plot(time, np.real(apply_ifft(modulated_audio2)), label="Modulated Audio 2", alpha=0.7)
plt.title("Modulated Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Combined signal
plt.subplot(3, 2, 4)
plt.plot(time, np.real(apply_ifft(combined_audio)), label="Combined Signal")
plt.title("Combined Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Recovered signals
plt.subplot(3, 2, 5)
plt.plot(time, recovered_audio1, label="Recovered Audio 1")
plt.plot(time, recovered_audio2, label="Recovered Audio 2", alpha=0.7)
plt.title("Recovered Signals (Hamming Filter)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Vergleich zwischen Original und rekonstruierten Signalen
plt.figure(figsize=(12, 6))

# Vergleich für Audio 1
plt.subplot(2, 1, 1)
plt.plot(time, audio1, label="Original Audio 1", linestyle="--")
plt.plot(time, recovered_audio1, label="Reconstructed Audio 1", alpha=0.7)
plt.title("Comparison: Original vs Reconstructed Audio 1")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Vergleich für Audio 2
plt.subplot(2, 1, 2)
plt.plot(time, audio2, label="Original Audio 2", linestyle="--")
plt.plot(time, recovered_audio2, label="Reconstructed Audio 2", alpha=0.7)
plt.title("Comparison: Original vs Reconstructed Audio 2")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
