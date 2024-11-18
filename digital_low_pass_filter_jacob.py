import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from function import Function  # Wir gehen davon aus, dass Function als function.py gespeichert ist

def test_harmonic_oscillation():
    fs = 1000  # Sampling frequency
    f_signal = 50  # Signal frequency
    omega_c_low = 30  # Cutoff frequency below signal frequency
    omega_c_high = 70  # Cutoff frequency above signal frequency
    N = 20  # Filter delay

    # Create harmonic oscillation
    n = np.arange(0, fs)
    Ts = 1.0 / fs
    signal = np.sin(2 * np.pi * f_signal * n * Ts)
    
    # Create Function objects
    signal_func = Function(n, Ts=Ts, f=signal)

    # Apply low-pass filter with low cutoff frequency
    filtered_low = signal_func.low_pass(omega_c_low, N)

    # Apply low-pass filter with high cutoff frequency
    filtered_high = signal_func.low_pass(omega_c_high, N)

    # Plot, trimming the filtered signals to match the original length
    t = n * Ts
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Original signal")
    plt.plot(t, filtered_low.f[:len(t)], label="Filtered signal (omega_c < f_signal)", linestyle='--')
    plt.plot(t, filtered_high.f[:len(t)], label="Filtered signal (omega_c > f_signal)", linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 1: Low-pass filtering of a harmonic oscillation")
    plt.show()

def test_two_frequencies_with_sawtooth():
    fs = 1000  # Sampling frequency
    f1, f2 = 30, 80  # Two different frequencies
    f_saw = 10  # Frequency of the sawtooth signal
    omega_c = 50  # Cutoff frequency between f1 and f2
    N = 50  # Filter delay

    # Create signal with two frequencies and a sawtooth signal
    n = np.arange(0, fs)
    Ts = 1.0 / fs
    sin_signal = np.sin(2 * np.pi * f1 * n * Ts) + np.sin(2 * np.pi * f2 * n * Ts)
    sawtooth_signal = 2 * (n * f_saw * Ts % 1) - 1  # Normalized sawtooth wave from -1 to 1
    combined_signal = sin_signal + sawtooth_signal

    # Create Function object
    signal_func = Function(n, Ts=Ts, f=combined_signal)

    # Apply low-pass filter
    filtered_signal = signal_func.low_pass(omega_c, N)

    # Plot, trimming the filtered signal to match the original length
    t = n * Ts
    plt.figure(figsize=(10, 6))
    plt.plot(t, combined_signal, label="Original signal (f1 + f2 + sawtooth)")
    plt.plot(t, filtered_signal.f[:len(t)], label="Filtered signal (f1 remains, f2 and sawtooth partially filtered)", linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 2: Low-pass filtering of a signal with two sine frequencies and a sawtooth wave")
    plt.show()

def test_finite_length_signal():
    fs = 1000  # Sampling frequency
    f_signal = 50  # Signal frequency
    omega_c = 40  # Cutoff frequency below the signal frequency
    N = 50  # Filter delay

    # Create a finite-length signal
    t = np.arange(0, 0.5, 1.0 / fs)  # Finite time interval
    n = np.arange(len(t))
    Ts = 1.0 / fs
    signal = np.sin(2 * np.pi * f_signal * t)

    # Create Function object
    signal_func = Function(n, Ts=Ts, f=signal)

    # Apply low-pass filter
    filtered_signal = signal_func.low_pass(omega_c, N)

    # Plot, trimming the filtered signal to match the original length
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Original signal (finite length)")
    plt.plot(t, filtered_signal.f[:len(t)], label="Filtered signal with transients", linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 3: Low-pass filtering of a finite signal (transients visible)")
    plt.show()

def test_music_filtering():
    file_path = "Jodler.wav"  # Path to the music file
    signal, fs = sf.read(file_path)

    # Convert stereo to mono if necessary
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Filter parameters
    omega_c = 2000  # Cutoff frequency of the low-pass filter in Hz
    N = 100  # Filter delay

    # Create Function object
    n = np.arange(len(signal))
    Ts = 1.0 / fs
    signal_func = Function(n, Ts=Ts, f=signal)

    # Apply low-pass filter
    filtered_signal = signal_func.low_pass(omega_c, N)

    # Save original and filtered music files
    sf.write("original_music.wav", signal, fs)
    sf.write("filtered_music.wav", filtered_signal.f[:len(signal)], fs)  # Trimmed to match the original length
    print("Original and filtered music signals have been saved as 'original_music.wav' and 'filtered_music.wav'.")

    # Plot waveform, trimming to view the first 10,000 samples
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:10000], label="Original signal", alpha=0.7)
    plt.plot(filtered_signal.f[:10000], label="Filtered signal", linestyle='--', alpha=0.7)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 4: Low-pass filtering of a music signal")
    plt.show()

# Run all tests
def run_all_tests():
    print("Running Test 1: Harmonic oscillation with cutoff frequencies above and below the signal frequency")
    test_harmonic_oscillation()
    
    print("Running Test 2: Two harmonic oscillations with different frequencies and an added sawtooth signal")
    test_two_frequencies_with_sawtooth()
    
    print("Running Test 3: Finite signal and transient observation")
    test_finite_length_signal()
    
    # print("Running Test 4: Music signal filtering and saving")
    # test_music_filtering()

run_all_tests()
