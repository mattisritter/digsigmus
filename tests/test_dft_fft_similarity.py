import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Pfad zu 'ablage' hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from function import Function
from dft import dft_optimized, idft_optimized
from fft import fft_recursive, fft_iterative

def test_identical_transformation():
    """
    Test: DFT, recursive FFT, and iterative FFT should reconstruct the original signal.
    """
    print("\n--- Test: Identical Transformation ---\n")

    # Generate a test signal
    N = 32
    t = np.linspace(0, 1, N, endpoint=False)
    signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)

    # Perform DFT
    dft_result = dft_optimized(signal)
    idft_result = idft_optimized(dft_result)

    # Perform recursive FFT and inverse FFT
    fft_recursive_result = fft_recursive(signal)
    ifft_recursive_result = fft_recursive(fft_recursive_result, inverse=True)

    # Perform iterative FFT and inverse FFT
    fft_iterative_result = fft_iterative(signal)
    ifft_iterative_result = fft_iterative(fft_iterative_result, inverse=True)

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, signal, label="Original Signal", linestyle="--")
    plt.plot(t, idft_result.real, label="Reconstructed (DFT)")
    plt.title("Signal Reconstruction using DFT")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, signal, label="Original Signal", linestyle="--")
    plt.plot(t, ifft_recursive_result.real, label="Reconstructed (Recursive FFT)")
    plt.title("Signal Reconstruction using Recursive FFT")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, signal, label="Original Signal", linestyle="--")
    plt.plot(t, ifft_iterative_result.real, label="Reconstructed (Iterative FFT)")
    plt.title("Signal Reconstruction using Iterative FFT")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def test_frequency_resolution():
    """
    Test: Show frequency resolution of DFT and FFT methods.
    """
    print("\n--- Test: Frequency Resolution ---\n")

    # Generate a signal with multiple frequencies
    N = 32
    t = np.linspace(0, 1, N, endpoint=False)
    signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)

    # Perform DFT
    dft_result = dft_optimized(signal)

    # Perform recursive FFT
    fft_recursive_result = fft_recursive(signal)

    # Perform iterative FFT
    fft_iterative_result = fft_iterative(signal)

    # Plot results
    freqs = np.fft.fftfreq(N, d=1 / N)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(freqs[:N // 2], np.abs(dft_result[:N // 2]), label="DFT", linestyle="--", marker="o")
    plt.title("Frequency Resolution using DFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(freqs[:N // 2], np.abs(fft_recursive_result[:N // 2]), label="Recursive FFT", linestyle="-.", marker="x")
    plt.title("Frequency Resolution using Recursive FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(freqs[:N // 2], np.abs(fft_iterative_result[:N // 2]), label="Iterative FFT", linestyle="-", marker="d")
    plt.title("Frequency Resolution using Iterative FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_identical_transformation()
    test_frequency_resolution()
