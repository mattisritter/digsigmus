import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Pfad zu 'ablage' hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '../ablage'))

from function import Function
from dft_matrix_multiplication import dft_matrix_multiplication
from fft_recursive import fft_recursive
from fft_iterative import fft_iterative

def compare_dft_fft_results():
    """
    Compare the frequency domain results of DFT, Recursive FFT, and Iterative FFT.
    Display numerical differences and visualizations.
    """
    print("\n--- Comparing DFT and FFT Results ---\n")

    # Generate a test signal
    N = 16  # Length of the signal
    t = np.linspace(0, 1, N, endpoint=False)
    signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 5 * t)

    # Perform DFT
    dft_result = dft_matrix_multiplication(Function(range(N), Ts=1 / N, f=signal))

    # Perform Recursive FFT
    fft_recursive_result = fft_recursive(Function(range(N), Ts=1 / N, f=signal))

    # Perform Iterative FFT
    fft_iterative_result = fft_iterative(signal)

    # Calculate absolute differences
    diff_recursive = np.abs(dft_result.f - fft_recursive_result.f)
    diff_iterative = np.abs(dft_result.f - fft_iterative_result)
    diff_fft = np.abs(fft_recursive_result.f - fft_iterative_result)

    # Print numerical differences
    print("Numerical Differences (DFT vs Recursive FFT):", diff_recursive)
    print("Numerical Differences (DFT vs Iterative FFT):", diff_iterative)
    print("Numerical Differences (Recursive FFT vs Iterative FFT):", diff_fft)

    # Plot numerical differences
    indices = np.arange(len(diff_recursive))

    plt.figure(figsize=(12, 8))

    plt.bar(indices - 0.3, diff_recursive, width=0.3, label="DFT vs Recursive FFT", color='blue')
    plt.bar(indices, diff_iterative, width=0.3, label="DFT vs Iterative FFT", color='orange')
    plt.bar(indices + 0.3, diff_fft, width=0.3, label="Recursive FFT vs Iterative FFT", color='green')

    plt.title("Numerical Differences between DFT and FFT Methods")
    plt.xlabel("Frequency Index")
    plt.ylabel("Difference (Magnitude)")
    plt.xticks(indices)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot results
    freqs = np.fft.fftfreq(N, d=1 / N)

    plt.figure(figsize=(12, 10))

    # Plot DFT results
    plt.subplot(3, 1, 1)
    plt.stem(freqs[:N // 2], np.abs(dft_result.f[:N // 2]), linefmt='b-', markerfmt='bo', basefmt='r-', label="DFT", use_line_collection=True)
    plt.title("DFT Results")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()

    # Plot Recursive FFT results
    plt.subplot(3, 1, 2)
    plt.stem(freqs[:N // 2], np.abs(fft_recursive_result.f[:N // 2]), linefmt='g-', markerfmt='go', basefmt='r-', label="Recursive FFT", use_line_collection=True)
    plt.title("Recursive FFT Results")
    plt.xlabel("Frequency (Hz")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()

    # Plot Iterative FFT results
    plt.subplot(3, 1, 3)
    plt.stem(freqs[:N // 2], np.abs(fft_iterative_result[:N // 2]), linefmt='m-', markerfmt='mo', basefmt='r-', label="Iterative FFT", use_line_collection=True)
    plt.title("Iterative FFT Results")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_dft_fft_results()