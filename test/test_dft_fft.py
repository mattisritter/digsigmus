# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from dft import dft
from fft import fft_recursive, fft_iterative

# Helper function to compare results
def compare_results(result1, result2, tolerance=1e-9):
    """
    Compare two results element-wise within a given tolerance.
    """
    return np.allclose(result1, result2, atol=tolerance)

def test_dft_fft_similarity():
    # Generate a test signal
    frequency = 5  # Frequency of the sinusoidal signal (Hz)
    duration = 1   # Duration of the signal (seconds)
    sampling_rate = 64  # Sampling rate (Hz)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)

    # Compute DFT
    dft_result = dft(signal)

    # Compute FFT Recursive
    fft_recursive_result = fft_recursive(signal)

    # Compute FFT Iterative
    fft_iterative_result = fft_iterative(signal)

    # Compare DFT and FFT Recursive
    assert compare_results(dft_result, fft_recursive_result), \
        "DFT and FFT Recursive results do not match within the tolerance!"

    # Compare DFT and FFT Iterative
    assert compare_results(dft_result, fft_iterative_result), \
        "DFT and FFT Iterative results do not match within the tolerance!"

    # Compare FFT Recursive and FFT Iterative
    assert compare_results(fft_recursive_result, fft_iterative_result), \
        "FFT Recursive and FFT Iterative results do not match within the tolerance!"

    print("All tests passed! DFT, FFT Recursive, and FFT Iterative produce similar results within the tolerance.")

if __name__ == "__main__":
    test_dft_fft_similarity()
