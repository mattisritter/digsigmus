import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

# Pfad zu 'ablage' hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from function import Function
from dft import dft_optimized, idft_optimized
from fft import fft_recursive, fft_iterative

def measure_memory_and_time(func, *args):
    """
    Measure execution time and peak memory usage of a function.
    """
    tracemalloc.start()  # Start memory tracking
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()  # Stop memory tracking
    return result, end_time - start_time, peak

def generate_signal(frequency, duration, sampling_frequency, signal_type='sin'):
    """
    Generate a sine or cosine signal.
    """
    Ts = 1 / sampling_frequency
    t = np.arange(0, duration, Ts)
    if signal_type == 'sin':
        values = np.sin(2 * np.pi * frequency * t)
    elif signal_type == 'cos':
        values = np.cos(2 * np.pi * frequency * t)
    else:
        raise ValueError("Unsupported signal type. Use 'sin' or 'cos'.")
    return values

def compare_dft_fft():
    """
    Compare DFT, recursive FFT, and iterative FFT in terms of time and memory usage.
    """
    # Generate a test signal
    frequency = 5  # Hz
    duration = 1   # seconds
    sampling_frequency = 2048  # Hz (higher value to better see differences)
    signal = generate_signal(frequency, duration, sampling_frequency, signal_type='sin')

    print("\n--- Starting DFT, Recursive FFT, and Iterative FFT Comparison ---\n")

    # Measure DFT
    dft_result, dft_time, dft_memory = measure_memory_and_time(dft_optimized, signal)

    # Measure Recursive FFT
    fft_recursive_result, fft_recursive_time, fft_recursive_memory = measure_memory_and_time(fft_recursive, signal)

    # Measure Iterative FFT
    fft_iterative_result, fft_iterative_time, fft_iterative_memory = measure_memory_and_time(fft_iterative, signal)

    # Print Results
    print(f"DFT: Zeit = {dft_time:.6f} s, Speicher = {dft_memory / 1024:.2f} KiB")
    print(f"Rekursive FFT: Zeit = {fft_recursive_time:.6f} s, Speicher = {fft_recursive_memory / 1024:.2f} KiB")
    print(f"Iterative FFT: Zeit = {fft_iterative_time:.6f} s, Speicher = {fft_iterative_memory / 1024:.2f} KiB")

    # Plot Results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.title("Vergleich der Transformierten")
    plt.plot(np.abs(dft_result), label='DFT', linestyle='--')
    plt.plot(np.abs(fft_recursive_result), label='Rekursive FFT', linestyle='-.')
    plt.plot(np.abs(fft_iterative_result), label='Iterative FFT', linestyle='-')
    plt.xlabel("Frequenzindex")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_dft_fft()
