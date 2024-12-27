import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

# Pfad zu 'ablage' hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '../ablage'))

from function import Function
from dft_matrix_multiplication import dft_matrix_multiplication
from fft_recursive import fft_recursive
from fft_iterative import fft_iterative

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
    return Function(range(len(t)), Ts=Ts, f=values)

def pad_to_power_of_two(signal: Function):
    """
    Pad the function values to the next power of 2 length.
    """
    next_power_of_2 = 2 ** int(np.ceil(np.log2(signal.len)))
    padded_values = np.pad(signal.f, (0, next_power_of_2 - signal.len), mode='constant')
    return Function(range(next_power_of_2), Ts=signal.Ts, f=padded_values)

def compare_dft_fft():
    """
    Compare DFT, recursive FFT, and iterative FFT in terms of time and memory usage.
    """
    # Generate a larger test signal
    frequency = 5  # Hz
    duration = 1   # seconds
    sampling_frequency_dft = 2048  # DFT: machbar
    sampling_frequency_fft = 8192  # FFT: große Signale

    # Signal für DFT
    signal_dft = generate_signal(frequency, duration, sampling_frequency_dft, signal_type='sin')
    # Signal für FFT
    signal_fft = generate_signal(frequency, duration, sampling_frequency_fft, signal_type='sin')

    print("\n--- Starting DFT, Recursive FFT, and Iterative FFT Comparison ---\n")

    # Measure DFT
    print("Running DFT...")
    dft_result, dft_time, dft_memory = measure_memory_and_time(dft_matrix_multiplication, signal_dft)

    # Padding für rekursive FFT
    signal_padded_recursive = pad_to_power_of_two(signal_fft)

    # Measure Recursive FFT
    print("Running Recursive FFT...")
    fft_recursive_result, fft_recursive_time, fft_recursive_memory = measure_memory_and_time(fft_recursive, signal_padded_recursive)

    # Padding für iterative FFT
    signal_padded_iterative = pad_to_power_of_two(signal_fft)

    # Measure Iterative FFT
    print("Running Iterative FFT...")
    fft_iterative_result, fft_iterative_time, fft_iterative_memory = measure_memory_and_time(fft_iterative, signal_padded_iterative.f)

    # Print Results
    print(f"DFT (N={len(signal_dft.f)}): Zeit = {dft_time:.6f} s, Speicher = {dft_memory / 1024:.2f} KiB")
    print(f"Rekursive FFT (N={len(signal_padded_recursive.f)}): Zeit = {fft_recursive_time:.6f} s, Speicher = {fft_recursive_memory / 1024:.2f} KiB")
    print(f"Iterative FFT (N={len(signal_padded_iterative.f)}): Zeit = {fft_iterative_time:.6f} s, Speicher = {fft_iterative_memory / 1024:.2f} KiB")

    # Plot Results
    plt.figure(figsize=(12, 6))
    plt.title("Vergleich der Transformierten")
    plt.plot(np.abs(dft_result.f), label='DFT', linestyle='--')
    plt.plot(np.abs(fft_recursive_result.f), label='Rekursive FFT', linestyle='-.')
    plt.plot(np.abs(fft_iterative_result), label='Iterative FFT', linestyle='-')
    plt.xlabel("Frequenzindex")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

def find_transition_point():
    """
    Find the transition point where recursive FFT becomes faster than DFT.
    Visualize the results for better understanding.
    """
    duration = 1  # Start duration in seconds
    sampling_frequency = 256  # Konstante Abtastrate

    durations = []
    dft_times = []
    fft_recursive_times = []

    print("\n--- DFT vs. Recursive FFT Transition Point ---\n")

    while True:
        print(f"Testing duration = {duration} seconds, sampling_frequency = {sampling_frequency} Hz")
        signal = generate_signal(frequency=5, duration=duration, sampling_frequency=sampling_frequency)

        # Pad for recursive FFT
        signal_padded = pad_to_power_of_two(signal)

        # Measure DFT
        _, dft_time, _ = measure_memory_and_time(dft_matrix_multiplication, signal)
        
        # Measure Recursive FFT
        _, fft_recursive_time, _ = measure_memory_and_time(fft_recursive, signal_padded)

        # Store results for visualization
        durations.append(duration)
        dft_times.append(dft_time)
        fft_recursive_times.append(fft_recursive_time)

        print(f"DFT Time: {dft_time:.6f} s | Recursive FFT Time: {fft_recursive_time:.6f} s\n")

        if fft_recursive_time < dft_time:
            print(f"Transition point found at duration = {duration} seconds\n")
            break

        duration += 1  # Increase the duration by 1 second

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(durations, dft_times, label="DFT Time", linestyle="--", marker="o")
    plt.plot(durations, fft_recursive_times, label="Recursive FFT Time", linestyle="-", marker="x")
    plt.axvline(x=duration, color='r', linestyle=":", label="Transition Point")
    plt.title("DFT vs Recursive FFT: Computing Time")
    plt.xlabel("Signaltime [s]")
    plt.ylabel("Computing Time [s]")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    compare_dft_fft()
    find_transition_point()

