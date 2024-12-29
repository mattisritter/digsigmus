import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

# Pfad zu 'ablage' hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from function import Function
from dft import dft_optimized, dft
from fft import fft_recursive, fft_iterative


def measure_memory_and_time(func, *args):
    """
    Measure execution time and peak memory usage of a function.
    """
    tracemalloc.start()
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end_time - start_time, peak_memory


def generate_signal(frequency, duration, sampling_frequency, signal_type='sin'):
    """
    Generate a sine or cosine signal.
    """
    Ts = 1 / sampling_frequency
    t = np.arange(0, duration, Ts)
    values = np.sin(2 * np.pi * frequency * t) if signal_type == 'sin' else np.cos(2 * np.pi * frequency * t)
    return values


def pad_to_power_of_two(signal):
    """
    Pad the function values to the next power of 2 length.
    """
    next_power_of_2 = 2 ** int(np.ceil(np.log2(len(signal))))
    padded_values = np.pad(signal, (0, next_power_of_2 - len(signal)), mode='constant')
    return padded_values


def compare_methods():
    """
    Compare DFT, recursive FFT, and iterative FFT for time and memory usage and visualize results.
    """
    length = 2**11

    signal = np.random.rand(length)

    # Measure DFT
    _, dft_time, dft_memory = measure_memory_and_time(dft_optimized, signal)

    # Measure Recursive FFT
    signal_padded_recursive = pad_to_power_of_two(signal)
    _, fft_recursive_time, fft_recursive_memory = measure_memory_and_time(fft_recursive, signal_padded_recursive)

    # Measure Iterative FFT
    signal_padded_iterative = pad_to_power_of_two(signal)
    _, fft_iterative_time, fft_iterative_memory = measure_memory_and_time(fft_iterative, signal_padded_iterative)

    # Collect results
    methods = ['DFT', 'Recursive FFT', 'Iterative FFT']
    times = [dft_time, fft_recursive_time, fft_iterative_time]
    memory = [dft_memory / 1024, fft_recursive_memory / 1024, fft_iterative_memory / 1024]  # Convert to KiB

    # Plot time and memory usage
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Computing Time Plot
    bars_time = ax[0].bar(methods, times, color=['blue', 'orange', 'green'], label="Time [s]")
    ax[0].set_title("Computing Time")
    ax[0].set_ylabel("Time [s]")
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # Add values above bars (rounded to 3 decimals)
    for bar in bars_time:
        height = bar.get_height()
        ax[0].annotate(f'{height:.3f}', 
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2),  # Offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    # Allocated Storage Plot
    bars_memory = ax[1].bar(methods, memory, color=['blue', 'orange', 'green'], label="Storage [KiB]")
    ax[1].set_title("Allocated Storage")
    ax[1].set_ylabel("Storage [KiB]")
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # Add values above bars (rounded to 3 decimals)
    for bar in bars_memory:
        height = bar.get_height()
        ax[1].annotate(f'{height:.3f}', 
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2),  # Offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



def find_transition_point():
    """
    Find and visualize the transition point where recursive FFT becomes faster than DFT.
    """
    duration = 1  # Start duration in seconds
    sampling_frequency = 256  # Constant sampling frequency

    durations = []
    dft_times = []
    fft_recursive_times = []

    print("\n--- Transition Point: DFT vs Recursive FFT ---\n")

    while True:
        signal = generate_signal(frequency=5, duration=duration, sampling_frequency=sampling_frequency)
        signal_padded = pad_to_power_of_two(signal)

        _, dft_time, _ = measure_memory_and_time(dft_optimized, signal)
        _, fft_recursive_time, _ = measure_memory_and_time(fft_recursive, signal_padded)

        durations.append(duration)
        dft_times.append(dft_time)
        fft_recursive_times.append(fft_recursive_time)

        print(f"Dauer: {duration} s | DFT: {dft_time:.6f} s | Rekursive FFT: {fft_recursive_time:.6f} s")

        if fft_recursive_time < dft_time:
            print(f"\nÜbergangspunkt gefunden bei Dauer = {duration} Sekunden\n")
            break

        duration += 0.1

    # Visualize transition point
    plt.figure(figsize=(10, 6))
    plt.plot(durations, dft_times, label="DFT", linestyle="--", marker="o")
    plt.plot(durations, fft_recursive_times, label="Recursive FFT", linestyle="-", marker="x")
    plt.axvline(x=duration, color='red', linestyle=":", label="Transition Point")
    plt.title("Computing Time Comparison: DFT vs Recursive FFT")
    plt.xlabel("Signaltime [s]")
    plt.ylabel("Computing Time [s]")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    compare_methods()
    # find_transition_point()
