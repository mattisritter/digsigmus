import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from dft import dft
from fft import fft_recursive, fft_iterative

# Helper function to pad signal to the next power of 2
def pad_to_next_power_of_2(signal):
    n = len(signal)
    next_power_of_2 = 1 << (n - 1).bit_length()
    padded_signal = np.pad(signal, (0, next_power_of_2 - n), mode='constant')
    return padded_signal

# Helper function to measure execution time and memory usage
def measure_performance(func, *args):
    tracemalloc.start()  # Start measuring memory
    start_time = time.perf_counter()  # Start timer
    result = func(*args)  # Execute the function
    end_time = time.perf_counter()  # Stop timer
    memory_used, _ = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop measuring memory
    execution_time = end_time - start_time
    return execution_time, memory_used, result

def visualize_differences():
    # Signal parameters
    frequency = 5  # Frequency of the sinusoidal signal (Hz)
    duration = 1   # Duration of the signal (seconds)
    sampling_rate = 64  # Sampling rate (Hz)

    # Generate a sinusoidal signal
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)

    # Pad the signal to the next power of 2 for FFT
    padded_signal = pad_to_next_power_of_2(signal)

    # Measure performance for each method
    dft_time, dft_memory, _ = measure_performance(dft, signal)
    fft_recursive_time, fft_recursive_memory, _ = measure_performance(fft_recursive, padded_signal)
    fft_iterative_time, fft_iterative_memory, _ = measure_performance(fft_iterative, padded_signal)

    # Plot computational time comparison
    plt.figure(figsize=(12, 6))
    times = [dft_time, fft_recursive_time, fft_iterative_time]
    labels = ['DFT', 'FFT Recursive', 'FFT Iterative']
    plt.bar(labels, times, color=['blue', 'green', 'magenta'])
    plt.ylabel('Execution Time (s)')
    plt.title('Computational Time Comparison')
    for i, time_val in enumerate(times):
        plt.text(i, time_val, f"{time_val:.4f}", ha='center', va='bottom')
    plt.grid(axis='y')
    plt.show()

    # Plot memory usage comparison
    plt.figure(figsize=(12, 6))
    memory = [dft_memory, fft_recursive_memory, fft_iterative_memory]
    plt.bar(labels, memory, color=['blue', 'green', 'magenta'])
    plt.ylabel('Memory Usage (Bytes)')
    plt.title('Memory Usage Comparison')
    for i, memory_val in enumerate(memory):
        plt.text(i, memory_val, f"{memory_val/1024:.2f} KB", ha='center', va='bottom')
    plt.grid(axis='y')
    plt.show()

    # Measure memory usage and computational time for varying signal lengths
    durations = np.arange(0.1, 10.1, 0.2)  # Signal durations from 0.1 to 10.0 seconds
    dft_memory_usage = []
    fft_recursive_memory_usage = []
    fft_iterative_memory_usage = []

    dft_time_usage = []
    fft_recursive_time_usage = []
    fft_iterative_time_usage = []

    for duration in durations:
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)

        padded_signal = pad_to_next_power_of_2(signal)

        dft_time, dft_memory, _ = measure_performance(dft, signal)
        fft_recursive_time, fft_recursive_memory, _ = measure_performance(fft_recursive, padded_signal)
        fft_iterative_time, fft_iterative_memory, _ = measure_performance(fft_iterative, padded_signal)

        dft_memory_usage.append(dft_memory)
        fft_recursive_memory_usage.append(fft_recursive_memory)
        fft_iterative_memory_usage.append(fft_iterative_memory)

        dft_time_usage.append(dft_time)
        fft_recursive_time_usage.append(fft_recursive_time)
        fft_iterative_time_usage.append(fft_iterative_time)

    # Plot memory usage over signal lengths
    plt.figure(figsize=(12, 6))
    plt.plot(durations[:len(dft_memory_usage)], np.array(dft_memory_usage) / 1024, label='DFT', marker='o', linestyle='-', color='blue')
    plt.plot(durations[:len(fft_recursive_memory_usage)], np.array(fft_recursive_memory_usage) / 1024, label='FFT Recursive', marker='s', linestyle='--', color='green')
    plt.plot(durations[:len(fft_iterative_memory_usage)], np.array(fft_iterative_memory_usage) / 1024, label='FFT Iterative', marker='^', linestyle='-.', color='magenta')
    plt.xlabel('Signal Duration (s)')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage vs Signal Length')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot computational time over signal lengths
    plt.figure(figsize=(12, 6))
    plt.plot(durations[:len(dft_time_usage)], dft_time_usage, label='DFT', marker='o', linestyle='-', color='blue')
    plt.plot(durations[:len(fft_recursive_time_usage)], fft_recursive_time_usage, label='FFT Recursive', marker='s', linestyle='--', color='green')
    plt.plot(durations[:len(fft_iterative_time_usage)], fft_iterative_time_usage, label='FFT Iterative', marker='^', linestyle='-.', color='magenta')
    plt.xlabel('Signal Duration (s)')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Signal Length')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    visualize_differences()
