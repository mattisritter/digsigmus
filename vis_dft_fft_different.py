import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from dft import dft_optimized
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
    # Signal length
    signal_length = 2**11
    # Signal
    signal = np.random.rand(signal_length)

    # Measure performance for each method
    dft_time, dft_memory, _ = measure_performance(dft_optimized, signal)
    fft_recursive_time, fft_recursive_memory, _ = measure_performance(fft_recursive, signal)
    fft_iterative_time, fft_iterative_memory, _ = measure_performance(fft_iterative, signal)

    # Plot computational time comparison
    plt.figure(figsize=(5, 4))
    times = [dft_time, fft_recursive_time, fft_iterative_time]
    labels = ['DFT', 'FFT Recursive', 'FFT Iterative']
    plt.bar(labels, times, color=['blue', 'green', 'magenta'])
    plt.ylabel('Execution Time (s)')
    plt.title('Computational Time Comparison')
    for i, time_val in enumerate(times):
        plt.text(i, time_val, f"{time_val:.4f}", ha='center', va='bottom')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot memory usage comparison
    plt.figure(figsize=(5, 4))
    memory = [dft_memory, fft_recursive_memory, fft_iterative_memory]
    plt.bar(labels, memory, color=['blue', 'green', 'magenta'])
    plt.ylabel('Memory Usage (Bytes)')
    plt.title('Memory Usage Comparison')
    for i, memory_val in enumerate(memory):
        plt.text(i, memory_val, f"{memory_val/1024:.2f} KB", ha='center', va='bottom')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Measure memory usage and computational time for varying signal lengths
    exp = range(1, 13)
    lengths = [2**i for i in exp]
    dft_memory_usage = []
    fft_recursive_memory_usage = []
    fft_iterative_memory_usage = []

    dft_time_usage = []
    fft_recursive_time_usage = []
    fft_iterative_time_usage = []

    for length in lengths:
        signal = np.random.rand(length)

        dft_time, dft_memory, _ = measure_performance(dft_optimized, signal)
        fft_recursive_time, fft_recursive_memory, _ = measure_performance(fft_recursive, signal)
        fft_iterative_time, fft_iterative_memory, _ = measure_performance(fft_iterative, signal)

        dft_memory_usage.append(dft_memory)
        fft_recursive_memory_usage.append(fft_recursive_memory)
        fft_iterative_memory_usage.append(fft_iterative_memory)

        dft_time_usage.append(dft_time)
        fft_recursive_time_usage.append(fft_recursive_time)
        fft_iterative_time_usage.append(fft_iterative_time)

        # Break if iterative FFT computing time exceeds recursive FFT 
        if fft_iterative_time < fft_recursive_time:
            break

    # Plot memory usage over signal lengths
    plt.figure(figsize=(5, 4))
    plt.plot(lengths[:len(dft_memory_usage)], np.array(dft_memory_usage) / 1024, label='DFT', marker='o', linestyle='-', color='blue')
    plt.plot(lengths[:len(fft_recursive_memory_usage)], np.array(fft_recursive_memory_usage) / 1024, label='FFT Recursive', marker='s', linestyle='--', color='green')
    plt.plot(lengths[:len(fft_iterative_memory_usage)], np.array(fft_iterative_memory_usage) / 1024, label='FFT Iterative', marker='^', linestyle='-.', color='magenta')
    plt.xlabel('Signal Lenght')
    plt.xscale('log')
    plt.ylabel('Memory Usage (KB)')
    plt.title('Memory Usage vs Signal Length')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot computational time over signal lengths
    plt.figure(figsize=(5, 4))
    plt.plot(lengths[:len(dft_time_usage)], dft_time_usage, label='DFT', marker='o', linestyle='-', color='blue')
    plt.plot(lengths[:len(fft_recursive_time_usage)], fft_recursive_time_usage, label='FFT Recursive', marker='s', linestyle='--', color='green')
    plt.plot(lengths[:len(fft_iterative_time_usage)], fft_iterative_time_usage, label='FFT Iterative', marker='^', linestyle='-.', color='magenta')
    plt.xlabel('Signal Length')
    plt.xscale('log')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Signal Length')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_differences()
