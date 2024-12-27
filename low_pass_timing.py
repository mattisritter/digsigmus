from low_pass_filter import low_pass_filter
import numpy as np
from function import Function
import matplotlib.pyplot as plt
import time

def test_call(function_len, filter_len, use_fast_convolution):
    fk = np.random.rand(function_len)
    f = Function(range(len(fk)), ws=100, f=fk)
    t0 = time.time()
    f_low_pass = low_pass_filter(f, 10, filter_len, hamming_window=True, use_fast_convolution=use_fast_convolution)
    t1 = time.time()
    return t1 - t0

def average_time(function_len, filter_len, use_fast_convolution, num_tests):
    times = [test_call(function_len, filter_len, use_fast_convolution) for _ in range(num_tests)]
    # Calculate the average time, standard deviation, min, max and median
    if type(times) == list:
        times = np.array(times)
    else:
        times = np.array([times])
    average_time = sum(times) / num_tests
    standard_deviation = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    return average_time, standard_deviation, min_time, max_time, median_time

def low_pass_impulse_response():
    N = 20
    wc = 5
    u = Function(range(0, 2*N+1), Ts=0.1, function_handle=lambda n: 1 if n == 0 else 0)
    f = low_pass_filter(u, wc, N, hamming_window=True)
    f_no_hamming = low_pass_filter(u, wc, N, hamming_window=False)
    plt.plot(f.f)
    plt.plot(f_no_hamming.f)
    plt.title("Low-pass filter impulse response")
    plt.legend(["Hamming window", "No windowing"])
    plt.xlabel("n")
    plt.grid()
    plt.show()
    
def low_pass_passband():
    A = []  # 2D list to store the amplitudes for all w and N combinations
    w_range = np.logspace(-1, 3, 200)  # Frequency range (logarithmic scale)
    n_range = [5, 15, 45, 135]  # Filter lengths

    for w in w_range:
        ws = 118.2 * w  # Sampling frequency based on w
        n = np.arange(0, 1000)  # Time points
        fcn_handle = lambda t: np.cos(w * t)  # Input function (cosine wave)
        f = Function(n, ws=ws, function_handle=fcn_handle)  # Create the Function object

        A_w = []  # Store amplitudes for current w

        for N in n_range:
            f_low_pass = low_pass_filter(f, 10, N)  # Apply low-pass filter
            amplitude = max(f_low_pass.f[2 * N : -2 * N])  # Calculate max amplitude
            A_w.append(amplitude)  # Store amplitude for current N

        f_low_pass_no_hamming = low_pass_filter(f, 10, N, hamming_window=False)
        amplitude_no_hamming = max(f_low_pass_no_hamming.f[2 * N : -2 * N])
        A_w.append(amplitude_no_hamming)
        A.append(A_w)

    # Convert A to a NumPy array for easier manipulation
    A = np.array(A)

    # Plot the amplitude of the filtered signal for each filter length
    for i, N in enumerate(n_range):
        plt.plot(w_range, A[:, i], label=f'N={N}')
    plt.plot(w_range, A[:, -1], label=f'N={N} (no windowing)')
    # Plot ideal filter
    A_ideal = [1, 1, 0, 0]
    w_ideal = [0.1, 10, 10, 1000]
    plt.plot(w_ideal, A_ideal, 'k--', label='Ideal Filter')

    plt.xscale('log')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude of the the Low-passed Signal')
    plt.legend()
    plt.grid()
    plt.show()

def low_pass_passband_compare_hamming():
    A = []  # 2D list to store the amplitudes for all w and N combinations
    w_range = np.logspace(-1, 3, 200)  # Frequency range (logarithmic scale)
    n_range = [15]  # Filter lengths

    for w in w_range:
        ws = 118.2 * w  # Sampling frequency based on w
        n = np.arange(0, 1000)  # Time points
        fcn_handle = lambda t: np.cos(w * t)  # Input function (cosine wave)
        f = Function(n, ws=ws, function_handle=fcn_handle)  # Create the Function object

        A_w = []  # Store amplitudes for current w

        for N in n_range:
            f_low_pass = low_pass_filter(f, 10, N)  # Apply low-pass filter
            amplitude = max(f_low_pass.f[2 * N : -2 * N])  # Calculate max amplitude
            A_w.append(amplitude)  # Store amplitude for current N
            # without hamming window
            f_low_pass_no_hamming = low_pass_filter(f, 10, N, hamming_window=False)
            amplitude_no_hamming = max(f_low_pass_no_hamming.f[2 * N : -2 * N])
            A_w.append(amplitude_no_hamming)

        A.append(A_w)  # Store amplitudes for current w

    # Convert A to a NumPy array for easier manipulation
    A = np.array(A)

    # Plot the amplitude of the filtered signal for each filter length
    for i, N in enumerate(n_range):
        plt.plot(w_range, A[:, 2*i], label=f'N={N}', color='tab:orange')
        plt.plot(w_range, A[:, 2*i+1], label=f'N={N} (no windowing)', color='darkred')
    # Plot ideal filter
    A_ideal = [1, 1, 0, 0]
    w_ideal = [0.1, 10, 10, 1000]
    plt.plot(w_ideal, A_ideal, 'k--', label='Ideal Filter')

    plt.xscale('log')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude of the Low-passed Signal')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # low_pass_impulse_response()
    # low_pass_passband()
    # low_pass_passband_compare_hamming()
    # Timing comparison
    function_len = int(2e4)
    filter_len = 50
    num_tests = 200
    print(f"Timing comparison for Low-Pass-Filter:\nFunction length: {function_len}\nFilter length {filter_len}\nNumber of tests: {num_tests}")

    use_fast_convolution = True
    avg, std, min, max, med = average_time(function_len, filter_len, use_fast_convolution, num_tests)
    print(f"Timing metrics using {'fast convolution' if use_fast_convolution else 'convolution in time-domain'}:")
    print(f"Average time: {avg:.4} s")
    print(f"Standard deviation: {std:.4} s")
    print(f"Min time: {min:.4} s")
    print(f"Max time: {max:.4} s")
    print(f"Median time: {med:.4} s")

    use_fast_convolution = False
    avg, std, min, max, med = average_time(function_len, filter_len, use_fast_convolution, num_tests)
    print(f"Timing metrics using {'fast convolution' if use_fast_convolution else 'convolution in time-domain'}:")
    print(f"Average time: {avg:.4} s")
    print(f"Standard deviation: {std:.4} s")
    print(f"Min time: {min:.4} s")
    print(f"Max time: {max:.4} s")
    print(f"Median time: {med:.4} s")