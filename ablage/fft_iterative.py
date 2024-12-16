import numpy as np

def fft_iterative(f_values):
    """
    Perform the Fast Fourier Transform (FFT) iteratively using the Cooley-Tukey algorithm.
    Parameters:
        f_values: list or np.array
            Input signal in the time domain.
    Return:
        np.array
            Transformed signal in the frequency domain.
    """
    N = len(f_values)
    if np.log2(N) % 1 > 0:
        raise ValueError("Input size must be a power of 2")

    f_values = np.array(f_values, dtype=complex)

    # Bit-reversal permutation
    indices = np.arange(N)
    indices = indices[np.argsort([int(bin(x)[2:].zfill(int(np.log2(N)))[::-1], 2) for x in indices])]
    f_values = f_values[indices]

    # Iterative FFT computation
    step = 2
    while step <= N:
        half_step = step // 2
        factor = np.exp(-2j * np.pi * np.arange(half_step) / step)
        for i in range(0, N, step):
            for j in range(half_step):
                temp = factor[j] * f_values[i + j + half_step]
                f_values[i + j + half_step] = f_values[i + j] - temp
                f_values[i + j] = f_values[i + j] + temp
        step *= 2

    return f_values

if __name__ == "__main__":
    # Example usage
    f_values = [0, 1, 2, 3, 4, 5, 6, 7]
    fft_result = fft_iterative(f_values)
    print("Input signal (time domain):", f_values)
    print("FFT result (frequency domain):", fft_result)
