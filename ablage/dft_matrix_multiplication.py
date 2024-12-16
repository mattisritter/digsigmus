import numpy as np
from function import Function

def dft_matrix_multiplication(f: Function) -> Function:
    """
    Perform the Discrete Fourier Transform (DFT) using matrix multiplication.
    Parameters:
        f: Function
            Input function in the time domain.
    Return:
        Function
            Transformed function in the frequency domain.
    """
    N = f.len  # Number of samples
    # Create the DFT matrix
    dft_matrix = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    # Perform the matrix multiplication
    F = np.dot(dft_matrix, f.f)
    # Return the frequency domain representation as a new Function object
    freq_sampling = np.fft.fftfreq(N, d=f.Ts)  # Frequency axis
    return Function(freq_sampling, Ts=f.Ts, f=F)

if __name__ == "__main__":
    # Example usage
    n = range(0, 8)
    Ts = 1
    f_values = [0, 1, 2, 3, 4, 5, 6, 7]
    time_domain_function = Function(n, Ts=Ts, f=f_values)

    # Perform DFT
    frequency_domain_function = dft_matrix_multiplication(time_domain_function)

    # Print the result
    print("Input function (time domain):", time_domain_function.f)
    print("DFT result (frequency domain):", frequency_domain_function.f)