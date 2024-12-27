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
    # Use precomputed B* matrix for efficiency
    B_star = calculate_B_star(N)
    F = np.dot(B_star, f.f) / N
    freq_sampling = np.fft.fftfreq(N, d=f.Ts)
    return Function(freq_sampling, Ts=f.Ts, f=F)

# Helper Functions
def calculate_B(n: int) -> np.ndarray:
    """
    Calculate the B matrix for the IDFT.
    Parameters:
        n: int
            Size of the matrix.
    Returns:
        np.ndarray
            B matrix.
    """
    return np.array([[np.exp(2j * np.pi * k * l / n) for l in range(n)] for k in range(n)], dtype=np.complex128)

def calculate_B_star(n: int) -> np.ndarray:
    """
    Calculate the B* matrix for the DFT.
    Parameters:
        n: int
            Size of the matrix.
    Returns:
        np.ndarray
            B* matrix.
    """
    return calculate_B(n).conj().T

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