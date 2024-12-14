import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor
from dft import dft, idft
import time

def _precompute_exponentials(n, inverse=False):
    """
    Precompute the exponential factors.
    Parameters:
        n: int
            The size of the FFT (must be a power of 2).
        inverse: bool
            If True, computes the exponentials of the inverse FFT.
    Returns:
        exp_factors: numpy array
            Array of exponential factors for FFT (or IFFT) computation.
    """
    if inverse:
        sign = 1
    else:
        sign = -1 
    exp_factors = np.zeros(n//2, dtype=np.complex128)
    for k in range(n//2):
        exp_factors[k] = np.exp(2j * sign * pi * k / n)
    return exp_factors

def _recursive_algorithm(f, exp_factors, inverse=False):
    """
    Recursive algorithm for the FFT.
    Parameters:
        f: numpy array
            Input sequence.
        exp_factors: numpy array
            Precomputed exponential factors.
        inverse: bool
            If True, computes the inverse FFT.
    Returns:
        f: numpy array
            FFT (or IFFT) of the input sequence.
    """
    n = len(f)
    if n == 1:
        return f
    a = _recursive_algorithm(f[::2], exp_factors[::2], inverse)
    b = _recursive_algorithm(f[1::2], exp_factors[::2], inverse)
    f = np.zeros(n, dtype=np.complex128)
    for k in range(n // 2):
        temp = exp_factors[k] * b[k]
        f[k] = a[k] + temp
        f[k + n // 2] = a[k] - temp
    return f

def fft_recursive(f, exp_factors=None, inverse=False):
    """
    Fast Fourier Transform with precomputed exponentials.
    Parameters:
        f: numpy array
            Input sequence.
        exp_factors: numpy array, optional
            Precomputed exponential factors.
        inverse: bool
            If True, computes the inverse FFT.
    Returns:
        f: numpy array
            FFT (or IFFT) of the input sequence.
    """
    # convert input to complex128
    f = np.array(f, dtype=np.complex128)
    n = len(f)
    assert (n & (n - 1)) == 0, "Size of input must be a power of 2"
    if exp_factors is None:
        exp_factors = _precompute_exponentials(n, inverse)
    f = _recursive_algorithm(f, exp_factors, inverse)
    if not inverse:
        f /= n
    return f

def _bit_reverse_indices(n):
    """
    Compute the bit-reverse order for indices.
    Parameters:
        n: int
            The size of the FFT (must be a power of 2).
    Returns:
        indices: numpy array
            Indices reordered by the bit-reverse scheme.
    """
    bits = int(np.log2(n))
    indices = np.arange(n)
    reversed_indices = np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in indices])
    return reversed_indices

def fft_iterative(f, inverse=False):
    """
    Iterative Fast Fourier Transform.
    Parameters:
        f: numpy array
            Input sequence (size must be a power of 2).
        inverse : bool, optional
            If True, computes the inverse FFT.
    Returns:
        f: numpy array
            FFT (or IFFT) of the input sequence.
    """
    # convert input to complex128
    f = np.array(f, dtype=np.complex128)
    n = len(f)
    assert (n & (n - 1)) == 0, "Size of input must be a power of 2"
    exp_factors = _precompute_exponentials(n, inverse)

    # Reorder input array by bit-reverse indexing
    indices = _bit_reverse_indices(n)
    f = f[indices]
    # Iterative FFT computation
    for s in range(1, int(np.log2(n)) + 1):
        m = 2 ** s  # Size of subproblem
        half_m = m // 2
        w_m = exp_factors[::n // m]  # Step size for coefficients
        for k in range(0, n, m):
            for j in range(half_m):
                t = w_m[j] * f[k + j + half_m]
                u = f[k + j]
                f[k + j] = u + t
                f[k + j + half_m] = u - t
    if not inverse:
        f /= n
    return f

if __name__ == "__main__":
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)

    # Perform the FFT
    z_fft_r = fft_recursive(f)
    z_fft_np = np.fft.fft(f)/n
    # Compare the results
    assert np.allclose(z_fft_r, z_fft_np), "The results of the recursive are not equal."
    # Perform the iterative FFT
    z_fft_i = fft_iterative(f)
    # Compare the results
    assert np.allclose(z_fft_i, z_fft_np), "The results of the iterative are not equal."
    # recursive and iterative FFT must be excatly the same
    diff = z_fft_i - z_fft_r
    assert diff.all() == np.zeros(n, dtype=np.complex128).all(), "The results of the recursive and iterative are not equal."

    # Perform the IFFT
    f_ifft_r = fft_recursive(z, inverse=True)
    f_ifft_np = np.fft.ifft(z)*n
    # Compare the results
    assert np.allclose(f_ifft_r, f_ifft_np), "The results of the recursive are not equal."
    # Perform the iterative IFFT
    f_ifft_i = fft_iterative(z, inverse=True)
    # Compare the results
    assert np.allclose(f_ifft_i, f_ifft_np), "The results of the iterative are not equal."

    # Test if IFFT(FFT(f)) = f and FFT(IFFT(z)) = z
    f_reconstructed = fft_iterative(fft_iterative(f), inverse=True)
    assert np.allclose(f, f_reconstructed), "IFFT(FFT(f)) != f"
    z_reconstructed = fft_iterative(fft_iterative(z, inverse=True))
    assert np.allclose(z, z_reconstructed), "FFT(IFFT(z)) != z"

    # Compare the cumpuational time of FFT and DFT
    n = 2**10
    f = np.random.rand(n)
    start = time.time()
    z_fft = fft_iterative(f)
    end = time.time()
    print(f"FFT time: {end-start}")
    start = time.time()
    z_dft = fft_recursive(f)
    end = time.time()
    print(f"FFT time recursive: {end-start}")
    start = time.time()
    z_dft = dft(f)
    end = time.time()
    print(f"DFT time: {end-start}")
    assert np.allclose(z_fft, z_dft), "The results of the FFT and DFT are not equal."

    
