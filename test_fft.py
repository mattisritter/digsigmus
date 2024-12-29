import numpy as np
from fft import fft_recursive, fft_iterative


def test_fft():
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

def test_ifft():
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)
    # Perform the IFFT
    f_ifft_r = fft_recursive(z, inverse=True)
    f_ifft_np = np.fft.ifft(z)*n
    # Compare the results
    assert np.allclose(f_ifft_r, f_ifft_np), "The results of the recursive are not equal."
    # Perform the iterative IFFT
    f_ifft_i = fft_iterative(z, inverse=True)
    # Compare the results
    assert np.allclose(f_ifft_i, f_ifft_np), "The results of the iterative are not equal."
    # recursive and iterative IFFT must be excatly the same
    diff = f_ifft_i - f_ifft_r
    assert diff.all() == np.zeros(n, dtype=np.complex128).all(), "The results of the recursive and iterative are not equal."

def test_inverse_operation():
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)
    # Test if IFFT(FFT(f)) = f and FFT(IFFT(z)) = z
    f_reconstructed = fft_iterative(fft_iterative(f), inverse=True)
    assert np.allclose(f, f_reconstructed), "IFFT(FFT(f)) != f"
    z_reconstructed = fft_iterative(fft_iterative(z, inverse=True))
    assert np.allclose(z, z_reconstructed), "FFT(IFFT(z)) != z"