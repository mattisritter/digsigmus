import numpy as np
from dft import dft, idft, dft_optimized, idft_optimized
from math import pi

def test_dft():
    # Test implementation of DFT and IDFT with random data
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)
    f_reconstructed = idft(dft(f))
    assert np.allclose(f, f_reconstructed), "IDFT(DFT(f)) != f"
    z_reconstructed = dft(idft(z))
    assert np.allclose(z, z_reconstructed), "DFT(IDFT(z)) != z"

def test_dft_optimized():
    # Test optimized implementation of DFT and IDFT with random data
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)
    f_reconstructed = idft_optimized(dft_optimized(f))
    assert np.allclose(f, f_reconstructed), "IDFT(DFT(f)) != f"
    z_reconstructed = dft_optimized(idft_optimized(z))
    assert np.allclose(z, z_reconstructed), "DFT(IDFT(z)) != z"

def test_dft_comparison():
    # Test if the optimized implementation gives the same result as the original implementation
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)
    z_original = dft(f)
    z_optimized = dft_optimized(f)
    assert np.allclose(z_original, z_optimized), "DFT != DFT optimized"
    f_original = idft(z)
    f_optimized = idft_optimized(z)
    assert np.allclose(f_original, f_optimized), "IDFT != IDFT optimized"

def test_dft_vs_numpy():
    # Compare to numpy implementation
    # Initialize random data
    n = 16
    f = np.random.rand(n)
    z = np.random.rand(n) + 1j*np.random.rand(n)
    z_original = dft(f)
    z_numpy = np.fft.fft(f)/n
    assert np.allclose(z_original, z_numpy), "DFT != DFT numpy"
    f_original = idft(z)
    f_numpy = np.fft.ifft(z)*n
    assert np.allclose(f_original, f_numpy), "IDFT != IDFT numpy"

def test_dft_cosine():
    # Test implementation of DFT and IDFT with a cosine wave
    t = np.linspace(0, 2*pi, 16, endpoint=False)
    f = 3 + np.cos(t+1) + 2*np.cos(3*t+2) - 5*np.cos(4*t-1)
    z = dft_optimized(f)
    z_abs_calculated = np.array([ 3, 0.5, 0, 1, 2.5, 0, 0, 0])
    assert np.allclose(np.abs(z[0:8]), z_abs_calculated), "DFT(f) != calculated"