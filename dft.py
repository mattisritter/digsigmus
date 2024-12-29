import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor

def dft(f):
    """
    Discrete Fourier Transform.
    Parameters:
        f: 
    Returns:
        z:
    """
    n = len(f)
    B_star = calculate_B_star(n)
    # Perform the matrix vector multiplication implemented with a loop
    z = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        for l in range(n):
            z[k] += B_star[k, l]*f[l]/n
    return z

def idft(z):
    """
    Inverse Discrete Fourier Transform.
    Parameters:
        z: 
    Returns:
        f:
    """
    n = len(z)
    B = calculate_B(n)
    # Perform the matrix vector multiplication implemented with a loop
    f = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        for l in range(n):
            f[k] += B[k, l]*z[l]
    return f

def calculate_B(n: int) -> np.ndarray:
    """
    Calculate the B matrix for the IDFT.
    Parameters:
        n: int
            size of the matrix
    Returns:
        B: np.ndarray
            B matrix
    """
    B = np.zeros((n, n), dtype=np.complex128)
    for k in range(n):
        for l in range(n):
            B[k, l] = np.exp(2j*pi*k*l/n)
    return B

def calculate_B_star(n: int) -> np.ndarray:
    """
    Calculate the B* matrix for the DFT.
    Parameters:
        n: int
            size of the matrix
    Returns:
        B_star: np.ndarray
            B* matrix
    """
    B_star = calculate_B(n)
    return B_star.conj().T

def dft_optimized(f):
    """
    Discrete Fourier Transform.
    Parameters:
        f: 
    Returns:
        z:
    """
    n = len(f)
    b_star = calculate_B_star_row1(n)
    # Perform the matrix vector multiplication implemented with a loop
    z = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        for l in range(n):
            z[k] += b_star[(k*l)%n]*f[l]/n
    return z

def idft_optimized(z):
    """
    Inverse Discrete Fourier Transform.
    Parameters:
        z: 
    Returns:
        f:
    """
    n = len(z)
    b = calculate_B_row1(n)
    # Perform the matrix vector multiplication implemented with a loop
    f = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        for l in range(n):
            f[k] += b[(k*l)%n]*z[l]
    return f

def calculate_B_row1(n: int) -> np.ndarray:
    """
    Calculate the row of the B matrix for the IDFT.
    Parameters:
        n: int
            size of the matrix
    Returns:
        B: np.ndarray
            B matrix
    """
    b = np.zeros(n, dtype=np.complex128)
    for l in range(n):
        b[l] = np.exp(2j*pi*l/n)
    return b

def calculate_B_star_row1(n: int) -> np.ndarray:
    """
    Calculate the row of the B* matrix for the DFT.
    Parameters:
        n: int
            size of the matrix
    Returns:
        B_star: np.ndarray
            B* matrix
    """
    b_star = calculate_B_row1(n)
    return b_star.conj()

if __name__ == "__main__":
    # Test implementation of DFT and IDFT with a cosine wave
    t = np.linspace(0, 2*pi, 16, endpoint=False)
    f = 3 + np.cos(t+1) + 2*np.cos(3*t+2) - 5*np.cos(4*t-1)
    z = dft_optimized(f)
    f_reconstructed = idft_optimized(z)
    # Plot
    plt.figure(1)
    plt.plot(f)
    plt.plot(f_reconstructed)
    plt.title('Cosine wave')
    plt.show()

    plt.figure(2)
    plt.scatter(range(len(z)), np.abs(z))
    plt.title('DFT of a cosine wave')
    plt.show()

               



    
