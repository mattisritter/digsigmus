from math import pi
import numpy as np
from function import Function

def fft_recursive(f: Function) -> Function:
    """
    Perform the Fast Fourier Transform (FFT) recursively on a Function object.
    Parameters:
        f: Function
            Input function in the time domain.
    Return:
        Function
            Transformed function in the frequency domain.
    """
    N = f.len
    if N <= 1:
        return f

    even = fft_recursive(Function(f.n[0::2], Ts=f.Ts, f=f.f[0::2]))
    odd = fft_recursive(Function(f.n[1::2], Ts=f.Ts, f=f.f[1::2]))

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    f_even = even.f
    f_odd = odd.f

    combined = np.concatenate([
        f_even + factor[:N // 2] * f_odd,
        f_even - factor[:N // 2] * f_odd
    ])

    return Function(f.n, Ts=f.Ts, f=combined)

if __name__ == "__main__":
    # Example usage
    n = range(8)
    Ts = 1
    f_values = [0, 1, 2, 3, 4, 5, 6, 7]
    time_domain_function = Function(n, Ts=Ts, f=f_values)

    # Perform FFT recursively
    frequency_domain_function = fft_recursive(time_domain_function)

    # Print the results
    print("Input function (time domain):", time_domain_function.f)
    print("FFT result (frequency domain):", frequency_domain_function.f)
