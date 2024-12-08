import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor
from convolution import convolution_time_domain

def low_pass_filter(f: Function, wc, N, hamming_window=True) -> Function:
    """
    Low pass filter a function.
    Parameters:
        f: Function
            Function to be filtered
        wc: float
            Cut-off frequency [rad/s]
        N: int
            Filter length
        hamming_window=True: bool
            Use Hamming window
    Return:
        Function
            Low pass filtered function
    """
    g = _calculate_filter(f, wc, N, hamming_window)
    return convolution_time_domain(f, g)

def _calculate_filter(f: Function, wc, N, hamming_window=True) -> Function:
    """
    Calculate the low pass filter.
    Parameters:
        f: Function
            Function to be filtered
        wc: float
            Cut-off frequency [rad/s]
        N: int
            Filter length
        hamming_window=True: bool
            Use Hamming window
    Return:
        Function
            Filter-function
    """
    # Initialize the low pass filter
    len_g = 2*N+1
    g = [0] * len_g
    w_hat = 2*wc/f.ws
    # Calculate the filter
    for k in range(len(g)):
        g[k] = w_hat*np.sinc(w_hat*(k-N))
        if hamming_window:
            g[k] *= 0.54 - 0.46*np.cos(2*pi*k/(2*N+1))
    return Function(range(len(g)), Ts=f.Ts, f=g)

if __name__ == "__main__":
    # Define first function
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    fcn_handle = lambda t: t % (77*Ts) # Sawtooth function
    f1 = Function(n, Ts=Ts, function_handle=fcn_handle)
    # Low pass filter the function
    wc = 8*pi # Cut-off frequency
    N = 30 # Filter length
    g = _calculate_filter(f1, wc, N, hamming_window=False)
    g_hamming = _calculate_filter(f1, wc, N, hamming_window=True)
    f1_low_pass = low_pass_filter(f1, wc, N, hamming_window=True)
    # Plot
    plt.figure(1)
    plt.plot(f1_low_pass.t, f1_low_pass.f, marker='x')
    plt.plot(f1.t, f1.f, marker='o')
    plt.title('Low-pass filtered function')
    plt.show()
    # Plot the filter
    plt.figure(2)
    plt.plot(g.n, g.f, marker='x')
    plt.plot(g_hamming.n, g_hamming.f, marker='o')
    plt.title('Filter')
    plt.show()



