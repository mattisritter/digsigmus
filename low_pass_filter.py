import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor
from convolution import discrete_convolution

def low_pass_filter(f: Function, wc, N, hamming_window=True) -> Function:
    # Initialize the low pass filter
    g = [0] * f.len
    filter_length = range(0, 2*N)
    w_hat = 2*wc/f.ws
    # Calculate the filter
    for k in range(f.len):
        if k in filter_length:
            g[k] = w_hat*np.sinc(w_hat*(k-N))
        if hamming_window:
            g[k] *= 0.54 - 0.46*np.cos(2*pi*k/(2*N))
    g = Function(f.n, Ts=f.Ts, f=g)
    return discrete_convolution(f, g)

if __name__ == "__main__":
    # Define first function
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    fcn_handle = lambda t: t % (77*Ts) # Sawtooth function
    f1 = Function(n, Ts=Ts, function_handle=fcn_handle)
    # Low pass filter the function
    wc = 8*pi # Cut-off frequency
    N = 15 # Filter length
    f1_low_pass = low_pass_filter(f1, wc, N, hamming_window=True)
    # Plot
    plt.figure(1)
    plt.plot(f1_low_pass.t, f1_low_pass.f, marker='x')
    plt.plot(f1.t, f1.f, marker='o')
    plt.title('Low-pass filtered function')
    plt.show()



