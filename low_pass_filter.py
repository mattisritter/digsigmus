import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor
from convolution import discrete_convolution

def low_pass_filter(f, wc, N, hamming_window=False):
    # f: input function
    # wc: cut-off frequency
    # N: filter length
    # hamming_window: whether to apply a Hamming window
    # h: output function
    # Determine the length of the output convolution
    len_g = len(f.n)
    
    # Initialize the output convolution
    g = [0] * len_g
    filter_length = range(0, 2*N)
    w_hat = 2*wc/f.ws
    # Perform the convolution
    for k in range(len_g):
        if k in filter_length:
            g[k] = w_hat*np.sinc(w_hat*(k-N))
        if hamming_window:
            g[k] *= 0.54 - 0.46*np.cos(2*pi*k/(2*N))
    g = Function(f.n, Ts=f.Ts, f=g)
    h = discrete_convolution(f, g)
    return h

# Example usage
w = 2 * pi # Frequency of the cosine wave
fcn_handle = lambda t: np.cos(w*t) + 0.5*np.cos(2*w*t) # Cosine function
Ts = 0.05 # Sampling rate
n = range(0, 101)
# Create the function object
f = Function(n, Ts=Ts, function_handle=fcn_handle)
# Cut-off frequency
wc = 1.5*w
# Filter length
N = 7
# Apply the low-pass filter
f_low_pass = low_pass_filter(f, wc, N, hamming_window=False)

# Plot the original and new signals
plt.plot(f.t, f.f, label='Original', marker='x')
# shift the new signal to the left by N
t_lp = [t_i - N*f.Ts for t_i in f_low_pass.t]
plt.plot(t_lp, f_low_pass.f, label='New', marker='o')
plt.legend()
plt.show()

