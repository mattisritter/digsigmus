from math import pi, ceil, floor
import numpy as np
from function import Function
import matplotlib.pyplot as plt

## Errors ad t=0
def convert_sampling_rate(f, N, ws_new=None, Ts_new=None):
    if ws_new is None:
        ws_new = 2*pi/Ts_new
    elif Ts_new is None:
        Ts_new = 2*pi/ws_new
    else:
        raise ValueError("Either Ts_new or ws_new must be provided.")
    #
    beta = f.ws/ws_new
    n_new = range(ceil(f.n[0]/beta), ceil(f.n[-1]/beta)+1)
    f_new = [0]*len(n_new)
    for l in n_new:
        n0 = ceil(l*beta)
        for n in range(n0-N, n0+N):
            if n in f.n:
                f_new[l] += f.f[f.n.index(n)]*np.sinc(n-l*beta)
    return Function(n_new, Ts=Ts_new, f=f_new)

# Example usage
w = 2 * pi # Frequency of the cosine wave
fcn_handle = lambda t: np.cos(w*t) # Cosine function
Ts = 0.1 # Sampling rate
n = range(0, 41)
# Create the function object
f = Function(n, Ts=Ts, function_handle=fcn_handle)

# Convert the sampling rate
Ts_new = 0.04 # New sampling rate
f_new = convert_sampling_rate(f, N=5, Ts_new=Ts_new)


# Plot the original and new signals
plt.plot(f.t, f.f, label='Original', marker='x')
plt.plot(f_new.t, f_new.f, label='New', marker='o')
plt.legend()
plt.show()


