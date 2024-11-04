from math import pi, ceil
import numpy as np

def convert_sampling_rate(fn, ws, ws_new):
    Ts = 2 * pi / ws
    Ts_new = 2 * pi / ws_new
    num_new_samples = ceil(len(fn) * ws_new / ws)
    fn_new = [0] * num_new_samples
    
    for i in range(num_new_samples):
        n_new = i * Ts_new
        for j in range(len(fn)):
            n = j * Ts
            fn_new[i] += fn[j] * np.sinc((n_new - n) / Ts)
    
    return fn_new

# Example usage
w = 2 * pi # Frequency of the cosine wave
ws = 20 # Sampling rate of the cosine wave
ws_new = 40 # New sampling rate
# Generate cosine wave
fn = [np.cos(w * i / ws) for i in range(-100, 100)]
# Convert the sampling rate
fn_new = convert_sampling_rate(fn, ws, ws_new)
print(fn_new)


