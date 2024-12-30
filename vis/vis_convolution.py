# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
from math import pi
import numpy as np
import time
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from convolution import convolution_time_domain
from fast_convolution import fast_convolution
from function import Function

# Generate sawtooth function
samples = 32
t = np.linspace(0, 2*pi, samples, endpoint=False)
# shifted dirac function
fcn = [0, 0, 0, 0, 0, 0, 0, 1]
f2 = Function(range(len(fcn)), Ts=1, f=fcn)
fcn = (t % pi)*0.33333
f1 = Function(range(samples), Ts=1, f=fcn)
# Convolve the functions
h = convolution_time_domain(f1, f2)
h_fast = fast_convolution(f1, f2, samples)
# Plot the functions
fig, axs = plt.subplots(3, 1, figsize=(8, 4))
axs[0].scatter(40, 0, color='white')
axs[0].stem(f1.t, f1.f, use_line_collection=True)
axs[0].set_title('Function f')
axs[1].scatter(40, 0, color='white')
axs[1].stem(f2.t, f2.f, use_line_collection=True)
axs[1].set_title('Function g')
axs[2].stem(h.t, h.f, use_line_collection=True)
axs[2].set_title('Convolution h = f$\star$g')
plt.xlabel('n')
plt.tight_layout()
plt.show()

diff = np.abs([h.f[i] - h_fast.f[i] for i in range(samples)])
print(f"Maximum difference between the two convolution methods: {np.max(diff)}")

# Compare the time taken by the two convolution methods
length_f = 2**14
length_g = 256
f = Function(range(length_f), Ts=1, f=np.random.rand(length_f))
g = Function(range(length_g), Ts=1, f=np.random.rand(length_g))
N = 10
times = []
for i in range(N):
    start = time.time()
    convolution_time_domain(f, g)
    end = time.time()
    times.append(end-start)
time_average = np.mean(times)
times = []
for i in range(N):
    start = time.time()
    fast_convolution(f, g, 1024)
    end = time.time()
    times.append(end-start)
time_average_fast = np.mean(times)
print(f"Average time for time domain convolution: {time_average:.6f} s")
print(f"Average time for fast convolution: {time_average_fast:.6f} s")
    