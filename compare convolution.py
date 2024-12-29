from convolution import convolution_time_domain
from fast_convolution import fast_convolution
from function import Function
from math import pi
import numpy as np
import time
import matplotlib.pyplot as plt

# Generate sawtooth function
samples = 32
t = np.linspace(0, 2*pi, samples, endpoint=False)
# shifted dirac function
fcn = [1 if i == 8 else 0 for i in range(samples)]
f1 = Function(range(samples), Ts=1, f=fcn)
fcn = (t % pi)*3
f2 = Function(range(samples), Ts=1, f=fcn)
# Convolve the functions
h = convolution_time_domain(f1, f2)
h_fast = fast_convolution(f1, f2, samples)
# Plot the functions
fig, axs = plt.subplots(3, 1)
axs[0].stem(f1.t, f1.f, use_line_collection=True)
axs[0].set_title('f1')
axs[1].stem(f2.t, f2.f, use_line_collection=True)
axs[1].set_title('f2')
axs[2].stem(h.t, h.f, use_line_collection=True)
axs[2].set_title('h')
plt.show()
    