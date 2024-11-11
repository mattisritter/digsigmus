from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from function import Function
from convolution import discrete_convolution
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate
from low_pass_filter import low_pass_filter
from add import add

# Define first function
w = 2 * pi # Frequency of the cosine wave
fcn_handle = lambda t: np.cos(w*t) # Cosine function
Ts = 0.01 # Sampling rate
n = range(0, 501)
f1 = Function(n, Ts=Ts, function_handle=fcn_handle)
# Define second function: sawtooth wave
fcn_handle = lambda t: t % (77*Ts) # Sawtooth function
f2 = Function(n, Ts=Ts, function_handle=fcn_handle)
# Plot
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(f1.t, f1.f, label='Cosine wave', marker='x')
plt.plot(f2.t, f2.f, label='Sawtooth wave', marker='o')
plt.title('Original functions')
plt.legend()
plt.show()

# Lowpass filter the functions
wc = 4*w # Cut-off frequency
N = 15 # Filter length
f1_low_pass = low_pass_filter(f1, wc, N)
f2_low_pass = low_pass_filter(f2, wc, N)
# Plot
plt.figure(2)
plt.plot(f1_low_pass.t, f1_low_pass.f, marker='x')
plt.plot(f2_low_pass.t, f2_low_pass.f, marker='o')
plt.title('Low-pass filtered functions')
plt.show()

# # Modulate the functions
# w_mod1 = 5*w
# w_mod2 = 15*w
# f1_mod = modulate(f1_low_pass, w_mod1)
# f2_mod = modulate(f2_low_pass, w_mod2)
# # Plot
# plt.figure(3)
# plt.plot(f1_mod.t, f1_mod.f, marker='x')
# plt.plot(f2_mod.t, f2_mod.f, marker='o')
# plt.title('Modulated functions')
# plt.show()
# print(f1_mod.len)
# print(f2_mod.len)

# # Add the modulated functions
# f_sum = add(f1_mod, f2_mod)
# # Plot
# plt.figure(4)
# plt.plot(f_sum.t, f_sum.f, marker='s', color='tab:green')
# plt.title('Sum of modulated functions')
# plt.show()
# print(f_sum.len)

# # Demodulate the sum
# f1_demod = demodulate(f_sum, w_mod1)
# f2_demod = demodulate(f_sum, w_mod2)
# # Plot
# plt.figure(5)
# plt.plot(f1_demod.t, f1_demod.f, marker='x')
# plt.plot(f2_demod.t, f2_demod.f, marker='o')
# plt.title('Demodulated functions')
# plt.show()

# Modulate using quadrature modulation
w_mod1 = 5*w
f1_mod, f2_mod = quadrature_modulate(f1_low_pass, f2_low_pass, w_mod1)
# Plot
plt.figure(3)
plt.plot(f1_mod.t, f1_mod.f, marker='x')
plt.plot(f2_mod.t, f2_mod.f, marker='o')
plt.title('Modulated functions')
plt.show()

# Add the modulated functions
f_sum = add(f1_mod, f2_mod)
# Plot
plt.figure(4)
plt.plot(f_sum.t, f_sum.f, marker='s', color='tab:green')
plt.title('Sum of modulated functions')
plt.show()

# Demodulate using quadrature demodulation
f1_demod, f2_demod = quadrature_demodulate(f_sum, f_sum, w_mod1)
# Plot
plt.figure(5)
plt.plot(f1_demod.t, f1_demod.f, marker='x')
plt.plot(f2_demod.t, f2_demod.f, marker='o')
plt.title('Demodulated functions')
plt.show()

# Lowpass filter the demodulated functions
f1_demod_low_pass = low_pass_filter(f1_demod, wc, N)
f2_demod_low_pass = low_pass_filter(f2_demod, wc, N)
# Plot
plt.figure(6)
plt.plot(f1_demod_low_pass.t, f1_demod_low_pass.f, marker='x')
plt.plot(f2_demod_low_pass.t, f2_demod_low_pass.f, marker='o')
plt.title('Reconstructed functions')
plt.show()

# compare the original and reconstructed functions
plt.figure(7)
# shift the new signal to the left by N
t1_lp = [t_i - (2*N)*f1_demod_low_pass.Ts for t_i in f1_demod_low_pass.t]
plt.plot(f1.t, f1.f, label='Original')
plt.plot(t1_lp, f1_demod_low_pass.f, label='Reconstructed', color='navy')
plt.legend()
plt.title('Comparison of original and reconstructed functions')
plt.show()
plt.figure(8)
# shift the new signal to the left by N
t2_lp = [t_i - (2*N)*f2_demod_low_pass.Ts for t_i in f2_demod_low_pass.t]
plt.plot(f2.t, f2.f, label='Original', color='tab:orange')
plt.plot(t2_lp, f2_demod_low_pass.f, label='Reconstructed', color='darkred')
plt.legend()
plt.title('Comparison of original and reconstructed functions')
plt.show()