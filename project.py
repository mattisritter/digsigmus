from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from function import Function
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate
from low_pass_filter import low_pass_filter
from fft import fft_iterative
from add import add

plot = False
analyse_in_frequency_domain = True

# Define first function
w = 2*pi/6.4 # Frequency of the cosine wave
Ts = 0.2 # Sampling rate
n = range(0, 256)
fcn_handle = lambda t: np.cos(w*t) # Cosine function
f1 = Function(n, Ts=Ts, function_handle=fcn_handle)
# Define second function: sawtooth wave
fcn_handle = lambda t: (t % 12)/8 - 0.5 # Sawtooth function
f2 = Function(n, Ts=Ts, function_handle=fcn_handle)
# Plot
import matplotlib.pyplot as plt
if plot:
    plt.figure(1)
    plt.plot(f1.t, f1.f, label='Cosine wave', marker='x')
    plt.plot(f2.t, f2.f, label='Sawtooth wave', marker='o')
    plt.title('Original functions')
    plt.legend()
    plt.show()

# Lowpass filter the functions
wc = 2*w # Cut-off frequency
N = 50 # Filter length
f1_low_pass = low_pass_filter(f1, wc, N)
f2_low_pass = low_pass_filter(f2, wc, N)
# Plot
if plot:
    plt.figure(2)
    plt.plot(f1_low_pass.t, f1_low_pass.f, marker='x')
    plt.plot(f2_low_pass.t, f2_low_pass.f, marker='o')
    plt.title('Low-pass filtered functions')
    plt.show()

# Modulate the functions
w_mod1 = 5*w
w_mod2 = 10*w
f1_mod = modulate(f1_low_pass, w_mod1)
f2_mod = modulate(f2_low_pass, w_mod2)
# Plot
if plot:
    plt.figure(3)
    plt.plot(f1_mod.t, f1_mod.f, marker='x')
    plt.plot(f2_mod.t, f2_mod.f, marker='o')
    plt.title('Modulated functions')
    plt.show()

# Add the modulated functions
f_sum = add(f1_mod, f2_mod)
# Plot
if plot:
    plt.figure(4)
    plt.plot(f_sum.t, f_sum.f, marker='s', color='tab:green')
    plt.title('Sum of modulated functions')
    plt.show()

# Demodulate the sum
f1_demod = demodulate(f_sum, w_mod1)
f2_demod = demodulate(f_sum, w_mod2)
# Plot
if plot:
    plt.figure(5)
    plt.plot(f1_demod.t, f1_demod.f, marker='x')
    plt.plot(f2_demod.t, f2_demod.f, marker='o')
    plt.title('Demodulated functions')
    plt.show()

# # Modulate using quadrature modulation
# w_mod1 = 5*w
# f1_mod, f2_mod = quadrature_modulate(f1_low_pass, f2_low_pass, w_mod1)
# # Plot
# plt.figure(3)
# plt.plot(f1_mod.t, f1_mod.f, marker='x')
# plt.plot(f2_mod.t, f2_mod.f, marker='o')
# plt.title('Modulated functions')
# plt.show()

# # Add the modulated functions
# f_sum = add(f1_mod, f2_mod)
# # Plot
# plt.figure(4)
# plt.plot(f_sum.t, f_sum.f, marker='s', color='tab:green')
# plt.title('Sum of modulated functions')
# plt.show()

# # Demodulate using quadrature demodulation
# f1_demod, f2_demod = quadrature_demodulate(f_sum, f_sum, w_mod1)
# # Plot
# plt.figure(5)
# plt.plot(f1_demod.t, f1_demod.f, marker='x')
# plt.plot(f2_demod.t, f2_demod.f, marker='o')
# plt.title('Demodulated functions')
# plt.show()

# Lowpass filter the demodulated functions
f1_demod_low_pass = low_pass_filter(f1_demod, wc, N)
f2_demod_low_pass = low_pass_filter(f2_demod, wc, N)
# Plot
if plot:
    plt.figure(6)
    plt.plot(f1_demod_low_pass.t, f1_demod_low_pass.f, marker='x')
    plt.plot(f2_demod_low_pass.t, f2_demod_low_pass.f, marker='o')
    plt.title('Reconstructed functions')
    plt.show()

if plot:
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

## Compare in frequency domain
if analyse_in_frequency_domain:
    # Perform FFT
    f1_fft = fft_iterative(f1.f)
    f1_low_pass_fft = fft_iterative(f1_low_pass.f[N:256+N])
    f1_mod_fft = fft_iterative(f1_mod.f[N:256+N])
    f_sum_fft = fft_iterative(f_sum.f[N:256+N])
    f1_demod_fft = fft_iterative(f1_demod.f[N:256+N])
    f1_reconstructed_fft = fft_iterative(f1_demod_low_pass.f[2*N:256+2*N])
    # Perform FFT
    f2_fft = fft_iterative(f2.f)
    f2_low_pass_fft = fft_iterative(f2_low_pass.f[N:256+N])
    f2_mod_fft = fft_iterative(f2_mod.f[N:256+N])
    f2_demod_fft = fft_iterative(f2_demod.f[N:256+N])
    f2_reconstructed_fft = fft_iterative(f2_demod_low_pass.f[2*N:256+2*N])
    # Plot FFT
    # plt.figure(1)
    # plt.scatter(range(256), np.abs(f1_fft[0:256]), label='Orginal')
    # plt.scatter(range(256), np.abs(f1_low_pass_fft[0:256]), label='Low-pass filtered')
    # plt.title('FFT of the functions')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Plot FFT
    # plt.figure(2)
    # plt.scatter(range(256), np.abs(f2_fft[0:256]), label='Orginal')
    # plt.scatter(range(256), np.abs(f2_low_pass_fft[0:256]), label='Low-pass filtered')
    # plt.title('FFT of the functions')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # # Plot Modulation
    # plt.figure(3)
    # plt.scatter(range(256), np.abs(f1_mod_fft[0:256]), label='Modulated')  
    # plt.scatter(range(256), np.abs(f2_mod_fft[0:256]), label='Modulated')
    # plt.title('FFT of the modulated functions')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].scatter(range(128), np.abs(f1_fft[0:128]), label='Cosine wave')  
    axs[0].scatter(range(128), np.abs(f2_fft[0:128]), label='Sawtooth wave')
    axs[0].set_title('FFT of the functions')
    axs[0].legend()
    axs[1].scatter(range(128), np.abs(f1_mod_fft[0:128]))
    axs[1].scatter(range(128), np.abs(f2_mod_fft[0:128]))
    axs[1].set_title('FFT of the modulated functions')
    axs[2].scatter(range(128), np.abs(f1_reconstructed_fft[0:128]))
    axs[2].scatter(range(128), np.abs(f2_reconstructed_fft[0:128]))
    axs[2].set_title('FFT of the reconstructed functions')
    plt.show()

plot_as_subplots = True
if plot_as_subplots:
    # Plot
    fig, axs = plt.subplots(6, 2, figsize=(10, 10))
    axs[0,0].plot(f1.t, f1.f, label='Cosine wave')
    axs[0,0].plot(f2.t, f2.f, label='Sawtooth wave')
    axs[0,0].set_title('Original functions')
    axs[0,0].legend()
    axs[1,0].plot(f1_low_pass.t[N: -N], f1_low_pass.f[N: -N])
    axs[1,0].plot(f2_low_pass.t[N: -N], f2_low_pass.f[N: -N])
    axs[1,0].set_title('Low-pass filtered functions')
    axs[2,0].plot(f1_mod.t[N: -N], f1_mod.f[N: -N])
    axs[2,0].plot(f2_mod.t[N: -N], f2_mod.f[N: -N])
    axs[2,0].set_title('Modulated functions')
    axs[3,0].plot(f_sum.t[N: -N], f_sum.f[N: -N], color='tab:green')
    axs[3,0].set_title('Sum of modulated functions')
    axs[4,0].plot(f1_demod.t[N: -N], f1_demod.f[N: -N])
    axs[4,0].plot(f2_demod.t[N: -N], f2_demod.f[N: -N])
    axs[4,0].set_title('Demodulated functions')
    axs[5,0].plot(f1_demod_low_pass.t[2*N: -2*N], f1_demod_low_pass.f[2*N: -2*N])
    axs[5,0].plot(f2_demod_low_pass.t[2*N: -2*N], f2_demod_low_pass.f[2*N: -2*N])
    axs[5,0].set_title('Reconstructed functions')
    # Compare to frequency domain
    axs[0,1].scatter(range(128), np.abs(f1_fft[0:128]), label='Cosine wave')
    axs[0,1].scatter(range(128), np.abs(f2_fft[0:128]), label='Sawtooth wave')
    axs[0,1].set_title('FFT of the functions')
    axs[0,1].legend()
    axs[1,1].scatter(range(128), np.abs(f1_low_pass_fft[0:128]))
    axs[1,1].scatter(range(128), np.abs(f2_low_pass_fft[0:128]))
    axs[1,1].set_title('FFT of the low-pass filtered functions')
    axs[2,1].scatter(range(128), np.abs(f1_mod_fft[0:128]))
    axs[2,1].scatter(range(128), np.abs(f2_mod_fft[0:128]))
    axs[2,1].set_title('FFT of the modulated functions')
    axs[3,1].scatter(range(128), np.abs(f_sum_fft[0:128]), color='tab:green')
    axs[3,1].set_title('FFT of the sum of modulated functions')
    axs[4,1].scatter(range(128), np.abs(f1_demod_fft[0:128]))
    axs[4,1].scatter(range(128), np.abs(f2_demod_fft[0:128]))
    axs[4,1].set_title('FFT of the demodulated functions')
    axs[5,1].scatter(range(128), np.abs(f1_reconstructed_fft[0:128]))
    axs[5,1].scatter(range(128), np.abs(f2_reconstructed_fft[0:128]))
    axs[5,1].set_title('FFT of the reconstructed functions')
    plt.show()
