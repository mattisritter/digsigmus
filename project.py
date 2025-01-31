from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from function import Function
from sampling_rate_conversion import convert_sampling_rate
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate
from low_pass_filter import low_pass_filter
from fft import fft_iterative
from add import add

# Define the functions
samples = 32
t = np.linspace(0, 2*pi, samples, endpoint=False)
fcn = 3 + np.cos(t+1) + 2*np.cos(3*t+2) - 5*np.cos(4*t-1) + np.cos(13*t)
f1 = Function(range(samples), Ts=2*pi/samples, f=fcn)
fcn = (t % pi)*3
f2 = Function(range(samples), Ts=2*pi/samples, f=fcn)
# Low-pass filter the functions
wc = 7 # Cut-off frequency
N = 15 # Filter length
f1_low_pass = low_pass_filter(f1, wc, N)
f2_low_pass = low_pass_filter(f2, wc, N)
# Increase the sampling rate
factor = 2 # remove when sampling rate conversion is implemented
if factor != 1:
    N = N*factor
    samples = samples*factor
    f1_low_pass = convert_sampling_rate(f1_low_pass, 15, Ts_new=f1.Ts/factor)
    f2_low_pass = convert_sampling_rate(f2_low_pass, 15, Ts_new=f2.Ts/factor)
# Modulate the functions
use_quadrature = False
if not use_quadrature:
    w_mod1 = 8 # Modulation frequency
    w_mod2 = 23 # Modulation frequency
    f1_mod = modulate(f1_low_pass, w_mod1)
    f2_mod = modulate(f2_low_pass, w_mod2)
else: # Quadrature modulation
    w_mod = 8 # Modulation frequency
    f1_mod, f2_mod = quadrature_modulate(f1_low_pass, f2_low_pass, w_mod)
# Add the modulated functions
f_sum = add(f1_mod, f2_mod)
# Demodulate the sum
if not use_quadrature:
    f1_demod = demodulate(f_sum, w_mod1)
    f2_demod = demodulate(f_sum, w_mod2)
else: # Quadrature demodulation
    f1_demod, f2_demod = quadrature_demodulate(f_sum, w_mod)
# Low-pass filter the demodulated functions
f1_demod_low_pass = low_pass_filter(f1_demod, wc, N)
f2_demod_low_pass = low_pass_filter(f2_demod, wc, N)

# Caclulate Fourier Transforms
f1_fft = fft_iterative(f1.f)
f2_fft = fft_iterative(f2.f)
f1_low_pass_fft = fft_iterative(f1_low_pass.f[N:samples+N])
f2_low_pass_fft = fft_iterative(f2_low_pass.f[N:samples+N])
f1_mod_fft = fft_iterative(f1_mod.f[N:samples+N])
f2_mod_fft = fft_iterative(f2_mod.f[N:samples+N])
f_sum_fft = fft_iterative(f_sum.f[N:samples+N])
f1_demod_fft = fft_iterative(f1_demod.f[N:samples+N])
f2_demod_fft = fft_iterative(f2_demod.f[N:samples+N])
f1_demod_low_pass_fft = fft_iterative(f1_demod_low_pass.f[2*N:samples+2*N])
f2_demod_low_pass_fft = fft_iterative(f2_demod_low_pass.f[2*N:samples+2*N])


# Plot in Subplots
fig, axs = plt.subplots(6, 2, figsize=(10, 10))
# Time Domain
axs[0,0].plot(f1.t, f1.f, label='Cosine wave')
axs[0,0].plot(f2.t, f2.f, label='Sawtooth wave')
axs[0,0].set_title('Original functions')
#axs[0,0].legend()
axs[1,0].plot(f1_low_pass.t[N:samples+N], f1_low_pass.f[N:samples+N])
axs[1,0].plot(f2_low_pass.t[N:samples+N], f2_low_pass.f[N:samples+N])
axs[1,0].set_title('Low-pass filtered functions')
axs[2,0].plot(f1_mod.t[N:samples+N], f1_mod.f[N:samples+N])
axs[2,0].plot(f2_mod.t[N:samples+N], f2_mod.f[N:samples+N])
axs[2,0].set_title('Modulated functions')
axs[3,0].plot(f_sum.t[N:samples+N], f_sum.f[N:samples+N], color='tab:green')
axs[3,0].set_title('Sum of modulated functions')
axs[4,0].plot(f1_demod.t[N:samples+N], f1_demod.f[N:samples+N])
axs[4,0].plot(f2_demod.t[N:samples+N], f2_demod.f[N:samples+N])
axs[4,0].set_title('Demodulated functions')
axs[5,0].plot(f1_demod_low_pass.t[2*N:samples+2*N], f1_demod_low_pass.f[2*N:samples+2*N])
axs[5,0].plot(f2_demod_low_pass.t[2*N:samples+2*N], f2_demod_low_pass.f[2*N:samples+2*N])
axs[5,0].set_title('Reconstructed functions')
axs[5,0].set_xlabel('t [s]')

# Frequency Domain
axs[0,1].scatter(samples/2-1, 0, color='white') # Add white point to make the x-axis visible
axs[0,1].scatter(range(int(samples/2/factor)), np.abs(f1_fft[:int(samples/2/factor)]))
axs[0,1].scatter(range(int(samples/2/factor)), np.abs(f2_fft[:int(samples/2/factor)]))
axs[0,1].set_title('Fourier Coefficients of the functions')
axs[1,1].scatter(range(int(samples/2)), np.abs(f1_low_pass_fft[:int(samples/2)]))
axs[1,1].scatter(range(int(samples/2)), np.abs(f2_low_pass_fft[:int(samples/2)]))
axs[1,1].axvline(x=wc, color='k', linestyle='--', label='Cut-off frequency') # Add vertical line at cut-off frequency
axs[1,1].text(8.3/32, 0.5, '$\omega_c$', transform=axs[1,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
axs[1,1].set_title('Fourier Coefficients of the low-pass filtered functions')
axs[2,1].scatter(range(int(samples/2)), np.abs(f1_mod_fft[:int(samples/2)]))
axs[2,1].scatter(range(int(samples/2)), np.abs(f2_mod_fft[:int(samples/2)]))
if not use_quadrature:
    axs[2,1].axvline(x=w_mod1, color='k', linestyle='--') # Add vertical line at modulation frequency
    axs[2,1].text(9.2/32, 0.5, '$\omega_{mod_1}$', transform=axs[2,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
    axs[2,1].fill_between([w_mod1-wc, w_mod1+wc], 0, max(np.abs(f1_mod_fft[:int(samples/2)])), color='tab:blue', alpha=0.3) # Add a sqaure showing the frequency band
    axs[2,1].axvline(x=w_mod2, color='k', linestyle='--') # Add vertical line at modulation frequency
    axs[2,1].text(19.2/32, 0.6, '$\omega_{mod_2}$', transform=axs[2,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
    axs[2,1].fill_between([w_mod2-wc, w_mod2+wc], 0, max(np.abs(f2_mod_fft[:int(samples/2)])), color='tab:orange', alpha=0.3) # Add a sqaure showing the frequency band
else:
    axs[2,1].axvline(x=w_mod, color='k', linestyle='--')
    axs[2,1].text(8.5/16, 0.5, '$\omega_{mod}$', transform=axs[2,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
    axs[2,1].fill_between([w_mod-wc, w_mod+wc], 0, max(np.abs(f1_mod_fft[:int(samples/2)])), color='tab:blue', alpha=0.3)
    axs[2,1].fill_between([w_mod-wc, w_mod+wc], 0, max(np.abs(f2_mod_fft[:int(samples/2)])), color='tab:orange', alpha=0.3)
axs[2,1].set_title('Fourier Coefficients of the modulated functions')
axs[3,1].scatter(range(int(samples/2)), np.abs(f_sum_fft[:int(samples/2)]), color='tab:green')
axs[3,1].set_title('Fourier Coefficients of the sum')
axs[4,1].scatter(range(int(samples/2)), np.abs(f1_demod_fft[:int(samples/2)]))
axs[4,1].scatter(range(int(samples/2)), np.abs(f2_demod_fft[:int(samples/2)]))
axs[4,1].set_title('Fourier Coefficients of the demodulated functions')
axs[5,1].scatter(range(int(samples/2)), np.abs(f1_demod_low_pass_fft[:int(samples/2)]))
axs[5,1].scatter(range(int(samples/2)), np.abs(f2_demod_low_pass_fft[:int(samples/2)]))
axs[5,1].set_title('Fourier Coefficients of the reconstructed functions')
axs[5,1].set_xlabel('k')

plt.tight_layout()
plt.show()

# Plot low original vs reconstructed signals
# shift signals to align
if True:
    t1_lp = [t_i - (N)*f1_demod_low_pass.Ts for t_i in f1_demod_low_pass.t]
    t2_lp = [t_i - (N)*f2_demod_low_pass.Ts for t_i in f2_demod_low_pass.t]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].plot(f1_low_pass.t[N:samples+N], f1_low_pass.f[N:samples+N])
    axs[0].plot(t1_lp[2*N:samples+2*N], f1_demod_low_pass.f[2*N:samples+2*N], color='darkblue')
    axs[0].set_xlabel('t [s]')
    axs[1].plot(f2_low_pass.t[N:samples+N], f2_low_pass.f[N:samples+N], color='tab:orange')
    axs[1].plot(t2_lp[2*N:samples+2*N], f2_demod_low_pass.f[2*N:samples+2*N], color='red')
    axs[1].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()

