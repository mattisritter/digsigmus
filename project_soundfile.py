from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from function import Function
from sampling_rate_conversion import convert_sampling_rate
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate
from low_pass_filter import low_pass_filter
from fft import fft_iterative
from add import add
# Soundfile
import soundfile as sf
import playsound as ps

# Import the soundfiles
f1, fs1 = sf.read("soundfiles/Jodler.wav")
f2, fs2 = sf.read("soundfiles/Violine.wav")
# Convert to mono
f1 = f1[:, 0].reshape(-1, 1)
f2 = f2[:, 0].reshape(-1, 1)
# Pad to length of 2**16
samples = 2**16
if len(f1) > samples:
    f1 = f1[:samples]
else:
    f1 = np.pad(f1, ((0, samples - len(f1)), (0, 0)), mode='constant')
if len(f2) > samples:
    f2 = f2[:samples]
else: 
    f2 = np.pad(f2, ((0, samples - len(f2)), (0, 0)), mode='constant')

# Create the Function objects
f1 = Function(range(len(f1)), Ts=1/fs1, f=f1)
f2 = Function(range(len(f2)), Ts=1/fs2, f=f2)
# Low-pass filter the functions
wc = 20000 # Cut-off frequency
N = 135 # Filter length
f1_low_pass = low_pass_filter(f1, wc, N)
f2_low_pass = low_pass_filter(f2, wc, N)
# Modulate the functions
w_mod1 = 25000 # Modulation frequency
w_mod2 = 47000 # Modulation frequency
f1_mod = modulate(f1_low_pass, w_mod1)
f2_mod = modulate(f2_low_pass, w_mod2)
# Add the modulated functions
f_sum = add(f1_mod, f2_mod)
# Demodulate the sum
f1_demod = demodulate(f_sum, w_mod1)
f2_demod = demodulate(f_sum, w_mod2)
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

relevant_samples = 2**16
# Base frequency
w_base = f1.ws/relevant_samples
cut_off = wc/w_base
mod1 = w_mod1/w_base
mod2 = w_mod2/w_base
# Frequency Domain
# Highest coefficient in the Fourier Transform
max_coeff1 = max(np.abs(f1_fft[:int(relevant_samples/2)]))
max_coeff2 = max(np.abs(f2_fft[:int(relevant_samples/2)]))
max_coeff = max(max_coeff1, max_coeff2)[0]
axs[0,1].scatter(range(int(relevant_samples/2)), np.abs(f1_fft[:int(relevant_samples/2)]))
axs[0,1].scatter(range(int(relevant_samples/2)), np.abs(f2_fft[:int(relevant_samples/2)]))
axs[0,1].set_title('Fourier Coefficients of the functions')
axs[0,1].set_ylim(0, max_coeff)
# Highest coefficient in the Fourier Transform
max_coeff1 = max(np.abs(f1_low_pass_fft[:int(relevant_samples/2)]))
max_coeff2 = max(np.abs(f2_low_pass_fft[:int(relevant_samples/2)]))
max_coeff = max(max_coeff1, max_coeff2)
axs[1,1].scatter(range(int(relevant_samples/2)), np.abs(f1_low_pass_fft[:int(relevant_samples/2)]))
axs[1,1].scatter(range(int(relevant_samples/2)), np.abs(f2_low_pass_fft[:int(relevant_samples/2)]))
axs[1,1].axvline(x=cut_off, color='k', linestyle='--', label='Cut-off frequency') # Add vertical line at cut-off frequency
axs[1,1].text(6/32, 0.5, '$\omega_c$', transform=axs[1,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
axs[1,1].set_title('Fourier Coefficients of the low-pass filtered functions')
axs[1,1].set_ylim(0, max_coeff)
axs[2,1].scatter(range(int(relevant_samples/2)), np.abs(f1_mod_fft[:int(relevant_samples/2)]))
axs[2,1].scatter(range(int(relevant_samples/2)), np.abs(f2_mod_fft[:int(relevant_samples/2)]))
axs[2,1].axvline(x=mod1, color='k', linestyle='--') # Add vertical line at modulation frequency
axs[2,1].text(7/32, 0.7, '$\omega_{mod_1}$', transform=axs[2,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
axs[2,1].fill_between([mod1-cut_off, mod1+cut_off], 0, max(np.abs(f1_mod_fft[:int(relevant_samples/2)])), color='tab:blue', alpha=0.3) # Add a sqaure showing the frequency band
axs[2,1].axvline(x=mod2, color='k', linestyle='--') # Add vertical line at modulation frequency
axs[2,1].text(12.2/32, 0.6, '$\omega_{mod_2}$', transform=axs[2,1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
axs[2,1].fill_between([mod2-cut_off, mod2+cut_off], 0, max(np.abs(f2_mod_fft[:int(relevant_samples/2)])), color='tab:orange', alpha=0.3) # Add a sqaure showing the frequency band
axs[2,1].set_title('Fourier Coefficients of the modulated functions')
axs[2,1].set_ylim(0, max_coeff)
axs[3,1].scatter(range(int(relevant_samples/2)), np.abs(f_sum_fft[:int(relevant_samples/2)]), color='tab:green')
axs[3,1].set_title('Fourier Coefficients of the sum')
axs[3,1].set_ylim(0, max_coeff)
axs[4,1].scatter(range(int(relevant_samples/2)), np.abs(f1_demod_fft[:int(relevant_samples/2)]))
axs[4,1].scatter(range(int(relevant_samples/2)), np.abs(f2_demod_fft[:int(relevant_samples/2)]))
axs[4,1].set_title('Fourier Coefficients of the demodulated functions')
axs[4,1].set_ylim(0, max_coeff)
axs[5,1].scatter(range(int(relevant_samples/2)), np.abs(f1_demod_low_pass_fft[:int(relevant_samples/2)]))
axs[5,1].scatter(range(int(relevant_samples/2)), np.abs(f2_demod_low_pass_fft[:int(relevant_samples/2)]))
axs[5,1].set_title('Fourier Coefficients of the reconstructed functions')
axs[5,1].set_xlabel('k')
axs[5,1].set_ylim(0, max_coeff)

plt.tight_layout()

plt.show()

# Plot low original vs reconstructed signals
if False:
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

# Save and play the soundfiles
if True:
    # Save the soundfiles
    #sf.write("soundfiles/Jodler_low_pass.wav", f1_low_pass.f, fs1)
    #sf.write("soundfiles/Violine_low_pass.wav", f2_low_pass.f, fs2)
    sf.write("soundfiles/Jodler_mod_overlap.wav", f1_mod.f, fs1)
    sf.write("soundfiles/Violine_mod_overlap.wav", f2_mod.f, fs2)
    sf.write("soundfiles/Sum_overlap.wav", f_sum.f, fs1)
    sf.write("soundfiles/Jodler_demod_overlap.wav", f1_demod.f, fs1)
    sf.write("soundfiles/Violine_demod_overlap.wav", f2_demod.f, fs2)
    sf.write("soundfiles/Jodler_demod_low_pass_overlap.wav", f1_demod_low_pass.f, fs1)
    sf.write("soundfiles/Violine_demod_low_pass_overlap.wav", f2_demod_low_pass.f, fs2)

    # Play the soundfiles
    # Jodler
    ps.playsound("soundfiles/Jodler.wav")
    ps.playsound("soundfiles/Jodler_low_pass.wav")
    # ps.playsound("soundfiles/Jodler_mod.wav")
    # ps.playsound("soundfiles/Sum.wav")
    # ps.playsound("soundfiles/Jodler_demod.wav")
    ps.playsound("soundfiles/Jodler_demod_low_pass_overlap.wav")
    # Violine
    ps.playsound("soundfiles/Violine.wav")
    ps.playsound("soundfiles/Violine_low_pass.wav")
    # ps.playsound("soundfiles/Violine_mod.wav")
    # ps.playsound("soundfiles/Sum.wav")
    # ps.playsound("soundfiles/Violine_demod.wav")
    ps.playsound("soundfiles/Violine_demod_low_pass_overlap.wav")
