import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Add the "ablage" directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
ablage_dir = os.path.join(current_dir, "../ablage")
sys.path.insert(0, ablage_dir)

from function import Function
from low_pass_filter import low_pass_filter
from modulation import quadrature_modulate, quadrature_demodulate
from add import add

# Parameters
w = 2 * pi  # Base frequency
Ts = 0.025  # Sampling rate
n = range(0, 501)  # Sample points
wc = 4 * w  # Cut-off frequency
N = 15  # Filter length
w_mod1 = 5 * w  # Modulation frequency

# Create signals
signal1 = Function(n, Ts=Ts, function_handle=lambda t: np.cos(w * t))  # Cosine wave
signal2 = Function(n, Ts=Ts, function_handle=lambda t: t % (77 * Ts))  # Sawtooth wave

# Low-pass filter the signals
signal1_filtered = low_pass_filter(signal1, wc, N)
signal2_filtered = low_pass_filter(signal2, wc, N)

# Modulate signals using quadrature modulation
modulated_signal1, modulated_signal2 = quadrature_modulate(signal1_filtered, signal2_filtered, w_mod1)

# Combine modulated signals
combined_signal = add(modulated_signal1, modulated_signal2)

# Demodulate signals
demodulated_signal1, demodulated_signal2 = quadrature_demodulate(combined_signal, combined_signal, w_mod1)

# Low-pass filter the demodulated signals
recovered_signal1 = low_pass_filter(demodulated_signal1, wc, N)
recovered_signal2 = low_pass_filter(demodulated_signal2, wc, N)

corr = np.correlate(signal1.f, recovered_signal1.f, mode="full")
delay_index = np.argmax(corr) - len(signal1.f) + 1
time_shift = delay_index * Ts
recovered_signal1.t = [t - time_shift for t in recovered_signal1.t]

corr = np.correlate(signal2.f, recovered_signal2.f, mode="full")
delay_index = np.argmax(corr) - len(signal1.f) + 1
time_shift = delay_index * Ts
recovered_signal2.t = [t - time_shift for t in recovered_signal2.t]

# Visualization in a single window
plt.figure(figsize=(16, 12))

# Plot original signals
plt.subplot(3, 3, 1)
plt.plot(signal1.t, signal1.f, label="Cosine Wave")
plt.plot(signal2.t, signal2.f, label="Sawtooth Wave")
plt.title("Original Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot filtered signals
plt.subplot(3, 3, 2)
plt.plot(signal1_filtered.t, signal1_filtered.f, label="Filtered Signal 1")
plt.plot(signal2_filtered.t, signal2_filtered.f, label="Filtered Signal 2")
plt.title("Filtered Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot modulated signals
plt.subplot(3, 3, 3)
plt.plot(modulated_signal1.t, modulated_signal1.f, label="Modulated Signal 1")
plt.plot(modulated_signal2.t, modulated_signal2.f, label="Modulated Signal 2")
plt.title("Modulated Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot combined signal
plt.subplot(3, 3, 4)
plt.plot(combined_signal.t, combined_signal.f, label="Combined Signal", color="purple")
plt.title("Combined Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot demodulated signals
plt.subplot(3, 3, 5)
plt.plot(demodulated_signal1.t, demodulated_signal1.f, label="Demodulated Signal 1")
plt.plot(demodulated_signal2.t, demodulated_signal2.f, label="Demodulated Signal 2")
plt.title("Demodulated Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot recovered signals
plt.subplot(3, 3, 6)
plt.plot(recovered_signal1.t, recovered_signal1.f, label="Recovered Signal 1")
plt.plot(recovered_signal2.t, recovered_signal2.f, label="Recovered Signal 2")
plt.title("Recovered Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Compare original and reconstructed signals
plt.subplot(3, 3, 7)
plt.plot(signal1.t, signal1.f, label="Original Signal 1", linestyle="--")
plt.plot(recovered_signal1.t, recovered_signal1.f, label="Recovered Signal 1", linestyle=":")
plt.title("Comparison: Original vs. Recovered (Signal 1)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(3, 3, 8)
plt.plot(signal2.t, signal2.f, label="Original Signal 2", linestyle="--")
plt.plot(recovered_signal2.t, recovered_signal2.f, label="Recovered Signal 2", linestyle=":")
plt.title("Comparison: Original vs. Recovered (Signal 2)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
