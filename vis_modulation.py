import matplotlib.pyplot as plt
from math import pi
from function import Function
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate

def visualize_modulation_and_demodulation():
    # Define original function
    n = list(range(0, 10))
    Ts = 1
    f_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    f = Function(n, Ts, f=f_values)

    # Modulate and demodulate
    w_mod = pi
    f_mod = modulate(f, w_mod)
    f_demod = demodulate(f_mod, w_mod)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(f.t, f.f, label="Original Signal", marker='o')
    plt.plot(f_mod.t, f_mod.f, label="Modulated Signal", linestyle='--')
    plt.plot(f_demod.t, f_demod.f, label="Demodulated Signal", linestyle='-.')
    plt.title("Modulation and Demodulation")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_quadrature_modulation_and_demodulation():
    # Define original functions
    n = list(range(0, 10))
    Ts = 1
    f_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    g_values = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    f = Function(n, Ts, f=f_values)
    g = Function(n, Ts, f=g_values)

    # Quadrature modulation and demodulation
    w_mod = pi
    f_mod, g_mod = quadrature_modulate(f, g, w_mod)
    f_demod, g_demod = quadrature_demodulate(f_mod, g_mod, w_mod)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Original signals
    plt.subplot(3, 1, 1)
    plt.plot(f.t, f.f, label="Original f(t)", marker='o')
    plt.plot(g.t, g.f, label="Original g(t)", marker='x')
    plt.title("Original Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Modulated signals
    plt.subplot(3, 1, 2)
    plt.plot(f_mod.t, f_mod.f, label="f_mod(t)", linestyle='--')
    plt.plot(g_mod.t, g_mod.f, label="g_mod(t)", linestyle=':')
    plt.title("Quadrature Modulated Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Demodulated signals
    plt.subplot(3, 1, 3)
    plt.plot(f_demod.t, f_demod.f, label="f_demod(t)", linestyle='-.')
    plt.plot(g_demod.t, g_demod.f, label="g_demod(t)", linestyle='-')
    plt.title("Quadrature Demodulated Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_modulation_and_demodulation()
    visualize_quadrature_modulation_and_demodulation()
