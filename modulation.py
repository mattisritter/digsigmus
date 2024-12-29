import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor

def modulate(f: Function, w_mod) -> Function:
    """
    Modulate a function.
    Parameters:
        f: Function
            Function to be modulated
        w_mod: float or int
            Modulation frequency [rad/s]
    Return:
        Function
            Modulated function
    """
    f_mod = [f.f[i] * np.cos(w_mod * f.t[i]) for i in range(f.len)]
    return Function(f.n, Ts=f.Ts, f=f_mod)

def demodulate(f: Function, w_mod) -> Function:
    """
    Demodulate a function.
    Parameters:
        f: Function
            Function to be demodulated
        w_mod: float or int
            Demodulation frequency [rad/s]
    Return:
        Function
            Demodulated function
    """
    f_demod = [f.f[i] * 2 * np.cos(w_mod * f.t[i]) for i in range(f.len)]
    return Function(f.n, Ts=f.Ts, f=f_demod)

def quadrature_modulate(f: Function, g: Function, w_mod):
    """
    Modulate two functions using quadrature modulation.
    Parameters:
        f: Function
            First function to be modulated
        g: Function
            Second function to be modulated
        w_mod: float or int
            Modulation frequency [rad/s]
    Return:
        Function, Function
            Modulated functions
    """
    f_mod = [f.f[i] * np.cos(w_mod * f.t[i]) for i in range(f.len)]
    g_mod = [g.f[i] * np.sin(w_mod * g.t[i]) for i in range(g.len)]
    return Function(f.n, Ts=f.Ts, f=f_mod), Function(g.n, Ts=g.Ts, f=g_mod)

def quadrature_demodulate(h, w_mod):
    """
    Demodulate two functions using quadrature modulation.
    Parameters:
        h: Function
            Function to be demodulated
        w_mod: float or int
            Demodulation frequency [rad/s]
    Return:
        Function, Function
            Demodulated functions
    """
    f_demod = [h.f[i] * 2 * np.cos(w_mod * h.t[i]) for i in range(h.len)]
    g_demod = [h.f[i] * 2 * np.sin(w_mod * h.t[i]) for i in range(h.len)]
    return Function(h.n, Ts=h.Ts, f=f_demod), Function(h.n, Ts=h.Ts, f=g_demod)


if __name__ == "__main__":
    # Define first function
    n = range(0, 50)
    Ts = 1
    f1 = list(n)
    function1 = Function(n, Ts, f=f1)
    # Modulate the function
    w_mod = 1
    function1_mod = modulate(function1, w_mod)
    # Plot
    plt.figure(1)
    plt.plot(function1_mod.t, function1_mod.f, marker='x')
    plt.plot(function1.t, function1.f, marker='o')
    plt.title('Modulated function')
    plt.show()
    # Demodulate the function
    function1_demod = demodulate(function1_mod, w_mod)
    # Plot
    plt.figure(2)
    plt.plot(function1_demod.t, function1_demod.f, marker='x')
    plt.plot(function1.t, function1.f, marker='o')
    plt.title('Demodulated function')
    plt.show()
