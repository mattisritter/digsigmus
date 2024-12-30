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
    n = range(0, 32)
    ws = pi
    f1 = list(n)
    f2 = [10] * len(n)
    function1 = Function(n, ws=ws, f=f1)
    function2 = Function(n, ws=ws, f=f2)
    # Modulate the function
    w_mod1 = 4
    function1_mod = modulate(function1, w_mod1)
    w_mod2 = 8
    function2_mod = modulate(function2, w_mod2)
   
    function1_demod = demodulate(function1_mod, w_mod1)
    function2_demod = demodulate(function2_mod, w_mod2)
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 4))
    axs[0,0].plot(function1_mod.t, function1_mod.f)
    axs[0,0].plot(function2_mod.t, function2_mod.f)
    axs[0,0].plot(function1.t, function1.f, color='darkblue')
    axs[0,0].plot(function2.t, function2.f, color='darkred')
    axs[0,0].set_title('Modulated functions')
    axs[0,0].grid()
    axs[1,0].plot(function1_demod.t, function1_demod.f)
    axs[1,0].plot(function2_demod.t, function2_demod.f)
    axs[1,0].plot(function1.t, function1.f, color='darkblue')
    axs[1,0].plot(function2.t, function2.f, color='darkred')
    axs[1,0].set_title('Demodulated functions')
    axs[1,0].grid()
    # Quadrature modulation
    w_mod = 6
    function1_mod, function2_mod = quadrature_modulate(function1, function2, w_mod)
    function1_demod, _ = quadrature_demodulate(function1_mod, w_mod)
    _, function2_demod = quadrature_demodulate(function2_mod, w_mod)
    axs[0,1].plot(function1_mod.t, function1_mod.f)
    axs[0,1].plot(function2_mod.t, function2_mod.f)
    axs[0,1].plot(function1.t, function1.f, color='darkblue')
    axs[0,1].plot(function2.t, function2.f, color='darkred')
    axs[0,1].set_title('Quadrature Modulated functions')
    axs[0,1].grid()
    axs[1,1].plot(function1_demod.t, function1_demod.f)
    axs[1,1].plot(function2_demod.t, function2_demod.f)
    axs[1,1].plot(function1.t, function1.f, color='darkblue')
    axs[1,1].plot(function2.t, function2.f, color='darkred')
    axs[1,1].set_title('Quadrature Demodulated functions')
    axs[1,1].grid()
    plt.tight_layout()
    plt.show()
