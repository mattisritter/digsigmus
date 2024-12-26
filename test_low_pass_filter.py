from low_pass_filter import low_pass_filter
import numpy as np
from function import Function
from math import pi
import matplotlib.pyplot as plt

def test_low_pass1():
    # Test case 1: cosine with wc >> w
    w = 2 * pi # Frequency of the cosine wave
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    fcn_handle = lambda t: np.cos(w*t) # Cosine function
    f = Function(n, Ts=Ts, function_handle=fcn_handle)
    N = 32
    f_low_pass = low_pass_filter(f, 4*w, N)
    assert np.allclose(f_low_pass.f[2*N:-2*N], f.f[N:-N], atol=0.01), "Test case 1 failed"

def test_low_pass2():
    # Test case 2: cosine with wc << w
    w = 2 * pi # Frequency of the cosine wave
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    fcn_handle = lambda t: np.cos(w*t) # Cosine function
    f = Function(n, Ts=Ts, function_handle=fcn_handle)
    N = 32
    f_low_pass = low_pass_filter(f, 0.1*w, N)
    assert np.allclose(f_low_pass.f, np.zeros_like(f_low_pass.f), atol=0.05), "Test case 2 failed"
    
def test_low_pass3():
    # Test case 3: remove noise from a cosine wave
    w = 2 * pi # Frequency of the cosine wave
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    fcn_handle = lambda t: np.cos(w*t) + 0.1*np.random.randn() # Cosine function with noise
    f = Function(n, Ts=Ts, function_handle=fcn_handle)
    N = 32
    f_low_pass = low_pass_filter(f, 3*w, N)
    # Compare to the original cosine wave without noise
    f_without_noise = Function(n, Ts=Ts, function_handle=lambda t: np.cos(w*t))
    assert np.allclose(f_low_pass.f[2*N:-2*N], f_without_noise.f[N:-N], atol=0.8), "Test case 3 failed"

if __name__ == '__main__':
    test_low_pass1()
    test_low_pass2()
    test_low_pass3()