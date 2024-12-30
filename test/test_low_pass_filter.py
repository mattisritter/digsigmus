# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
import numpy as np
from math import pi
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from function import Function
from low_pass_filter import low_pass_filter


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

def test_fast_convolution():
    # Test case 4: compare the result of the low pass filter with fast and regular convolution
    w = 2 * pi # Frequency of the cosine wave
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    fcn_handle = lambda t: np.cos(w*t) # Cosine function
    f = Function(n, Ts=Ts, function_handle=fcn_handle)
    N = 32
    f_low_pass_fast = low_pass_filter(f, 4*w, N)
    f_low_pass = low_pass_filter(f, 4*w, N, use_fast_convolution=False)
    assert np.allclose(f_low_pass_fast.f[2*N:-2*N], f_low_pass.f[2*N:-2*N], atol=1e-6), "Test case 4 failed"
