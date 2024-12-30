# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from function import Function
from fast_convolution import fast_convolution

def test_discrete_convolution_case1():
    # Test case 1: Shift with dirac pulse
    f = Function(range(4), Ts=1, f=[0, 0, 0, 1])
    g = Function(range(4), Ts=1, f=[1, 2, 3, 4])
    expected = [0, 0, 0, 1, 2, 3, 4]
    result = fast_convolution(f, g, 4)
    assert np.allclose(result.f, expected), "Test case 1 failed"

def test_discrete_convolution_case2():
    # Test case 2: Test communitve property
    f = Function(range(4), Ts=1, f=[1, 2, 3, 4])
    g = Function(range(4), Ts=1, f=[0, 0, 0, 1])
    expected = [0, 0, 0, 1, 2, 3, 4]
    result1 = fast_convolution(f, g, 4)
    result2 = fast_convolution(g, f, 4)
    assert np.allclose(result1.f, expected) and np.allclose(result2.f, expected), "Test case 2 failed"

def test_discrete_convolution_case3():
    # Test case 3: Test with more complex functions
    f = Function(range(10), Ts=1, f=[np.cos(i) for i in range(10)])
    g = Function(range(20), Ts=1, f=[np.cos(0.5 * i + 1) + i*i for i in range(20)])
    expected = np.convolve(f.f, g.f)
    result = fast_convolution(f, g, 32)
    assert np.allclose(result.f, expected), "Test case 3 failed"
