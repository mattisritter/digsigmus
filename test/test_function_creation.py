# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from function import Function

def test_direct_creation():
    n = [0, 1, 2, 3]
    Ts = 1
    f = [1, 2, 3, 4]
    function = Function(n, Ts, f=f)
    assert function.n == n, "Test case 1 failed"
    assert function.Ts == Ts, "Test case 1 failed"
    assert function.f == f, "Test case 1 failed"

def test_function_handle():
    n = [0, 1, 2, 3]
    Ts = 1
    function_handle = lambda t: t**2
    function = Function(n, Ts, function_handle=function_handle)
    assert function.n == n, "Test case 2 failed"
    assert function.Ts == Ts, "Test case 2 failed"
    assert function.f == [0, 1, 4, 9], "Test case 2 failed"

def test_time_values():
    n = [0, 1, 2, 3]
    Ts = 2
    f = [1, 2, 3, 4]
    function = Function(n, Ts, f=f)
    assert function.t == [0, 2, 4, 6], "Test case 3 failed"

def test_frequency_values():
    n = [0, 1, 2, 3]
    ws = 4*pi
    f = [1, 2, 3, 4]
    function = Function(n, ws=ws, f=f)
    assert function.t == [0, 0.5, 1, 1.5], "Test case 4 failed"