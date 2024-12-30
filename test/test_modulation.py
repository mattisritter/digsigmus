# ======================================
# Digital Signal Processing
# Jakob Kurz (210262)
# Mattis Tom Ritter (210265)
# Heilbronn University of Applied Sciences
# (C) Jakob Kurz, Mattis Tom Ritter 2024
# ======================================
from math import pi
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../module/'))
from function import Function
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate

def test_modulation():
    f = Function([0, 1, 2, 3], Ts=1, f=[0, 0, 0, 1])
    w_mod = pi
    f_mod = modulate(f, w_mod)
    assert f_mod.f == [0, 0, 0, -1], "Test case 1 failed"

def test_demodulation():
    f = Function([0, 1, 2, 3], Ts=1, f=[0, 0, 0, 1])
    w_mod = pi
    f_demod = demodulate(f, w_mod)
    assert f_demod.f == [0, 0, 0, -2], "Test case 2 failed"

def test_quadrature_modulation():
    f = Function([0, 1, 2, 3], Ts=1, f=[0, 0, 0, 1])
    g = Function([0, 1, 2, 3], Ts=1, f=[1, 0, 0, 0])
    w_mod = pi
    f_mod, g_mod = quadrature_modulate(f, g, w_mod)
    assert f_mod.f == [0, 0, 0, -1] and g_mod.f == [0, 0, 0, 0], "Test case 3 failed"

def test_quadrature_demodulation():
    h = Function([0, 1, 2, 3], Ts=1, f=[0, 0, 0, 1])
    w_mod = pi
    f_demod, g_demod = quadrature_demodulate(h, w_mod)
    assert np.allclose(f_demod.f, [0, 0, 0, -2]) and np.allclose(g_demod.f, [0, 0, 0, 0]), "Test case 4 failed"