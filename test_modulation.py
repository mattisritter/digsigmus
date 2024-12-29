from function import Function
from modulation import modulate, demodulate, quadrature_modulate, quadrature_demodulate
from math import pi
import numpy as np

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