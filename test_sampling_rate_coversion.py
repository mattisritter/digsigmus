from sampling_rate_conversion import convert_sampling_rate
from function import Function
from math import pi
import numpy as np

def setup():
    fcn_handle = lambda t: np.sin(t)
    Ts = pi/5
    n = range(0, 61)
    f = Function(n, Ts=Ts, function_handle=fcn_handle)
    return f, Ts

def test_convert_sampling_rate():
    f, Ts = setup()
    f_new = convert_sampling_rate(f, N=15, Ts_new=Ts/2)
    # Check if the new sampling rate is correct
    assert np.isclose(f_new.Ts, Ts/2), "New sampling rate is incorrect"

def test_convert_sampling_rate_values():
    f, Ts = setup()
    f_new = convert_sampling_rate(f, N=15, Ts_new=0.1)
    # Check if the new sample points are correct
    sin_wave = [np.sin(t) for t in f_new.t]
    assert np.allclose(f_new.f, sin_wave, atol=0.07), "New sample points are incorrect"

def test_convert_sampling_rate_old_values():
    f, Ts = setup()
    f_new = convert_sampling_rate(f, N=15, Ts_new=Ts/2)
    # Check if the old sample points remain unchanged (up to numerical precision)
    assert np.allclose(f_new.f[::2], f.f, atol=1e-16), "Old sample points are changed"
