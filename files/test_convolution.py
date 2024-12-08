import numpy as np
from function import Function
from convolution import convolution_time_domain

def test_discrete_convolution_case1():
    # Test case 1: Shift with dirac pulse
    f = Function(range(4), Ts=1, f=[0, 0, 0, 1])
    g = Function(range(4), Ts=1, f=[1, 2, 3, 4])
    expected = [0, 0, 0, 1, 2, 3, 4]
    result = convolution_time_domain(f, g)
    assert np.array_equal(result.f, expected), "Test case 1 failed"

def test_discrete_convolution_case2():
    # Test case 2: Test communitve property
    f = Function(range(4), Ts=1, f=[1, 2, 3, 4])
    g = Function(range(4), Ts=1, f=[0, 0, 0, 1])
    expected = [0, 0, 0, 1, 2, 3, 4]
    result1 = convolution_time_domain(f, g)
    result2 = convolution_time_domain(g, f)
    assert np.array_equal(result1.f, expected) and np.array_equal(result2.f, expected), "Test case 2 failed"

def test_discrete_convolution_case3():
    # Test case 3: Test with more complex functions
    f = Function(range(10), Ts=1, f=[np.cos(i) for i in range(10)])
    g = Function(range(20), Ts=1, f=[np.cos(0.5 * i + 1) + i*i for i in range(20)])
    expected = np.convolve(f.f, g.f)
    result = convolution_time_domain(f, g)
    assert np.allclose(result.f, expected), "Test case 3 failed"
