import numpy as np
from convolution import discrete_convolution

def test_discrete_convolution_case1():
    # Test case 1: Shift with dirac pulse
    f = [1, 1, 1]
    g = [0, 0, 0, 0, 0, 0, 1]
    expected = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    result = discrete_convolution(f, g)
    assert np.array_equal(result, expected), "Test case 1 failed"

def test_discrete_convolution_case2():
    # Test case 2: Test with floating point numbers
    f = [0.1, 0.2, 0.3]
    g = [0.4, 0.5, 0.6]
    expected = np.convolve(f, g)
    result = discrete_convolution(f, g)
    assert np.array_equal(result, expected), "Test case 2 failed"

def test_discrete_convolution_case3():
    # Test case 3: Test with cosine functions
    f = [np.cos(i) for i in range(10)]
    g = [np.cos(0.5 * i) for i in range(20)]
    expected = np.convolve(f, g)
    result = discrete_convolution(f, g)
    assert np.allclose(result, expected), "Test case 3 failed"
