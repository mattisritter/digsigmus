from low_pass_filter import low_pass_filter
import numpy as np
from function import Function
import matplotlib.pyplot as plt
import time

def test_call(function_len, filter_len, use_fast_convolution):
    fk = np.random.rand(function_len)
    f = Function(range(len(fk)), ws=100, f=fk)
    t0 = time.time()
    f_low_pass = low_pass_filter(f, 10, filter_len, hamming_window=True, use_fast_convolution=use_fast_convolution)
    t1 = time.time()
    return t1 - t0

def average_time(function_len, filter_len, use_fast_convolution, num_tests):
    times = [test_call(function_len, filter_len, use_fast_convolution) for _ in range(num_tests)]
    return sum(times) / num_tests

# Test the function
function_len = int(1e5)
filter_len = 50
num_tests = 5
use_fast_convolution = True
time_fast = average_time(function_len, filter_len, use_fast_convolution, num_tests)
print(f"Average time using fast convolution: {time_fast:.4} s")
use_fast_convolution = False
time_slow = average_time(function_len, filter_len, use_fast_convolution, num_tests)
print(f"Average time without fast convolution: {time_slow:.4} s")
    