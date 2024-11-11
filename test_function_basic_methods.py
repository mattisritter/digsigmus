from math import pi, ceil
import numpy as np
import matplotlib.pyplot as plt
from function import Function

def test_add():
    n = range(0, 4)
    Ts = 1
    f = [1, 2, 3, 4]
    g = [1, 2, 1, 2]
    fn = Function(n, Ts, f=f)
    gn = Function(n, Ts, f=g)
    result = fn + gn
    #assert result.f == [2, 4, 4, 6], "Test case 1 failed"

test_add()