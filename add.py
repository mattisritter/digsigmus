import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor

def add(f: Function, g: Function) -> Function:
    """Add two functions."""
    # Check for compatible sampling points
    if len(f.f) != len(g.f) or not np.allclose(f.t, g.t):
        raise ValueError("Functions must have the same sampling points to be added.")
    # add elements of f and g with the same time points
    h = [f.f[i] + g.f[i] for i in range(f.len)]
    return Function(f.n, Ts=f.Ts, f=h)

if __name__ == "__main__":
    # Define first function
    f1 = Function([0, 1, 2, 3], Ts=1, f=[1, 2, 3, 4])
    # Define second function
    f2 = Function([0, 1, 2, 3], Ts=1, f=[4, 3, 2, 1])
    # Add the functions
    f3 = add(f1, f2)
    print(f3.f)