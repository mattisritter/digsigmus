import numpy as np
from function import Function
import matplotlib.pyplot as plt
from math import pi, ceil, floor

def add(f: Function, g: Function) -> Function:
    """
    Add two functions.
    Parameters:
        f: Function
            First summand
        g: Function
            Second summand
    Return:
        Function
            Sum of the two functions
    Raises:
        ValueError: The functions must have the same sampling time.
    """
    # Check for same sampling time
    if f.Ts != g.Ts:
        raise ValueError("The functions must have the same sampling time.")
    # Fill with zeros if the functions have different lengths
    len_h = f.len
    if f.len > g.len:
        g.f = np.append(g.f, [0]*(f.len-g.len))
        len_h = f.len
    elif g.len > f.len:
        f.f = np.append(f.f, [0]*(g.len-f.len))
        len_h = g.len
    # Add the functions
    h = [f.f[i] + g.f[i] for i in range(len_h)]
    return Function(range(len_h), Ts=f.Ts, f=h)

if __name__ == "__main__":
    # Define first function
    f1 = Function([0, 1, 2, 3, 4], Ts=1, f=[1, 2, 3, 4, 1])
    # Define second function
    f2 = Function([0, 1, 2, 3], Ts=1, f=[4, 3, 2, 1])
    # Add the functions
    f3 = add(f1, f2)
    print(f3.f)