from function import Function
import numpy as np

def discrete_convolution(f: Function, g: Function) -> Function:
    # Determine the lengths of the input functions
    len_f = len(f.f)
    len_g = len(g.f)
    
    # Determine the length of the output convolution
    len_h = len_f + len_g - 1
    
    # Initialize the output convolution
    h = [0] * len_h
    
    # Perform the convolution
    for i in range(len_f):
        for j in range(len_g):
            h[i + j] += f.f[i] * g.f[j]
    # delete the zeros at the end
    while h[-1] == 0:
        h.pop()
    return Function(range(len(h)), Ts=f.Ts, f=h)

