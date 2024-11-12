from function import Function
import numpy as np

def convolution_time_domain(f: Function, g: Function) -> Function:
    """
    Perform the discrete convolution of two functions in time domain.
    Parameters:
        f: Function
            First function
        g: Function
            Second function
    Return:
        Function
            Convolution of the two functions
    """
    # Determine the length of the output convolution
    len_h = f.len + g.len - 1
    # Initialize the output convolution
    h = [0] * len_h
    # Perform the convolution
    for i in range(f.len):
        for j in range(g.len):
            h[i + j] += f.f[i] * g.f[j]
    # Delete the zeros at the end
    while h[-1] == 0:
        h.pop()
    return Function(range(len(h)), Ts=f.Ts, f=h)

if __name__ == "__main__":
    # Define the functions
    n = [0, 1, 2, 3]
    Ts = 1
    f1 = [0, 0, 0, 1]
    f2 = [1, 2, 3, 4]
    function1 = Function(n, Ts, f=f1)
    function2 = Function(n, Ts, f=f2)
    
    # Perform the convolution
    convolution = convolution_time_domain(function1, function2)
    
    # Print the result
    print(convolution.f)

