def discrete_convolution(f, g):
    # Determine the lengths of the input functions
    len_f = len(f)
    len_g = len(g)
    
    # Determine the length of the output convolution
    len_h = len_f + len_g - 1
    
    # Initialize the output convolution
    h = [0] * len_h
    
    # Perform the convolution
    for i in range(len_f):
        for j in range(len_g):
            h[i + j] += f[i] * g[j]
    
    return h