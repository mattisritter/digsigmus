from fft import fft_iterative, _iterative_algorithm, _precompute_exponentials
from function import Function
import numpy as np

def fast_convolution(f: Function, g: Function, F) -> Function:
    """
    Perform fast convolution using FFT.
    Parameters:
        f: Function
            Input signal f0, f1, ...
        g: Function
            Impulse response g0, g1, ..., gG-1
        F: int
            Dimension of FFT (must be greater than or equal to length of impulse response)
    Return:
        Function
            Convolved output signal h0, h1, ...
    """
    # Lengths of signal and impulse response
    G = g.len
    assert F >= G, "F must be greater than or equal to the length of the impulse response."
    
    # Zero-pad the impulse response to F dimensions
    g_padded = np.zeros(F, dtype=np.complex128)
    g_padded[:G] = g.f

    # Precompute the exponential factors
    exp_factors = _precompute_exponentials(F)
    exp_factors_inv = _precompute_exponentials(F, inverse=True)
    
    # Compute FFT of the padded impulse response
    G_fft = _iterative_algorithm(g_padded, exp_factors)
    
    # Initialize variables
    h = []
    s = -G + 1  # Start index
    delta = F - G + 1  # Shift width
    
    while s < f.len:
        # Extract F-dimensional sample block from the signal
        f_padded = np.zeros(F, dtype=np.complex128)
        for i in range(F):
            if 0 <= s + i < f.len:
                f_padded[i] = f.f[s + i]
        
        # Perform cyclic convolution using FFT
        F_fft = _iterative_algorithm(f_padded, exp_factors)
        z = F * np.real(_iterative_algorithm(F_fft * G_fft, exp_factors_inv, inverse=True))
        
        # Append the correct sample values to the output
        h.extend(z[G - 1:F])
        
        # Shift the start index
        s += delta

    # Remove the zeros at the end
    while abs(h[-1]) < 1e-12:
        h.pop()
    
    return Function(range(len(h)), Ts=f.Ts, f=h)


if __name__ == "__main__":
    gk = [0, 0, 0, 1]
    fk = np.random.rand(100000)
    f = Function(range(len(fk)), Ts=1, f=fk)
    g = Function(range(len(gk)), Ts=1, f=gk)
    
    # Perform the convolution
    convolution = fast_convolution(f, g, 256)
    
    # compare to numpy implementation
    convolution_np = np.convolve(f.f, g.f)
    # Pad with zeros to match the length of the fast convolution
    #convolution_np = np.pad(convolution_np, (0, len(convolution) - len(convolution_np)))

    assert np.allclose(convolution.f, convolution_np), "The results of the fast convolution are not equal."