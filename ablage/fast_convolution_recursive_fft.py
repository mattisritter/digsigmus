from function import Function
import numpy as np
from fft_recursive import fft_recursive

def fast_convolution_fft(f: Function, g: Function) -> Function:
    """
    Perform convolution of two functions using the Fast Fourier Transform (FFT).
    Parameters:
        f: Function
            First input function in the time domain.
        g: Function
            Second input function in the time domain.
    Return:
        Function
            Convolved function in the time domain.
    """
    # Ensure the length is the next power of 2 for efficient FFT
    len_result = f.len + g.len - 1
    next_power_of_2 = 2**int(np.ceil(np.log2(len_result)))

    # Zero-pad both functions to the same length
    f_padded = np.pad(f.f, (0, next_power_of_2 - f.len))
    g_padded = np.pad(g.f, (0, next_power_of_2 - g.len))

    # Perform FFT on both padded functions
    F_fft = fft_recursive(Function(range(next_power_of_2), Ts=f.Ts, f=f_padded)).f
    G_fft = fft_recursive(Function(range(next_power_of_2), Ts=g.Ts, f=g_padded)).f

    # Multiply in the frequency domain
    H_fft = F_fft * G_fft

    # Perform inverse FFT to get the result in the time domain
    h_time = np.fft.ifft(H_fft).real  # Use the real part only

    # Truncate to the original convolution length
    h_time = h_time[:len_result]

    return Function(range(len_result), Ts=f.Ts, f=h_time)

if __name__ == "__main__":
    # Define the functions
    n = [0, 1, 2, 3]
    Ts = 1
    f1 = [0, 0, 0, 1]
    f2 = [1, 2, 3, 4]
    function1 = Function(n, Ts, f=f1)
    function2 = Function(n, Ts, f=f2)

    # Perform fast convolution
    convolution = fast_convolution_fft(function1, function2)

    # Print the result
    print(convolution.f)
