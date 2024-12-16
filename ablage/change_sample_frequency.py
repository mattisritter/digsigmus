from function import Function
import numpy as np

def change_sampling_frequency(f: Function, new_fs: float) -> Function:
    """
    Change the sampling frequency of a function using interpolation.
    Parameters:
        f: Function
            Input function.
        new_fs: float
            New sampling frequency [Hz].
    Return:
        Function
            Function resampled to the new sampling frequency.
    """
    old_time = f.t
    new_Ts = 1 / new_fs
    new_time = np.arange(old_time[0], old_time[-1] + new_Ts, new_Ts)

    # Interpolate the function values to the new time points
    new_values = np.interp(new_time, old_time, f.f)

    return Function(range(len(new_time)), Ts=new_Ts, f=new_values)

if __name__ == "__main__":
    # Define the original function
    n = range(10)
    Ts = 0.1  # Original sampling time
    f_values = [np.sin(2 * np.pi * 1 * t) for t in np.arange(0, 1, Ts)]
    original_function = Function(n, Ts=Ts, f=f_values)

    # Change sampling frequency
    new_fs = 20  # New sampling frequency
    resampled_function = change_sampling_frequency(original_function, new_fs)

    # Print the results
    print("Original function (time domain):", original_function.f)
    print("Original sampling frequency:", 1 / original_function.Ts, "Hz")
    print("Resampled function (time domain):", resampled_function.f)
    print("New sampling frequency:", 1 / resampled_function.Ts, "Hz")
