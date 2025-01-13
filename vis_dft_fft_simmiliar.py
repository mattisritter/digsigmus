import numpy as np
import matplotlib.pyplot as plt
from dft import dft, idft
from fft import fft_recursive, fft_iterative

# Helper function to generate a sinusoidal signal
def generate_signal(frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

def ifft_from_fft(fft_result):
    """
    Perform the Inverse FFT (IFFT) using the implemented FFT results.
    """
    n = len(fft_result)
    return fft_recursive(fft_result, inverse=True)

def test_dft_fft_commonalities():
    # Signal parameters
    frequency = 5  # Frequency of the sinusoidal signal (Hz)
    duration = 1   # Duration of the signal (seconds)
    sampling_rate = 64  # Sampling rate (Hz)

    # Generate a sinusoidal signal
    t, signal = generate_signal(frequency, duration, sampling_rate)

    # Apply DFT
    dft_result = dft(signal)
    idft_result = idft(dft_result)

    # Apply Recursive FFT
    fft_recursive_result = fft_recursive(signal)
    ifft_recursive_result = ifft_from_fft(fft_recursive_result)

    # Apply Iterative FFT
    fft_iterative_result = fft_iterative(signal)
    ifft_iterative_result = ifft_from_fft(fft_iterative_result)

    # Plot frequency domain results
    freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    plt.figure(figsize=(12, 18))
    plt.suptitle("Frequency Domain Comparison")

    # DFT Plot
    plt.subplot(3, 1, 1)
    plt.stem(freqs[:len(freqs)//2], np.abs(dft_result)[:len(freqs)//2], linefmt='b-', markerfmt='bo', basefmt='r-', label='DFT', use_line_collection=True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Recursive FFT Plot
    plt.subplot(3, 1, 2)
    plt.stem(freqs[:len(freqs)//2], np.abs(fft_recursive_result)[:len(freqs)//2], linefmt='g-', markerfmt='go', basefmt='r-', label='FFT Recursive', use_line_collection=True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Iterative FFT Plot
    plt.subplot(3, 1, 3)
    plt.stem(freqs[:len(freqs)//2], np.abs(fft_iterative_result)[:len(freqs)//2], linefmt='m-', markerfmt='mo', basefmt='r-', label='FFT Iterative', use_line_collection=True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout(h_pad=6.0, rect=[0, 0.05, 1, 0.96])
    plt.show()

    # Plot time domain results
    plt.figure(figsize=(12, 18))
    plt.suptitle("Reconstructed Time Domain Comparison")

    # Original Signal Plot
    plt.subplot(4, 1, 1)
    plt.plot(t, signal, label="Original Signal", linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='lower left')
    plt.grid()

    # DFT Reconstructed Plot
    plt.subplot(4, 1, 2)
    plt.plot(t, idft_result.real, label="DFT", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='lower left')
    plt.grid()

    # Recursive FFT Reconstructed Plot
    plt.subplot(4, 1, 3)
    plt.plot(t, ifft_recursive_result.real, label="FFT Recursive", linestyle='-.')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='lower left')
    plt.grid()

    # Iterative FFT Reconstructed Plot
    plt.subplot(4, 1, 4)
    plt.plot(t, ifft_iterative_result.real, label="FFT Iterative", linestyle=':')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='lower left')
    plt.grid()

    plt.tight_layout(h_pad=6.0, rect=[0, 0.05, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    test_dft_fft_commonalities()
