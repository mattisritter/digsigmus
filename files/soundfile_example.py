import matplotlib.pyplot as plt
import numpy as np
from math import pi
from function import Function
import soundfile as sf
import playsound as ps
from modulation import modulate, demodulate
from low_pass_filter import low_pass_filter
from convolution import discrete_convolution
from add import add

def play_soundfile(file):
    ps.playsound(file)

def get_soundfiles(file1: str, file2: str):
    """
    Read two sound files at once and return the corresponding Function objects.
    Parameters: 
        file1: str
            Path to the first sound file (.wav)
        file2: str
            Path to the second sound file (.wav)
    Return:
        Function, Function
            Function objects of the sound files
    """
    return get_soundfile(file1), get_soundfile(file2)

def get_soundfile(file: str) -> Function:
    """
    Read a sound file and return the corresponding Function object.
    Parameter:
        file: str
            Path to the sound file (.wav)
    Return:
        Function
            Function object of the sound file
    """
    # Read the sound file
    f, fs = sf.read(file)
    _convert_to_mono(f1)
    return Function(range(len(f1)), Ts=1/fs, f=f)

def _convert_to_mono(f: np.ndarray) -> np.ndarray:
    """
    Convert a stereo sound file to mono.
    parameter:
        f: numpy.ndarray
            Data of the sound file
    return:
        numpy.ndarray
            Data of mono sound file
    """
    if len(f.shape) == 2:
        return f[:, 0]
    else:
        return f

if __name__ == "__main__":
    f1, f2 = get_soundfiles("soundfiles/Jodler.wav", "soundfiles/Spock.wav")
    # Plot the sound files
    # plt.figure(1)
    # plt.plot(f1.n, f1.f, label="Jodler")
    # #plt.plot(f2.n, f2.f, label="Spock")
    # plt.title("Sound files")
    # plt.legend()
    # plt.show()
    # Low pass filter the sound files
    ws = f1.ws
    wc = 2000
    N = 20
    f1_low_pass = low_pass_filter(f1, wc, N, hamming_window=True)
    f2_low_pass = low_pass_filter(f2, wc, N, hamming_window=True)
    # Plot the low pass filtered sound files
    plt.figure(2)
    plt.plot(f1_low_pass.t, f1_low_pass.f, label="Jodler Low Pass")
    plt.plot(f2_low_pass.t, f2_low_pass.f, label="Spock Low Pass")
    plt.title("Low pass filtered sound files")
    plt.legend()
    plt.show()
    # write the low pass filtered sound files
    sf.write("soundfiles/Jodler_low_pass.wav", f1_low_pass.f, int(1/f1_low_pass.Ts))
    sf.write("soundfiles/Spock_low_pass.wav", f2_low_pass.f, int(1/f2_low_pass.Ts))
    # play the sound files
    # play_soundfile("soundfiles/Jodler.wav")
    # play_soundfile("soundfiles/Jodler_low_pass.wav")
    # play_soundfile("soundfiles/Spock.wav")
    # play_soundfile("soundfiles/Spock_low_pass.wav")