import matplotlib.pyplot as plt
import numpy as np
from math import pi
from function import Function
import soundfile as sf
import playsound as ps
from modulation import modulate, demodulate
from low_pass_filter import low_pass_filter
from convolution import convolution_time_domain
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
    f = _convert_to_mono(f)
    return Function(range(len(f)), Ts=1/fs, f=f)

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
        return f[:, 0].reshape(-1, 1)
    else:
        return f

if __name__ == "__main__":
    f1, f2 = get_soundfiles("soundfiles/Jodler.wav", "soundfiles/Violine.wav")
    # Plot the sound files
    if False:
        plt.figure(1)
        plt.plot(f1.n, f1.f, label="Jodler")
        plt.plot(f2.n, f2.f, label="Elefant")
        plt.title("Sound files")
        plt.legend()
        plt.show()
    # Low pass filter the sound files
    ws = f1.ws
    wc = 20000
    N = 128
    f1_low_pass = low_pass_filter(f1, wc, N, hamming_window=True)
    f2_low_pass = low_pass_filter(f2, wc, N, hamming_window=True)
    # Plot the low pass filtered sound files
    if False:
        plt.figure(2)
        plt.plot(f1_low_pass.n, f1_low_pass.f, label="Jodler Low Pass")
        plt.plot(f2_low_pass.n, f2_low_pass.f, label="Elefant Low Pass")
        plt.title("Low pass filtered sound files")
        plt.legend()
        plt.show()
    w_mod1 = 20000
    w_mod2 = 60000
    f1_mod = modulate(f1_low_pass, w_mod1)
    f2_mod = modulate(f2_low_pass, w_mod2)
    # Add the modulated sound files
    f_sum = add(f1_mod, f2_mod)
    # Demodulate the sum
    f1_demod = demodulate(f_sum, w_mod1)
    f2_demod = demodulate(f_sum, w_mod2)
    # Low pass filter the demodulated sound files
    f1_demod_low_pass = low_pass_filter(f1_demod, wc, N, hamming_window=True)
    f2_demod_low_pass = low_pass_filter(f2_demod, wc, N, hamming_window=True)

    # write to sound files
    #sf.write("soundfiles/Jodler_modulated.wav", f1_mod.f, int(1/f1_mod.Ts))
    #sf.write("soundfiles/Elefant_modulated.wav", f2_mod.f, int(1/f2_mod.Ts))
    #sf.write("soundfiles/Sound_sum.wav", f_sum.f, int(1/f_sum.Ts))
    sf.write("soundfiles/Jodler_reconstructed.wav", f1_demod_low_pass.f, int(1/f1_low_pass.Ts))
    sf.write("soundfiles/Violine_reconstructed.wav", f2_demod_low_pass.f, int(1/f2_low_pass.Ts))
    # play the sound files
    if True:
        play_soundfile("soundfiles/Jodler.wav")
        play_soundfile("soundfiles/Jodler_reconstructed.wav")
        #play_soundfile("soundfiles/Jodler_modulated.wav")
        play_soundfile("soundfiles/Violine.wav")
        play_soundfile("soundfiles/Violine_reconstructed.wav")
        #play_soundfile("soundfiles/Elefant_modulated.wav")
        #play_soundfile("soundfiles/Sound_sum.wav")