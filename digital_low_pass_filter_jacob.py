import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import soundfile as sf

def create_lowpass_filter(omega_c, omega_s, N):
    """Berechnet die Koeffizienten des Tiefpassfilters"""
    hat_omega_c = 2 * omega_c / omega_s  # Normalisierte Grenzfrequenz
    g = np.zeros(2 * N + 1)
    for k in range(2 * N + 1):
        if k == N:
            g[k] = hat_omega_c  # Zentraler Wert
        else:
            g[k] = hat_omega_c * np.sinc((k - N) * hat_omega_c)
    return g

def lowpass_filter(signal, omega_c, omega_s, N):
    """Führt die Tiefpassfilterung durch"""
    g = create_lowpass_filter(omega_c, omega_s, N)
    filtered_signal = convolve(signal, g, mode='same')
    return filtered_signal

# Test 1: Harmonische Schwingung mit Grenzfrequenz über und unter der Signalfrequenz
def test_harmonic_oscillation():
    fs = 1000  # Abtastfrequenz
    f_signal = 50  # Signalfrequenz
    omega_c_low = 30  # Grenzfrequenz unter der Signalfrequenz
    omega_c_high = 70  # Grenzfrequenz über der Signalfrequenz
    N = 50  # Filterverzögerung

    # Erzeuge harmonische Schwingung
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * f_signal * t)

    # Filterung mit niedriger Grenzfrequenz
    filtered_low = lowpass_filter(signal, omega_c_low, fs, N)

    # Filterung mit hoher Grenzfrequenz
    filtered_high = lowpass_filter(signal, omega_c_high, fs, N)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Originalsignal")
    plt.plot(t, filtered_low, label="Gefiltertes Signal (omega_c < f_signal)", linestyle='--')
    plt.plot(t, filtered_high, label="Gefiltertes Signal (omega_c > f_signal)", linestyle='--')
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 1: Tiefpassfilterung einer harmonischen Schwingung")
    plt.show()

# Test 2: Zwei harmonische Schwingungen mit unterschiedlichen Frequenzen und Grenzfrequenz dazwischen
def test_two_frequencies():
    fs = 1000  # Abtastfrequenz
    f1, f2 = 30, 80  # Zwei unterschiedliche Frequenzen
    omega_c = 50  # Grenzfrequenz zwischen f1 und f2
    N = 50  # Filterverzögerung

    # Erzeuge Signal mit zwei Frequenzen
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    # Filterung
    filtered_signal = lowpass_filter(signal, omega_c, fs, N)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Originalsignal (f1 + f2)")
    plt.plot(t, filtered_signal, label="Gefiltertes Signal (f1 bleibt, f2 verschwindet)", linestyle='--')
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 2: Tiefpassfilterung eines Signals mit zwei Frequenzen")
    plt.show()

# Test 3: Endliches Signal und Beobachtung von Transienten
def test_finite_length_signal():
    fs = 1000  # Abtastfrequenz
    f_signal = 50  # Signalfrequenz
    omega_c = 40  # Grenzfrequenz unterhalb der Signalfrequenz
    N = 50  # Filterverzögerung

    # Erzeuge ein endliches Signal (harmonische Schwingung, aber nur für einen Teil der Zeit)
    t = np.arange(0, 0.5, 1.0 / fs)  # Endliches Zeitintervall
    signal = np.sin(2 * np.pi * f_signal * t)

    # Filterung
    filtered_signal = lowpass_filter(signal, omega_c, fs, N)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Originalsignal (endliche Länge)")
    plt.plot(t, filtered_signal, label="Gefiltertes Signal mit Transienten", linestyle='--')
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 3: Tiefpassfilterung eines endlichen Signals (Transienten sichtbar)")
    plt.show()

# Test 4: Musiksignal filtern und speichern
def test_music_filtering():
    # Musikdatei laden
    file_path = "Jodler.wav"  # Pfad zur Musikdatei
    signal, fs = sf.read(file_path)

    # Falls die Musik stereo ist, konvertieren wir sie in mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Filterparameter
    omega_c = 2000  # Grenzfrequenz des Tiefpassfilters in Hz
    N = 100  # Filterverzögerung

    # Musiksignal filtern
    filtered_signal = lowpass_filter(signal, omega_c, fs, N)

    # Musikdateien speichern
    sf.write("original_music.wav", signal, fs)
    sf.write("filtered_music.wav", filtered_signal, fs)
    print("Original- und gefiltertes Musiksignal wurden als 'original_music.wav' und 'filtered_music.wav' gespeichert.")

    # Plot der Wellenformen
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:10000], label="Originalsignal", alpha=0.7)
    plt.plot(filtered_signal[:10000], label="Gefiltertes Signal", linestyle='--', alpha=0.7)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Test 4: Tiefpassfilterung eines Musiksignals")
    plt.show()

# Funktion zum Ausführen aller Tests
def run_all_tests():
    print("Starte Test 1: Harmonische Schwingung mit Grenzfrequenz über und unter der Signalfrequenz")
    test_harmonic_oscillation()
    
    print("Starte Test 2: Zwei harmonische Schwingungen mit unterschiedlichen Frequenzen")
    test_two_frequencies()
    
    print("Starte Test 3: Endliches Signal und Beobachtung von Transienten")
    test_finite_length_signal()
    
    print("Starte Test 4: Musiksignal filtern und speichern")
    test_music_filtering()

# Alle Tests ausführen
run_all_tests()
