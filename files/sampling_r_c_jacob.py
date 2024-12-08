import numpy as np
import matplotlib.pyplot as plt

# Sinc-Interpolation-Funktion zur Umabtastung (ohne Hann-Fenster)
def sinc_interp(x, s, N):
    return np.sinc(x - s)  # Entfernen des Hann-Fensters, damit nur ein Wert zurückgegeben wird

def resample(f_n, omega_s, omega_s_prime, N):
    T_s = 1 / omega_s  # Ursprüngliches Abtastintervall
    T_s_prime = 1 / omega_s_prime  # Neues Abtastintervall

    # Länge des ursprünglichen Abtastwerte-Arrays
    L = len(f_n)
    # Zeitpunkte für die neue Samplingrate
    t_prime = np.arange(0, L * T_s, T_s_prime)

    # Erzeugung der neuen Abtastwerte f'_n durch die begrenzte Sinc-Summierung
    f_n_prime = []
    for t in t_prime:
        # Berechnung des neuen Abtastwertes durch Summieren über den begrenzten Sinc-Interpolationsbereich
        sample_sum = sum(f_n[n] * sinc_interp(t / T_s - n, 0, N) for n in range(max(0, int(t / T_s - N)), min(L, int(t / T_s + N))))
        f_n_prime.append(sample_sum)

    return np.array(f_n_prime[:len(t_prime)])  # Auf die gleiche Länge wie t_prime begrenzen

# Test der Funktion mit einem Kosinussignal und verschiedenen Werten für N
def test_resampling():
    # Sicherstellen, dass das Plot-Fenster gelöscht ist, bevor ein neuer Plot erstellt wird
    plt.figure()  # Startet ein neues Plot-Fenster
    
    # Ursprüngliche Signalparameter
    omega_hat = 2 * np.pi * 1  # Grenzfrequenz
    omega_s = 5 * omega_hat  # Ursprüngliche Samplingrate
    omega_s_prime = 3 * omega_hat  # Neue Samplingrate, unterschiedlich zu omega_s

    # Generierung der ursprünglichen Abtastwerte
    T_s = 1 / omega_s
    num_samples = 100
    t_values = np.arange(num_samples) * T_s
    f_n = np.cos(omega_hat * t_values)

    # Vergleichswerte mit der Kosinusfunktion bei neuer Samplingrate
    T_s_prime = 1 / omega_s_prime
    t_values_prime = np.arange(0, num_samples * T_s, T_s_prime)
    expected_values = np.cos(omega_hat * t_values_prime)

    # Plotten der erwarteten Werte
    plt.plot(t_values_prime, expected_values, label='Erwartete Kosinus-Werte', linestyle='--')

    # Verschiedene Werte für N ausprobieren: 2, 4, 6, ..., 20
    for N in range(2, 21, 2):
        f_n_prime = resample(f_n, omega_s, omega_s_prime, N)
        
        # Plotten der umgesampelten Kurve für jedes N
        plt.plot(t_values_prime[:len(f_n_prime)], f_n_prime, label=f'Umgesampeltes Signal (N={N})', marker='o', markersize=4)

    # Grafikeinstellungen
    plt.legend()
    plt.xlabel('Zeit')
    plt.ylabel('Amplitude')
    plt.title('Vergleich zwischen erwarteten und umgesampelten Werten für verschiedene N')
    plt.show()

# Test ausführen und die Grafik anzeigen
test_resampling()
