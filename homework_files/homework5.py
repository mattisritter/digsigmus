import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 1: Matrix B erstellen
def compute_matrix_B(n):
    """
    Erstellt eine Matrix B ∈ C^{n x n}, bei der b_kl = e^{2πjkl/n}.
    """
    B = np.zeros((n, n), dtype=complex)
    for k in range(n):
        for l in range(n):
            B[k, l] = np.exp(2j * np.pi * k * l / n)
    return B

# Aufgabe 2: Signal f berechnen
def compute_samples(B, z):
    """
    Berechnet die Werte des Signals f durch Matrix-Vektor-Multiplikation.
    """
    f = np.dot(B, z)
    return f

# Aufgabe 3: DFT und IDFT implementieren
def DFT(f):
    """
    Diskrete Fourier-Transformation (DFT).
    """
    n = len(f)
    B = compute_matrix_B(n)
    return np.dot(B, f)

def IDFT(z):
    """
    Inverse Diskrete Fourier-Transformation (IDFT).
    """
    n = len(z)
    B = compute_matrix_B(n)
    return np.dot(np.conjugate(B.T), z) / n

# Aufgabe 4: Funktion f(t) abtasten
def sample_function():
    """
    Definiert und sampelt die Funktion f(t) = 3 + cos(t + 1) + 2cos(3t + 2) - 5cos(4t - 1).
    """
    n = 16
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    f = 3 + np.cos(t + 1) + 2 * np.cos(3 * t + 2) - 5 * np.cos(4 * t - 1)
    return f

# Ausgabe mit Runden und Vergleichen
def rounded_matrix(matrix, decimals=2):
    """
    Rundet eine Matrix auf die angegebene Anzahl von Nachkommastellen.
    """
    return np.round(matrix, decimals)

# Hauptprogramm mit Tests
if __name__ == "__main__":
    n = 16  # Beispielgröße

    # Aufgabe 1: Matrix B
    B = compute_matrix_B(n)
    B_rounded = rounded_matrix(B, decimals=2)
    print("Matrix B (gerundet):")
    print(B_rounded)

    # Test: Orthogonalität von B prüfen
    identity_check = np.dot(B, np.conjugate(B.T)) / n
    identity_check_rounded = rounded_matrix(identity_check, decimals=2)
    print("\nOrthogonalität von B geprüft (sollte Einheitsmatrix sein, gerundet):")
    print(identity_check_rounded)

    # Aufgabe 2: Signalberechnung
    z = np.zeros(n, dtype=complex)
    z[0] = 3  # Konstante Komponente
    f = compute_samples(B, z)
    f_rounded = rounded_matrix(f.real, decimals=2)
    print("\nSignal f für konstantes z (sollte konstant und real sein, gerundet):")
    print(f_rounded)

    # Aufgabe 3: Test von DFT und IDFT
    f_original = np.array([1, 2, 3, 4] + [0] * (n - 4), dtype=float)
    z_dft = DFT(f_original)
    f_reconstructed = IDFT(z_dft)
    f_original_rounded = rounded_matrix(f_original, decimals=2)
    f_reconstructed_rounded = rounded_matrix(f_reconstructed.real, decimals=2)
    print("\nOriginal f (gerundet):", f_original_rounded)
    print("Reconstructed f (gerundet):", f_reconstructed_rounded)
    print("Rekonstruktionsfehler (gerundet, sollte nahe 0 sein):")
    print(rounded_matrix(np.abs(f_original - f_reconstructed.real), decimals=2))

    # Aufgabe 4: Abtasten und Fourier-Koeffizienten
    f_sampled = sample_function()
    z_fourier = DFT(f_sampled)
    f_reconstructed = IDFT(z_fourier)
    f_sampled_rounded = rounded_matrix(f_sampled, decimals=2)
    z_fourier_rounded = rounded_matrix(z_fourier, decimals=2)
    error_sampled_rounded = rounded_matrix(np.abs(f_sampled - f_reconstructed.real), decimals=2)
    print("\nAbgetastete Werte f(t) (gerundet):")
    print(f_sampled_rounded)
    print("Fourier-Koeffizienten z (gerundet):")
    print(z_fourier_rounded)
    print("Fehler bei abgetastetem Signal (gerundet, sollte nahe 0 sein):")
    print(error_sampled_rounded)

    # Signal und Rekonstruktion plotten
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    plt.figure(figsize=(10, 6))
    plt.plot(t, f_sampled, label="Original Signal f(t)")
    plt.plot(t, f_reconstructed.real, linestyle="--", label="Reconstructed Signal")
    plt.xlabel("t")
    plt.ylabel("Signalwert")
    plt.legend()
    plt.title("Signal und Rekonstruktion")
    plt.grid()
    plt.show()
