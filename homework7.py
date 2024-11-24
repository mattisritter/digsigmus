import numpy as np
import unittest

# Globale Variable für komplexe Multiplikationen
complex_multiplications = 0

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

# Aufgabe 1: Optimierte Berechnung von B*f
def optimized_bf_multiplication(f):
    """
    Berechnet das Matrix-Vektor-Produkt B*f unter Verwendung der Matrixdefinition aus Aufgabe 1.
    """
    n = len(f)
    b = np.zeros(n, dtype=complex)
    for k in range(n):
        for l in range(n):
            b[k] += np.exp(2j * np.pi * k * l / n) * f[l]
    return b

# Aufgabe 4: Rekursive FFT
def fft_recursive(f, w):
    """
    Rekursive Implementierung der FFT.
    """
    global complex_multiplications
    n = len(f)
    if n <= 1:
        return f
    even_fft = fft_recursive(f[::2], w[::2])
    odd_fft = fft_recursive(f[1::2], w[::2])
    combined = np.zeros(n, dtype=complex)
    for k in range(n // 2):
        complex_multiplications += 1
        t = w[k] * odd_fft[k]
        combined[k] = even_fft[k] + t
        combined[k + n // 2] = even_fft[k] - t
    return combined

# Aufgabe 5: Iterative FFT
def fft_iterative(f):
    """
    Iterative Implementierung der FFT.
    """
    global complex_multiplications
    n = len(f)
    levels = int(np.log2(n))
    f = np.asarray(f, dtype=complex)
    f = f[np.argsort([int(bin(i)[2:].zfill(levels)[::-1], 2) for i in range(n)])]
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    step = 2
    while step <= n:
        for k in range(0, n, step):
            for j in range(step // 2):
                complex_multiplications += 1
                t = w[n // step * j] * f[k + j + step // 2]
                f[k + j + step // 2] = f[k + j] - t
                f[k + j] += t
        step *= 2
    return f

# Aufgabe 6: Inverse FFT
def ifft(f):
    """
    Implementierung der inversen FFT.
    """
    n = len(f)
    f_conjugate = np.conjugate(f)
    fft_result = fft_iterative(f_conjugate)
    return np.conjugate(fft_result) / n

# Funktionen zur Demonstration der Ausgaben
def demonstrate_task1(n):
    print("\nAufgabe 1: Berechnung von B*f")
    B = compute_matrix_B(n)
    f = np.random.rand(n)
    result = optimized_bf_multiplication(f)
    print(f"Matrix B (gerundet auf 2 Dezimalstellen):\n{np.round(B, 2)}")
    print(f"Vektor f:\n{f}")
    print(f"Ergebnis B*f:\n{result}\n")

def demonstrate_task4(n):
    print("\nAufgabe 4: Rekursive FFT")
    f = np.random.rand(n)
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    fft_result = fft_recursive(f, w)
    print(f"Eingabevektor f:\n{f}")
    print(f"Ergebnis der rekursiven FFT:\n{fft_result}")
    print(f"Referenz (NumPy FFT):\n{np.fft.fft(f)}\n")

def demonstrate_task5(n):
    print("\nAufgabe 5: Iterative FFT")
    f = np.random.rand(n)
    fft_result = fft_iterative(f)
    print(f"Eingabevektor f:\n{f}")
    print(f"Ergebnis der iterativen FFT:\n{fft_result}")
    print(f"Referenz (NumPy FFT):\n{np.fft.fft(f)}\n")

def demonstrate_task6(n):
    print("\nAufgabe 6: Inverse FFT")
    f = np.random.rand(n)
    fft_result = np.fft.fft(f)
    ifft_result = ifft(fft_result)
    print(f"Eingabevektor f:\n{f}")
    print(f"FFT von f:\n{fft_result}")
    print(f"Rekonstruierter Vektor (iFFT):\n{ifft_result}")
    print(f"Rekonstruktionsfehler (sollte nahe 0 sein):\n{np.abs(f - ifft_result)}\n")

# Hauptprogramm
if __name__ == "__main__":
    print("Demonstration der Aufgaben für n = 4")
    n = 4  # Beispielgröße, für die Aufgaben sinnvoll
    demonstrate_task1(n)
    demonstrate_task4(n)
    demonstrate_task5(n)
    demonstrate_task6(n)
