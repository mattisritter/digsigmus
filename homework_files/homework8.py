import numpy as np
import time

def non_cyclic_convolution(x, h):
    """Berechnet die nicht-zyklische Faltung von x und h."""
    return np.convolve(x, h)

def cyclic_convolution_time_domain(x, h):
    """Berechnet die zyklische Faltung im Zeitbereich."""
    n = len(x)
    h = np.pad(h, (0, n - len(h)), 'constant')
    result = np.zeros(n)
    for i in range(n):
        for j in range(len(h)):
            result[i] += x[j] * h[(i - j) % n]
    return result

def cyclic_convolution_fft(x, h):
    """Berechnet die zyklische Faltung mithilfe der FFT."""
    n = len(x)
    h = np.pad(h, (0, n - len(h)), 'constant')
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    Y = X * H
    return np.fft.ifft(Y).real

# Testvektoren
x = np.array([1, 2, 3, 4, 5], dtype=int)
h = np.array([1, 2, 1], dtype=int)

# Nicht-zyklische Faltung
result_non_cyclic = non_cyclic_convolution(x, h)
print("Nicht-zyklische Faltung:", np.round(result_non_cyclic))

# Zyklische Faltung im Zeitbereich
start_time = time.time()
result_cyclic_time = cyclic_convolution_time_domain(x, h)
time_time_domain = time.time() - start_time
print("Zyklische Faltung (Zeitbereich):", np.round(result_cyclic_time))

# Zyklische Faltung mit FFT
start_time = time.time()
result_cyclic_fft = cyclic_convolution_fft(x, h)
time_fft = time.time() - start_time
print("Zyklische Faltung (FFT):", np.round(result_cyclic_fft))

# Laufzeitvergleich
print(f"Laufzeit Zeitbereich: {time_time_domain:.6f} Sekunden")
print(f"Laufzeit FFT: {time_fft:.6f} Sekunden")

# Ergebnisse vergleichen
print("Ergebnisse identisch (nach Runden):", 
      np.allclose(np.round(result_cyclic_time), np.round(result_cyclic_fft)))
