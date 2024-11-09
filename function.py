from math import pi, ceil
import numpy as np

class Function:
    def __init__(self, n, Ts=None, ws=None, f=None, function_handle=None):
        if Ts is None:
            Ts = 2*pi/ws
        elif ws is None:
            ws = 2*pi/Ts
        else:
            raise ValueError("Either Ts or ws must be provided.")
        self.ws = ws # Sampling frequency
        self.Ts = Ts # Sampling rate
        self.n = n # Sampling points
        self.t = [n_i * Ts for n_i in n] # Time points
        self.len = len(n) # Number of samples
        if f is not None:
            self.f = f
        elif function_handle is not None:
            self.f = self._evaluate_function(function_handle)
        else:
            raise ValueError("Either f or function_handle must be provided.")

    def _evaluate_function(self, function_handle):
        # Evaluate the function handle to get the sampling points and values
        # Replace this with your own implementation
        f = [function_handle(t_i) for t_i in self.t]
        return f
    
    def __add__(self, g):
        """Add another function or a scalar."""
        if isinstance(g, Function):
            # Check for compatible sampling points
            if len(self.f) != len(g.f) or not np.allclose(self.t, g.t):
                raise ValueError("Functions must have the same sampling points to be added.")
            return Function(self.n, Ts=self.Ts, f=self.f + g.f)
        elif isinstance(g, (int, float)):
            # Scalar addition
            return Function(self.n, Ts=self.Ts, f=self.f + g)
        else:
            raise TypeError("Addition with unsupported type.")

    def __sub__(self, g):
        """Subtract another function or a scalar."""
        if isinstance(g, Function):
            # Check for compatible sampling points
            if len(self.f) != len(g.f) or not np.allclose(self.t, g.t):
                raise ValueError("Functions must have the same sampling points to be subtracted.")
            return Function(self.n, Ts=self.Ts, f=self.f - g.f)
        elif isinstance(g, (int, float)):
            # Scalar subtraction
            return Function(self.n, Ts=self.Ts, f=self.f - g)
        else:
            raise TypeError("Subtraction with unsupported type.")

    def __mul__(self, g):
        """Multiply by a scalar or another function."""
        if isinstance(g, (int, float)):
            # Scalar multiplication
            return Function(self.n, Ts=self.Ts, f=self.f * g)
        elif isinstance(g, Function):
            # Pointwise multiplication
            if len(self.f) != len(g.f) or not np.allclose(self.t, g.t):
                raise ValueError("Functions must have the same sampling points to multiply.")
            return Function(self.n, Ts=self.Ts, f=self.f * g.f)
        else:
            raise TypeError("Multiplication with unsupported type.")
        
    def convolute(self, g):
        # Determine the length of the output convolution
        len_h = self.len + g.len - 1
        # Initialize the output convolution
        h = [0] * len_h
        # Perform the convolution
        for i in range(self.len):
            for j in range(g.len):
                h[i + j] += self.f[i] * g.f[j]
        # delete the zeros at the end
        while h[-1] == 0:
            h.pop()
        return Function(range(len(h)), Ts=self.Ts, f=h)
    
    def low_pass(self, wc, N, hamming_window=False):
        # Initialize 
        g = [0] * self.len
        filter_length = range(0, 2*N)
        w_hat = 2*wc/self.ws
        # 
        for k in range(self.len):
            if k in filter_length:
                g[k] = w_hat*np.sinc(w_hat*(k-N))
            if hamming_window:
                g[k] *= 0.54 - 0.46*np.cos(2*pi*k/(2*N))
        g = Function(self.n, Ts=self.Ts, f=g)
        return self.convolute(g)
    
    def convert_sampling_rate(self, N, ws_new=None, Ts_new=None):
        if ws_new is None:
            ws_new = 2*pi/Ts_new
        elif Ts_new is None:
            Ts_new = 2*pi/ws_new
        else:
            raise ValueError("Either Ts_new or ws_new must be provided.")
        #
        beta = self.ws/ws_new
        n_new = range(ceil(self.n[0]/beta), ceil(self.n[-1]/beta)+1)
        f_new = [0]*len(n_new)
        for l in n_new:
            n0 = ceil(l*beta)
            for n in range(n0-N, n0+N):
                if n in self.n:
                    f_new[l] += self.f[self.n.index(n)]*np.sinc(n-l*beta)
        return Function(n_new, Ts=Ts_new, f=f_new)
    
    def modulate(self, w):
        return Function(self.n, Ts=self.Ts, f=self.f*np.cos(w*np.array(self.t)))
    
    def demodulate(self, w):
        return Function(self.n, Ts=self.Ts, f=self.f*2*np.cos(w*np.array(self.t)))

# Example usage
if __name__ == "__main__":
    # Define first function
    w = 2 * pi # Frequency of the cosine wave
    fcn_handle = lambda t: np.cos(w*t) # Cosine function
    Ts = 0.01 # Sampling rate
    n = range(0, 501)
    f1 = Function(n, Ts=Ts, function_handle=fcn_handle)
    # Define second function: sawtooth wave
    fcn_handle = lambda t: t % (77*Ts) # Sawtooth function
    f2 = Function(n, Ts=Ts, function_handle=fcn_handle)
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(f1.t, f1.f, label='Cosine wave', marker='x')
    plt.plot(f2.t, f2.f, label='Sawtooth wave', marker='o')
    plt.title('Original functions')
    plt.legend()
    plt.show()

    # Lowpass filter the functions
    wc = 4*w # Cut-off frequency
    N = 15 # Filter length
    f1_low_pass = f1.low_pass(wc, N, hamming_window=True)
    f2_low_pass = f2.low_pass(wc, N, hamming_window=True)
    # Plot
    plt.figure(2)
    plt.plot(f1_low_pass.t, f1_low_pass.f, marker='x')
    plt.plot(f2_low_pass.t, f2_low_pass.f, marker='o')
    plt.title('Low-pass filtered functions')
    plt.show()

    # Modulate the functions
    w_mod1 = 5*w
    w_mod2 = 15*w
    f1_mod = f1.modulate(w_mod1)
    f2_mod = f2.modulate(w_mod2)
    # Plot
    plt.figure(3)
    plt.plot(f1_mod.t, f1_mod.f, marker='x')
    plt.plot(f2_mod.t, f2_mod.f, marker='o')
    plt.title('Modulated functions')
    plt.show()

    # Add the modulated functions
    f_sum = f1_mod + f2_mod
    # Plot
    plt.figure(4)
    plt.plot(f_sum.t, f_sum.f, marker='s', color='tab:green')
    plt.title('Sum of modulated functions')
    plt.show()

    # Demodulate the sum
    f1_demod = f_sum.demodulate(w_mod1)
    f2_demod = f_sum.demodulate(w_mod2)
    # Plot
    plt.figure(5)
    plt.plot(f1_demod.t, f1_demod.f, marker='x')
    plt.plot(f2_demod.t, f2_demod.f, marker='o')
    plt.title('Demodulated functions')
    plt.show()

    # Lowpass filter the demodulated functions
    f1_demod_low_pass = f1_demod.low_pass(wc, N, hamming_window=True)
    f2_demod_low_pass = f2_demod.low_pass(wc, N, hamming_window=True)
    # Plot
    plt.figure(6)
    plt.plot(f1_demod_low_pass.t, f1_demod_low_pass.f, marker='x')
    plt.plot(f2_demod_low_pass.t, f2_demod_low_pass.f, marker='o')
    plt.title('Reconstructed functions')
    plt.show()

    # compare the original and reconstructed functions
    plt.figure(7)
    # shift the new signal to the left by N
    t1_lp = [t_i - N*f1_demod_low_pass.Ts for t_i in f1_demod_low_pass.t]
    plt.plot(f1.t, f1.f, label='Original')
    plt.plot(t1_lp, f1_demod_low_pass.f, label='Reconstructed', color='navy')
    plt.legend()
    plt.title('Comparison of original and reconstructed functions')
    plt.show()
    plt.figure(8)
    # shift the new signal to the left by N
    t2_lp = [t_i - N*f2_demod_low_pass.Ts for t_i in f2_demod_low_pass.t]
    plt.plot(f2.t, f2.f, label='Original', color='tab:orange')
    plt.plot(t2_lp, f2_demod_low_pass.f, label='Reconstructed', color='darkred')
    plt.legend()
    plt.title('Comparison of original and reconstructed functions')
    plt.show()



