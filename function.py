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


