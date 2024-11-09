from math import pi

class Function:
    def __init__(self, n, Ts=None, ws=None, f=None, function_handle=None):
        if Ts is None:
            Ts = 2*pi/ws
        elif ws is None:
            ws = 2*pi/Ts
        else:
            raise ValueError("Either Ts or ws must be provided.")
        self.ws = ws
        self.Ts = Ts
        self.n = n
        self.t = [n_i * Ts for n_i in n]
        if f is not None:
            self.f = f
        elif function_handle is not None:
            self.f = self.evaluate_function(function_handle)
        else:
            raise ValueError("Either f or function_handle must be provided.")

    def evaluate_function(self, function_handle):
        # Evaluate the function handle to get the sampling points and values
        # Replace this with your own implementation
        f = [function_handle(t_i) for t_i in self.t]
        return f
