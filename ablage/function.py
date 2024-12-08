from math import pi
import numpy as np

class Function:
    """
    Object to represent a function.
    Attributes:
        n: iterable
            Sampling points
        Ts: float or int
            Sampling time [s]
        ws: float or int
            Sampling frequency [rad/s]
        f: iterable
            Function values
        t: iterable
            Time points [s]
        len: int
            Number of samples
    """
    def __init__(self, n, Ts=None, ws=None, f=None, function_handle=None):
        """
        Initialize the function object.
        Parameters:
            n: iterable
                Sampling points
            Ts=None: float or int
                Sampling time [s]
            ws=None: float or int
                Sampling frequency [rad/s]
            f=None: iterable
                Function values
            function_handle=None: function
                Function handle
        Raises:
            ValueError: Either Ts or ws must be provided.
            ValueError: Either f or function_handle must be provided.
        """
        if Ts is None and ws is not None:
            Ts = 2 * pi / ws
        elif ws is None and Ts is not None:
            ws = 2 * pi / Ts
        else:
            raise ValueError("Either Ts or ws must be provided.")

        self.ws = ws  # Sampling frequency
        self.Ts = Ts  # Sampling time
        self.n = np.array(n)  # Sampling points
        self.t = np.array([n_i * Ts for n_i in self.n])  # Time points
        self.len = len(self.n)  # Number of samples

        if f is not None:
            self.f = np.array(f)
        elif function_handle is not None:
            self.f = np.array(self._evaluate_function(function_handle))
        else:
            raise ValueError("Either f or function_handle must be provided.")

        # Validate that time and function lengths match
        if len(self.t) != len(self.f):
            raise ValueError(f"Time and function lengths do not match! t: {len(self.t)}, f: {len(self.f)}")

    def _evaluate_function(self, function_handle):
        """
        Evaluate the function handle to get the sampling values.
        Parameters:
            function_handle: function
                Function handle
        Return:
            iterable
                Sampling values
        """
        return [function_handle(t_i) for t_i in self.t]

    def update_length(self):
        """Update the length of the function based on the current size of f."""
        self.len = min(len(self.t), len(self.f))
