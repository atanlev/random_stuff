import torch
import torch.nn.functional as F

def torch_filtfilt(b, a, x):
    """
    PyTorch implementation of SciPy's filtfilt with Gustafsson’s method.
    Matches SciPy's reflection padding logic for FIR/IIR filters.
    
    Parameters
    ----------
    b : torch.Tensor
        Numerator filter coefficients (1D).
        (also called taps)
    a : torch.Tensor
        Denominator filter coefficients (1D).
    x : torch.Tensor
        Signal to be filtered (1D).
    """
    # Ensure float64 for numerical parity with SciPy
    b = b.clone().detach().to(dtype=torch.float64)
    a = a.clone().detach().to(dtype=torch.float64)
    x = x.clone().detach().to(dtype=torch.float64)

    if a.numel() != 1:
        if a[0] != 1.0:
            b = b / a[0]
            a = a / a[0]

    # Pad length 
    padlen = 3 * max(len(a), len(b))
    if x.numel() <= padlen:
        raise ValueError("Input signal too short for filtfilt padding")

    # Reflection padding 
    # Front pad
    front = 2 * x[0] - x[1:padlen+1].flip(0)
    # End pad
    end = 2 * x[-1] - x[-padlen-1:-1].flip(0)
    x_pad = torch.cat([front, x, end])

    # Forward filter
    y = torch_lfilter(b, a, x_pad)

    # Reverse
    y = torch.flip(y, dims=[0])

    # Backward filter
    y = torch_lfilter(b, a, y)

    # Reverse again
    y = torch.flip(y, dims=[0])

    # Remove padding
    y = y[padlen:-padlen]

    return y


def torch_lfilter(b, a, x):
    """
    Direct-form II transposed IIR filter implementation in PyTorch.
    if a = [1] feedback is skiped resulting in an FIR filter.
    """
    b = b.to(dtype=torch.float64)
    a = a.to(dtype=torch.float64)
    x = x.to(dtype=torch.float64)

    N = max(len(a), len(b))
    a_pad = torch.zeros(N, dtype=torch.float64)
    b_pad = torch.zeros(N, dtype=torch.float64)
    a_pad[:len(a)] = a
    b_pad[:len(b)] = b

    # State buffer
    zi = torch.zeros(N - 1, dtype=torch.float64)
    y = torch.zeros_like(x)

    for n in range(len(x)):
        y[n] = b_pad[0] * x[n] + zi[0]
        for i in range(1, N - 1):
            zi[i - 1] = b_pad[i] * x[n] + zi[i] - a_pad[i] * y[n]
        zi[-1] = b_pad[N - 1] * x[n] - a_pad[N - 1] * y[n]

    return y

class RealTimeIIR:
    """
    PyTorch implementation of a real time direct form I causal FIR/IIR filter
    i.e. y[n] = (1/a[0]) * ( Σ_{k=0}^{M} b[k] x[n-k]  -  Σ_{k=1}^{N} a[k] y[n-k] )
    """
    def __init__(self, b, a, device="cpu"):
        """
        Initialize filter with given filter coefficients,
        initialize sample history, filter history.

        Note
        ----
        - sample_history stores the past input samples (length len(b)-1).
        - filter_history stores the past output samples (length len(a)-1).
          Only used when len(a) > 1 (i.e. IIR case).        

        Parameters
        ----------
        b : torch.Tensor
            Numerator (FIR) coefficients.
        a : torch.Tensor
            Denominator (IIR) coefficients. a[0] is assumed nonzero (usually 1).
        """
        self.b = torch.tensor(b, dtype=torch.float32, device=device)
        self.a = torch.tensor(a, dtype=torch.float32, device=device)
        # normalize to ensure a[0] == 1
        self.b = self.b / self.a[0]
        self.a = self.a / self.a[0]        

        self.sample_history = torch.zeros(len(b)-1, dtype=torch.float32, device=device)
        if len(self.a) > 1:
            self.filter_history = torch.zeros(len(a)-1, dtype=torch.float32, device=device)

    def step(self, new_sample):
        sample_window = torch.cat((new_sample.view(1), self.sample_history))
        if len(self.a) > 1:
            y_new = self.b @ sample_window - self.a[1:] @ self.filter_history
        else:
            y_new = self.b @ sample_window

        self.sample_history = sample_window[:-1]
        if len(self.a) > 1:
            self.filter_history = torch.cat((y_new.view(1), self.filter_history[:-1]))

        return y_new
    

