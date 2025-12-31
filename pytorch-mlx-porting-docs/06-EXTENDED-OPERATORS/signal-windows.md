# Signal Processing Windows (torch.signal.windows)

## Purpose

Window functions are essential for spectral analysis, filter design, and signal processing. They reduce spectral leakage when computing Discrete Fourier Transforms (DFT) by tapering the signal at its boundaries. PyTorch provides a comprehensive set of window functions in `torch.signal.windows`.

**Reference Files:**
- `torch/signal/windows/windows.py` - All window implementations

---

## Overview

### Why Window Functions?

When analyzing a finite segment of a signal with FFT:
1. The signal is implicitly treated as periodic
2. Discontinuities at boundaries cause "spectral leakage"
3. Window functions taper the signal to reduce this artifact

```
Raw Signal           Windowed Signal
┌─────────────┐     ┌─────────────┐
│  ▄▄▄▄▄▄▄▄▄ │     │    ▄▄▄▄    │
│ ▀▀▀▀▀▀▀▀▀▀ │  →  │  ▄▀    ▀▄  │
│             │     │ ▀        ▀ │
└─────────────┘     └─────────────┘
Discontinuity        Smooth taper
at edges             at edges
```

### Common Parameters

All window functions share these parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `M` | int | Window length (number of points) |
| `sym` | bool | `True` for symmetric (filter design), `False` for periodic (spectral analysis) |
| `dtype` | torch.dtype | Data type (float32 or float64) |
| `device` | torch.device | Target device |
| `layout` | torch.layout | Tensor layout (must be strided) |
| `requires_grad` | bool | Enable gradient tracking |

### Symmetric vs Periodic

- **Symmetric (`sym=True`)**: Window is symmetric around center, suitable for filter design
- **Periodic (`sym=False`)**: Window suitable for spectral analysis with FFT

```python
# Symmetric: endpoints are equal, length M
hann_sym = torch.signal.windows.hann(10, sym=True)
# [0, ..., 1, ..., 0]  # Symmetric

# Periodic: one complete period, more natural for FFT
hann_per = torch.signal.windows.hann(10, sym=False)
# Endpoint would equal start if extended
```

---

## Available Windows

### bartlett

Triangular window with zero endpoints.

**Formula:**
$$w_n = 1 - \left| \frac{2n}{M-1} - 1 \right|$$

```python
torch.signal.windows.bartlett(M, *, sym=True, dtype=None, device=None)
```

**Characteristics:**
- Simple triangular shape
- First sidelobe: -26.5 dB
- Sidelobe rolloff: -12 dB/octave

**Example:**
```python
>>> torch.signal.windows.bartlett(10)
tensor([0.0000, 0.2222, 0.4444, 0.6667, 0.8889, 0.8889, 0.6667, 0.4444, 0.2222, 0.0000])
```

---

### blackman

Three-term cosine window with excellent sidelobe suppression.

**Formula:**
$$w_n = 0.42 - 0.5\cos\left(\frac{2\pi n}{M-1}\right) + 0.08\cos\left(\frac{4\pi n}{M-1}\right)$$

```python
torch.signal.windows.blackman(M, *, sym=True, dtype=None, device=None)
```

**Characteristics:**
- First sidelobe: -58 dB
- Excellent sidelobe suppression
- Wider main lobe than Hann/Hamming

**Example:**
```python
>>> torch.signal.windows.blackman(5)
tensor([-1.4901e-08,  3.4000e-01,  1.0000e+00,  3.4000e-01, -1.4901e-08])
```

---

### cosine

Simple sine window (half-period cosine).

**Formula:**
$$w_n = \sin\left(\frac{\pi(n + 0.5)}{M}\right)$$

```python
torch.signal.windows.cosine(M, *, sym=True, dtype=None, device=None)
```

**Characteristics:**
- Non-zero endpoints
- Simple computation
- Also known as "sine window"

**Example:**
```python
>>> torch.signal.windows.cosine(10)
tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540, 0.1564])
```

---

### exponential

Exponential decay window (Poisson window).

**Formula:**
$$w_n = \exp\left(-\frac{|n - c|}{\tau}\right)$$

where $c$ is the center and $\tau$ controls decay rate.

```python
torch.signal.windows.exponential(
    M,
    *,
    center=None,  # Default: M/2 (periodic) or (M-1)/2 (symmetric)
    tau=1.0,      # Decay constant (0-100, higher = slower decay)
    sym=True,
    dtype=None,
    device=None
)
```

**Characteristics:**
- Asymmetric if center is specified
- tau=100 approximates rectangular window
- Useful for certain audio applications

**Example:**
```python
>>> torch.signal.windows.exponential(10, tau=1.0)
tensor([0.0111, 0.0302, 0.0821, 0.2231, 0.6065, 0.6065, 0.2231, 0.0821, 0.0302, 0.0111])

>>> torch.signal.windows.exponential(10, tau=0.5, sym=False)
tensor([4.5400e-05, 3.3546e-04, 2.4788e-03, 1.8316e-02, 1.3534e-01, 1.0000e+00, ...])
```

---

### gaussian

Gaussian-shaped window.

**Formula:**
$$w_n = \exp\left(-\left(\frac{n}{2\sigma}\right)^2\right)$$

```python
torch.signal.windows.gaussian(
    M,
    *,
    std=1.0,  # Standard deviation (controls width)
    sym=True,
    dtype=None,
    device=None
)
```

**Characteristics:**
- Smooth, no sharp transitions
- Controlled by standard deviation
- No perfect nulls in frequency domain

**Example:**
```python
>>> torch.signal.windows.gaussian(10, std=1.0)
tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, ...])

>>> torch.signal.windows.gaussian(10, std=0.9, sym=False)
tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, ...])
```

---

### hamming

Raised cosine window with non-zero endpoints.

**Formula:**
$$w_n = 0.54 - 0.46\cos\left(\frac{2\pi n}{M-1}\right)$$

```python
torch.signal.windows.hamming(M, *, sym=True, dtype=None, device=None)
```

**Characteristics:**
- Non-zero endpoints (0.08)
- First sidelobe: -42.7 dB
- Good frequency resolution

**Example:**
```python
>>> torch.signal.windows.hamming(10)
tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])
```

---

### hann

Raised cosine window with zero endpoints.

**Formula:**
$$w_n = \frac{1}{2}\left[1 - \cos\left(\frac{2\pi n}{M-1}\right)\right] = \sin^2\left(\frac{\pi n}{M-1}\right)$$

```python
torch.signal.windows.hann(M, *, sym=True, dtype=None, device=None)
```

**Characteristics:**
- Zero endpoints
- First sidelobe: -31.5 dB
- Very common for general-purpose spectral analysis

**Example:**
```python
>>> torch.signal.windows.hann(10)
tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170, 0.0000])
```

---

### kaiser

Parameterized window with adjustable main lobe width vs sidelobe level tradeoff.

**Formula:**
$$w_n = \frac{I_0\left(\beta\sqrt{1 - \left(\frac{n - N/2}{N/2}\right)^2}\right)}{I_0(\beta)}$$

where $I_0$ is the zeroth-order modified Bessel function of the first kind.

```python
torch.signal.windows.kaiser(
    M,
    *,
    beta=12.0,  # Shape parameter (higher = narrower main lobe, higher sidelobes)
    sym=True,
    dtype=None,
    device=None
)
```

**Characteristics:**
- Adjustable tradeoff via β parameter
- β=0: rectangular window
- β=5: similar to Hamming
- β=6: similar to Hann
- β=8.6: similar to Blackman
- Uses `torch.special.i0` internally

**Example:**
```python
>>> torch.signal.windows.kaiser(10, beta=5)
tensor([0.0367, 0.1664, 0.4014, 0.6719, 0.9017, 0.9017, 0.6719, 0.4014, 0.1664, 0.0367])

>>> torch.signal.windows.kaiser(10, beta=14)
tensor([7.7e-06, 4.1e-04, 6.8e-03, 5.2e-02, 2.1e-01, ...])
```

---

### nuttall

Minimum 4-term Blackman-Harris window.

**Formula:**
$$w_n = 0.3635819 - 0.4891775\cos(z_n) + 0.1365995\cos(2z_n) - 0.0106411\cos(3z_n)$$

where $z_n = \frac{2\pi n}{M}$.

```python
torch.signal.windows.nuttall(M, *, sym=True, dtype=None, device=None)
```

**Characteristics:**
- First sidelobe: -93 dB
- Excellent sidelobe suppression
- Wider main lobe

**Example:**
```python
>>> torch.signal.windows.nuttall(10)
tensor([3.6280e-04, 5.5e-02, 3.5e-01, 7.5e-01, 9.9e-01, ...])
```

---

### general_cosine

General weighted sum of cosines.

**Formula:**
$$w_n = \sum_{i=0}^{M-1} (-1)^i a_i \cos\left(\frac{2\pi i n}{M-1}\right)$$

```python
torch.signal.windows.general_cosine(
    M,
    *,
    a,      # Iterable of coefficients
    sym=True,
    dtype=None,
    device=None
)
```

**Example:**
```python
>>> # Blackman window via general_cosine
>>> torch.signal.windows.general_cosine(10, a=[0.42, 0.5, 0.08])

>>> # Custom 3-term window
>>> torch.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31])
tensor([0.5400, 0.3376, 0.1288, 0.4200, 0.9136, ...])
```

---

### general_hamming

Generalized Hamming window.

**Formula:**
$$w_n = \alpha - (1 - \alpha)\cos\left(\frac{2\pi n}{M-1}\right)$$

```python
torch.signal.windows.general_hamming(
    M,
    *,
    alpha=0.54,  # Window coefficient
    sym=True,
    dtype=None,
    device=None
)
```

**Relationships:**
- `alpha=0.54`: Standard Hamming
- `alpha=0.5`: Hann window

**Example:**
```python
>>> # Hamming (alpha=0.54)
>>> torch.signal.windows.general_hamming(10)
tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, ...])

>>> # Hann (alpha=0.5)
>>> torch.signal.windows.general_hamming(10, alpha=0.5)
tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, ...])
```

---

## Window Comparison

| Window | First Sidelobe | Sidelobe Rolloff | Main Lobe Width | Use Case |
|--------|---------------|------------------|-----------------|----------|
| Rectangular | -13 dB | -6 dB/octave | Narrowest | Maximum frequency resolution |
| Bartlett | -27 dB | -12 dB/octave | Moderate | Simple applications |
| Hann | -32 dB | -18 dB/octave | Moderate | General purpose |
| Hamming | -43 dB | -6 dB/octave | Moderate | Audio processing |
| Blackman | -58 dB | -18 dB/octave | Wide | High dynamic range |
| Kaiser (β=8) | -60 dB | Adjustable | Wide | Flexible tradeoff |
| Nuttall | -93 dB | -18 dB/octave | Very wide | Very low sidelobes |

---

## Usage with FFT

### Spectral Analysis

```python
import torch
import torch.fft as fft

# Generate signal
t = torch.linspace(0, 1, 1024)
signal = torch.sin(2 * torch.pi * 50 * t) + 0.5 * torch.sin(2 * torch.pi * 120 * t)

# Apply window
window = torch.signal.windows.hann(1024, sym=False)
windowed_signal = signal * window

# Compute FFT
spectrum = fft.fft(windowed_signal)
magnitude = torch.abs(spectrum[:512])  # Positive frequencies
```

### Short-Time Fourier Transform (STFT)

```python
def stft(signal, window_size=256, hop_size=64, window_type='hann'):
    # Create window
    window = torch.signal.windows.hann(window_size, sym=False)

    # Compute STFT frames
    num_frames = (len(signal) - window_size) // hop_size + 1
    frames = []

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + window_size] * window
        spectrum = torch.fft.rfft(frame)
        frames.append(spectrum)

    return torch.stack(frames)
```

### Filter Design

```python
# Design a low-pass FIR filter with Hamming window
def fir_lowpass(cutoff, num_taps, fs):
    # Ideal impulse response
    n = torch.arange(num_taps) - (num_taps - 1) / 2
    h_ideal = torch.sinc(2 * cutoff / fs * n)

    # Apply window (symmetric for filter design)
    window = torch.signal.windows.hamming(num_taps, sym=True)
    h = h_ideal * window

    # Normalize
    return h / h.sum()
```

---

## Implementation Details

### Internal Structure

All windows follow this pattern:

```python
def window_function(M, *, sym=True, dtype=None, device=None, requires_grad=False):
    # Default dtype
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Validation
    _window_function_checks("name", M, dtype, layout)

    # Edge cases
    if M == 0:
        return torch.empty((0,), dtype=dtype, device=device)
    if M == 1:
        return torch.ones((1,), dtype=dtype, device=device)

    # Create index array
    # For periodic: divide by M
    # For symmetric: divide by M-1
    constant = ... / (M if not sym else M - 1)
    k = torch.linspace(start, end, steps=M, dtype=dtype, device=device)

    # Apply window formula
    return window_formula(k)
```

### Validation

```python
def _window_function_checks(name, M, dtype, layout):
    if M < 0:
        raise ValueError(f"{name} requires non-negative window length")
    if layout is not torch.strided:
        raise ValueError(f"{name} is implemented for strided tensors only")
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(f"{name} expects float32 or float64 dtypes")
```

---

## MLX Porting Considerations

### Direct Implementation

Most windows are straightforward to implement in MLX:

```python
import mlx.core as mx
import math

def hann(M, sym=True):
    """Hann window in MLX."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    N = M - 1 if sym else M
    n = mx.arange(M)
    return 0.5 * (1 - mx.cos(2 * math.pi * n / N))

def hamming(M, sym=True, alpha=0.54):
    """Hamming window in MLX."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    N = M - 1 if sym else M
    n = mx.arange(M)
    return alpha - (1 - alpha) * mx.cos(2 * math.pi * n / N)

def blackman(M, sym=True):
    """Blackman window in MLX."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    N = M - 1 if sym else M
    n = mx.arange(M)
    return (0.42
            - 0.5 * mx.cos(2 * math.pi * n / N)
            + 0.08 * mx.cos(4 * math.pi * n / N))

def gaussian(M, std=1.0, sym=True):
    """Gaussian window in MLX."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    start = -(M if not sym else M - 1) / 2.0
    n = mx.arange(M) + start
    return mx.exp(-(n / (2 * std)) ** 2)

def bartlett(M, sym=True):
    """Bartlett (triangular) window in MLX."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    N = M if not sym else M - 1
    n = mx.arange(M)
    return 1 - mx.abs(2 * n / N - 1)
```

### Kaiser Window (Requires Bessel)

The Kaiser window requires the modified Bessel function I₀:

```python
def i0_approx(x):
    """Approximate I0 using polynomial expansion for MLX."""
    # Use rational approximation or series expansion
    # For small x: I0(x) ≈ 1 + (x/2)^2 + (x/2)^4/4 + ...
    t = x / 3.75
    t2 = t * t

    # Polynomial coefficients for |x| < 3.75
    small = (1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (
             1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))))))

    # For larger x, use asymptotic expansion
    # ... (more complex)

    return small

def kaiser(M, beta=12.0, sym=True):
    """Kaiser window in MLX (requires Bessel function)."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    N = M if not sym else M - 1
    n = mx.arange(M)
    alpha = (M - 1) / 2.0

    arg = beta * mx.sqrt(1 - ((n - alpha) / alpha) ** 2)
    return i0_approx(arg) / i0_approx(mx.array(beta))
```

### General Cosine Framework

```python
def general_cosine(M, coeffs, sym=True):
    """General cosine window in MLX."""
    if M == 0:
        return mx.array([])
    if M == 1:
        return mx.array([1.0])

    N = M if not sym else M - 1
    n = mx.arange(M)

    result = mx.zeros(M)
    for i, a in enumerate(coeffs):
        sign = (-1) ** i
        result = result + sign * a * mx.cos(2 * math.pi * i * n / N)

    return result
```

---

## Priority for MLX Implementation

**High Priority:**
- `hann` - Most common general-purpose window
- `hamming` - Common in audio processing
- `blackman` - Good sidelobe suppression

**Medium Priority:**
- `gaussian` - Smooth characteristics
- `bartlett` - Simple implementation
- `general_cosine` - Enables custom windows

**Lower Priority:**
- `kaiser` - Requires Bessel function implementation
- `exponential` - Specialized use
- `nuttall` - Specialized use

---

## Summary

| Window | Formula Type | Special Requirements |
|--------|--------------|---------------------|
| bartlett | Linear | None |
| blackman | Cosine sum | None |
| cosine | Single sin | None |
| exponential | Exponential | None |
| gaussian | Gaussian | None |
| hamming | Cosine | None |
| hann | Cosine | None |
| kaiser | Bessel I₀ | `torch.special.i0` |
| nuttall | Cosine sum | None |
| general_cosine | Cosine sum | None |
| general_hamming | Cosine | None |

Window functions are fundamental for spectral analysis and filter design. The core windows (hann, hamming, blackman) are simple to implement in MLX, while kaiser requires a Bessel function approximation.
