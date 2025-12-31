# Fourier Transform Operators (torch.fft)

## Purpose

Fourier Transform operators convert signals between time/spatial domain and frequency domain. These are essential for:
- Signal processing and audio analysis
- Image filtering and convolution (FFT-based convolution)
- Spectral analysis in scientific computing
- Some transformer architectures using spectral methods

This document covers all 20 FFT operations in PyTorch's `torch.fft` module.

**Reference Files:**
- `torch/fft/__init__.py` - Python API and documentation
- `torch/_refs/fft.py` - Reference implementations
- `aten/src/ATen/native/SpectralOps.cpp` - Native implementations

---

## Overview

### Transform Categories

```
torch.fft Module
├── Complex FFT (full spectrum)
│   ├── fft, ifft         - 1D transforms
│   ├── fft2, ifft2       - 2D transforms
│   └── fftn, ifftn       - N-D transforms
│
├── Real FFT (one-sided, memory efficient)
│   ├── rfft, irfft       - 1D real transforms
│   ├── rfft2, irfft2     - 2D real transforms
│   └── rfftn, irfftn     - N-D real transforms
│
├── Hermitian FFT (time-domain Hermitian)
│   ├── hfft, ihfft       - 1D Hermitian transforms
│   ├── hfft2, ihfft2     - 2D Hermitian transforms
│   └── hfftn, ihfftn     - N-D Hermitian transforms
│
└── Utilities
    ├── fftfreq           - DFT sample frequencies
    ├── rfftfreq          - Real FFT sample frequencies
    ├── fftshift          - Shift zero-frequency to center
    └── ifftshift         - Inverse of fftshift
```

### Key Concepts

**Discrete Fourier Transform (DFT)**:
```
X[k] = Σ_{n=0}^{N-1} x[n] · e^{-2πikn/N}
```

**Inverse DFT**:
```
x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^{2πikn/N}
```

**Hermitian Symmetry**: For real input signals:
```
X[i] = conj(X[-i])
```
This property allows `rfft` to return only half the spectrum.

---

## Normalization Modes

All FFT functions support three normalization modes via the `norm` parameter:

| Mode | Forward Transform | Backward Transform | Combined |
|------|------------------|-------------------|----------|
| `"backward"` (default) | No normalization | Divide by `n` | `1/n` |
| `"forward"` | Divide by `n` | No normalization | `1/n` |
| `"ortho"` | Divide by `√n` | Divide by `√n` | `1/n` |

Where `n = prod(s)` is the logical FFT size.

**Orthonormal mode** (`"ortho"`) makes the transform unitary, preserving energy:
```python
torch.linalg.norm(x) == torch.linalg.norm(torch.fft.fft(x, norm="ortho"))
```

---

## Complex FFT Operations

### fft (1D Forward FFT)

**Purpose**: Compute the one-dimensional discrete Fourier transform.

**Signature**:
```python
torch.fft.fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor (real or complex) |
| `n` | int, optional | Signal length (zero-pad or trim) |
| `dim` | int | Dimension to transform (default: -1) |
| `norm` | str | Normalization mode |

**Output Shape**: Same as input, output is always complex.

**Algorithm**: Fast Fourier Transform (Cooley-Tukey algorithm)
- Complexity: O(n log n)
- Radix-2 for power-of-2 lengths
- Mixed-radix for other lengths

**Usage Examples**:
```python
import torch

# Basic FFT
t = torch.arange(4)
X = torch.fft.fft(t)
# tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

# Complex input
t = torch.tensor([0.+1.j, 2.+3.j, 4.+5.j, 6.+7.j])
X = torch.fft.fft(t)
# tensor([12.+16.j, -8.+0.j, -4.-4.j,  0.-8.j])

# With zero-padding
t = torch.arange(4)
X = torch.fft.fft(t, n=8)  # Zero-pad to length 8

# Orthonormal transform
X = torch.fft.fft(t, norm="ortho")
```

**Gradient**: Supported via `torch.fft.ifft` with conjugate.

---

### ifft (1D Inverse FFT)

**Purpose**: Compute the one-dimensional inverse discrete Fourier transform.

**Signature**:
```python
torch.fft.ifft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor
```

**Parameters**: Same as `fft`.

**Inverse Relationship**:
```python
x = torch.randn(10, dtype=torch.complex64)
X = torch.fft.fft(x)
x_recovered = torch.fft.ifft(X)
torch.allclose(x, x_recovered)  # True
```

**Usage Examples**:
```python
X = torch.tensor([6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
x = torch.fft.ifft(X)
# tensor([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j])
```

---

### fft2 (2D Forward FFT)

**Purpose**: Compute the 2-dimensional discrete Fourier transform.

**Signature**:
```python
torch.fft.fft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | Tuple[int] | Signal size in each dimension |
| `dim` | Tuple[int] | Dimensions to transform (default: last 2) |

**Separability**: 2D FFT is separable:
```python
x = torch.rand(10, 10, dtype=torch.complex64)
fft2_result = torch.fft.fft2(x)

# Equivalent to two 1D FFTs:
two_ffts = torch.fft.fft(torch.fft.fft(x, dim=0), dim=1)
torch.allclose(fft2_result, two_ffts)  # True
```

**Common Use Cases**:
- Image processing (frequency domain filtering)
- 2D convolution via FFT
- Spectral analysis of 2D data

---

### ifft2 (2D Inverse FFT)

**Purpose**: Compute the 2-dimensional inverse discrete Fourier transform.

**Signature**:
```python
torch.fft.ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor
```

---

### fftn (N-D Forward FFT)

**Purpose**: Compute the N-dimensional discrete Fourier transform.

**Signature**:
```python
torch.fft.fftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | Tuple[int], optional | Dimensions to transform (default: all) |

**Usage Examples**:
```python
# 3D FFT
x = torch.randn(10, 20, 30, dtype=torch.complex64)
X = torch.fft.fftn(x)  # Transform all dimensions

# Selective dimensions
X = torch.fft.fftn(x, dim=(0, 2))  # Only dims 0 and 2
```

---

### ifftn (N-D Inverse FFT)

**Purpose**: Compute the N-dimensional inverse discrete Fourier transform.

**Signature**:
```python
torch.fft.ifftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor
```

---

## Real FFT Operations

Real FFTs exploit Hermitian symmetry to use half the memory.

### rfft (1D Real FFT)

**Purpose**: Compute the FFT of real-valued input, returning only positive frequencies.

**Signature**:
```python
torch.fft.rfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor
```

**Output Shape**: For input of length `n`, output has length `n // 2 + 1`.

**Hermitian Property**: `X[i] = conj(X[-i])`, so negative frequencies are redundant.

**Usage Examples**:
```python
t = torch.arange(4)
X_full = torch.fft.fft(t)   # tensor([ 6.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
X_real = torch.fft.rfft(t)  # tensor([ 6.+0.j, -2.+2.j, -2.+0.j])

# Note: X_real omits X[-1] = conj(X[1])
```

**Memory Advantage**:
```python
x = torch.randn(1000)
fft_result = torch.fft.fft(x)    # 1000 complex values
rfft_result = torch.fft.rfft(x)  # 501 complex values (half + 1)
```

---

### irfft (1D Inverse Real FFT)

**Purpose**: Compute the inverse of `rfft`, returning real output.

**Signature**:
```python
torch.fft.irfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor
```

**Important**: The output length `n` should be specified for odd-length signals:
```python
t = torch.linspace(0, 1, 5)  # Odd length
T = torch.fft.rfft(t)

# Without n, assumes even length
wrong = torch.fft.irfft(T)  # Length 4 (wrong!)

# With n, correct round-trip
correct = torch.fft.irfft(T, n=5)  # Length 5 (correct)
torch.allclose(correct, t)  # True
```

---

### rfft2 (2D Real FFT)

**Purpose**: Compute the 2D FFT of real input.

**Signature**:
```python
torch.fft.rfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor
```

**Output Shape**: Last dimension is `s[-1] // 2 + 1`.

**Usage Examples**:
```python
t = torch.rand(10, 10)
X = torch.fft.rfft2(t)
X.size()  # torch.Size([10, 6]) - last dim is 10//2 + 1 = 6

# Compared to full fft2
X_full = torch.fft.fft2(t)
torch.allclose(X_full[..., :6], X)  # True
```

---

### irfft2 (2D Inverse Real FFT)

**Purpose**: Compute the inverse of `rfft2`.

**Signature**:
```python
torch.fft.irfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor
```

---

### rfftn (N-D Real FFT)

**Purpose**: Compute the N-dimensional FFT of real input.

**Signature**:
```python
torch.fft.rfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor
```

---

### irfftn (N-D Inverse Real FFT)

**Purpose**: Compute the inverse of `rfftn`.

**Signature**:
```python
torch.fft.irfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor
```

---

## Hermitian FFT Operations

Hermitian FFTs are the "opposite" of real FFTs:
- **Real FFT**: Real input → Hermitian-symmetric output
- **Hermitian FFT**: Hermitian-symmetric input → Real output

### hfft (1D Hermitian FFT)

**Purpose**: Compute the FFT of a Hermitian-symmetric signal, producing real output.

**Signature**:
```python
torch.fft.hfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor
```

**Relationship to rfft/irfft**:
```
rfft: Real time-domain → Hermitian frequency-domain
irfft: Hermitian frequency-domain → Real time-domain

hfft: Hermitian time-domain → Real frequency-domain
ihfft: Real frequency-domain → Hermitian time-domain
```

**Usage Examples**:
```python
# Real frequency signal
t = torch.linspace(0, 1, 5)
# Bring to time domain (gives Hermitian symmetric)
T = torch.fft.ifft(t)

# hfft converts Hermitian time-domain back to real frequency
result = torch.fft.hfft(T[:3], n=5)
torch.allclose(result, t)  # True
```

---

### ihfft (1D Inverse Hermitian FFT)

**Purpose**: Compute the inverse of `hfft`.

**Signature**:
```python
torch.fft.ihfft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor
```

---

### hfft2 / ihfft2 (2D Hermitian FFT)

**Signatures**:
```python
torch.fft.hfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor
torch.fft.ihfft2(input, s=None, dim=(-2, -1), norm=None, *, out=None) -> Tensor
```

---

### hfftn / ihfftn (N-D Hermitian FFT)

**Signatures**:
```python
torch.fft.hfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor
torch.fft.ihfftn(input, s=None, dim=None, norm=None, *, out=None) -> Tensor
```

---

## Utility Functions

### fftfreq (Sample Frequencies)

**Purpose**: Compute the DFT sample frequencies.

**Signature**:
```python
torch.fft.fftfreq(n, d=1.0, *, dtype=None, device=None) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | int | FFT length |
| `d` | float | Sample spacing (default: 1.0) |

**Output Ordering**: Positive frequencies first, then negative:
```
f = [0, 1, ..., (n-1)//2, -(n//2), ..., -1] / (d * n)
```

**Usage Examples**:
```python
torch.fft.fftfreq(5)
# tensor([ 0.0000,  0.2000,  0.4000, -0.4000, -0.2000])

torch.fft.fftfreq(4)
# tensor([ 0.0000,  0.2500, -0.5000, -0.2500])
# Note: Nyquist frequency at index 2 is negative by convention

# With physical sample spacing
sample_rate = 1000  # Hz
freqs = torch.fft.fftfreq(1000, d=1/sample_rate)  # Frequencies in Hz
```

---

### rfftfreq (Real FFT Sample Frequencies)

**Purpose**: Compute sample frequencies for `rfft` (positive only).

**Signature**:
```python
torch.fft.rfftfreq(n, d=1.0, *, dtype=None, device=None) -> Tensor
```

**Output**: Only non-negative frequencies:
```
f = [0, 1, ..., n//2] / (d * n)
```

**Usage Examples**:
```python
torch.fft.rfftfreq(5)
# tensor([0.0000, 0.2000, 0.4000])

torch.fft.rfftfreq(4)
# tensor([0.0000, 0.2500, 0.5000])
# Note: Nyquist is positive for rfftfreq
```

---

### fftshift (Center Zero-Frequency)

**Purpose**: Shift zero-frequency component to the center of the spectrum.

**Signature**:
```python
torch.fft.fftshift(input, dim=None) -> Tensor
```

**Usage**: Rearranges FFT output from `[0, 1, ..., n/2-1, -n/2, ..., -1]` to `[-n/2, ..., -1, 0, 1, ..., n/2-1]`.

**Usage Examples**:
```python
f = torch.fft.fftfreq(4)
# tensor([ 0.0000,  0.2500, -0.5000, -0.2500])

torch.fft.fftshift(f)
# tensor([-0.5000, -0.2500,  0.0000,  0.2500])

# 2D example
x = torch.fft.fft2(torch.randn(10, 10))
x_centered = torch.fft.fftshift(x)  # Zero at center
```

**Common Use Cases**:
- Visualization (center DC component)
- Filtering in frequency domain
- Centered spatial data processing

---

### ifftshift (Inverse Center Shift)

**Purpose**: Inverse of `fftshift`.

**Signature**:
```python
torch.fft.ifftshift(input, dim=None) -> Tensor
```

**Usage Examples**:
```python
f = torch.fft.fftfreq(5)
shifted = torch.fft.fftshift(f)
restored = torch.fft.ifftshift(shifted)
torch.allclose(f, restored)  # True
```

---

## Hardware Support

### CUDA Half-Precision

- Supported on GPU Architecture SM53 or greater
- **Restriction**: Only powers of 2 signal lengths

```python
# Half-precision FFT on CUDA
x = torch.randn(1024, device='cuda', dtype=torch.float16)
X = torch.fft.rfft(x)  # Works on SM53+

# Complex half-precision
x = torch.randn(1024, device='cuda', dtype=torch.complex32)
X = torch.fft.fft(x)  # Works on SM53+
```

### MPS (Metal) Backend

FFT operations are supported on Apple Silicon via Metal Performance Shaders:

```python
device = torch.device('mps')
x = torch.randn(1024, device=device)
X = torch.fft.rfft(x)  # Uses Metal
```

---

## Common Patterns

### Convolution via FFT

For large kernels, FFT convolution is faster than direct convolution:

```python
def fft_conv1d(signal, kernel):
    """1D convolution using FFT (circular/periodic)"""
    n = signal.shape[-1]

    # FFT both signals (zero-pad kernel to signal length)
    S = torch.fft.rfft(signal, n=n)
    K = torch.fft.rfft(kernel, n=n)

    # Multiply in frequency domain = convolve in time domain
    Y = S * K

    # Inverse FFT
    return torch.fft.irfft(Y, n=n)

def fft_conv2d(image, kernel):
    """2D convolution using FFT"""
    h, w = image.shape[-2:]

    S = torch.fft.rfft2(image)
    K = torch.fft.rfft2(kernel, s=(h, w))

    return torch.fft.irfft2(S * K, s=(h, w))
```

### Spectral Analysis

```python
def power_spectrum(signal, sample_rate):
    """Compute power spectrum density"""
    n = signal.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/sample_rate)

    X = torch.fft.rfft(signal, norm="ortho")
    power = torch.abs(X) ** 2

    return freqs, power

# Usage
signal = torch.randn(1000)
freqs, power = power_spectrum(signal, sample_rate=1000)
```

### Low-Pass Filter

```python
def lowpass_filter(signal, cutoff_freq, sample_rate):
    """Apply low-pass filter in frequency domain"""
    n = signal.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/sample_rate)

    # FFT
    X = torch.fft.rfft(signal)

    # Zero out frequencies above cutoff
    mask = freqs <= cutoff_freq
    X_filtered = X * mask

    # Inverse FFT
    return torch.fft.irfft(X_filtered, n=n)
```

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `torch.fft.fft` | `mx.fft.fft` |
| `torch.fft.ifft` | `mx.fft.ifft` |
| `torch.fft.fft2` | `mx.fft.fft2` |
| `torch.fft.ifft2` | `mx.fft.ifft2` |
| `torch.fft.fftn` | `mx.fft.fftn` |
| `torch.fft.ifftn` | `mx.fft.ifftn` |
| `torch.fft.rfft` | `mx.fft.rfft` |
| `torch.fft.irfft` | `mx.fft.irfft` |
| `torch.fft.rfft2` | `mx.fft.rfft2` |
| `torch.fft.irfft2` | `mx.fft.irfft2` |
| `torch.fft.rfftn` | `mx.fft.rfftn` |
| `torch.fft.irfftn` | `mx.fft.irfftn` |

### MLX Implementation Notes

1. **Normalization**: MLX uses the same normalization modes as PyTorch
2. **Real FFT**: MLX `rfft` returns complex output with half+1 elements
3. **Axis vs Dim**: MLX uses `axis` parameter, PyTorch uses `dim`

### Example Translation

```python
# PyTorch
import torch
x = torch.randn(1024)
X = torch.fft.rfft(x, norm="ortho")
x_back = torch.fft.irfft(X, n=1024, norm="ortho")

# MLX
import mlx.core as mx
x = mx.random.normal(shape=(1024,))
X = mx.fft.rfft(x, norm="ortho")
x_back = mx.fft.irfft(X, n=1024, norm="ortho")
```

### Utility Function Implementations for MLX

```python
import mlx.core as mx

def fftfreq(n, d=1.0):
    """MLX implementation of fftfreq"""
    results = mx.zeros(n)
    N = (n - 1) // 2 + 1
    results[:N] = mx.arange(N)
    results[N:] = mx.arange(-(n // 2), 0)
    return results / (n * d)

def rfftfreq(n, d=1.0):
    """MLX implementation of rfftfreq"""
    return mx.arange(n // 2 + 1) / (n * d)

def fftshift(x, axes=None):
    """MLX implementation of fftshift"""
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    shifts = [x.shape[ax] // 2 for ax in axes]
    return mx.roll(x, shifts, axes)

def ifftshift(x, axes=None):
    """MLX implementation of ifftshift"""
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    shifts = [-(x.shape[ax] // 2) for ax in axes]
    return mx.roll(x, shifts, axes)
```

---

## Gradient Support

All FFT operations support automatic differentiation:

```python
x = torch.randn(100, requires_grad=True)
X = torch.fft.fft(x)
loss = X.abs().sum()
loss.backward()
# x.grad is defined
```

**Gradient Formulas**:
- `grad(fft(x)) = fft(grad_output) * normalization_factor`
- `grad(ifft(x)) = ifft(grad_output) * normalization_factor`

---

## Implementation Files

**PyTorch Locations**:
- `torch/fft/__init__.py` - Python API
- `aten/src/ATen/native/SpectralOps.cpp` - CPU implementations
- `aten/src/ATen/native/cuda/SpectralOps.cu` - CUDA implementations
- `aten/src/ATen/native/mps/operations/Spectral.mm` - Metal implementations

---

## Summary Table

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `fft` | any | complex | General 1D frequency analysis |
| `ifft` | complex | complex | Inverse 1D transform |
| `fft2` | any | complex | 2D image frequency analysis |
| `ifft2` | complex | complex | Inverse 2D transform |
| `fftn` | any | complex | N-D frequency analysis |
| `ifftn` | complex | complex | Inverse N-D transform |
| `rfft` | real | complex | Memory-efficient 1D (real input) |
| `irfft` | complex | real | Inverse of rfft |
| `rfft2` | real | complex | Memory-efficient 2D (real input) |
| `irfft2` | complex | real | Inverse of rfft2 |
| `rfftn` | real | complex | Memory-efficient N-D (real input) |
| `irfftn` | complex | real | Inverse of rfftn |
| `hfft` | complex | real | Hermitian time-domain → real freq |
| `ihfft` | real | complex | Real freq → Hermitian time-domain |
| `fftfreq` | n, d | real | Sample frequencies for fft |
| `rfftfreq` | n, d | real | Sample frequencies for rfft |
| `fftshift` | any | any | Center zero-frequency |
| `ifftshift` | any | any | Inverse of fftshift |
