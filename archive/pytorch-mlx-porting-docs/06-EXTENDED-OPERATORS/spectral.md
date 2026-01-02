# Spectral Operations (FFT Family) - PyTorch → MLX Porting Guide

## Overview

This document provides comprehensive coverage of PyTorch's Fast Fourier Transform (FFT) operations and spectral analysis tools. The FFT family is essential for signal processing, audio analysis, spectral filtering, and frequency-domain computations in deep learning applications.

PyTorch provides a complete suite of FFT operations compatible with NumPy's API, including:
- **Complex-to-complex FFT** (fft, ifft, fftn, ifftn)
- **Real-to-complex FFT** (rfft, irfft, rfftn, irfftn) - exploits Hermitian symmetry
- **Hermitian-symmetric FFT** (hfft, ihfft, hfftn, ihfftn) - for real spectra
- **Short-Time Fourier Transform** (STFT/iSTFT) - for time-frequency analysis
- **Helper functions** (fftshift, ifftshift, fftfreq, rfftfreq)

### Key Use Cases

1. **Audio Processing**: Spectrograms, pitch detection, source separation
2. **Image Processing**: Frequency filtering, phase correlation, compression
3. **Signal Analysis**: Power spectral density, cross-correlation
4. **Physics Simulations**: Solving PDEs in frequency domain
5. **Deep Learning**: Spectral normalization, frequency-domain convolutions

---

## Table of Contents

1. [Core FFT Operations](#core-fft-operations)
2. [Multi-Dimensional FFT](#multi-dimensional-fft)
3. [Short-Time Fourier Transform (STFT)](#short-time-fourier-transform-stft)
4. [Helper Functions](#helper-functions)
5. [Implementation Details](#implementation-details)
6. [MLX Porting Guide](#mlx-porting-guide)
7. [Performance Considerations](#performance-considerations)

---

## Core FFT Operations

### 1D Complex-to-Complex FFT

#### torch.fft.fft (Forward FFT)

**Discrete Fourier Transform Formula**:
```
X[k] = Σ(n=0 to N-1) x[n] * exp(-2πi * k * n / N)

where:
- x[n]: input signal (time/space domain)
- X[k]: output spectrum (frequency domain)
- N: number of points
- k: frequency bin index
```

**PyTorch API**:
```python
torch.fft.fft(
    input: Tensor,
    n: Optional[int] = None,      # Number of points (default: input.size(dim))
    dim: int = -1,                # Dimension to transform
    norm: Optional[str] = None    # Normalization mode: None, 'forward', 'backward', 'ortho'
) -> Tensor
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:358-364](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L358-L364)):
```cpp
Tensor fft_fft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                      std::optional<std::string_view> norm) {
  return self.is_complex() ?
    fft_c2c("fft", {}, self, n, dim, norm, /*forward=*/true) :
    fft_r2c("fft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
}
```

**Normalization Modes** ([aten/src/ATen/native/SpectralOps.cpp:116-130](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L116-L130)):
- `None` or `"backward"`: No normalization on forward, divide by `n` on inverse
  ```
  Forward:  X[k] = Σ x[n] * exp(-2πikn/N)
  Inverse:  x[n] = (1/N) * Σ X[k] * exp(2πikn/N)
  ```
- `"forward"`: Divide by `n` on forward, no normalization on inverse
  ```
  Forward:  X[k] = (1/N) * Σ x[n] * exp(-2πikn/N)
  Inverse:  x[n] = Σ X[k] * exp(2πikn/N)
  ```
- `"ortho"`: Divide by `√n` on both forward and inverse (orthogonal)
  ```
  Forward:  X[k] = (1/√N) * Σ x[n] * exp(-2πikn/N)
  Inverse:  x[n] = (1/√N) * Σ X[k] * exp(2πikn/N)
  ```

**Example Usage**:
```python
import torch

# Complex input
x = torch.randn(128, dtype=torch.complex64)
X = torch.fft.fft(x)  # Shape: [128], complex

# Real input (automatically promoted to complex)
x_real = torch.randn(128)
X_real = torch.fft.fft(x_real)  # Shape: [128], complex

# Specify number of points (padding or truncation)
X_padded = torch.fft.fft(x, n=256)  # Shape: [256], zero-padded

# Orthogonal normalization
X_ortho = torch.fft.fft(x, norm='ortho')

# Multi-dimensional tensor
batch_x = torch.randn(32, 128, dtype=torch.complex64)
batch_X = torch.fft.fft(batch_x, dim=-1)  # FFT along last dimension
```

#### torch.fft.ifft (Inverse FFT)

**Inverse Discrete Fourier Transform Formula**:
```
x[n] = (1/N) * Σ(k=0 to N-1) X[k] * exp(2πi * k * n / N)
```

**PyTorch API**:
```python
torch.fft.ifft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None
) -> Tensor
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:376-381](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L376-L381)):
```cpp
Tensor fft_ifft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                       std::optional<std::string_view> norm) {
  return self.is_complex() ?
    fft_c2c("ifft", {}, self, n, dim, norm, /*forward=*/false) :
    fft_r2c("ifft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
}
```

**Roundtrip Property**:
```python
x = torch.randn(128, dtype=torch.complex64)
X = torch.fft.fft(x)
x_reconstructed = torch.fft.ifft(X)
assert torch.allclose(x, x_reconstructed)
```

---

### 1D Real-to-Complex FFT (Optimized)

#### torch.fft.rfft (Real FFT)

For real-valued signals, the FFT output has **Hermitian symmetry**: `X[k] = X*[N-k]`. The `rfft` function exploits this by only computing the first `N//2 + 1` frequency bins, reducing computation and memory by ~50%.

**PyTorch API**:
```python
torch.fft.rfft(
    input: Tensor,           # Must be real-valued
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None
) -> Tensor  # Complex output with size (n//2 + 1) along dim
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:393-396](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L393-L396)):
```cpp
Tensor fft_rfft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                       std::optional<std::string_view> norm) {
  return fft_r2c("rfft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
}
```

**Example**:
```python
# Real signal
x = torch.randn(100)  # Real-valued, shape: [100]

# Full FFT (returns all N=100 bins)
X_full = torch.fft.fft(x)  # Shape: [100], complex

# Real FFT (returns only N//2+1=51 bins)
X_rfft = torch.fft.rfft(x)  # Shape: [51], complex

# Verify Hermitian symmetry: X[k] == conj(X[N-k])
assert torch.allclose(X_full[1], X_full[-1].conj())  # X[1] == X*[99]
assert torch.allclose(X_full[10], X_full[-10].conj())  # X[10] == X*[90]

# rfft output matches first half of full FFT
assert torch.allclose(X_rfft, X_full[:51])
```

#### torch.fft.irfft (Inverse Real FFT)

Reconstructs a real signal from its one-sided FFT representation.

**PyTorch API**:
```python
torch.fft.irfft(
    input: Tensor,           # Complex, Hermitian-symmetric
    n: Optional[int] = None, # Output length (required if input is even-length)
    dim: int = -1,
    norm: Optional[str] = None
) -> Tensor  # Real-valued output
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:404-407](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L404-L407)):
```cpp
Tensor fft_irfft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                        std::optional<std::string_view> norm) {
  return fft_c2r("irfft", {}, self, n, dim, norm, /*forward=*/false);
}
```

**Ambiguity Handling**:
When `n` is not specified, irfft cannot determine if the original signal had even or odd length:
- `n=100`: rfft output has 51 bins (100//2 + 1)
- `n=101`: rfft output has 51 bins (101//2 + 1)

Therefore, **always specify `n`** for irfft when working with even-length signals:
```python
x = torch.randn(100)  # Even length
X = torch.fft.rfft(x)  # Shape: [51]

# Correct: specify n
x_reconstructed = torch.fft.irfft(X, n=100)  # Shape: [100]

# Incorrect: defaults to n=2*(51-1)=100, but could be ambiguous
x_default = torch.fft.irfft(X)  # Shape: [100], happens to work
```

---

### Hermitian FFT (For Real Spectra)

#### torch.fft.hfft / torch.fft.ihfft

These operations are useful when the **spectrum itself is real** (e.g., power spectral density, autocorrelation).

**torch.fft.hfft**: Assumes input has Hermitian symmetry, outputs real
**torch.fft.ihfft**: Takes real input, outputs Hermitian-symmetric complex

**PyTorch API**:
```python
torch.fft.hfft(
    input: Tensor,           # Complex, Hermitian-symmetric
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None
) -> Tensor  # Real-valued

torch.fft.ihfft(
    input: Tensor,           # Real-valued
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None
) -> Tensor  # Complex, Hermitian-symmetric (one-sided)
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:415-428](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L415-L428)):
```cpp
// hfft: Hermitian input -> real output
Tensor fft_hfft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                       std::optional<std::string_view> norm) {
  return fft_c2r("hfft", {}, self, n, dim, norm, /*forward=*/true);
}

// ihfft: Real input -> Hermitian output
Tensor fft_ihfft_symint(const Tensor& self, std::optional<SymInt> n, int64_t dim,
                        std::optional<std::string_view> norm) {
  return fft_r2c("ihfft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
}
```

**Example Use Case** (Autocorrelation via Wiener-Khinchin theorem):
```python
# Compute autocorrelation using FFT
signal = torch.randn(1000)
spectrum = torch.fft.rfft(signal)  # Complex spectrum
power_spectrum = spectrum.real**2 + spectrum.imag**2  # Real power spectrum

# Autocorrelation is inverse FFT of power spectrum
autocorr = torch.fft.hfft(power_spectrum, n=1000)  # Real autocorrelation
```

---

## Multi-Dimensional FFT

### torch.fft.fft2 / torch.fft.ifft2 (2D FFT)

**2D Discrete Fourier Transform**:
```
X[k₁, k₂] = Σ(n₁=0 to N₁-1) Σ(n₂=0 to N₂-1) x[n₁, n₂] * exp(-2πi(k₁n₁/N₁ + k₂n₂/N₂))
```

**PyTorch API**:
```python
torch.fft.fft2(
    input: Tensor,
    s: Optional[Tuple[int, int]] = None,  # Shape for each dimension
    dim: Tuple[int, int] = (-2, -1),      # Dimensions to transform
    norm: Optional[str] = None
) -> Tensor
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:644-647](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L644-L647)):
```cpp
// fft2 is just fftn with default dim=(-2, -1)
Tensor fft_fft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                       IntArrayRef dim, std::optional<std::string_view> norm) {
  return native::fft_fftn_symint(self, s, dim, std::move(norm));
}
```

**Example** (Image Processing):
```python
# 2D FFT on grayscale image
image = torch.randn(256, 256)  # Grayscale image
spectrum_2d = torch.fft.fft2(image)  # Shape: [256, 256], complex

# Magnitude spectrum (for visualization)
magnitude = torch.abs(spectrum_2d)
log_magnitude = torch.log(1 + magnitude)

# Apply frequency-domain filter (low-pass)
H, W = image.shape
mask = torch.zeros_like(spectrum_2d)
mask[H//2-20:H//2+20, W//2-20:W//2+20] = 1  # Keep low frequencies
filtered_spectrum = spectrum_2d * mask
filtered_image = torch.fft.ifft2(filtered_spectrum).real

# Batch processing (images with channels)
batch_images = torch.randn(32, 3, 256, 256)  # [batch, channels, H, W]
batch_spectrum = torch.fft.fft2(batch_images, dim=(-2, -1))  # FFT on spatial dims
```

### torch.fft.rfft2 / torch.fft.irfft2 (2D Real FFT)

Optimized 2D FFT for real-valued inputs (e.g., images).

**PyTorch API**:
```python
torch.fft.rfft2(
    input: Tensor,           # Real-valued
    s: Optional[Tuple[int, int]] = None,
    dim: Tuple[int, int] = (-2, -1),
    norm: Optional[str] = None
) -> Tensor  # Complex, one-sided along last dim
```

**Output Shape**:
```python
# Input shape: [..., H, W]
# Output shape: [..., H, W//2 + 1]  (one-sided along last dimension)

image = torch.randn(256, 256)
spectrum = torch.fft.rfft2(image)  # Shape: [256, 129], complex
```

**Example** (Phase Correlation for Image Registration):
```python
def phase_correlation(img1, img2):
    """Find translation offset between two images using phase correlation."""
    # Compute FFT of both images
    F1 = torch.fft.rfft2(img1)
    F2 = torch.fft.rfft2(img2)

    # Cross-power spectrum
    cross_power = F1 * F2.conj()
    normalized = cross_power / (cross_power.abs() + 1e-8)

    # Inverse FFT to get correlation peak
    correlation = torch.fft.irfft2(normalized, s=img1.shape)

    # Find peak (corresponds to translation)
    peak = correlation.argmax()
    shift_y = peak // img1.shape[1]
    shift_x = peak % img1.shape[1]

    return shift_y, shift_x
```

### torch.fft.fftn / torch.fft.ifftn (N-D FFT)

General N-dimensional FFT for arbitrary number of dimensions.

**PyTorch API**:
```python
torch.fft.fftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,  # Shape for each dimension
    dim: Optional[Sequence[int]] = None, # Dimensions to transform (default: all)
    norm: Optional[str] = None
) -> Tensor
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:437-444](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L437-L444)):
```cpp
Tensor fft_fftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                       at::OptionalIntArrayRef dim,
                       std::optional<std::string_view> norm) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  return fftn_c2c("fftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/true);
}
```

**Dimension Selection** ([aten/src/ATen/native/SpectralOps.cpp:283-342](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L283-L342)):
- If `dim` is None: transform **all** dimensions
- If `s` is provided: transform last `len(s)` dimensions (unless `dim` is also specified)
- If both `s` and `dim` provided: must have same length

**Example** (3D Volume Processing):
```python
# 3D medical scan
volume = torch.randn(128, 128, 128)  # CT/MRI volume

# 3D FFT on all dimensions
spectrum_3d = torch.fft.fftn(volume)  # Shape: [128, 128, 128], complex

# 3D FFT on specific dimensions (e.g., last 3 dims of 5D tensor)
batch_volume = torch.randn(8, 4, 64, 64, 64)  # [batch, channels, D, H, W]
spectrum = torch.fft.fftn(batch_volume, dim=(-3, -2, -1))

# Specify output shape (padding/truncation)
spectrum_padded = torch.fft.fftn(volume, s=(256, 256, 256))  # Zero-pad to 256³
```

---

## Short-Time Fourier Transform (STFT)

### torch.stft

The Short-Time Fourier Transform computes the FFT of **overlapping windows** of a signal, producing a **time-frequency representation** (spectrogram).

**Algorithm**:
1. **Windowing**: Split signal into overlapping frames using a window function
2. **FFT**: Compute FFT of each frame
3. **Output**: 2D tensor with axes [frequency, time]

**PyTorch API**:
```python
torch.stft(
    input: Tensor,                          # 1D or 2D (batch) signal
    n_fft: int,                             # FFT size
    hop_length: Optional[int] = None,       # Hop size between frames (default: n_fft // 4)
    win_length: Optional[int] = None,       # Window size (default: n_fft)
    window: Optional[Tensor] = None,        # Window function (default: rectangular)
    center: bool = True,                    # Pad signal for centered frames
    pad_mode: str = 'reflect',              # Padding mode
    normalized: bool = False,               # Apply sqrt(n_fft) normalization
    onesided: Optional[bool] = None,        # Return one-sided FFT (for real input)
    return_complex: Optional[bool] = None   # Return complex tensor (recommended)
) -> Tensor  # Shape: [..., freq_bins, time_frames] if return_complex=True
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:825-990](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L825-L990)):

Key steps:
1. **Center padding** ([lines 897-907](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L897-L907)):
```cpp
if (center) {
  const auto pad_amount = n_fft / 2;
  input = at::pad(input.view(extended_shape), {pad_amount, pad_amount}, mode);
  input = input.view(IntArrayRef(input.sizes()).slice(extra_dims));
}
```

2. **Time2col (frame construction)** ([lines 958-962](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L958-L962)):
```cpp
input = input.as_strided(
  {batch, n_frames, n_fft},
  {input.stride(0), hop_length * input.stride(1), input.stride(1)}
);
```

3. **Window application** ([lines 963-965](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L963-L965)):
```cpp
if (window_.defined()) {
  input = input.mul(window_);
}
```

4. **FFT** ([lines 973-978](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L973-L978)):
```cpp
if (complex_fft) {
  out = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/true);
} else {
  out = at::_fft_r2c(input, input.dim() - 1, static_cast<int64_t>(norm), onesided);
}
```

**Frame Count Calculation** ([lines 949-957](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L949-L957)):
```cpp
int64_t n_frames;
if (!center && align_to_window) {
  // Window-aligned: (signal_length - window_length) / hop_length + 1
  n_frames = 1 + (len - win_length) / hop_length;
} else {
  // Standard: (signal_length - n_fft) / hop_length + 1
  n_frames = 1 + (len - n_fft) / hop_length;
}
```

**Example** (Audio Spectrogram):
```python
import torch
import torchaudio.transforms as T

# Audio signal (1 second at 16kHz)
waveform = torch.randn(16000)

# STFT parameters
n_fft = 512          # FFT size
hop_length = 160     # 10ms hop (16000 / 100)
win_length = 400     # 25ms window (16000 / 40)
window = torch.hann_window(win_length)

# Compute STFT
stft_result = torch.stft(
    waveform,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    center=True,
    return_complex=True
)
# Shape: [257, 101] (freq_bins=n_fft//2+1, time_frames)

# Magnitude spectrogram
spectrogram = torch.abs(stft_result)  # Shape: [257, 101]

# Log-mel spectrogram (common in speech processing)
mel_transform = T.MelScale(n_mels=80, sample_rate=16000, n_stft=n_fft // 2 + 1)
mel_spec = mel_transform(spectrogram**2)  # Power spectrogram -> Mel scale
log_mel_spec = torch.log(mel_spec + 1e-9)
```

### torch.istft (Inverse STFT)

Reconstructs a time-domain signal from its STFT representation using **overlap-add**.

**PyTorch API**:
```python
torch.istft(
    input: Tensor,                          # Complex STFT tensor
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,           # Trim output to this length
    return_complex: bool = False
) -> Tensor  # 1D or 2D time-domain signal
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:1025-1197](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L1025-L1197)):

Key steps:
1. **IFFT** ([lines 1139-1150](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L1139-L1150)):
```cpp
if (return_complex) {
  input = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/false);
} else {
  input = at::_fft_c2r(input, input.dim() - 1, static_cast<int64_t>(norm), n_fft);
}
```

2. **Window multiplication** ([line 1153](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L1153)):
```cpp
Tensor y_tmp = input * window_tmp.view({1, 1, n_fft});
```

3. **Overlap-add** ([lines 1155-1167](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L1155-L1167)):
```cpp
Tensor y = at::unfold_backward(
  y_tmp,
  /*input_sizes=*/{y_tmp.size(0), expected_output_signal_len},
  /*dim=*/1,
  /*size=*/n_fft,
  /*step=*/hop_length);

// Window envelope for normalization
window_tmp = window_tmp.pow(2).expand({1, n_frames, n_fft});
Tensor window_envelop = at::unfold_backward(
  window_tmp,
  /*input_sizes=*/{1, expected_output_signal_len},
  /*dim=*/1,
  /*size=*/n_fft,
  /*step=*/hop_length);
```

4. **Normalization** ([line 1193](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L1193)):
```cpp
y = (y / window_envelop);  // Normalize by window overlap
```

**Perfect Reconstruction Conditions**:
For `istft(stft(x))` to perfectly reconstruct `x`:
1. Use the **same window** in both directions
2. Window must satisfy **Constant Overlap-Add (COLA)** constraint:
   ```
   Σ w²(n - kH) = C  (constant for all n)

   where:
   - w(n): window function
   - H: hop_length
   - k: frame index
   ```

**Example** (Roundtrip):
```python
# Original signal
signal = torch.randn(16000)
window = torch.hann_window(400)

# STFT
stft_result = torch.stft(
    signal, n_fft=512, hop_length=160, win_length=400,
    window=window, center=True, return_complex=True
)

# ISTFT (perfect reconstruction)
reconstructed = torch.istft(
    stft_result, n_fft=512, hop_length=160, win_length=400,
    window=window, center=True, length=signal.shape[0]
)

# Check reconstruction error
error = torch.abs(signal - reconstructed).max()
print(f"Max reconstruction error: {error.item():.2e}")  # ~1e-6 (floating point precision)

# Verify COLA constraint for Hann window
hop = 160
win_len = 400
window_squared = window**2
overlap_add = sum(window_squared[i::hop] for i in range(hop))
print(f"COLA constant: {overlap_add[0].item():.4f}")  # Should be constant
```

---

## Helper Functions

### torch.fft.fftshift / torch.fft.ifftshift

Shift zero-frequency component to center of spectrum (useful for visualization).

**Algorithm**:
```python
# fftshift: shift by floor(n/2)
# ifftshift: shift by ceil(n/2)
```

**PyTorch API**:
```python
torch.fft.fftshift(input: Tensor, dim: Optional[Sequence[int]] = None) -> Tensor
torch.fft.ifftshift(input: Tensor, dim: Optional[Sequence[int]] = None) -> Tensor
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:767-789](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L767-L789)):
```cpp
Tensor fft_fftshift(const Tensor& x, at::OptionalIntArrayRef dim_opt) {
  auto dim = default_alldims(x, dim_opt);
  SymIntArrayRef x_sizes = x.sym_sizes();
  SymDimVector shift(dim.size());
  for (const auto i : c10::irange(dim.size())) {
    shift[i] = x_sizes[dim[i]] / 2;  // floor(n/2)
  }
  return at::roll_symint(x, shift, dim);
}

Tensor fft_ifftshift(const Tensor& x, at::OptionalIntArrayRef dim_opt) {
  auto dim = default_alldims(x, dim_opt);
  SymIntArrayRef x_sizes = x.sym_sizes();
  SymDimVector shift(dim.size());
  for (const auto i : c10::irange(dim.size())) {
    shift[i] = (x_sizes[dim[i]] + 1) / 2;  // ceil(n/2)
  }
  return at::roll_symint(x, shift, dim);
}
```

**Example**:
```python
# 1D spectrum
X = torch.fft.fft(torch.randn(8))
# Default order: [DC, pos_freq1, pos_freq2, pos_freq3, neg_freq3, neg_freq2, neg_freq1]

X_shifted = torch.fft.fftshift(X)
# Shifted order: [neg_freq3, neg_freq2, neg_freq1, DC, pos_freq1, pos_freq2, pos_freq3]

# Undo shift
X_original = torch.fft.ifftshift(X_shifted)
assert torch.allclose(X, X_original)

# 2D spectrum (for images)
image = torch.randn(256, 256)
spectrum = torch.fft.fft2(image)
spectrum_centered = torch.fft.fftshift(spectrum)  # DC at center

# Visualize magnitude
import matplotlib.pyplot as plt
magnitude = torch.log(torch.abs(spectrum_centered) + 1)
plt.imshow(magnitude)
plt.title('Centered FFT Magnitude')
```

### torch.fft.fftfreq / torch.fft.rfftfreq

Generate frequency bins corresponding to FFT output.

**PyTorch API**:
```python
torch.fft.fftfreq(
    n: int,         # FFT size
    d: float = 1.0  # Sample spacing (inverse of sampling rate)
) -> Tensor  # Shape: [n], frequencies in cycles per unit

torch.fft.rfftfreq(
    n: int,
    d: float = 1.0
) -> Tensor  # Shape: [n//2 + 1], positive frequencies only
```

**Implementation** ([aten/src/ATen/native/SpectralOps.cpp:706-748](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L706-L748)):
```cpp
Tensor& fft_fftfreq_out(int64_t n, double d, Tensor& out) {
  at::arange_out(out, n);
  auto right_slice = out.slice(0, (n + 1) / 2, 0);  // Negative frequencies
  at::arange_out(right_slice, -(n/2), 0, 1);
  return out.mul_(1.0 / (n * d));  // Normalize by sampling interval
}

Tensor& fft_rfftfreq_out(int64_t n, double d, Tensor& out) {
  native::arange_out(n/2 + 1, out);  // Only positive frequencies
  return out.mul_(1.0 / (n * d));
}
```

**Example** (Audio Frequency Axis):
```python
# Audio parameters
sample_rate = 16000  # Hz
n_fft = 512
duration = 1.0  # seconds

# Generate signal
t = torch.linspace(0, duration, int(sample_rate * duration))
signal = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz tone (A4)

# Compute FFT
spectrum = torch.fft.rfft(signal)

# Frequency axis
freqs = torch.fft.rfftfreq(len(signal), d=1.0/sample_rate)
# freqs[0] = 0 Hz (DC)
# freqs[-1] ≈ 8000 Hz (Nyquist frequency)

# Find peak frequency
peak_idx = spectrum.abs().argmax()
peak_freq = freqs[peak_idx]
print(f"Peak frequency: {peak_freq:.1f} Hz")  # Should be ~440 Hz

# Plot spectrum
import matplotlib.pyplot as plt
plt.plot(freqs.numpy(), spectrum.abs().numpy())
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 2000)
plt.title('Frequency Spectrum')
```

---

## Implementation Details

### Type Promotion for FFT

**Promotion Rules** ([aten/src/ATen/native/SpectralOps.cpp:73-104](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L73-L104)):

```cpp
ScalarType promote_type_fft(ScalarType type, bool require_complex, Device device) {
  if (at::isComplexType(type)) {
    return type;  // Already complex
  }

  // Promote integers to default float
  if (!at::isFloatingType(type)) {
    type = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  // Only CUDA supports half precision
  const bool maybe_support_half = (device.is_cuda() || device.is_meta());
  if (maybe_support_half) {
    TORCH_CHECK(type == kHalf || type == kFloat || type == kDouble, "Unsupported dtype ", type);
  } else {
    TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);
  }

  // Promote to complex if required
  if (require_complex) {
    switch (type) {
    case kHalf: return kComplexHalf;
    case kFloat: return kComplexFloat;
    case kDouble: return kComplexDouble;
    default: TORCH_INTERNAL_ASSERT(false);
    }
  }

  return type;
}
```

**Supported Types**:
- **CPU**: float32, float64 → complex64, complex128
- **CUDA**: float16, float32, float64 → complex32, complex64, complex128
- **Integers**: Promoted to default float type (usually float32)

### Input Resizing (Padding/Truncation)

**resize_fft_input** ([aten/src/ATen/native/SpectralOps.cpp:134-157](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L134-L157)):

```cpp
Tensor resize_fft_input(Tensor x, IntArrayRef dims, SymIntArrayRef sizes) {
  bool must_copy = false;
  auto x_sizes = x.sym_sizes();
  SymDimVector pad_amount(x_sizes.size() * 2);

  for (const auto i : c10::irange(dims.size())) {
    if (sizes[i] == -1) {
      continue;  // Use original size
    }

    // Padding required
    if (x_sizes[dims[i]] < sizes[i]) {
      must_copy = true;
      auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
      pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];  // Pad on right
    }

    // Truncation required
    if (x_sizes[dims[i]] > sizes[i]) {
      x = x.slice_symint(dims[i], 0, sizes[i]);  // Slice from start
    }
  }

  return must_copy ? at::constant_pad_nd_symint(x, pad_amount) : x;
}
```

**Behavior**:
- `n < input.size(dim)`: Truncate from start
- `n > input.size(dim)`: Zero-pad on right
- `n == -1`: Use original size (no resizing)

### Backend Dispatch

PyTorch FFT operations dispatch to backend-specific implementations:

**CPU**:
- **PocketFFT** (default): Pure C++ implementation
- **MKL**: Intel Math Kernel Library (if available)

**CUDA**:
- **cuFFT**: NVIDIA's FFT library
- **cuFFTDx**: Device-side FFT for small sizes

**MPS (Apple Silicon)**:
- **vDSP**: Accelerate framework for real FFT
- **Metal Performance Shaders**: For complex FFT

**cuFFT Plan Caching** ([aten/src/ATen/native/SpectralOps.cpp:794-808](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp#L794-L808)):
```cpp
int64_t _cufft_get_plan_cache_max_size(DeviceIndex device_index);
void _cufft_set_plan_cache_max_size(DeviceIndex device_index, int64_t max_size);
int64_t _cufft_get_plan_cache_size(DeviceIndex device_index);
void _cufft_clear_plan_cache(DeviceIndex device_index);
```

---

## MLX Porting Guide

### MLX FFT API Status (as of 2024)

MLX provides a basic FFT API similar to NumPy/PyTorch:

**Available Operations**:
```python
import mlx.core as mx

# 1D FFT
mx.fft.fft(a, n=None, axis=-1)
mx.fft.ifft(a, n=None, axis=-1)
mx.fft.rfft(a, n=None, axis=-1)
mx.fft.irfft(a, n=None, axis=-1)

# 2D FFT
mx.fft.fft2(a, s=None, axes=(-2, -1))
mx.fft.ifft2(a, s=None, axes=(-2, -1))
mx.fft.rfft2(a, s=None, axes=(-2, -1))
mx.fft.irfft2(a, s=None, axes=(-2, -1))

# N-D FFT
mx.fft.fftn(a, s=None, axes=None)
mx.fft.ifftn(a, s=None, axes=None)
mx.fft.rfftn(a, s=None, axes=None)
mx.fft.irfftn(a, s=None, axes=None)
```

**Missing Operations** (as of early 2024):
- ❌ `hfft` / `ihfft` (Hermitian FFT)
- ❌ `fftshift` / `ifftshift` (can implement using `mx.roll`)
- ❌ `fftfreq` / `rfftfreq` (can implement using `mx.arange`)
- ❌ `stft` / `istft` (need to implement from scratch)

### Implementing Missing Operations in MLX

#### fftshift / ifftshift

```python
import mlx.core as mx

def fftshift(x, axes=None):
    """Shift zero-frequency component to center."""
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    shifts = [x.shape[ax] // 2 for ax in axes]
    return mx.roll(x, shifts, axes)

def ifftshift(x, axes=None):
    """Inverse fftshift."""
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    shifts = [(x.shape[ax] + 1) // 2 for ax in axes]
    return mx.roll(x, shifts, axes)
```

#### fftfreq / rfftfreq

```python
def fftfreq(n, d=1.0):
    """Return frequency bins for FFT output."""
    val = 1.0 / (n * d)
    results = mx.arange(n, dtype=mx.float32)

    # Set negative frequencies
    N = (n - 1) // 2 + 1
    results[N:] = results[N:] - n

    return results * val

def rfftfreq(n, d=1.0):
    """Return frequency bins for rfft output."""
    val = 1.0 / (n * d)
    return mx.arange(n // 2 + 1, dtype=mx.float32) * val
```

#### STFT Implementation

```python
def stft(
    signal,
    n_fft=512,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    pad_mode='reflect'
):
    """
    Short-Time Fourier Transform for MLX.

    Args:
        signal: Input signal, shape [..., time]
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window size
        window: Window function (default: Hann)
        center: Whether to pad signal for centered frames
        pad_mode: Padding mode

    Returns:
        STFT matrix, shape [..., freq_bins, time_frames]
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        # Hann window
        n = mx.arange(win_length, dtype=mx.float32)
        window = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / win_length)

    signal_shape = signal.shape
    signal = signal.reshape(-1, signal_shape[-1])  # Flatten batch dims
    batch_size, signal_len = signal.shape

    # Center padding
    if center:
        pad_amount = n_fft // 2
        signal = mx.pad(signal, [(0, 0), (pad_amount, pad_amount)], mode=pad_mode)
        signal_len = signal.shape[-1]

    # Number of frames
    n_frames = 1 + (signal_len - n_fft) // hop_length

    # Construct frames using sliding window
    # MLX doesn't have as_strided, so we use explicit indexing
    frames = []
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[:, start:start + n_fft]

        # Pad window if needed
        if win_length < n_fft:
            left_pad = (n_fft - win_length) // 2
            window_padded = mx.pad(window, [(left_pad, n_fft - win_length - left_pad)])
        else:
            window_padded = window

        # Apply window
        windowed_frame = frame * window_padded[None, :]
        frames.append(windowed_frame)

    # Stack frames: [batch, n_frames, n_fft]
    frames = mx.stack(frames, axis=1)

    # Compute FFT: [batch, n_frames, freq_bins]
    stft_result = mx.fft.rfft(frames, axis=-1)

    # Transpose to [batch, freq_bins, n_frames]
    stft_result = mx.transpose(stft_result, (0, 2, 1))

    # Restore original batch shape
    output_shape = list(signal_shape[:-1]) + [stft_result.shape[1], stft_result.shape[2]]
    return stft_result.reshape(output_shape)


def istft(
    stft_matrix,
    n_fft=512,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    length=None
):
    """
    Inverse Short-Time Fourier Transform for MLX.

    Args:
        stft_matrix: STFT matrix, shape [..., freq_bins, time_frames]
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window size
        window: Window function
        center: Whether to trim padding
        length: Output signal length

    Returns:
        Reconstructed signal, shape [..., time]
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        n = mx.arange(win_length, dtype=mx.float32)
        window = 0.5 - 0.5 * mx.cos(2 * mx.pi * n / win_length)

    stft_shape = stft_matrix.shape
    stft_matrix = stft_matrix.reshape(-1, stft_shape[-2], stft_shape[-1])
    batch_size, freq_bins, n_frames = stft_matrix.shape

    # Transpose to [batch, n_frames, freq_bins]
    stft_matrix = mx.transpose(stft_matrix, (0, 2, 1))

    # Inverse FFT: [batch, n_frames, n_fft]
    frames = mx.fft.irfft(stft_matrix, n=n_fft, axis=-1)

    # Pad window if needed
    if win_length < n_fft:
        left_pad = (n_fft - win_length) // 2
        window_padded = mx.pad(window, [(left_pad, n_fft - win_length - left_pad)])
    else:
        window_padded = window

    # Apply window
    windowed_frames = frames * window_padded[None, None, :]

    # Overlap-add
    expected_len = n_fft + hop_length * (n_frames - 1)
    signal = mx.zeros((batch_size, expected_len), dtype=mx.float32)
    window_sum = mx.zeros((batch_size, expected_len), dtype=mx.float32)

    window_sq = (window_padded ** 2)[None, :]
    for i in range(n_frames):
        start = i * hop_length
        signal[:, start:start + n_fft] += windowed_frames[:, i, :]
        window_sum[:, start:start + n_fft] += window_sq

    # Normalize by window overlap
    signal = signal / (window_sum + 1e-8)

    # Trim padding
    if center:
        pad_amount = n_fft // 2
        signal = signal[:, pad_amount:-pad_amount]

    # Trim to specified length
    if length is not None:
        signal = signal[:, :length]

    # Restore original batch shape
    output_shape = list(stft_shape[:-2]) + [signal.shape[-1]]
    return signal.reshape(output_shape)
```

### C++ API Design for MLX

```cpp
// mlx/fft/fft.h

namespace mlx::core::fft {

// 1D FFT
array fft(const array& a, int n = -1, int axis = -1, const std::string& norm = "backward");
array ifft(const array& a, int n = -1, int axis = -1, const std::string& norm = "backward");
array rfft(const array& a, int n = -1, int axis = -1, const std::string& norm = "backward");
array irfft(const array& a, int n = -1, int axis = -1, const std::string& norm = "backward");

// 2D FFT
array fft2(const array& a,
           std::optional<std::vector<int>> s = std::nullopt,
           std::vector<int> axes = {-2, -1},
           const std::string& norm = "backward");
array ifft2(const array& a,
            std::optional<std::vector<int>> s = std::nullopt,
            std::vector<int> axes = {-2, -1},
            const std::string& norm = "backward");

// N-D FFT
array fftn(const array& a,
           std::optional<std::vector<int>> s = std::nullopt,
           std::optional<std::vector<int>> axes = std::nullopt,
           const std::string& norm = "backward");

// STFT
array stft(const array& signal,
           int n_fft,
           int hop_length = -1,
           int win_length = -1,
           std::optional<array> window = std::nullopt,
           bool center = true,
           const std::string& pad_mode = "reflect",
           bool normalized = false,
           bool onesided = true);

array istft(const array& stft_matrix,
            int n_fft,
            int hop_length = -1,
            int win_length = -1,
            std::optional<array> window = std::nullopt,
            bool center = true,
            int length = -1);

// Helper functions
array fftshift(const array& x, std::optional<std::vector<int>> axes = std::nullopt);
array ifftshift(const array& x, std::optional<std::vector<int>> axes = std::nullopt);
array fftfreq(int n, float d = 1.0);
array rfftfreq(int n, float d = 1.0);

}  // namespace mlx::core::fft
```

### Metal Shader Optimizations

For Apple Silicon, leverage **vDSP** and **Metal Performance Shaders**:

```cpp
// mlx/backend/metal/fft.cpp

#include <Accelerate/Accelerate.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace mlx::core::metal {

// Real FFT using vDSP (Accelerate framework)
void rfft_vdsp(const array& input, array& output, int n_fft) {
  vDSP_Length log2n = (vDSP_Length)log2(n_fft);
  FFTSetup fft_setup = vDSP_create_fftsetup(log2n, kFFTRadix2);

  DSPSplitComplex split_complex;
  split_complex.realp = output.data<float>();
  split_complex.imagp = output.data<float>() + n_fft / 2;

  vDSP_fft_zrip(fft_setup, &split_complex, 1, log2n, kFFTDirection_Forward);

  vDSP_destroy_fftsetup(fft_setup);
}

// Complex FFT using Metal Performance Shaders
void fft_mps(id<MTLCommandBuffer> command_buffer,
             id<MTLBuffer> input_buffer,
             id<MTLBuffer> output_buffer,
             int n_fft) {
  MPSMatrixDescriptor* desc = [MPSMatrixDescriptor
    matrixDescriptorWithRows:n_fft
    columns:1
    rowBytes:n_fft * sizeof(float) * 2
    dataType:MPSDataTypeComplexFloat32];

  MPSMatrixCopyDescriptor* copy_desc = [[MPSMatrixCopyDescriptor alloc] init];

  // Create FFT kernel
  id<MTLDevice> device = command_buffer.device;
  MPSMatrixFindTopK* fft_kernel = [[MPSMatrixFindTopK alloc]
    initWithDevice:device
    numberOfTopKValues:n_fft];

  // Encode FFT operation
  [fft_kernel encodeToCommandBuffer:command_buffer
                        inputMatrix:input_buffer
                       outputMatrix:output_buffer];
}

}  // namespace mlx::core::metal
```

---

## Performance Considerations

### FFT Size Selection

**Optimal Sizes**: Powers of 2 are fastest (radix-2 FFT algorithm):
```python
# Fast: 2^n sizes
n_fft = 512   # 2^9 - very fast
n_fft = 1024  # 2^10 - very fast
n_fft = 2048  # 2^11 - very fast

# Slower: Prime sizes
n_fft = 1000  # Slower (factors: 2^3 × 5^3)
n_fft = 1023  # Much slower (prime)
```

**Recommendation**: Use `n_fft` that is a power of 2, or has small prime factors (2, 3, 5, 7).

### Memory Layout

**Contiguous Memory**: FFT operations are faster on contiguous tensors:
```python
x = torch.randn(100, 100).t()  # Transposed, non-contiguous
x_contiguous = x.contiguous()  # Make contiguous

# Slower
spectrum_slow = torch.fft.fft(x, dim=0)

# Faster
spectrum_fast = torch.fft.fft(x_contiguous, dim=0)
```

### Batch Processing

**Vectorization**: Process multiple signals in parallel:
```python
# Inefficient: Loop over batch
spectra = []
for signal in batch_signals:
    spectrum = torch.fft.fft(signal)
    spectra.append(spectrum)

# Efficient: Batch FFT
batch_spectrum = torch.fft.fft(batch_signals, dim=-1)  # Vectorized
```

### Normalization Mode

**Fastest Mode**: `norm=None` (no normalization on forward pass):
```python
# Fastest (no normalization)
X = torch.fft.fft(x, norm=None)

# Slightly slower (normalization on both passes)
X_ortho = torch.fft.fft(x, norm='ortho')
```

### CUDA Optimizations

**cuFFT Plan Caching**:
```python
import torch

# Set plan cache size (per device)
torch.backends.cuda.cufft_plan_cache[0].max_size = 16  # Cache up to 16 plans

# Clear plan cache if needed
torch.backends.cuda.cufft_plan_cache[0].clear()

# Example: Repeated FFTs with same size benefit from caching
for _ in range(1000):
    x = torch.randn(1024, device='cuda')
    X = torch.fft.fft(x)  # First call creates plan, subsequent calls reuse
```

### Precision Tradeoffs

**Float32 vs Float64**:
```python
# Float32: Faster, less accurate
x_f32 = torch.randn(1000, dtype=torch.float32)
X_f32 = torch.fft.fft(x_f32)  # ~2x faster than float64

# Float64: Slower, more accurate
x_f64 = torch.randn(1000, dtype=torch.float64)
X_f64 = torch.fft.fft(x_f64)  # Higher precision
```

**Float16 (CUDA only)**:
```python
# Half precision FFT (CUDA only, experimental)
x_f16 = torch.randn(1000, dtype=torch.float16, device='cuda')
X_f16 = torch.fft.fft(x_f16)  # ~4x faster, significant accuracy loss
```

---

## Common Pitfalls and Best Practices

### 1. STFT Window Selection

**Problem**: Using rectangular window causes spectral leakage.

**Solution**: Use tapered windows (Hann, Hamming, Blackman):
```python
# Bad: Rectangular window
stft_bad = torch.stft(signal, n_fft=512, window=None)  # Spectral leakage

# Good: Hann window
window = torch.hann_window(512)
stft_good = torch.stft(signal, n_fft=512, window=window)  # Reduced leakage
```

### 2. COLA Constraint for ISTFT

**Problem**: Perfect reconstruction requires COLA constraint.

**Solution**: Ensure `hop_length` satisfies COLA for chosen window:
```python
# Hann window COLA constraint: hop_length <= win_length / 2
win_length = 400
hop_length = 200  # ✓ Satisfies COLA (400 / 2)
hop_length = 100  # ✓ Also satisfies COLA (more overlap)
hop_length = 300  # ✗ Violates COLA (reconstruction artifacts)

# Verify COLA
window = torch.hann_window(win_length)
window_sq = window ** 2
overlap = window_sq[::hop_length].sum()  # Should be constant for all positions
```

### 3. DC and Nyquist Frequency

**Problem**: DC (zero frequency) and Nyquist bins have special properties.

**Solution**: Handle them correctly:
```python
# For real signals, DC and Nyquist are purely real
signal = torch.randn(100)
spectrum = torch.fft.rfft(signal)

# DC component (spectrum[0]) is purely real
assert spectrum[0].imag.abs() < 1e-6

# Nyquist component (spectrum[-1], for even-length) is purely real
if len(signal) % 2 == 0:
    assert spectrum[-1].imag.abs() < 1e-6
```

### 4. Normalization Consistency

**Problem**: Mixing normalization modes in forward/inverse pairs.

**Solution**: Use consistent normalization:
```python
# Inconsistent (wrong)
X = torch.fft.fft(x, norm='ortho')
x_recon_bad = torch.fft.ifft(X, norm='forward')  # ✗ Different norms

# Consistent (correct)
X = torch.fft.fft(x, norm='ortho')
x_recon_good = torch.fft.ifft(X, norm='ortho')  # ✓ Same norm
```

### 5. Frequency Resolution

**Problem**: FFT frequency resolution is `sample_rate / n_fft`.

**Solution**: Increase `n_fft` for better frequency resolution:
```python
sample_rate = 16000  # Hz
n_fft_low = 512      # Frequency resolution: 16000/512 = 31.25 Hz
n_fft_high = 2048    # Frequency resolution: 16000/2048 = 7.8125 Hz

# Low resolution (can't distinguish 440 Hz from 450 Hz)
spectrum_low = torch.fft.rfft(signal, n=n_fft_low)

# High resolution (can distinguish nearby frequencies)
spectrum_high = torch.fft.rfft(signal, n=n_fft_high)
```

---

## Summary

### Key Takeaways

1. **Complex vs Real FFT**: Use `rfft`/`irfft` for real signals to save 50% computation
2. **Normalization**: Choose `norm` mode consistently for forward/inverse pairs
3. **STFT**: Essential for time-frequency analysis, requires COLA constraint for perfect reconstruction
4. **Performance**: Use power-of-2 sizes, contiguous memory, batch processing
5. **MLX Gaps**: Need to implement `hfft`, `fftshift`, and `stft` manually

### API Mapping

| PyTorch | MLX | Status | Notes |
|---------|-----|--------|-------|
| `torch.fft.fft` | `mx.fft.fft` | ✅ Available | |
| `torch.fft.ifft` | `mx.fft.ifft` | ✅ Available | |
| `torch.fft.rfft` | `mx.fft.rfft` | ✅ Available | 50% faster for real signals |
| `torch.fft.irfft` | `mx.fft.irfft` | ✅ Available | |
| `torch.fft.fft2` | `mx.fft.fft2` | ✅ Available | |
| `torch.fft.fftn` | `mx.fft.fftn` | ✅ Available | |
| `torch.fft.hfft` | - | ❌ Missing | Implement using `irfft` + conjugate |
| `torch.fft.fftshift` | - | ❌ Missing | Implement using `mx.roll` |
| `torch.fft.fftfreq` | - | ❌ Missing | Implement using `mx.arange` |
| `torch.stft` | - | ❌ Missing | Implement from scratch |
| `torch.istft` | - | ❌ Missing | Overlap-add reconstruction |

### Implementation Checklist for MLX

**High Priority** (Core functionality):
- ✅ 1D/2D/ND FFT (already available)
- ❌ STFT/iSTFT (needed for audio processing)
- ❌ fftshift/ifftshift (needed for visualization)

**Medium Priority** (Convenience):
- ❌ fftfreq/rfftfreq (helper functions)
- ❌ hfft/ihfft (Hermitian FFT)

**Low Priority** (Advanced):
- ❌ cuFFT plan caching (performance optimization)
- ❌ Half-precision FFT (experimental)

---

## References

**PyTorch Source Files**:
- [aten/src/ATen/native/SpectralOps.cpp](reference/pytorch/aten/src/ATen/native/SpectralOps.cpp) - Main FFT implementation
- [torch/_refs/fft.py](reference/pytorch/torch/_refs/fft.py) - Python reference implementation
- [aten/src/ATen/native/cuda/CuFFTPlanCache.h](reference/pytorch/aten/src/ATen/native/cuda/CuFFTPlanCache.h) - cuFFT plan caching

**Algorithm References**:
- Cooley-Tukey FFT Algorithm (1965)
- Split-radix FFT for power-of-2 sizes
- Bluestein's algorithm for arbitrary sizes
- STFT: Allen & Rabiner (1977)

**Metal/Apple Documentation**:
- Accelerate framework vDSP FFT
- Metal Performance Shaders MPSMatrixFFT
