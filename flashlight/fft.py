"""
FFT Module

PyTorch-compatible torch.fft module for MLX.
Provides Fast Fourier Transform operations.
"""

from typing import Optional, Sequence, Union

import mlx.core as mx
import mlx.core.fft as mlx_fft

from .tensor import Tensor

# Re-export Tensor for compatibility with torch.fft.Tensor
Tensor = Tensor


def _to_mlx(x: Tensor) -> mx.array:
    """Convert Tensor to MLX array."""
    if isinstance(x, Tensor):
        return x._mlx_array
    return x


def _to_tensor(x: mx.array) -> Tensor:
    """Convert MLX array to Tensor."""
    return Tensor(x)


def fft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the one-dimensional discrete Fourier Transform.

    Args:
        input: Input tensor
        n: Signal length. If None, uses input.shape[dim]
        dim: Dimension along which to take the FFT
        norm: Normalization mode ('forward', 'backward', 'ortho')

    Returns:
        Complex tensor containing the FFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.fft(data, n=n, axis=dim)
    return _to_tensor(result)


def ifft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    Args:
        input: Input tensor
        n: Signal length
        dim: Dimension along which to compute IFFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the IFFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.ifft(data, n=n, axis=dim)
    return _to_tensor(result)


def fft2(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the 2-dimensional discrete Fourier Transform.

    Args:
        input: Input tensor
        s: Signal sizes for each dimension
        dim: Dimensions over which to compute the FFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the 2D FFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.fft2(data, s=s, axes=dim)
    return _to_tensor(result)


def ifft2(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.

    Args:
        input: Input tensor
        s: Signal sizes
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the 2D IFFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.ifft2(data, s=s, axes=dim)
    return _to_tensor(result)


def fftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the N-dimensional discrete Fourier Transform.

    Args:
        input: Input tensor
        s: Signal sizes for each dimension
        dim: Dimensions over which to compute the FFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the N-D FFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.fftn(data, s=s, axes=dim)
    return _to_tensor(result)


def ifftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the N-dimensional inverse discrete Fourier Transform.

    Args:
        input: Input tensor
        s: Signal sizes
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the N-D IFFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.ifftn(data, s=s, axes=dim)
    return _to_tensor(result)


def rfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the one-dimensional FFT of real input.

    Args:
        input: Real input tensor
        n: Signal length
        dim: Dimension along which to compute FFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the one-sided FFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.rfft(data, n=n, axis=dim)
    return _to_tensor(result)


def irfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the inverse FFT of real input.

    Args:
        input: Complex input tensor
        n: Output signal length
        dim: Dimension along which to compute IFFT
        norm: Normalization mode

    Returns:
        Real tensor containing the IFFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.irfft(data, n=n, axis=dim)
    return _to_tensor(result)


def rfft2(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the 2-dimensional FFT of real input.

    Args:
        input: Real input tensor
        s: Signal sizes
        dim: Dimensions over which to compute FFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the 2D one-sided FFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.rfft2(data, s=s, axes=dim)
    return _to_tensor(result)


def irfft2(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the inverse 2D FFT of real input.

    Args:
        input: Complex input tensor
        s: Output signal sizes
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode

    Returns:
        Real tensor containing the 2D IFFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.irfft2(data, s=s, axes=dim)
    return _to_tensor(result)


def rfftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the N-dimensional FFT of real input.

    Args:
        input: Real input tensor
        s: Signal sizes
        dim: Dimensions over which to compute FFT
        norm: Normalization mode

    Returns:
        Complex tensor containing the N-D one-sided FFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.rfftn(data, s=s, axes=dim)
    return _to_tensor(result)


def irfftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the inverse N-dimensional FFT of real input.

    Args:
        input: Complex input tensor
        s: Output signal sizes
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode

    Returns:
        Real tensor containing the N-D IFFT result
    """
    data = _to_mlx(input)
    result = mlx_fft.irfftn(data, s=s, axes=dim)
    return _to_tensor(result)


def fftshift(
    input: Tensor,
    dim: Optional[Union[int, Sequence[int]]] = None,
) -> Tensor:
    """
    Shift the zero-frequency component to the center of the spectrum.

    Args:
        input: Input tensor
        dim: Dimensions over which to shift

    Returns:
        Shifted tensor
    """
    data = _to_mlx(input)
    result = mlx_fft.fftshift(data, axes=dim)
    return _to_tensor(result)


def ifftshift(
    input: Tensor,
    dim: Optional[Union[int, Sequence[int]]] = None,
) -> Tensor:
    """
    Inverse of fftshift.

    Args:
        input: Input tensor
        dim: Dimensions over which to shift

    Returns:
        Shifted tensor
    """
    data = _to_mlx(input)
    result = mlx_fft.ifftshift(data, axes=dim)
    return _to_tensor(result)


def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the discrete Fourier Transform sample frequencies.

    Args:
        n: Window length
        d: Sample spacing
        dtype: Data type
        layout: Memory layout (ignored)
        device: Device (ignored in MLX)
        requires_grad: Whether to track gradients

    Returns:
        Tensor of frequencies
    """
    # Pure MLX implementation of fftfreq
    # Frequencies are: [0, 1, ..., n//2-1, -n//2, ..., -1] / (n * d) for even n
    # Frequencies are: [0, 1, ..., (n-1)//2, -(n-1)//2, ..., -1] / (n * d) for odd n
    if n % 2 == 0:
        pos = mx.arange(0, n // 2, dtype=mx.float32)
        neg = mx.arange(-n // 2, 0, dtype=mx.float32)
    else:
        pos = mx.arange(0, (n + 1) // 2, dtype=mx.float32)
        neg = mx.arange(-(n // 2), 0, dtype=mx.float32)
    freqs = mx.concatenate([pos, neg]) / (n * d)
    result = Tensor(freqs)
    if dtype is not None:
        result = result.to(dtype)
    result.requires_grad = requires_grad
    return result


def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the discrete Fourier Transform sample frequencies for rfft.

    Args:
        n: Window length
        d: Sample spacing
        dtype: Data type
        layout: Memory layout (ignored)
        device: Device (ignored in MLX)
        requires_grad: Whether to track gradients

    Returns:
        Tensor of frequencies (one-sided)
    """
    # Pure MLX implementation of rfftfreq
    # Frequencies are: [0, 1, ..., n//2] / (n * d)
    freqs = mx.arange(0, n // 2 + 1, dtype=mx.float32) / (n * d)
    result = Tensor(freqs)
    if dtype is not None:
        result = result.to(dtype)
    result.requires_grad = requires_grad
    return result


def hfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the FFT of a Hermitian symmetric signal.

    The input is assumed to be Hermitian symmetric in the frequency domain,
    so the output will be real.

    Args:
        input: Input tensor (complex)
        n: Output signal length
        dim: Dimension along which to compute FFT
        norm: Normalization mode

    Returns:
        Real tensor
    """
    # hfft(x, n) = n * irfft(conj(x), n)
    # This is the inverse of ihfft
    data = _to_mlx(input)
    conj_data = mx.conj(data)
    result = mlx_fft.irfft(conj_data, n=n, axis=dim)

    # Determine output length for scaling
    if n is None:
        n = 2 * (input.shape[dim] - 1)

    # Apply scaling: hfft includes factor of n
    result = result * n

    return _to_tensor(result)


def ihfft(
    input: Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the inverse FFT of a Hermitian symmetric signal.

    Args:
        input: Input tensor (real)
        n: Signal length
        dim: Dimension along which to compute IFFT
        norm: Normalization mode

    Returns:
        Complex tensor
    """
    # ihfft(x, n) = conj(rfft(x, n)) / n
    # This is the inverse of hfft
    data = _to_mlx(input)
    result = mlx_fft.rfft(data, n=n, axis=dim)

    # Determine input length for scaling
    if n is None:
        n = input.shape[dim]

    # Apply conjugate and scaling
    # Note: We need to materialize the conjugate to avoid the "conjugate bit" issue
    # when converting to numpy. Multiplying by 1+0j forces materialization.
    result = mx.conj(result) / n
    # Force materialization by adding zero to clear conjugate bit
    result = result + 0j

    return _to_tensor(result)


def hfft2(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the 2D FFT of a Hermitian symmetric signal.

    Args:
        input: Input tensor (complex)
        s: Output signal sizes
        dim: Dimensions over which to compute FFT
        norm: Normalization mode

    Returns:
        Real tensor
    """
    data = _to_mlx(input)
    conj_data = mx.conj(data)
    result = mlx_fft.irfft2(conj_data, s=s, axes=dim)
    return _to_tensor(result)


def ihfft2(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Sequence[int] = (-2, -1),
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the inverse 2D FFT of a Hermitian symmetric signal.

    Args:
        input: Input tensor (real)
        s: Signal sizes
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode

    Returns:
        Complex tensor
    """
    data = _to_mlx(input)
    result = mlx_fft.rfft2(data, s=s, axes=dim)
    result = mx.conj(result)
    return _to_tensor(result)


def hfftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the N-dimensional FFT of a Hermitian symmetric signal.

    Args:
        input: Input tensor (complex)
        s: Output signal sizes
        dim: Dimensions over which to compute FFT
        norm: Normalization mode

    Returns:
        Real tensor
    """
    data = _to_mlx(input)
    conj_data = mx.conj(data)
    result = mlx_fft.irfftn(conj_data, s=s, axes=dim)
    return _to_tensor(result)


def ihfftn(
    input: Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> Tensor:
    """
    Compute the inverse N-dimensional FFT of a Hermitian symmetric signal.

    Args:
        input: Input tensor (real)
        s: Signal sizes
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode

    Returns:
        Complex tensor
    """
    data = _to_mlx(input)
    result = mlx_fft.rfftn(data, s=s, axes=dim)
    result = mx.conj(result)
    return _to_tensor(result)


__all__ = [
    "Tensor",
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
]
