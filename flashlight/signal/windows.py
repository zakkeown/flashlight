"""
Window Functions Module

PyTorch-compatible torch.signal.windows module for MLX.
Provides various window functions for signal processing.
"""

from typing import Optional, Iterable
import math

import mlx.core as mx

from ..tensor import Tensor
from ..dtype import get_dtype


def _to_mlx_dtype(dtype):
    """Convert dtype to MLX dtype."""
    if dtype is None:
        return mx.float32
    dt = get_dtype(dtype)
    return dt._mlx_dtype if dt else mx.float32


def bartlett(
    M: int,
    *,
    sym: bool = True,
    dtype=None,
    layout=None,  # Ignored - MLX uses strided
    device=None,  # Ignored - MLX uses unified memory
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the Bartlett window.

    The Bartlett window is a triangular window.

    Args:
        M: Number of points in the window
        sym: If True, generates a symmetric window for filter design.
             If False, generates a periodic window for spectral analysis.
        dtype: The desired data type
        layout: Ignored (for PyTorch compatibility)
        device: Ignored (for PyTorch compatibility)
        requires_grad: If True, the resulting tensor requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    n = M if sym else M + 1
    half = (n - 1) / 2.0

    indices = mx.arange(n, dtype=mx.float32)
    window = 1.0 - mx.abs((indices - half) / half)

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def blackman(
    M: int,
    *,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the Blackman window.

    Args:
        M: Number of points in the window
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    n = M if sym else M + 1
    indices = mx.arange(n, dtype=mx.float32)

    a0, a1, a2 = 0.42, 0.5, 0.08
    window = a0 - a1 * mx.cos(2 * math.pi * indices / (n - 1)) + a2 * mx.cos(4 * math.pi * indices / (n - 1))

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def cosine(
    M: int,
    *,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the cosine window (also known as sine window).

    Args:
        M: Number of points in the window
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    # Fix: PyTorch uses sin(pi*(n+0.5)/M) formula, not sin(pi*n/(M-1))
    # This gives a window that starts and ends with non-zero values
    n = M if sym else M + 1
    indices = mx.arange(n, dtype=mx.float32)

    window = mx.sin(math.pi * (indices + 0.5) / n)

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def exponential(
    M: int,
    *,
    center: Optional[float] = None,
    tau: float = 1.0,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the exponential (or Poisson) window.

    Args:
        M: Number of points in the window
        center: Center of the window. If None, defaults to (M-1)/2 for symmetric
                and M/2 for periodic windows.
        tau: Decay parameter
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    n = M if sym else M + 1

    if center is None:
        # PyTorch uses (M-1)/2 for symmetric and M/2 for periodic
        center = (M - 1) / 2.0 if sym else M / 2.0

    indices = mx.arange(n, dtype=mx.float32)
    window = mx.exp(-mx.abs(indices - center) / tau)

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def gaussian(
    M: int,
    *,
    std: float = 1.0,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the Gaussian window.

    Args:
        M: Number of points in the window
        std: Standard deviation of the Gaussian
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    n = M if sym else M + 1
    center = (n - 1) / 2.0

    indices = mx.arange(n, dtype=mx.float32)
    window = mx.exp(-0.5 * ((indices - center) / std) ** 2)

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def general_cosine(
    M: int,
    *,
    a: Iterable,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute a generic weighted sum of cosine terms window.

    Args:
        M: Number of points in the window
        a: Sequence of weighting coefficients
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    a = list(a)
    n = M if sym else M + 1
    indices = mx.arange(n, dtype=mx.float32)

    window = mx.zeros((n,), dtype=mx.float32)
    for k, coef in enumerate(a):
        sign = 1 if k % 2 == 0 else -1
        window = window + sign * coef * mx.cos(2 * math.pi * k * indices / (n - 1))

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def general_hamming(
    M: int,
    *,
    alpha: float = 0.54,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the generalized Hamming window.

    Args:
        M: Number of points in the window
        alpha: Window coefficient (0.54 for Hamming, 0.5 for Hann)
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    return general_cosine(
        M,
        a=[alpha, 1 - alpha],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def hamming(
    M: int,
    *,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the Hamming window.

    Args:
        M: Number of points in the window
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    return general_hamming(
        M,
        alpha=0.54,
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def hann(
    M: int,
    *,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the Hann window (also known as Hanning).

    Args:
        M: Number of points in the window
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    return general_hamming(
        M,
        alpha=0.5,
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


def kaiser(
    M: int,
    *,
    beta: float = 12.0,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Args:
        M: Number of points in the window
        beta: Shape parameter for the window. Determines the trade-off between
              main-lobe width and side-lobe level. Higher beta means narrower
              main lobe but higher side lobes.
        sym: If True, generates a symmetric window for filter design.
             If False, generates a periodic window for spectral analysis.
        dtype: The desired data type
        layout: Ignored (for PyTorch compatibility)
        device: Ignored (for PyTorch compatibility)
        requires_grad: If True, the resulting tensor requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    if M <= 0:
        return Tensor(mx.array([], dtype=_to_mlx_dtype(dtype)))
    if M == 1:
        return Tensor(mx.ones((1,), dtype=_to_mlx_dtype(dtype)))

    n = M if sym else M + 1

    # Pure MLX implementation using Bessel I0 approximation
    indices = mx.arange(n, dtype=mx.float32)
    alpha = (n - 1) / 2.0
    ratio = (indices - alpha) / alpha
    # Clamp to avoid sqrt of negative due to numerical precision
    arg = beta * mx.sqrt(mx.maximum(1 - ratio ** 2, mx.array(0.0)))

    # I0 approximation via series expansion
    window = _i0_approx(arg) / _i0_approx(mx.array(beta))

    if not sym:
        window = window[:-1]

    result = Tensor(window.astype(_to_mlx_dtype(dtype)))
    result.requires_grad = requires_grad
    return result


def _i0_approx(x):
    """Approximate modified Bessel function I0 using series expansion."""
    # Use more terms for better accuracy
    result = mx.ones_like(x)
    term = mx.ones_like(x)
    for k in range(1, 25):
        term = term * (x / (2 * k)) ** 2
        result = result + term
    return result


def nuttall(
    M: int,
    *,
    sym: bool = True,
    dtype=None,
    layout=None,
    device=None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Compute the minimum 4-term Blackman-Harris window (Nuttall window).

    Args:
        M: Number of points in the window
        sym: If True, generates a symmetric window
        dtype: The desired data type
        layout: Ignored
        device: Ignored
        requires_grad: If True, requires gradient

    Returns:
        A 1-D tensor of size M containing the window
    """
    return general_cosine(
        M,
        a=[0.3635819, 0.4891775, 0.1365995, 0.0106411],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


__all__ = [
    'bartlett',
    'blackman',
    'cosine',
    'exponential',
    'gaussian',
    'general_cosine',
    'general_hamming',
    'hamming',
    'hann',
    'kaiser',
    'nuttall',
]
