"""
Special Functions Module

PyTorch-compatible torch.special module for MLX.
Provides special mathematical functions.
"""

from typing import Optional, Union
import math
import mlx.core as mx

from .tensor import Tensor


def _to_mlx(x: Union[Tensor, float, int]) -> mx.array:
    """Convert Tensor or scalar to MLX array."""
    if isinstance(x, Tensor):
        return x._mlx_array
    elif isinstance(x, (int, float)):
        return mx.array(x)
    return x


def _to_tensor(x: mx.array) -> Tensor:
    """Convert MLX array to Tensor."""
    return Tensor(x)


def _to_numpy_n(n: Union[Tensor, int, 'mx.array']) -> Union[int, 'np.ndarray']:
    """Convert n (degree) to numpy array or scalar for scipy functions.

    PyTorch supports both scalar and tensor n for polynomial functions.
    Scipy requires numpy array or scalar.
    """
    import numpy as np
    if isinstance(n, Tensor):
        return np.array(n._mlx_array)
    elif isinstance(n, mx.array):
        return np.array(n)
    return n


# Error functions

def erf(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the error function.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Error function values
    """
    data = _to_mlx(input)
    result = mx.erf(data)
    return _to_tensor(result)


def erfc(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the complementary error function.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Complementary error function values (1 - erf(x))
    """
    data = _to_mlx(input)
    result = mx.array(1.0) - mx.erf(data)
    return _to_tensor(result)


def erfcx(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the scaled complementary error function.

    erfcx(x) = exp(x^2) * erfc(x)

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Scaled complementary error function values
    """
    data = _to_mlx(input)
    # erfcx(x) = exp(x^2) * erfc(x)
    result = mx.exp(data * data) * (mx.array(1.0) - mx.erf(data))
    return _to_tensor(result)


def erfinv(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the inverse error function.

    Args:
        input: Input tensor with values in (-1, 1)
        out: Output tensor (ignored in MLX)

    Returns:
        Inverse error function values
    """
    data = _to_mlx(input)
    result = mx.erfinv(data)
    return _to_tensor(result)


# Gamma functions

def gammaln(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the natural log of the absolute value of the gamma function.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Log-gamma values
    """
    data = _to_mlx(input)
    # MLX doesn't have gammaln directly, use approximation
    # For positive values: lgamma(x) = log(gamma(x))
    # Using Stirling's approximation for large x, and reflection for negative
    import numpy as np
    np_result = np.vectorize(math.lgamma)(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def digamma(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the logarithmic derivative of the gamma function.

    Also known as psi function.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Digamma function values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.digamma(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


# Alias for digamma
psi = digamma


def polygamma(n: int, input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the n-th derivative of the digamma function.

    Args:
        n: Order of the derivative
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Polygamma function values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.polygamma(n, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def multigammaln(input: Tensor, p: int, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the log of the multivariate gamma function.

    Args:
        input: Input tensor
        p: Dimension
        out: Output tensor (ignored in MLX)

    Returns:
        Log multivariate gamma values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.multigammaln(np.array(data), p)
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def gammainc(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the regularized lower incomplete gamma function.

    Args:
        input: Non-negative input tensor (a parameter)
        other: Non-negative input tensor (x parameter)
        out: Output tensor (ignored in MLX)

    Returns:
        Lower incomplete gamma values
    """
    import numpy as np
    from scipy import special as sp
    a = _to_mlx(input)
    x = _to_mlx(other)
    np_result = sp.gammainc(np.array(a), np.array(x))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def gammaincc(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the regularized upper incomplete gamma function.

    Args:
        input: Non-negative input tensor (a parameter)
        other: Non-negative input tensor (x parameter)
        out: Output tensor (ignored in MLX)

    Returns:
        Upper incomplete gamma values
    """
    import numpy as np
    from scipy import special as sp
    a = _to_mlx(input)
    x = _to_mlx(other)
    np_result = sp.gammaincc(np.array(a), np.array(x))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


# Bessel functions

def bessel_j0(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Bessel function of the first kind of order 0.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Bessel J0 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.j0(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def bessel_j1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Bessel function of the first kind of order 1.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Bessel J1 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.j1(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def bessel_y0(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Bessel function of the second kind of order 0.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Bessel Y0 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.y0(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def bessel_y1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Bessel function of the second kind of order 1.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Bessel Y1 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.y1(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def modified_bessel_i0(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Modified Bessel function of the first kind of order 0.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Modified Bessel I0 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.i0(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def modified_bessel_i1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Modified Bessel function of the first kind of order 1.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Modified Bessel I1 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.i1(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def modified_bessel_k0(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Modified Bessel function of the second kind of order 0.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Modified Bessel K0 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.k0(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def modified_bessel_k1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Modified Bessel function of the second kind of order 1.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Modified Bessel K1 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.k1(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def scaled_modified_bessel_k0(input: Tensor = None, *, x: Tensor = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Scaled modified Bessel function of the second kind of order 0.

    Returns exp(x) * K0(x).

    Args:
        input: Input tensor (also accepts 'x' for PyTorch compatibility)
        x: Input tensor (alternative to 'input')
        out: Output tensor (ignored in MLX)

    Returns:
        Scaled modified Bessel K0 values
    """
    import numpy as np
    from scipy import special as sp
    # Support both 'input' and 'x' parameter names
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("scaled_modified_bessel_k0() missing required argument: 'input' or 'x'")
    data = _to_mlx(tensor)
    np_result = sp.k0e(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def scaled_modified_bessel_k1(input: Tensor = None, *, x: Tensor = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Scaled modified Bessel function of the second kind of order 1.

    Returns exp(x) * K1(x).

    Args:
        input: Input tensor (also accepts 'x' for PyTorch compatibility)
        x: Input tensor (alternative to 'input')
        out: Output tensor (ignored in MLX)

    Returns:
        Scaled modified Bessel K1 values
    """
    import numpy as np
    from scipy import special as sp
    # Support both 'input' and 'x' parameter names
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("scaled_modified_bessel_k1() missing required argument: 'input' or 'x'")
    data = _to_mlx(tensor)
    np_result = sp.k1e(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def spherical_bessel_j0(input: Tensor = None, *, x: Tensor = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Spherical Bessel function of the first kind of order 0.

    Args:
        input: Input tensor (also accepts 'x' for PyTorch compatibility)
        x: Input tensor (alternative to 'input')
        out: Output tensor (ignored in MLX)

    Returns:
        Spherical Bessel j0 values
    """
    import numpy as np
    from scipy import special as sp
    # Support both 'input' and 'x' parameter names
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("spherical_bessel_j0() missing required argument: 'input' or 'x'")
    data = _to_mlx(tensor)
    np_result = sp.spherical_jn(0, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def i0(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Modified Bessel function of the first kind of order 0.

    Alias for modified_bessel_i0.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        I0 values
    """
    return modified_bessel_i0(input, out=out)


def i0e(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Exponentially scaled modified Bessel function of the first kind of order 0.

    Returns exp(-|x|) * I0(x).

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Scaled I0 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.i0e(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def i1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Modified Bessel function of the first kind of order 1.

    Alias for modified_bessel_i1.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        I1 values
    """
    return modified_bessel_i1(input, out=out)


def i1e(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Exponentially scaled modified Bessel function of the first kind of order 1.

    Returns exp(-|x|) * I1(x).

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Scaled I1 values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.i1e(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


# Exponential and logarithmic functions

def exp2(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute 2^x element-wise.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        2^x values
    """
    data = _to_mlx(input)
    result = mx.power(mx.array(2.0), data)
    return _to_tensor(result)


def expm1(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute exp(x) - 1 element-wise.

    More accurate than exp(x) - 1 for small x.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        exp(x) - 1 values
    """
    data = _to_mlx(input)
    result = mx.expm1(data)
    return _to_tensor(result)


def log1p(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute log(1 + x) element-wise.

    More accurate than log(1 + x) for small x.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        log(1 + x) values
    """
    data = _to_mlx(input)
    result = mx.log1p(data)
    return _to_tensor(result)


def xlog1py(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute x * log1p(y) element-wise.

    Args:
        input: First input tensor (x)
        other: Second input tensor (y)
        out: Output tensor (ignored in MLX)

    Returns:
        x * log1p(y) values
    """
    x = _to_mlx(input)
    y = _to_mlx(other)
    result = x * mx.log1p(y)
    return _to_tensor(result)


def xlogy(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute x * log(y) element-wise.

    Returns 0 when x == 0.

    Args:
        input: First input tensor (x)
        other: Second input tensor (y)
        out: Output tensor (ignored in MLX)

    Returns:
        x * log(y) values (0 where x == 0)
    """
    x = _to_mlx(input)
    y = _to_mlx(other)
    result = mx.where(x == 0, mx.array(0.0), x * mx.log(y))
    return _to_tensor(result)


# Logit and sigmoid functions

def expit(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the sigmoid (expit) function.

    expit(x) = 1 / (1 + exp(-x))

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Sigmoid values
    """
    data = _to_mlx(input)
    result = mx.sigmoid(data)
    return _to_tensor(result)


def logit(input: Tensor, eps: Optional[float] = None, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the logit function.

    logit(x) = log(x / (1 - x))

    Args:
        input: Input tensor with values in (0, 1)
        eps: Small epsilon for numerical stability
        out: Output tensor (ignored in MLX)

    Returns:
        Logit values
    """
    data = _to_mlx(input)
    if eps is not None:
        data = mx.clip(data, eps, 1 - eps)
    result = mx.log(data / (mx.array(1.0) - data))
    return _to_tensor(result)


# Softmax functions

def softmax(input: Tensor, dim: int, *, dtype=None) -> Tensor:
    """
    Compute softmax.

    Args:
        input: Input tensor
        dim: Dimension along which to compute softmax
        dtype: Output dtype

    Returns:
        Softmax values
    """
    data = _to_mlx(input)
    result = mx.softmax(data, axis=dim)
    return _to_tensor(result)


def log_softmax(input: Tensor, dim: int, *, dtype=None) -> Tensor:
    """
    Compute log softmax.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log softmax
        dtype: Output dtype

    Returns:
        Log softmax values
    """
    data = _to_mlx(input)
    # log_softmax = x - logsumexp(x)
    max_val = mx.max(data, axis=dim, keepdims=True)
    exp_data = mx.exp(data - max_val)
    sum_exp = mx.sum(exp_data, axis=dim, keepdims=True)
    result = data - max_val - mx.log(sum_exp)
    return _to_tensor(result)


def logsumexp(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """
    Compute the log of the sum of exponentials.

    Args:
        input: Input tensor
        dim: Dimension along which to compute
        keepdim: Whether to keep the reduced dimension

    Returns:
        Log-sum-exp values
    """
    data = _to_mlx(input)
    result = mx.logsumexp(data, axis=dim, keepdims=keepdim)
    return _to_tensor(result)


# Normal distribution functions

def ndtr(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the cumulative distribution function of the standard normal.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        CDF values
    """
    data = _to_mlx(input)
    # ndtr(x) = 0.5 * (1 + erf(x / sqrt(2)))
    result = mx.array(0.5) * (mx.array(1.0) + mx.erf(data / mx.sqrt(mx.array(2.0))))
    return _to_tensor(result)


def ndtri(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the inverse of the CDF of the standard normal.

    Args:
        input: Input tensor with values in (0, 1)
        out: Output tensor (ignored in MLX)

    Returns:
        Inverse CDF values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.ndtri(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def log_ndtr(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the log of the CDF of the standard normal.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Log CDF values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    np_result = sp.log_ndtr(np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


# Entropy

def entr(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the entropy function: -x * log(x).

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Entropy values
    """
    data = _to_mlx(input)
    result = mx.where(data > 0, -data * mx.log(data), mx.array(0.0))
    return _to_tensor(result)


# Sinc function

def sinc(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the normalized sinc function: sin(pi * x) / (pi * x).

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Sinc values
    """
    data = _to_mlx(input)
    pi_x = mx.array(math.pi) * data
    # sinc(0) = 1
    result = mx.where(data == 0, mx.array(1.0), mx.sin(pi_x) / pi_x)
    return _to_tensor(result)


# Round function

def round(input: Tensor, decimals: int = 0, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Round elements to the given number of decimals.

    Args:
        input: Input tensor
        decimals: Number of decimal places to round to
        out: Output tensor (ignored in MLX)

    Returns:
        Rounded values
    """
    data = _to_mlx(input)
    if decimals == 0:
        result = mx.round(data)
    else:
        factor = 10.0 ** decimals
        result = mx.round(data * factor) / factor
    return _to_tensor(result)


# Zeta function

def zeta(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the Hurwitz zeta function.

    Args:
        input: First input tensor
        other: Second input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Zeta function values
    """
    import numpy as np
    from scipy import special as sp
    x = _to_mlx(input)
    q = _to_mlx(other)
    np_result = sp.zeta(np.array(x), np.array(q))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


# Airy function

def airy_ai(input: Tensor = None, *, x: Tensor = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the Airy function Ai.

    Args:
        input: Input tensor (also accepts 'x' for PyTorch compatibility)
        x: Input tensor (alternative to 'input')
        out: Output tensor (ignored in MLX)

    Returns:
        Airy Ai values
    """
    import numpy as np
    from scipy import special as sp
    # Support both 'input' and 'x' parameter names
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("airy_ai() missing required argument: 'input' or 'x'")
    data = _to_mlx(tensor)
    ai, _, _, _ = sp.airy(np.array(data))
    result = mx.array(ai.astype(np.float32))
    return _to_tensor(result)


# Polynomial functions (Chebyshev, Hermite, Laguerre, Legendre)

def chebyshev_polynomial_t(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Chebyshev polynomial of the first kind.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Chebyshev T_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_chebyt(n_np, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def chebyshev_polynomial_u(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Chebyshev polynomial of the second kind.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Chebyshev U_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_chebyu(n_np, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def chebyshev_polynomial_v(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Chebyshev polynomial of the third kind.

    V_n: V_0 = 1, V_1 = 2x - 1, V_{n+1} = 2x * V_n - V_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Chebyshev V_n values
    """
    import numpy as np
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_x = np.array(data).astype(np.float64)
    original_shape = np_x.shape

    # Flatten both x and n for element-wise processing
    np_x_flat = np_x.flatten()

    if np.isscalar(n_np):
        # Scalar n - apply same degree to all x values
        n_val = int(n_np)
        if n_val < 0:
            np_result = np.zeros_like(np_x_flat)
        elif n_val == 0:
            np_result = np.ones_like(np_x_flat)
        elif n_val == 1:
            np_result = 2.0 * np_x_flat - 1.0
        else:
            v_prev = np.ones_like(np_x_flat)
            v_curr = 2.0 * np_x_flat - 1.0
            for k in range(2, n_val + 1):
                v_next = 2.0 * np_x_flat * v_curr - v_prev
                v_prev = v_curr
                v_curr = v_next
            np_result = v_curr
    else:
        # Array n - compute element-wise
        n_arr = np.array(n_np).flatten().astype(int)
        np_result = np.zeros_like(np_x_flat)

        for i in range(len(np_x_flat)):
            n_val = n_arr[i]
            x_val = np_x_flat[i]
            if n_val < 0:
                continue
            elif n_val == 0:
                np_result[i] = 1.0
            elif n_val == 1:
                np_result[i] = 2.0 * x_val - 1.0
            else:
                v_prev = 1.0
                v_curr = 2.0 * x_val - 1.0
                for k in range(2, n_val + 1):
                    v_next = 2.0 * x_val * v_curr - v_prev
                    v_prev = v_curr
                    v_curr = v_next
                np_result[i] = v_curr

    np_result = np_result.reshape(original_shape)
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def chebyshev_polynomial_w(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Chebyshev polynomial of the fourth kind.

    W_n: W_0 = 1, W_1 = 2x + 1, W_{n+1} = 2x * W_n - W_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Chebyshev W_n values
    """
    import numpy as np
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_x = np.array(data).astype(np.float64)
    original_shape = np_x.shape

    # Flatten both x and n for element-wise processing
    np_x_flat = np_x.flatten()

    if np.isscalar(n_np):
        # Scalar n - apply same degree to all x values
        n_val = int(n_np)
        if n_val < 0:
            np_result = np.zeros_like(np_x_flat)
        elif n_val == 0:
            np_result = np.ones_like(np_x_flat)
        elif n_val == 1:
            np_result = 2.0 * np_x_flat + 1.0
        else:
            w_prev = np.ones_like(np_x_flat)
            w_curr = 2.0 * np_x_flat + 1.0
            for k in range(2, n_val + 1):
                w_next = 2.0 * np_x_flat * w_curr - w_prev
                w_prev = w_curr
                w_curr = w_next
            np_result = w_curr
    else:
        # Array n - compute element-wise
        n_arr = np.array(n_np).flatten().astype(int)
        np_result = np.zeros_like(np_x_flat)

        for i in range(len(np_x_flat)):
            n_val = n_arr[i]
            x_val = np_x_flat[i]
            if n_val < 0:
                continue
            elif n_val == 0:
                np_result[i] = 1.0
            elif n_val == 1:
                np_result[i] = 2.0 * x_val + 1.0
            else:
                w_prev = 1.0
                w_curr = 2.0 * x_val + 1.0
                for k in range(2, n_val + 1):
                    w_next = 2.0 * x_val * w_curr - w_prev
                    w_prev = w_curr
                    w_curr = w_next
                np_result[i] = w_curr

    np_result = np_result.reshape(original_shape)
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def shifted_chebyshev_polynomial_t(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the first kind.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev T*_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    # T*_n(x) = T_n(2x - 1)
    np_result = sp.eval_chebyt(n_np, 2 * np.array(data) - 1)
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def shifted_chebyshev_polynomial_u(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the second kind.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev U*_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_chebyu(n_np, 2 * np.array(data) - 1)
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def shifted_chebyshev_polynomial_v(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the third kind.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev V*_n values
    """
    # Fix: Shifted Chebyshev V*_n(x) = V_n(2x - 1)
    # Transform input from [0,1] to [-1,1] and use regular V polynomial
    shifted_input = Tensor(2 * _to_mlx(input) - 1)
    return chebyshev_polynomial_v(shifted_input, n, out=out)


def shifted_chebyshev_polynomial_w(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the fourth kind.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev W*_n values
    """
    # Fix: Shifted Chebyshev W*_n(x) = W_n(2x - 1)
    # Transform input from [0,1] to [-1,1] and use regular W polynomial
    shifted_input = Tensor(2 * _to_mlx(input) - 1)
    return chebyshev_polynomial_w(shifted_input, n, out=out)


def hermite_polynomial_h(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Physicist's Hermite polynomial.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Hermite H_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_hermite(n_np, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def hermite_polynomial_he(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Probabilist's Hermite polynomial.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Hermite He_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_hermitenorm(n_np, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def laguerre_polynomial_l(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Laguerre polynomial.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Laguerre L_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_laguerre(n_np, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


def legendre_polynomial_p(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Legendre polynomial.

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Legendre P_n values
    """
    import numpy as np
    from scipy import special as sp
    data = _to_mlx(input)
    n_np = _to_numpy_n(n)
    np_result = sp.eval_legendre(n_np, np.array(data))
    result = mx.array(np_result.astype(np.float32))
    return _to_tensor(result)


__all__ = [
    # Error functions
    'erf',
    'erfc',
    'erfcx',
    'erfinv',
    # Gamma functions
    'gammaln',
    'digamma',
    'psi',
    'polygamma',
    'multigammaln',
    'gammainc',
    'gammaincc',
    # Bessel functions
    'bessel_j0',
    'bessel_j1',
    'bessel_y0',
    'bessel_y1',
    'modified_bessel_i0',
    'modified_bessel_i1',
    'modified_bessel_k0',
    'modified_bessel_k1',
    'scaled_modified_bessel_k0',
    'scaled_modified_bessel_k1',
    'spherical_bessel_j0',
    'i0',
    'i0e',
    'i1',
    'i1e',
    # Exponential/logarithmic
    'exp2',
    'expm1',
    'log1p',
    'xlog1py',
    'xlogy',
    # Logit/sigmoid
    'expit',
    'logit',
    # Softmax
    'softmax',
    'log_softmax',
    'logsumexp',
    # Normal distribution
    'ndtr',
    'ndtri',
    'log_ndtr',
    # Other
    'entr',
    'sinc',
    'round',
    'zeta',
    'airy_ai',
    # Polynomials
    'chebyshev_polynomial_t',
    'chebyshev_polynomial_u',
    'chebyshev_polynomial_v',
    'chebyshev_polynomial_w',
    'shifted_chebyshev_polynomial_t',
    'shifted_chebyshev_polynomial_u',
    'shifted_chebyshev_polynomial_v',
    'shifted_chebyshev_polynomial_w',
    'hermite_polynomial_h',
    'hermite_polynomial_he',
    'laguerre_polynomial_l',
    'legendre_polynomial_p',
]
