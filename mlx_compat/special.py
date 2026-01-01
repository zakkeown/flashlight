"""
Special Functions Module

PyTorch-compatible torch.special module for MLX.
Provides special mathematical functions implemented in pure MLX.
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


# =============================================================================
# Error functions
# =============================================================================

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

    Uses asymptotic expansion for large positive x to avoid precision loss
    from computing 1 - erf(x) when erf(x) is very close to 1.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Complementary error function values (1 - erf(x))
    """
    data = _to_mlx(input)

    # For small |x|, direct computation works well
    # For large positive x, use asymptotic: erfc(x) ~ exp(-x^2) / (x * sqrt(pi)) * series
    # For large negative x, use erfc(-x) = 2 - erfc(x)
    # Threshold of 3.5 is chosen because MLX's erf saturates to 1.0 around x=4
    # causing 1-erf(x) to return 0. Asymptotic is accurate for x > 3.
    threshold = 3.5

    # Direct computation for moderate values
    direct_result = mx.array(1.0) - mx.erf(data)

    # Asymptotic expansion for large positive x
    # erfc(x) ~ exp(-x^2) / (sqrt(pi) * x) * (1 - 1/(2x^2) + 3/(2x^2)^2 - ...)
    x = mx.maximum(mx.abs(data), 1e-10)  # Avoid division by zero
    x2 = x * x
    inv_2x2 = 0.5 / x2

    # Asymptotic series coefficients: (-1)^n * (2n-1)!! / (2x^2)^n
    inv_sqrt_pi = 0.5641895835477563  # 1/sqrt(pi)
    series = (1.0
              - inv_2x2 * (1.0
              - inv_2x2 * (3.0
              - inv_2x2 * (15.0
              - inv_2x2 * (105.0
              - inv_2x2 * (945.0
              - inv_2x2 * 10395.0))))))

    # erfc(x) = exp(-x^2) * (1/(sqrt(pi)*x)) * series
    asymptotic_result = mx.exp(-x2) * inv_sqrt_pi / x * series

    # Use asymptotic only for large positive x
    use_asymptotic = (data > threshold)
    result = mx.where(use_asymptotic, asymptotic_result, direct_result)

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

    # For small |x|, use direct computation
    # For large positive x, use asymptotic expansion to avoid overflow
    # For negative x, use reflection: erfcx(-x) = 2*exp(x^2) - erfcx(x)

    # Threshold where asymptotic expansion becomes accurate (|x| > 4)
    threshold = 4.0
    small_x = mx.abs(data) < threshold

    # Direct computation for small |x| - this works well in the range [-4, 4]
    direct_result = mx.exp(data * data) * (mx.array(1.0) - mx.erf(data))

    # Asymptotic expansion for large positive x
    # erfcx(x) ~ 1/(sqrt(pi) * x) * sum_{n=0}^{N} (-1)^n * (2n-1)!! / (2*x^2)^n
    # Coefficients: 1, -1/2, 3/4, -15/8, 105/16, -945/32, 10395/64, ...
    x = mx.abs(data)
    x2 = x * x
    inv_2x2 = 0.5 / x2  # 1/(2*x^2)

    # More terms in the asymptotic series for better accuracy
    # (2n-1)!! = 1, 1, 3, 15, 105, 945, 10395, 135135, ...
    inv_sqrt_pi = 0.5641895835477563  # 1/sqrt(pi)
    series = (1.0
              - inv_2x2 * (1.0
              - inv_2x2 * (3.0
              - inv_2x2 * (15.0
              - inv_2x2 * (105.0
              - inv_2x2 * (945.0
              - inv_2x2 * 10395.0))))))
    asymptotic_result = inv_sqrt_pi / x * series

    result = mx.where(small_x, direct_result, asymptotic_result)

    # For negative x, use the direct computation which handles this correctly
    # (erfc(-x) = 2 - erfc(x), and exp(x^2) grows, but the product is stable)
    result = mx.where(data < 0, direct_result, result)

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


# =============================================================================
# Gamma functions - Pure MLX implementations
# =============================================================================

def _gammaln_mlx(x: mx.array) -> mx.array:
    """
    Pure MLX implementation of log-gamma function using Lanczos approximation.

    Uses the Lanczos approximation with g=7 coefficients for good accuracy.
    """
    # Lanczos approximation coefficients (g=7)
    g = 7.0
    coeffs = mx.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ], dtype=mx.float32)

    # Handle reflection for x < 0.5 using: gamma(x) * gamma(1-x) = pi / sin(pi*x)
    # So: lgamma(x) = log(pi) - log(sin(pi*x)) - lgamma(1-x)
    needs_reflection = x < 0.5

    # For the main computation, use z = x - 1 for x >= 0.5, or z = -x for x < 0.5
    z = mx.where(needs_reflection, -x, x - 1.0)

    # Lanczos series
    ag = coeffs[0]
    for i in range(1, 9):
        ag = ag + coeffs[i] / (z + float(i))

    # log(gamma(z+1)) = 0.5*log(2*pi) + (z+0.5)*log(z+g+0.5) - (z+g+0.5) + log(ag)
    zgh = z + g + 0.5
    log_gamma = 0.5 * math.log(2 * math.pi) + (z + 0.5) * mx.log(zgh) - zgh + mx.log(ag)

    # Apply reflection formula for x < 0.5
    # lgamma(x) = log(pi) - log(|sin(pi*x)|) - lgamma(1-x)
    sin_pi_x = mx.sin(mx.array(math.pi) * x)
    reflected = mx.array(math.log(math.pi)) - mx.log(mx.abs(sin_pi_x)) - log_gamma

    result = mx.where(needs_reflection, reflected, log_gamma)
    return result


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
    result = _gammaln_mlx(data)
    return _to_tensor(result)


def _digamma_mlx(x: mx.array) -> mx.array:
    """
    Pure MLX implementation of digamma function.

    Uses asymptotic expansion for large x and recurrence relation for small x.
    psi(x) = psi(x+n) - sum_{k=0}^{n-1} 1/(x+k) for shifting to large x region.
    """
    # Shift x to be >= 10 for better convergence of asymptotic expansion
    # Higher threshold gives better accuracy (was 8, now 10)
    min_x = 10.0

    # Calculate how much we need to shift
    n = mx.maximum(mx.ceil(min_x - x), mx.zeros_like(x))
    n = n.astype(mx.int32)
    x_shifted = x + n.astype(mx.float32)

    # Asymptotic expansion for large x:
    # psi(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{N} B_{2k}/(2k * x^{2k})
    # where B_{2k} are Bernoulli numbers:
    # B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730, B14=7/6
    inv_x = 1.0 / x_shifted
    inv_x2 = inv_x * inv_x

    # Extended asymptotic expansion with 7 Bernoulli terms for improved accuracy
    # Coefficients: B_{2k}/(2k) for k=1..7
    # B2/2=1/12, B4/4=-1/120, B6/6=1/252, B8/8=-1/240, B10/10=1/132,
    # B12/12=-691/32760, B14/14=1/12
    result = mx.log(x_shifted) - 0.5 * inv_x
    result = result - inv_x2 * (
        1.0/12.0
        - inv_x2 * (1.0/120.0
        - inv_x2 * (1.0/252.0
        - inv_x2 * (1.0/240.0
        - inv_x2 * (1.0/132.0
        - inv_x2 * (691.0/32760.0
        - inv_x2 * (1.0/12.0))))))
    )

    # Apply recurrence relation to shift back: psi(x) = psi(x+n) - sum 1/(x+k)
    # We need to subtract sum_{k=0}^{n-1} 1/(x+k)
    max_n = 20  # Maximum shift we support
    for k in range(max_n):
        k_float = mx.array(float(k), dtype=mx.float32)
        should_subtract = k_float < n.astype(mx.float32)
        term = mx.where(should_subtract, 1.0 / (x + k_float), mx.zeros_like(x))
        result = result - term

    # Handle x <= 0 (poles at non-positive integers)
    # psi has poles at 0, -1, -2, ... ; return NaN there
    is_nonpositive_int = (x <= 0) & (mx.abs(x - mx.round(x)) < 1e-10)
    result = mx.where(is_nonpositive_int, mx.array(float('nan')), result)

    return result


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
    data = _to_mlx(input)
    result = _digamma_mlx(data)
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
    data = _to_mlx(input)

    if n == 0:
        return digamma(input, out=out)

    # For n >= 1, use the series representation:
    # psi^(n)(x) = (-1)^(n+1) * n! * sum_{k=0}^{inf} 1/(x+k)^(n+1)

    # Shift x to be large enough for convergence
    min_x = 10.0 + n  # Need larger x for higher derivatives
    shift = mx.maximum(mx.ceil(min_x - data), mx.zeros_like(data))
    shift = shift.astype(mx.int32)
    x_shifted = data + shift.astype(mx.float32)

    # Compute asymptotic expansion for polygamma
    # psi^(n)(x) ~ (-1)^(n+1) * [(n-1)!/x^n + n!/(2*x^(n+1)) + sum of Bernoulli terms]
    sign = (-1.0) ** (n + 1)

    # Factorial
    factorial_n = 1.0
    for i in range(1, n + 1):
        factorial_n *= i
    factorial_n_minus_1 = factorial_n / n if n > 0 else 1.0

    inv_x = 1.0 / x_shifted
    inv_x_pow = mx.power(inv_x, float(n))

    # Leading terms of asymptotic expansion
    result = sign * factorial_n_minus_1 * inv_x_pow
    result = result + sign * factorial_n * 0.5 * inv_x_pow * inv_x

    # Add correction terms (Bernoulli terms)
    inv_x2 = inv_x * inv_x
    if n >= 1:
        # B_2 / 2! * (n+1)! / x^(n+2)
        coeff = factorial_n * (n + 1) / 12.0
        result = result + sign * coeff * inv_x_pow * inv_x2

    # Apply recurrence to shift back
    # psi^(n)(x) = psi^(n)(x+m) + (-1)^(n+1) * n! * sum_{k=0}^{m-1} 1/(x+k)^(n+1)
    max_shift = 30
    for k in range(max_shift):
        k_float = mx.array(float(k), dtype=mx.float32)
        should_add = k_float < shift.astype(mx.float32)
        term = sign * factorial_n / mx.power(data + k_float, float(n + 1))
        term = mx.where(should_add, term, mx.zeros_like(data))
        result = result + term

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
    data = _to_mlx(input)

    # Multivariate gamma: Gamma_p(a) = pi^(p(p-1)/4) * prod_{j=1}^{p} Gamma(a + (1-j)/2)
    # log form: log(Gamma_p(a)) = p(p-1)/4 * log(pi) + sum_{j=1}^{p} lgamma(a + (1-j)/2)

    result = mx.array(p * (p - 1) / 4.0 * math.log(math.pi))

    for j in range(1, p + 1):
        result = result + _gammaln_mlx(data + (1.0 - j) / 2.0)

    return _to_tensor(result)


def _gammainc_series(a: mx.array, x: mx.array, max_iter: int = 300) -> mx.array:
    """
    Compute lower incomplete gamma using series expansion.
    P(a,x) = (x^a * e^-x / Gamma(a)) * sum_{n=0}^{inf} x^n / (a+1)...(a+n)
    Good for x < a + 1.

    Increased max_iter from 200 to 300 for better convergence with large a.
    """
    # Series: sum_{n=0}^{inf} x^n / [(a+1)(a+2)...(a+n)]
    term = mx.ones_like(x)
    sum_val = term

    for n in range(1, max_iter):
        term = term * x / (a + float(n))
        sum_val = sum_val + term

    # P(a,x) = x^a * exp(-x) * sum / Gamma(a+1)
    # But Gamma(a+1) = a * Gamma(a), so we use exp(a*log(x) - x - lgamma(a+1))
    # Use mx.maximum to avoid log(0) for very small x
    safe_x = mx.maximum(x, 1e-38)
    log_prefix = a * mx.log(safe_x) - x - _gammaln_mlx(a + 1.0)
    result = mx.exp(log_prefix) * sum_val

    # Handle x = 0 case
    result = mx.where(x == 0, mx.zeros_like(x), result)

    return result


def _gammainc_cf(a: mx.array, x: mx.array, max_iter: int = 300) -> mx.array:
    """
    Compute upper incomplete gamma using continued fraction (Lentz's algorithm).
    Q(a,x) = (x^a * e^-x / Gamma(a)) * CF
    Good for x >= a + 1.
    Returns 1 - P(a,x).

    Increased max_iter from 200 to 300 for better convergence.
    Uses modified Lentz's algorithm with underflow protection.
    """
    # Modified Lentz's algorithm for continued fraction
    tiny = 1e-30
    eps = 1e-10  # Convergence threshold

    # Initialize
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / mx.maximum(mx.abs(b), tiny)
    d = mx.where(b < 0, -d, d)  # Preserve sign
    h = d

    for i in range(1, max_iter):
        an = -float(i) * (float(i) - a)
        b = b + 2.0
        d = an * d + b
        d = mx.where(mx.abs(d) < tiny, mx.array(tiny), d)
        c = b + an / c
        c = mx.where(mx.abs(c) < tiny, mx.array(tiny), c)
        d = 1.0 / d
        delta = d * c
        h = h * delta

    # Q(a,x) = exp(a*log(x) - x - lgamma(a)) * h
    log_prefix = a * mx.log(x) - x - _gammaln_mlx(a)
    result = mx.exp(log_prefix) * h

    return result


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
    a = _to_mlx(input)
    x = _to_mlx(other)

    # Use series for x < a+1, continued fraction otherwise
    use_series = x < (a + 1.0)

    # Compute both methods
    series_result = _gammainc_series(a, x)
    cf_result = 1.0 - _gammainc_cf(a, x)

    result = mx.where(use_series, series_result, cf_result)

    # Clamp to [0, 1]
    result = mx.clip(result, 0.0, 1.0)

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
    a = _to_mlx(input)
    x = _to_mlx(other)

    # Q(a,x) = 1 - P(a,x)
    use_series = x < (a + 1.0)

    series_result = 1.0 - _gammainc_series(a, x)
    cf_result = _gammainc_cf(a, x)

    result = mx.where(use_series, series_result, cf_result)

    # Clamp to [0, 1]
    result = mx.clip(result, 0.0, 1.0)

    return _to_tensor(result)


# =============================================================================
# Bessel functions - Pure MLX implementations using polynomial approximations
# =============================================================================

def bessel_j0(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Bessel function of the first kind of order 0.

    Uses polynomial approximations from Abramowitz & Stegun.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Bessel J0 values
    """
    x = _to_mlx(input)
    ax = mx.abs(x)

    # For |x| < 8, use polynomial approximation
    # For |x| >= 8, use asymptotic form

    # Polynomial for |x| < 8
    y = x * x
    p1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (
        -11214424.18 + y * (77392.33017 + y * (-184.9052456)))))
    p2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (
        59272.64853 + y * (267.8532712 + y * 1.0))))
    small_result = p1 / p2

    # Asymptotic form for |x| >= 8
    z = 8.0 / ax
    y_large = z * z
    xx = ax - 0.785398164  # pi/4

    p3 = 1.0 + y_large * (-0.1098628627e-2 + y_large * (0.2734510407e-4 + y_large * (
        -0.2073370639e-5 + y_large * 0.2093887211e-6)))
    p4 = -0.1562499995e-1 + y_large * (0.1430488765e-3 + y_large * (
        -0.6911147651e-5 + y_large * (0.7621095161e-6 - y_large * 0.934945152e-7)))

    large_result = mx.sqrt(0.636619772 / ax) * (mx.cos(xx) * p3 - z * mx.sin(xx) * p4)

    result = mx.where(ax < 8.0, small_result, large_result)
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
    x = _to_mlx(input)
    ax = mx.abs(x)

    # Polynomial for |x| < 8
    y = x * x
    p1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (
        -2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
    p2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (
        99447.43394 + y * (376.9991397 + y * 1.0))))
    small_result = p1 / p2

    # Asymptotic form for |x| >= 8
    z = 8.0 / ax
    y_large = z * z
    xx = ax - 2.356194491  # 3*pi/4

    p3 = 1.0 + y_large * (0.183105e-2 + y_large * (-0.3516396496e-4 + y_large * (
        0.2457520174e-5 + y_large * (-0.240337019e-6))))
    p4 = 0.04687499995 + y_large * (-0.2002690873e-3 + y_large * (
        0.8449199096e-5 + y_large * (-0.88228987e-6 + y_large * 0.105787412e-6)))

    large_result = mx.sqrt(0.636619772 / ax) * (mx.cos(xx) * p3 - z * mx.sin(xx) * p4)
    large_result = mx.where(x < 0, -large_result, large_result)

    result = mx.where(ax < 8.0, small_result, large_result)
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
    x = _to_mlx(input)

    # Y0 is only defined for x > 0
    # For x < 8, use polynomial + J0 * ln(x)
    # For x >= 8, use asymptotic form

    # Small x polynomial
    y = x * x
    p1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (
        10879881.29 + y * (-86327.92757 + y * 228.4622733))))
    p2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (
        47447.26470 + y * (226.1030244 + y * 1.0))))

    # Y0(x) = (2/pi) * J0(x) * ln(x) + polynomial
    j0_val = bessel_j0(input)._mlx_array
    small_result = p1 / p2 + 0.636619772 * j0_val * mx.log(x)

    # Asymptotic form for x >= 8
    z = 8.0 / x
    y_large = z * z
    xx = x - 0.785398164

    p3 = 1.0 + y_large * (-0.1098628627e-2 + y_large * (0.2734510407e-4 + y_large * (
        -0.2073370639e-5 + y_large * 0.2093887211e-6)))
    p4 = -0.1562499995e-1 + y_large * (0.1430488765e-3 + y_large * (
        -0.6911147651e-5 + y_large * (0.7621095161e-6 - y_large * 0.934945152e-7)))

    large_result = mx.sqrt(0.636619772 / x) * (mx.sin(xx) * p3 + z * mx.cos(xx) * p4)

    result = mx.where(x < 8.0, small_result, large_result)

    # Handle x <= 0 (undefined)
    result = mx.where(x <= 0, mx.array(float('nan')), result)

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
    x = _to_mlx(input)

    # Small x polynomial
    y = x * x
    p1 = x * (-4900604943000.0 + y * (1275274390000.0 + y * (-51534381390.0 + y * (
        734926455.1 + y * (-4237922.726 + y * 8511.937935)))))
    p2 = 24909857410000.0 + y * (424441966400.0 + y * (3733650367.0 + y * (
        22459040.02 + y * (102042.605 + y * (354.9632885 + y)))))

    # Y1(x) = (2/pi) * (J1(x) * ln(x) - 1/x) + polynomial
    j1_val = bessel_j1(input)._mlx_array
    small_result = p1 / p2 + 0.636619772 * (j1_val * mx.log(x) - 1.0 / x)

    # Asymptotic form for x >= 8
    z = 8.0 / x
    y_large = z * z
    xx = x - 2.356194491

    p3 = 1.0 + y_large * (0.183105e-2 + y_large * (-0.3516396496e-4 + y_large * (
        0.2457520174e-5 + y_large * (-0.240337019e-6))))
    p4 = 0.04687499995 + y_large * (-0.2002690873e-3 + y_large * (
        0.8449199096e-5 + y_large * (-0.88228987e-6 + y_large * 0.105787412e-6)))

    large_result = mx.sqrt(0.636619772 / x) * (mx.sin(xx) * p3 + z * mx.cos(xx) * p4)

    result = mx.where(x < 8.0, small_result, large_result)
    result = mx.where(x <= 0, mx.array(float('nan')), result)

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
    x = _to_mlx(input)
    ax = mx.abs(x)

    # Polynomial for |x| < 3.75
    y = (x / 3.75) ** 2
    small_result = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (
        0.2659732 + y * (0.0360768 + y * 0.0045813)))))

    # Polynomial for |x| >= 3.75
    y_large = 3.75 / ax
    large_result = (mx.exp(ax) / mx.sqrt(ax)) * (0.39894228 + y_large * (
        0.01328592 + y_large * (0.00225319 + y_large * (-0.00157565 + y_large * (
            0.00916281 + y_large * (-0.02057706 + y_large * (0.02635537 + y_large * (
                -0.01647633 + y_large * 0.00392377))))))))

    result = mx.where(ax < 3.75, small_result, large_result)
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
    x = _to_mlx(input)
    ax = mx.abs(x)

    # Polynomial for |x| < 3.75
    y = (x / 3.75) ** 2
    small_result = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (
        0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))

    # Polynomial for |x| >= 3.75
    y_large = 3.75 / ax
    large_result = (mx.exp(ax) / mx.sqrt(ax)) * (0.39894228 + y_large * (
        -0.03988024 + y_large * (-0.00362018 + y_large * (0.00163801 + y_large * (
            -0.01031555 + y_large * (0.02282967 + y_large * (-0.02895312 + y_large * (
                0.01787654 + y_large * (-0.00420059)))))))))

    result = mx.where(ax < 3.75, small_result, large_result)
    result = mx.where(x < 0, -result, result)

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
    x = _to_mlx(input)

    # For x <= 2, use polynomial with I0 * ln(x) term
    y = x * x / 4.0
    small_result = (-mx.log(x / 2.0) * modified_bessel_i0(Tensor(x))._mlx_array) + (
        -0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (
            0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))))

    # For x > 2, use different polynomial
    y_large = 2.0 / x
    large_result = (mx.exp(-x) / mx.sqrt(x)) * (1.25331414 + y_large * (
        -0.07832358 + y_large * (0.02189568 + y_large * (-0.01062446 + y_large * (
            0.00587872 + y_large * (-0.00251540 + y_large * 0.00053208))))))

    result = mx.where(x <= 2.0, small_result, large_result)
    result = mx.where(x <= 0, mx.array(float('inf')), result)

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
    x = _to_mlx(input)

    # For x <= 2
    y = x * x / 4.0
    small_result = (mx.log(x / 2.0) * modified_bessel_i1(Tensor(x))._mlx_array) + (1.0 / x) * (
        1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 + y * (
            -0.01919402 + y * (-0.00110404 + y * (-0.00004686)))))))

    # For x > 2
    y_large = 2.0 / x
    large_result = (mx.exp(-x) / mx.sqrt(x)) * (1.25331414 + y_large * (
        0.23498619 + y_large * (-0.03655620 + y_large * (0.01504268 + y_large * (
            -0.00780353 + y_large * (0.00325614 + y_large * (-0.00068245)))))))

    result = mx.where(x <= 2.0, small_result, large_result)
    result = mx.where(x <= 0, mx.array(float('inf')), result)

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
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("scaled_modified_bessel_k0() missing required argument: 'input' or 'x'")

    data = _to_mlx(tensor)
    k0_val = modified_bessel_k0(Tensor(data))._mlx_array
    result = mx.exp(data) * k0_val
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
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("scaled_modified_bessel_k1() missing required argument: 'input' or 'x'")

    data = _to_mlx(tensor)
    k1_val = modified_bessel_k1(Tensor(data))._mlx_array
    result = mx.exp(data) * k1_val
    return _to_tensor(result)


def spherical_bessel_j0(input: Tensor = None, *, x: Tensor = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Spherical Bessel function of the first kind of order 0.

    j0(x) = sin(x) / x

    Args:
        input: Input tensor (also accepts 'x' for PyTorch compatibility)
        x: Input tensor (alternative to 'input')
        out: Output tensor (ignored in MLX)

    Returns:
        Spherical Bessel j0 values
    """
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("spherical_bessel_j0() missing required argument: 'input' or 'x'")

    data = _to_mlx(tensor)
    # j0(x) = sin(x) / x, with j0(0) = 1
    result = mx.where(mx.abs(data) < 1e-10, mx.ones_like(data), mx.sin(data) / data)
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
    data = _to_mlx(input)
    i0_val = modified_bessel_i0(Tensor(data))._mlx_array
    result = mx.exp(-mx.abs(data)) * i0_val
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
    data = _to_mlx(input)
    i1_val = modified_bessel_i1(Tensor(data))._mlx_array
    result = mx.exp(-mx.abs(data)) * i1_val
    return _to_tensor(result)


# =============================================================================
# Exponential and logarithmic functions
# =============================================================================

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


# =============================================================================
# Logit and sigmoid functions
# =============================================================================

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


# =============================================================================
# Softmax functions
# =============================================================================

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
    Compute log softmax in a numerically stable way.

    Uses MLX's fused logsumexp for better numerical stability and performance.
    log_softmax(x) = x - logsumexp(x)

    Args:
        input: Input tensor
        dim: Dimension along which to compute log softmax
        dtype: Output dtype

    Returns:
        Log softmax values
    """
    data = _to_mlx(input)
    # Use MLX's fused logsumexp (numerically stable and efficient)
    result = data - mx.logsumexp(data, axis=dim, keepdims=True)
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


# =============================================================================
# Normal distribution functions
# =============================================================================

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
    Compute the inverse of the CDF of the standard normal (quantile function).

    Uses rational approximation (Moro's algorithm).

    Args:
        input: Input tensor with values in (0, 1)
        out: Output tensor (ignored in MLX)

    Returns:
        Inverse CDF values
    """
    p = _to_mlx(input)

    # Moro's algorithm - rational approximation for inverse normal CDF
    # Split into central region and tails

    # Central region coefficients (|p - 0.5| <= 0.425)
    a = mx.array([
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ], dtype=mx.float32)

    b = mx.array([
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ], dtype=mx.float32)

    # Tail region coefficients
    c = mx.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ], dtype=mx.float32)

    d = mx.array([
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ], dtype=mx.float32)

    # Define break points
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Central region
    q = p - 0.5
    r_central = q * q

    num_central = (((((a[0] * r_central + a[1]) * r_central + a[2]) * r_central + a[3]) * r_central + a[4]) * r_central + a[5])
    den_central = (((((b[0] * r_central + b[1]) * r_central + b[2]) * r_central + b[3]) * r_central + b[4]) * r_central + 1.0)
    central_result = q * num_central / den_central

    # Lower tail
    q_low = mx.sqrt(-2.0 * mx.log(p))
    num_low = (((((c[0] * q_low + c[1]) * q_low + c[2]) * q_low + c[3]) * q_low + c[4]) * q_low + c[5])
    den_low = ((((d[0] * q_low + d[1]) * q_low + d[2]) * q_low + d[3]) * q_low + 1.0)
    low_result = num_low / den_low

    # Upper tail
    q_high = mx.sqrt(-2.0 * mx.log(1.0 - p))
    num_high = (((((c[0] * q_high + c[1]) * q_high + c[2]) * q_high + c[3]) * q_high + c[4]) * q_high + c[5])
    den_high = ((((d[0] * q_high + d[1]) * q_high + d[2]) * q_high + d[3]) * q_high + 1.0)
    high_result = -num_high / den_high

    # Combine results
    in_central = (p > p_low) & (p < p_high)
    in_low = p <= p_low

    result = mx.where(in_central, central_result, mx.where(in_low, low_result, high_result))

    # Handle edge cases
    result = mx.where(p <= 0, mx.array(float('-inf')), result)
    result = mx.where(p >= 1, mx.array(float('inf')), result)

    return _to_tensor(result)


def log_ndtr(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the log of the CDF of the standard normal.

    Uses numerically stable computation for extreme values.

    Args:
        input: Input tensor
        out: Output tensor (ignored in MLX)

    Returns:
        Log CDF values
    """
    x = _to_mlx(input)

    # For large positive x, log(ndtr(x)) ~ 0
    # For large negative x, use asymptotic: log(ndtr(x)) ~ -x^2/2 - log(-x) - 0.5*log(2*pi)

    # Normal computation
    ndtr_val = mx.array(0.5) * (mx.array(1.0) + mx.erf(x / mx.sqrt(mx.array(2.0))))
    normal_result = mx.log(ndtr_val)

    # Asymptotic for very negative x (more stable)
    # log(erfc(x/sqrt(2))/2) for x << 0
    # erfc(y) ~ exp(-y^2) / (y * sqrt(pi)) for large y
    # So log(ndtr(x)) ~ -x^2/2 - log(-x*sqrt(2)) - 0.5*log(pi) for x << 0
    asymp_result = -0.5 * x * x - mx.log(-x) - 0.5 * mx.log(mx.array(2.0 * math.pi))

    # Use asymptotic for x < -5 (where erfc becomes very small)
    result = mx.where(x < -5.0, asymp_result, normal_result)

    return _to_tensor(result)


# =============================================================================
# Entropy
# =============================================================================

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


# =============================================================================
# Sinc function
# =============================================================================

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


# =============================================================================
# Round function
# =============================================================================

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


# =============================================================================
# Zeta function
# =============================================================================

def zeta(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the Hurwitz zeta function.

    zeta(s, q) = sum_{n=0}^{inf} 1/(q+n)^s

    Uses Euler-Maclaurin formula for computation.

    Args:
        input: First input tensor (s parameter)
        other: Second input tensor (q parameter)
        out: Output tensor (ignored in MLX)

    Returns:
        Zeta function values
    """
    s = _to_mlx(input)
    q = _to_mlx(other)

    # Use Euler-Maclaurin with N terms and p correction terms
    N = 20
    p = 10  # Number of Bernoulli correction terms

    # Sum first N terms exactly
    result = mx.zeros_like(s)
    for n in range(N):
        result = result + 1.0 / mx.power(q + float(n), s)

    # Integral approximation: integral from N to inf of 1/(q+x)^s dx = (q+N)^(1-s) / (s-1)
    integral = mx.power(q + float(N), 1.0 - s) / (s - 1.0)
    result = result + integral

    # First Euler-Maclaurin correction: 1/2 * f(N)
    result = result + 0.5 / mx.power(q + float(N), s)

    # Bernoulli number corrections (B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, ...)
    # Only use first few for numerical stability
    bernoulli = [1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0, 5.0/66.0]

    qN = q + float(N)
    s_prod = s  # s
    qN_pow = 1.0 / mx.power(qN, s + 1.0)

    for k, b in enumerate(bernoulli):
        if k >= p:
            break
        s_prod = s_prod * (s + 2*k) * (s + 2*k + 1)  # Rising factorial
        factorial = 1.0
        for j in range(1, 2*k + 3):
            factorial *= j
        qN_pow = qN_pow / (qN * qN)
        result = result + b * s_prod * qN_pow / factorial

    return _to_tensor(result)


# =============================================================================
# Airy function
# =============================================================================

def airy_ai(input: Tensor = None, *, x: Tensor = None, out: Optional[Tensor] = None) -> Tensor:
    """
    Compute the Airy function Ai.

    Uses polynomial/asymptotic approximations.

    Args:
        input: Input tensor (also accepts 'x' for PyTorch compatibility)
        x: Input tensor (alternative to 'input')
        out: Output tensor (ignored in MLX)

    Returns:
        Airy Ai values
    """
    tensor = input if input is not None else x
    if tensor is None:
        raise TypeError("airy_ai() missing required argument: 'input' or 'x'")

    z = _to_mlx(tensor)

    # Use different approximations for different regions
    # For small |z|, use Taylor series
    # For large positive z, Ai(z) ~ exp(-2/3 * z^(3/2)) / (2*sqrt(pi)*z^(1/4))
    # For large negative z, use oscillatory asymptotic

    # Coefficients for Taylor series around z=0
    # Ai(z) = c1 * f(z) - c2 * g(z) where f, g are power series
    c1 = 0.355028053887817
    c2 = 0.258819403792807

    # Taylor series (valid for |z| < ~3)
    z3 = z * z * z
    f = 1.0 + z3 / 6.0 + z3 * z3 / 180.0 + z3 * z3 * z3 / 12600.0
    g = z + z3 * z / 12.0 + z3 * z3 * z / 504.0 + z3 * z3 * z3 * z / 45360.0
    taylor_result = c1 * f - c2 * g

    # Asymptotic for large positive z
    # Ai(z) ~ exp(-zeta) / (2*sqrt(pi)*z^(1/4)) where zeta = 2/3 * z^(3/2)
    zeta_pos = (2.0 / 3.0) * mx.power(z, 1.5)
    asymp_pos = mx.exp(-zeta_pos) / (2.0 * mx.sqrt(mx.array(math.pi)) * mx.power(z, 0.25))

    # Asymptotic for large negative z (oscillatory)
    # Ai(-z) ~ sin(zeta + pi/4) / (sqrt(pi)*z^(1/4)) where zeta = 2/3 * z^(3/2)
    z_neg = mx.abs(z)
    zeta_neg = (2.0 / 3.0) * mx.power(z_neg, 1.5)
    asymp_neg = mx.sin(zeta_neg + math.pi / 4.0) / (mx.sqrt(mx.array(math.pi)) * mx.power(z_neg, 0.25))

    # Combine regions
    result = mx.where(mx.abs(z) < 3.0, taylor_result,
                      mx.where(z > 0, asymp_pos, asymp_neg))

    return _to_tensor(result)


# =============================================================================
# Polynomial functions (Chebyshev, Hermite, Laguerre, Legendre)
# Pure MLX implementations using recurrence relations
# =============================================================================

def _chebyshev_t_scalar(x: mx.array, n: int) -> mx.array:
    """Chebyshev T polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return x
    else:
        t_prev = mx.ones_like(x)
        t_curr = x
        for _ in range(2, n + 1):
            t_next = 2.0 * x * t_curr - t_prev
            t_prev = t_curr
            t_curr = t_next
        return t_curr


def chebyshev_polynomial_t(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Chebyshev polynomial of the first kind.

    T_n(x): T_0 = 1, T_1 = x, T_{n+1} = 2x*T_n - T_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Chebyshev T_n values
    """
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _chebyshev_t_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        # For tensor n, we need to compute element-wise
        # This is less efficient but handles the general case
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        # Compute all polynomials up to max_n
        polys = [mx.ones_like(x)]  # T_0
        if max_n >= 1:
            polys.append(x)  # T_1
        for k in range(2, max_n + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])

        # Select based on n
        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        # Assume it's an int-like value
        result = _chebyshev_t_scalar(x, int(n))

    return _to_tensor(result)


def _chebyshev_u_scalar(x: mx.array, n: int) -> mx.array:
    """Chebyshev U polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        u_prev = mx.ones_like(x)
        u_curr = 2.0 * x
        for _ in range(2, n + 1):
            u_next = 2.0 * x * u_curr - u_prev
            u_prev = u_curr
            u_curr = u_next
        return u_curr


def chebyshev_polynomial_u(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Chebyshev polynomial of the second kind.

    U_n(x): U_0 = 1, U_1 = 2x, U_{n+1} = 2x*U_n - U_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Chebyshev U_n values
    """
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _chebyshev_u_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(2.0 * x)
        for k in range(2, max_n + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _chebyshev_u_scalar(x, int(n))

    return _to_tensor(result)


def _chebyshev_v_scalar(x: mx.array, n: int) -> mx.array:
    """Chebyshev V polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return 2.0 * x - 1.0
    else:
        v_prev = mx.ones_like(x)
        v_curr = 2.0 * x - 1.0
        for _ in range(2, n + 1):
            v_next = 2.0 * x * v_curr - v_prev
            v_prev = v_curr
            v_curr = v_next
        return v_curr


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
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _chebyshev_v_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(2.0 * x - 1.0)
        for k in range(2, max_n + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _chebyshev_v_scalar(x, int(n))

    return _to_tensor(result)


def _chebyshev_w_scalar(x: mx.array, n: int) -> mx.array:
    """Chebyshev W polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return 2.0 * x + 1.0
    else:
        w_prev = mx.ones_like(x)
        w_curr = 2.0 * x + 1.0
        for _ in range(2, n + 1):
            w_next = 2.0 * x * w_curr - w_prev
            w_prev = w_curr
            w_curr = w_next
        return w_curr


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
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _chebyshev_w_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(2.0 * x + 1.0)
        for k in range(2, max_n + 1):
            polys.append(2.0 * x * polys[-1] - polys[-2])

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _chebyshev_w_scalar(x, int(n))

    return _to_tensor(result)


def shifted_chebyshev_polynomial_t(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the first kind.

    T*_n(x) = T_n(2x - 1)

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev T*_n values
    """
    x = _to_mlx(input)
    shifted_x = 2.0 * x - 1.0
    return chebyshev_polynomial_t(Tensor(shifted_x), n, out=out)


def shifted_chebyshev_polynomial_u(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the second kind.

    U*_n(x) = U_n(2x - 1)

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev U*_n values
    """
    x = _to_mlx(input)
    shifted_x = 2.0 * x - 1.0
    return chebyshev_polynomial_u(Tensor(shifted_x), n, out=out)


def shifted_chebyshev_polynomial_v(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the third kind.

    V*_n(x) = V_n(2x - 1)

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev V*_n values
    """
    x = _to_mlx(input)
    shifted_x = 2.0 * x - 1.0
    return chebyshev_polynomial_v(Tensor(shifted_x), n, out=out)


def shifted_chebyshev_polynomial_w(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Shifted Chebyshev polynomial of the fourth kind.

    W*_n(x) = W_n(2x - 1)

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Shifted Chebyshev W*_n values
    """
    x = _to_mlx(input)
    shifted_x = 2.0 * x - 1.0
    return chebyshev_polynomial_w(Tensor(shifted_x), n, out=out)


def _hermite_h_scalar(x: mx.array, n: int) -> mx.array:
    """Physicist's Hermite polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h_prev = mx.ones_like(x)
        h_curr = 2.0 * x
        for k in range(2, n + 1):
            h_next = 2.0 * x * h_curr - 2.0 * (k - 1) * h_prev
            h_prev = h_curr
            h_curr = h_next
        return h_curr


def hermite_polynomial_h(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Physicist's Hermite polynomial.

    H_n(x): H_0 = 1, H_1 = 2x, H_{n+1} = 2x*H_n - 2n*H_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Hermite H_n values
    """
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _hermite_h_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(2.0 * x)
        for k in range(2, max_n + 1):
            polys.append(2.0 * x * polys[-1] - 2.0 * (k - 1) * polys[-2])

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _hermite_h_scalar(x, int(n))

    return _to_tensor(result)


def _hermite_he_scalar(x: mx.array, n: int) -> mx.array:
    """Probabilist's Hermite polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return x
    else:
        he_prev = mx.ones_like(x)
        he_curr = x
        for k in range(2, n + 1):
            he_next = x * he_curr - (k - 1) * he_prev
            he_prev = he_curr
            he_curr = he_next
        return he_curr


def hermite_polynomial_he(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Probabilist's Hermite polynomial.

    He_n(x): He_0 = 1, He_1 = x, He_{n+1} = x*He_n - n*He_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Hermite He_n values
    """
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _hermite_he_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(x)
        for k in range(2, max_n + 1):
            polys.append(x * polys[-1] - (k - 1) * polys[-2])

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _hermite_he_scalar(x, int(n))

    return _to_tensor(result)


def _laguerre_l_scalar(x: mx.array, n: int) -> mx.array:
    """Laguerre polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return 1.0 - x
    else:
        l_prev = mx.ones_like(x)
        l_curr = 1.0 - x
        for k in range(2, n + 1):
            # (k+1)*L_{k+1} = (2k+1-x)*L_k - k*L_{k-1}
            # So L_{k} = ((2(k-1)+1-x)*L_{k-1} - (k-1)*L_{k-2}) / k
            l_next = ((2.0 * (k - 1) + 1.0 - x) * l_curr - (k - 1) * l_prev) / k
            l_prev = l_curr
            l_curr = l_next
        return l_curr


def laguerre_polynomial_l(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Laguerre polynomial.

    L_n(x): L_0 = 1, L_1 = 1-x, (n+1)*L_{n+1} = (2n+1-x)*L_n - n*L_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Laguerre L_n values
    """
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _laguerre_l_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(1.0 - x)
        for k in range(2, max_n + 1):
            polys.append(((2.0 * (k - 1) + 1.0 - x) * polys[-1] - (k - 1) * polys[-2]) / k)

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _laguerre_l_scalar(x, int(n))

    return _to_tensor(result)


def _legendre_p_scalar(x: mx.array, n: int) -> mx.array:
    """Legendre polynomial for scalar n using recurrence."""
    if n < 0:
        return mx.zeros_like(x)
    elif n == 0:
        return mx.ones_like(x)
    elif n == 1:
        return x
    else:
        p_prev = mx.ones_like(x)
        p_curr = x
        for k in range(2, n + 1):
            # (k+1)*P_{k+1} = (2k+1)*x*P_k - k*P_{k-1}
            # So P_k = ((2(k-1)+1)*x*P_{k-1} - (k-1)*P_{k-2}) / k
            p_next = ((2.0 * (k - 1) + 1.0) * x * p_curr - (k - 1) * p_prev) / k
            p_prev = p_curr
            p_curr = p_next
        return p_curr


def legendre_polynomial_p(input: Tensor, n: Union[int, Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """
    Legendre polynomial.

    P_n(x): P_0 = 1, P_1 = x, (n+1)*P_{n+1} = (2n+1)*x*P_n - n*P_{n-1}

    Args:
        input: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor (ignored in MLX)

    Returns:
        Legendre P_n values
    """
    x = _to_mlx(input)

    if isinstance(n, int):
        result = _legendre_p_scalar(x, n)
    elif isinstance(n, Tensor):
        n_arr = n._mlx_array
        max_n = int(mx.max(n_arr).item()) if n_arr.size > 0 else 0

        polys = [mx.ones_like(x)]
        if max_n >= 1:
            polys.append(x)
        for k in range(2, max_n + 1):
            polys.append(((2.0 * (k - 1) + 1.0) * x * polys[-1] - (k - 1) * polys[-2]) / k)

        result = mx.zeros_like(x)
        for k in range(max_n + 1):
            mask = (n_arr == k)
            result = mx.where(mask, polys[k], result)
    else:
        result = _legendre_p_scalar(x, int(n))

    return _to_tensor(result)


# =============================================================================
# Module exports
# =============================================================================

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
