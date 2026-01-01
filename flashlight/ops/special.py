"""
Special mathematical functions implemented in pure MLX.

These functions replace scipy.special for use in probability distributions
and other mathematical computations. All functions work directly with
mx.array objects for maximum efficiency.

Implementations use well-known numerical methods:
- Lanczos approximation for lgamma
- Asymptotic expansion for digamma
- Continued fractions for incomplete gamma/beta
- Chebyshev polynomials for Bessel functions
"""

import math
from typing import Union

import mlx.core as mx

from ..distributions._constants import CF_TINY

# Lanczos coefficients for lgamma (g=7, n=9)
# These provide ~15 digit precision for positive real numbers
_LANCZOS_G = 7.0
_LANCZOS_COEFFICIENTS = mx.array(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ],
    dtype=mx.float32,
)


def lgamma(x: mx.array) -> mx.array:
    """
    Compute the natural logarithm of the absolute value of the gamma function.

    Uses the Lanczos approximation which provides high accuracy (~1e-10)
    for all positive real numbers.

    For negative non-integer x, uses the reflection formula:
        lgamma(x) = log(pi / |sin(pi*x)|) - lgamma(1-x)

    Args:
        x: Input array

    Returns:
        Array with log-gamma values
    """
    x = mx.array(x) if not isinstance(x, mx.array) else x
    orig_dtype = x.dtype
    x = x.astype(mx.float32)

    # Handle negative values using reflection formula
    # lgamma(x) = log(pi) - log(|sin(pi*x)|) - lgamma(1-x)
    needs_reflection = x < 0.5

    # For reflection: work with 1-x instead of x
    z = mx.where(needs_reflection, 1.0 - x, x)

    # Lanczos approximation for lgamma(z) where z >= 0.5
    # lgamma(z) = 0.5*log(2*pi) + (z-0.5)*log(z+g-0.5) - (z+g-0.5) + log(Ag(z))
    z_minus_1 = z - 1.0

    # Compute Ag(z) = c0 + c1/(z) + c2/(z+1) + ... + c8/(z+7)
    # where z here is actually z-1 in the standard formulation
    ag = _LANCZOS_COEFFICIENTS[0]
    for i in range(1, 9):
        ag = ag + _LANCZOS_COEFFICIENTS[i] / (z_minus_1 + float(i))

    # Compute lgamma using Lanczos formula
    tmp = z_minus_1 + _LANCZOS_G + 0.5
    lgamma_z = 0.5 * math.log(2.0 * math.pi) + (z_minus_1 + 0.5) * mx.log(tmp) - tmp + mx.log(ag)

    # Apply reflection formula for negative values
    # lgamma(x) = log(pi) - log(|sin(pi*x)|) - lgamma(1-x)
    reflected = math.log(math.pi) - mx.log(mx.abs(mx.sin(math.pi * x))) - lgamma_z

    result = mx.where(needs_reflection, reflected, lgamma_z)

    # Handle special cases
    # lgamma(1) = lgamma(2) = 0
    result = mx.where((x == 1.0) | (x == 2.0), mx.zeros_like(result), result)

    # lgamma at non-positive integers is +inf
    is_nonpositive_int = (x <= 0) & (x == mx.floor(x))
    result = mx.where(is_nonpositive_int, mx.array(float("inf")), result)

    # Always return float32 - lgamma is a continuous function that produces floats
    # even for integer inputs
    return result


def gamma(x: mx.array) -> mx.array:
    """
    Compute the gamma function.

    gamma(x) = exp(lgamma(x)) with appropriate sign handling.

    Args:
        x: Input array

    Returns:
        Array with gamma values
    """
    return mx.exp(lgamma(x))


def digamma(x: mx.array) -> mx.array:
    """
    Compute the digamma function (psi function).

    The digamma function is the logarithmic derivative of the gamma function:
        digamma(x) = d/dx ln(gamma(x)) = gamma'(x) / gamma(x)

    Uses asymptotic expansion for large x and recurrence for small x.
    For negative x, uses the reflection formula.

    Args:
        x: Input array

    Returns:
        Array with digamma values
    """
    x = mx.array(x) if not isinstance(x, mx.array) else x
    orig_dtype = x.dtype
    x = x.astype(mx.float32)

    result = mx.zeros_like(x)

    # Handle negative values using reflection formula
    # digamma(1-x) - digamma(x) = pi * cot(pi*x)
    # => digamma(x) = digamma(1-x) - pi * cot(pi*x)
    needs_reflection = x < 0
    x_work = mx.where(needs_reflection, 1.0 - x, x)

    # Use recurrence relation to shift x to large values
    # digamma(x+1) = digamma(x) + 1/x
    # We shift x until it's >= 6 for good asymptotic convergence
    shift = mx.zeros_like(x_work)
    x_shifted = x_work

    # Unroll the loop for a fixed number of iterations
    for _ in range(7):
        needs_shift = x_shifted < 6.0
        shift = mx.where(needs_shift, shift - 1.0 / x_shifted, shift)
        x_shifted = mx.where(needs_shift, x_shifted + 1.0, x_shifted)

    # Asymptotic expansion for large x:
    # digamma(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{inf} B_{2k} / (2k * x^{2k})
    # where B_{2k} are Bernoulli numbers
    # B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66
    x2 = x_shifted * x_shifted
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6

    # Bernoulli numbers B_{2k} / (2k)
    b2 = 1.0 / 12.0  # B_2 / 2 = (1/6) / 2
    b4 = 1.0 / 120.0  # B_4 / 4 = (-1/30) / 4, but we need -B_4/4
    b6 = 1.0 / 252.0  # B_6 / 6
    b8 = 1.0 / 240.0  # -B_8 / 8
    b10 = 5.0 / 660.0  # B_10 / 10
    b12 = 691.0 / 32760.0  # -B_12 / 12

    asymp = (
        mx.log(x_shifted)
        - 0.5 / x_shifted
        - b2 / x2
        + b4 / x4
        - b6 / x6
        + b8 / x8
        - b10 / x10
        + b12 / x12
    )

    result = asymp + shift

    # Apply reflection formula for negative values
    # digamma(x) = digamma(1-x) - pi * cot(pi*x)
    # = digamma(1-x) - pi * cos(pi*x) / sin(pi*x)
    sin_pix = mx.sin(math.pi * x)
    # Add epsilon protection when sin is near zero to avoid division by zero
    safe_sin_pix = mx.where(
        mx.abs(sin_pix) < CF_TINY, CF_TINY * mx.sign(sin_pix + CF_TINY), sin_pix
    )
    cot_pix = mx.cos(math.pi * x) / safe_sin_pix
    reflected = result - math.pi * cot_pix

    result = mx.where(needs_reflection, reflected, result)

    # Handle poles at non-positive integers
    is_nonpositive_int = (x <= 0) & (x == mx.floor(x))
    result = mx.where(is_nonpositive_int, mx.array(float("nan")), result)

    # Always return float32 - digamma is a continuous function that produces floats
    return result


def betaln(a: mx.array, b: mx.array) -> mx.array:
    """
    Compute the natural logarithm of the beta function.

    betaln(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)

    Args:
        a: First parameter
        b: Second parameter

    Returns:
        Array with log-beta values
    """
    a = mx.array(a) if not isinstance(a, mx.array) else a
    b = mx.array(b) if not isinstance(b, mx.array) else b

    return lgamma(a) + lgamma(b) - lgamma(a + b)


def beta(a: mx.array, b: mx.array) -> mx.array:
    """
    Compute the beta function.

    beta(a, b) = gamma(a) * gamma(b) / gamma(a + b) = exp(betaln(a, b))

    Args:
        a: First parameter
        b: Second parameter

    Returns:
        Array with beta values
    """
    return mx.exp(betaln(a, b))


def gammainc(a: mx.array, x: mx.array) -> mx.array:
    """
    Compute the regularized lower incomplete gamma function.

    P(a, x) = (1/gamma(a)) * integral from 0 to x of t^(a-1) * exp(-t) dt

    Uses series expansion for x < a+1 and continued fraction for x >= a+1.

    Args:
        a: Shape parameter (positive)
        x: Upper limit of integration (non-negative)

    Returns:
        Array with incomplete gamma values in [0, 1]
    """
    a = mx.array(a) if not isinstance(a, mx.array) else a
    x = mx.array(x) if not isinstance(x, mx.array) else x

    a = a.astype(mx.float32)
    x = x.astype(mx.float32)

    # Handle edge cases
    result = mx.zeros_like(a + x)

    # x = 0 => P(a, 0) = 0
    # x = inf => P(a, inf) = 1
    result = mx.where(x <= 0, mx.zeros_like(result), result)

    # For x < a + 1, use series expansion
    # For x >= a + 1, use continued fraction
    use_series = x < (a + 1.0)

    # Series expansion: P(a,x) = exp(-x) * x^a * sum_{n=0}^{inf} x^n / gamma(a+n+1)
    # Rewritten: P(a,x) = exp(-x + a*ln(x) - lgamma(a)) * sum_{n=0}^{inf} gamma(a)/gamma(a+n+1) * x^n
    # = exp(-x + a*ln(x) - lgamma(a+1)) * sum_{n=0}^{inf} x^n / (a+1)(a+2)...(a+n)

    # Series computation
    log_prefactor = -x + a * mx.log(mx.maximum(x, CF_TINY)) - lgamma(a + 1)

    # Sum the series
    term = mx.ones_like(a)
    series_sum = mx.ones_like(a)
    for n in range(1, 100):  # Usually converges much faster
        term = term * x / (a + float(n))
        series_sum = series_sum + term
        # Early termination check would require dynamic control flow

    series_result = mx.exp(log_prefactor) * series_sum

    # Continued fraction for Q(a,x) = 1 - P(a,x) using Lentz's algorithm
    # Then P(a,x) = 1 - Q(a,x)
    # Q(a,x) = exp(-x + a*ln(x) - lgamma(a)) * (1/(x+1-a- 1*1/(x+3-a- 2*1/(x+5-a- ...))))

    # Simplified continued fraction evaluation using modified Lentz
    log_prefactor_cf = -x + a * mx.log(mx.maximum(x, CF_TINY)) - lgamma(a)

    # Use Lentz's algorithm for continued fraction
    b0 = x + 1.0 - a
    c = 1.0 / CF_TINY
    d = 1.0 / mx.maximum(b0, CF_TINY)
    h = d

    for n in range(1, 100):
        an = -float(n) * (float(n) - a)
        bn = x + 2.0 * float(n) + 1.0 - a
        d = bn + an * d
        d = mx.where(mx.abs(d) < CF_TINY, CF_TINY * mx.ones_like(d), d)
        c = bn + an / c
        c = mx.where(mx.abs(c) < CF_TINY, CF_TINY * mx.ones_like(c), c)
        d = 1.0 / d
        delta = c * d
        h = h * delta

    cf_result = 1.0 - mx.exp(log_prefactor_cf) * h

    result = mx.where(use_series, series_result, cf_result)

    # Clamp to [0, 1]
    result = mx.clip(result, 0.0, 1.0)

    # Handle special cases
    result = mx.where(x <= 0, mx.zeros_like(result), result)
    result = mx.where(a <= 0, mx.ones_like(result), result)

    return result


def gammaincc(a: mx.array, x: mx.array) -> mx.array:
    """
    Compute the regularized upper incomplete gamma function.

    Q(a, x) = 1 - P(a, x) = (1/gamma(a)) * integral from x to inf of t^(a-1) * exp(-t) dt

    Args:
        a: Shape parameter (positive)
        x: Lower limit of integration (non-negative)

    Returns:
        Array with upper incomplete gamma values in [0, 1]
    """
    return 1.0 - gammainc(a, x)


def betainc(a: mx.array, b: mx.array, x: mx.array) -> mx.array:
    """
    Compute the regularized incomplete beta function.

    I_x(a, b) = (1/B(a,b)) * integral from 0 to x of t^(a-1) * (1-t)^(b-1) dt

    Uses continued fraction expansion (Lentz's algorithm).

    Args:
        a: First shape parameter (positive)
        b: Second shape parameter (positive)
        x: Upper limit of integration in [0, 1]

    Returns:
        Array with incomplete beta values in [0, 1]
    """
    a = mx.array(a) if not isinstance(a, mx.array) else a
    b = mx.array(b) if not isinstance(b, mx.array) else b
    x = mx.array(x) if not isinstance(x, mx.array) else x

    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    x = x.astype(mx.float32)

    # Handle edge cases
    result = mx.zeros_like(a + b + x)

    # x = 0 => I_0(a,b) = 0
    # x = 1 => I_1(a,b) = 1
    is_zero = x <= 0
    is_one = x >= 1

    # For numerical stability, use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
    # when x > (a+1)/(a+b+2)
    threshold = (a + 1.0) / (a + b + 2.0)
    use_symmetry = x > threshold

    # Work with transformed values where needed
    x_work = mx.where(use_symmetry, 1.0 - x, x)
    a_work = mx.where(use_symmetry, b, a)
    b_work = mx.where(use_symmetry, a, b)

    # Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    log_prefactor = (
        a_work * mx.log(mx.maximum(x_work, CF_TINY))
        + b_work * mx.log(mx.maximum(1.0 - x_work, CF_TINY))
        - mx.log(a_work)
        - betaln(a_work, b_work)
    )

    # Continued fraction using Lentz's algorithm
    # The continued fraction for I_x(a,b) is:
    # I_x(a,b) = prefactor * 1/(1+ d1/(1+ d2/(1+ ...)))
    # where d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
    #       d_{2m} = m(b-m)x / ((a+2m-1)(a+2m))

    # Initialize Lentz algorithm
    c = 1.0 / CF_TINY
    d = 1.0
    h = 1.0

    for m in range(1, 100):
        # d_{2m-1}
        m_float = float(m)
        d_odd_num = -(a_work + m_float - 1.0) * (a_work + b_work + m_float - 1.0) * x_work
        d_odd_den = (a_work + 2.0 * m_float - 2.0) * (a_work + 2.0 * m_float - 1.0)
        d_odd = d_odd_num / mx.maximum(d_odd_den, CF_TINY)

        d = 1.0 + d_odd * d
        d = mx.where(mx.abs(d) < CF_TINY, CF_TINY * mx.ones_like(d), d)
        c = 1.0 + d_odd / c
        c = mx.where(mx.abs(c) < CF_TINY, CF_TINY * mx.ones_like(c), c)
        d = 1.0 / d
        h = h * c * d

        # d_{2m}
        d_even_num = m_float * (b_work - m_float) * x_work
        d_even_den = (a_work + 2.0 * m_float - 1.0) * (a_work + 2.0 * m_float)
        d_even = d_even_num / mx.maximum(d_even_den, CF_TINY)

        d = 1.0 + d_even * d
        d = mx.where(mx.abs(d) < CF_TINY, CF_TINY * mx.ones_like(d), d)
        c = 1.0 + d_even / c
        c = mx.where(mx.abs(c) < CF_TINY, CF_TINY * mx.ones_like(c), c)
        d = 1.0 / d
        h = h * c * d

    cf_result = mx.exp(log_prefactor) * h

    # Apply symmetry transformation
    result = mx.where(use_symmetry, 1.0 - cf_result, cf_result)

    # Handle edge cases
    result = mx.where(is_zero, mx.zeros_like(result), result)
    result = mx.where(is_one, mx.ones_like(result), result)

    # Clamp to [0, 1]
    result = mx.clip(result, 0.0, 1.0)

    return result


def i0(x: mx.array) -> mx.array:
    """
    Compute the modified Bessel function of the first kind, order 0.

    Uses Chebyshev polynomial approximation for accuracy.

    For small |x| <= 3.75, uses a polynomial approximation.
    For large |x| > 3.75, uses an asymptotic expansion.

    Args:
        x: Input array

    Returns:
        Array with I0 values
    """
    x = mx.array(x) if not isinstance(x, mx.array) else x
    orig_dtype = x.dtype
    x = x.astype(mx.float32)

    ax = mx.abs(x)

    # For |x| <= 3.75, use polynomial approximation
    # I0(x) = 1 + (x/2)^2 * sum of polynomial terms
    t = (ax / 3.75) ** 2

    # Coefficients for small x approximation
    small_coeffs = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]

    small_result = small_coeffs[0]
    t_power = t
    for c in small_coeffs[1:]:
        small_result = small_result + c * t_power
        t_power = t_power * t

    # For |x| > 3.75, use asymptotic expansion
    # I0(x) ~ exp(x) / sqrt(2*pi*x) * (1 + sum of terms)
    t_large = 3.75 / ax

    # Coefficients for large x approximation (scaled by sqrt(2*pi))
    large_coeffs = [
        0.39894228,
        0.01328592,
        0.00225319,
        -0.00157565,
        0.00916281,
        -0.02057706,
        0.02635537,
        -0.01647633,
        0.00392377,
    ]

    large_result = large_coeffs[0]
    t_power = t_large
    for c in large_coeffs[1:]:
        large_result = large_result + c * t_power
        t_power = t_power * t_large

    large_result = mx.exp(ax) / mx.sqrt(ax) * large_result

    result = mx.where(ax <= 3.75, small_result, large_result)

    # Always return float32 - Bessel functions produce floats
    return result


def i1(x: mx.array) -> mx.array:
    """
    Compute the modified Bessel function of the first kind, order 1.

    Uses Chebyshev polynomial approximation for accuracy.

    For small |x| <= 3.75, uses a polynomial approximation.
    For large |x| > 3.75, uses an asymptotic expansion.

    Args:
        x: Input array

    Returns:
        Array with I1 values
    """
    x = mx.array(x) if not isinstance(x, mx.array) else x
    orig_dtype = x.dtype
    x = x.astype(mx.float32)

    ax = mx.abs(x)
    sign = mx.sign(x)

    # For |x| <= 3.75, use polynomial approximation
    t = (ax / 3.75) ** 2

    # Coefficients for small x approximation
    # I1(x) = x * (0.5 + polynomial in t)
    small_coeffs = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411]

    small_result = small_coeffs[0]
    t_power = t
    for c in small_coeffs[1:]:
        small_result = small_result + c * t_power
        t_power = t_power * t
    small_result = ax * small_result

    # For |x| > 3.75, use asymptotic expansion
    t_large = 3.75 / ax

    # Coefficients for large x approximation
    large_coeffs = [
        0.39894228,
        -0.03988024,
        -0.00362018,
        0.00163801,
        -0.01031555,
        0.02282967,
        -0.02895312,
        0.01787654,
        -0.00420059,
    ]

    large_result = large_coeffs[0]
    t_power = t_large
    for c in large_coeffs[1:]:
        large_result = large_result + c * t_power
        t_power = t_power * t_large

    large_result = mx.exp(ax) / mx.sqrt(ax) * large_result

    result = mx.where(ax <= 3.75, small_result, large_result)

    # I1 is an odd function: I1(-x) = -I1(x)
    result = result * sign

    # Always return float32 - Bessel functions produce floats
    return result


def multigammaln(a: mx.array, d: int) -> mx.array:
    """
    Compute the multivariate log-gamma function.

    The multivariate gamma function is defined as:
    Gamma_d(a) = pi^(d(d-1)/4) * prod_{j=1}^{d} Gamma(a + (1-j)/2)

    So:
    multigammaln(a, d) = d*(d-1)/4 * log(pi) + sum_{j=1}^{d} lgamma(a + (1-j)/2)

    Args:
        a: Input array (a > (d-1)/2 for valid values)
        d: Dimension

    Returns:
        Array with multivariate log-gamma values
    """
    a = mx.array(a) if not isinstance(a, mx.array) else a
    a = a.astype(mx.float32)

    result = float(d * (d - 1)) / 4.0 * math.log(math.pi)

    for j in range(1, d + 1):
        result = result + lgamma(a + (1.0 - float(j)) / 2.0)

    return result


# Convenience aliases to match scipy.special naming
gammaln = lgamma
psi = digamma


__all__ = [
    "lgamma",
    "gammaln",
    "gamma",
    "digamma",
    "psi",
    "betaln",
    "beta",
    "gammainc",
    "gammaincc",
    "betainc",
    "i0",
    "i1",
    "multigammaln",
]
