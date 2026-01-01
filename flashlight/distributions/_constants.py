"""
Numerical constants for probability distributions.

Provides standardized epsilon values for numerical stability across all
distribution implementations. Using consistent values improves reproducibility
and makes it easier to reason about numerical precision.
"""

import mlx.core as mx

# =============================================================================
# Float32 Machine Constants
# =============================================================================

# Machine epsilon for float32: smallest x such that 1.0 + x != 1.0
# This is 2^-23 ≈ 1.1920929e-7
FLOAT32_EPS = 1.1920929e-7

# Smallest positive normal float32 value (avoids subnormal territory)
# This is 2^-126 ≈ 1.175494e-38
FLOAT32_TINY = 1.175494e-38

# Largest finite float32 value
# This is (2 - 2^-23) * 2^127 ≈ 3.4028235e+38
FLOAT32_MAX = 3.4028235e38

# =============================================================================
# Numerical Stability Constants
# =============================================================================

# Epsilon for log operations to avoid log(0) = -inf
# Using FLOAT32_TINY as the smallest positive float32 value that doesn't cause
# numerical issues when passed to log()
LOG_EPSILON = FLOAT32_TINY

# Epsilon for probability clamping to avoid exact 0 or 1
# sqrt(float32_eps) ≈ 3.45e-4, but we use 1e-7 for better precision
# while still avoiding numerical issues in probability computations
PROB_EPSILON = 1e-7

# Uniform sampling bounds to avoid exact 0 and 1 for Gumbel trick
# These values ensure -log(-log(u)) doesn't produce inf
UNIFORM_LOW = 1e-10
UNIFORM_HIGH = 1.0 - 1e-10

# Threshold for switching between direct and asymptotic computation
# Used in special functions like erfc, gammainc, etc.
ASYMPTOTIC_THRESHOLD = 4.0

# Tiny value for continued fraction algorithms (Lentz's method)
# Slightly larger than FLOAT32_TINY to provide some headroom
CF_TINY = 1e-30


def xlogy(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute x * log(y) with proper handling of x=0.

    Returns 0 when x=0, even if y=0 or y<0 (which would make log(y) undefined).
    This is the correct mathematical limit: lim_{x->0} x*log(y) = 0.

    Args:
        x: First array (the multiplier)
        y: Second array (argument to log)

    Returns:
        Array with x * log(y), with 0 where x=0
    """
    # Broadcast to common shape for proper zero handling
    x = mx.array(x) if not isinstance(x, mx.array) else x
    y = mx.array(y) if not isinstance(y, mx.array) else y
    # Use zeros_like on the result of x * log(y) to ensure proper shape/dtype
    log_y = mx.log(mx.maximum(y, FLOAT32_TINY))  # Avoid log(0) during computation
    result = x * log_y
    return mx.where(x == 0, mx.zeros_like(result), result)


def xlog1py(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute x * log1p(y) with proper handling of x=0.

    Returns 0 when x=0, even if y=-1 (which would make log1p(y) = -inf).
    This is the correct mathematical limit: lim_{x->0} x*log1p(y) = 0.

    More numerically stable than x * log(1 + y) for small y.

    Args:
        x: First array (the multiplier)
        y: Second array (argument to log1p)

    Returns:
        Array with x * log1p(y), with 0 where x=0
    """
    x = mx.array(x) if not isinstance(x, mx.array) else x
    y = mx.array(y) if not isinstance(y, mx.array) else y
    result = x * mx.log1p(y)
    return mx.where(x == 0, mx.zeros_like(result), result)


def safe_log(x: mx.array, eps: float = LOG_EPSILON) -> mx.array:
    """
    Compute log(x) with numerical safety for small positive values.

    Args:
        x: Input array (should be positive)
        eps: Minimum value to clamp x to (default: LOG_EPSILON)

    Returns:
        log(max(x, eps))
    """
    return mx.log(mx.maximum(x, eps))


def safe_exp(x: mx.array, max_val: float = 88.0) -> mx.array:
    """
    Compute exp(x) with overflow protection.

    Clamps input to avoid overflow (exp(88) ≈ 1.65e38 is near float32 max).

    Args:
        x: Input array
        max_val: Maximum input value (default: 88.0, safe for float32)

    Returns:
        exp(min(x, max_val))
    """
    return mx.exp(mx.minimum(x, max_val))


def log_sum_exp(x: mx.array, axis: int = -1, keepdims: bool = False) -> mx.array:
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Uses the identity: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

    Args:
        x: Input array
        axis: Axis along which to compute
        keepdims: Whether to keep reduced dimensions

    Returns:
        Log-sum-exp values
    """
    return mx.logsumexp(x, axis=axis, keepdims=keepdims)
