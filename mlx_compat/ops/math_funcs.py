"""
Math Utility Functions

Implements PyTorch-compatible math utility functions with MLX backend.
"""

from typing import Optional, Union
import mlx.core as mx
from ..tensor import Tensor
from ..autograd.context import is_grad_enabled


def clamp(input: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
    """
    Clamp tensor values to a range.

    Args:
        input: Input tensor
        min: Minimum value (None means no lower bound)
        max: Maximum value (None means no upper bound)

    Returns:
        Clamped tensor
    """
    result_array = input._mlx_array

    if min is not None:
        result_array = mx.maximum(result_array, min)
    if max is not None:
        result_array = mx.minimum(result_array, max)

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# Alias for clamp
clip = clamp


def clamp_min(input: Tensor, min: float) -> Tensor:
    """
    Clamp tensor values to have minimum value.

    Args:
        input: Input tensor
        min: Minimum value

    Returns:
        Clamped tensor
    """
    result_array = mx.maximum(input._mlx_array, min)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def clamp_max(input: Tensor, max: float) -> Tensor:
    """
    Clamp tensor values to have maximum value.

    Args:
        input: Input tensor
        max: Maximum value

    Returns:
        Clamped tensor
    """
    result_array = mx.minimum(input._mlx_array, max)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def clamp_min_(input: Tensor, min: float) -> Tensor:
    """
    In-place version of clamp_min.

    Args:
        input: Input tensor (modified in-place)
        min: Minimum value

    Returns:
        Modified input tensor
    """
    result_array = mx.maximum(input._mlx_array, min)
    input._mlx_array = result_array
    return input


def clamp_max_(input: Tensor, max: float) -> Tensor:
    """
    In-place version of clamp_max.

    Args:
        input: Input tensor (modified in-place)
        max: Maximum value

    Returns:
        Modified input tensor
    """
    result_array = mx.minimum(input._mlx_array, max)
    input._mlx_array = result_array
    return input


def clip_(input: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
    """
    In-place version of clip/clamp.

    Args:
        input: Input tensor (modified in-place)
        min: Minimum value (None means no lower bound)
        max: Maximum value (None means no upper bound)

    Returns:
        Modified input tensor
    """
    result_array = input._mlx_array
    if min is not None:
        result_array = mx.maximum(result_array, min)
    if max is not None:
        result_array = mx.minimum(result_array, max)
    input._mlx_array = result_array
    return input


def floor(input: Tensor) -> Tensor:
    """
    Floor of input elements.

    Args:
        input: Input tensor

    Returns:
        Tensor with floor of each element
    """
    result_array = mx.floor(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def ceil(input: Tensor) -> Tensor:
    """
    Ceiling of input elements.

    Args:
        input: Input tensor

    Returns:
        Tensor with ceiling of each element
    """
    result_array = mx.ceil(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def round(input: Tensor, decimals: int = 0) -> Tensor:
    """
    Round input elements to nearest integer or decimal place.

    Args:
        input: Input tensor
        decimals: Number of decimal places to round to

    Returns:
        Rounded tensor
    """
    if decimals == 0:
        result_array = mx.round(input._mlx_array)
    else:
        scale = 10 ** decimals
        result_array = mx.round(input._mlx_array * scale) / scale

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def trunc(input: Tensor) -> Tensor:
    """
    Truncate input elements towards zero.

    Args:
        input: Input tensor

    Returns:
        Truncated tensor
    """
    # trunc(x) = floor(x) for x >= 0, ceil(x) for x < 0
    result_array = mx.where(
        input._mlx_array >= 0,
        mx.floor(input._mlx_array),
        mx.ceil(input._mlx_array)
    )
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def frac(input: Tensor) -> Tensor:
    """
    Fractional part of input elements.

    Args:
        input: Input tensor

    Returns:
        Tensor with fractional parts
    """
    result_array = input._mlx_array - trunc(input)._mlx_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# fix is an alias for trunc
fix = trunc


def fix_(input: Tensor) -> Tensor:
    """
    In-place version of fix (trunc towards zero).

    Args:
        input: Input tensor (modified in-place)

    Returns:
        Modified input tensor
    """
    result_array = mx.where(
        input._mlx_array >= 0,
        mx.floor(input._mlx_array),
        mx.ceil(input._mlx_array)
    )
    input._mlx_array = result_array
    return input


def sign(input: Tensor) -> Tensor:
    """
    Sign of input elements (-1, 0, or 1).

    Args:
        input: Input tensor

    Returns:
        Tensor with signs
    """
    result_array = mx.sign(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def isnan(input: Tensor) -> Tensor:
    """
    Check for NaN values.

    Args:
        input: Input tensor

    Returns:
        Boolean tensor indicating NaN positions
    """
    result_array = mx.isnan(input._mlx_array)
    return Tensor._from_mlx_array(result_array)


def isinf(input: Tensor) -> Tensor:
    """
    Check for infinite values.

    Args:
        input: Input tensor

    Returns:
        Boolean tensor indicating infinite positions
    """
    result_array = mx.isinf(input._mlx_array)
    return Tensor._from_mlx_array(result_array)


def isfinite(input: Tensor) -> Tensor:
    """
    Check for finite values (not NaN or Inf).

    Args:
        input: Input tensor

    Returns:
        Boolean tensor indicating finite positions
    """
    result_array = mx.isfinite(input._mlx_array)
    return Tensor._from_mlx_array(result_array)


def logical_and(input: Tensor, other: Tensor) -> Tensor:
    """
    Logical AND of two tensors.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Boolean tensor with logical AND results
    """
    result_array = mx.logical_and(input._mlx_array, other._mlx_array)
    return Tensor._from_mlx_array(result_array)


def logical_or(input: Tensor, other: Tensor) -> Tensor:
    """
    Logical OR of two tensors.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Boolean tensor with logical OR results
    """
    result_array = mx.logical_or(input._mlx_array, other._mlx_array)
    return Tensor._from_mlx_array(result_array)


def logical_not(input: Tensor) -> Tensor:
    """
    Logical NOT of a tensor.

    Args:
        input: Input tensor

    Returns:
        Boolean tensor with logical NOT results
    """
    result_array = mx.logical_not(input._mlx_array)
    return Tensor._from_mlx_array(result_array)


def logical_xor(input: Tensor, other: Tensor) -> Tensor:
    """
    Logical XOR of two tensors.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Boolean tensor with logical XOR results
    """
    # XOR = (A OR B) AND NOT (A AND B)
    a = input._mlx_array
    b = other._mlx_array
    result_array = mx.logical_and(
        mx.logical_or(a, b),
        mx.logical_not(mx.logical_and(a, b))
    )
    return Tensor._from_mlx_array(result_array)


def reciprocal(input: Tensor) -> Tensor:
    """
    Reciprocal (1/x) of input elements.

    Args:
        input: Input tensor

    Returns:
        Tensor with reciprocals
    """
    result_array = mx.reciprocal(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def rsqrt(input: Tensor) -> Tensor:
    """
    Reciprocal square root (1/sqrt(x)) of input elements.

    Args:
        input: Input tensor

    Returns:
        Tensor with reciprocal square roots
    """
    result_array = mx.rsqrt(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def square(input: Tensor) -> Tensor:
    """
    Square of input elements.

    Args:
        input: Input tensor

    Returns:
        Tensor with squared elements
    """
    result_array = mx.square(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def lerp(input: Tensor, end: Tensor, weight: Union[float, Tensor]) -> Tensor:
    """
    Linear interpolation: input + weight * (end - input).

    Args:
        input: Start tensor
        end: End tensor
        weight: Interpolation weight (0 = input, 1 = end)

    Returns:
        Interpolated tensor
    """
    if isinstance(weight, Tensor):
        weight_array = weight._mlx_array
    else:
        weight_array = weight

    result_array = input._mlx_array + weight_array * (end._mlx_array - input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or end.requires_grad or
                               (isinstance(weight, Tensor) and weight.requires_grad)):
        result.requires_grad = True

    return result


def addcmul(input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0) -> Tensor:
    """
    Computes input + value * tensor1 * tensor2.

    Args:
        input: Input tensor
        tensor1: First tensor to multiply
        tensor2: Second tensor to multiply
        value: Multiplier for product

    Returns:
        Result tensor
    """
    result_array = input._mlx_array + value * tensor1._mlx_array * tensor2._mlx_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or tensor1.requires_grad or tensor2.requires_grad):
        result.requires_grad = True

    return result


def addcdiv(input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0) -> Tensor:
    """
    Computes input + value * tensor1 / tensor2.

    Args:
        input: Input tensor
        tensor1: Numerator tensor
        tensor2: Denominator tensor
        value: Multiplier for quotient

    Returns:
        Result tensor
    """
    result_array = input._mlx_array + value * tensor1._mlx_array / tensor2._mlx_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or tensor1.requires_grad or tensor2.requires_grad):
        result.requires_grad = True

    return result


def fmod(input: Tensor, other) -> Tensor:
    """
    Floating-point remainder of division.

    Args:
        input: Dividend tensor
        other: Divisor (tensor or scalar)

    Returns:
        Remainder tensor
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    # fmod is the remainder after truncation division
    result_array = input._mlx_array - trunc(Tensor._from_mlx_array(input._mlx_array / other_array))._mlx_array * other_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def remainder(input: Tensor, other) -> Tensor:
    """
    Python-style remainder of division (follows sign of divisor).

    Args:
        input: Dividend tensor
        other: Divisor (tensor or scalar)

    Returns:
        Remainder tensor
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    # Use modulo operator for Python-style remainder
    result_array = input._mlx_array % other_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def cumsum(input: Tensor, dim: int, dtype=None) -> Tensor:
    """
    Cumulative sum along a dimension.

    Args:
        input: Input tensor
        dim: Dimension to sum along
        dtype: Optional output dtype

    Returns:
        Tensor with cumulative sums
    """
    result_array = mx.cumsum(input._mlx_array, axis=dim)
    if dtype is not None:
        from ..dtype import get_mlx_dtype
        result_array = result_array.astype(get_mlx_dtype(dtype))
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def cumprod(input: Tensor, dim: int, dtype=None) -> Tensor:
    """
    Cumulative product along a dimension.

    Args:
        input: Input tensor
        dim: Dimension to multiply along
        dtype: Optional output dtype

    Returns:
        Tensor with cumulative products
    """
    result_array = mx.cumprod(input._mlx_array, axis=dim)
    if dtype is not None:
        from ..dtype import get_mlx_dtype
        result_array = result_array.astype(get_mlx_dtype(dtype))
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def deg2rad(input: Tensor) -> Tensor:
    """
    Convert angles from degrees to radians.

    Args:
        input: Input tensor in degrees

    Returns:
        Tensor in radians
    """
    import math
    result_array = input._mlx_array * (math.pi / 180.0)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def rad2deg(input: Tensor) -> Tensor:
    """
    Convert angles from radians to degrees.

    Args:
        input: Input tensor in radians

    Returns:
        Tensor in degrees
    """
    import math
    result_array = input._mlx_array * (180.0 / math.pi)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def nan_to_num(input: Tensor, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None) -> Tensor:
    """
    Replace NaN, positive infinity, and negative infinity values.

    Args:
        input: Input tensor
        nan: Value to replace NaN with
        posinf: Value to replace positive infinity with (default: max float)
        neginf: Value to replace negative infinity with (default: min float)

    Returns:
        Tensor with replaced values
    """
    import numpy as np
    result_array = input._mlx_array

    # Replace NaN
    result_array = mx.where(mx.isnan(result_array), nan, result_array)

    # Replace positive infinity
    if posinf is None:
        posinf = float(np.finfo(np.float32).max)
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array > 0),
        posinf,
        result_array
    )

    # Replace negative infinity
    if neginf is None:
        neginf = float(np.finfo(np.float32).min)
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array < 0),
        neginf,
        result_array
    )

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def signbit(input: Tensor) -> Tensor:
    """
    Check if sign bit is set (negative).

    Args:
        input: Input tensor

    Returns:
        Boolean tensor indicating negative values
    """
    result_array = input._mlx_array < 0
    return Tensor._from_mlx_array(result_array)


def count_nonzero(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Count non-zero elements.

    Args:
        input: Input tensor
        dim: Dimension to count along (None for all elements)

    Returns:
        Tensor with count of non-zero elements
    """
    nonzero_mask = input._mlx_array != 0
    if dim is None:
        result_array = mx.sum(nonzero_mask.astype(mx.int32))
    else:
        result_array = mx.sum(nonzero_mask.astype(mx.int32), axis=dim)
    return Tensor._from_mlx_array(result_array.astype(mx.int64))


def diff(input: Tensor, n: int = 1, dim: int = -1, prepend: Optional[Tensor] = None, append: Optional[Tensor] = None) -> Tensor:
    """
    Compute n-th discrete difference along a dimension.

    Args:
        input: Input tensor
        n: Number of times to differentiate
        dim: Dimension to compute difference along
        prepend: Values to prepend before differencing
        append: Values to append before differencing

    Returns:
        Tensor with differences
    """
    result_array = input._mlx_array

    # Handle prepend/append
    if prepend is not None:
        result_array = mx.concatenate([prepend._mlx_array, result_array], axis=dim)
    if append is not None:
        result_array = mx.concatenate([result_array, append._mlx_array], axis=dim)

    # Compute differences n times
    for _ in range(n):
        ndim = len(result_array.shape)
        dim_normalized = dim if dim >= 0 else ndim + dim

        # Create slices for [1:] and [:-1] along dim
        slices_end = [slice(None)] * ndim
        slices_start = [slice(None)] * ndim
        slices_end[dim_normalized] = slice(1, None)
        slices_start[dim_normalized] = slice(None, -1)

        result_array = result_array[tuple(slices_end)] - result_array[tuple(slices_start)]

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# In-place variants
# =============================================================================

def deg2rad_(input: Tensor) -> Tensor:
    """In-place version of deg2rad."""
    import math
    input._mlx_array = input._mlx_array * (math.pi / 180.0)
    return input


def rad2deg_(input: Tensor) -> Tensor:
    """In-place version of rad2deg."""
    import math
    input._mlx_array = input._mlx_array * (180.0 / math.pi)
    return input


def nan_to_num_(input: Tensor, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None) -> Tensor:
    """In-place version of nan_to_num."""
    import numpy as np
    result_array = input._mlx_array
    result_array = mx.where(mx.isnan(result_array), nan, result_array)
    if posinf is None:
        posinf = float(np.finfo(np.float32).max)
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array > 0),
        posinf, result_array
    )
    if neginf is None:
        neginf = float(np.finfo(np.float32).min)
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array < 0),
        neginf, result_array
    )
    input._mlx_array = result_array
    return input


def negative_(input: Tensor) -> Tensor:
    """In-place negation."""
    input._mlx_array = -input._mlx_array
    return input


def square_(input: Tensor) -> Tensor:
    """In-place square."""
    input._mlx_array = mx.square(input._mlx_array)
    return input


# =============================================================================
# New math functions (Phase 7.3)
# =============================================================================

def logit(input: Tensor, eps: Optional[float] = None) -> Tensor:
    """
    Compute the logit (inverse of sigmoid) of input.

    logit(x) = log(x / (1 - x))

    Args:
        input: Input tensor with values in (0, 1)
        eps: Small value for numerical stability

    Returns:
        Logit of input
    """
    x = input._mlx_array
    if eps is not None:
        x = mx.clip(x, eps, 1 - eps)
    result_array = mx.log(x / (1 - x))
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def logit_(input: Tensor, eps: Optional[float] = None) -> Tensor:
    """In-place version of logit."""
    x = input._mlx_array
    if eps is not None:
        x = mx.clip(x, eps, 1 - eps)
    input._mlx_array = mx.log(x / (1 - x))
    return input


def sgn(input: Tensor) -> Tensor:
    """
    Sign function that handles complex numbers.
    For real: same as sign.
    For complex: returns x / |x|.

    Args:
        input: Input tensor

    Returns:
        Sign tensor
    """
    # For now, treat as alias to sign (MLX doesn't support complex well)
    result_array = mx.sign(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def rsub(input: Tensor, other, alpha: float = 1.0) -> Tensor:
    """
    Reverse subtraction: other - alpha * input.

    Args:
        input: Input tensor
        other: Value to subtract from
        alpha: Multiplier for input

    Returns:
        Result tensor
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = other_array - alpha * input._mlx_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def subtract(input: Tensor, other, alpha: float = 1.0) -> Tensor:
    """
    Subtraction with alpha: input - alpha * other.

    Args:
        input: Input tensor
        other: Value to subtract
        alpha: Multiplier for other

    Returns:
        Result tensor
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = input._mlx_array - alpha * other_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def floor_divide(input: Tensor, other) -> Tensor:
    """
    Floor division: floor(input / other).

    Args:
        input: Dividend tensor
        other: Divisor (tensor or scalar)

    Returns:
        Floor division result
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.floor(input._mlx_array / other_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def true_divide(input: Tensor, other) -> Tensor:
    """
    True division (same as regular division).

    Args:
        input: Dividend tensor
        other: Divisor (tensor or scalar)

    Returns:
        Division result
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = input._mlx_array / other_array
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def gcd(input: Tensor, other: Tensor) -> Tensor:
    """
    Greatest common divisor of input and other.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        GCD tensor
    """
    import numpy as np
    input_np = np.array(input._mlx_array).astype(np.int64)
    other_np = np.array(other._mlx_array).astype(np.int64)
    result_np = np.gcd(input_np, other_np)
    return Tensor._from_mlx_array(mx.array(result_np))


def gcd_(input: Tensor, other: Tensor) -> Tensor:
    """In-place version of gcd."""
    import numpy as np
    input_np = np.array(input._mlx_array).astype(np.int64)
    other_np = np.array(other._mlx_array).astype(np.int64)
    result_np = np.gcd(input_np, other_np)
    input._mlx_array = mx.array(result_np)
    return input


def lcm(input: Tensor, other: Tensor) -> Tensor:
    """
    Least common multiple of input and other.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        LCM tensor
    """
    import numpy as np
    input_np = np.array(input._mlx_array).astype(np.int64)
    other_np = np.array(other._mlx_array).astype(np.int64)
    result_np = np.lcm(input_np, other_np)
    return Tensor._from_mlx_array(mx.array(result_np))


def lcm_(input: Tensor, other: Tensor) -> Tensor:
    """In-place version of lcm."""
    import numpy as np
    input_np = np.array(input._mlx_array).astype(np.int64)
    other_np = np.array(other._mlx_array).astype(np.int64)
    result_np = np.lcm(input_np, other_np)
    input._mlx_array = mx.array(result_np)
    return input


def trapezoid(y: Tensor, x: Optional[Tensor] = None, dx: float = 1.0, dim: int = -1) -> Tensor:
    """
    Trapezoidal rule integration.

    Args:
        y: Values to integrate
        x: Points at which y is sampled (optional)
        dx: Spacing between sample points (used if x is None)
        dim: Dimension along which to integrate

    Returns:
        Integrated values
    """
    import numpy as np
    y_np = np.array(y._mlx_array)
    x_np = np.array(x._mlx_array) if x is not None else None
    result_np = np.trapezoid(y_np, x=x_np, dx=dx, axis=dim)
    result = Tensor._from_mlx_array(mx.array(result_np))

    if is_grad_enabled() and y.requires_grad:
        result.requires_grad = True

    return result


# Alias
trapz = trapezoid


def cumulative_trapezoid(y: Tensor, x: Optional[Tensor] = None, dx: float = 1.0, dim: int = -1) -> Tensor:
    """
    Cumulative trapezoidal rule integration.

    Args:
        y: Values to integrate
        x: Points at which y is sampled (optional)
        dx: Spacing between sample points (used if x is None)
        dim: Dimension along which to integrate

    Returns:
        Cumulative integrated values
    """
    import numpy as np
    from scipy import integrate
    y_np = np.array(y._mlx_array)
    x_np = np.array(x._mlx_array) if x is not None else None
    result_np = integrate.cumulative_trapezoid(y_np, x=x_np, dx=dx, axis=dim)
    result = Tensor._from_mlx_array(mx.array(result_np))

    if is_grad_enabled() and y.requires_grad:
        result.requires_grad = True

    return result


def gradient(input: Tensor, spacing: float = 1.0, dim: Optional[int] = None, edge_order: int = 1) -> tuple:
    """
    Compute the gradient of an array.

    Args:
        input: Input tensor
        spacing: Spacing between samples
        dim: Dimension along which to compute gradient (None for all)
        edge_order: Edge order for boundary (1 or 2)

    Returns:
        Tuple of gradient tensors (one per dimension if dim is None, one element if dim is specified)
    """
    import numpy as np
    input_np = np.array(input._mlx_array)

    if dim is not None:
        # Compute gradient along specific dimension - still returns a tuple with one element
        result_np = np.gradient(input_np, spacing, axis=dim, edge_order=edge_order)
        return (Tensor._from_mlx_array(mx.array(result_np)),)
    else:
        # Compute gradient along all dimensions
        results = np.gradient(input_np, spacing, edge_order=edge_order)
        if isinstance(results, np.ndarray):
            # 1D input returns a single array, wrap in tuple
            return (Tensor._from_mlx_array(mx.array(results)),)
        return tuple(Tensor._from_mlx_array(mx.array(r)) for r in results)


def is_same_size(input: Tensor, other: Tensor) -> bool:
    """
    Check if two tensors have the same size.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        True if tensors have the same shape
    """
    return input.shape == other.shape


def is_signed(input: Tensor) -> bool:
    """
    Check if tensor dtype is a signed type.

    Args:
        input: Input tensor

    Returns:
        True if dtype is signed
    """
    dtype = input._mlx_array.dtype
    return dtype in (mx.int8, mx.int16, mx.int32, mx.int64, mx.float16, mx.float32, mx.bfloat16)


def vander(x: Tensor, N: Optional[int] = None, increasing: bool = False) -> Tensor:
    """
    Generate a Vandermonde matrix.

    Args:
        x: 1-D input tensor
        N: Number of columns (default: len(x))
        increasing: If True, powers increase left to right

    Returns:
        Vandermonde matrix
    """
    import numpy as np
    x_np = np.array(x._mlx_array)
    result_np = np.vander(x_np, N=N, increasing=increasing)
    return Tensor._from_mlx_array(mx.array(result_np))


def unravel_index(indices: Tensor, shape: tuple) -> tuple:
    """
    Convert flat indices to coordinate arrays.

    Args:
        indices: Flat indices
        shape: Shape of the array

    Returns:
        Tuple of coordinate tensors
    """
    import numpy as np
    indices_np = np.array(indices._mlx_array)
    coords = np.unravel_index(indices_np, shape)
    return tuple(Tensor._from_mlx_array(mx.array(c)) for c in coords)


def tril_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """
    Return indices for lower-triangular part of matrix.

    Args:
        row: Number of rows
        col: Number of columns
        offset: Diagonal offset

    Returns:
        Tensor of shape (2, N) with row indices in [0] and column indices in [1]
    """
    import numpy as np
    r, c = np.tril_indices(row, offset, col)
    # PyTorch returns a stacked tensor of shape (2, N), not a tuple
    stacked = np.stack([r, c], axis=0)
    return Tensor._from_mlx_array(mx.array(stacked))


def triu_indices(row: int, col: int, offset: int = 0) -> Tensor:
    """
    Return indices for upper-triangular part of matrix.

    Args:
        row: Number of rows
        col: Number of columns
        offset: Diagonal offset

    Returns:
        Tensor of shape (2, N) with row indices in [0] and column indices in [1]
    """
    import numpy as np
    r, c = np.triu_indices(row, offset, col)
    # PyTorch returns a stacked tensor of shape (2, N), not a tuple
    stacked = np.stack([r, c], axis=0)
    return Tensor._from_mlx_array(mx.array(stacked))


def range_(*args, **kwargs):
    """
    Deprecated alias for arange.

    Use torch.arange instead.
    """
    import warnings
    warnings.warn("torch.range is deprecated, use torch.arange instead", DeprecationWarning)
    from ..factories import arange
    return arange(*args, **kwargs)


def bitwise_left_shift(input: Tensor, other) -> Tensor:
    """
    Left bit shift.

    Args:
        input: Input tensor
        other: Number of bits to shift

    Returns:
        Shifted tensor
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.left_shift(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def bitwise_right_shift(input: Tensor, other) -> Tensor:
    """
    Right bit shift.

    Args:
        input: Input tensor
        other: Number of bits to shift

    Returns:
        Shifted tensor
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.right_shift(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


__all__ = [
    'clamp', 'clip',
    'clamp_min', 'clamp_max', 'clamp_min_', 'clamp_max_', 'clip_',
    'floor', 'ceil', 'round', 'trunc', 'frac',
    'fix', 'fix_',
    'sign', 'signbit',
    'isnan', 'isinf', 'isfinite',
    'logical_and', 'logical_or', 'logical_not', 'logical_xor',
    'reciprocal', 'rsqrt', 'square',
    'lerp', 'addcmul', 'addcdiv',
    'fmod', 'remainder',
    'cumsum', 'cumprod',
    'deg2rad', 'rad2deg', 'deg2rad_', 'rad2deg_',
    'nan_to_num', 'nan_to_num_',
    'count_nonzero',
    'diff',
    # In-place variants
    'negative_', 'square_',
    # New math functions
    'logit', 'logit_',
    'sgn', 'rsub', 'subtract',
    'floor_divide', 'true_divide',
    'gcd', 'gcd_', 'lcm', 'lcm_',
    'trapezoid', 'trapz', 'cumulative_trapezoid',
    'gradient',
    'is_same_size', 'is_signed',
    'vander', 'unravel_index',
    'tril_indices', 'triu_indices',
    'range_',
    'bitwise_left_shift', 'bitwise_right_shift',
]
