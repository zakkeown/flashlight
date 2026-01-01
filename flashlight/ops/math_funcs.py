"""
Math Utility Functions

Implements PyTorch-compatible math utility functions with MLX backend.
"""

from typing import Optional, Union

import mlx.core as mx

from ..autograd.context import is_grad_enabled
from ..distributions._constants import FLOAT32_MAX
from ..tensor import Tensor


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
    arr = input._mlx_array

    # Use mx.clip when both bounds are provided (single fused operation)
    if min is not None and max is not None:
        result_array = mx.clip(arr, min, max)
    elif min is not None:
        result_array = mx.maximum(arr, min)
    elif max is not None:
        result_array = mx.minimum(arr, max)
    else:
        result_array = arr

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
        scale = 10**decimals
        result_array = mx.round(input._mlx_array * scale) / scale

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


_trunc_kernel = None
_TRUNC_METAL_THRESHOLD = 1_000_000  # Use Metal kernel for tensors > 1M elements


def _get_trunc_kernel():
    """Get or create cached Metal kernel for trunc operation."""
    global _trunc_kernel
    if _trunc_kernel is None:
        _trunc_kernel = mx.fast.metal_kernel(
            name="trunc_kernel",
            input_names=["inp"],
            output_names=["out"],
            source="""
                uint idx = thread_position_in_grid.x;
                if (idx >= inp_shape[0]) return;
                out[idx] = metal::trunc(inp[idx]);
            """,
        )
    return _trunc_kernel


def trunc(input: Tensor) -> Tensor:
    """
    Truncate input elements towards zero.

    Args:
        input: Input tensor

    Returns:
        Truncated tensor
    """
    arr = input._mlx_array
    original_dtype = arr.dtype
    original_shape = arr.shape

    # For large tensors, use custom Metal kernel with metal::trunc
    # (faster than int32 cast for large data)
    if arr.size > _TRUNC_METAL_THRESHOLD:
        kernel = _get_trunc_kernel()
        flat = mx.reshape(arr, (-1,))
        outputs = kernel(
            inputs=[flat],
            template=[("T", arr.dtype)],
            grid=(flat.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[flat.shape],
            output_dtypes=[arr.dtype],
        )
        result_array = mx.reshape(outputs[0], original_shape)
    else:
        # For smaller tensors, int32 cast has lower overhead
        # Cast to int32 and back - naturally truncates towards zero
        result_array = arr.astype(mx.int32).astype(original_dtype)

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
        input._mlx_array >= 0, mx.floor(input._mlx_array), mx.ceil(input._mlx_array)
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
    result_array = mx.logical_and(mx.logical_or(a, b), mx.logical_not(mx.logical_and(a, b)))
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

    if is_grad_enabled() and (
        input.requires_grad
        or end.requires_grad
        or (isinstance(weight, Tensor) and weight.requires_grad)
    ):
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

    if is_grad_enabled() and (
        input.requires_grad or tensor1.requires_grad or tensor2.requires_grad
    ):
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

    if is_grad_enabled() and (
        input.requires_grad or tensor1.requires_grad or tensor2.requires_grad
    ):
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
    result_array = (
        input._mlx_array
        - trunc(Tensor._from_mlx_array(input._mlx_array / other_array))._mlx_array * other_array
    )
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


def nan_to_num(
    input: Tensor, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None
) -> Tensor:
    """
    Replace NaN, positive infinity, and negative infinity values.

    Args:
        input: Input tensor
        nan: Value to replace NaN with
        posinf: Value to replace positive infinity with (default: max float)
        neginf: Value to replace negative infinity with (default: min float)

    Returns:
        Tensor with replaced values

    Pure MLX implementation.
    """
    result_array = input._mlx_array

    # Replace NaN
    result_array = mx.where(mx.isnan(result_array), nan, result_array)

    # Replace positive infinity with float32 max
    if posinf is None:
        posinf = FLOAT32_MAX
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array > 0), posinf, result_array
    )

    # Replace negative infinity with negative float32 max
    if neginf is None:
        neginf = -FLOAT32_MAX
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array < 0), neginf, result_array
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


def diff(
    input: Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
) -> Tensor:
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


def nan_to_num_(
    input: Tensor, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None
) -> Tensor:
    """In-place version of nan_to_num. Pure MLX implementation."""
    result_array = input._mlx_array
    result_array = mx.where(mx.isnan(result_array), nan, result_array)
    if posinf is None:
        posinf = FLOAT32_MAX
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array > 0), posinf, result_array
    )
    if neginf is None:
        neginf = -FLOAT32_MAX
    result_array = mx.where(
        mx.logical_and(mx.isinf(result_array), result_array < 0), neginf, result_array
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

    Uses the Euclidean algorithm implemented in pure MLX.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        GCD tensor
    """
    a = mx.abs(input._mlx_array).astype(mx.int32)
    b = mx.abs(other._mlx_array).astype(mx.int32)

    # Euclidean algorithm: gcd(a, b) = gcd(b, a % b) until b = 0
    # For 32-bit integers, at most 32 iterations are needed
    for _ in range(32):
        # Where b != 0, compute a % b; otherwise keep a
        remainder = mx.remainder(a, mx.maximum(b, mx.array(1, dtype=mx.int32)))
        # Update: a = b, b = remainder (where b != 0)
        mask = b != 0
        new_a = mx.where(mask, b, a)
        new_b = mx.where(mask, remainder, mx.zeros_like(b))
        a = new_a
        b = new_b

    return Tensor._from_mlx_array(a)


def gcd_(input: Tensor, other: Tensor) -> Tensor:
    """In-place version of gcd."""
    result = gcd(input, other)
    input._mlx_array = result._mlx_array
    return input


def lcm(input: Tensor, other: Tensor) -> Tensor:
    """
    Least common multiple of input and other.

    Uses the formula: lcm(a, b) = |a * b| / gcd(a, b)

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        LCM tensor
    """
    a = mx.abs(input._mlx_array).astype(mx.int32)
    b = mx.abs(other._mlx_array).astype(mx.int32)

    # Use the relationship: lcm(a, b) = |a * b| / gcd(a, b)
    g = gcd(input, other)._mlx_array

    # Handle zero case: lcm(0, x) = lcm(x, 0) = 0
    # Divide before multiply to avoid overflow
    result = mx.where(g == 0, mx.zeros_like(a), (a // g) * b)

    return Tensor._from_mlx_array(result)


def lcm_(input: Tensor, other: Tensor) -> Tensor:
    """In-place version of lcm."""
    result = lcm(input, other)
    input._mlx_array = result._mlx_array
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
    y_arr = y._mlx_array
    ndim = len(y_arr.shape)

    # Normalize dim to positive
    if dim < 0:
        dim = ndim + dim

    # Move the integration axis to the last position for easier slicing
    if dim != ndim - 1:
        axes = list(range(ndim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        y_arr = mx.transpose(y_arr, axes=axes)
    else:
        axes = None

    # Trapezoidal rule: sum((y[i] + y[i+1]) / 2 * dx[i])
    if x is not None:
        x_arr = x._mlx_array
        # Compute dx from x values
        d = x_arr[1:] - x_arr[:-1]  # Differences
        # Broadcast d to match y shape
        # d shape is (n-1,), we need to expand it
        for _ in range(len(y_arr.shape) - 1):
            d = mx.expand_dims(d, axis=0)
        integral = mx.sum((y_arr[..., :-1] + y_arr[..., 1:]) / 2.0 * d, axis=-1)
    else:
        # Uniform spacing
        integral = mx.sum((y_arr[..., :-1] + y_arr[..., 1:]) / 2.0 * dx, axis=-1)

    result = Tensor._from_mlx_array(integral)

    if is_grad_enabled() and y.requires_grad:
        result.requires_grad = True

    return result


# Alias
trapz = trapezoid


def cumulative_trapezoid(
    y: Tensor, x: Optional[Tensor] = None, dx: float = 1.0, dim: int = -1
) -> Tensor:
    """
    Cumulative trapezoidal rule integration.

    Args:
        y: Values to integrate
        x: Points at which y is sampled (optional)
        dx: Spacing between sample points (used if x is None)
        dim: Dimension along which to integrate

    Returns:
        Cumulative integrated values (length n-1 along dim)
    """
    y_arr = y._mlx_array
    ndim = len(y_arr.shape)

    # Normalize dim to positive
    if dim < 0:
        dim = ndim + dim

    # Move the integration axis to the last position for easier slicing
    if dim != ndim - 1:
        axes = list(range(ndim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        y_arr = mx.transpose(y_arr, axes=axes)
        need_transpose_back = True
    else:
        axes = None
        need_transpose_back = False

    # Trapezoidal rule for each interval: (y[i] + y[i+1]) / 2 * dx
    if x is not None:
        x_arr = x._mlx_array
        # Compute dx from x values
        d = x_arr[1:] - x_arr[:-1]  # Differences
        # Broadcast d to match y shape
        for _ in range(len(y_arr.shape) - 1):
            d = mx.expand_dims(d, axis=0)
        intervals = (y_arr[..., :-1] + y_arr[..., 1:]) / 2.0 * d
    else:
        # Uniform spacing
        intervals = (y_arr[..., :-1] + y_arr[..., 1:]) / 2.0 * dx

    # Cumulative sum along last axis
    cumulative = mx.cumsum(intervals, axis=-1)

    # Transpose back if needed
    if need_transpose_back:
        cumulative = mx.transpose(cumulative, axes=axes)

    result = Tensor._from_mlx_array(cumulative)

    if is_grad_enabled() and y.requires_grad:
        result.requires_grad = True

    return result


def gradient(
    input: Tensor, spacing: float = 1.0, dim: Optional[int] = None, edge_order: int = 1
) -> tuple:
    """
    Compute the gradient of an array using finite differences.

    Args:
        input: Input tensor
        spacing: Spacing between samples
        dim: Dimension along which to compute gradient (None for all)
        edge_order: Edge order for boundary (1 or 2)

    Returns:
        Tuple of gradient tensors (one per dimension if dim is None, one element if dim is specified)
    """
    arr = input._mlx_array

    def _gradient_1d(arr_1d: mx.array, h: float, edge_order: int) -> mx.array:
        """Compute gradient along a 1D array."""
        n = arr_1d.shape[0]
        if n < 2:
            return mx.zeros_like(arr_1d)

        # Interior points: central difference (f[i+1] - f[i-1]) / (2*h)
        # We need to handle this for each position

        # Allocate result
        result = mx.zeros_like(arr_1d)

        if n == 2:
            # Only two points: forward/backward difference
            grad = (arr_1d[1] - arr_1d[0]) / h
            result = mx.full_like(arr_1d, grad)
            return result

        # Central differences for interior points
        central = (arr_1d[2:] - arr_1d[:-2]) / (2.0 * h)

        if edge_order == 1:
            # First-order forward/backward differences at edges
            left_edge = (arr_1d[1] - arr_1d[0]) / h
            right_edge = (arr_1d[-1] - arr_1d[-2]) / h
        else:  # edge_order == 2
            # Second-order forward/backward differences at edges
            # Left: (-3*f[0] + 4*f[1] - f[2]) / (2*h)
            left_edge = (-3.0 * arr_1d[0] + 4.0 * arr_1d[1] - arr_1d[2]) / (2.0 * h)
            # Right: (3*f[-1] - 4*f[-2] + f[-3]) / (2*h)
            right_edge = (3.0 * arr_1d[-1] - 4.0 * arr_1d[-2] + arr_1d[-3]) / (2.0 * h)

        # Concatenate: [left_edge, central..., right_edge]
        result = mx.concatenate([mx.array([left_edge]), central, mx.array([right_edge])])

        return result

    def _gradient_along_axis(arr: mx.array, axis: int, h: float, edge_order: int) -> mx.array:
        """Compute gradient along a specific axis."""
        ndim = arr.ndim
        n = arr.shape[axis]

        if n < 2:
            return mx.zeros_like(arr)

        # Move axis to last position, compute gradient, move back
        arr_moved = mx.moveaxis(arr, axis, -1)
        original_shape = arr_moved.shape

        # Flatten all but last dimension
        flat_shape = (-1, n)
        arr_flat = arr_moved.reshape(flat_shape)

        # Compute gradient for each row
        num_rows = arr_flat.shape[0]
        results = []
        for i in range(num_rows):
            row_grad = _gradient_1d(arr_flat[i], h, edge_order)
            results.append(row_grad)

        result_flat = mx.stack(results, axis=0)
        result_moved = result_flat.reshape(original_shape)
        result = mx.moveaxis(result_moved, -1, axis)

        return result

    if dim is not None:
        # Compute gradient along specific dimension
        result = _gradient_along_axis(arr, dim, spacing, edge_order)
        return (Tensor._from_mlx_array(result),)
    else:
        # Compute gradient along all dimensions
        ndim = arr.ndim
        results = []
        for axis in range(ndim):
            result = _gradient_along_axis(arr, axis, spacing, edge_order)
            results.append(Tensor._from_mlx_array(result))
        return tuple(results)


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
    arr = x._mlx_array
    n = arr.shape[0]
    num_cols = N if N is not None else n

    # Create powers array [0, 1, 2, ..., num_cols-1]
    if increasing:
        powers = mx.arange(num_cols, dtype=mx.float32)
    else:
        powers = mx.arange(num_cols - 1, -1, -1, dtype=mx.float32)

    # Reshape for broadcasting: x is (n,), powers is (num_cols,)
    # Result should be (n, num_cols) where result[i,j] = x[i] ** powers[j]
    x_col = arr.reshape(-1, 1).astype(mx.float32)  # (n, 1)
    powers_row = powers.reshape(1, -1)  # (1, num_cols)

    result = mx.power(x_col, powers_row)
    return Tensor._from_mlx_array(result.astype(arr.dtype))


def unravel_index(indices: Tensor, shape: tuple) -> tuple:
    """
    Convert flat indices to coordinate arrays.

    Args:
        indices: Flat indices
        shape: Shape of the array

    Returns:
        Tuple of coordinate tensors
    """
    idx = indices._mlx_array.astype(mx.int64)
    coords = []

    # Compute strides (from last dimension to first)
    # For shape (3, 4, 5): strides are [20, 5, 1]
    strides = []
    stride = 1
    for dim_size in reversed(shape):
        strides.append(stride)
        stride *= dim_size
    strides = list(reversed(strides))

    # Extract coordinates using divmod
    remaining = idx
    for stride in strides:
        coord = mx.floor_divide(remaining, stride)
        remaining = mx.remainder(remaining, stride)
        coords.append(Tensor._from_mlx_array(coord))

    return tuple(coords)


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
    # Build indices directly by iterating - the total number of elements
    # in lower triangular can be computed, then we use vectorized ops
    # For efficiency, we compute counts per row and use concatenation

    row_indices_list = []
    col_indices_list = []

    for r in range(row):
        # For row r, lower triangular includes columns where c <= r + offset
        # But we also need c < col
        max_c = min(r + offset + 1, col)
        if max_c > 0:
            row_indices_list.append(mx.full((max_c,), r, dtype=mx.int64))
            col_indices_list.append(mx.arange(max_c, dtype=mx.int64))

    if row_indices_list:
        row_indices = mx.concatenate(row_indices_list)
        col_indices = mx.concatenate(col_indices_list)
    else:
        row_indices = mx.array([], dtype=mx.int64)
        col_indices = mx.array([], dtype=mx.int64)

    # Stack to match PyTorch's return format (2, N)
    stacked = mx.stack([row_indices, col_indices], axis=0)
    return Tensor._from_mlx_array(stacked)


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
    # Build indices directly by iterating
    # For efficiency, we compute counts per row and use concatenation

    row_indices_list = []
    col_indices_list = []

    for r in range(row):
        # For row r, upper triangular includes columns where c >= r + offset
        # But we also need c < col
        min_c = max(r + offset, 0)
        if min_c < col:
            num_cols = col - min_c
            row_indices_list.append(mx.full((num_cols,), r, dtype=mx.int64))
            col_indices_list.append(mx.arange(min_c, col, dtype=mx.int64))

    if row_indices_list:
        row_indices = mx.concatenate(row_indices_list)
        col_indices = mx.concatenate(col_indices_list)
    else:
        row_indices = mx.array([], dtype=mx.int64)
        col_indices = mx.array([], dtype=mx.int64)

    # Stack to match PyTorch's return format (2, N)
    stacked = mx.stack([row_indices, col_indices], axis=0)
    return Tensor._from_mlx_array(stacked)


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
    "clamp",
    "clip",
    "clamp_min",
    "clamp_max",
    "clamp_min_",
    "clamp_max_",
    "clip_",
    "floor",
    "ceil",
    "round",
    "trunc",
    "frac",
    "fix",
    "fix_",
    "sign",
    "signbit",
    "isnan",
    "isinf",
    "isfinite",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "reciprocal",
    "rsqrt",
    "square",
    "lerp",
    "addcmul",
    "addcdiv",
    "fmod",
    "remainder",
    "cumsum",
    "cumprod",
    "deg2rad",
    "rad2deg",
    "deg2rad_",
    "rad2deg_",
    "nan_to_num",
    "nan_to_num_",
    "count_nonzero",
    "diff",
    # In-place variants
    "negative_",
    "square_",
    # New math functions
    "logit",
    "logit_",
    "sgn",
    "rsub",
    "subtract",
    "floor_divide",
    "true_divide",
    "gcd",
    "gcd_",
    "lcm",
    "lcm_",
    "trapezoid",
    "trapz",
    "cumulative_trapezoid",
    "gradient",
    "is_same_size",
    "is_signed",
    "vander",
    "unravel_index",
    "tril_indices",
    "triu_indices",
    "range_",
    "bitwise_left_shift",
    "bitwise_right_shift",
]
