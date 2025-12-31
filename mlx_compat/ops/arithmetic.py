"""
Arithmetic Operations

Implements PyTorch-compatible arithmetic operations with MLX backend.
These operations build the autograd computation graph.
"""

from typing import Union, Optional
import mlx.core as mx

from ..tensor import Tensor
from ..dtype import get_dtype
from ..autograd.function import (
    AddBackward, SubBackward, MulBackward, DivBackward,
    MatmulBackward, PowBackward, SqrtBackward, ExpBackward,
    LogBackward, AbsBackward, NegBackward
)
from ..autograd.context import is_grad_enabled


def add(input: Tensor, other: Union[Tensor, float, int], *, alpha: float = 1) -> Tensor:
    """
    Add two tensors element-wise.

    out = input + alpha * other

    Args:
        input: First tensor
        other: Second tensor or scalar
        alpha: Multiplier for other (default: 1)

    Returns:
        Result tensor

    Example:
        >>> a = mlx_compat.tensor([1, 2, 3])
        >>> b = mlx_compat.tensor([4, 5, 6])
        >>> mlx_compat.add(a, b)
        tensor([5, 7, 9])
    """
    if isinstance(other, (int, float)):
        other_array = other
    elif isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        raise TypeError(f"Unsupported type for other: {type(other)}")

    if alpha != 1:
        other_array = mx.multiply(other_array, alpha)

    mlx_result = mx.add(input._mlx_array, other_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
        # Convert scalar to Tensor for autograd
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        grad_fn = AddBackward(input, other_tensor, alpha)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def sub(input: Tensor, other: Union[Tensor, float, int], *, alpha: float = 1) -> Tensor:
    """
    Subtract two tensors element-wise.

    out = input - alpha * other

    Args:
        input: First tensor
        other: Second tensor or scalar
        alpha: Multiplier for other (default: 1)

    Returns:
        Result tensor
    """
    if isinstance(other, (int, float)):
        other_array = other
    elif isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        raise TypeError(f"Unsupported type for other: {type(other)}")

    if alpha != 1:
        other_array = mx.multiply(other_array, alpha)

    mlx_result = mx.subtract(input._mlx_array, other_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        grad_fn = SubBackward(input, other_tensor, alpha)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def mul(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """
    Multiply two tensors element-wise.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Result tensor
    """
    if isinstance(other, (int, float)):
        other_array = other
    elif isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        raise TypeError(f"Unsupported type for other: {type(other)}")

    mlx_result = mx.multiply(input._mlx_array, other_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        grad_fn = MulBackward(input, other_tensor)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def div(input: Tensor, other: Union[Tensor, float, int], *, rounding_mode: Optional[str] = None) -> Tensor:
    """
    Divide two tensors element-wise.

    Args:
        input: Dividend tensor
        other: Divisor tensor or scalar
        rounding_mode: Type of rounding ('trunc', 'floor', or None)

    Returns:
        Result tensor
    """
    if isinstance(other, (int, float)):
        other_array = other
    elif isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        raise TypeError(f"Unsupported type for other: {type(other)}")

    mlx_result = mx.divide(input._mlx_array, other_array)

    if rounding_mode == 'trunc':
        # MLX doesn't have trunc, so implement it manually
        # trunc(x) = sign(x) * floor(abs(x))
        mlx_result = mx.sign(mlx_result) * mx.floor(mx.abs(mlx_result))
    elif rounding_mode == 'floor':
        mlx_result = mx.floor(mlx_result)
    elif rounding_mode is not None:
        raise ValueError(f"Invalid rounding_mode: {rounding_mode}")

    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction (only for non-rounding modes for now)
    if rounding_mode is None and is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
        other_tensor = other if isinstance(other, Tensor) else Tensor(other)
        grad_fn = DivBackward(input, other_tensor)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """
    Matrix multiplication of two tensors.

    Behavior:
    - 1D x 1D: dot product
    - 2D x 2D: matrix multiplication
    - 1D x 2D: (1, N) @ (N, M) → (M,)
    - 2D x 1D: (N, M) @ (M,) → (N,)
    - Batched: Broadcasting on batch dimensions

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Result tensor
    """
    mlx_result = mx.matmul(input._mlx_array, other._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
        grad_fn = MatmulBackward(input, other)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def mm(input: Tensor, mat2: Tensor) -> Tensor:
    """
    Matrix multiplication (2D tensors only).

    Equivalent to matmul for 2D tensors.

    Args:
        input: First matrix (2D)
        mat2: Second matrix (2D)

    Returns:
        Result matrix
    """
    if input.ndim != 2 or mat2.ndim != 2:
        raise RuntimeError(f"mm expects 2D tensors, got {input.ndim}D and {mat2.ndim}D")

    return matmul(input, mat2)


def bmm(input: Tensor, mat2: Tensor) -> Tensor:
    """
    Batch matrix multiplication.

    Both tensors must be 3D with matching batch dimensions.

    Args:
        input: First batched matrix (B, N, M)
        mat2: Second batched matrix (B, M, P)

    Returns:
        Result batched matrix (B, N, P)
    """
    if input.ndim != 3 or mat2.ndim != 3:
        raise RuntimeError(f"bmm expects 3D tensors, got {input.ndim}D and {mat2.ndim}D")

    if input.shape[0] != mat2.shape[0]:
        raise RuntimeError(f"Batch dimensions must match: {input.shape[0]} vs {mat2.shape[0]}")

    return matmul(input, mat2)


def pow(input: Tensor, exponent: Union[Tensor, float, int]) -> Tensor:
    """
    Raise tensor to a power element-wise.

    Args:
        input: Base tensor
        exponent: Exponent tensor or scalar

    Returns:
        Result tensor
    """
    if isinstance(exponent, (int, float)):
        exponent_array = exponent
    elif isinstance(exponent, Tensor):
        exponent_array = exponent._mlx_array
    else:
        raise TypeError(f"Unsupported type for exponent: {type(exponent)}")

    mlx_result = mx.power(input._mlx_array, exponent_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and (input.requires_grad or (isinstance(exponent, Tensor) and exponent.requires_grad)):
        result.requires_grad = True
        exponent_tensor = exponent if isinstance(exponent, Tensor) else Tensor(exponent)
        grad_fn = PowBackward(input, exponent_tensor, result)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def sqrt(input: Tensor) -> Tensor:
    """
    Compute square root element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor
    """
    mlx_result = mx.sqrt(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = SqrtBackward(input, result)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def exp(input: Tensor) -> Tensor:
    """
    Compute exponential element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor (e^input)
    """
    mlx_result = mx.exp(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = ExpBackward(input, result)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def log(input: Tensor) -> Tensor:
    """
    Compute natural logarithm element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor (ln(input))
    """
    mlx_result = mx.log(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = LogBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def abs(input: Tensor) -> Tensor:
    """
    Compute absolute value element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor (|input|)
    """
    mlx_result = mx.abs(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = AbsBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def neg(input: Tensor) -> Tensor:
    """
    Negate tensor element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor (-input)
    """
    mlx_result = mx.negative(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = NegBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def sin(input: Tensor) -> Tensor:
    """
    Compute sine element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor (sin(input))
    """
    mlx_result = mx.sin(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        # Gradient of sin is cos
        # For now, just mark as requiring grad
        # TODO: Add proper backward function
        result._grad_fn = None

    return result


def cos(input: Tensor) -> Tensor:
    """
    Compute cosine element-wise.

    Args:
        input: Input tensor

    Returns:
        Result tensor (cos(input))
    """
    mlx_result = mx.cos(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        # Gradient of cos is -sin
        # For now, just mark as requiring grad
        # TODO: Add proper backward function
        result._grad_fn = None

    return result


def tan(input: Tensor) -> Tensor:
    """Compute tangent element-wise."""
    mlx_result = mx.tan(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def sinh(input: Tensor) -> Tensor:
    """Compute hyperbolic sine element-wise."""
    mlx_result = mx.sinh(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def cosh(input: Tensor) -> Tensor:
    """Compute hyperbolic cosine element-wise."""
    mlx_result = mx.cosh(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def asin(input: Tensor) -> Tensor:
    """Compute arcsine element-wise."""
    mlx_result = mx.arcsin(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def acos(input: Tensor) -> Tensor:
    """Compute arccosine element-wise."""
    mlx_result = mx.arccos(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def atan(input: Tensor) -> Tensor:
    """Compute arctangent element-wise."""
    mlx_result = mx.arctan(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def atan2(input: Tensor, other: Tensor) -> Tensor:
    """Compute arctangent of input/other element-wise."""
    mlx_result = mx.arctan2(input._mlx_array, other._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
    return result


def log2(input: Tensor) -> Tensor:
    """Compute base-2 logarithm element-wise."""
    mlx_result = mx.log2(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def log10(input: Tensor) -> Tensor:
    """Compute base-10 logarithm element-wise."""
    mlx_result = mx.log10(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def log1p(input: Tensor) -> Tensor:
    """Compute log(1 + input) element-wise."""
    mlx_result = mx.log1p(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def expm1(input: Tensor) -> Tensor:
    """Compute exp(input) - 1 element-wise."""
    mlx_result = mx.expm1(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def asinh(input: Tensor) -> Tensor:
    """Compute inverse hyperbolic sine element-wise."""
    mlx_result = mx.arcsinh(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def acosh(input: Tensor) -> Tensor:
    """Compute inverse hyperbolic cosine element-wise."""
    mlx_result = mx.arccosh(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def atanh(input: Tensor) -> Tensor:
    """Compute inverse hyperbolic tangent element-wise."""
    mlx_result = mx.arctanh(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


# Aliases for PyTorch compatibility
multiply = mul
divide = div
arcsin = asin
arccos = acos
arctan = atan
arctan2 = atan2
arcsinh = asinh
arccosh = acosh
arctanh = atanh
# PyTorch 'absolute' alias for abs
absolute = abs


# ==================== Extended Math Functions ====================

def angle(input: Tensor) -> Tensor:
    """
    Compute the angle (argument) of complex numbers element-wise.

    For complex inputs, returns atan2(imag, real).
    For real inputs, returns 0 for positive values (including +inf) and pi for negative values
    (including -inf). NaN inputs return NaN.

    Args:
        input: Input tensor

    Returns:
        Tensor of angles in radians (always real-valued)
    """
    x = input._mlx_array

    # Check if input is complex
    if x.dtype == mx.complex64:
        # For complex numbers: angle(z) = atan2(imag(z), real(z))
        real_part = x.real
        imag_part = x.imag
        mlx_result = mx.arctan2(imag_part, real_part)
    else:
        # For real tensors, angle is 0 for positive, pi for negative
        # NaN should remain NaN, which we handle by checking for NaN explicitly
        pi_array = mx.full(x.shape, 3.141592653589793, dtype=x.dtype)
        zeros = mx.zeros_like(x)

        # First compute the basic result: 0 for x >= 0, pi for x < 0
        mlx_result = mx.where(x >= 0, zeros, pi_array)

        # For NaN values, the result should be NaN
        # mx.isnan returns True for NaN values
        nan_array = mx.full(x.shape, float('nan'), dtype=x.dtype)
        mlx_result = mx.where(mx.isnan(x), nan_array, mlx_result)

    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def exp2(input: Tensor) -> Tensor:
    """
    Compute 2**input element-wise.

    Args:
        input: Input tensor

    Returns:
        Tensor with 2 raised to the power of each element
    """
    mlx_result = mx.power(2.0, input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def sinc(input: Tensor) -> Tensor:
    """
    Compute sinc(x) = sin(pi*x) / (pi*x) element-wise.

    Note: sinc(0) = 1 by definition.

    Args:
        input: Input tensor

    Returns:
        Tensor with sinc values
    """
    import math
    pi = math.pi
    x = input._mlx_array

    # sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
    pi_x = pi * x
    # Use where to handle x=0 case
    mlx_result = mx.where(
        x == 0,
        mx.ones_like(x),
        mx.sin(pi_x) / pi_x
    )
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def hypot(input: Tensor, other: Tensor) -> Tensor:
    """
    Compute the hypotenuse sqrt(input^2 + other^2) element-wise.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Tensor with hypotenuse values
    """
    mlx_result = mx.sqrt(
        mx.power(input._mlx_array, 2) + mx.power(other._mlx_array, 2)
    )
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
    return result


def copysign(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """
    Create a tensor with magnitude of input and sign of other.

    Args:
        input: Tensor providing magnitude
        other: Tensor or scalar providing sign

    Returns:
        Tensor with copysigned values
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    # copysign(x, y) = abs(x) * sign(y)
    mlx_result = mx.abs(input._mlx_array) * mx.sign(other_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
    return result


def heaviside(input: Tensor, values: Tensor) -> Tensor:
    """
    Compute the Heaviside step function element-wise.

    heaviside(x) = 0 if x < 0, 1 if x > 0, values if x == 0

    Args:
        input: Input tensor
        values: Values to use where input == 0

    Returns:
        Tensor with Heaviside values
    """
    x = input._mlx_array
    vals = values._mlx_array

    mlx_result = mx.where(
        x < 0,
        mx.zeros_like(x),
        mx.where(x > 0, mx.ones_like(x), vals)
    )
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or values.requires_grad):
        result.requires_grad = True
    return result


def fmax(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """
    Compute element-wise maximum, ignoring NaNs.

    If one input is NaN and the other is not, returns the non-NaN value.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Tensor with element-wise maximum
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    # fmax ignores NaN: if one is NaN, return the other
    x = input._mlx_array
    y = other_array

    mlx_result = mx.where(
        mx.isnan(x), y,
        mx.where(mx.isnan(y), x, mx.maximum(x, y))
    )
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
    return result


def fmin(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """
    Compute element-wise minimum, ignoring NaNs.

    If one input is NaN and the other is not, returns the non-NaN value.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Tensor with element-wise minimum
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    # fmin ignores NaN: if one is NaN, return the other
    x = input._mlx_array
    y = other_array

    mlx_result = mx.where(
        mx.isnan(x), y,
        mx.where(mx.isnan(y), x, mx.minimum(x, y))
    )
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
    return result


def erfc(input: Tensor) -> Tensor:
    """
    Compute the complementary error function element-wise.

    erfc(x) = 1 - erf(x)

    Args:
        input: Input tensor

    Returns:
        Tensor with complementary error function values
    """
    mlx_result = 1.0 - mx.erf(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def lgamma(input: Tensor) -> Tensor:
    """
    Compute the natural logarithm of the absolute value of the gamma function.

    Args:
        input: Input tensor

    Returns:
        Tensor with log-gamma values
    """
    import numpy as np
    from scipy import special

    # Use scipy for lgamma
    np_result = special.gammaln(np.array(input._mlx_array))
    mlx_result = mx.array(np_result)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def digamma(input: Tensor) -> Tensor:
    """
    Compute the digamma function (logarithmic derivative of gamma) element-wise.

    Also known as psi function.

    Args:
        input: Input tensor

    Returns:
        Tensor with digamma values
    """
    import numpy as np
    from scipy import special

    # Use scipy for digamma
    np_result = special.digamma(np.array(input._mlx_array))
    mlx_result = mx.array(np_result)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def i0(input: Tensor) -> Tensor:
    """
    Compute the modified Bessel function of the first kind, order 0.

    Args:
        input: Input tensor

    Returns:
        Tensor with I0 values
    """
    import numpy as np
    from scipy import special

    # Use scipy for i0
    np_result = special.i0(np.array(input._mlx_array))
    mlx_result = mx.array(np_result)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def isreal(input: Tensor) -> Tensor:
    """
    Check if tensor elements are real (have zero imaginary part).

    For real-valued tensors, always returns True for all elements.

    Args:
        input: Input tensor

    Returns:
        Boolean tensor
    """
    # For real tensors, everything is real
    mlx_result = mx.ones(input.shape, dtype=mx.bool_)
    return Tensor._from_mlx_array(mlx_result)


def isneginf(input: Tensor) -> Tensor:
    """
    Check if tensor elements are negative infinity.

    Args:
        input: Input tensor

    Returns:
        Boolean tensor
    """
    x = input._mlx_array
    mlx_result = mx.logical_and(mx.isinf(x), x < 0)
    return Tensor._from_mlx_array(mlx_result)


def isposinf(input: Tensor) -> Tensor:
    """
    Check if tensor elements are positive infinity.

    Args:
        input: Input tensor

    Returns:
        Boolean tensor
    """
    x = input._mlx_array
    mlx_result = mx.logical_and(mx.isinf(x), x > 0)
    return Tensor._from_mlx_array(mlx_result)


def float_power(input: Tensor, exponent: Union[Tensor, float, int]) -> Tensor:
    """
    Raise input to the power of exponent, promoting to float.

    Unlike pow, this always returns a float tensor.

    Args:
        input: Base tensor
        exponent: Exponent tensor or scalar

    Returns:
        Float tensor with power values
    """
    # Cast to float first
    x = input._mlx_array.astype(mx.float32)

    if isinstance(exponent, Tensor):
        exp_array = exponent._mlx_array.astype(mx.float32)
    else:
        exp_array = float(exponent)

    mlx_result = mx.power(x, exp_array)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or (isinstance(exponent, Tensor) and exponent.requires_grad)):
        result.requires_grad = True
    return result


def logaddexp2(input: Tensor, other: Tensor) -> Tensor:
    """
    Compute log2(2^input + 2^other) element-wise.

    This is useful for adding probabilities in log2-space.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Tensor with log2-sum-exp values
    """
    # log2(2^a + 2^b) = max(a,b) + log2(1 + 2^(min(a,b) - max(a,b)))
    x = input._mlx_array
    y = other._mlx_array

    max_val = mx.maximum(x, y)
    min_val = mx.minimum(x, y)

    # Use log1p for numerical stability
    import math
    log2_e = 1.0 / math.log(2)

    mlx_result = max_val + mx.log1p(mx.power(2.0, min_val - max_val)) * log2_e
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
    return result


def nextafter(input: Tensor, other: Tensor) -> Tensor:
    """
    Return the next representable floating-point value after input towards other.

    Args:
        input: Input tensor
        other: Direction tensor

    Returns:
        Tensor with next representable values
    """
    import numpy as np

    # Use numpy for nextafter
    np_result = np.nextafter(
        np.array(input._mlx_array),
        np.array(other._mlx_array)
    )
    mlx_result = mx.array(np_result)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
    return result


def frexp(input: Tensor):
    """
    Decompose input into mantissa and exponent.

    Returns (mantissa, exponent) where input = mantissa * 2^exponent.

    Args:
        input: Input tensor

    Returns:
        Tuple of (mantissa tensor, exponent tensor)
    """
    import numpy as np

    # Use numpy for frexp
    mantissa, exponent = np.frexp(np.array(input._mlx_array))
    mantissa_result = Tensor._from_mlx_array(mx.array(mantissa))
    exponent_result = Tensor._from_mlx_array(mx.array(exponent))

    return mantissa_result, exponent_result


def ldexp(input: Tensor, other: Tensor) -> Tensor:
    """
    Multiply input by 2 raised to the power of other.

    ldexp(x, n) = x * 2^n

    Args:
        input: Mantissa tensor
        other: Exponent tensor (integer)

    Returns:
        Tensor with scaled values
    """
    import numpy as np

    # Use numpy for ldexp
    np_result = np.ldexp(
        np.array(input._mlx_array),
        np.array(other._mlx_array).astype(np.int32)
    )
    mlx_result = mx.array(np_result)
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def xlogy(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """
    Compute input * log(other) element-wise.

    Returns 0 when input is 0, even if other is 0 or negative.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Tensor with xlogy values
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    x = input._mlx_array
    y = other_array

    # xlogy(x, y) = x * log(y), but 0 * log(y) = 0 even when log(y) is undefined
    mlx_result = mx.where(
        x == 0,
        mx.zeros_like(x),
        x * mx.log(y)
    )
    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True
    return result
