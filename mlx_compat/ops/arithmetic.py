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
        from ..autograd.function import SinBackward
        result.requires_grad = True
        grad_fn = SinBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

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
        from ..autograd.function import CosBackward
        result.requires_grad = True
        grad_fn = CosBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

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

    Implementation uses the Lanczos approximation which is accurate to about
    15 significant digits for positive arguments.

    Args:
        input: Input tensor

    Returns:
        Tensor with log-gamma values
    """
    x = input._mlx_array.astype(mx.float32)

    # Lanczos approximation coefficients (g=7, n=9)
    # These coefficients give good precision for float32
    g = 7.0
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    # For x < 0.5, use reflection formula: lgamma(x) = log(pi/sin(pi*x)) - lgamma(1-x)
    reflect_mask = x < 0.5

    # Work with reflected values for x < 0.5
    x_work = mx.where(reflect_mask, 1.0 - x, x)

    # Lanczos approximation: lgamma(x) = (x - 0.5) * log(x + g - 0.5) - (x + g - 0.5) + 0.5*log(2*pi) + log(A_g(x))
    # where A_g(x) = c0 + sum(c_i / (x + i - 1))

    # Compute A_g(x)
    ag = mx.array(c[0], dtype=mx.float32)
    for i in range(1, len(c)):
        ag = ag + mx.array(c[i], dtype=mx.float32) / (x_work + mx.array(i - 1, dtype=mx.float32))

    # Compute lgamma
    t = x_work + g - 0.5
    half_log_2pi = 0.9189385332046727  # 0.5 * log(2 * pi)

    lgamma_positive = (x_work - 0.5) * mx.log(t) - t + half_log_2pi + mx.log(ag)

    # Apply reflection formula for x < 0.5
    # lgamma(x) = log(pi) - log(|sin(pi * x)|) - lgamma(1 - x)
    log_pi = mx.array(1.1447298858494002, dtype=mx.float32)  # log(pi)
    sin_pi_x = mx.sin(mx.array(3.141592653589793, dtype=mx.float32) * x)
    log_sin_pi_x = mx.log(mx.abs(sin_pi_x) + mx.array(1e-38, dtype=mx.float32))

    lgamma_negative = log_pi - log_sin_pi_x - lgamma_positive

    mlx_result = mx.where(reflect_mask, lgamma_negative, lgamma_positive)

    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def digamma(input: Tensor) -> Tensor:
    """
    Compute the digamma function (logarithmic derivative of gamma) element-wise.

    Also known as psi function. Implementation uses asymptotic expansion
    for large x and recurrence relation for small x.

    Args:
        input: Input tensor

    Returns:
        Tensor with digamma values
    """
    x = input._mlx_array.astype(mx.float32)

    # Asymptotic expansion coefficients (Bernoulli numbers)
    # digamma(x) ≈ log(x) - 1/(2x) - sum(B_{2k}/(2k*x^{2k}))
    B2 = 1.0 / 6.0  # B_2 = 1/6
    B4 = -1.0 / 30.0  # B_4 = -1/30
    B6 = 1.0 / 42.0  # B_6 = 1/42
    B8 = -1.0 / 30.0  # B_8 = -1/30
    B10 = 5.0 / 66.0  # B_10 = 5/66

    # Use recurrence for small x: digamma(x) = digamma(x+1) - 1/x
    # Shift x up until it's large enough for asymptotic expansion
    min_x = 6.0  # Threshold for asymptotic expansion

    # Count how many times we need to shift
    shift_needed = mx.maximum(mx.ceil(min_x - x), mx.array(0.0, dtype=mx.float32))

    # Compute the shifted value
    x_shifted = x + shift_needed

    # Asymptotic expansion for digamma(x_shifted)
    x2 = x_shifted * x_shifted
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x6 * x2
    x10 = x8 * x2

    psi = mx.log(x_shifted) - 0.5 / x_shifted
    psi = psi - B2 / (2.0 * x2)
    psi = psi - B4 / (4.0 * x4)
    psi = psi - B6 / (6.0 * x6)
    psi = psi - B8 / (8.0 * x8)
    psi = psi - B10 / (10.0 * x10)

    # Apply recurrence relation backwards: digamma(x) = digamma(x+n) - sum(1/(x+k)) for k=0..n-1
    # We need to subtract 1/(x+k) for k = 0, 1, ..., shift_needed-1
    max_shift = int(mx.max(shift_needed).item())
    for k in range(max_shift):
        k_float = mx.array(k, dtype=mx.float32)
        # Only subtract if this shift was needed for this element
        should_subtract = (shift_needed > k_float).astype(mx.float32)
        psi = psi - should_subtract / (x + k_float)

    # Handle negative x using reflection formula:
    # digamma(x) = digamma(1-x) + pi * cot(pi * x)
    negative_mask = x < 0.0

    # For negative x, compute digamma(1-x) + pi*cot(pi*x)
    # cot(pi*x) = cos(pi*x) / sin(pi*x)
    pi = mx.array(3.141592653589793, dtype=mx.float32)
    sin_pi_x = mx.sin(pi * x)
    cos_pi_x = mx.cos(pi * x)
    cot_pi_x = cos_pi_x / (sin_pi_x + mx.array(1e-38, dtype=mx.float32))

    # Compute digamma(1-x) using the same shifted approach
    x_neg = 1.0 - x
    shift_neg = mx.maximum(mx.ceil(min_x - x_neg), mx.array(0.0, dtype=mx.float32))
    x_neg_shifted = x_neg + shift_neg

    x2_neg = x_neg_shifted * x_neg_shifted
    x4_neg = x2_neg * x2_neg
    x6_neg = x4_neg * x2_neg
    x8_neg = x6_neg * x2_neg
    x10_neg = x8_neg * x2_neg

    psi_neg = mx.log(x_neg_shifted) - 0.5 / x_neg_shifted
    psi_neg = psi_neg - B2 / (2.0 * x2_neg)
    psi_neg = psi_neg - B4 / (4.0 * x4_neg)
    psi_neg = psi_neg - B6 / (6.0 * x6_neg)
    psi_neg = psi_neg - B8 / (8.0 * x8_neg)
    psi_neg = psi_neg - B10 / (10.0 * x10_neg)

    max_shift_neg = int(mx.max(shift_neg).item())
    for k in range(max_shift_neg):
        k_float = mx.array(k, dtype=mx.float32)
        should_subtract = (shift_neg > k_float).astype(mx.float32)
        psi_neg = psi_neg - should_subtract / (x_neg + k_float)

    psi_reflection = psi_neg + pi * cot_pi_x

    mlx_result = mx.where(negative_mask, psi_reflection, psi)

    result = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def i0(input: Tensor) -> Tensor:
    """
    Compute the modified Bessel function of the first kind, order 0.

    Implementation uses polynomial approximations for small and large |x|.
    Based on Abramowitz and Stegun formulas.

    Args:
        input: Input tensor

    Returns:
        Tensor with I0 values
    """
    x = input._mlx_array.astype(mx.float32)
    ax = mx.abs(x)

    # For |x| <= 3.75, use polynomial approximation
    # I0(x) ≈ 1 + 3.5156229*(x/3.75)^2 + 3.0899424*(x/3.75)^4 + ...
    small_threshold = 3.75

    t = ax / small_threshold
    t2 = t * t

    # Coefficients for small x approximation
    small_result = (1.0 +
                    t2 * (3.5156229 +
                    t2 * (3.0899424 +
                    t2 * (1.2067492 +
                    t2 * (0.2659732 +
                    t2 * (0.0360768 +
                    t2 * 0.0045813))))))

    # For |x| > 3.75, use asymptotic expansion
    # I0(x) ≈ exp(x) / sqrt(x) * polynomial(3.75/x)
    t_large = small_threshold / ax
    t_large2 = t_large * t_large

    # Coefficients for large x approximation
    poly_large = (0.39894228 +
                  t_large * (0.01328592 +
                  t_large * (0.00225319 +
                  t_large * (-0.00157565 +
                  t_large * (0.00916281 +
                  t_large * (-0.02057706 +
                  t_large * (0.02635537 +
                  t_large * (-0.01647633 +
                  t_large * 0.00392377))))))))

    large_result = mx.exp(ax) / mx.sqrt(ax) * poly_large

    # Choose based on threshold
    mlx_result = mx.where(ax <= small_threshold, small_result, large_result)

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

    Implementation uses the property that for positive floats, incrementing/decrementing
    the bit representation gives the next/previous representable value.

    Args:
        input: Input tensor
        other: Direction tensor

    Returns:
        Tensor with next representable values
    """
    x = input._mlx_array.astype(mx.float32)
    y = other._mlx_array.astype(mx.float32)

    # Handle the case where x == y (return x unchanged)
    equal_mask = mx.equal(x, y)

    # For float32, machine epsilon is 2^-23 ≈ 1.19e-7
    eps = mx.array(1.1920929e-7, dtype=mx.float32)  # float32 machine epsilon

    # Smallest positive float32 (subnormal)
    min_positive = mx.array(1.4012985e-45, dtype=mx.float32)

    # Direction masks
    going_up = mx.greater(y, x)
    going_down = mx.less(y, x)

    # Handle x == 0 specially: nextafter(0, y) where y > 0 should be min_positive
    # and nextafter(0, y) where y < 0 should be -min_positive
    # Note: we use mx.where to avoid subnormal * 1.0 = 0 issue
    is_zero = mx.equal(x, mx.array(0.0, dtype=mx.float32))
    zero_result = mx.where(going_up, min_positive, mx.where(going_down, -min_positive, mx.zeros_like(x)))

    # For non-zero x: use ulp-based stepping
    # ULP (unit in last place) = eps * |x| for normal numbers
    # For subnormals, ulp = min_positive
    abs_x = mx.abs(x)
    ulp = mx.maximum(eps * abs_x, min_positive)

    # Step in the direction using where to avoid multiplication issues
    nonzero_result = mx.where(going_up, x + ulp, mx.where(going_down, x - ulp, x))

    # Select between zero and nonzero cases
    result = mx.where(is_zero, zero_result, nonzero_result)

    # If x == y, return x unchanged
    mlx_result = mx.where(equal_mask, x, result)

    result_tensor = Tensor._from_mlx_array(mlx_result)
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result_tensor.requires_grad = True
    return result_tensor


def frexp(input: Tensor):
    """
    Decompose input into mantissa and exponent.

    Returns (mantissa, exponent) where input = mantissa * 2^exponent,
    with 0.5 <= |mantissa| < 1.0.

    Args:
        input: Input tensor

    Returns:
        Tuple of (mantissa tensor, exponent tensor)
    """
    arr = input._mlx_array.astype(mx.float32)

    # Handle zero case
    is_zero = mx.equal(arr, mx.array(0.0, dtype=mx.float32))

    # For non-zero values:
    # exponent = floor(log2(|x|)) + 1
    # mantissa = x / 2^exponent

    abs_arr = mx.abs(arr)

    # Compute log2 for non-zero values (add tiny to avoid log(0))
    tiny = mx.array(1e-45, dtype=mx.float32)
    log2_val = mx.log2(mx.maximum(abs_arr, tiny))

    # Exponent: floor(log2(|x|)) + 1
    exponent = mx.floor(log2_val).astype(mx.int32) + 1

    # Mantissa: x / 2^exponent
    # Convert exponent to float for power calculation
    exp_float = exponent.astype(mx.float32)
    mantissa = arr / mx.power(mx.array(2.0, dtype=mx.float32), exp_float)

    # Handle zero: mantissa = 0, exponent = 0
    mantissa = mx.where(is_zero, mx.zeros_like(mantissa), mantissa)
    exponent = mx.where(is_zero, mx.zeros_like(exponent), exponent)

    mantissa_result = Tensor._from_mlx_array(mantissa)
    exponent_result = Tensor._from_mlx_array(exponent)

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
    # Pure MLX implementation: x * 2^n = x * pow(2, n)
    x = input._mlx_array
    n = other._mlx_array.astype(mx.float32)

    # Compute 2^n using exp2 or power
    two_to_n = mx.power(mx.array(2.0, dtype=mx.float32), n)

    mlx_result = x * two_to_n
    result = Tensor._from_mlx_array(mlx_result.astype(x.dtype))
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
