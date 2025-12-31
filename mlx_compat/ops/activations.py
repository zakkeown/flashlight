"""
Activation Functions

Implements PyTorch-compatible activation functions with MLX backend.
"""

from typing import Optional
import mlx.core as mx
import mlx.nn as nn

from ..tensor import Tensor
from ..autograd.function import (
    ReLUBackward, SigmoidBackward, TanhBackward,
    SoftmaxBackward, LogSoftmaxBackward, SiLUBackward,
    LeakyReLUBackward, ELUBackward, GELUBackward
)
from ..autograd.context import is_grad_enabled


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply Rectified Linear Unit activation.

    relu(x) = max(0, x)

    Args:
        input: Input tensor
        inplace: If True, modify in-place (ignored in MLX, creates new tensor)

    Returns:
        Result tensor
    """
    if inplace:
        # MLX doesn't support true in-place ops, but we'll modify the tensor's data
        pass  # Will be handled below

    mlx_result = mx.maximum(input._mlx_array, 0)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = ReLUBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def gelu(input: Tensor, approximate: str = 'none') -> Tensor:
    """
    Apply Gaussian Error Linear Unit activation.

    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal

    Args:
        input: Input tensor
        approximate: Approximation method ('none' or 'tanh')

    Returns:
        Result tensor
    """
    # MLX has a built-in gelu that uses the exact formula
    mlx_result = nn.gelu(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = GELUBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def sigmoid(input: Tensor) -> Tensor:
    """
    Apply sigmoid activation.

    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        input: Input tensor

    Returns:
        Result tensor
    """
    mlx_result = nn.sigmoid(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = SigmoidBackward(input, result)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def tanh(input: Tensor) -> Tensor:
    """
    Apply hyperbolic tangent activation.

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        input: Input tensor

    Returns:
        Result tensor
    """
    mlx_result = mx.tanh(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = TanhBackward(input, result)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    """
    Apply softmax activation along a dimension.

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        input: Input tensor
        dim: Dimension along which to apply softmax (default: last dimension)
        _stacklevel: Internal parameter for warning stack level
        dtype: Optional dtype to cast input to before computation

    Returns:
        Result tensor
    """
    if dim is None:
        dim = -1

    # Cast input if dtype specified
    input_array = input._mlx_array
    if dtype is not None:
        # Convert PyTorch dtype to MLX dtype if needed
        from ..dtype import to_mlx_dtype
        mlx_dtype = to_mlx_dtype(dtype)
        input_array = input_array.astype(mlx_dtype)

    mlx_result = nn.softmax(input_array, axis=dim)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = SoftmaxBackward(input, result, dim)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    """
    Apply log(softmax(x)) along a dimension.

    More numerically stable than computing log(softmax(x)) separately.

    Args:
        input: Input tensor
        dim: Dimension along which to apply log_softmax (default: last dimension)
        _stacklevel: Internal parameter for warning stack level
        dtype: Optional dtype to cast input to before computation

    Returns:
        Result tensor
    """
    if dim is None:
        dim = -1

    # Cast input if dtype specified
    input_array = input._mlx_array
    if dtype is not None:
        # Convert PyTorch dtype to MLX dtype if needed
        from ..dtype import to_mlx_dtype
        mlx_dtype = to_mlx_dtype(dtype)
        input_array = input_array.astype(mlx_dtype)

    # Optimized: use MLX's fused logsumexp (2 ops instead of 6)
    # log_softmax(x) = x - logsumexp(x)
    # logsumexp is numerically stable and fused for performance
    mlx_result = input_array - mx.logsumexp(input_array, axis=dim, keepdims=True)

    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = LogSoftmaxBackward(input, result, dim)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply Sigmoid Linear Unit (SiLU/Swish) activation.

    silu(x) = x * sigmoid(x)

    Also known as Swish.

    Args:
        input: Input tensor
        inplace: If True, modify in-place (ignored in MLX, creates new tensor)

    Returns:
        Result tensor
    """
    mlx_result = nn.silu(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = SiLUBackward(input)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    """
    Apply Leaky ReLU activation.

    leaky_relu(x) = max(0, x) + negative_slope * min(0, x)

    Args:
        input: Input tensor
        negative_slope: Slope for negative values
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    mlx_result = nn.leaky_relu(input._mlx_array, negative_slope=negative_slope)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = LeakyReLUBackward(input, negative_slope)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """
    Apply Exponential Linear Unit activation.

    elu(x) = x if x > 0 else alpha * (exp(x) - 1)

    Args:
        input: Input tensor
        alpha: Scale factor for negative values
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    mlx_result = nn.elu(input._mlx_array, alpha=alpha)
    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = ELUBackward(input, result, alpha)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """
    Apply Continuously Differentiable Exponential Linear Unit activation.

    celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))

    Args:
        input: Input tensor
        alpha: Scale parameter (default: 1.0)
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    x = input._mlx_array
    mlx_result = mx.maximum(x, 0) + mx.minimum(0, alpha * (mx.exp(x / alpha) - 1))
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def selu(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply Scaled Exponential Linear Unit activation.

    selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))

    With scale = 1.0507009873554804934193349852946 and
    alpha = 1.6732632423543772848170429916717

    Args:
        input: Input tensor
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    # SELU constants from the paper
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    x = input._mlx_array
    mlx_result = scale * mx.where(x > 0, x, alpha * (mx.exp(x) - 1))
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    """
    Apply HardTanh activation (clipping).

    hardtanh(x) = clip(x, min_val, max_val)

    Args:
        input: Input tensor
        min_val: Minimum value (default: -1.0)
        max_val: Maximum value (default: 1.0)
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    mlx_result = mx.clip(input._mlx_array, min_val, max_val)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def hardtanh_(input: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    """In-place hardtanh (returns new tensor in MLX)."""
    return hardtanh(input, min_val, max_val, inplace=True)


def hardshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    """
    Apply Hard Shrinkage activation.

    hardshrink(x) = x if |x| > lambd else 0

    Args:
        input: Input tensor
        lambd: Lambda threshold (default: 0.5)

    Returns:
        Result tensor
    """
    x = input._mlx_array
    mlx_result = mx.where(mx.abs(x) > lambd, x, 0)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def softshrink(input: Tensor, lambd: float = 0.5) -> Tensor:
    """
    Apply Soft Shrinkage activation.

    softshrink(x) = x - lambd if x > lambd
                  = x + lambd if x < -lambd
                  = 0 otherwise

    Args:
        input: Input tensor
        lambd: Lambda threshold (default: 0.5)

    Returns:
        Result tensor
    """
    x = input._mlx_array
    mlx_result = mx.where(
        x > lambd,
        x - lambd,
        mx.where(x < -lambd, x + lambd, 0)
    )
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def tanhshrink(input: Tensor) -> Tensor:
    """
    Apply Tanh Shrinkage activation.

    tanhshrink(x) = x - tanh(x)

    Args:
        input: Input tensor

    Returns:
        Result tensor
    """
    x = input._mlx_array
    mlx_result = x - mx.tanh(x)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def threshold(input: Tensor, threshold: float, value: float, inplace: bool = False) -> Tensor:
    """
    Apply Threshold activation.

    threshold(x) = x if x > threshold else value

    Args:
        input: Input tensor
        threshold: Threshold value
        value: Value to replace with when x <= threshold
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    x = input._mlx_array
    mlx_result = mx.where(x > threshold, x, value)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def threshold_(input: Tensor, threshold: float, value: float) -> Tensor:
    """In-place threshold (returns new tensor in MLX)."""
    # Inline implementation to avoid parameter name shadowing
    x = input._mlx_array
    mlx_result = mx.where(x > threshold, x, value)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def glu(input: Tensor, dim: int = -1) -> Tensor:
    """
    Apply Gated Linear Unit activation.

    glu(x) = a * sigmoid(b) where x is split into a and b along dim

    Args:
        input: Input tensor (must have even size along dim)
        dim: Dimension to split along (default: -1)

    Returns:
        Result tensor
    """
    x = input._mlx_array
    # Split into two halves along dim
    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(f"Dimension {dim} size must be even, got {size}")

    half_size = size // 2
    # Use slicing to split
    if dim == -1 or dim == len(x.shape) - 1:
        a = x[..., :half_size]
        b = x[..., half_size:]
    else:
        # General case using split
        slices_a = [slice(None)] * len(x.shape)
        slices_b = [slice(None)] * len(x.shape)
        slices_a[dim] = slice(0, half_size)
        slices_b[dim] = slice(half_size, None)
        a = x[tuple(slices_a)]
        b = x[tuple(slices_b)]

    mlx_result = a * nn.sigmoid(b)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def logsigmoid(input: Tensor) -> Tensor:
    """
    Apply Log Sigmoid activation.

    logsigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x)))
                  = -softplus(-x) = -log(1 + exp(-x))

    More numerically stable implementation.

    Args:
        input: Input tensor

    Returns:
        Result tensor
    """
    x = input._mlx_array
    # Numerically stable: -softplus(-x)
    # softplus(x) = log(1 + exp(x))
    # For large negative x: logsigmoid(x) ≈ x
    # For large positive x: logsigmoid(x) ≈ 0
    mlx_result = -nn.softplus(-x)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def prelu(input: Tensor, weight: Tensor) -> Tensor:
    """
    Apply Parametric ReLU activation.

    prelu(x) = max(0, x) + weight * min(0, x)

    Args:
        input: Input tensor
        weight: Learnable weight tensor (can be scalar or per-channel)

    Returns:
        Result tensor
    """
    x = input._mlx_array
    w = weight._mlx_array

    # Broadcast weight if needed
    if w.ndim == 1 and x.ndim > 1:
        # Reshape weight for broadcasting (typically for channel dimension)
        shape = [1] * x.ndim
        shape[1] = -1  # Channel dimension
        w = mx.reshape(w, shape)

    mlx_result = mx.maximum(x, 0) + w * mx.minimum(x, 0)
    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and (input.requires_grad or weight.requires_grad):
        result.requires_grad = True

    return result


def softmin(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype=None) -> Tensor:
    """
    Apply Softmin activation.

    softmin(x) = softmax(-x)

    Args:
        input: Input tensor
        dim: Dimension along which to apply softmin (default: last dimension)
        _stacklevel: Internal parameter for warning stack level
        dtype: Optional dtype to cast input to before computation

    Returns:
        Result tensor
    """
    if dim is None:
        dim = -1

    # Negate input and apply softmax
    neg_input = Tensor._from_mlx_array(-input._mlx_array)
    neg_input.requires_grad = input.requires_grad

    return softmax(neg_input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)


def rrelu(input: Tensor, lower: float = 1.0/8, upper: float = 1.0/3,
          training: bool = False, inplace: bool = False) -> Tensor:
    """
    Apply Randomized Leaky ReLU activation.

    During training: rrelu(x) = x if x >= 0 else x * a (a sampled from U(lower, upper))
    During eval: rrelu(x) = x if x >= 0 else x * (lower + upper) / 2

    Args:
        input: Input tensor
        lower: Lower bound of uniform distribution (default: 1/8)
        upper: Upper bound of uniform distribution (default: 1/3)
        training: If True, sample random slopes; else use mean
        inplace: If True, modify in-place (ignored in MLX)

    Returns:
        Result tensor
    """
    x = input._mlx_array

    if training:
        # Sample random slopes for each element
        slopes = mx.random.uniform(low=lower, high=upper, shape=x.shape)
        mlx_result = mx.where(x >= 0, x, x * slopes)
    else:
        # Use average slope during inference
        mean_slope = (lower + upper) / 2
        mlx_result = mx.where(x >= 0, x, x * mean_slope)

    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def rrelu_(input: Tensor, lower: float = 1.0/8, upper: float = 1.0/3, training: bool = False) -> Tensor:
    """In-place rrelu (returns new tensor in MLX)."""
    return rrelu(input, lower, upper, training, inplace=True)


def celu_(input: Tensor, alpha: float = 1.0) -> Tensor:
    """In-place celu (returns new tensor in MLX)."""
    return celu(input, alpha, inplace=True)


def selu_(input: Tensor) -> Tensor:
    """In-place selu (returns new tensor in MLX)."""
    return selu(input, inplace=True)


# Aliases
swish = silu  # Swish is another name for SiLU
