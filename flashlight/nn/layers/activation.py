"""
Activation Layers

Implements PyTorch-compatible activation function layers:
- ReLU, LeakyReLU, ELU
- Sigmoid, Tanh
- GELU, SiLU (Swish)
- Softmax, LogSoftmax
"""

from typing import Optional

from ... import ops
from ...tensor import Tensor
from ..module import Module


class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise.

    ReLU(x) = max(0, x)

    Args:
        inplace: Unused (for PyTorch compatibility). MLX arrays are immutable.

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.ReLU()
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        result = ops.relu(input)
        if self.inplace:
            # Simulate inplace by copying result back to input
            input._mlx_array = result._mlx_array
            return input
        return result

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}" if self.inplace else ""


class LeakyReLU(Module):
    """
    Applies the leaky rectified linear unit function element-wise.

    LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)

    Args:
        negative_slope: Controls the angle of the negative slope (default: 0.01)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.LeakyReLU(0.1)
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        result = ops.leaky_relu(input, negative_slope=self.negative_slope)
        if self.inplace:
            # Simulate inplace by copying result back to input
            input._mlx_array = result._mlx_array
            return input
        return result

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class ELU(Module):
    """
    Applies the exponential linear unit function element-wise.

    ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))

    Args:
        alpha: The alpha value for the ELU formulation (default: 1.0)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.ELU()
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return ops.elu(input, alpha=self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class Sigmoid(Module):
    """
    Applies the sigmoid function element-wise.

    Sigmoid(x) = 1 / (1 + exp(-x))

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.Sigmoid()
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return ops.sigmoid(input)


class Tanh(Module):
    """
    Applies the hyperbolic tangent function element-wise.

    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.Tanh()
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return ops.tanh(input)


class GELU(Module):
    """
    Applies the Gaussian Error Linear Unit function.

    GELU(x) = x * Φ(x) where Φ(x) is the Gaussian CDF

    Args:
        approximate: If 'tanh', use tanh approximation (default: 'none')

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.GELU()
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return ops.gelu(input)

    def extra_repr(self) -> str:
        return f"approximate='{self.approximate}'"


class SiLU(Module):
    """
    Applies the Sigmoid Linear Unit (SiLU) function, also known as Swish.

    SiLU(x) = x * sigmoid(x)

    Args:
        inplace: Not used (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.SiLU()
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return ops.silu(input)


class Softmax(Module):
    """
    Applies the Softmax function to an n-dimensional input tensor.

    Softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    Args:
        dim: Dimension along which Softmax will be computed (default: -1)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.Softmax(dim=1)
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim if dim is not None else -1

    def forward(self, input: Tensor) -> Tensor:
        return ops.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """
    Applies the LogSoftmax function to an n-dimensional input tensor.

    LogSoftmax(x_i) = log(exp(x_i) / sum_j exp(x_j))

    Args:
        dim: Dimension along which LogSoftmax will be computed (default: None)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input

    Example:
        >>> m = nn.LogSoftmax(dim=1)
        >>> x = flashlight.randn(2, 3)
        >>> output = m(x)
    """

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim if dim is not None else -1

    def forward(self, input: Tensor) -> Tensor:
        return ops.log_softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class ReLU6(Module):
    """
    Applies ReLU6 function element-wise.

    ReLU6(x) = min(max(0, x), 6)

    Args:
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.minimum(mx.maximum(x, 0), 6)
        return Tensor._from_mlx_array(result)


class SELU(Module):
    """
    Applies the Scaled Exponential Linear Unit function element-wise.

    SELU(x) = scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
    with alpha = 1.6732632423543772848170429916717
    and scale = 1.0507009873554804934193349852946

    Args:
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        x = input._mlx_array
        result = scale * mx.where(x > 0, x, alpha * (mx.exp(x) - 1))
        return Tensor._from_mlx_array(result)


class CELU(Module):
    """
    Applies the Continuously Differentiable ELU function element-wise.

    CELU(x) = max(0,x) + min(0, alpha * (exp(x/alpha) - 1))

    Args:
        alpha: The alpha value for CELU (default: 1.0)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.maximum(x, 0) + mx.minimum(0, self.alpha * (mx.exp(x / self.alpha) - 1))
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class Hardtanh(Module):
    """
    Applies the HardTanh function element-wise.

    Hardtanh(x) = max(min_val, min(max_val, x))

    Args:
        min_val: Minimum value (default: -1)
        max_val: Maximum value (default: 1)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: float = None,
        max_value: float = None,
    ):
        super().__init__()
        # min_value and max_value are deprecated aliases for min_val and max_val
        if min_value is not None:
            import warnings

            warnings.warn("min_value is deprecated, use min_val instead", DeprecationWarning)
            min_val = min_value
        if max_value is not None:
            import warnings

            warnings.warn("max_value is deprecated, use max_val instead", DeprecationWarning)
            max_val = max_value
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.clip(x, self.min_val, self.max_val)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"


class Hardsigmoid(Module):
    """
    Applies the Hardsigmoid function element-wise.

    Hardsigmoid(x) = max(0, min(1, (x + 3) / 6))

    Args:
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.clip((x + 3) / 6, 0, 1)
        return Tensor._from_mlx_array(result)


class Hardswish(Module):
    """
    Applies the Hardswish function element-wise.

    Hardswish(x) = x * Hardsigmoid(x)

    Args:
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = x * mx.clip((x + 3) / 6, 0, 1)
        return Tensor._from_mlx_array(result)


class Hardshrink(Module):
    """
    Applies the Hardshrink function element-wise.

    Hardshrink(x) = x if |x| > lambda else 0

    Args:
        lambd: The lambda value (default: 0.5)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.where(mx.abs(x) > self.lambd, x, 0)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class Softplus(Module):
    """
    Applies the Softplus function element-wise.

    Softplus(x) = (1/beta) * log(1 + exp(beta * x))

    Args:
        beta: The beta value (default: 1)
        threshold: Values above this revert to linear function (default: 20)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        beta_x = self.beta * x
        # Use linear approximation for large values to avoid overflow
        result = mx.where(beta_x > self.threshold, x, mx.log1p(mx.exp(beta_x)) / self.beta)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"


class Softshrink(Module):
    """
    Applies the Softshrink function element-wise.

    Softshrink(x) = x - lambda if x > lambda
                    x + lambda if x < -lambda
                    0 otherwise

    Args:
        lambd: The lambda value (default: 0.5)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.where(
            x > self.lambd, x - self.lambd, mx.where(x < -self.lambd, x + self.lambd, 0)
        )
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class Softsign(Module):
    """
    Applies the Softsign function element-wise.

    Softsign(x) = x / (1 + |x|)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = x / (1 + mx.abs(x))
        return Tensor._from_mlx_array(result)


class Tanhshrink(Module):
    """
    Applies the Tanhshrink function element-wise.

    Tanhshrink(x) = x - tanh(x)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = x - mx.tanh(x)
        return Tensor._from_mlx_array(result)


class LogSigmoid(Module):
    """
    Applies the LogSigmoid function element-wise.

    LogSigmoid(x) = log(1 / (1 + exp(-x)))

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        # Numerically stable: -softplus(-x)
        result = -mx.log1p(mx.exp(-x))
        return Tensor._from_mlx_array(result)


class Softmin(Module):
    """
    Applies Softmin to an n-dimensional input tensor.

    Softmin(x_i) = exp(-x_i) / sum_j exp(-x_j)

    Args:
        dim: Dimension along which Softmin will be computed (default: None)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim if dim is not None else -1

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.softmax(-x, axis=self.dim)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Mish(Module):
    """
    Applies the Mish function element-wise.

    Mish(x) = x * tanh(softplus(x))

    Args:
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = x * mx.tanh(mx.log1p(mx.exp(x)))
        return Tensor._from_mlx_array(result)


class GLU(Module):
    """
    Applies the Gated Linear Unit function.

    GLU(a, b) = a ⊗ σ(b)

    where input is split in half along dim to form a and b.

    Args:
        dim: Dimension on which to split the input (default: -1)

    Shape:
        - Input: (*, N, *) where N is even at dimension dim
        - Output: (*, N/2, *)
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        a, b = mx.split(x, 2, axis=self.dim)
        result = a * mx.sigmoid(b)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class PReLU(Module):
    """
    Applies PReLU (Parametric ReLU) function element-wise.

    PReLU(x) = max(0, x) + a * min(0, x)

    where a is a learnable parameter.

    Args:
        num_parameters: Number of parameters. 1 for shared, or number of channels.
        init: Initial value of a (default: 0.25)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None):
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)
        self.num_parameters = num_parameters
        import mlx.core as mx

        from ..parameter import Parameter

        self.weight = Parameter(Tensor._from_mlx_array(mx.full((num_parameters,), init)))

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        w = self.weight._mlx_array
        # Broadcast weight if needed
        if self.num_parameters == 1:
            result = mx.maximum(x, 0) + w * mx.minimum(x, 0)
        else:
            # Reshape weight for broadcasting with channel dimension
            shape = [1] * x.ndim
            shape[1] = self.num_parameters
            w = w.reshape(shape)
            result = mx.maximum(x, 0) + w * mx.minimum(x, 0)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"


class Threshold(Module):
    """
    Thresholds each element of the input tensor.

    Threshold(x) = x if x > threshold else value

    Args:
        threshold: The threshold value
        value: The value to replace with
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, threshold: float, value: float, inplace: bool = False):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        result = mx.where(x > self.threshold, x, self.value)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, value={self.value}"


class RReLU(Module):
    """
    Applies Randomized Leaky ReLU.

    During training, the negative slope is sampled uniformly from [lower, upper].
    During evaluation, uses the midpoint (lower + upper) / 2.

    Args:
        lower: Lower bound of random slope (default: 1/8)
        upper: Upper bound of random slope (default: 1/3)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*, ) any shape
        - Output: (*, ) same shape as input
    """

    def __init__(self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        if self.training:
            slope = mx.random.uniform(low=self.lower, high=self.upper, shape=x.shape)
        else:
            slope = (self.lower + self.upper) / 2
        result = mx.where(x >= 0, x, x * slope)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"lower={self.lower}, upper={self.upper}"


class Softmax2d(Module):
    """
    Applies Softmax over features to each spatial location.

    When given an image of shape (N, C, H, W), applies softmax over C at each (H, W) location.

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) same shape as input
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        # Apply softmax over channel dimension (dim=1)
        result = mx.softmax(input._mlx_array, axis=1)
        return Tensor._from_mlx_array(result)


class ChannelShuffle(Module):
    """
    Divides and rearranges channels in a tensor.

    Useful for ShuffleNet architectures.

    Args:
        groups: Number of groups to divide channels into

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) same shape as input
    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        N, C, H, W = x.shape
        g = self.groups
        # Reshape to (N, g, C//g, H, W)
        x = mx.reshape(x, (N, g, C // g, H, W))
        # Transpose to (N, C//g, g, H, W)
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        # Flatten back to (N, C, H, W)
        x = mx.reshape(x, (N, C, H, W))
        return Tensor._from_mlx_array(x)

    def extra_repr(self) -> str:
        return f"groups={self.groups}"


class Bilinear(Module):
    """
    Applies bilinear transformation: y = x1^T A x2 + b.

    Args:
        in1_features: Size of each first input sample
        in2_features: Size of each second input sample
        out_features: Size of each output sample
        bias: If False, no bias is added

    Shape:
        - Input1: (*, in1_features)
        - Input2: (*, in2_features)
        - Output: (*, out_features)
    """

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        import mlx.core as mx

        from ..parameter import Parameter

        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        # Weight shape: (out_features, in1_features, in2_features)
        k = 1.0 / in1_features
        weight = mx.random.uniform(
            low=-(k**0.5), high=k**0.5, shape=(out_features, in1_features, in2_features)
        )
        self.weight = Parameter(Tensor._from_mlx_array(weight))

        if bias:
            bias_init = mx.random.uniform(low=-(k**0.5), high=k**0.5, shape=(out_features,))
            self.bias = Parameter(Tensor._from_mlx_array(bias_init))
        else:
            self.bias = None

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        from ..functional import bilinear

        return bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in1_features={self.in1_features}, in2_features={self.in2_features}, out_features={self.out_features}, bias={self.bias is not None}"


__all__ = [
    "ReLU",
    "LeakyReLU",
    "ELU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "SiLU",
    "Softmax",
    "LogSoftmax",
    "ReLU6",
    "SELU",
    "CELU",
    "Hardtanh",
    "Hardsigmoid",
    "Hardswish",
    "Hardshrink",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanhshrink",
    "LogSigmoid",
    "Softmin",
    "Mish",
    "GLU",
    "PReLU",
    "Threshold",
    "RReLU",
    "Softmax2d",
    "ChannelShuffle",
    "Bilinear",
]
