"""
Linear Layers

Implements fully-connected (linear) layers.
"""

import math
from typing import Optional, Any

from ...tensor import Tensor
from ..module import Module
from ..parameter import Parameter
from ... import ops


class Linear(Module):
    """
    Applies a linear transformation to the incoming data: y = xW^T + b

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias (default: True)

    Shape:
        - Input: (*, in_features) where * means any number of dimensions
        - Output: (*, out_features)

    Attributes:
        weight: Learnable weights of shape (out_features, in_features)
        bias: Learnable bias of shape (out_features) if bias=True

    Example:
        >>> m = nn.Linear(20, 30)
        >>> input = mlx_compat.randn(128, 20)
        >>> output = m(input)
        >>> output.shape
        (128, 30)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        """
        Initialize Linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            device: Device to place the layer on (ignored, MLX uses unified memory)
            dtype: Data type for the layer parameters (ignored, uses default)
        """
        # device and dtype are accepted for PyTorch compatibility but ignored
        # MLX uses unified memory and default dtype
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameter
        # Shape is (out_features, in_features) to match PyTorch
        from ... import randn
        self.weight = Parameter(randn(out_features, in_features))

        # Initialize bias if requested
        if bias:
            self.bias = Parameter(randn(out_features))
        else:
            # Register None to be explicit
            self.bias = None

        # Initialize parameters using Kaiming uniform
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize parameters using Kaiming uniform initialization.

        This follows PyTorch's default initialization for Linear layers.
        """
        from ... import empty
        import mlx.core as mx

        # Kaiming uniform initialization
        # bound = sqrt(1 / in_features)
        bound = math.sqrt(1.0 / self.in_features)

        # Initialize weight
        # Create uniform random values in [-bound, bound]
        weight_data = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(self.out_features, self.in_features)
        )
        self.weight.data = Tensor._from_mlx_array(weight_data)

        # Initialize bias if it exists
        if self.bias is not None:
            bias_data = mx.random.uniform(
                low=-bound,
                high=bound,
                shape=(self.out_features,)
            )
            self.bias.data = Tensor._from_mlx_array(bias_data)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of linear layer.

        Args:
            input: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Compute xW^T
        # input: (batch, in_features)
        # weight: (out_features, in_features)
        # We need: input @ weight.T = (batch, in_features) @ (in_features, out_features)
        from ... import transpose
        weight_t = transpose(self.weight, 0, 1)
        output = ops.matmul(input, weight_t)

        # Add bias if it exists
        if self.bias is not None:
            output = ops.add(output, self.bias)

        return output

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


__all__ = ['Linear']
