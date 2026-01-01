"""
Shape Manipulation Layers

Implements shape manipulation layers for neural networks.
"""

import mlx.core as mx
from ..module import Module
from ...tensor import Tensor
from typing import Union, Tuple


class Flatten(Module):
    """
    Flattens a contiguous range of dimensions.

    Args:
        start_dim: First dimension to flatten (default: 1)
        end_dim: Last dimension to flatten (default: -1)

    Shape:
        - Input: [*, start_dim, ..., end_dim, *]
        - Output: [*, flattened_dims, *]

    Example:
        >>> flatten = nn.Flatten()
        >>> x = flashlight.randn(4, 3, 32, 32)
        >>> output = flatten(x)  # [4, 3072]
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """Flatten the tensor."""
        x = input._mlx_array
        ndim = x.ndim

        # Handle negative indices
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim

        if start < 0 or start >= ndim:
            raise IndexError(f"start_dim {self.start_dim} out of range for tensor with {ndim} dimensions")
        if end < 0 or end >= ndim:
            raise IndexError(f"end_dim {self.end_dim} out of range for tensor with {ndim} dimensions")
        if start > end:
            raise ValueError(f"start_dim ({start}) cannot be greater than end_dim ({end})")

        # Build new shape
        shape = x.shape
        new_shape = list(shape[:start])
        flattened_size = 1
        for i in range(start, end + 1):
            flattened_size *= shape[i]
        new_shape.append(flattened_size)
        new_shape.extend(shape[end + 1:])

        result = mx.reshape(x, new_shape)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'


class Unflatten(Module):
    """
    Unflattens a dimension of a tensor.

    Args:
        dim: Dimension to unflatten
        unflattened_size: New shape for the unflattened dimension

    Shape:
        - Input: [*, S, *] where S is the size at dim
        - Output: [*, unflattened_size[0], ..., unflattened_size[n], *]

    Example:
        >>> unflatten = nn.Unflatten(1, (3, 32, 32))
        >>> x = flashlight.randn(4, 3072)
        >>> output = unflatten(x)  # [4, 3, 32, 32]
    """

    def __init__(self, dim: int, unflattened_size: Tuple[int, ...]):
        super().__init__()
        self.dim = dim
        self.unflattened_size = tuple(unflattened_size)

    def forward(self, input: Tensor) -> Tensor:
        """Unflatten the tensor."""
        x = input._mlx_array
        ndim = x.ndim
        shape = x.shape

        # Handle negative index
        dim = self.dim if self.dim >= 0 else ndim + self.dim

        if dim < 0 or dim >= ndim:
            raise IndexError(f"dim {self.dim} out of range for tensor with {ndim} dimensions")

        # Verify the unflattened size is compatible
        expected_size = 1
        for s in self.unflattened_size:
            expected_size *= s
        if shape[dim] != expected_size:
            raise ValueError(
                f"Cannot unflatten dimension {dim} with size {shape[dim]} "
                f"into shape {self.unflattened_size} (product = {expected_size})"
            )

        # Build new shape
        new_shape = list(shape[:dim]) + list(self.unflattened_size) + list(shape[dim + 1:])

        result = mx.reshape(x, new_shape)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, unflattened_size={self.unflattened_size}'


class Identity(Module):
    """
    A placeholder identity operator.

    A module that passes input through unchanged. Useful as a placeholder
    when a module is expected but no operation is needed.

    Args:
        *args: Any arguments (ignored)
        **kwargs: Any keyword arguments (ignored)

    Shape:
        - Input: Any
        - Output: Same as input

    Example:
        >>> identity = nn.Identity()
        >>> x = flashlight.randn(4, 64)
        >>> output = identity(x)  # Same as x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Return input unchanged."""
        return input

    def extra_repr(self) -> str:
        return ''


__all__ = ['Flatten', 'Unflatten', 'Identity']
