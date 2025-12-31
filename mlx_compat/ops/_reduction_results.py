"""
Lazy result classes for min/max reductions.

These classes defer computation of indices until they're actually accessed,
avoiding the overhead of computing argmax/argmin when only values are needed.
"""

import mlx.core as mx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor import Tensor


class MaxResult:
    """
    Lazy result for max() with axis - computes indices only when accessed.

    This matches PyTorch's torch.return_types.max behavior, providing both
    .values and .indices attributes, but only computing indices on demand.

    Usage:
        >>> result = max(tensor, dim=1)
        >>> values = result.values  # Computed immediately
        >>> indices = result.indices  # Computed only when accessed
        >>> # Or unpack like a tuple:
        >>> values, indices = max(tensor, dim=1)
    """

    def __init__(self, input_array, dim: int, keepdim: bool):
        self._input = input_array
        self._dim = dim
        self._keepdim = keepdim
        self._values = None
        self._indices = None

    @property
    def values(self) -> 'Tensor':
        if self._values is None:
            from ..tensor import Tensor
            self._values = Tensor._from_mlx_array(
                mx.max(self._input, axis=self._dim, keepdims=self._keepdim)
            )
        return self._values

    @property
    def indices(self) -> 'Tensor':
        if self._indices is None:
            from ..tensor import Tensor
            self._indices = Tensor._from_mlx_array(
                mx.argmax(self._input, axis=self._dim, keepdims=self._keepdim)
            )
        return self._indices

    def __iter__(self):
        """Allow tuple unpacking: values, indices = max(x, dim=1)"""
        return iter((self.values, self.indices))

    def __getitem__(self, idx: int):
        """Allow indexing: result[0] for values, result[1] for indices"""
        return (self.values, self.indices)[idx]

    def __repr__(self):
        return f"MaxResult(values={self.values}, indices={self.indices})"


class MinResult:
    """
    Lazy result for min() with axis - computes indices only when accessed.

    This matches PyTorch's torch.return_types.min behavior, providing both
    .values and .indices attributes, but only computing indices on demand.

    Usage:
        >>> result = min(tensor, dim=1)
        >>> values = result.values  # Computed immediately
        >>> indices = result.indices  # Computed only when accessed
        >>> # Or unpack like a tuple:
        >>> values, indices = min(tensor, dim=1)
    """

    def __init__(self, input_array, dim: int, keepdim: bool):
        self._input = input_array
        self._dim = dim
        self._keepdim = keepdim
        self._values = None
        self._indices = None

    @property
    def values(self) -> 'Tensor':
        if self._values is None:
            from ..tensor import Tensor
            self._values = Tensor._from_mlx_array(
                mx.min(self._input, axis=self._dim, keepdims=self._keepdim)
            )
        return self._values

    @property
    def indices(self) -> 'Tensor':
        if self._indices is None:
            from ..tensor import Tensor
            self._indices = Tensor._from_mlx_array(
                mx.argmin(self._input, axis=self._dim, keepdims=self._keepdim)
            )
        return self._indices

    def __iter__(self):
        """Allow tuple unpacking: values, indices = min(x, dim=1)"""
        return iter((self.values, self.indices))

    def __getitem__(self, idx: int):
        """Allow indexing: result[0] for values, result[1] for indices"""
        return (self.values, self.indices)[idx]

    def __repr__(self):
        return f"MinResult(values={self.values}, indices={self.indices})"


__all__ = ['MaxResult', 'MinResult']
