"""Cauchy Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Cauchy(Distribution):
    """Cauchy distribution."""

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc: Union[Tensor, float], scale: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else mx.array(scale)
        batch_shape = mx.broadcast_shapes(self.loc.shape, self.scale.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        # Use mx.full instead of mx.full_like (MLX doesn't have full_like)
        return Tensor(mx.full(self.loc.shape, float('nan'), dtype=self.loc.dtype))

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        # Use mx.full instead of mx.full_like (MLX doesn't have full_like)
        return Tensor(mx.full(self.loc.shape, float('inf'), dtype=self.loc.dtype))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(self.loc + self.scale * mx.tan(math.pi * (u - 0.5)))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(-mx.log(math.pi * self.scale * (1 + ((data - self.loc) / self.scale) ** 2)))

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(mx.arctan((data - self.loc) / self.scale) / math.pi + 0.5)

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(self.loc + self.scale * mx.tan(math.pi * (data - 0.5)))

    def entropy(self) -> Tensor:
        return Tensor(mx.log(4 * math.pi * self.scale))


__all__ = ['Cauchy']
