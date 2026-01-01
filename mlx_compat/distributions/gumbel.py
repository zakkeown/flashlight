"""Gumbel Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Gumbel(Distribution):
    """Gumbel distribution."""

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
        euler_gamma = 0.5772156649015329
        return Tensor(self.loc + self.scale * euler_gamma)

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        return Tensor((math.pi ** 2 / 6) * self.scale ** 2)

    @property
    def stddev(self) -> Tensor:
        return Tensor((math.pi / math.sqrt(6)) * self.scale)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(self.loc - self.scale * mx.log(-mx.log(u + 1e-10) + 1e-10))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        z = (data - self.loc) / self.scale
        return Tensor(-z - mx.exp(-z) - mx.log(self.scale))

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(mx.exp(-mx.exp(-(data - self.loc) / self.scale)))

    def entropy(self) -> Tensor:
        euler_gamma = 0.5772156649015329
        return Tensor(mx.log(self.scale) + 1 + euler_gamma)


__all__ = ['Gumbel']
