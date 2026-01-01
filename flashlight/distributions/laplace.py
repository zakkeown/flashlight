"""Laplace Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Laplace(Distribution):
    """Laplace distribution."""

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
        return Tensor(self.loc)

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        return Tensor(2 * self.scale ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape) - 0.5
        return Tensor(self.loc - self.scale * mx.sign(u) * mx.log1p(-2 * mx.abs(u)))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(-mx.log(2 * self.scale) - mx.abs(data - self.loc) / self.scale)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(0.5 - 0.5 * mx.sign(data - self.loc) * mx.expm1(-mx.abs(data - self.loc) / self.scale))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(self.loc - self.scale * mx.sign(data - 0.5) * mx.log1p(-2 * mx.abs(data - 0.5)))

    def entropy(self) -> Tensor:
        return Tensor(1 + mx.log(2 * self.scale))


__all__ = ['Laplace']
