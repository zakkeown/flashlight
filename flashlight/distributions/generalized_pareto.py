"""Generalized Pareto Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class GeneralizedPareto(Distribution):
    """Generalized Pareto distribution."""

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'concentration': constraints.real}
    has_rsample = True

    def __init__(
        self,
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        concentration: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else mx.array(scale)
        self.concentration = concentration._mlx_array if isinstance(concentration, Tensor) else mx.array(concentration)
        batch_shape = mx.broadcast_shapes(self.loc.shape, mx.broadcast_shapes(self.scale.shape, self.concentration.shape))
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints.greater_than_eq(self.loc)

    @property
    def mean(self) -> Tensor:
        xi = self.concentration
        return Tensor(mx.where(xi < 1, self.loc + self.scale / (1 - xi), mx.array(float('inf'))))

    @property
    def variance(self) -> Tensor:
        xi = self.concentration
        return Tensor(mx.where(xi < 0.5, self.scale ** 2 / ((1 - xi) ** 2 * (1 - 2 * xi)), mx.array(float('inf'))))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        xi = self.concentration
        # GPD quantile function
        return Tensor(self.loc + self.scale / xi * (mx.power(1 - u, -xi) - 1))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        xi = self.concentration
        z = (data - self.loc) / self.scale
        log_prob = -mx.log(self.scale) - (1 + 1/xi) * mx.log(1 + xi * z)
        return Tensor(log_prob)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        xi = self.concentration
        z = (data - self.loc) / self.scale
        return Tensor(1 - mx.power(1 + xi * z, -1/xi))


__all__ = ['GeneralizedPareto']
