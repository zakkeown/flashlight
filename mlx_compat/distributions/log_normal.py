"""Log-Normal Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class LogNormal(Distribution):
    """Log-Normal distribution."""

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc: Union[Tensor, float], scale: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.loc = loc._data if isinstance(loc, Tensor) else mx.array(loc)
        self.scale = scale._data if isinstance(scale, Tensor) else mx.array(scale)
        batch_shape = mx.broadcast_shapes(self.loc.shape, self.scale.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(mx.exp(self.loc + self.scale ** 2 / 2))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.exp(self.loc - self.scale ** 2))

    @property
    def variance(self) -> Tensor:
        return Tensor((mx.exp(self.scale ** 2) - 1) * mx.exp(2 * self.loc + self.scale ** 2))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        return Tensor(mx.exp(self.loc + self.scale * mx.random.normal(shape)))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        var = self.scale ** 2
        log_value = mx.log(data)
        return Tensor(-((log_value - self.loc) ** 2) / (2 * var) - mx.log(self.scale * data) - 0.5 * math.log(2 * math.pi))

    def cdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(0.5 * (1 + mx.erf((mx.log(data) - self.loc) / (self.scale * math.sqrt(2)))))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(mx.exp(self.loc + self.scale * mx.erfinv(2 * data - 1) * math.sqrt(2)))

    def entropy(self) -> Tensor:
        return Tensor(0.5 + 0.5 * math.log(2 * math.pi) + mx.log(self.scale) + self.loc)


__all__ = ['LogNormal']
