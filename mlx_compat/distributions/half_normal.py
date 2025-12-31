"""Half-Normal Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class HalfNormal(Distribution):
    """Half-Normal distribution (absolute value of Normal)."""

    arg_constraints = {'scale': constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, scale: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.scale = scale._data if isinstance(scale, Tensor) else mx.array(scale)
        super().__init__(self.scale.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.scale * math.sqrt(2 / math.pi))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.zeros_like(self.scale))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.scale ** 2 * (1 - 2 / math.pi))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        return Tensor(mx.abs(self.scale * mx.random.normal(shape)))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        log_prob = 0.5 * math.log(2 / math.pi) - mx.log(self.scale) - (data ** 2) / (2 * self.scale ** 2)
        return Tensor(log_prob)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(mx.erf(data / (self.scale * math.sqrt(2))))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(self.scale * mx.erfinv(data) * math.sqrt(2))

    def entropy(self) -> Tensor:
        return Tensor(0.5 * math.log(math.pi * math.e / 2) + mx.log(self.scale))


__all__ = ['HalfNormal']
