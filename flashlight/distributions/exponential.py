"""Exponential Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .exp_family import ExponentialFamily


class Exponential(ExponentialFamily):
    """Exponential distribution with rate parameter."""

    arg_constraints = {"rate": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, rate: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.rate = rate._mlx_array if isinstance(rate, Tensor) else mx.array(rate)
        super().__init__(self.rate.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(1 / self.rate)

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.zeros_like(self.rate))

    @property
    def variance(self) -> Tensor:
        return Tensor(1 / self.rate**2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        return Tensor(-mx.log(mx.random.uniform(shape)) / self.rate)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(mx.log(self.rate) - self.rate * data)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(1 - mx.exp(-self.rate * data))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(-mx.log(1 - data) / self.rate)

    def entropy(self) -> Tensor:
        return Tensor(1 - mx.log(self.rate))

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(-self.rate),)

    def _log_normalizer(self, x) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(-mx.log(-x_data))


__all__ = ["Exponential"]
