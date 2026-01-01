"""Uniform Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class Uniform(Distribution):
    """
    Uniform distribution over [low, high).

    Args:
        low: Lower bound
        high: Upper bound
        validate_args: Whether to validate arguments
    """

    arg_constraints = {"low": constraints.dependent, "high": constraints.dependent}
    has_rsample = True

    def __init__(
        self,
        low: Union[Tensor, float],
        high: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.low = low._mlx_array if isinstance(low, Tensor) else mx.array(low)
        self.high = high._mlx_array if isinstance(high, Tensor) else mx.array(high)

        batch_shape = mx.broadcast_shapes(self.low.shape, self.high.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints.interval(self.low, self.high)

    @property
    def mean(self) -> Tensor:
        return Tensor((self.low + self.high) / 2)

    @property
    def mode(self) -> Tensor:
        # Uniform has no unique mode
        return Tensor((self.low + self.high) / 2)

    @property
    def variance(self) -> Tensor:
        return Tensor((self.high - self.low) ** 2 / 12)

    @property
    def stddev(self) -> Tensor:
        return Tensor((self.high - self.low) / mx.sqrt(mx.array(12.0)))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        return Tensor(mx.random.uniform(self.low, self.high, shape))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        lb = data >= self.low
        ub = data < self.high
        log_prob = mx.where(lb & ub, -mx.log(self.high - self.low), mx.array(float("-inf")))
        return Tensor(log_prob)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        result = (data - self.low) / (self.high - self.low)
        result = mx.clip(result, 0, 1)
        return Tensor(result)

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(self.low + data * (self.high - self.low))

    def entropy(self) -> Tensor:
        return Tensor(mx.log(self.high - self.low))


__all__ = ["Uniform"]
