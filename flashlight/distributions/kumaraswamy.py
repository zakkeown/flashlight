"""Kumaraswamy Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..ops.special import beta
from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class Kumaraswamy(Distribution):
    """Kumaraswamy distribution."""

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Union[Tensor, float],
        concentration0: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.concentration1 = (
            concentration1._mlx_array
            if isinstance(concentration1, Tensor)
            else mx.array(concentration1)
        )
        self.concentration0 = (
            concentration0._mlx_array
            if isinstance(concentration0, Tensor)
            else mx.array(concentration0)
        )
        batch_shape = mx.broadcast_shapes(self.concentration1.shape, self.concentration0.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        # E[X] = b * Beta(1 + 1/a, b)
        a, b = self.concentration1, self.concentration0
        mean_val = b * beta(1 + 1 / a, b)
        return Tensor(mean_val)

    @property
    def mode(self) -> Tensor:
        a, b = self.concentration1, self.concentration0
        return Tensor(
            mx.where(
                (a >= 1) & (b >= 1) & ((a > 1) | (b > 1)),
                mx.power((a - 1) / (a * b - 1), 1 / a),
                mx.array(float("nan")),
            )
        )

    @property
    def variance(self) -> Tensor:
        # Var[X] = b * Beta(1 + 2/a, b) - (b * Beta(1 + 1/a, b))^2
        a, b = self.concentration1, self.concentration0
        moment2 = b * beta(1 + 2 / a, b)
        mean_val = b * beta(1 + 1 / a, b)
        return Tensor(moment2 - mean_val**2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Inverse CDF method: x = (1 - (1-u)^(1/b))^(1/a)
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape=shape)
        return Tensor(
            mx.power(1 - mx.power(1 - u, 1 / self.concentration0), 1 / self.concentration1)
        )

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else mx.array(value)
        log_prob = (
            mx.log(self.concentration1)
            + mx.log(self.concentration0)
            + (self.concentration1 - 1) * mx.log(data)
            + (self.concentration0 - 1) * mx.log(1 - mx.power(data, self.concentration1))
        )
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        from ..ops.special import digamma

        a, b = self.concentration1, self.concentration0
        # Entropy = (1 - 1/b) + (1 - 1/a) * H_b + log(a*b)
        # where H_b = digamma(b+1) + euler_gamma
        euler_gamma = 0.5772156649015329
        H_b = digamma(b + 1) + euler_gamma
        return Tensor((1 - 1 / b) + (1 - 1 / a) * H_b - mx.log(a * b))


__all__ = ["Kumaraswamy"]
