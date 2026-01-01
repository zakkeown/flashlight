"""Pareto Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class Pareto(Distribution):
    """Pareto distribution (Type I)."""

    arg_constraints = {"scale": constraints.positive, "alpha": constraints.positive}
    has_rsample = True

    def __init__(
        self,
        scale: Union[Tensor, float],
        alpha: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else mx.array(scale)
        self.alpha = alpha._mlx_array if isinstance(alpha, Tensor) else mx.array(alpha)
        batch_shape = mx.broadcast_shapes(self.scale.shape, self.alpha.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints.greater_than_eq(self.scale)

    @property
    def mean(self) -> Tensor:
        return Tensor(
            mx.where(
                self.alpha > 1, self.alpha * self.scale / (self.alpha - 1), mx.array(float("inf"))
            )
        )

    @property
    def mode(self) -> Tensor:
        return Tensor(self.scale)

    @property
    def variance(self) -> Tensor:
        return Tensor(
            mx.where(
                self.alpha > 2,
                self.scale**2 * self.alpha / ((self.alpha - 1) ** 2 * (self.alpha - 2)),
                mx.array(float("inf")),
            )
        )

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(self.scale / mx.power(u, 1 / self.alpha))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(
            mx.log(self.alpha) + self.alpha * mx.log(self.scale) - (self.alpha + 1) * mx.log(data)
        )

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(1 - mx.power(self.scale / data, self.alpha))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(self.scale / mx.power(1 - data, 1 / self.alpha))

    def entropy(self) -> Tensor:
        return Tensor(mx.log(self.scale / self.alpha) + 1 + 1 / self.alpha)


__all__ = ["Pareto"]
