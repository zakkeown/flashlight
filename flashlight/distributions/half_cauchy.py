"""Half-Cauchy Distribution"""

import math
from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class HalfCauchy(Distribution):
    """Half-Cauchy distribution (absolute value of Cauchy)."""

    arg_constraints = {"scale": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, scale: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else mx.array(scale)
        super().__init__(self.scale.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        # Use mx.full instead of mx.full_like (MLX doesn't have full_like)
        return Tensor(mx.full(self.scale.shape, float("inf"), dtype=self.scale.dtype))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.zeros_like(self.scale))

    @property
    def variance(self) -> Tensor:
        # Use mx.full instead of mx.full_like (MLX doesn't have full_like)
        return Tensor(mx.full(self.scale.shape, float("inf"), dtype=self.scale.dtype))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(mx.abs(self.scale * mx.tan(math.pi * (u - 0.5))))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(
            math.log(2 / math.pi) - mx.log(self.scale) - mx.log(1 + (data / self.scale) ** 2)
        )

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(2 / math.pi * mx.arctan(data / self.scale))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(self.scale * mx.tan(math.pi / 2 * data))

    def entropy(self) -> Tensor:
        return Tensor(mx.log(2 * math.pi * self.scale))


__all__ = ["HalfCauchy"]
