"""Normal Distribution"""

import math
from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution
from .exp_family import ExponentialFamily


class Normal(ExponentialFamily):
    """
    Normal (Gaussian) distribution.

    Args:
        loc: Mean of the distribution
        scale: Standard deviation of the distribution
        validate_args: Whether to validate arguments
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
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
        return Tensor(self.scale**2)

    @property
    def stddev(self) -> Tensor:
        return Tensor(self.scale)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        eps = mx.random.normal(shape)
        return Tensor(self.loc + self.scale * eps)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        var = self.scale**2
        log_scale = mx.log(self.scale)
        log_prob = -((data - self.loc) ** 2) / (2 * var) - log_scale - 0.5 * math.log(2 * math.pi)
        return Tensor(log_prob)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(0.5 * (1 + mx.erf((data - self.loc) / (self.scale * math.sqrt(2)))))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(self.loc + self.scale * mx.erfinv(2 * data - 1) * math.sqrt(2))

    def entropy(self) -> Tensor:
        return Tensor(0.5 + 0.5 * math.log(2 * math.pi) + mx.log(self.scale))

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.loc / self.scale**2), Tensor(-0.5 / self.scale**2))

    def _log_normalizer(self, x, y) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        y_data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(-0.25 * x_data**2 / y_data + 0.5 * mx.log(-math.pi / y_data))


__all__ = ["Normal"]
