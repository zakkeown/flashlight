"""Gamma Distribution"""

import math
from typing import Optional, Tuple, Union

import mlx.core as mx

from ..ops.special import digamma, gammainc, lgamma
from ..tensor import Tensor
from . import constraints
from .exp_family import ExponentialFamily


class Gamma(ExponentialFamily):
    """
    Gamma distribution.

    Args:
        concentration: Shape parameter (alpha/k)
        rate: Rate parameter (beta = 1/scale)
        validate_args: Whether to validate arguments
    """

    arg_constraints = {"concentration": constraints.positive, "rate": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        concentration: Union[Tensor, float],
        rate: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.concentration = (
            concentration._mlx_array
            if isinstance(concentration, Tensor)
            else mx.array(concentration)
        )
        self.rate = rate._mlx_array if isinstance(rate, Tensor) else mx.array(rate)

        batch_shape = mx.broadcast_shapes(self.concentration.shape, self.rate.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.concentration / self.rate)

    @property
    def mode(self) -> Tensor:
        return Tensor(
            mx.where(self.concentration >= 1, (self.concentration - 1) / self.rate, mx.array(0.0))
        )

    @property
    def variance(self) -> Tensor:
        return Tensor(self.concentration / self.rate**2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Sample from Gamma distribution using MLX's native gamma sampler.

        MLX's mx.random.gamma uses shape parameterization (scale=1), so we need
        to divide by rate to get proper Gamma(concentration, rate) samples.

        Gamma(shape, rate) = Gamma(shape, scale=1) / rate
        """
        shape = sample_shape + self._batch_shape

        # Broadcast concentration to output shape
        concentration = mx.broadcast_to(self.concentration, shape)
        rate = mx.broadcast_to(self.rate, shape)

        # Use MLX's native gamma sampling (which uses shape parameterization with scale=1)
        # mx.random.gamma(shape_param, sample_shape) samples from Gamma(shape_param, scale=1)
        samples = mx.random.gamma(concentration, shape)

        # Convert from scale=1 parameterization to rate parameterization
        # Gamma(alpha, rate=beta) = Gamma(alpha, scale=1) / beta
        return Tensor(samples / rate)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        log_gamma_val = lgamma(self.concentration)
        log_prob = (
            self.concentration * mx.log(self.rate)
            + (self.concentration - 1) * mx.log(data)
            - self.rate * data
            - log_gamma_val
        )
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        log_gamma_val = lgamma(self.concentration)
        psi = digamma(self.concentration)
        return Tensor(
            self.concentration - mx.log(self.rate) + log_gamma_val + (1 - self.concentration) * psi
        )

    def cdf(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        result = gammainc(self.concentration, self.rate * data)
        return Tensor(result)

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.concentration - 1), Tensor(-self.rate))

    def _log_normalizer(self, x, y) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        y_data = y._mlx_array if isinstance(y, Tensor) else y
        log_gamma_val = lgamma(x_data + 1)
        return Tensor(log_gamma_val - (x_data + 1) * mx.log(-y_data))


__all__ = ["Gamma"]
