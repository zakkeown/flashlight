"""Inverse Gamma Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..ops.special import lgamma
from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class InverseGamma(Distribution):
    """Inverse Gamma distribution."""

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
        return Tensor(
            mx.where(
                self.concentration > 1, self.rate / (self.concentration - 1), mx.array(float("inf"))
            )
        )

    @property
    def mode(self) -> Tensor:
        return Tensor(self.rate / (self.concentration + 1))

    @property
    def variance(self) -> Tensor:
        return Tensor(
            mx.where(
                self.concentration > 2,
                self.rate**2 / ((self.concentration - 1) ** 2 * (self.concentration - 2)),
                mx.array(float("inf")),
            )
        )

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # Sample from Gamma(concentration, 1), then compute rate / gamma_sample
        # InverseGamma(alpha, beta) = 1 / Gamma(alpha, beta)
        # where Gamma(alpha, beta) has mean alpha/beta
        gamma_sample = mx.random.gamma(self.concentration, shape)
        return Tensor(self.rate / gamma_sample)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else mx.array(value)
        log_gamma_conc = lgamma(self.concentration)
        log_prob = (
            self.concentration * mx.log(self.rate)
            - log_gamma_conc
            - (self.concentration + 1) * mx.log(data)
            - self.rate / data
        )
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        from ..ops.special import digamma

        log_gamma_conc = lgamma(self.concentration)
        psi = digamma(self.concentration)
        # Entropy = alpha + log(beta * Gamma(alpha)) - (1 + alpha) * psi(alpha)
        return Tensor(
            self.concentration + mx.log(self.rate) + log_gamma_conc - (1 + self.concentration) * psi
        )


__all__ = ["InverseGamma"]
