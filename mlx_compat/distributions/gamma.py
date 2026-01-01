"""Gamma Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma, digamma, gammainc
from .exp_family import ExponentialFamily
from . import constraints


class Gamma(ExponentialFamily):
    """
    Gamma distribution.

    Args:
        concentration: Shape parameter (alpha/k)
        rate: Rate parameter (beta = 1/scale)
        validate_args: Whether to validate arguments
    """

    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        concentration: Union[Tensor, float],
        rate: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.concentration = concentration._mlx_array if isinstance(concentration, Tensor) else mx.array(concentration)
        self.rate = rate._mlx_array if isinstance(rate, Tensor) else mx.array(rate)

        batch_shape = mx.broadcast_shapes(self.concentration.shape, self.rate.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.concentration / self.rate)

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.where(self.concentration >= 1, (self.concentration - 1) / self.rate, mx.array(0.0)))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.concentration / self.rate ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # Gamma sampling using Marsaglia and Tsang's method for alpha >= 1
        # For alpha < 1, we use the relation: gamma(alpha) = gamma(alpha+1) * U^(1/alpha)
        alpha = mx.broadcast_to(self.concentration, shape)
        rate = mx.broadcast_to(self.rate, shape)

        # Marsaglia and Tsang's method
        # For alpha >= 1: d = alpha - 1/3, c = 1/sqrt(9*d)
        # Generate: x = normal(0,1), v = (1 + c*x)^3
        # Accept if: log(u) < 0.5*x^2 + d - d*v + d*log(v)

        # For simplicity and to handle all alpha values, we use the
        # transformation: if alpha < 1, sample gamma(alpha+1) then multiply by U^(1/alpha)
        needs_boost = alpha < 1.0
        alpha_work = mx.where(needs_boost, alpha + 1.0, alpha)

        d = alpha_work - 1.0 / 3.0
        c = 1.0 / mx.sqrt(9.0 * d)

        # Rejection sampling loop (we run a fixed number of iterations and take last valid)
        # In practice, the method has very high acceptance rate
        samples = mx.zeros(shape)

        for _ in range(10):  # Usually accepts in 1-2 iterations
            x = mx.random.normal(shape)
            v = (1.0 + c * x) ** 3
            valid_v = v > 0

            u = mx.random.uniform(shape=shape)

            # Acceptance condition
            accept = (
                (mx.log(u) < 0.5 * x * x + d - d * v + d * mx.log(v)) &
                valid_v
            )

            candidate = d * v
            samples = mx.where(accept & (samples == 0), candidate, samples)

        # Fallback: if still zero, use a simple approximation
        samples = mx.where(samples == 0, d, samples)

        # Boost for alpha < 1
        u_boost = mx.random.uniform(shape=shape)
        samples = mx.where(needs_boost, samples * (u_boost ** (1.0 / alpha)), samples)

        # Apply rate
        return Tensor(samples / rate)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        log_gamma_val = lgamma(self.concentration)
        log_prob = (self.concentration * mx.log(self.rate) +
                   (self.concentration - 1) * mx.log(data) -
                   self.rate * data - log_gamma_val)
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        log_gamma_val = lgamma(self.concentration)
        psi = digamma(self.concentration)
        return Tensor(self.concentration - mx.log(self.rate) + log_gamma_val + (1 - self.concentration) * psi)

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


__all__ = ['Gamma']
