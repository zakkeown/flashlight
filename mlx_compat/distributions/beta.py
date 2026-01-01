"""Beta Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import betaln, digamma
from .exp_family import ExponentialFamily
from . import constraints
from .gamma import Gamma


class Beta(ExponentialFamily):
    """
    Beta distribution.

    Args:
        concentration1: Alpha parameter
        concentration0: Beta parameter
        validate_args: Whether to validate arguments
    """

    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Union[Tensor, float],
        concentration0: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.concentration1 = concentration1._mlx_array if isinstance(concentration1, Tensor) else mx.array(concentration1)
        self.concentration0 = concentration0._mlx_array if isinstance(concentration0, Tensor) else mx.array(concentration0)

        batch_shape = mx.broadcast_shapes(self.concentration1.shape, self.concentration0.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.concentration1 / (self.concentration1 + self.concentration0))

    @property
    def mode(self) -> Tensor:
        a, b = self.concentration1, self.concentration0
        return Tensor(mx.where((a > 1) & (b > 1), (a - 1) / (a + b - 2), mx.array(float('nan'))))

    @property
    def variance(self) -> Tensor:
        total = self.concentration1 + self.concentration0
        return Tensor(self.concentration1 * self.concentration0 / (total ** 2 * (total + 1)))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Sample using gamma ratio: Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
        gamma_a = Gamma(self.concentration1, 1.0)
        gamma_b = Gamma(self.concentration0, 1.0)
        x = gamma_a.sample(sample_shape)._mlx_array
        y = gamma_b.sample(sample_shape)._mlx_array
        return Tensor(x / (x + y))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        a, b = self.concentration1, self.concentration0
        log_beta_val = betaln(a, b)
        log_prob = (a - 1) * mx.log(data) + (b - 1) * mx.log(1 - data) - log_beta_val
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        a, b = self.concentration1, self.concentration0
        log_beta_val = betaln(a, b)
        psi_sum = digamma(a + b)
        psi_a = digamma(a)
        psi_b = digamma(b)
        return Tensor(log_beta_val - (a - 1) * psi_a - (b - 1) * psi_b + (a + b - 2) * psi_sum)

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.concentration1), Tensor(self.concentration0))

    def _log_normalizer(self, x, y) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        y_data = y._mlx_array if isinstance(y, Tensor) else y
        log_beta_val = betaln(x_data, y_data)
        return Tensor(log_beta_val)


__all__ = ['Beta']
