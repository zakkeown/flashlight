"""Dirichlet Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma, digamma
from .exp_family import ExponentialFamily
from .gamma import Gamma
from . import constraints
from ._constants import xlogy


class Dirichlet(ExponentialFamily):
    """Dirichlet distribution."""

    arg_constraints = {'concentration': constraints.independent(constraints.positive, 1)}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration: Union[Tensor, mx.array], validate_args: Optional[bool] = None):
        self.concentration = concentration._mlx_array if isinstance(concentration, Tensor) else mx.array(concentration)
        batch_shape = self.concentration.shape[:-1]
        event_shape = self.concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.concentration / mx.sum(self.concentration, axis=-1, keepdims=True))

    @property
    def mode(self) -> Tensor:
        # Mode only exists when all concentration > 1
        return Tensor((self.concentration - 1) / (mx.sum(self.concentration, axis=-1, keepdims=True) - self.concentration.shape[-1]))

    @property
    def variance(self) -> Tensor:
        con0 = mx.sum(self.concentration, axis=-1, keepdims=True)
        return Tensor(self.concentration * (con0 - self.concentration) / (con0 ** 2 * (con0 + 1)))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Sample gamma variates for each concentration parameter
        # Dirichlet samples are normalized gamma variates
        shape = sample_shape + self._batch_shape + self._event_shape

        # Create a Gamma distribution with concentration parameters and rate=1
        gamma_dist = Gamma(
            concentration=Tensor(mx.broadcast_to(self.concentration, shape)),
            rate=Tensor(mx.ones(shape))
        )
        gammas = gamma_dist.sample()._mlx_array

        # Normalize to get Dirichlet samples
        return Tensor(gammas / mx.sum(gammas, axis=-1, keepdims=True))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        # Log Dirichlet PDF
        log_gamma_sum = lgamma(mx.sum(self.concentration, axis=-1))
        log_gamma_each = lgamma(self.concentration)
        log_beta = mx.sum(log_gamma_each, axis=-1) - log_gamma_sum
        # Use xlogy for numerical stability when concentration=1 (which makes coeff=0)
        log_prob = -log_beta + mx.sum(xlogy(self.concentration - 1, data), axis=-1)
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        k = self.concentration.shape[-1]
        a0 = mx.sum(self.concentration, axis=-1)
        log_gamma_each = lgamma(self.concentration)
        log_gamma_sum = lgamma(a0)
        log_beta = mx.sum(log_gamma_each, axis=-1) - log_gamma_sum
        digamma_each = digamma(self.concentration)
        digamma_sum = digamma(a0)
        return Tensor(log_beta + (a0 - k) * digamma_sum - mx.sum((self.concentration - 1) * digamma_each, axis=-1))

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.concentration - 1),)

    def _log_normalizer(self, x) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        log_gamma_each = lgamma(x_data + 1)
        log_gamma_sum = lgamma(mx.sum(x_data + 1, axis=-1))
        return Tensor(mx.sum(log_gamma_each, axis=-1) - log_gamma_sum)


__all__ = ['Dirichlet']
