"""Inverse Gamma Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class InverseGamma(Distribution):
    """Inverse Gamma distribution."""

    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        concentration: Union[Tensor, float],
        rate: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.concentration = concentration._data if isinstance(concentration, Tensor) else mx.array(concentration)
        self.rate = rate._data if isinstance(rate, Tensor) else mx.array(rate)
        batch_shape = mx.broadcast_shapes(self.concentration.shape, self.rate.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(mx.where(self.concentration > 1, self.rate / (self.concentration - 1), mx.array(float('inf'))))

    @property
    def mode(self) -> Tensor:
        return Tensor(self.rate / (self.concentration + 1))

    @property
    def variance(self) -> Tensor:
        return Tensor(mx.where(self.concentration > 2,
                              self.rate ** 2 / ((self.concentration - 1) ** 2 * (self.concentration - 2)),
                              mx.array(float('inf'))))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # Sample from Gamma, then invert
        gamma_sample = mx.random.gamma(self.concentration, shape) / self.rate
        return Tensor(1 / gamma_sample)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        log_gamma = sp.gammaln(np.array(self.concentration))
        log_gamma = mx.array(log_gamma.astype(np.float32))
        log_prob = (self.concentration * mx.log(self.rate) -
                   log_gamma -
                   (self.concentration + 1) * mx.log(data) -
                   self.rate / data)
        return Tensor(log_prob)


__all__ = ['InverseGamma']
