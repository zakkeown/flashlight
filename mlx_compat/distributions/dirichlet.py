"""Dirichlet Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .exp_family import ExponentialFamily
from . import constraints


class Dirichlet(ExponentialFamily):
    """Dirichlet distribution."""

    arg_constraints = {'concentration': constraints.independent(constraints.positive, 1)}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, concentration: Union[Tensor, mx.array], validate_args: Optional[bool] = None):
        self.concentration = concentration._data if isinstance(concentration, Tensor) else mx.array(concentration)
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
        shape = sample_shape + self._batch_shape + self._event_shape
        # Sample gamma for each component and normalize
        gammas = mx.random.gamma(self.concentration, shape)
        return Tensor(gammas / mx.sum(gammas, axis=-1, keepdims=True))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        # Log Dirichlet PDF
        log_gamma_sum = sp.gammaln(np.array(mx.sum(self.concentration, axis=-1)))
        log_gamma_sum = mx.array(log_gamma_sum.astype(np.float32))
        log_gamma_each = sp.gammaln(np.array(self.concentration))
        log_gamma_each = mx.array(log_gamma_each.astype(np.float32))
        log_beta = mx.sum(log_gamma_each, axis=-1) - log_gamma_sum
        log_prob = -log_beta + mx.sum((self.concentration - 1) * mx.log(data + 1e-10), axis=-1)
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        import numpy as np
        from scipy import special as sp
        k = self.concentration.shape[-1]
        a0 = mx.sum(self.concentration, axis=-1)
        log_gamma_each = sp.gammaln(np.array(self.concentration))
        log_gamma_each = mx.array(log_gamma_each.astype(np.float32))
        log_gamma_sum = sp.gammaln(np.array(a0))
        log_gamma_sum = mx.array(log_gamma_sum.astype(np.float32))
        log_beta = mx.sum(log_gamma_each, axis=-1) - log_gamma_sum
        digamma_each = sp.digamma(np.array(self.concentration))
        digamma_each = mx.array(digamma_each.astype(np.float32))
        digamma_sum = sp.digamma(np.array(a0))
        digamma_sum = mx.array(digamma_sum.astype(np.float32))
        return Tensor(log_beta + (a0 - k) * digamma_sum - mx.sum((self.concentration - 1) * digamma_each, axis=-1))

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.concentration - 1),)

    def _log_normalizer(self, x) -> Tensor:
        import numpy as np
        from scipy import special as sp
        x_data = x._data if isinstance(x, Tensor) else x
        log_gamma_each = sp.gammaln(np.array(x_data + 1))
        log_gamma_each = mx.array(log_gamma_each.astype(np.float32))
        log_gamma_sum = sp.gammaln(np.array(mx.sum(x_data + 1, axis=-1)))
        log_gamma_sum = mx.array(log_gamma_sum.astype(np.float32))
        return Tensor(mx.sum(log_gamma_each, axis=-1) - log_gamma_sum)


__all__ = ['Dirichlet']
