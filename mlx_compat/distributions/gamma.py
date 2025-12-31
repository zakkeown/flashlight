"""Gamma Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
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
        self.concentration = concentration._data if isinstance(concentration, Tensor) else mx.array(concentration)
        self.rate = rate._data if isinstance(rate, Tensor) else mx.array(rate)

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
        return Tensor(mx.random.gamma(self.concentration, shape) / self.rate)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        log_gamma = sp.gammaln(np.array(self.concentration))
        log_gamma = mx.array(log_gamma.astype(np.float32))
        log_prob = (self.concentration * mx.log(self.rate) +
                   (self.concentration - 1) * mx.log(data) -
                   self.rate * data - log_gamma)
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        import numpy as np
        from scipy import special as sp
        log_gamma = sp.gammaln(np.array(self.concentration))
        log_gamma = mx.array(log_gamma.astype(np.float32))
        psi = sp.digamma(np.array(self.concentration))
        psi = mx.array(psi.astype(np.float32))
        return Tensor(self.concentration - mx.log(self.rate) + log_gamma + (1 - self.concentration) * psi)

    def cdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        result = sp.gammainc(np.array(self.concentration), np.array(self.rate * data))
        return Tensor(mx.array(result.astype(np.float32)))

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.concentration - 1), Tensor(-self.rate))

    def _log_normalizer(self, x, y) -> Tensor:
        import numpy as np
        from scipy import special as sp
        x_data = x._data if isinstance(x, Tensor) else x
        y_data = y._data if isinstance(y, Tensor) else y
        log_gamma = sp.gammaln(np.array(x_data + 1))
        log_gamma = mx.array(log_gamma.astype(np.float32))
        return Tensor(log_gamma - (x_data + 1) * mx.log(-y_data))


__all__ = ['Gamma']
