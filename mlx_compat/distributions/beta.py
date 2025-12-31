"""Beta Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .exp_family import ExponentialFamily
from . import constraints


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
        self.concentration1 = concentration1._data if isinstance(concentration1, Tensor) else mx.array(concentration1)
        self.concentration0 = concentration0._data if isinstance(concentration0, Tensor) else mx.array(concentration0)

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
        shape = sample_shape + self._batch_shape
        # Sample using gamma ratio
        x = mx.random.gamma(self.concentration1, shape)
        y = mx.random.gamma(self.concentration0, shape)
        return Tensor(x / (x + y))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        a, b = self.concentration1, self.concentration0
        # Use scipy for log beta function
        import numpy as np
        from scipy import special as sp
        log_beta = sp.betaln(np.array(a), np.array(b))
        log_beta = mx.array(log_beta.astype(np.float32))
        log_prob = (a - 1) * mx.log(data) + (b - 1) * mx.log(1 - data) - log_beta
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        a, b = self.concentration1, self.concentration0
        import numpy as np
        from scipy import special as sp
        log_beta = sp.betaln(np.array(a), np.array(b))
        log_beta = mx.array(log_beta.astype(np.float32))
        psi_sum = sp.digamma(np.array(a + b))
        psi_sum = mx.array(psi_sum.astype(np.float32))
        psi_a = sp.digamma(np.array(a))
        psi_a = mx.array(psi_a.astype(np.float32))
        psi_b = sp.digamma(np.array(b))
        psi_b = mx.array(psi_b.astype(np.float32))
        return Tensor(log_beta - (a - 1) * psi_a - (b - 1) * psi_b + (a + b - 2) * psi_sum)

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.concentration1), Tensor(self.concentration0))

    def _log_normalizer(self, x, y) -> Tensor:
        import numpy as np
        from scipy import special as sp
        x_data = x._data if isinstance(x, Tensor) else x
        y_data = y._data if isinstance(y, Tensor) else y
        log_beta = sp.betaln(np.array(x_data), np.array(y_data))
        return Tensor(mx.array(log_beta.astype(np.float32)))


__all__ = ['Beta']
