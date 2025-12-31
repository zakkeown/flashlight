"""Weibull Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Weibull(Distribution):
    """Weibull distribution."""

    arg_constraints = {'scale': constraints.positive, 'concentration': constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, scale: Union[Tensor, float], concentration: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.scale = scale._data if isinstance(scale, Tensor) else mx.array(scale)
        self.concentration = concentration._data if isinstance(concentration, Tensor) else mx.array(concentration)
        batch_shape = mx.broadcast_shapes(self.scale.shape, self.concentration.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        import numpy as np
        from scipy import special as sp
        gamma_val = sp.gamma(1 + 1 / np.array(self.concentration))
        return Tensor(self.scale * mx.array(gamma_val.astype(np.float32)))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.where(self.concentration > 1,
                              self.scale * mx.power((self.concentration - 1) / self.concentration, 1 / self.concentration),
                              mx.array(0.0)))

    @property
    def variance(self) -> Tensor:
        import numpy as np
        from scipy import special as sp
        k = np.array(self.concentration)
        gamma1 = sp.gamma(1 + 2 / k)
        gamma2 = sp.gamma(1 + 1 / k) ** 2
        return Tensor(self.scale ** 2 * mx.array((gamma1 - gamma2).astype(np.float32)))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(self.scale * mx.power(-mx.log(1 - u), 1 / self.concentration))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(mx.log(self.concentration / self.scale) +
                     (self.concentration - 1) * mx.log(data / self.scale) -
                     mx.power(data / self.scale, self.concentration))

    def cdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(1 - mx.exp(-mx.power(data / self.scale, self.concentration)))

    def icdf(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        return Tensor(self.scale * mx.power(-mx.log(1 - data), 1 / self.concentration))

    def entropy(self) -> Tensor:
        euler_gamma = 0.5772156649015329
        return Tensor(euler_gamma * (1 - 1 / self.concentration) + mx.log(self.scale / self.concentration) + 1)


__all__ = ['Weibull']
