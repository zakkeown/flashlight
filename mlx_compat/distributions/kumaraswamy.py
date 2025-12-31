"""Kumaraswamy Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Kumaraswamy(Distribution):
    """Kumaraswamy distribution."""

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
        import numpy as np
        from scipy import special as sp
        a, b = np.array(self.concentration1), np.array(self.concentration0)
        mean = b * sp.beta(1 + 1/a, b)
        return Tensor(mx.array(mean.astype(np.float32)))

    @property
    def mode(self) -> Tensor:
        a, b = self.concentration1, self.concentration0
        return Tensor(mx.where((a >= 1) & (b >= 1) & ((a > 1) | (b > 1)),
                              mx.power((a - 1) / (a * b - 1), 1 / a),
                              mx.array(float('nan'))))

    @property
    def variance(self) -> Tensor:
        import numpy as np
        from scipy import special as sp
        a, b = np.array(self.concentration1), np.array(self.concentration0)
        moment2 = b * sp.beta(1 + 2/a, b)
        mean = b * sp.beta(1 + 1/a, b)
        return Tensor(mx.array((moment2 - mean ** 2).astype(np.float32)))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(mx.power(1 - mx.power(1 - u, 1 / self.concentration0), 1 / self.concentration1))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        log_prob = (mx.log(self.concentration1) + mx.log(self.concentration0) +
                   (self.concentration1 - 1) * mx.log(data) +
                   (self.concentration0 - 1) * mx.log(1 - mx.power(data, self.concentration1)))
        return Tensor(log_prob)


__all__ = ['Kumaraswamy']
