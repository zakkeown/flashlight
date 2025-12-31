"""Poisson Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .exp_family import ExponentialFamily
from . import constraints


class Poisson(ExponentialFamily):
    """Poisson distribution."""

    arg_constraints = {'rate': constraints.nonnegative}
    support = constraints.nonnegative_integer

    def __init__(self, rate: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.rate = rate._data if isinstance(rate, Tensor) else mx.array(rate)
        super().__init__(self.rate.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.rate)

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.floor(self.rate))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.rate)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # Use inverse transform sampling approximation
        import numpy as np
        samples = np.random.poisson(np.array(self.rate), shape)
        return Tensor(mx.array(samples.astype(np.float32)))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        log_factorial = sp.gammaln(np.array(data) + 1)
        log_factorial = mx.array(log_factorial.astype(np.float32))
        return Tensor(data * mx.log(self.rate + 1e-10) - self.rate - log_factorial)

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(mx.log(self.rate + 1e-10)),)

    def _log_normalizer(self, x) -> Tensor:
        x_data = x._data if isinstance(x, Tensor) else x
        return Tensor(mx.exp(x_data))


__all__ = ['Poisson']
