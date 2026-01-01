"""Poisson Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma
from .exp_family import ExponentialFamily
from . import constraints


class Poisson(ExponentialFamily):
    """Poisson distribution."""

    arg_constraints = {'rate': constraints.nonnegative}
    support = constraints.nonnegative_integer

    def __init__(self, rate: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.rate = rate._mlx_array if isinstance(rate, Tensor) else mx.array(rate)
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
        rate = mx.broadcast_to(self.rate, shape)

        # Poisson sampling using inverse transform method
        # For small lambda, use direct method; for large lambda, use normal approximation
        max_rate = float(mx.max(rate))

        if max_rate < 30:
            # Direct method: generate uniforms and count how many needed
            L = mx.exp(-rate)
            samples = mx.zeros(shape)
            p = mx.ones(shape)

            for _ in range(int(max_rate * 3) + 30):
                u = mx.random.uniform(shape=shape)
                p = p * u
                should_increment = p > L
                samples = samples + should_increment.astype(mx.float32)
        else:
            # Normal approximation for large lambda
            samples = mx.round(rate + mx.sqrt(rate) * mx.random.normal(shape))
            samples = mx.maximum(samples, mx.zeros_like(samples))

        return Tensor(samples)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        log_factorial = lgamma(data + 1)
        return Tensor(data * mx.log(self.rate + 1e-10) - self.rate - log_factorial)

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(mx.log(self.rate + 1e-10)),)

    def _log_normalizer(self, x) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.exp(x_data))


__all__ = ['Poisson']
