"""Binomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma
from .distribution import Distribution
from . import constraints


class Binomial(Distribution):
    """Binomial distribution."""

    arg_constraints = {'total_count': constraints.nonnegative_integer,
                      'probs': constraints.unit_interval,
                      'logits': constraints.real}

    def __init__(
        self,
        total_count: Union[int, Tensor] = 1,
        probs: Optional[Union[Tensor, float]] = None,
        logits: Optional[Union[Tensor, float]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        self.total_count = total_count._mlx_array if isinstance(total_count, Tensor) else mx.array(total_count)
        if probs is not None:
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs) - mx.log(1 - self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.sigmoid(self.logits)

        batch_shape = mx.broadcast_shapes(self.total_count.shape, self.probs.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints.integer_interval(0, int(mx.max(self.total_count)))

    @property
    def mean(self) -> Tensor:
        return Tensor(self.total_count * self.probs)

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.floor((self.total_count + 1) * self.probs))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.total_count * self.probs * (1 - self.probs))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Binomial sampling using n independent Bernoulli trials
        shape = sample_shape + self._batch_shape
        n = mx.broadcast_to(self.total_count, shape)
        p = mx.broadcast_to(self.probs, shape)

        # For simplicity, sum n Bernoulli trials
        # This works well for small n; for large n, consider normal approximation
        max_n = int(mx.max(n))
        if max_n <= 100:
            samples = mx.zeros(shape)
            for _ in range(max_n):
                u = mx.random.uniform(shape=shape)
                samples = samples + (u < p).astype(mx.float32)
            # Clamp to actual n
            samples = mx.minimum(samples, n.astype(mx.float32))
        else:
            # Use normal approximation for large n
            mean = n * p
            std = mx.sqrt(n * p * (1 - p))
            samples = mx.round(mean + std * mx.random.normal(shape))
            samples = mx.clip(samples, 0, n)

        return Tensor(samples)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        n = self.total_count
        k = data
        # log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_comb = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        return Tensor(log_comb + k * mx.log(self.probs + 1e-10) + (n - k) * mx.log(1 - self.probs + 1e-10))


__all__ = ['Binomial']
