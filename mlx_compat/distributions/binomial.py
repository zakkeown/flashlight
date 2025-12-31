"""Binomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
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

        self.total_count = total_count._data if isinstance(total_count, Tensor) else mx.array(total_count)
        if probs is not None:
            self.probs = probs._data if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs) - mx.log(1 - self.probs)
        else:
            self.logits = logits._data if isinstance(logits, Tensor) else mx.array(logits)
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
        shape = sample_shape + self._batch_shape
        import numpy as np
        samples = np.random.binomial(int(mx.max(self.total_count)), np.array(self.probs), shape)
        return Tensor(mx.array(samples.astype(np.float32)))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        n = self.total_count
        k = data
        log_comb = sp.gammaln(np.array(n) + 1) - sp.gammaln(np.array(k) + 1) - sp.gammaln(np.array(n - k) + 1)
        log_comb = mx.array(log_comb.astype(np.float32))
        return Tensor(log_comb + k * mx.log(self.probs + 1e-10) + (n - k) * mx.log(1 - self.probs + 1e-10))


__all__ = ['Binomial']
