"""Multinomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Multinomial(Distribution):
    """Multinomial distribution."""

    arg_constraints = {'probs': constraints.simplex, 'logits': constraints.real_vector}

    def __init__(
        self,
        total_count: int = 1,
        probs: Optional[Union[Tensor, mx.array]] = None,
        logits: Optional[Union[Tensor, mx.array]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        self.total_count = total_count
        if probs is not None:
            self.probs = probs._data if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs)
        else:
            self.logits = logits._data if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.softmax(self.logits, axis=-1)

        batch_shape = self.probs.shape[:-1]
        event_shape = self.probs.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints.multinomial(self.total_count)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.probs * self.total_count)

    @property
    def variance(self) -> Tensor:
        return Tensor(self.total_count * self.probs * (1 - self.probs))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        import numpy as np
        samples = np.random.multinomial(self.total_count, np.array(self.probs), shape)
        return Tensor(mx.array(samples.astype(np.float32)))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        # Log multinomial coefficient + sum of log probs
        log_factorial_n = sp.gammaln(self.total_count + 1)
        log_factorial_k = sp.gammaln(np.array(data) + 1)
        log_coeff = log_factorial_n - mx.sum(mx.array(log_factorial_k.astype(np.float32)), axis=-1)
        log_probs = mx.sum(data * mx.log(self.probs + 1e-10), axis=-1)
        return Tensor(log_coeff + log_probs)


__all__ = ['Multinomial']
