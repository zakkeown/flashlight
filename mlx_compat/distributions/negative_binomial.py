"""Negative Binomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class NegativeBinomial(Distribution):
    """Negative Binomial distribution."""

    arg_constraints = {'total_count': constraints.nonnegative,
                      'probs': constraints.half_open_interval(0, 1),
                      'logits': constraints.real}
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Union[Tensor, float],
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
    def mean(self) -> Tensor:
        return Tensor(self.total_count * self.probs / (1 - self.probs))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.where(self.total_count > 1,
                              mx.floor((self.total_count - 1) * self.probs / (1 - self.probs)),
                              mx.array(0.0)))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.total_count * self.probs / (1 - self.probs) ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        import numpy as np
        # Negative binomial as Poisson-gamma mixture
        r = np.array(self.total_count)
        p = np.array(self.probs)
        samples = np.random.negative_binomial(r, 1 - p, shape)
        return Tensor(mx.array(samples.astype(np.float32)))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        r = self.total_count
        k = data
        log_comb = sp.gammaln(np.array(k + r)) - sp.gammaln(np.array(k + 1)) - sp.gammaln(np.array(r))
        log_comb = mx.array(log_comb.astype(np.float32))
        return Tensor(log_comb + r * mx.log(1 - self.probs) + k * mx.log(self.probs))


__all__ = ['NegativeBinomial']
