"""Relaxed Bernoulli Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class RelaxedBernoulli(Distribution):
    """Relaxed Bernoulli distribution (Binary Concrete)."""

    arg_constraints = {'temperature': constraints.positive,
                      'probs': constraints.unit_interval,
                      'logits': constraints.real}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        temperature: Union[Tensor, float],
        probs: Optional[Union[Tensor, float]] = None,
        logits: Optional[Union[Tensor, float]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        self.temperature = temperature._data if isinstance(temperature, Tensor) else mx.array(temperature)
        if probs is not None:
            self.probs = probs._data if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs) - mx.log(1 - self.probs)
        else:
            self.logits = logits._data if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.sigmoid(self.logits)

        batch_shape = mx.broadcast_shapes(self.temperature.shape, self.probs.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        # Gumbel-softmax trick
        return Tensor(mx.sigmoid((mx.log(u) - mx.log(1 - u) + self.logits) / self.temperature))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        diff = self.logits - self.temperature * (mx.log(data) - mx.log(1 - data))
        return Tensor(mx.log(self.temperature) - mx.log(data) - mx.log(1 - data) -
                     2 * mx.logaddexp(mx.array(0.0), -diff))


__all__ = ['RelaxedBernoulli']
