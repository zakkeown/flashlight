"""Categorical Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Categorical(Distribution):
    """
    Categorical distribution over integers [0, K).

    Args:
        probs: Event probabilities
        logits: Log probabilities (unnormalized)
        validate_args: Whether to validate arguments
    """

    arg_constraints = {'probs': constraints.simplex, 'logits': constraints.real_vector}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[Union[Tensor, mx.array]] = None,
        logits: Optional[Union[Tensor, mx.array]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        if probs is not None:
            self.probs = probs._data if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs)
        else:
            self.logits = logits._data if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.softmax(self.logits, axis=-1)

        self._num_events = self.probs.shape[-1]
        batch_shape = self.probs.shape[:-1]
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @property
    def mean(self) -> Tensor:
        return Tensor(mx.sum(self.probs * mx.arange(self._num_events), axis=-1))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.argmax(self.probs, axis=-1))

    @property
    def variance(self) -> Tensor:
        mean = mx.sum(self.probs * mx.arange(self._num_events), axis=-1)
        mean_sq = mx.sum(self.probs * mx.arange(self._num_events) ** 2, axis=-1)
        return Tensor(mean_sq - mean ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # Use Gumbel-max trick
        gumbel = -mx.log(-mx.log(mx.random.uniform(shape + (self._num_events,)) + 1e-10) + 1e-10)
        return Tensor(mx.argmax(self.logits + gumbel, axis=-1))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        data = data.astype(mx.int32)
        # Gather log probs at indices
        log_probs = mx.log(self.probs + 1e-10)
        # Index into last dimension
        result = mx.take_along_axis(log_probs, mx.expand_dims(data, -1), axis=-1)
        return Tensor(mx.squeeze(result, -1))

    def entropy(self) -> Tensor:
        return Tensor(-mx.sum(self.probs * mx.log(self.probs + 1e-10), axis=-1))

    def enumerate_support(self, expand: bool = True) -> Tensor:
        values = mx.arange(self._num_events)
        if expand:
            values = mx.broadcast_to(values.reshape(-1, *([1] * len(self._batch_shape))),
                                    (self._num_events,) + self._batch_shape)
        return Tensor(values)


__all__ = ['Categorical']
