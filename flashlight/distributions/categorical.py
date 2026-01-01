"""Categorical Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints
from ._constants import UNIFORM_LOW, UNIFORM_HIGH, xlogy


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
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
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
        # Use Gumbel-max trick with proper uniform bounds to avoid log(0)
        u = mx.random.uniform(low=UNIFORM_LOW, high=UNIFORM_HIGH, shape=shape + (self._num_events,))
        gumbel = -mx.log(-mx.log(u))
        return Tensor(mx.argmax(self.logits + gumbel, axis=-1))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        data = data.astype(mx.int32)
        # Use pre-computed logits (which are log(probs)) for numerical stability
        # Index into last dimension
        result = mx.take_along_axis(self.logits, mx.expand_dims(data, -1), axis=-1)
        return Tensor(mx.squeeze(result, -1))

    def entropy(self) -> Tensor:
        # Use xlogy for numerical stability: xlogy(p, p) = 0 when p = 0
        return Tensor(-mx.sum(xlogy(self.probs, self.probs), axis=-1))

    def enumerate_support(self, expand: bool = True) -> Tensor:
        values = mx.arange(self._num_events)
        if expand:
            values = mx.broadcast_to(values.reshape(-1, *([1] * len(self._batch_shape))),
                                    (self._num_events,) + self._batch_shape)
        return Tensor(values)


__all__ = ['Categorical']
