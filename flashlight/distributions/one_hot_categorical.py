"""One-Hot Categorical Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from .categorical import Categorical
from . import constraints


class OneHotCategorical(Distribution):
    """One-hot categorical distribution."""

    arg_constraints = {'probs': constraints.simplex, 'logits': constraints.real_vector}
    support = constraints.one_hot
    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[Union[Tensor, mx.array]] = None,
        logits: Optional[Union[Tensor, mx.array]] = None,
        validate_args: Optional[bool] = None,
    ):
        self._categorical = Categorical(probs=probs, logits=logits)
        batch_shape = self._categorical._batch_shape
        event_shape = (self._categorical._num_events,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def probs(self):
        return self._categorical.probs

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def mean(self) -> Tensor:
        return Tensor(self.probs)

    @property
    def mode(self) -> Tensor:
        idx = mx.argmax(self.probs, axis=-1)
        one_hot = mx.zeros_like(self.probs)
        # Create one-hot at argmax position
        return Tensor(mx.eye(self._event_shape[0])[idx])

    @property
    def variance(self) -> Tensor:
        return Tensor(self.probs * (1 - self.probs))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        indices = self._categorical.sample(sample_shape)
        return Tensor(mx.eye(self._event_shape[0])[indices._mlx_array.astype(mx.int32)])

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        indices = mx.argmax(data, axis=-1)
        return self._categorical.log_prob(Tensor(indices))

    def entropy(self) -> Tensor:
        return self._categorical.entropy()

    def enumerate_support(self, expand: bool = True) -> Tensor:
        n = self._event_shape[0]
        values = mx.eye(n)
        if expand:
            values = mx.broadcast_to(values.reshape(n, *([1] * len(self._batch_shape)), n),
                                    (n,) + self._batch_shape + (n,))
        return Tensor(values)


class OneHotCategoricalStraightThrough(OneHotCategorical):
    """
    One-hot categorical with straight-through gradient estimator.

    During sampling, returns a one-hot vector.
    During backprop, uses the softmax probabilities for gradients.
    """

    has_rsample = True

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        samples = self.sample(sample_shape)
        # Straight-through: in forward pass use samples, but backprop through probs
        probs = mx.broadcast_to(self.probs, samples._mlx_array.shape)
        return Tensor(samples._mlx_array + probs - mx.stop_gradient(probs))


__all__ = ['OneHotCategorical', 'OneHotCategoricalStraightThrough']
