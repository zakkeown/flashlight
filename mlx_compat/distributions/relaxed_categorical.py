"""Relaxed Categorical Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class RelaxedOneHotCategorical(Distribution):
    """Relaxed One-Hot Categorical distribution (Concrete/Gumbel-Softmax)."""

    arg_constraints = {'temperature': constraints.positive,
                      'probs': constraints.simplex,
                      'logits': constraints.real_vector}
    support = constraints.simplex
    has_rsample = True

    def __init__(
        self,
        temperature: Union[Tensor, float],
        probs: Optional[Union[Tensor, mx.array]] = None,
        logits: Optional[Union[Tensor, mx.array]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        self.temperature = temperature._mlx_array if isinstance(temperature, Tensor) else mx.array(temperature)
        if probs is not None:
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.softmax(self.logits, axis=-1)

        batch_shape = self.probs.shape[:-1]
        event_shape = self.probs.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape + self._event_shape
        u = mx.random.uniform(shape)
        # Gumbel noise
        gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
        # Softmax with temperature
        return Tensor(mx.softmax((self.logits + gumbel) / self.temperature, axis=-1))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        K = self._event_shape[0]
        log_scale = (K - 1) * mx.log(self.temperature)
        score = self.logits - self.temperature * mx.log(data + 1e-10)
        score = score - mx.logsumexp(score, axis=-1, keepdims=True)
        log_prob = log_scale + mx.sum(score - mx.log(data + 1e-10), axis=-1)
        return Tensor(log_prob)


__all__ = ['RelaxedOneHotCategorical']
