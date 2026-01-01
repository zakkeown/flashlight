"""Multinomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma
from .distribution import Distribution
from . import constraints
from ._constants import UNIFORM_LOW, UNIFORM_HIGH, xlogy


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
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
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
        # Multinomial sampling using repeated categorical sampling
        shape = sample_shape + self._batch_shape
        num_categories = self.probs.shape[-1]

        # Initialize counts to zero
        if len(shape) > 0:
            full_shape = shape + (num_categories,)
        else:
            full_shape = (num_categories,)

        counts = mx.zeros(full_shape, dtype=mx.float32)

        # Sample total_count times and accumulate counts
        # For each sample, draw from categorical and increment the count
        probs_expanded = mx.broadcast_to(self.probs, full_shape)

        for _ in range(self.total_count):
            # Sample from categorical distribution using Gumbel-max trick
            # Use proper uniform bounds to avoid log(0)
            u = mx.random.uniform(low=UNIFORM_LOW, high=UNIFORM_HIGH, shape=full_shape)
            gumbel = -mx.log(-mx.log(u))
            # Use pre-computed logits for numerical stability
            logits = mx.broadcast_to(self.logits, full_shape) + gumbel

            # One-hot encode the selected category
            indices = mx.argmax(logits, axis=-1, keepdims=True)
            one_hot = mx.zeros(full_shape)
            # Create one-hot using scatter-like operation
            # For each position, set 1 at the sampled index
            for cat_idx in range(num_categories):
                mask = (indices == cat_idx)
                one_hot = one_hot.at[..., cat_idx].add(mx.squeeze(mask.astype(mx.float32), axis=-1))

            counts = counts + one_hot

        return Tensor(counts)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        # Log multinomial coefficient + sum of log probs
        log_factorial_n = lgamma(mx.array(self.total_count + 1, dtype=mx.float32))
        log_factorial_k = lgamma(data + 1)
        log_coeff = log_factorial_n - mx.sum(log_factorial_k, axis=-1)
        # Use xlogy for numerical stability: xlogy(k, p) = 0 when k = 0
        log_probs = mx.sum(xlogy(data, self.probs), axis=-1)
        return Tensor(log_coeff + log_probs)


__all__ = ['Multinomial']
