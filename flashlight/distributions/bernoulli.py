"""Bernoulli Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from ._constants import xlogy
from .exp_family import ExponentialFamily


class Bernoulli(ExponentialFamily):
    """
    Bernoulli distribution.

    Args:
        probs: Probability of sampling 1
        logits: Log-odds of sampling 1
        validate_args: Whether to validate arguments
    """

    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[Union[Tensor, float]] = None,
        logits: Optional[Union[Tensor, float]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        if probs is not None:
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs) - mx.log(1 - self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.sigmoid(self.logits)

        batch_shape = self.probs.shape
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.probs)

    @property
    def mode(self) -> Tensor:
        return Tensor((self.probs >= 0.5).astype(mx.float32))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.probs * (1 - self.probs))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        return Tensor((mx.random.uniform(shape) < self.probs).astype(mx.float32))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        log_prob = data * self.logits - mx.logaddexp(mx.array(0.0), self.logits)
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        # Use xlogy for numerical stability: xlogy(p, p) = 0 when p = 0
        p = self.probs
        return Tensor(-xlogy(p, p) - xlogy(1 - p, 1 - p))

    def enumerate_support(self, expand: bool = True) -> Tensor:
        values = mx.array([0.0, 1.0])
        if expand:
            values = mx.broadcast_to(
                values.reshape(-1, *([1] * len(self._batch_shape))), (2,) + self._batch_shape
            )
        return Tensor(values)

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        return (Tensor(self.logits),)

    def _log_normalizer(self, x) -> Tensor:
        x_data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.logaddexp(mx.array(0.0), x_data))


__all__ = ["Bernoulli"]
