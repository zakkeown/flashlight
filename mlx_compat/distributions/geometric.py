"""Geometric Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Geometric(Distribution):
    """Geometric distribution."""

    arg_constraints = {'probs': constraints.unit_interval, 'logits': constraints.real}
    support = constraints.nonnegative_integer

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

        super().__init__(self.probs.shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor((1 - self.probs) / self.probs)

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.zeros_like(self.probs))

    @property
    def variance(self) -> Tensor:
        return Tensor((1 - self.probs) / self.probs ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        return Tensor(mx.floor(mx.log(u) / mx.log(1 - self.probs)))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        return Tensor(data * mx.log(1 - self.probs) + mx.log(self.probs))

    def entropy(self) -> Tensor:
        return Tensor(-(1 - self.probs) * mx.log(1 - self.probs + 1e-10) / self.probs - mx.log(self.probs + 1e-10))


__all__ = ['Geometric']
