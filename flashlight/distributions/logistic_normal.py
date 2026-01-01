"""Logistic Normal Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class LogisticNormal(Distribution):
    """Logistic Normal distribution."""

    arg_constraints = {"loc": constraints.real_vector, "scale": constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(
        self,
        loc: Union[Tensor, mx.array],
        scale: Union[Tensor, mx.array],
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else mx.array(scale)
        batch_shape = mx.broadcast_shapes(self.loc.shape[:-1], self.scale.shape[:-1])
        event_shape = (self.loc.shape[-1] + 1,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        # No closed form
        return Tensor(mx.softmax(self.loc, axis=-1))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape + self.loc.shape[-1:]
        eps = mx.random.normal(shape)
        x = self.loc + self.scale * eps
        # Softmax to get simplex values
        return Tensor(
            mx.softmax(mx.concatenate([x, mx.zeros(shape[:-1] + (1,))], axis=-1), axis=-1)
        )

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        # Transform to unconstrained space
        x = mx.log(data[..., :-1]) - mx.log(data[..., -1:])
        # Normal log prob
        import math

        log_prob = -0.5 * mx.sum(
            ((x - self.loc) / self.scale) ** 2 + mx.log(2 * math.pi * self.scale**2), axis=-1
        )
        # Jacobian correction
        log_prob = log_prob - mx.sum(mx.log(data + 1e-10), axis=-1)
        return Tensor(log_prob)


__all__ = ["LogisticNormal"]
