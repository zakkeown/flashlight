"""Independent Distribution Wrapper"""

from typing import Optional, Tuple

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class Independent(Distribution):
    """
    Reinterpret batch dims as event dims.

    This is useful for constructing distributions with diagonal covariance, e.g.:
    - A batch of independent Normal becomes MultivariateNormal with diagonal cov.
    """

    arg_constraints = {}
    has_rsample = True
    has_enumerate_support = False

    def __init__(
        self,
        base_distribution: Distribution,
        reinterpreted_batch_ndims: int,
        validate_args: Optional[bool] = None,
    ):
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        # Compute new batch and event shapes
        base_batch_shape = base_distribution.batch_shape
        base_event_shape = base_distribution.event_shape

        if reinterpreted_batch_ndims > len(base_batch_shape):
            raise ValueError("reinterpreted_batch_ndims is too large")

        batch_shape = base_batch_shape[: len(base_batch_shape) - reinterpreted_batch_ndims]
        event_shape = (
            base_batch_shape[len(base_batch_shape) - reinterpreted_batch_ndims :] + base_event_shape
        )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @property
    def support(self):
        return constraints.independent(self.base_dist.support, self.reinterpreted_batch_ndims)

    @property
    def mean(self) -> Tensor:
        return self.base_dist.mean

    @property
    def mode(self) -> Tensor:
        return self.base_dist.mode

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        log_prob = self.base_dist.log_prob(value)
        data = log_prob._mlx_array if isinstance(log_prob, Tensor) else log_prob
        # Sum over reinterpreted dims
        for _ in range(self.reinterpreted_batch_ndims):
            data = mx.sum(data, axis=-1)
        return Tensor(data)

    def entropy(self) -> Tensor:
        entropy = self.base_dist.entropy()
        data = entropy._mlx_array if isinstance(entropy, Tensor) else entropy
        for _ in range(self.reinterpreted_batch_ndims):
            data = mx.sum(data, axis=-1)
        return Tensor(data)

    def enumerate_support(self, expand: bool = True) -> Tensor:
        return self.base_dist.enumerate_support(expand=expand)


__all__ = ["Independent"]
