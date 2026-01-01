"""Transformed Distribution"""

from typing import List, Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from .transforms import Transform, ComposeTransform
from . import constraints


class TransformedDistribution(Distribution):
    """
    Distribution class for applying a sequence of transforms to a base distribution.
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Union[Transform, List[Transform]],
        validate_args: Optional[bool] = None,
    ):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms]
        else:
            self.transforms = list(transforms)

        # Compute transformed event shape
        shape = base_distribution.event_shape
        for t in self.transforms:
            shape = self._transform_shape(t, shape)

        super().__init__(base_distribution.batch_shape, shape, validate_args=validate_args)

    def _transform_shape(self, transform: Transform, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape of transform."""
        # For most transforms, shape is preserved
        return shape

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def support(self):
        if self.transforms:
            return self.transforms[-1].codomain
        return self.base_dist.support

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        x = self.base_dist.sample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        x = self.base_dist.rsample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def log_prob(self, value: Tensor) -> Tensor:
        # Invert transforms
        y = value
        log_det = Tensor(mx.zeros(self._batch_shape))
        for t in reversed(self.transforms):
            x = t.inv(y)
            log_det_jacobian = t.log_abs_det_jacobian(x, y)
            if isinstance(log_det_jacobian, Tensor):
                log_det_jacobian = log_det_jacobian._mlx_array
            if isinstance(log_det, Tensor):
                log_det = log_det._mlx_array
            log_det = log_det - log_det_jacobian
            y = x

        log_prob = self.base_dist.log_prob(y)
        if isinstance(log_prob, Tensor):
            log_prob = log_prob._mlx_array
        return Tensor(log_prob + log_det)

    @property
    def mean(self) -> Tensor:
        # No closed form in general
        raise NotImplementedError("mean not available for TransformedDistribution")

    @property
    def variance(self) -> Tensor:
        raise NotImplementedError("variance not available for TransformedDistribution")


__all__ = ['TransformedDistribution']
