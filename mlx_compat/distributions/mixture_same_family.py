"""Mixture of Same Family Distribution"""

from typing import Optional, Tuple
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class MixtureSameFamily(Distribution):
    """
    Mixture of distributions from the same family.

    Args:
        mixture_distribution: Categorical mixing distribution
        component_distribution: Distribution for each mixture component
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(
        self,
        mixture_distribution: Distribution,
        component_distribution: Distribution,
        validate_args: Optional[bool] = None,
    ):
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution

        # Validate shapes
        mix_batch_shape = mixture_distribution.batch_shape
        comp_batch_shape = component_distribution.batch_shape
        num_components = comp_batch_shape[-1]

        batch_shape = comp_batch_shape[:-1]
        event_shape = component_distribution.event_shape

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def support(self):
        return self.component_distribution.support

    @property
    def mean(self) -> Tensor:
        probs = self.mixture_distribution.probs
        if isinstance(probs, Tensor):
            probs = probs._data
        comp_mean = self.component_distribution.mean
        if isinstance(comp_mean, Tensor):
            comp_mean = comp_mean._data
        # Weighted average of component means
        probs = mx.expand_dims(probs, -1)
        return Tensor(mx.sum(probs * comp_mean, axis=-2))

    @property
    def variance(self) -> Tensor:
        probs = self.mixture_distribution.probs
        if isinstance(probs, Tensor):
            probs = probs._data
        probs = mx.expand_dims(probs, -1)
        comp_mean = self.component_distribution.mean
        if isinstance(comp_mean, Tensor):
            comp_mean = comp_mean._data
        comp_var = self.component_distribution.variance
        if isinstance(comp_var, Tensor):
            comp_var = comp_var._data
        mean = self.mean._data
        # Var = E[Var(X|Z)] + Var(E[X|Z])
        return Tensor(mx.sum(probs * (comp_var + comp_mean ** 2), axis=-2) - mean ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Sample mixture indices
        indices = self.mixture_distribution.sample(sample_shape)
        if isinstance(indices, Tensor):
            indices = indices._data.astype(mx.int32)
        # Sample from all components
        comp_samples = self.component_distribution.sample(sample_shape)
        if isinstance(comp_samples, Tensor):
            comp_samples = comp_samples._data
        # Select based on indices
        result = mx.take_along_axis(comp_samples, mx.expand_dims(indices, -1), axis=-2)
        return Tensor(mx.squeeze(result, -2))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        # Expand value for each component
        data = mx.expand_dims(data, -len(self._event_shape) - 1)
        # Log prob for each component
        comp_log_prob = self.component_distribution.log_prob(Tensor(data))
        if isinstance(comp_log_prob, Tensor):
            comp_log_prob = comp_log_prob._data
        # Log mixture weights
        mix_log_prob = mx.log(self.mixture_distribution.probs)
        if isinstance(mix_log_prob, Tensor):
            mix_log_prob = mix_log_prob._data
        # Log-sum-exp over components
        return Tensor(mx.logsumexp(mix_log_prob + comp_log_prob, axis=-1))


__all__ = ['MixtureSameFamily']
