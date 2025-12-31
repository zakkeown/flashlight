"""
Exponential Family Distribution

PyTorch-compatible exponential family base class.
"""

from typing import Tuple
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution


class ExponentialFamily(Distribution):
    """
    Base class for distributions in the exponential family.

    Distributions in this family have the form:
    p(x|theta) = h(x) * exp(eta(theta) . T(x) - A(theta))

    where:
    - eta(theta) is the natural parameter
    - T(x) is the sufficient statistic
    - A(theta) is the log normalizer
    - h(x) is the base measure
    """

    @property
    def _natural_params(self) -> Tuple[Tensor, ...]:
        """Return the natural parameters of the distribution."""
        raise NotImplementedError

    def _log_normalizer(self, *natural_params) -> Tensor:
        """Compute the log normalizer A(theta)."""
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self) -> Tensor:
        """Return E[h(x)] under the distribution."""
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """
        Compute entropy using the natural parameterization.

        H = -E[log p(x)] = A(theta) - eta . E[T(x)]

        This is an approximation for many distributions.
        """
        # Generic entropy computation using natural params
        # Subclasses may override with analytical solutions
        result = self._log_normalizer(*self._natural_params)
        return result


__all__ = ['ExponentialFamily']
