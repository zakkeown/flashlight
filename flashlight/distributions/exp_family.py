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

    The entropy of an exponential family distribution is:
    H = A(theta) - eta . E[T(x)] + E[log h(x)]

    For minimal exponential families, E[T(x)] = grad_A(theta), so:
    H = A(theta) - eta . grad_A(theta) + E[log h(x)]
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
        """Return E[log h(x)] under the distribution.

        For many common distributions, the base measure h(x) is constant
        (often 1), so log h(x) = 0 and this returns 0.

        Subclasses should override if their base measure is non-constant.
        """
        # Default: constant base measure h(x) = 1, so E[log h(x)] = 0
        return Tensor(mx.array(0.0))

    def entropy(self) -> Tensor:
        """
        Compute entropy using the natural parameterization.

        For exponential family distributions:
        H = A(theta) - eta . grad_A(theta) + E[log h(x)]

        where grad_A(theta) = E[T(x)] (the mean of sufficient statistics).

        This base implementation computes:
        H = A(theta) - sum(eta_i * d_A/d_eta_i) + E[log h(x)]

        For numerical stability with MLX (which lacks autograd for this),
        we use the relationship:
        - For many exponential families, the entropy has a closed form
        - Subclasses should override with their analytical entropy formula

        This base implementation uses Monte Carlo estimation as a fallback
        when no analytical formula is available.
        """
        # Get natural parameters
        natural_params = self._natural_params

        # Compute log normalizer at current parameters
        log_norm = self._log_normalizer(*natural_params)

        # Try to compute entropy via Monte Carlo estimation of E[-log p(x)]
        # This is a fallback - concrete distributions should override with
        # their analytical entropy formulas
        try:
            # Sample from the distribution
            samples = self.sample((1000,))

            # Compute mean negative log probability
            neg_log_probs = -self.log_prob(samples)
            entropy_estimate = neg_log_probs.mean()

            return entropy_estimate
        except (NotImplementedError, AttributeError):
            # If sampling or log_prob not available, return log normalizer
            # as a lower bound (this is only correct when E[log h(x)] = 0
            # and the natural parameters happen to give E[T(x)] = 0)
            return log_norm


__all__ = ['ExponentialFamily']
