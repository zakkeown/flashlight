"""Continuous Bernoulli Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from ._constants import PROB_EPSILON
from .distribution import Distribution


class ContinuousBernoulli(Distribution):
    """Continuous Bernoulli distribution."""

    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        probs: Optional[Union[Tensor, float]] = None,
        logits: Optional[Union[Tensor, float]] = None,
        lims: Tuple[float, float] = (0.499, 0.501),
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        self._lims = lims
        if probs is not None:
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs) - mx.log(1 - self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.sigmoid(self.logits)

        super().__init__(self.probs.shape, validate_args=validate_args)

    def _cont_bern_log_norm(self):
        """Log normalizing constant."""
        # For lambda near 0.5, use Taylor approximation
        cut_probs = mx.clip(self.probs, self._lims[0], self._lims[1])
        is_unstable = (self.probs < self._lims[0]) | (self.probs > self._lims[1])
        log_norm = mx.where(
            is_unstable,
            mx.log(mx.abs(2 * mx.arctanh(1 - 2 * self.probs)) + PROB_EPSILON)
            - mx.log(mx.abs(1 - 2 * self.probs) + PROB_EPSILON),
            mx.log(mx.array(2.0)),  # Approximation near 0.5
        )
        return log_norm

    @property
    def mean(self) -> Tensor:
        p = self.probs
        is_unstable = (p < self._lims[0]) | (p > self._lims[1])
        mean = mx.where(
            is_unstable,
            p / (2 * p - 1) + 1 / (2 * mx.arctanh(1 - 2 * p) + PROB_EPSILON),
            mx.array(0.5),
        )
        return Tensor(mean)

    @property
    def variance(self) -> Tensor:
        # Simplified variance computation
        mean = self.mean._mlx_array
        return Tensor(mean * (1 - mean) / 3)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        u = mx.random.uniform(shape)
        # Inverse CDF sampling
        p = self.probs
        is_unstable = (p < self._lims[0]) | (p > self._lims[1])
        samples = mx.where(
            is_unstable,
            mx.log(1 + (2 * p - 1) / (1 - p) * (mx.exp(mx.log(1 - p) - mx.log(p) * u) - 1))
            / (2 * mx.arctanh(1 - 2 * p) + PROB_EPSILON),
            u,  # Uniform approximation near 0.5
        )
        return Tensor(mx.clip(samples, 0, 1))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        log_norm = self._cont_bern_log_norm()
        # PDF: C(lambda) * lambda^x * (1-lambda)^(1-x)
        # log PDF: log C + x*log(lambda) + (1-x)*log(1-lambda)
        #        = log C + x*logits - log(1 + exp(logits))
        log_prob = log_norm + data * self.logits - mx.logaddexp(mx.array(0.0), self.logits)
        return Tensor(log_prob)


__all__ = ["ContinuousBernoulli"]
