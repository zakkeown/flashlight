"""Negative Binomial Distribution"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..ops.special import lgamma
from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class NegativeBinomial(Distribution):
    """Negative Binomial distribution."""

    arg_constraints = {
        "total_count": constraints.nonnegative,
        "probs": constraints.half_open_interval(0, 1),
        "logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Union[Tensor, float],
        probs: Optional[Union[Tensor, float]] = None,
        logits: Optional[Union[Tensor, float]] = None,
        validate_args: Optional[bool] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified")

        self.total_count = (
            total_count._mlx_array if isinstance(total_count, Tensor) else mx.array(total_count)
        )
        if probs is not None:
            self.probs = probs._mlx_array if isinstance(probs, Tensor) else mx.array(probs)
            self.logits = mx.log(self.probs) - mx.log(1 - self.probs)
        else:
            self.logits = logits._mlx_array if isinstance(logits, Tensor) else mx.array(logits)
            self.probs = mx.sigmoid(self.logits)

        batch_shape = mx.broadcast_shapes(self.total_count.shape, self.probs.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.total_count * self.probs / (1 - self.probs))

    @property
    def mode(self) -> Tensor:
        return Tensor(
            mx.where(
                self.total_count > 1,
                mx.floor((self.total_count - 1) * self.probs / (1 - self.probs)),
                mx.array(0.0),
            )
        )

    @property
    def variance(self) -> Tensor:
        return Tensor(self.total_count * self.probs / (1 - self.probs) ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Sample from Negative Binomial distribution using Gamma-Poisson mixture.

        The Negative Binomial distribution can be represented as a Gamma-Poisson mixture:
        If G ~ Gamma(r, (1-p)/p) then X ~ Poisson(G) has NegBin(r, p)

        This uses MLX's native gamma sampling for accurate results.
        """
        shape = sample_shape + self._batch_shape
        r = mx.broadcast_to(self.total_count, shape)
        p = mx.broadcast_to(self.probs, shape)

        # Gamma rate = (1-p)/p, so scale = p/(1-p)
        # MLX's gamma uses shape parameterization (scale=1)
        # Gamma(shape=r, scale=p/(1-p)) = Gamma(shape=r, scale=1) * p/(1-p)
        scale = p / (1 - p)

        # Sample from Gamma(r, scale=1) using MLX's native gamma
        gamma_samples = mx.random.gamma(r, shape)

        # Scale by p/(1-p) to get Gamma(r, scale=p/(1-p))
        lambda_samples = gamma_samples * scale

        # Sample from Poisson(lambda_samples)
        poisson_samples = self._sample_poisson(lambda_samples)

        return Tensor(poisson_samples)

    def _sample_poisson(self, rate: mx.array) -> mx.array:
        """Sample from Poisson distribution using inverse transform."""
        # For rate > 30, use normal approximation
        # For smaller rates, use inverse transform

        shape = rate.shape

        # Normal approximation for large rates
        normal_samples = mx.round(rate + mx.sqrt(rate) * mx.random.normal(shape))
        normal_samples = mx.maximum(normal_samples, 0)

        # Inverse transform for small rates
        # P(X <= k) = sum_{i=0}^k exp(-lambda) * lambda^i / i!
        u = mx.random.uniform(shape)

        # Iterate to find k such that CDF(k) > u
        k = mx.zeros(shape)
        p = mx.exp(-rate)  # P(X = 0)
        cdf = p

        for i in range(1, 50):  # Limit iterations
            found = cdf >= u
            k = mx.where(found, k, mx.array(float(i)))
            p = p * rate / float(i)
            cdf = cdf + p

        # Use normal approximation for large rates
        result = mx.where(rate > 30, normal_samples, k)

        return result

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else mx.array(value)
        r = self.total_count
        k = data
        # log(C(k+r-1, k)) = lgamma(k+r) - lgamma(k+1) - lgamma(r)
        log_comb = lgamma(k + r) - lgamma(k + 1) - lgamma(r)
        return Tensor(log_comb + r * mx.log(1 - self.probs) + k * mx.log(self.probs))


__all__ = ["NegativeBinomial"]
