"""Negative Binomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints
from ..ops.special import lgamma


class NegativeBinomial(Distribution):
    """Negative Binomial distribution."""

    arg_constraints = {'total_count': constraints.nonnegative,
                      'probs': constraints.half_open_interval(0, 1),
                      'logits': constraints.real}
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

        self.total_count = total_count._mlx_array if isinstance(total_count, Tensor) else mx.array(total_count)
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
        return Tensor(mx.where(self.total_count > 1,
                              mx.floor((self.total_count - 1) * self.probs / (1 - self.probs)),
                              mx.array(0.0)))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.total_count * self.probs / (1 - self.probs) ** 2)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Negative binomial as Gamma-Poisson mixture:
        # If G ~ Gamma(r, (1-p)/p) then X ~ Poisson(G) has NegBin(r, p)
        shape = sample_shape + self._batch_shape
        r = self.total_count
        p = self.probs

        # Gamma rate = (1-p)/p, so scale = p/(1-p)
        # Gamma shape = r
        # Sample from Gamma(r, scale=p/(1-p))
        # MLX gamma takes shape and uses scale=1, so we need to scale ourselves
        scale = p / (1 - p)

        # Sample gamma using shape r
        # gamma_samples = mx.random.gamma(r, shape) doesn't exist in MLX
        # Use inverse transform for gamma when shape >= 1, or rejection sampling
        # For simplicity, use the relation: Gamma(k, theta) can be sampled as sum of k Exp(theta) for integer k
        # But for non-integer k, we need a different approach

        # MLX doesn't have direct gamma sampling, so we use a workaround
        # For integer r, NegBin(r, p) = sum of r Geometric(1-p) samples
        # For general r, we approximate using the gamma-poisson mixture with rejection sampling

        # Simple approach: Use the fact that for large r, NegBin approaches Normal
        # For small r, use explicit gamma sampling approximation

        # Use Marsaglia and Tsang's method for gamma sampling
        gamma_samples = self._sample_gamma(r, shape)

        # Scale by p/(1-p)
        lambda_samples = gamma_samples * scale

        # Sample from Poisson(lambda_samples)
        poisson_samples = self._sample_poisson(lambda_samples)

        return Tensor(poisson_samples)

    def _sample_gamma(self, shape_param: mx.array, sample_shape: Tuple[int, ...]) -> mx.array:
        """Sample from Gamma distribution using Marsaglia and Tsang's method."""
        # For shape >= 1, use Marsaglia and Tsang's method
        # For shape < 1, use boosting: Gamma(a) = Gamma(a+1) * U^(1/a)

        alpha = mx.broadcast_to(shape_param, sample_shape)

        # Handle shape < 1 by boosting
        boost = alpha < 1
        alpha_boosted = mx.where(boost, alpha + 1, alpha)

        # Marsaglia and Tsang's method
        d = alpha_boosted - 1.0 / 3.0
        c = 1.0 / mx.sqrt(9.0 * d)

        # Generate samples (simplified - using a fixed number of iterations)
        # In practice, this is rejection sampling, but we approximate
        result = mx.zeros(sample_shape)

        for _ in range(10):  # Multiple attempts to get valid samples
            # Generate normal samples
            z = mx.random.normal(sample_shape)
            v = (1 + c * z) ** 3

            # Rejection criterion
            u = mx.random.uniform(sample_shape)
            valid = (v > 0) & (mx.log(u) < 0.5 * z * z + d - d * v + d * mx.log(v))

            result = mx.where(valid & (result == 0), d * v, result)

        # Apply boost correction for shape < 1
        u_boost = mx.random.uniform(sample_shape)
        result = mx.where(boost, result * mx.power(u_boost, 1.0 / alpha), result)

        # Fallback for any remaining zeros (use mean as approximation)
        result = mx.where(result <= 0, alpha, result)

        return result

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


__all__ = ['NegativeBinomial']
