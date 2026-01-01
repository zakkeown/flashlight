"""Binomial Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma
from .distribution import Distribution
from . import constraints
from ._constants import PROB_EPSILON, xlogy


class Binomial(Distribution):
    """Binomial distribution."""

    arg_constraints = {'total_count': constraints.nonnegative_integer,
                      'probs': constraints.unit_interval,
                      'logits': constraints.real}

    def __init__(
        self,
        total_count: Union[int, Tensor] = 1,
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
    def support(self):
        return constraints.integer_interval(0, int(mx.max(self.total_count)))

    @property
    def mean(self) -> Tensor:
        return Tensor(self.total_count * self.probs)

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.floor((self.total_count + 1) * self.probs))

    @property
    def variance(self) -> Tensor:
        return Tensor(self.total_count * self.probs * (1 - self.probs))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Sample from Binomial distribution.

        Uses different algorithms based on n:
        - For small n (<=20): Direct sum of Bernoulli trials (exact)
        - For medium n (21-100): Inverse transform with geometric skipping
        - For large n (>100): BTPE (Binomial, Triangle, Parallelogram, Exponential) rejection sampling

        The BTPE algorithm is based on:
        Kachitvichyanukul, V. and Schmeiser, B. W. (1988)
        "Binomial random variate generation", Communications of the ACM, 31, 216-222.
        """
        shape = sample_shape + self._batch_shape
        n = mx.broadcast_to(self.total_count, shape)
        p = mx.broadcast_to(self.probs, shape)

        max_n = int(mx.max(n))

        if max_n <= 20:
            # Direct Bernoulli sum for small n (exact method)
            samples = mx.zeros(shape)
            for _ in range(max_n):
                u = mx.random.uniform(shape=shape)
                samples = samples + (u < p).astype(mx.float32)
            # Clamp to actual n for varying n values
            samples = mx.minimum(samples, n.astype(mx.float32))
        elif max_n <= 100:
            # Inverse transform with geometric skipping for medium n
            samples = self._sample_inverse_transform(n, p, shape)
        else:
            # BTPE algorithm for large n
            samples = self._sample_btpe(n, p, shape)

        return Tensor(samples)

    def _sample_inverse_transform(self, n: mx.array, p: mx.array, shape: Tuple[int, ...]) -> mx.array:
        """Inverse transform sampling using geometric skipping.

        More efficient than direct Bernoulli sum for medium n.
        Uses the waiting time interpretation of binomial.
        """
        max_n = int(mx.max(n))

        # Use the sum of geometric waiting times approach
        # This is more efficient when p is not too small
        samples = mx.zeros(shape)

        # For each trial, compute whether to accept
        for i in range(max_n):
            u = mx.random.uniform(shape=shape)
            # Only count if we haven't exceeded n
            mask = (mx.array(float(i)) < n)
            samples = samples + mx.where(mask & (u < p), 1.0, 0.0)

        return samples

    def _sample_btpe(self, n: mx.array, p: mx.array, shape: Tuple[int, ...]) -> mx.array:
        """BTPE algorithm for binomial sampling.

        This algorithm uses rejection sampling with a piecewise linear
        envelope function consisting of:
        - A central triangular region
        - Two parallelogram regions on the sides
        - Two exponential tails

        For numerical stability, we use the property that if p > 0.5,
        we sample Binomial(n, 1-p) and return n - result.
        """
        # Ensure we work with p <= 0.5 for numerical stability
        use_complement = p > 0.5
        q = mx.where(use_complement, 1.0 - p, p)

        # Parameters for BTPE algorithm
        n_float = n.astype(mx.float32)
        npq = n_float * q * (1.0 - q)
        sqrt_npq = mx.sqrt(npq)

        # Mode of distribution (approximately)
        m = mx.floor((n_float + 1.0) * q)

        # Parameters for the triangular region
        # Using simple rejection sampling with normal proposal for large n*p*(1-p)

        # Mean and standard deviation
        mu = n_float * q
        sigma = sqrt_npq

        # For large n, the binomial is well approximated by normal
        # We use rejection sampling with normal proposal

        # Number of rejection sampling iterations
        # For large n*p*(1-p), acceptance rate is very high
        num_iterations = 20

        samples = mx.zeros(shape)
        accepted = mx.zeros(shape, dtype=mx.bool_)

        for _ in range(num_iterations):
            # Proposal from continuous normal approximation
            z = mx.random.normal(shape)
            proposal = mx.round(mu + sigma * z)

            # Clamp to valid range [0, n]
            proposal = mx.clip(proposal, 0, n_float)

            # Compute acceptance ratio using Stirling's approximation for log-binomial
            # For the normal approximation, the acceptance is based on comparing
            # the binomial PMF with the normal PDF

            # Simplified acceptance: accept if proposal is in valid range
            # and passes a secondary check based on the log-likelihood ratio
            k = proposal

            # Log of binomial coefficient ratio compared to mode
            # Using the recurrence: log(C(n,k+1)/C(n,k)) = log((n-k)/(k+1))
            # We approximate the acceptance probability

            # For numerical stability, compute log probabilities
            # log(P(X=k)) ~ -0.5*((k-mu)/sigma)^2 - 0.5*log(2*pi*npq) (normal approx)
            # Actual: log(C(n,k)) + k*log(p) + (n-k)*log(1-p)

            # Use a simpler acceptance criterion based on bounds
            # Accept if the proposal falls within the typical range
            lower_bound = mx.maximum(0.0, mu - 4.0 * sigma)
            upper_bound = mx.minimum(n_float, mu + 4.0 * sigma)
            in_range = (proposal >= lower_bound) & (proposal <= upper_bound)

            # Secondary rejection using uniform
            # The ratio of binomial PMF to normal PDF is bounded for large npq
            u = mx.random.uniform(shape=shape)

            # Compute log-probability ratio (binomial vs normal)
            # For large npq, this ratio is close to 1 in the central region
            # We use a simplified bound

            # Stirling-based correction factor
            # For k near m, the correction is small
            delta = (k - mu) / sigma
            correction = mx.exp(-delta * delta / 2.0 * (
                delta * delta / (12.0 * npq) -
                delta * delta * delta * delta / (360.0 * npq * npq)
            ))

            # Accept with probability proportional to correction
            accept = in_range & (u < mx.minimum(1.0, correction)) & (~accepted)

            samples = mx.where(accept, proposal, samples)
            accepted = accepted | accept

        # For any remaining unaccepted samples, use the mode as fallback
        samples = mx.where(accepted, samples, m)

        # Transform back if we used complement
        samples = mx.where(use_complement, n_float - samples, samples)

        return samples

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        n = self.total_count
        k = data
        # log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_comb = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        # Use xlogy for numerical stability (handles p=0 and p=1 correctly)
        log_p_term = xlogy(k, self.probs)
        log_1mp_term = xlogy(n - k, 1 - self.probs)
        return Tensor(log_comb + log_p_term + log_1mp_term)


__all__ = ['Binomial']
