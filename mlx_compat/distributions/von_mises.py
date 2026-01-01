"""Von Mises Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints
from ..ops.special import i0, i1


class VonMises(Distribution):
    """Von Mises distribution (circular distribution on angles)."""

    arg_constraints = {'loc': constraints.real, 'concentration': constraints.nonnegative}
    support = constraints.interval(-math.pi, math.pi)
    has_rsample = True

    def __init__(
        self,
        loc: Union[Tensor, float],
        concentration: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)
        self.concentration = concentration._mlx_array if isinstance(concentration, Tensor) else mx.array(concentration)
        batch_shape = mx.broadcast_shapes(self.loc.shape, self.concentration.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        # Circular variance = 1 - I_1(kappa) / I_0(kappa)
        kappa = self.concentration
        i0_val = i0(kappa)
        i1_val = i1(kappa)
        return Tensor(1.0 - i1_val / i0_val)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Sample using rejection sampling with uniform proposal.

        Uses the algorithm from Best & Fisher (1979) for sampling from
        the von Mises distribution via rejection sampling.
        """
        shape = sample_shape + self._batch_shape

        # Broadcast parameters to full shape
        loc = mx.broadcast_to(self.loc, shape)
        kappa = mx.broadcast_to(self.concentration, shape)

        # Best-Fisher algorithm for von Mises sampling
        # tau = 1 + sqrt(1 + 4*kappa^2)
        tau = 1.0 + mx.sqrt(1.0 + 4.0 * kappa * kappa)
        # rho = (tau - sqrt(2*tau)) / (2*kappa)
        rho = (tau - mx.sqrt(2.0 * tau)) / (2.0 * kappa)
        # r = (1 + rho^2) / (2*rho)
        r = (1.0 + rho * rho) / (2.0 * rho)

        # Handle kappa = 0 case (uniform distribution)
        kappa_is_zero = kappa < 1e-10

        # Initialize samples
        samples = mx.zeros(shape)

        # Rejection sampling
        # We need to iterate until all samples are accepted
        # Since MLX doesn't have while loops, we'll do fixed iterations
        # with masking for already-accepted samples
        accepted = mx.zeros(shape, dtype=mx.bool_)
        max_iterations = 100

        for _ in range(max_iterations):
            # Generate uniform samples
            u1 = mx.random.uniform(shape=shape)
            u2 = mx.random.uniform(shape=shape)
            u3 = mx.random.uniform(shape=shape)

            # z = cos(pi * u1)
            z = mx.cos(math.pi * u1)

            # f = (1 + r*z) / (r + z)
            f = (1.0 + r * z) / (r + z)

            # c = kappa * (r - f)
            c = kappa * (r - f)

            # Accept if u2 < c * (2 - c) or log(c/u2) + 1 - c >= 0
            accept_cond1 = u2 < c * (2.0 - c)
            accept_cond2 = mx.log(c / u2) + 1.0 - c >= 0.0
            accept = (accept_cond1 | accept_cond2) & ~accepted

            # theta = sign(u3 - 0.5) * arccos(f)
            theta = mx.where(u3 > 0.5, mx.arccos(f), -mx.arccos(f))

            # Update samples where accepted
            samples = mx.where(accept, theta, samples)
            accepted = accepted | accept

            # Early exit if all accepted (can't actually break in MLX)
            if mx.all(accepted):
                break

        # For kappa = 0, just use uniform on [-pi, pi]
        uniform_samples = mx.random.uniform(shape=shape) * 2.0 * math.pi - math.pi
        samples = mx.where(kappa_is_zero, uniform_samples, samples)

        # Shift by location
        samples = samples + loc

        # Wrap to [-pi, pi]
        samples = mx.remainder(samples + math.pi, 2.0 * math.pi) - math.pi

        return Tensor(samples)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        log_i0_val = mx.log(i0(self.concentration))
        log_prob = self.concentration * mx.cos(data - self.loc) - math.log(2 * math.pi) - log_i0_val
        return Tensor(log_prob)


__all__ = ['VonMises']
