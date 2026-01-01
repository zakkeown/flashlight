"""LKJ Cholesky Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints
from ..ops.special import betaln


class LKJCholesky(Distribution):
    """LKJ distribution for Cholesky factors of correlation matrices."""

    arg_constraints = {'dim': constraints.positive_integer, 'concentration': constraints.positive}
    support = constraints.corr_cholesky

    def __init__(
        self,
        dim: int,
        concentration: Union[Tensor, float] = 1.0,
        validate_args: Optional[bool] = None,
    ):
        self.dim = dim
        self.concentration = concentration._mlx_array if isinstance(concentration, Tensor) else mx.array(concentration)
        batch_shape = self.concentration.shape
        event_shape = (dim, dim)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape

        # Sample using onion method
        d = self.dim
        L = mx.zeros(shape + (d, d))
        L = L.at[..., 0, 0].add(1.0)

        for i in range(1, d):
            # Sample beta for partial correlations
            alpha = float(mx.max(self.concentration)) + (d - 1 - i) / 2
            # Use MLX random for beta sampling
            beta_sample = mx.random.beta(mx.array(alpha), mx.array(alpha), shape)
            # Sample uniformly on sphere using normal distribution
            z = mx.random.normal(shape + (i,))
            z = z / mx.sqrt(mx.sum(z * z, axis=-1, keepdims=True) + 1e-10)
            L = L.at[..., i, :i].add(z * mx.sqrt(beta_sample)[..., None])
            L = L.at[..., i, i].add(mx.sqrt(1 - beta_sample))

        return Tensor(L)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value

        d = self.dim
        eta = self.concentration

        # Log probability of LKJ
        diag = mx.diagonal(data, axis1=-2, axis2=-1)
        log_diag = mx.log(diag[..., 1:] + 1e-10)

        # Sum of (d - i) * log(L_ii) for i = 2, ..., d
        weights = mx.arange(d - 1, 0, -1)
        log_prob = mx.sum((2 * eta - 2 + weights) * log_diag, axis=-1)

        # Normalize
        log_norm = mx.array(0.0)
        for k in range(1, d):
            log_norm = log_norm + betaln(eta + (d - 1 - k) / 2, eta + (d - 1 - k) / 2)

        return Tensor(log_prob - log_norm)


__all__ = ['LKJCholesky']
