"""LKJ Cholesky Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


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
        self.concentration = concentration._data if isinstance(concentration, Tensor) else mx.array(concentration)
        batch_shape = self.concentration.shape
        event_shape = (dim, dim)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        import numpy as np

        # Sample using onion method
        d = self.dim
        L = np.zeros(shape + (d, d))
        L[..., 0, 0] = 1.0

        for i in range(1, d):
            # Sample beta for partial correlations
            alpha = float(mx.max(self.concentration)) + (d - 1 - i) / 2
            beta_sample = np.random.beta(alpha, alpha, shape)
            # Sample uniformly on sphere
            z = np.random.normal(size=shape + (i,))
            z = z / np.linalg.norm(z, axis=-1, keepdims=True)
            L[..., i, :i] = z * np.sqrt(beta_sample)[..., None]
            L[..., i, i] = np.sqrt(1 - beta_sample)

        return Tensor(mx.array(L.astype(np.float32)))

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp

        d = self.dim
        eta = self.concentration

        # Log probability of LKJ
        diag = mx.diagonal(data, axis1=-2, axis2=-1)
        log_diag = mx.log(diag[..., 1:] + 1e-10)

        # Sum of (d - i) * log(L_ii) for i = 2, ..., d
        weights = mx.arange(d - 1, 0, -1)
        log_prob = mx.sum((2 * eta - 2 + weights) * log_diag, axis=-1)

        # Normalize
        log_norm = 0
        for k in range(1, d):
            log_norm += sp.betaln(eta + (d - 1 - k) / 2, eta + (d - 1 - k) / 2)
        log_norm = mx.array(log_norm)

        return Tensor(log_prob - log_norm)


__all__ = ['LKJCholesky']
