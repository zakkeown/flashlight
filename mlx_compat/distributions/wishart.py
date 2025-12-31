"""Wishart Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class Wishart(Distribution):
    """Wishart distribution over positive definite matrices."""

    arg_constraints = {'df': constraints.positive,
                      'covariance_matrix': constraints.positive_definite,
                      'precision_matrix': constraints.positive_definite,
                      'scale_tril': constraints.lower_cholesky}
    support = constraints.positive_definite
    has_rsample = True

    def __init__(
        self,
        df: Union[Tensor, float],
        covariance_matrix: Optional[Union[Tensor, mx.array]] = None,
        precision_matrix: Optional[Union[Tensor, mx.array]] = None,
        scale_tril: Optional[Union[Tensor, mx.array]] = None,
        validate_args: Optional[bool] = None,
    ):
        self.df = df._data if isinstance(df, Tensor) else mx.array(df)

        if sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril]) != 1:
            raise ValueError("Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified")

        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix._data if isinstance(covariance_matrix, Tensor) else mx.array(covariance_matrix)
            self._scale_tril = mx.linalg.cholesky(self.covariance_matrix)
        elif scale_tril is not None:
            self._scale_tril = scale_tril._data if isinstance(scale_tril, Tensor) else mx.array(scale_tril)
            self.covariance_matrix = mx.matmul(self._scale_tril, mx.swapaxes(self._scale_tril, -2, -1))
        else:
            self._precision_matrix = precision_matrix._data if isinstance(precision_matrix, Tensor) else mx.array(precision_matrix)
            self.covariance_matrix = mx.linalg.inv(self._precision_matrix)
            self._scale_tril = mx.linalg.cholesky(self.covariance_matrix)

        p = self._scale_tril.shape[-1]
        batch_shape = mx.broadcast_shapes(self.df.shape, self._scale_tril.shape[:-2])
        event_shape = (p, p)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.df[..., None, None] * self.covariance_matrix)

    @property
    def mode(self) -> Tensor:
        p = self._event_shape[0]
        return Tensor((self.df[..., None, None] - p - 1) * self.covariance_matrix)

    @property
    def variance(self) -> Tensor:
        p = self._event_shape[0]
        var = 2 * self.df[..., None, None] * (self.covariance_matrix ** 2 +
              mx.einsum('...ij,...ji->...', self.covariance_matrix, self.covariance_matrix)[..., None, None])
        return Tensor(var)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Bartlett decomposition
        p = self._event_shape[0]
        shape = sample_shape + self._batch_shape

        import numpy as np
        # Sample A from Bartlett decomposition
        A = np.zeros(shape + (p, p))
        for i in range(p):
            A[..., i, i] = np.sqrt(np.random.chisquare(float(mx.max(self.df)) - i, shape))
        for i in range(p):
            for j in range(i):
                A[..., i, j] = np.random.normal(0, 1, shape)
        A = mx.array(A.astype(np.float32))

        # W = L @ A @ A.T @ L.T
        LA = mx.matmul(self._scale_tril, A)
        return Tensor(mx.matmul(LA, mx.swapaxes(LA, -2, -1)))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp

        p = self._event_shape[0]
        df = self.df

        # Log determinant of value
        log_det_value = mx.sum(mx.log(mx.diagonal(mx.linalg.cholesky(data), axis1=-2, axis2=-1)), axis=-1) * 2
        # Log determinant of scale
        log_det_scale = mx.sum(mx.log(mx.diagonal(self._scale_tril, axis1=-2, axis2=-1)), axis=-1) * 2

        # Multivariate log gamma
        log_mvgamma = sp.multigammaln(np.array(df / 2), p)
        log_mvgamma = mx.array(log_mvgamma.astype(np.float32))

        # Trace term
        scale_inv = mx.linalg.inv(self.covariance_matrix)
        trace_term = mx.trace(mx.matmul(scale_inv, data))

        log_prob = ((df - p - 1) / 2 * log_det_value -
                   trace_term / 2 -
                   df * p / 2 * np.log(2) -
                   df / 2 * log_det_scale -
                   log_mvgamma)
        return Tensor(log_prob)


__all__ = ['Wishart']
