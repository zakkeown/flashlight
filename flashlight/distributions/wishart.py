"""Wishart Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints
from ..ops.special import lgamma, multigammaln


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
        self.df = df._mlx_array if isinstance(df, Tensor) else mx.array(df)

        if sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril]) != 1:
            raise ValueError("Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified")

        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix._mlx_array if isinstance(covariance_matrix, Tensor) else mx.array(covariance_matrix)
            # Use CPU stream for cholesky (required by MLX)
            self._scale_tril = mx.linalg.cholesky(self.covariance_matrix, stream=mx.cpu)
            mx.eval(self._scale_tril)
        elif scale_tril is not None:
            self._scale_tril = scale_tril._mlx_array if isinstance(scale_tril, Tensor) else mx.array(scale_tril)
            self.covariance_matrix = mx.matmul(self._scale_tril, mx.swapaxes(self._scale_tril, -2, -1))
        else:
            self._precision_matrix = precision_matrix._mlx_array if isinstance(precision_matrix, Tensor) else mx.array(precision_matrix)
            # Use CPU stream for inv and cholesky (required by MLX)
            self.covariance_matrix = mx.linalg.inv(self._precision_matrix, stream=mx.cpu)
            mx.eval(self.covariance_matrix)
            self._scale_tril = mx.linalg.cholesky(self.covariance_matrix, stream=mx.cpu)
            mx.eval(self._scale_tril)

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
        """Sample using Bartlett decomposition.

        The Wishart distribution W(df, Sigma) can be sampled as:
        W = L @ A @ A.T @ L.T

        where L is the Cholesky factor of Sigma (scale_tril), and A is a
        lower triangular matrix with:
        - A[i,i] ~ sqrt(Chi2(df - i)) for i = 0, ..., p-1
        - A[i,j] ~ N(0, 1) for i > j

        Chi2(k) = Gamma(k/2, 2), which we sample as 2 * Gamma(k/2, 1).
        """
        p = self._event_shape[0]
        shape = sample_shape + self._batch_shape

        # Build the lower triangular matrix A using Bartlett decomposition
        # We need to construct it element by element

        # Create storage for A
        full_shape = shape + (p, p)

        # Sample diagonal elements: A[i,i] ~ sqrt(Chi2(df - i))
        # Chi2(k) = Gamma(k/2, 2)
        # We sample sqrt(Chi2(df - i)) = sqrt(2 * Gamma((df - i)/2, 1))

        # Sample all diagonal and lower triangular elements
        # Diagonal: sqrt of chi-squared with decreasing degrees of freedom
        # Off-diagonal: standard normal

        # Initialize A as zeros
        A = mx.zeros(full_shape)

        # Broadcast df to sample shape for proper sampling
        df_broadcast = mx.broadcast_to(self.df, shape)

        # Fill diagonal elements
        for i in range(p):
            # Sample chi-squared with df - i degrees of freedom
            # Chi2(k) = Gamma(k/2, scale=2) = 2 * Gamma(k/2, scale=1)
            chi2_df = df_broadcast - float(i)
            chi2_samples = mx.random.gamma(chi2_df / 2, shape) * 2
            sqrt_chi2 = mx.sqrt(chi2_samples)

            # Set diagonal element A[..., i, i]
            # We need to update the diagonal elements
            indices = [slice(None)] * len(full_shape)
            indices[-2] = i
            indices[-1] = i
            A = A.at[tuple(indices)].add(sqrt_chi2)

        # Fill lower triangular elements with standard normal
        for i in range(p):
            for j in range(i):
                normal_samples = mx.random.normal(shape)
                indices = [slice(None)] * len(full_shape)
                indices[-2] = i
                indices[-1] = j
                A = A.at[tuple(indices)].add(normal_samples)

        # W = L @ A @ A.T @ L.T
        # First compute L @ A
        LA = mx.matmul(self._scale_tril, A)

        # Then compute LA @ LA.T
        W = mx.matmul(LA, mx.swapaxes(LA, -2, -1))

        return Tensor(W)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value

        p = self._event_shape[0]
        df = self.df

        # Log determinant of value (use CPU stream for cholesky)
        chol_data = mx.linalg.cholesky(data, stream=mx.cpu)
        mx.eval(chol_data)
        log_det_value = mx.sum(mx.log(mx.diagonal(chol_data, axis1=-2, axis2=-1)), axis=-1) * 2
        # Log determinant of scale
        log_det_scale = mx.sum(mx.log(mx.diagonal(self._scale_tril, axis1=-2, axis2=-1)), axis=-1) * 2

        # Multivariate log gamma using pure MLX
        log_mvgamma = multigammaln(df / 2, p)

        # Trace term (use CPU stream for inv)
        scale_inv = mx.linalg.inv(self.covariance_matrix, stream=mx.cpu)
        mx.eval(scale_inv)
        trace_term = mx.trace(mx.matmul(scale_inv, data))

        log_prob = ((df - p - 1) / 2 * log_det_value -
                   trace_term / 2 -
                   df * p / 2 * math.log(2) -
                   df / 2 * log_det_scale -
                   log_mvgamma)
        return Tensor(log_prob)


__all__ = ['Wishart']
