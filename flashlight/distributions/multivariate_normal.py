"""Multivariate Normal Distribution"""

import math
from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from . import constraints
from .distribution import Distribution


class MultivariateNormal(Distribution):
    """Multivariate Normal distribution."""

    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc: Union[Tensor, mx.array],
        covariance_matrix: Optional[Union[Tensor, mx.array]] = None,
        precision_matrix: Optional[Union[Tensor, mx.array]] = None,
        scale_tril: Optional[Union[Tensor, mx.array]] = None,
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)

        if sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril]) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified"
            )

        if covariance_matrix is not None:
            self.covariance_matrix = (
                covariance_matrix._mlx_array
                if isinstance(covariance_matrix, Tensor)
                else mx.array(covariance_matrix)
            )
            # Use CPU stream for cholesky (required by MLX)
            self._scale_tril = mx.linalg.cholesky(self.covariance_matrix, stream=mx.cpu)
            mx.eval(self._scale_tril)
        elif scale_tril is not None:
            self._scale_tril = (
                scale_tril._mlx_array if isinstance(scale_tril, Tensor) else mx.array(scale_tril)
            )
            self.covariance_matrix = mx.matmul(
                self._scale_tril, mx.swapaxes(self._scale_tril, -2, -1)
            )
        else:
            self._precision_matrix = (
                precision_matrix._mlx_array
                if isinstance(precision_matrix, Tensor)
                else mx.array(precision_matrix)
            )
            # Invert precision to get covariance (use CPU stream)
            self.covariance_matrix = mx.linalg.inv(self._precision_matrix, stream=mx.cpu)
            mx.eval(self.covariance_matrix)
            self._scale_tril = mx.linalg.cholesky(self.covariance_matrix, stream=mx.cpu)
            mx.eval(self._scale_tril)

        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def scale_tril(self) -> Tensor:
        return Tensor(self._scale_tril)

    @property
    def precision_matrix(self) -> Tensor:
        # inv also requires CPU stream
        result = mx.linalg.inv(self.covariance_matrix, stream=mx.cpu)
        mx.eval(result)
        return Tensor(result)

    @property
    def mean(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        return Tensor(mx.diagonal(self.covariance_matrix, axis1=-2, axis2=-1))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape + self._event_shape
        eps = mx.random.normal(shape)
        # Transform: loc + L @ eps
        result = self.loc + mx.squeeze(mx.matmul(self._scale_tril, mx.expand_dims(eps, -1)), -1)
        return Tensor(result)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else value
        diff = data - self.loc
        k = self._event_shape[0]

        # Solve L @ y = diff for y, then y.T @ y = diff.T @ cov^-1 @ diff
        # MLX uses upper=False for lower triangular (opposite of PyTorch's lower=True)
        # Also needs CPU stream like cholesky
        y = mx.linalg.solve_triangular(
            self._scale_tril, mx.expand_dims(diff, -1), upper=False, stream=mx.cpu
        )
        mx.eval(y)
        M = mx.sum(y**2, axis=(-2, -1))

        # Log det of covariance = 2 * sum(log(diag(L)))
        log_det = 2 * mx.sum(mx.log(mx.diagonal(self._scale_tril, axis1=-2, axis2=-1)), axis=-1)

        log_prob = -0.5 * (k * math.log(2 * math.pi) + log_det + M)
        return Tensor(log_prob)

    def entropy(self) -> Tensor:
        k = self._event_shape[0]
        log_det = 2 * mx.sum(mx.log(mx.diagonal(self._scale_tril, axis1=-2, axis2=-1)), axis=-1)
        return Tensor(0.5 * (k * (1 + math.log(2 * math.pi)) + log_det))


class LowRankMultivariateNormal(Distribution):
    """
    Multivariate Normal with low-rank plus diagonal covariance.

    Covariance = cov_factor @ cov_factor.T + diag(cov_diag)
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_factor": constraints.real,
        "cov_diag": constraints.positive,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc: Union[Tensor, mx.array],
        cov_factor: Union[Tensor, mx.array],
        cov_diag: Union[Tensor, mx.array],
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)
        self.cov_factor = (
            cov_factor._mlx_array if isinstance(cov_factor, Tensor) else mx.array(cov_factor)
        )
        self.cov_diag = cov_diag._mlx_array if isinstance(cov_diag, Tensor) else mx.array(cov_diag)

        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def covariance_matrix(self) -> Tensor:
        return Tensor(
            mx.matmul(self.cov_factor, mx.swapaxes(self.cov_factor, -2, -1))
            + mx.diag(self.cov_diag)
        )

    @property
    def mean(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        return Tensor(mx.sum(self.cov_factor**2, axis=-1) + self.cov_diag)

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape + self._event_shape
        eps = mx.random.normal(shape)
        # Low-rank + diagonal structure
        rank = self.cov_factor.shape[-1]
        eps_factor = mx.random.normal(sample_shape + self._batch_shape + (rank,))
        result = (
            self.loc
            + mx.squeeze(mx.matmul(self.cov_factor, mx.expand_dims(eps_factor, -1)), -1)
            + mx.sqrt(self.cov_diag) * eps
        )
        return Tensor(result)

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        # Simplified - compute via full covariance
        cov = self.covariance_matrix._mlx_array
        return MultivariateNormal(Tensor(self.loc), covariance_matrix=Tensor(cov)).log_prob(value)


__all__ = ["MultivariateNormal", "LowRankMultivariateNormal"]
