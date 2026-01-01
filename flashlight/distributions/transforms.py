"""
Transforms Module

PyTorch-compatible transforms for distributions.
"""

from typing import List, Optional, Sequence
import math
import mlx.core as mx

from ..tensor import Tensor
from . import constraints


class Transform:
    """
    Abstract class for invertible transforms with computable log det Jacobians.
    """

    bijective = True
    domain: constraints.Constraint = constraints.real
    codomain: constraints.Constraint = constraints.real

    def __init__(self, cache_size=0):
        self._cache_size = cache_size
        self._cached_x_y = [None, None]

    def __call__(self, x):
        return self._call(x)

    def _call(self, x):
        raise NotImplementedError

    def _inverse(self, y):
        raise NotImplementedError

    def inv(self, y):
        """Compute the inverse transform."""
        return self._inverse(y)

    def log_abs_det_jacobian(self, x, y):
        """Compute the log abs det Jacobian."""
        raise NotImplementedError

    @property
    def sign(self):
        """Returns sign of the Jacobian determinant."""
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ComposeTransform(Transform):
    """Compose multiple transforms."""

    def __init__(self, parts: List[Transform], cache_size=0):
        super().__init__(cache_size)
        self.parts = parts

    @property
    def domain(self):
        if not self.parts:
            return constraints.real
        return self.parts[0].domain

    @property
    def codomain(self):
        if not self.parts:
            return constraints.real
        return self.parts[-1].codomain

    @property
    def bijective(self):
        return all(p.bijective for p in self.parts)

    def _call(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def _inverse(self, y):
        for part in reversed(self.parts):
            y = part.inv(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        result = mx.zeros_like(data)
        for part in self.parts:
            y_tmp = part(x)
            result = result + part.log_abs_det_jacobian(x, y_tmp)._mlx_array
            x = y_tmp
        return Tensor(result)


class ExpTransform(Transform):
    """Transform via exp."""

    domain = constraints.real
    codomain = constraints.positive

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.exp(data))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(mx.log(data))

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(data)

    @property
    def sign(self):
        return 1


class PowerTransform(Transform):
    """Transform via x^exponent."""

    domain = constraints.positive
    codomain = constraints.positive

    def __init__(self, exponent, cache_size=0):
        super().__init__(cache_size)
        self.exponent = exponent

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.power(data, self.exponent))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(mx.power(data, 1.0 / self.exponent))

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.log(mx.abs(self.exponent * mx.power(data, self.exponent - 1))))

    @property
    def sign(self):
        return 1 if self.exponent > 0 else -1


class SigmoidTransform(Transform):
    """Transform via sigmoid."""

    domain = constraints.real
    codomain = constraints.unit_interval

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.sigmoid(data))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(mx.log(data) - mx.log(1 - data))

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        # log(sigmoid(x) * (1 - sigmoid(x))) = x - 2*softplus(x)
        return Tensor(data - 2 * mx.logaddexp(mx.array(0.0), data))

    @property
    def sign(self):
        return 1


class TanhTransform(Transform):
    """Transform via tanh."""

    domain = constraints.real
    codomain = constraints.interval(-1, 1)

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.tanh(data))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(0.5 * (mx.log1p(data) - mx.log1p(-data)))

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        # log(1 - tanh^2(x)) = 2 * (log(2) - x - softplus(-2x))
        return Tensor(2 * (math.log(2) - data - mx.logaddexp(mx.array(0.0), -2 * data)))

    @property
    def sign(self):
        return 1


class AbsTransform(Transform):
    """Transform via absolute value."""

    domain = constraints.real
    codomain = constraints.nonnegative
    bijective = False

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.abs(data))

    def _inverse(self, y):
        # Not uniquely invertible
        return y


class AffineTransform(Transform):
    """Transform via affine: y = loc + scale * x."""

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super().__init__(cache_size)
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else loc
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else scale
        self.event_dim = event_dim

    @property
    def domain(self):
        if mx.any(self.scale < 0):
            return constraints.real
        return constraints.real

    @property
    def codomain(self):
        return constraints.real

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(self.loc + self.scale * data)

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor((data - self.loc) / self.scale)

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        shape = data.shape
        scale = self.scale
        if isinstance(scale, (int, float)):
            result = mx.full(shape, math.log(abs(scale)))
        else:
            result = mx.broadcast_to(mx.log(mx.abs(scale)), shape)
        # Sum over event dims
        for _ in range(self.event_dim):
            result = mx.sum(result, axis=-1)
        return Tensor(result)

    @property
    def sign(self):
        return 1 if mx.all(self.scale > 0) else -1


class SoftmaxTransform(Transform):
    """Transform via softmax."""

    domain = constraints.real_vector
    codomain = constraints.simplex

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.softmax(data, axis=-1))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(mx.log(data))


class SoftplusTransform(Transform):
    """Transform via softplus."""

    domain = constraints.real
    codomain = constraints.positive

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.logaddexp(mx.array(0.0), data))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        return Tensor(data + mx.log(-mx.expm1(-data)))

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(-mx.logaddexp(mx.array(0.0), -data))


class LowerCholeskyTransform(Transform):
    """Transform to lower Cholesky factor."""

    domain = constraints.real
    codomain = constraints.lower_cholesky

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        # Apply exp to diagonal
        diag = mx.diagonal(data, axis1=-2, axis2=-1)
        result = mx.tril(data, -1) + mx.diag(mx.exp(diag))
        return Tensor(result)

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        diag = mx.diagonal(data, axis1=-2, axis2=-1)
        result = mx.tril(data, -1) + mx.diag(mx.log(diag))
        return Tensor(result)


class PositiveDefiniteTransform(Transform):
    """Transform to positive definite matrix."""

    domain = constraints.real
    codomain = constraints.positive_definite

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        # L L^T where L is lower Cholesky
        L = mx.tril(data)
        diag = mx.diagonal(L, axis1=-2, axis2=-1)
        L = mx.tril(L, -1) + mx.diag(mx.exp(diag))
        result = mx.matmul(L, mx.swapaxes(L, -2, -1))
        return Tensor(result)


class CorrCholeskyTransform(Transform):
    """Transform to correlation Cholesky factor."""

    domain = constraints.real
    codomain = constraints.corr_cholesky

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        # Construct correlation Cholesky from unconstrained
        z = mx.tanh(data)
        L = mx.tril(mx.ones_like(data))
        # Simplified: just normalize rows
        norms = mx.sqrt(mx.sum(L ** 2, axis=-1, keepdims=True))
        result = L / norms
        return Tensor(result)


class StickBreakingTransform(Transform):
    """Stick breaking transform for simplex."""

    domain = constraints.real
    codomain = constraints.simplex

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        z = mx.sigmoid(data)
        # Stick breaking: x_i = z_i * prod(1 - z_j for j < i)
        z1m_cumprod = mx.cumprod(1 - z, axis=-1)
        # Shift and pad
        z1m_cumprod_shifted = mx.concatenate([mx.ones((*data.shape[:-1], 1)), z1m_cumprod[..., :-1]], axis=-1)
        p = z * z1m_cumprod_shifted
        # Last element
        last = z1m_cumprod[..., -1:]
        result = mx.concatenate([p, last], axis=-1)
        return Tensor(result)


class CatTransform(Transform):
    """Concatenate transforms along a dimension."""

    def __init__(self, tseq: List[Transform], dim=0, lengths=None, cache_size=0):
        super().__init__(cache_size)
        self.tseq = tseq
        self.dim = dim
        self.lengths = lengths

    def _call(self, x):
        # Split and apply
        data = x._mlx_array if isinstance(x, Tensor) else x
        if self.lengths is None:
            n = len(self.tseq)
            split_size = data.shape[self.dim] // n
            lengths = [split_size] * n
        else:
            lengths = self.lengths

        # Split along dim
        splits = []
        start = 0
        for length in lengths:
            slices = [slice(None)] * len(data.shape)
            slices[self.dim] = slice(start, start + length)
            splits.append(data[tuple(slices)])
            start += length

        # Apply transforms
        results = [t(Tensor(s))._mlx_array for t, s in zip(self.tseq, splits)]
        result = mx.concatenate(results, axis=self.dim)
        return Tensor(result)


class StackTransform(Transform):
    """Stack transforms along a dimension."""

    def __init__(self, tseq: List[Transform], dim=0, cache_size=0):
        super().__init__(cache_size)
        self.tseq = tseq
        self.dim = dim

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        # Split along stacked dim
        n = len(self.tseq)
        splits = [data[i] for i in range(n)]
        results = [t(Tensor(s))._mlx_array for t, s in zip(self.tseq, splits)]
        result = mx.stack(results, axis=self.dim)
        return Tensor(result)


class IndependentTransform(Transform):
    """Treat batch dims as event dims."""

    def __init__(self, base_transform: Transform, reinterpreted_batch_ndims: int, cache_size=0):
        super().__init__(cache_size)
        self.base_transform = base_transform
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def _call(self, x):
        return self.base_transform(x)

    def _inverse(self, y):
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        result = self.base_transform.log_abs_det_jacobian(x, y)
        data = result._mlx_array if isinstance(result, Tensor) else result
        for _ in range(self.reinterpreted_batch_ndims):
            data = mx.sum(data, axis=-1)
        return Tensor(data)


class ReshapeTransform(Transform):
    """Reshape transform."""

    def __init__(self, in_shape: Sequence[int], out_shape: Sequence[int], cache_size=0):
        super().__init__(cache_size)
        self.in_shape = in_shape
        self.out_shape = out_shape

    def _call(self, x):
        data = x._mlx_array if isinstance(x, Tensor) else x
        batch_shape = data.shape[:-len(self.in_shape)]
        return Tensor(mx.reshape(data, batch_shape + tuple(self.out_shape)))

    def _inverse(self, y):
        data = y._mlx_array if isinstance(y, Tensor) else y
        batch_shape = data.shape[:-len(self.out_shape)]
        return Tensor(mx.reshape(data, batch_shape + tuple(self.in_shape)))

    def log_abs_det_jacobian(self, x, y):
        data = x._mlx_array if isinstance(x, Tensor) else x
        return Tensor(mx.zeros(data.shape[:-len(self.in_shape)]))


class CumulativeDistributionTransform(Transform):
    """Transform via CDF of a distribution."""

    def __init__(self, distribution, cache_size=0):
        super().__init__(cache_size)
        self.distribution = distribution

    @property
    def domain(self):
        return self.distribution.support

    @property
    def codomain(self):
        return constraints.unit_interval

    def _call(self, x):
        return self.distribution.cdf(x)

    def _inverse(self, y):
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.distribution.log_prob(x)


# Identity transform (empty ComposeTransform like PyTorch)
identity_transform = ComposeTransform([])

__all__ = [
    'Transform',
    'ComposeTransform',
    'ExpTransform',
    'PowerTransform',
    'SigmoidTransform',
    'TanhTransform',
    'AbsTransform',
    'AffineTransform',
    'SoftmaxTransform',
    'SoftplusTransform',
    'LowerCholeskyTransform',
    'PositiveDefiniteTransform',
    'CorrCholeskyTransform',
    'StickBreakingTransform',
    'CatTransform',
    'StackTransform',
    'IndependentTransform',
    'ReshapeTransform',
    'CumulativeDistributionTransform',
    'identity_transform',
]
