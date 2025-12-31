"""Student's t Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class StudentT(Distribution):
    """Student's t distribution."""

    arg_constraints = {'df': constraints.positive, 'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        df: Union[Tensor, float],
        loc: Union[Tensor, float] = 0.0,
        scale: Union[Tensor, float] = 1.0,
        validate_args: Optional[bool] = None,
    ):
        self.df = df._data if isinstance(df, Tensor) else mx.array(df)
        self.loc = loc._data if isinstance(loc, Tensor) else mx.array(loc)
        self.scale = scale._data if isinstance(scale, Tensor) else mx.array(scale)
        batch_shape = mx.broadcast_shapes(self.df.shape, mx.broadcast_shapes(self.loc.shape, self.scale.shape))
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(mx.where(self.df > 1, self.loc, mx.array(float('nan'))))

    @property
    def mode(self) -> Tensor:
        return Tensor(self.loc)

    @property
    def variance(self) -> Tensor:
        return Tensor(mx.where(self.df > 2, self.scale ** 2 * self.df / (self.df - 2),
                              mx.where(self.df > 1, mx.array(float('inf')), mx.array(float('nan')))))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # t = Z / sqrt(V/df) where Z ~ N(0,1), V ~ Chi2(df)
        z = mx.random.normal(shape)
        chi2 = mx.random.gamma(self.df / 2, shape) * 2
        return Tensor(self.loc + self.scale * z / mx.sqrt(chi2 / self.df))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        y = (data - self.loc) / self.scale
        log_unnorm = -0.5 * (self.df + 1) * mx.log1p(y ** 2 / self.df)
        log_gamma_ratio = sp.gammaln(np.array((self.df + 1) / 2)) - sp.gammaln(np.array(self.df / 2))
        log_gamma_ratio = mx.array(log_gamma_ratio.astype(np.float32))
        log_norm = log_gamma_ratio - 0.5 * mx.log(self.df * math.pi) - mx.log(self.scale)
        return Tensor(log_unnorm + log_norm)

    def entropy(self) -> Tensor:
        import numpy as np
        from scipy import special as sp
        log_gamma_half = sp.gammaln(np.array(self.df / 2))
        log_gamma_half = mx.array(log_gamma_half.astype(np.float32))
        log_gamma_half_p1 = sp.gammaln(np.array((self.df + 1) / 2))
        log_gamma_half_p1 = mx.array(log_gamma_half_p1.astype(np.float32))
        digamma_half_p1 = sp.digamma(np.array((self.df + 1) / 2))
        digamma_half_p1 = mx.array(digamma_half_p1.astype(np.float32))
        digamma_half = sp.digamma(np.array(self.df / 2))
        digamma_half = mx.array(digamma_half.astype(np.float32))
        return Tensor((self.df + 1) / 2 * (digamma_half_p1 - digamma_half) +
                     mx.log(self.scale) + log_gamma_half - log_gamma_half_p1 + 0.5 * mx.log(self.df) + 0.5 * math.log(math.pi))


__all__ = ['StudentT']
