"""Student's t Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..ops.special import lgamma, digamma
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
        self.df = df._mlx_array if isinstance(df, Tensor) else mx.array(df)
        self.loc = loc._mlx_array if isinstance(loc, Tensor) else mx.array(loc)
        self.scale = scale._mlx_array if isinstance(scale, Tensor) else mx.array(scale)
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
        # Chi2(df) = Gamma(shape=df/2, scale=2) = Gamma(shape=df/2) * 2
        z = mx.random.normal(shape)
        chi2 = mx.random.gamma(self.df / 2, shape) * 2
        return Tensor(self.loc + self.scale * z / mx.sqrt(chi2 / self.df))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._mlx_array if isinstance(value, Tensor) else mx.array(value)
        y = (data - self.loc) / self.scale
        log_unnorm = -0.5 * (self.df + 1) * mx.log1p(y ** 2 / self.df)
        log_gamma_ratio = lgamma((self.df + 1) / 2) - lgamma(self.df / 2)
        log_norm = log_gamma_ratio - 0.5 * mx.log(self.df * math.pi) - mx.log(self.scale)
        return Tensor(log_unnorm + log_norm)

    def entropy(self) -> Tensor:
        log_gamma_half = lgamma(self.df / 2)
        log_gamma_half_p1 = lgamma((self.df + 1) / 2)
        digamma_half_p1 = digamma((self.df + 1) / 2)
        digamma_half = digamma(self.df / 2)
        return Tensor((self.df + 1) / 2 * (digamma_half_p1 - digamma_half) +
                     mx.log(self.scale) + log_gamma_half - log_gamma_half_p1 + 0.5 * mx.log(self.df) + 0.5 * math.log(math.pi))


__all__ = ['StudentT']
