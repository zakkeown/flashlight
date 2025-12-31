"""Fisher-Snedecor (F) Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


class FisherSnedecor(Distribution):
    """Fisher-Snedecor (F) distribution."""

    arg_constraints = {'df1': constraints.positive, 'df2': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        df1: Union[Tensor, float],
        df2: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ):
        self.df1 = df1._data if isinstance(df1, Tensor) else mx.array(df1)
        self.df2 = df2._data if isinstance(df2, Tensor) else mx.array(df2)
        batch_shape = mx.broadcast_shapes(self.df1.shape, self.df2.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        return Tensor(mx.where(self.df2 > 2, self.df2 / (self.df2 - 2), mx.array(float('inf'))))

    @property
    def mode(self) -> Tensor:
        return Tensor(mx.where(self.df1 > 2,
                              (self.df1 - 2) / self.df1 * self.df2 / (self.df2 + 2),
                              mx.array(0.0)))

    @property
    def variance(self) -> Tensor:
        return Tensor(mx.where(self.df2 > 4,
                              2 * self.df2 ** 2 * (self.df1 + self.df2 - 2) /
                              (self.df1 * (self.df2 - 2) ** 2 * (self.df2 - 4)),
                              mx.array(float('inf'))))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        # F = (X1/df1) / (X2/df2) where X1 ~ Chi2(df1), X2 ~ Chi2(df2)
        x1 = mx.random.gamma(self.df1 / 2, shape) * 2
        x2 = mx.random.gamma(self.df2 / 2, shape) * 2
        return Tensor((x1 / self.df1) / (x2 / self.df2))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        d1, d2 = self.df1, self.df2
        log_beta = sp.betaln(np.array(d1/2), np.array(d2/2))
        log_beta = mx.array(log_beta.astype(np.float32))
        log_prob = (d1/2 * mx.log(d1/d2) + (d1/2 - 1) * mx.log(data) -
                   (d1 + d2)/2 * mx.log(1 + d1/d2 * data) - log_beta)
        return Tensor(log_prob)


__all__ = ['FisherSnedecor']
