"""Von Mises Distribution"""

import math
from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .distribution import Distribution
from . import constraints


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
        self.loc = loc._data if isinstance(loc, Tensor) else mx.array(loc)
        self.concentration = concentration._data if isinstance(concentration, Tensor) else mx.array(concentration)
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
        import numpy as np
        from scipy import special as sp
        # Circular variance = 1 - I_1(kappa) / I_0(kappa)
        kappa = np.array(self.concentration)
        i0 = sp.i0(kappa)
        i1 = sp.i1(kappa)
        return Tensor(mx.array((1 - i1 / i0).astype(np.float32)))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        shape = sample_shape + self._batch_shape
        import numpy as np
        # Use numpy's von Mises sampler
        samples = np.random.vonmises(np.array(self.loc), np.array(self.concentration), shape)
        return Tensor(mx.array(samples.astype(np.float32)))

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self.sample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        data = value._data if isinstance(value, Tensor) else value
        import numpy as np
        from scipy import special as sp
        log_i0 = np.log(sp.i0(np.array(self.concentration)))
        log_i0 = mx.array(log_i0.astype(np.float32))
        log_prob = self.concentration * mx.cos(data - self.loc) - math.log(2 * math.pi) - log_i0
        return Tensor(log_prob)


__all__ = ['VonMises']
