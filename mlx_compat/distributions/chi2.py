"""Chi-squared Distribution"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from .gamma import Gamma
from . import constraints


class Chi2(Gamma):
    """Chi-squared distribution (special case of Gamma)."""

    arg_constraints = {'df': constraints.positive}

    def __init__(self, df: Union[Tensor, float], validate_args: Optional[bool] = None):
        self.df = df._data if isinstance(df, Tensor) else mx.array(df)
        super().__init__(self.df / 2, mx.array(0.5), validate_args=validate_args)


__all__ = ['Chi2']
