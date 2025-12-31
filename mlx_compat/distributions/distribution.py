"""
Base Distribution Class

PyTorch-compatible base distribution class.
"""

from typing import Dict, Optional, Tuple
import mlx.core as mx

from ..tensor import Tensor
from . import constraints


class Distribution:
    """
    Base class for probability distributions.

    Subclasses should implement:
    - sample()
    - log_prob()
    - And optionally: rsample(), cdf(), icdf(), entropy(), mean, variance, etc.
    """

    has_rsample = False
    has_enumerate_support = False
    arg_constraints: Dict[str, constraints.Constraint] = {}
    support: constraints.Constraint = None

    def __init__(
        self,
        batch_shape: Tuple[int, ...] = (),
        event_shape: Tuple[int, ...] = (),
        validate_args: Optional[bool] = None,
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._validate_args = validate_args

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of batch dimensions."""
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of a single sample (event)."""
        return self._event_shape

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        raise NotImplementedError

    @property
    def mode(self) -> Tensor:
        """Mode of the distribution."""
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        """Variance of the distribution."""
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        """Standard deviation of the distribution."""
        var = self.variance
        return Tensor(mx.sqrt(var._data))

    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Generate a sample."""
        raise NotImplementedError

    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """
        Generate a reparameterized sample.

        Default implementation raises an error.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement rsample")

    def log_prob(self, value: Tensor) -> Tensor:
        """Compute log probability of a value."""
        raise NotImplementedError

    def cdf(self, value: Tensor) -> Tensor:
        """Compute cumulative distribution function."""
        raise NotImplementedError

    def icdf(self, value: Tensor) -> Tensor:
        """Compute inverse cumulative distribution function (quantile)."""
        raise NotImplementedError

    def entropy(self) -> Tensor:
        """Compute entropy of the distribution."""
        raise NotImplementedError

    def enumerate_support(self, expand: bool = True) -> Tensor:
        """Return tensor of all values supported by a discrete distribution."""
        raise NotImplementedError

    def expand(self, batch_shape: Tuple[int, ...], _instance=None):
        """
        Returns a new distribution with batch dimensions expanded.
        """
        raise NotImplementedError

    def _extended_shape(self, sample_shape: Tuple[int, ...] = ()) -> Tuple[int, ...]:
        """Return the shape of a sample including sample, batch, and event dims."""
        return sample_shape + self._batch_shape + self._event_shape

    def _validate_sample(self, value: Tensor):
        """Validate that a value is in the support of the distribution."""
        if self._validate_args:
            if self.support is not None:
                valid = self.support.check(value)
                if not mx.all(valid._data):
                    raise ValueError("Value is not in the support of the distribution")

    def __repr__(self):
        param_names = [k for k in self.arg_constraints.keys() if hasattr(self, k)]
        args_string = ", ".join(
            [f"{p}: {getattr(self, p)}" for p in param_names[:2]]
        )
        return f"{self.__class__.__name__}({args_string})"


__all__ = ['Distribution']
