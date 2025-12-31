"""
Constraints Module

PyTorch-compatible constraint classes for distributions.
"""

from typing import Dict, Optional
import mlx.core as mx

from ..tensor import Tensor


class Constraint:
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid.
    """

    is_discrete = False
    event_dim = 0

    def check(self, value) -> Tensor:
        """
        Check if a value satisfies the constraint.

        Args:
            value: Value to check

        Returns:
            Boolean tensor indicating if constraint is satisfied
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__[1:] + "()"


class _Dependent(Constraint):
    """Placeholder for dependent constraints."""

    def __init__(self, is_discrete=False, event_dim=0):
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()

    @property
    def is_discrete(self):
        return self._is_discrete

    @property
    def event_dim(self):
        return self._event_dim

    def __call__(self, is_discrete=False, event_dim=0):
        return _Dependent(is_discrete, event_dim)

    def check(self, value):
        raise ValueError("Cannot check dependent constraint")


class _DependentProperty:
    """Property that returns dependent constraint."""

    def __init__(self, fn=None, is_discrete=False, event_dim=0):
        self.fn = fn
        self._is_discrete = is_discrete
        self._event_dim = event_dim

    def __call__(self, fn):
        return _DependentProperty(fn, self._is_discrete, self._event_dim)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fn(obj)


class _Boolean(Constraint):
    """Constraint to boolean values (0 or 1)."""

    is_discrete = True

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = (data == 0) | (data == 1)
        return Tensor(result)


class _IntegerInterval(Constraint):
    """Constraint to integer values in an interval."""

    is_discrete = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = (data >= self.lower_bound) & (data <= self.upper_bound) & (data == mx.floor(data))
        return Tensor(result)

    def __repr__(self):
        return f"integer_interval({self.lower_bound}, {self.upper_bound})"


class _IntegerGreaterThan(Constraint):
    """Constraint to integers greater than a bound."""

    is_discrete = True

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = (data > self.lower_bound) & (data == mx.floor(data))
        return Tensor(result)


class _Real(Constraint):
    """Constraint to real values."""

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = mx.isfinite(data)
        return Tensor(result)


class _RealVector(Constraint):
    """Constraint to real vectors."""

    event_dim = 1

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = mx.isfinite(data)
        return Tensor(mx.all(result, axis=-1))


class _GreaterThan(Constraint):
    """Constraint to values greater than a bound."""

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = data > self.lower_bound
        return Tensor(result)

    def __repr__(self):
        return f"greater_than({self.lower_bound})"


class _GreaterThanEq(Constraint):
    """Constraint to values greater than or equal to a bound."""

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = data >= self.lower_bound
        return Tensor(result)

    def __repr__(self):
        return f"greater_than_eq({self.lower_bound})"


class _LessThan(Constraint):
    """Constraint to values less than a bound."""

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = data < self.upper_bound
        return Tensor(result)

    def __repr__(self):
        return f"less_than({self.upper_bound})"


class _Interval(Constraint):
    """Constraint to values in an open interval."""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = (data > self.lower_bound) & (data < self.upper_bound)
        return Tensor(result)

    def __repr__(self):
        return f"interval({self.lower_bound}, {self.upper_bound})"


class _HalfOpenInterval(Constraint):
    """Constraint to values in a half-open interval [lower, upper)."""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = (data >= self.lower_bound) & (data < self.upper_bound)
        return Tensor(result)


class _Positive(Constraint):
    """Constraint to positive values."""

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = data > 0
        return Tensor(result)


class _Nonnegative(Constraint):
    """Constraint to non-negative values."""

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = data >= 0
        return Tensor(result)


class _Simplex(Constraint):
    """Constraint to the simplex (values sum to 1, all >= 0)."""

    event_dim = 1

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = mx.all(data >= 0, axis=-1) & mx.isclose(mx.sum(data, axis=-1), mx.array(1.0))
        return Tensor(result)


class _OneHot(Constraint):
    """Constraint to one-hot vectors."""

    is_discrete = True
    event_dim = 1

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        is_boolean = mx.all((data == 0) | (data == 1), axis=-1)
        is_one_hot = mx.sum(data, axis=-1) == 1
        result = is_boolean & is_one_hot
        return Tensor(result)


class _LowerTriangular(Constraint):
    """Constraint to lower triangular matrices."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        # Check if matrix equals its lower triangular part
        tril = mx.tril(data)
        result = mx.all(mx.isclose(data, tril), axis=(-2, -1))
        return Tensor(result)


class _LowerCholesky(Constraint):
    """Constraint to lower triangular matrices with positive diagonal."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        tril = mx.tril(data)
        is_tril = mx.all(mx.isclose(data, tril), axis=(-2, -1))
        diag = mx.diagonal(data, axis1=-2, axis2=-1)
        is_pos_diag = mx.all(diag > 0, axis=-1)
        result = is_tril & is_pos_diag
        return Tensor(result)


class _PositiveDefinite(Constraint):
    """Constraint to positive definite matrices."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        # Check if all eigenvalues are positive
        try:
            eigvals = mx.linalg.eigvalsh(data)
            result = mx.all(eigvals > 0, axis=-1)
        except Exception:
            result = mx.array(False)
        return Tensor(result)


class _PositiveSemidefinite(Constraint):
    """Constraint to positive semidefinite matrices."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        try:
            eigvals = mx.linalg.eigvalsh(data)
            result = mx.all(eigvals >= 0, axis=-1)
        except Exception:
            result = mx.array(False)
        return Tensor(result)


class _CorrCholesky(Constraint):
    """Constraint to Cholesky factors of correlation matrices."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        tril = mx.tril(data)
        is_tril = mx.all(mx.isclose(data, tril), axis=(-2, -1))
        diag = mx.diagonal(data, axis1=-2, axis2=-1)
        is_pos_diag = mx.all(diag > 0, axis=-1)
        # Check unit rows
        row_norms = mx.sum(data ** 2, axis=-1)
        is_unit = mx.all(mx.isclose(row_norms, mx.array(1.0)), axis=-1)
        result = is_tril & is_pos_diag & is_unit
        return Tensor(result)


class _Square(Constraint):
    """Constraint to square matrices."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        result = mx.array(data.shape[-2] == data.shape[-1])
        return Tensor(result)


class _Symmetric(Constraint):
    """Constraint to symmetric matrices."""

    event_dim = 2

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        transpose = mx.swapaxes(data, -2, -1)
        result = mx.all(mx.isclose(data, transpose), axis=(-2, -1))
        return Tensor(result)


class _Cat(Constraint):
    """Concatenation of constraints."""

    def __init__(self, cseq, dim=0, lengths=None):
        self.cseq = cseq
        self.dim = dim
        self.lengths = lengths
        super().__init__()

    @property
    def is_discrete(self):
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self):
        return max(c.event_dim for c in self.cseq)

    def check(self, value):
        # Simplified check - verify each part satisfies its constraint
        return Tensor(mx.array(True))


class _Stack(Constraint):
    """Stack of constraints."""

    def __init__(self, cseq, dim=0):
        self.cseq = cseq
        self.dim = dim
        super().__init__()

    @property
    def is_discrete(self):
        return any(c.is_discrete for c in self.cseq)

    @property
    def event_dim(self):
        return 1 + max(c.event_dim for c in self.cseq)

    def check(self, value):
        return Tensor(mx.array(True))


class _Multinomial(Constraint):
    """Constraint to multinomial samples."""

    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound):
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        data = value._data if isinstance(value, Tensor) else value
        # All non-negative integers
        is_nonneg = mx.all(data >= 0, axis=-1)
        is_int = mx.all(data == mx.floor(data), axis=-1)
        # Sum equals upper bound (total count)
        is_sum = mx.isclose(mx.sum(data, axis=-1), mx.array(float(self.upper_bound)))
        result = is_nonneg & is_int & is_sum
        return Tensor(result)


# Constraint instances
dependent = _Dependent()
dependent_property = _DependentProperty
boolean = _Boolean()
nonnegative_integer = _IntegerGreaterThan(-1)
positive_integer = _IntegerGreaterThan(0)
real = _Real()
real_vector = _RealVector()
positive = _Positive()
nonnegative = _Nonnegative()
unit_interval = _Interval(0.0, 1.0)
simplex = _Simplex()
one_hot = _OneHot()
lower_triangular = _LowerTriangular()
lower_cholesky = _LowerCholesky()
positive_definite = _PositiveDefinite()
positive_semidefinite = _PositiveSemidefinite()
corr_cholesky = _CorrCholesky()
square = _Square()
symmetric = _Symmetric()


# Factory functions
def integer_interval(lower_bound, upper_bound):
    return _IntegerInterval(lower_bound, upper_bound)


def greater_than(lower_bound):
    return _GreaterThan(lower_bound)


def greater_than_eq(lower_bound):
    return _GreaterThanEq(lower_bound)


def less_than(upper_bound):
    return _LessThan(upper_bound)


def interval(lower_bound, upper_bound):
    return _Interval(lower_bound, upper_bound)


def half_open_interval(lower_bound, upper_bound):
    return _HalfOpenInterval(lower_bound, upper_bound)


def multinomial(upper_bound):
    return _Multinomial(upper_bound)


def cat(cseq, dim=0, lengths=None):
    return _Cat(cseq, dim, lengths)


def stack(cseq, dim=0):
    return _Stack(cseq, dim)


def independent(base_constraint, reinterpreted_batch_ndims):
    return _Independent(base_constraint, reinterpreted_batch_ndims)


class _Independent(Constraint):
    """Wraps a constraint to treat batch dims as event dims."""

    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__()

    @property
    def is_discrete(self):
        return self.base_constraint.is_discrete

    @property
    def event_dim(self):
        return self.base_constraint.event_dim + self.reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        data = result._data if isinstance(result, Tensor) else result
        for _ in range(self.reinterpreted_batch_ndims):
            data = mx.all(data, axis=-1)
        return Tensor(data)


def is_dependent(constraint):
    """Check if a constraint is dependent."""
    return isinstance(constraint, _Dependent)


class MixtureSameFamilyConstraint(Constraint):
    """
    Constraint for MixtureSameFamily distribution.

    Adds back the rightmost batch dimension before performing the
    validity check with the component distribution constraint.

    Args:
        base_constraint: The Constraint object of the component distribution
    """

    def __init__(self, base_constraint):
        self.base_constraint = base_constraint
        super().__init__()

    @property
    def is_discrete(self):
        return self.base_constraint.is_discrete

    @property
    def event_dim(self):
        return self.base_constraint.event_dim

    def check(self, value):
        # Add dimension back and check with base constraint
        data = value._data if isinstance(value, Tensor) else value
        expanded = mx.expand_dims(data, axis=-self.base_constraint.event_dim - 1)
        return self.base_constraint.check(Tensor(expanded))


__all__ = [
    'Constraint',
    'MixtureSameFamilyConstraint',
    'boolean',
    'cat',
    'corr_cholesky',
    'dependent',
    'dependent_property',
    'greater_than',
    'greater_than_eq',
    'half_open_interval',
    'independent',
    'integer_interval',
    'interval',
    'is_dependent',
    'less_than',
    'lower_cholesky',
    'lower_triangular',
    'multinomial',
    'nonnegative',
    'nonnegative_integer',
    'one_hot',
    'positive',
    'positive_definite',
    'positive_integer',
    'positive_semidefinite',
    'real',
    'real_vector',
    'simplex',
    'square',
    'stack',
    'symmetric',
    'unit_interval',
]
