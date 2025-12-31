"""Constraint Registry"""

from typing import Callable, Dict, Type
import mlx.core as mx

from . import constraints
from .transforms import (
    Transform, ExpTransform, SigmoidTransform, AffineTransform,
    SoftmaxTransform, LowerCholeskyTransform, StickBreakingTransform,
    ComposeTransform, AbsTransform, PowerTransform
)


# Registry for constraint -> transform mappings
_BIJECT_TO_REGISTRY: Dict[type, Callable] = {}
_TRANSFORM_TO_REGISTRY: Dict[type, Callable] = {}


def biject_to(constraint: constraints.Constraint) -> Transform:
    """
    Get a bijective transform that maps the reals to the constraint's support.

    Args:
        constraint: The target constraint

    Returns:
        A Transform that maps reals to the constraint's support
    """
    constraint_type = type(constraint)
    if constraint_type in _BIJECT_TO_REGISTRY:
        return _BIJECT_TO_REGISTRY[constraint_type](constraint)

    # Try parent classes
    for ctype, fn in _BIJECT_TO_REGISTRY.items():
        if isinstance(constraint, ctype):
            return fn(constraint)

    raise NotImplementedError(f"biject_to not implemented for {constraint}")


def transform_to(constraint: constraints.Constraint) -> Transform:
    """
    Get a transform that maps the reals to the constraint's support.

    Unlike biject_to, this may not be bijective.

    Args:
        constraint: The target constraint

    Returns:
        A Transform that maps reals to the constraint's support
    """
    constraint_type = type(constraint)
    if constraint_type in _TRANSFORM_TO_REGISTRY:
        return _TRANSFORM_TO_REGISTRY[constraint_type](constraint)

    # Fall back to biject_to
    return biject_to(constraint)


def _register_transforms():
    """Register default constraint->transform mappings."""

    # Real: identity
    @_register_biject(constraints._Real)
    def _biject_real(c):
        return AffineTransform(0, 1)

    # Positive: exp
    @_register_biject(constraints._Positive)
    def _biject_positive(c):
        return ExpTransform()

    @_register_biject(constraints._Nonnegative)
    def _biject_nonnegative(c):
        return ExpTransform()

    # Unit interval: sigmoid
    @_register_biject(constraints._Interval)
    def _biject_interval(c):
        if c.lower_bound == 0 and c.upper_bound == 1:
            return SigmoidTransform()
        return ComposeTransform([
            SigmoidTransform(),
            AffineTransform(c.lower_bound, c.upper_bound - c.lower_bound)
        ])

    # Greater than: shift + exp
    @_register_biject(constraints._GreaterThan)
    def _biject_greater_than(c):
        return ComposeTransform([ExpTransform(), AffineTransform(c.lower_bound, 1)])

    # Less than: negate + shift + exp
    @_register_biject(constraints._LessThan)
    def _biject_less_than(c):
        return ComposeTransform([ExpTransform(), AffineTransform(0, -1), AffineTransform(c.upper_bound, 1)])

    # Simplex: stick breaking
    @_register_biject(constraints._Simplex)
    def _biject_simplex(c):
        return StickBreakingTransform()

    # Lower Cholesky
    @_register_biject(constraints._LowerCholesky)
    def _biject_lower_cholesky(c):
        return LowerCholeskyTransform()


def _register_biject(constraint_type: type):
    """Decorator to register a biject_to transform."""
    def decorator(fn: Callable):
        _BIJECT_TO_REGISTRY[constraint_type] = fn
        _TRANSFORM_TO_REGISTRY[constraint_type] = fn
        return fn
    return decorator


# Initialize
_register_transforms()


__all__ = ['biject_to', 'transform_to']
