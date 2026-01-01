"""
Comparison Operations

Implements PyTorch-compatible comparison operations with MLX backend.
"""

import mlx.core as mx
from ..tensor import Tensor
from ..autograd.context import is_grad_enabled


def eq(input: Tensor, other) -> Tensor:
    """
    Element-wise equality comparison.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Boolean tensor where True indicates equal elements
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.equal(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def ne(input: Tensor, other) -> Tensor:
    """
    Element-wise inequality comparison.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Boolean tensor where True indicates unequal elements
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.not_equal(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def lt(input: Tensor, other) -> Tensor:
    """
    Element-wise less than comparison.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Boolean tensor where True indicates input < other
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.less(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def le(input: Tensor, other) -> Tensor:
    """
    Element-wise less than or equal comparison.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Boolean tensor where True indicates input <= other
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.less_equal(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def gt(input: Tensor, other) -> Tensor:
    """
    Element-wise greater than comparison.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Boolean tensor where True indicates input > other
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.greater(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def ge(input: Tensor, other) -> Tensor:
    """
    Element-wise greater than or equal comparison.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Boolean tensor where True indicates input >= other
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.greater_equal(input._mlx_array, other_array)
    return Tensor._from_mlx_array(result_array)


def equal(input: Tensor, other: Tensor) -> bool:
    """
    Check if two tensors are equal (same shape and all elements equal).

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        True if tensors are equal
    """
    if input.shape != other.shape:
        return False
    return bool(mx.all(mx.equal(input._mlx_array, other._mlx_array)).item())


def allclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two tensors are element-wise equal within a tolerance.

    Args:
        input: First tensor
        other: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if tensors are close
    """
    diff = mx.abs(input._mlx_array - other._mlx_array)
    tolerance = atol + rtol * mx.abs(other._mlx_array)
    return bool(mx.all(diff <= tolerance).item())


def isclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> Tensor:
    """
    Element-wise comparison within tolerance.

    Args:
        input: First tensor
        other: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Boolean tensor indicating where elements are close
    """
    diff = mx.abs(input._mlx_array - other._mlx_array)
    tolerance = atol + rtol * mx.abs(other._mlx_array)
    result_array = diff <= tolerance
    return Tensor._from_mlx_array(result_array)


def maximum(input: Tensor, other) -> Tensor:
    """
    Element-wise maximum of two tensors.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Tensor with element-wise maximum
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.maximum(input._mlx_array, other_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True

    return result


def minimum(input: Tensor, other) -> Tensor:
    """
    Element-wise minimum of two tensors.

    Args:
        input: First tensor
        other: Second tensor or scalar

    Returns:
        Tensor with element-wise minimum
    """
    if isinstance(other, Tensor):
        other_array = other._mlx_array
    else:
        other_array = other

    result_array = mx.minimum(input._mlx_array, other_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or (isinstance(other, Tensor) and other.requires_grad)):
        result.requires_grad = True

    return result


# PyTorch-style aliases
greater = gt
greater_equal = ge
less = lt
less_equal = le
not_equal = ne


__all__ = [
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'equal', 'allclose', 'isclose',
    'maximum', 'minimum',
    # Aliases
    'greater', 'greater_equal', 'less', 'less_equal', 'not_equal',
]
