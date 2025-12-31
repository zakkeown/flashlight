"""
View and shape manipulation operations.

Implements PyTorch-compatible view semantics on top of MLX arrays.

Note: MLX arrays are immutable, so "views" create new arrays but track
the relationship to the base tensor for autograd purposes.

Reference:
- pytorch-mlx-porting-docs/01-FOUNDATIONS/tensor-core.md (lines 250-350)
- pytorch-mlx-porting-docs/02-OPERATORS/operator-reference/shape-manipulation.md
"""

from typing import Union, Tuple, Sequence, Optional

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


def reshape(input: 'Tensor', shape: Union[Tuple[int, ...], Sequence[int]]) -> 'Tensor':
    """
    Reshape tensor to new shape.

    Args:
        input: Input tensor
        shape: New shape (can contain -1 for inferred dimension)

    Returns:
        Reshaped tensor (view)

    Example:
        >>> x = randn(12)
        >>> y = reshape(x, (3, 4))
    """
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    mlx_array = mx.reshape(input._mlx_array, shape)
    result = Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)

    # Mark as view
    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def view(input: 'Tensor', *shape: int) -> 'Tensor':
    """
    View tensor with new shape (PyTorch alias for reshape).

    Args:
        input: Input tensor
        *shape: New shape dimensions

    Returns:
        Reshaped tensor (view)

    Example:
        >>> x = randn(12)
        >>> y = x.view(3, 4)
        >>> z = x.view(-1, 2)  # -1 inferred as 6
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])

    return reshape(input, shape)


def transpose(input: 'Tensor', dim0: int, dim1: int) -> 'Tensor':
    """
    Transpose two dimensions of a tensor.

    Args:
        input: Input tensor
        dim0: First dimension
        dim1: Second dimension

    Returns:
        Transposed tensor (view)

    Example:
        >>> x = randn(2, 3, 4)
        >>> y = transpose(x, 0, 2)  # Shape (4, 3, 2)
    """
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Build permutation axes
    ndim = len(input.shape)
    axes = list(range(ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

    mlx_array = mx.transpose(input._mlx_array, axes)
    result = Tensor._from_mlx_array(mlx_array)

    result._is_view = True
    result._base = input if input._base is None else input._base

    # Attach gradient function if needed
    from .autograd.context import is_grad_enabled
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        from .autograd.function import TransposeBackward
        grad_fn = TransposeBackward(input, dim0, dim1)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def permute(input: 'Tensor', *dims: int) -> 'Tensor':
    """
    Permute dimensions of a tensor.

    Args:
        input: Input tensor
        *dims: New order of dimensions

    Returns:
        Permuted tensor (view)

    Example:
        >>> x = randn(2, 3, 4)
        >>> y = permute(x, 2, 0, 1)  # Shape (4, 2, 3)
    """
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Handle both permute(x, 2, 0, 1) and permute(x, (2, 0, 1))
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])

    mlx_array = mx.transpose(input._mlx_array, dims)
    result = Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)

    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def squeeze(input: 'Tensor', dim: Optional[int] = None) -> 'Tensor':
    """
    Remove dimensions of size 1.

    Args:
        input: Input tensor
        dim: If specified, only squeeze this dimension (if size 1)

    Returns:
        Squeezed tensor (view)

    Example:
        >>> x = randn(1, 3, 1, 4)
        >>> y = squeeze(x)  # Shape (3, 4)
        >>> z = squeeze(x, 0)  # Shape (3, 1, 4)
    """
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    if dim is None:
        mlx_array = mx.squeeze(input._mlx_array)
    else:
        # Squeeze specific dimension only if it's size 1
        if input.shape[dim] == 1:
            mlx_array = mx.squeeze(input._mlx_array, axis=dim)
        else:
            # No-op if dimension is not size 1
            mlx_array = input._mlx_array

    result = Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)
    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def unsqueeze(input: 'Tensor', dim: int) -> 'Tensor':
    """
    Add a dimension of size 1.

    Args:
        input: Input tensor
        dim: Position to insert new dimension

    Returns:
        Tensor with new dimension (view)

    Example:
        >>> x = randn(3, 4)
        >>> y = unsqueeze(x, 0)  # Shape (1, 3, 4)
        >>> z = unsqueeze(x, 2)  # Shape (3, 4, 1)
    """
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    mlx_array = mx.expand_dims(input._mlx_array, axis=dim)
    result = Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)

    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def flatten(
    input: 'Tensor',
    start_dim: int = 0,
    end_dim: int = -1
) -> 'Tensor':
    """
    Flatten dimensions from start_dim to end_dim.

    Args:
        input: Input tensor
        start_dim: First dimension to flatten
        end_dim: Last dimension to flatten (inclusive)

    Returns:
        Flattened tensor (view)

    Example:
        >>> x = randn(2, 3, 4, 5)
        >>> y = flatten(x)  # Shape (120,)
        >>> z = flatten(x, 1, 2)  # Shape (2, 12, 5)
    """
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    ndim = len(input.shape)

    # Handle negative indices
    if start_dim < 0:
        start_dim = ndim + start_dim
    if end_dim < 0:
        end_dim = ndim + end_dim

    # Compute new shape
    new_shape = list(input.shape[:start_dim])

    # Flatten middle dimensions
    flat_dim = 1
    for i in range(start_dim, end_dim + 1):
        flat_dim *= input.shape[i]
    new_shape.append(flat_dim)

    # Add remaining dimensions
    new_shape.extend(input.shape[end_dim + 1:])

    mlx_array = mx.reshape(input._mlx_array, new_shape)
    result = Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)

    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def contiguous(input: 'Tensor') -> 'Tensor':
    """
    Return a contiguous tensor (no-op in MLX).

    MLX arrays are always contiguous, so this is provided for API compatibility.

    Args:
        input: Input tensor

    Returns:
        Contiguous tensor (same as input in MLX)
    """
    # In MLX, arrays are always contiguous
    # Return a copy for proper semantics
    from .tensor import Tensor

    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    mlx_array = mx.array(input._mlx_array)
    return Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)


__all__ = [
    'reshape',
    'view',
    'transpose',
    'permute',
    'squeeze',
    'unsqueeze',
    'flatten',
    'contiguous',
]
