"""
Sorting and Selection Operations

Implements PyTorch-compatible sorting operations with MLX backend.
"""

from typing import Optional, Tuple, Union
import mlx.core as mx

from ..tensor import Tensor
from ..autograd.context import is_grad_enabled


def sort(input: Tensor, dim: int = -1, descending: bool = False, stable: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Sorts the elements of input along a given dimension.

    Args:
        input: Input tensor
        dim: Dimension to sort along (default: -1, last dimension)
        descending: If True, sort in descending order
        stable: If True, preserve order of equal elements (not supported in MLX, ignored)

    Returns:
        Tuple of (sorted_values, sorted_indices)

    Example:
        >>> x = mlx_compat.tensor([3, 1, 4, 1, 5])
        >>> values, indices = mlx_compat.sort(x)
        >>> values
        tensor([1, 1, 3, 4, 5])
    """
    mlx_array = input._mlx_array

    # MLX sort returns values, argsort returns indices
    sorted_values = mx.sort(mlx_array, axis=dim)
    sorted_indices = mx.argsort(mlx_array, axis=dim)

    if descending:
        # Reverse along the sorted dimension using slicing (MLX doesn't have flip)
        ndim = len(mlx_array.shape)
        dim_normalized = dim if dim >= 0 else ndim + dim
        slices = [slice(None)] * ndim
        slices[dim_normalized] = slice(None, None, -1)
        sorted_values = sorted_values[tuple(slices)]
        sorted_indices = sorted_indices[tuple(slices)]

    values_tensor = Tensor._from_mlx_array(sorted_values)
    indices_tensor = Tensor._from_mlx_array(sorted_indices.astype(mx.int64))

    # Note: sort gradients are complex (permutation-based), not implementing autograd here
    # PyTorch's sort also doesn't track gradients through indices
    if is_grad_enabled() and input.requires_grad:
        values_tensor.requires_grad = True

    return values_tensor, indices_tensor


def argsort(input: Tensor, dim: int = -1, descending: bool = False, stable: bool = False) -> Tensor:
    """
    Returns the indices that would sort a tensor along a given dimension.

    Args:
        input: Input tensor
        dim: Dimension to sort along (default: -1)
        descending: If True, sort in descending order
        stable: If True, preserve order of equal elements (not supported, ignored)

    Returns:
        Tensor of indices

    Example:
        >>> x = mlx_compat.tensor([3, 1, 4, 1, 5])
        >>> mlx_compat.argsort(x)
        tensor([1, 3, 0, 2, 4])
    """
    mlx_array = input._mlx_array
    indices = mx.argsort(mlx_array, axis=dim)

    if descending:
        # Reverse using slicing (MLX doesn't have flip)
        ndim = len(mlx_array.shape)
        dim_normalized = dim if dim >= 0 else ndim + dim
        slices = [slice(None)] * ndim
        slices[dim_normalized] = slice(None, None, -1)
        indices = indices[tuple(slices)]

    return Tensor._from_mlx_array(indices.astype(mx.int64))


def topk(input: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Returns the k largest (or smallest) elements along a dimension.

    Args:
        input: Input tensor
        k: Number of top elements to return
        dim: Dimension to find top-k along (default: -1)
        largest: If True, return k largest; if False, return k smallest
        sorted: If True, return sorted (default: True)

    Returns:
        Tuple of (values, indices)

    Example:
        >>> x = mlx_compat.tensor([3, 1, 4, 1, 5, 9, 2, 6])
        >>> values, indices = mlx_compat.topk(x, 3)
        >>> values
        tensor([9, 6, 5])
    """
    mlx_array = input._mlx_array

    # MLX topk only returns values, not indices
    # We need to use argpartition or argsort to get indices
    if largest:
        # Get indices of top k by sorting
        all_indices = mx.argsort(mlx_array, axis=dim)
        # Take last k indices (largest values)
        ndim = len(mlx_array.shape)
        dim_normalized = dim if dim >= 0 else ndim + dim
        slices = [slice(None)] * ndim
        slices[dim_normalized] = slice(-k, None)
        indices = all_indices[tuple(slices)]
        # Reverse to get descending order
        slices[dim_normalized] = slice(None, None, -1)
        indices = indices[tuple(slices)]
        # Gather values
        values = mx.take_along_axis(mlx_array, indices, axis=dim)
    else:
        # Get indices of bottom k
        all_indices = mx.argsort(mlx_array, axis=dim)
        ndim = len(mlx_array.shape)
        dim_normalized = dim if dim >= 0 else ndim + dim
        slices = [slice(None)] * ndim
        slices[dim_normalized] = slice(None, k)
        indices = all_indices[tuple(slices)]
        values = mx.take_along_axis(mlx_array, indices, axis=dim)

    values_tensor = Tensor._from_mlx_array(values)
    indices_tensor = Tensor._from_mlx_array(indices.astype(mx.int64))

    if is_grad_enabled() and input.requires_grad:
        values_tensor.requires_grad = True

    return values_tensor, indices_tensor


def kthvalue(input: Tensor, k: int, dim: int = -1, keepdim: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Returns the k-th smallest element along a dimension.

    Args:
        input: Input tensor
        k: k for the k-th smallest element (1-indexed)
        dim: Dimension to find k-th value along
        keepdim: Whether to keep the reduced dimension

    Returns:
        Tuple of (value, index)
    """
    # Sort and take the k-th element (k is 1-indexed in PyTorch)
    sorted_vals, sorted_idx = sort(input, dim=dim, descending=False)

    # Index into the k-1 position (0-indexed)
    k_idx = k - 1

    # Take slice at position k_idx along dim
    ndim = len(input.shape)
    dim = dim if dim >= 0 else ndim + dim

    # Build slice tuple
    slices = [slice(None)] * ndim
    slices[dim] = k_idx

    value = Tensor._from_mlx_array(sorted_vals._mlx_array[tuple(slices)])
    index = Tensor._from_mlx_array(sorted_idx._mlx_array[tuple(slices)])

    if keepdim:
        # Add dimension back
        value = value.unsqueeze(dim)
        index = index.unsqueeze(dim)

    return value, index


def msort(input: Tensor) -> Tensor:
    """
    Sorts the elements of input along its first dimension.

    Equivalent to torch.sort(input, dim=0).values

    Args:
        input: Input tensor

    Returns:
        Sorted tensor
    """
    values, _ = sort(input, dim=0)
    return values


def unique(
    input: Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: Optional[int] = None
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """
    Returns the unique elements of the input tensor.

    Args:
        input: Input tensor
        sorted: Whether to sort the unique elements (default: True)
        return_inverse: Whether to return inverse indices
        return_counts: Whether to return counts of each unique value
        dim: Dimension to operate on (None for flattened tensor)

    Returns:
        Tensor of unique values, optionally with inverse indices and/or counts

    Example:
        >>> x = mlx_compat.tensor([1, 2, 1, 3, 2, 1])
        >>> mlx_compat.unique(x)
        tensor([1, 2, 3])
    """
    mlx_array = input._mlx_array

    if dim is not None:
        raise NotImplementedError("unique with dim parameter is not yet supported")

    # Flatten for unique operation
    flat = mlx_array.reshape(-1)

    # Sort first to find unique values
    sorted_arr = mx.sort(flat)

    n = sorted_arr.shape[0]
    if n == 0:
        unique_vals = sorted_arr
        if return_inverse or return_counts:
            results = [Tensor._from_mlx_array(unique_vals)]
            if return_inverse:
                results.append(Tensor._from_mlx_array(mx.array([], dtype=mx.int64)))
            if return_counts:
                results.append(Tensor._from_mlx_array(mx.array([], dtype=mx.int64)))
            return tuple(results)
        return Tensor._from_mlx_array(unique_vals)

    # Compare adjacent elements - find where consecutive elements differ
    diff_mask = mx.concatenate([
        mx.array([True]),
        sorted_arr[1:] != sorted_arr[:-1]
    ])

    # Use cumsum to count unique elements and create indices
    # cumsum gives us the unique index for each position
    unique_idx_per_elem = mx.cumsum(diff_mask.astype(mx.int32)) - 1
    num_unique = unique_idx_per_elem[-1].item() + 1

    # Create output array by taking first occurrence of each unique value
    # Use where to extract indices where diff_mask is True
    # Build unique_vals by iterating (MLX doesn't have boolean indexing or argwhere)
    unique_indices_list = []
    for i in range(n):
        if diff_mask[i].item():
            unique_indices_list.append(i)
    unique_indices = mx.array(unique_indices_list, dtype=mx.int32)
    unique_vals = sorted_arr[unique_indices]

    result = Tensor._from_mlx_array(unique_vals)

    if not return_inverse and not return_counts:
        return result

    results = [result]

    if return_inverse:
        # Compute inverse indices: for each element in flat, find its index in unique_vals
        # Since we sorted first, we need to track original positions through sort
        sort_indices = mx.argsort(flat)
        inverse_sorted = mx.zeros(n, dtype=mx.int32)

        # unique_idx_per_elem tells us which unique value index each sorted position maps to
        # We need to unsort this back to original order
        inverse_list = [0] * n
        for i in range(n):
            orig_idx = sort_indices[i].item()
            unique_idx = unique_idx_per_elem[i].item()
            inverse_list[orig_idx] = unique_idx
        inverse_arr = mx.array(inverse_list, dtype=mx.int64)
        results.append(Tensor._from_mlx_array(inverse_arr))

    if return_counts:
        # Count occurrences by finding gaps between unique indices
        end_indices = mx.concatenate([unique_indices[1:], mx.array([n])])
        counts = end_indices - unique_indices
        results.append(Tensor._from_mlx_array(counts.astype(mx.int64)))

    return tuple(results)


def unique_consecutive(
    *args,
    **kwargs
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """
    Eliminates all but the first element from every consecutive group of equivalent elements.

    Args:
        *args, **kwargs: Flexible arguments matching PyTorch signature
            input: Input tensor
            return_inverse: Whether to return inverse indices
            return_counts: Whether to return counts
            dim: Dimension to operate on

    Returns:
        Tensor with consecutive duplicates removed
    """
    # Parse args/kwargs to match PyTorch's flexible signature
    if len(args) >= 1:
        input = args[0]
        args = args[1:]
    else:
        input = kwargs.pop('input')

    return_inverse = kwargs.pop('return_inverse', False)
    return_counts = kwargs.pop('return_counts', False)
    dim = kwargs.pop('dim', None)

    mlx_array = input._mlx_array

    if dim is not None:
        raise NotImplementedError("unique_consecutive with dim is not yet supported")

    flat = mlx_array.reshape(-1)
    n = flat.shape[0]

    if n == 0:
        result = Tensor._from_mlx_array(flat)
        if return_inverse or return_counts:
            results = [result]
            if return_inverse:
                results.append(Tensor._from_mlx_array(mx.array([], dtype=mx.int64)))
            if return_counts:
                results.append(Tensor._from_mlx_array(mx.array([], dtype=mx.int64)))
            return tuple(results)
        return result

    # Find where consecutive elements differ
    diff_mask = mx.concatenate([
        mx.array([True]),
        flat[1:] != flat[:-1]
    ])

    # Get indices where diff_mask is True (MLX doesn't support boolean indexing or argwhere)
    unique_indices_list = []
    for i in range(n):
        if diff_mask[i].item():
            unique_indices_list.append(i)
    unique_indices = mx.array(unique_indices_list, dtype=mx.int32)
    unique_vals = flat[unique_indices]
    result = Tensor._from_mlx_array(unique_vals)

    if not return_inverse and not return_counts:
        return result

    results = [result]

    if return_inverse:
        # Each element maps to cumsum of diff_mask - 1
        inverse = mx.cumsum(diff_mask.astype(mx.int32)) - 1
        results.append(Tensor._from_mlx_array(inverse.astype(mx.int64)))

    if return_counts:
        # Count consecutive runs
        end_indices = mx.concatenate([unique_indices[1:], mx.array([n])])
        counts = end_indices - unique_indices
        results.append(Tensor._from_mlx_array(counts.astype(mx.int64)))

    return tuple(results)


__all__ = [
    'sort',
    'argsort',
    'topk',
    'kthvalue',
    'msort',
    'unique',
    'unique_consecutive',
]
