"""
Indexing Operations

Implements PyTorch-compatible indexing operations with MLX backend.
"""

from typing import Optional, Union
import mlx.core as mx

from ..tensor import Tensor


def scatter(input: Tensor, dim: int, index: Tensor, src: Union[Tensor, float]) -> Tensor:
    """
    Scatter values from src into input at indices.

    Args:
        input: Input tensor to scatter into
        dim: Dimension along which to scatter
        index: Indices to scatter to
        src: Source values (tensor or scalar)

    Returns:
        Tensor with scattered values
    """
    import numpy as np

    # Convert to numpy for scatter operation
    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)

    if isinstance(src, Tensor):
        src_np = np.array(src._mlx_array)
    else:
        # Scalar case - create array filled with scalar
        src_np = np.full(index_np.shape, src)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Perform scatter operation using numpy's advanced indexing
    # We need to iterate over all indices and place values
    it = np.nditer(index_np, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        target_idx = list(idx)
        target_idx[dim] = int(it[0])
        result_np[tuple(target_idx)] = src_np[idx]
        it.iternext()

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or (isinstance(src, Tensor) and src.requires_grad):
        result.requires_grad = True

    return result


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """
    Add values from src into input at indices (in-place scatter add).

    Args:
        input: Input tensor
        dim: Dimension along which to scatter
        index: Indices to scatter to
        src: Source values to add

    Returns:
        Tensor with added values
    """
    import numpy as np

    # Convert to numpy for scatter_add operation
    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)
    src_np = np.array(src._mlx_array)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Perform scatter_add operation using numpy's advanced indexing
    it = np.nditer(index_np, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        target_idx = list(idx)
        target_idx[dim] = int(it[0])
        result_np[tuple(target_idx)] += src_np[idx]
        it.iternext()

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result


def index_select(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Select values along dimension using indices.

    Args:
        input: Input tensor
        dim: Dimension to select from
        index: Indices to select

    Returns:
        Selected tensor
    """
    # MLX uses take for index selection
    mlx_result = mx.take(input._mlx_array, index._mlx_array, axis=dim)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def where(condition: Tensor, x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
    """
    Select elements from x or y based on condition.

    Args:
        condition: Boolean condition tensor
        x: Values where condition is True
        y: Values where condition is False

    Returns:
        Tensor with selected values
    """
    if isinstance(x, Tensor):
        x_array = x._mlx_array
    else:
        x_array = x

    if isinstance(y, Tensor):
        y_array = y._mlx_array
    else:
        y_array = y

    mlx_result = mx.where(condition._mlx_array, x_array, y_array)
    result = Tensor._from_mlx_array(mlx_result)

    # Propagate requires_grad
    requires_grad = False
    if isinstance(x, Tensor) and x.requires_grad:
        requires_grad = True
    if isinstance(y, Tensor) and y.requires_grad:
        requires_grad = True

    if requires_grad:
        result.requires_grad = True

    return result


def masked_fill(input: Tensor, mask: Tensor, value: float) -> Tensor:
    """
    Fill elements of input with value where mask is True.

    Args:
        input: Input tensor
        mask: Boolean mask
        value: Fill value

    Returns:
        Tensor with filled values
    """
    # Use where to implement masked_fill
    mlx_result = mx.where(mask._mlx_array, value, input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    """
    Select elements from input where mask is True.

    Args:
        input: Input tensor
        mask: Boolean mask

    Returns:
        1D tensor with selected elements
    """
    # Flatten and select based on mask
    # MLX doesn't support boolean indexing directly, use where to get indices
    flat_input = input._mlx_array.reshape(-1)
    flat_mask = mask._mlx_array.reshape(-1)

    # Create index array and use where to select
    indices = mx.arange(flat_mask.shape[0])
    # Get indices where mask is true
    selected_indices = mx.where(flat_mask, indices, flat_mask.shape[0])
    # Sort and take valid ones (< size)
    # Count true values
    num_true = int(mx.sum(flat_mask).item())
    # Compact the indices
    sorted_idx = mx.sort(selected_indices)[:num_true]

    mlx_result = mx.take(flat_input, sorted_idx)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: float = 1) -> Tensor:
    """
    Add values from source to input at indices.

    Args:
        input: Input tensor
        dim: Dimension along which to add
        index: Indices where to add
        source: Source values to add
        alpha: Multiplier for source

    Returns:
        Tensor with added values
    """
    import numpy as np

    # Convert to numpy for index_add operation
    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)
    source_np = np.array(source._mlx_array)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Perform index_add operation
    # For each index, add the corresponding slice from source to result
    for i, idx in enumerate(index_np):
        # Build slicing tuples for source and destination
        src_slices = [slice(None)] * ndim
        src_slices[dim] = i

        dst_slices = [slice(None)] * ndim
        dst_slices[dim] = int(idx)

        result_np[tuple(dst_slices)] += alpha * source_np[tuple(src_slices)]

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or source.requires_grad:
        result.requires_grad = True

    return result


def nonzero(input: Tensor, as_tuple: bool = False) -> Union[Tensor, tuple]:
    """
    Return indices of non-zero elements.

    Args:
        input: Input tensor
        as_tuple: If True, return tuple of 1D tensors (one per dimension)

    Returns:
        Indices of non-zero elements
    """
    # Create boolean mask of non-zero elements
    nonzero_mask = input._mlx_array != 0

    if as_tuple:
        # Return tuple of indices for each dimension
        # For each dimension, find indices where any element in that dimension is non-zero
        result = []
        shape = input.shape

        # Flatten the array and mask
        flat_mask = nonzero_mask.reshape(-1)
        num_nonzero = int(mx.sum(flat_mask).item())

        if num_nonzero == 0:
            # No non-zero elements
            return tuple(Tensor._from_mlx_array(mx.array([], dtype=mx.int32)) for _ in range(input.ndim))

        # Get flat indices
        flat_indices = mx.arange(flat_mask.shape[0])
        selected_flat = mx.where(flat_mask, flat_indices, flat_mask.shape[0])
        valid_flat = mx.sort(selected_flat)[:num_nonzero]

        # Convert flat indices to multi-dimensional indices
        for dim in range(input.ndim):
            # Calculate stride for this dimension
            stride = 1
            for d in range(dim + 1, input.ndim):
                stride *= shape[d]
            dim_indices = (valid_flat // stride) % shape[dim]
            result.append(Tensor._from_mlx_array(dim_indices))

        return tuple(result)
    else:
        # Return 2D tensor of indices [num_nonzero, ndim]
        flat_mask = nonzero_mask.reshape(-1)
        num_nonzero = int(mx.sum(flat_mask).item())

        if num_nonzero == 0:
            # Return empty array with correct shape
            return Tensor._from_mlx_array(mx.zeros((0, input.ndim), dtype=mx.int32))

        # Get flat indices
        flat_indices = mx.arange(flat_mask.shape[0])
        selected_flat = mx.where(flat_mask, flat_indices, flat_mask.shape[0])
        valid_flat = mx.sort(selected_flat)[:num_nonzero]

        # Convert to multi-dimensional indices
        indices_list = []
        shape = input.shape
        for dim in range(input.ndim):
            stride = 1
            for d in range(dim + 1, input.ndim):
                stride *= shape[d]
            dim_indices = (valid_flat // stride) % shape[dim]
            indices_list.append(dim_indices)

        # Stack into 2D array
        indices = mx.stack(indices_list, axis=1)
        result = Tensor._from_mlx_array(indices)
        return result


def take(input: Tensor, index: Tensor) -> Tensor:
    """
    Take elements from input at flat indices.

    Args:
        input: Input tensor
        index: Flat indices

    Returns:
        Selected elements
    """
    flat_input = input._mlx_array.reshape(-1)
    mlx_result = mx.take(flat_input, index._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def put(input: Tensor, index: Tensor, source: Tensor, accumulate: bool = False) -> Tensor:
    """
    Put values from source into input at flat indices.

    Args:
        input: Input tensor
        index: Flat indices
        source: Source values
        accumulate: If True, add values instead of replacing

    Returns:
        Tensor with put values
    """
    import numpy as np

    # Convert to numpy for put operation
    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)
    source_np = np.array(source._mlx_array)

    # Flatten the result for indexing
    original_shape = result_np.shape
    result_flat = result_np.reshape(-1)

    # Put values at flat indices
    if accumulate:
        # Add values instead of replacing
        for i, idx in enumerate(index_np.flat):
            result_flat[int(idx)] += source_np.flat[i]
    else:
        # Replace values
        for i, idx in enumerate(index_np.flat):
            result_flat[int(idx)] = source_np.flat[i]

    # Reshape back to original shape
    result_np = result_flat.reshape(original_shape)
    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or source.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# Extended Indexing Operations (Sprint 5)
# =============================================================================

def index_fill(input: Tensor, dim: int, index: Tensor, value: float) -> Tensor:
    """
    Fill elements of input with value at indices along dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to fill
        index: Indices where to fill
        value: Fill value

    Returns:
        Tensor with filled values
    """
    import numpy as np

    # Convert to numpy for easier manipulation
    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)

    # Fill at indices along dimension
    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Create slices for indexing
    for idx in index_np:
        slices = [slice(None)] * ndim
        slices[dim] = int(idx)
        result_np[tuple(slices)] = value

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad:
        result.requires_grad = True

    return result


def index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    """
    Copy values from source to input at indices along dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to copy
        index: Indices where to copy
        source: Source values

    Returns:
        Tensor with copied values
    """
    import numpy as np

    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)
    source_np = np.array(source._mlx_array)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Copy from source at indices
    for i, idx in enumerate(index_np):
        src_slices = [slice(None)] * source_np.ndim
        src_slices[dim] = i

        dst_slices = [slice(None)] * ndim
        dst_slices[dim] = int(idx)

        result_np[tuple(dst_slices)] = source_np[tuple(src_slices)]

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or source.requires_grad:
        result.requires_grad = True

    return result


def index_put(input: Tensor, indices: tuple, values: Union[Tensor, float], accumulate: bool = False) -> Tensor:
    """
    Put values into input at indices.

    Args:
        input: Input tensor
        indices: Tuple of index tensors
        values: Values to put
        accumulate: If True, add values instead of replacing

    Returns:
        Tensor with put values
    """
    import numpy as np

    result_np = np.array(input._mlx_array)

    # Convert indices to numpy
    np_indices = []
    for idx in indices:
        if isinstance(idx, Tensor):
            np_indices.append(np.array(idx._mlx_array).astype(int))
        else:
            np_indices.append(idx)

    if isinstance(values, Tensor):
        values_np = np.array(values._mlx_array)
    else:
        values_np = values

    if accumulate:
        result_np[tuple(np_indices)] += values_np
    else:
        result_np[tuple(np_indices)] = values_np

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or (isinstance(values, Tensor) and values.requires_grad):
        result.requires_grad = True

    return result


def fill(input: Tensor, value: float) -> Tensor:
    """
    Fill tensor with a scalar value.

    Args:
        input: Input tensor
        value: Fill value

    Returns:
        Tensor filled with value
    """
    mlx_result = mx.full(input.shape, value, dtype=input._mlx_array.dtype)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def take_along_dim(input: Tensor, indices: Tensor, dim: int = None) -> Tensor:
    """
    Take values from input along a dimension using indices.

    Args:
        input: Input tensor
        indices: Indices tensor
        dim: Dimension along which to take. If None, flattens input first.

    Returns:
        Gathered values
    """
    if dim is None:
        # Flatten input if dim is None
        input_arr = input._mlx_array.flatten()
        indices_arr = indices._mlx_array.flatten()
        mlx_result = mx.take(input_arr, indices_arr)
    else:
        mlx_result = mx.take_along_axis(input._mlx_array, indices._mlx_array, axis=dim)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def argwhere(input: Tensor) -> Tensor:
    """
    Return indices of non-zero elements as (N, ndim) tensor.

    Args:
        input: Input tensor

    Returns:
        2D tensor with indices
    """
    # Use nonzero with as_tuple=False
    return nonzero(input, as_tuple=False)


def isin(elements: Tensor, test_elements: Tensor, assume_unique: bool = False, invert: bool = False) -> Tensor:
    """
    Test if elements are contained in test_elements.

    Args:
        elements: Elements to test
        test_elements: Elements to test against
        assume_unique: If True, assume both are unique (unused)
        invert: If True, return elements NOT in test_elements

    Returns:
        Boolean tensor of same shape as elements
    """
    import numpy as np

    elements_np = np.array(elements._mlx_array)
    test_np = np.array(test_elements._mlx_array)

    result_np = np.isin(elements_np, test_np, invert=invert)
    result = Tensor._from_mlx_array(mx.array(result_np))

    return result


def diagonal_scatter(input: Tensor, src: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
    """
    Embed values from src into the diagonals of input.

    Args:
        input: Input tensor (at least 2D)
        src: Source tensor for diagonal values
        offset: Diagonal offset (positive = above main, negative = below main)
        dim1: First dimension for the 2D sub-matrix
        dim2: Second dimension for the 2D sub-matrix

    Returns:
        Tensor with src embedded in the diagonal

    Example:
        >>> a = torch.zeros(3, 3)
        >>> diagonal_scatter(a, torch.tensor([1, 2, 3]), 0)
        tensor([[1., 0., 0.],
                [0., 2., 0.],
                [0., 0., 3.]])
    """
    import numpy as np

    result_np = np.array(input._mlx_array)
    src_np = np.array(src._mlx_array)

    ndim = result_np.ndim
    dim1 = dim1 if dim1 >= 0 else ndim + dim1
    dim2 = dim2 if dim2 >= 0 else ndim + dim2

    # Handle different dimensions by transposing
    # Move dim1 and dim2 to the last two dimensions
    axes = list(range(ndim))
    axes.remove(dim1)
    axes.remove(dim2)
    axes.extend([dim1, dim2])

    result_transposed = np.transpose(result_np, axes)

    # Now operate on last two dimensions
    shape = result_transposed.shape
    h, w = shape[-2], shape[-1]

    # Set diagonal
    if offset >= 0:
        diag_len = min(h, w - offset)
        for i in range(diag_len):
            # Handle multi-dimensional case
            idx = tuple([...] + [i, i + offset])
            result_transposed[idx] = src_np[..., i] if src_np.ndim > 1 else src_np[i]
    else:
        diag_len = min(h + offset, w)
        for i in range(diag_len):
            idx = tuple([...] + [i - offset, i])
            result_transposed[idx] = src_np[..., i] if src_np.ndim > 1 else src_np[i]

    # Transpose back
    inv_axes = [0] * ndim
    for i, ax in enumerate(axes):
        inv_axes[ax] = i
    result_np = np.transpose(result_transposed, inv_axes)

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result


def slice_scatter(input: Tensor, src: Tensor, dim: int = 0, start: Optional[int] = None,
                  end: Optional[int] = None, step: int = 1) -> Tensor:
    """
    Embed values from src into input at a slice.

    Args:
        input: Input tensor
        src: Source tensor to embed
        dim: Dimension to slice along
        start: Start index of slice
        end: End index of slice
        step: Step of slice

    Returns:
        Tensor with src embedded at the slice
    """
    import numpy as np

    result_np = np.array(input._mlx_array)
    src_np = np.array(src._mlx_array)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Build the slice
    slices = [slice(None)] * ndim
    slices[dim] = slice(start, end, step)

    result_np[tuple(slices)] = src_np

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result


def select_scatter(input: Tensor, src: Tensor, dim: int, index: int) -> Tensor:
    """
    Embed values from src into input at a selected index.

    Args:
        input: Input tensor
        src: Source tensor (one dimension less than input)
        dim: Dimension to select along
        index: Index to embed at

    Returns:
        Tensor with src embedded at the index
    """
    import numpy as np

    result_np = np.array(input._mlx_array)
    src_np = np.array(src._mlx_array)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Build the index tuple
    slices = [slice(None)] * ndim
    slices[dim] = index

    result_np[tuple(slices)] = src_np

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result


def scatter_reduce(input: Tensor, dim: int, index: Tensor, src: Tensor,
                   reduce: str, include_self: bool = True) -> Tensor:
    """
    Scatter and reduce values from src into input.

    Args:
        input: Input tensor
        dim: Dimension to scatter along
        index: Indices for scattering
        src: Source values
        reduce: Reduction operation ('sum', 'prod', 'mean', 'amax', 'amin')
        include_self: Whether to include input values in reduction

    Returns:
        Tensor with reduced values
    """
    import numpy as np

    result_np = np.array(input._mlx_array)
    index_np = np.array(index._mlx_array).astype(int)
    src_np = np.array(src._mlx_array)

    ndim = result_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # If not including self, initialize based on reduction
    if not include_self:
        if reduce == 'sum':
            result_np = np.zeros_like(result_np)
        elif reduce == 'prod':
            result_np = np.ones_like(result_np)
        elif reduce == 'amax':
            result_np = np.full_like(result_np, -np.inf)
        elif reduce == 'amin':
            result_np = np.full_like(result_np, np.inf)

    # Simple loop-based scatter reduce (can be optimized)
    it = np.nditer(index_np, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        target_idx = list(idx)
        target_idx[dim] = int(it[0])

        if reduce == 'sum':
            result_np[tuple(target_idx)] += src_np[idx]
        elif reduce == 'prod':
            result_np[tuple(target_idx)] *= src_np[idx]
        elif reduce == 'mean':
            result_np[tuple(target_idx)] += src_np[idx]  # Need count for proper mean
        elif reduce == 'amax':
            result_np[tuple(target_idx)] = max(result_np[tuple(target_idx)], src_np[idx])
        elif reduce == 'amin':
            result_np[tuple(target_idx)] = min(result_np[tuple(target_idx)], src_np[idx])

        it.iternext()

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result
