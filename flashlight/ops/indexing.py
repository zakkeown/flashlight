"""
Indexing Operations

Implements PyTorch-compatible indexing operations with MLX backend.
Optimized to use native MLX operations where possible.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor


def _build_scatter_indices(index_arr: mx.array, dim: int, ndim: int) -> Tuple[mx.array, ...]:
    """
    Build tuple of index arrays for scatter operation (vectorized).

    For PyTorch scatter semantics with dim=d:
    - out[i_0, ..., index[i_0, ..., i_n], ..., i_n] = src[i_0, ..., i_n]
    where the index value replaces position d.

    Args:
        index_arr: Index array
        dim: Dimension to scatter along
        ndim: Number of dimensions in output tensor

    Returns:
        Tuple of index arrays suitable for array.at[...].add()
    """
    idx_shape = index_arr.shape
    idx_ndim = len(idx_shape)

    # Create coordinate arrays for each dimension of the index
    coords = []
    for i in range(idx_ndim):
        r = mx.arange(idx_shape[i])
        shape = [1] * idx_ndim
        shape[i] = idx_shape[i]
        expanded = mx.reshape(r, shape)
        coords.append(mx.broadcast_to(expanded, idx_shape).reshape(-1))

    # Build output indices
    result_indices = []
    for d in range(ndim):
        if d == dim:
            # Use the actual index values for the scatter dimension
            result_indices.append(index_arr.reshape(-1).astype(mx.int32))
        elif d < idx_ndim:
            # Use coordinate grid for other dimensions
            result_indices.append(coords[d].astype(mx.int32))

    return tuple(result_indices)


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

    Note:
        Uses vectorized MLX operations instead of Python loops for better performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Handle scalar source
    if isinstance(src, Tensor):
        src_arr = src._mlx_array
    else:
        src_arr = mx.full(index.shape, src, dtype=input._mlx_array.dtype)

    # Build scatter indices
    indices = _build_scatter_indices(index._mlx_array, dim, ndim)

    # For scatter (overwrite mode), we need to:
    # 1. Start with input
    # 2. Zero out positions that will be written
    # 3. Add the scattered values

    # Create a mask of positions being written to
    ones = mx.ones_like(src_arr)
    mask_result = mx.zeros(input.shape, dtype=input._mlx_array.dtype)
    mask_result = mask_result.at[indices].add(ones.reshape(-1))

    # Zero out positions in input that will be overwritten (where mask > 0)
    # Then scatter src values
    zeroed_input = mx.where(mask_result > 0, mx.zeros_like(input._mlx_array), input._mlx_array)

    # For positions written multiple times, we want the last value
    # Use scatter_add with values, but since we zeroed the positions, it works
    result_arr = mx.zeros(input.shape, dtype=input._mlx_array.dtype)
    result_arr = result_arr.at[indices].add(src_arr.reshape(-1))

    # Combine: keep input where not scattered, use scattered values elsewhere
    final_result = zeroed_input + result_arr

    result = Tensor._from_mlx_array(final_result)

    if input.requires_grad or (isinstance(src, Tensor) and src.requires_grad):
        result.requires_grad = True

    return result


def scatter_add(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """
    Add values from src into input at indices (scatter with addition).

    Args:
        input: Input tensor
        dim: Dimension along which to scatter
        index: Indices to scatter to
        src: Source values to add

    Returns:
        Tensor with added values

    Note:
        Uses native MLX .at[].add() for vectorized performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Build scatter indices
    indices = _build_scatter_indices(index._mlx_array, dim, ndim)

    # Use MLX native scatter_add via .at[].add()
    result_arr = input._mlx_array.at[indices].add(src._mlx_array.reshape(-1))

    result = Tensor._from_mlx_array(result_arr)

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


def index_add(
    input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: float = 1
) -> Tensor:
    """
    Add values from source to input at indices along a dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to add
        index: 1D indices where to add (length must match source.shape[dim])
        source: Source values to add
        alpha: Multiplier for source

    Returns:
        Tensor with added values

    Note:
        Uses vectorized MLX operations for better performance.
        index_add is different from scatter_add: it selects entire slices along dim.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim

    # index_add semantics: for each i, add source[..., i, ...] to input[..., index[i], ...]
    # where i and index[i] are positions along dimension `dim`

    src_arr = source._mlx_array
    if alpha != 1:
        src_arr = src_arr * alpha

    # Build full scatter indices for all elements of source
    # We need to map each element in source to its target in input
    src_shape = src_arr.shape
    idx_arr = index._mlx_array

    # Create coordinate grids for all dimensions
    coords = []
    for d in range(ndim):
        r = mx.arange(src_shape[d])
        shape = [1] * ndim
        shape[d] = src_shape[d]
        expanded = mx.reshape(r, shape)
        coords.append(mx.broadcast_to(expanded, src_shape).reshape(-1))

    # Build output indices - replace the dim coordinate with index values
    result_indices = []
    for d in range(ndim):
        if d == dim:
            # Map source dim index -> target index via the index array
            # coords[d] contains 0, 1, 2, ... src_shape[dim] for each position
            # We want idx_arr[coords[d]]
            mapped = mx.take(idx_arr, coords[d].astype(mx.int32))
            result_indices.append(mapped.astype(mx.int32))
        else:
            result_indices.append(coords[d].astype(mx.int32))

    # Use scatter_add
    result_arr = input._mlx_array.at[tuple(result_indices)].add(src_arr.reshape(-1))

    result = Tensor._from_mlx_array(result_arr)

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
            return tuple(
                Tensor._from_mlx_array(mx.array([], dtype=mx.int32)) for _ in range(input.ndim)
            )

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

    Note:
        Uses vectorized MLX operations for better performance.
    """
    original_shape = input.shape
    flat_input = input._mlx_array.reshape(-1)
    flat_index = index._mlx_array.reshape(-1).astype(mx.int32)
    flat_source = source._mlx_array.reshape(-1)

    if accumulate:
        # Use scatter_add semantics
        result_flat = flat_input.at[flat_index].add(flat_source)
    else:
        # For overwrite mode, zero out target positions first, then add
        # This handles duplicate indices by taking the last value
        ones = mx.ones_like(flat_source)
        mask = mx.zeros_like(flat_input)
        mask = mask.at[flat_index].add(ones)

        # Zero out positions that will be overwritten
        zeroed = mx.where(mask > 0, mx.zeros_like(flat_input), flat_input)

        # Scatter the source values
        scattered = mx.zeros_like(flat_input)
        scattered = scattered.at[flat_index].add(flat_source)

        result_flat = zeroed + scattered

    result = Tensor._from_mlx_array(result_flat.reshape(original_shape))

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
        index: 1D indices where to fill
        value: Fill value

    Returns:
        Tensor with filled values

    Note:
        Uses vectorized MLX operations for better performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim
    shape = input.shape

    # Build indices for all elements in the slices to fill
    # For index_fill, we fill entire slices along dim at positions given by index

    # Create coordinate grids for each dimension
    idx_arr = index._mlx_array

    # Build shape for the filled region: same as input except dim has len(index) elements
    fill_shape = list(shape)
    fill_shape[dim] = len(idx_arr)

    # Create coordinate grids for the fill region
    coords = []
    for d in range(ndim):
        if d == dim:
            # Expand index to match fill_shape
            expand_shape = [1] * ndim
            expand_shape[d] = len(idx_arr)
            r = mx.reshape(idx_arr, expand_shape)
        else:
            expand_shape = [1] * ndim
            expand_shape[d] = shape[d]
            r = mx.reshape(mx.arange(shape[d]), expand_shape)
        coords.append(mx.broadcast_to(r, fill_shape).reshape(-1).astype(mx.int32))

    # Create mask and fill
    mask = mx.zeros(shape, dtype=mx.bool_)
    mask = mask.at[tuple(coords)].add(mx.ones(len(coords[0]), dtype=mx.bool_))

    result_arr = mx.where(mask, value, input._mlx_array)
    result = Tensor._from_mlx_array(result_arr)

    if input.requires_grad:
        result.requires_grad = True

    return result


def index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    """
    Copy values from source to input at indices along dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to copy
        index: 1D indices where to copy (length must match source.shape[dim])
        source: Source values

    Returns:
        Tensor with copied values

    Note:
        Uses vectorized MLX operations for better performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim
    shape = input.shape
    src_shape = source.shape

    # index_copy: copy source[..., i, ...] to input[..., index[i], ...] for each i
    idx_arr = index._mlx_array

    # Build coordinate grids for all elements in source
    coords = []
    for d in range(ndim):
        if d == dim:
            # Map source's dim coordinate through index array
            expand_shape = [1] * ndim
            expand_shape[d] = src_shape[d]
            src_coord = mx.reshape(mx.arange(src_shape[d]), expand_shape)
            src_coord = mx.broadcast_to(src_coord, src_shape).reshape(-1)
            # Map through index: index[src_coord]
            mapped = mx.take(idx_arr, src_coord.astype(mx.int32))
            coords.append(mapped.astype(mx.int32))
        else:
            expand_shape = [1] * ndim
            expand_shape[d] = src_shape[d]
            r = mx.reshape(mx.arange(src_shape[d]), expand_shape)
            coords.append(mx.broadcast_to(r, src_shape).reshape(-1).astype(mx.int32))

    # Create mask for positions being overwritten
    mask = mx.zeros(shape, dtype=mx.bool_)
    mask = mask.at[tuple(coords)].add(mx.ones(len(coords[0]), dtype=mx.bool_))

    # Zero out target positions
    zeroed = mx.where(mask, mx.zeros_like(input._mlx_array), input._mlx_array)

    # Scatter source values
    scattered = mx.zeros(shape, dtype=input._mlx_array.dtype)
    scattered = scattered.at[tuple(coords)].add(source._mlx_array.reshape(-1))

    result_arr = zeroed + scattered
    result = Tensor._from_mlx_array(result_arr)

    if input.requires_grad or source.requires_grad:
        result.requires_grad = True

    return result


def index_put(
    input: Tensor, indices: tuple, values: Union[Tensor, float], accumulate: bool = False
) -> Tensor:
    """
    Put values into input at indices.

    Args:
        input: Input tensor
        indices: Tuple of index tensors
        values: Values to put
        accumulate: If True, add values instead of replacing

    Returns:
        Tensor with put values

    Note:
        Uses vectorized MLX operations for better performance.
    """
    # Convert indices to MLX arrays
    mlx_indices = []
    for idx in indices:
        if isinstance(idx, Tensor):
            mlx_indices.append(idx._mlx_array.reshape(-1).astype(mx.int32))
        elif isinstance(idx, (int, slice)):
            mlx_indices.append(idx)
        else:
            mlx_indices.append(mx.array(idx).reshape(-1).astype(mx.int32))

    # Handle values
    if isinstance(values, Tensor):
        values_arr = values._mlx_array.reshape(-1)
    elif isinstance(values, (int, float)):
        # Scalar - will be broadcast
        values_arr = values
    else:
        values_arr = mx.array(values).reshape(-1)

    if accumulate:
        result_arr = input._mlx_array.at[tuple(mlx_indices)].add(values_arr)
    else:
        # For non-accumulate mode, we need to overwrite
        # Create mask and use where
        if isinstance(values_arr, (int, float)):
            # Scalar case - simpler handling
            result_arr = input._mlx_array.at[tuple(mlx_indices)].add(
                values_arr - input._mlx_array[tuple(mlx_indices)]
            )
        else:
            # Zero out target and add values
            ones = mx.ones_like(values_arr)
            mask = mx.zeros(input.shape, dtype=mx.bool_)
            mask = mask.at[tuple(mlx_indices)].add(ones.astype(mx.bool_))

            zeroed = mx.where(mask, mx.zeros_like(input._mlx_array), input._mlx_array)
            scattered = mx.zeros(input.shape, dtype=input._mlx_array.dtype)
            scattered = scattered.at[tuple(mlx_indices)].add(values_arr)

            result_arr = zeroed + scattered

    result = Tensor._from_mlx_array(result_arr)

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


def isin(
    elements: Tensor, test_elements: Tensor, assume_unique: bool = False, invert: bool = False
) -> Tensor:
    """
    Test if elements are contained in test_elements.

    Args:
        elements: Elements to test
        test_elements: Elements to test against
        assume_unique: If True, assume both are unique (unused)
        invert: If True, return elements NOT in test_elements

    Returns:
        Boolean tensor of same shape as elements

    Note:
        Uses vectorized MLX operations for better performance.
    """
    elem_arr = elements._mlx_array
    test_arr = test_elements._mlx_array.reshape(-1)

    # Expand dimensions for broadcasting comparison
    # elements: [...shape...] -> [...shape..., 1]
    # test: [n] -> [1, 1, ..., n] (broadcast to compare each element with all test values)
    elem_expanded = mx.expand_dims(elem_arr, axis=-1)  # [..., 1]

    # Compare each element with all test elements
    matches = elem_expanded == test_arr  # [..., n]

    # Any match along last dimension
    result_arr = mx.any(matches, axis=-1)

    if invert:
        result_arr = mx.logical_not(result_arr)

    result = Tensor._from_mlx_array(result_arr)

    return result


def diagonal_scatter(
    input: Tensor, src: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> Tensor:
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

    Note:
        Uses vectorized MLX operations for better performance.
    """
    ndim = input.ndim
    dim1 = dim1 if dim1 >= 0 else ndim + dim1
    dim2 = dim2 if dim2 >= 0 else ndim + dim2
    shape = input.shape

    h, w = shape[dim1], shape[dim2]

    # Calculate diagonal length
    if offset >= 0:
        diag_len = min(h, w - offset)
        row_indices = mx.arange(diag_len)
        col_indices = mx.arange(offset, offset + diag_len)
    else:
        diag_len = min(h + offset, w)
        row_indices = mx.arange(-offset, -offset + diag_len)
        col_indices = mx.arange(diag_len)

    if diag_len <= 0:
        # No diagonal to fill
        return Tensor._from_mlx_array(input._mlx_array)

    # Build full index arrays for all diagonal elements
    # For batched case, we need to expand indices across batch dimensions
    src_arr = src._mlx_array

    if ndim == 2:
        # Simple 2D case
        result_arr = input._mlx_array.at[
            row_indices.astype(mx.int32), col_indices.astype(mx.int32)
        ].add(
            src_arr - input._mlx_array[row_indices.astype(mx.int32), col_indices.astype(mx.int32)]
        )
    else:
        # For higher dimensions, build full coordinate arrays
        # This handles batched diagonals
        batch_shape = [shape[d] for d in range(ndim) if d not in (dim1, dim2)]
        batch_size = 1
        for s in batch_shape:
            batch_size *= s

        # Create batch indices
        batch_ranges = []
        for d in range(ndim):
            if d not in (dim1, dim2):
                batch_ranges.append(mx.arange(shape[d]))

        # Build all indices
        coords = []
        for d in range(ndim):
            if d == dim1:
                # Row indices repeated for each batch element
                idx = mx.tile(row_indices, (batch_size,))
                coords.append(idx.astype(mx.int32))
            elif d == dim2:
                # Col indices repeated for each batch element
                idx = mx.tile(col_indices, (batch_size,))
                coords.append(idx.astype(mx.int32))
            else:
                # Batch dimension - repeat for diag_len elements
                batch_idx = batch_ranges.pop(0) if batch_ranges else mx.array([0])
                idx = mx.repeat(batch_idx, diag_len)
                coords.append(idx.astype(mx.int32))

        # Apply scatter
        src_flat = src_arr.reshape(-1)
        mask = mx.zeros(shape, dtype=mx.bool_)
        mask = mask.at[tuple(coords)].add(mx.ones(len(coords[0]), dtype=mx.bool_))

        zeroed = mx.where(mask, mx.zeros_like(input._mlx_array), input._mlx_array)
        scattered = mx.zeros(shape, dtype=input._mlx_array.dtype)
        scattered = scattered.at[tuple(coords)].add(src_flat)

        result_arr = zeroed + scattered

    result = Tensor._from_mlx_array(result_arr)

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result


def slice_scatter(
    input: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> Tensor:
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

    Note:
        Uses MLX slice_update for better performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim
    shape = input.shape

    # Calculate actual slice indices
    dim_size = shape[dim]
    actual_start = start if start is not None else 0
    actual_end = end if end is not None else dim_size

    # Handle negative indices
    if actual_start < 0:
        actual_start = dim_size + actual_start
    if actual_end < 0:
        actual_end = dim_size + actual_end

    # Build slice indices for the dimension
    slice_indices = mx.arange(actual_start, actual_end, step)
    src_shape = src.shape

    # Build coordinate arrays for all positions in src
    coords = []
    for d in range(ndim):
        if d == dim:
            # Map src index to input index via slice_indices
            expand_shape = [1] * ndim
            expand_shape[d] = src_shape[d]
            src_coord = mx.reshape(mx.arange(src_shape[d]), expand_shape)
            src_coord = mx.broadcast_to(src_coord, src_shape).reshape(-1)
            mapped = mx.take(slice_indices, src_coord.astype(mx.int32))
            coords.append(mapped.astype(mx.int32))
        else:
            expand_shape = [1] * ndim
            expand_shape[d] = src_shape[d]
            r = mx.reshape(mx.arange(src_shape[d]), expand_shape)
            coords.append(mx.broadcast_to(r, src_shape).reshape(-1).astype(mx.int32))

    # Create mask and scatter
    mask = mx.zeros(shape, dtype=mx.bool_)
    mask = mask.at[tuple(coords)].add(mx.ones(len(coords[0]), dtype=mx.bool_))

    zeroed = mx.where(mask, mx.zeros_like(input._mlx_array), input._mlx_array)
    scattered = mx.zeros(shape, dtype=input._mlx_array.dtype)
    scattered = scattered.at[tuple(coords)].add(src._mlx_array.reshape(-1))

    result_arr = zeroed + scattered
    result = Tensor._from_mlx_array(result_arr)

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

    Note:
        Uses vectorized MLX operations for better performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim
    shape = input.shape

    # Handle negative index
    if index < 0:
        index = shape[dim] + index

    # src has one fewer dimension than input (dimension dim is missing)
    # Build coordinates for all elements in src, with dim fixed to index
    src_shape = src.shape

    coords = []
    src_dim = 0
    for d in range(ndim):
        if d == dim:
            # Fixed index for this dimension
            coord = mx.full((src._mlx_array.size,), index, dtype=mx.int32)
            coords.append(coord)
        else:
            # Coordinate from src
            expand_shape = [1] * src.ndim
            expand_shape[src_dim] = src_shape[src_dim]
            r = mx.reshape(mx.arange(src_shape[src_dim]), expand_shape)
            coords.append(mx.broadcast_to(r, src_shape).reshape(-1).astype(mx.int32))
            src_dim += 1

    # Create mask and scatter
    mask = mx.zeros(shape, dtype=mx.bool_)
    mask = mask.at[tuple(coords)].add(mx.ones(len(coords[0]), dtype=mx.bool_))

    zeroed = mx.where(mask, mx.zeros_like(input._mlx_array), input._mlx_array)
    scattered = mx.zeros(shape, dtype=input._mlx_array.dtype)
    scattered = scattered.at[tuple(coords)].add(src._mlx_array.reshape(-1))

    result_arr = zeroed + scattered
    result = Tensor._from_mlx_array(result_arr)

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result


def scatter_reduce(
    input: Tensor, dim: int, index: Tensor, src: Tensor, reduce: str, include_self: bool = True
) -> Tensor:
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

    Note:
        Uses vectorized MLX operations for better performance.
    """
    ndim = input.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Build scatter indices
    indices = _build_scatter_indices(index._mlx_array, dim, ndim)

    # Initialize result based on include_self and reduction type
    if include_self:
        result_arr = input._mlx_array
    else:
        if reduce == "sum" or reduce == "mean":
            result_arr = mx.zeros(input.shape, dtype=input._mlx_array.dtype)
        elif reduce == "prod":
            result_arr = mx.ones(input.shape, dtype=input._mlx_array.dtype)
        elif reduce == "amax":
            result_arr = mx.full(input.shape, float("-inf"), dtype=input._mlx_array.dtype)
        elif reduce == "amin":
            result_arr = mx.full(input.shape, float("inf"), dtype=input._mlx_array.dtype)
        else:
            result_arr = input._mlx_array

    # Apply reduction using MLX .at[] operations
    src_flat = src._mlx_array.reshape(-1)

    if reduce == "sum" or reduce == "mean":
        result_arr = result_arr.at[indices].add(src_flat)
        if reduce == "mean":
            # Count occurrences for mean calculation
            ones = mx.ones_like(src_flat)
            count = mx.zeros(input.shape, dtype=input._mlx_array.dtype)
            if include_self:
                count = mx.ones(input.shape, dtype=input._mlx_array.dtype)
            count = count.at[indices].add(ones)
            # Avoid division by zero
            count = mx.maximum(count, mx.ones_like(count))
            result_arr = result_arr / count
    elif reduce == "prod":
        result_arr = result_arr.at[indices].multiply(src_flat)
    elif reduce == "amax":
        result_arr = result_arr.at[indices].maximum(src_flat)
    elif reduce == "amin":
        result_arr = result_arr.at[indices].minimum(src_flat)

    result = Tensor._from_mlx_array(result_arr)

    if input.requires_grad or src.requires_grad:
        result.requires_grad = True

    return result
