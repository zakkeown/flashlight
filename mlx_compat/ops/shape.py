"""
Shape Manipulation Operations

Implements PyTorch-compatible shape manipulation operations with MLX backend.
"""

from typing import Union, Tuple, List, Optional
import mlx.core as mx

from ..tensor import Tensor


def cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int = 0) -> Tensor:
    """
    Concatenate tensors along a dimension.

    Args:
        tensors: Sequence of tensors to concatenate
        dim: Dimension to concatenate along

    Returns:
        Concatenated tensor
    """
    from ..autograd.context import is_grad_enabled

    if not tensors:
        raise ValueError("cat expects at least one tensor")

    mlx_arrays = [t._mlx_array for t in tensors]
    mlx_result = mx.concatenate(mlx_arrays, axis=dim)
    result = Tensor._from_mlx_array(mlx_result)

    # Attach gradient function if any tensor requires grad
    if is_grad_enabled() and any(t.requires_grad for t in tensors):
        result.requires_grad = True
        from ..autograd.function import CatBackward
        grad_fn = CatBackward(list(tensors), dim=dim)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int = 0) -> Tensor:
    """
    Stack tensors along a new dimension.

    Args:
        tensors: Sequence of tensors to stack
        dim: Dimension to insert

    Returns:
        Stacked tensor
    """
    if not tensors:
        raise ValueError("stack expects at least one tensor")

    mlx_arrays = [t._mlx_array for t in tensors]
    mlx_result = mx.stack(mlx_arrays, axis=dim)
    result = Tensor._from_mlx_array(mlx_result)

    if any(t.requires_grad for t in tensors):
        result.requires_grad = True

    return result


def split(tensor: Tensor, split_size_or_sections: Union[int, List[int]], dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Split tensor into chunks.

    Args:
        tensor: Tensor to split
        split_size_or_sections: Size of each chunk or list of sizes
        dim: Dimension to split along

    Returns:
        Tuple of tensors
    """
    if isinstance(split_size_or_sections, int):
        # Split into equal chunks of size split_size_or_sections
        size = split_size_or_sections
        dim_size = tensor.shape[dim]
        num_splits = (dim_size + size - 1) // size  # Ceiling division

        # Create indices for splitting
        indices = [i * size for i in range(1, num_splits)]
        mlx_results = mx.split(tensor._mlx_array, indices, axis=dim)
    else:
        # Split into sections with given sizes
        sections = split_size_or_sections
        indices = []
        cumsum = 0
        for s in sections[:-1]:  # All but last
            cumsum += s
            indices.append(cumsum)
        mlx_results = mx.split(tensor._mlx_array, indices, axis=dim)

    results = tuple(Tensor._from_mlx_array(arr) for arr in mlx_results)

    if tensor.requires_grad:
        for r in results:
            r.requires_grad = True

    return results


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Split tensor into a specific number of chunks.

    Args:
        tensor: Tensor to split
        chunks: Number of chunks
        dim: Dimension to split along

    Returns:
        Tuple of tensors
    """
    dim_size = tensor.shape[dim]
    chunk_size = (dim_size + chunks - 1) // chunks  # Ceiling division

    return split(tensor, chunk_size, dim=dim)


def expand(tensor: Tensor, *sizes: int) -> Tensor:
    """
    Expand tensor to a larger size by broadcasting.

    Args:
        tensor: Input tensor
        *sizes: Desired expanded size

    Returns:
        Expanded tensor (view)
    """
    # Handle tuple argument
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])

    # MLX uses broadcast_to for expansion
    mlx_result = mx.broadcast_to(tensor._mlx_array, sizes)
    result = Tensor._from_mlx_array(mlx_result)

    if tensor.requires_grad:
        result.requires_grad = True

    # Mark as view
    result._is_view = True
    result._base = tensor if tensor._base is None else tensor._base

    return result


def repeat(tensor: Tensor, *sizes: int) -> Tensor:
    """
    Repeat tensor along dimensions.

    Args:
        tensor: Input tensor
        *sizes: Number of repetitions for each dimension

    Returns:
        Repeated tensor
    """
    # Handle tuple argument
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])

    # MLX uses tile for repeating
    mlx_result = mx.tile(tensor._mlx_array, sizes)
    result = Tensor._from_mlx_array(mlx_result)

    if tensor.requires_grad:
        result.requires_grad = True

    return result


def tile(tensor: Tensor, dims: Tuple[int, ...]) -> Tensor:
    """
    Tile tensor (alias for repeat).

    Args:
        tensor: Input tensor
        dims: Number of repetitions for each dimension

    Returns:
        Tiled tensor
    """
    return repeat(tensor, dims)


def repeat_interleave(tensor: Tensor, repeats: Union[int, Tensor], dim: Optional[int] = None) -> Tensor:
    """
    Repeat elements of tensor.

    Args:
        tensor: Input tensor
        repeats: Number of repetitions for each element
        dim: Dimension along which to repeat (None = flatten first)

    Returns:
        Tensor with repeated elements
    """
    if dim is None:
        # Flatten then repeat
        flat = tensor._mlx_array.reshape(-1)
        if isinstance(repeats, Tensor):
            repeats = repeats._mlx_array
        mlx_result = mx.repeat(flat, repeats, axis=0)
    else:
        if isinstance(repeats, Tensor):
            repeats = repeats._mlx_array
        mlx_result = mx.repeat(tensor._mlx_array, repeats, axis=dim)

    result = Tensor._from_mlx_array(mlx_result)

    if tensor.requires_grad:
        result.requires_grad = True

    return result


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Gather values along an axis specified by dim.

    Args:
        input: Source tensor
        dim: Dimension along which to index
        index: Indices to gather

    Returns:
        Gathered tensor
    """
    # MLX uses take_along_axis for gather
    mlx_result = mx.take_along_axis(input._mlx_array, index._mlx_array, axis=dim)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def narrow(tensor: Tensor, dim: int, start: int, length: int) -> Tensor:
    """
    Return a narrowed version of the tensor.

    Args:
        tensor: Input tensor
        dim: Dimension to narrow
        start: Starting index
        length: Length of slice

    Returns:
        Narrowed tensor (view)
    """
    # Create slice for the specified dimension
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, start + length)

    mlx_result = tensor._mlx_array[tuple(slices)]
    result = Tensor._from_mlx_array(mlx_result)

    if tensor.requires_grad:
        result.requires_grad = True

    result._is_view = True
    result._base = tensor if tensor._base is None else tensor._base

    return result


def select(tensor: Tensor, dim: int, index: int) -> Tensor:
    """
    Select a slice along a dimension at the given index.

    Args:
        tensor: Input tensor
        dim: Dimension to select from
        index: Index to select

    Returns:
        Selected tensor with one fewer dimension
    """
    # Create slice that selects the index
    slices = [slice(None)] * tensor.ndim
    slices[dim] = index

    mlx_result = tensor._mlx_array[tuple(slices)]
    result = Tensor._from_mlx_array(mlx_result)

    if tensor.requires_grad:
        result.requires_grad = True

    result._is_view = True
    result._base = tensor if tensor._base is None else tensor._base

    return result


def unbind(tensor: Tensor, dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Remove a dimension and return a tuple of all slices along that dimension.

    Args:
        tensor: Input tensor
        dim: Dimension to unbind

    Returns:
        Tuple of tensors
    """
    # Use split to create individual slices
    size = tensor.shape[dim]
    results = []

    for i in range(size):
        slices = [slice(None)] * tensor.ndim
        slices[dim] = i
        mlx_slice = tensor._mlx_array[tuple(slices)]
        results.append(Tensor._from_mlx_array(mlx_slice))

    if tensor.requires_grad:
        for r in results:
            r.requires_grad = True

    return tuple(results)


def roll(input: Tensor, shifts: Union[int, Tuple[int, ...]], dims: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Roll the tensor along the given dimension(s).

    Elements that roll beyond the last position are re-introduced at the first.

    Args:
        input: Input tensor
        shifts: Number of places to shift. If tuple, must match dims tuple.
        dims: Dimension(s) along which to roll. If None, flatten and roll.

    Returns:
        Rolled tensor

    Example:
        >>> x = mlx_compat.tensor([1, 2, 3, 4, 5])
        >>> mlx_compat.roll(x, 2)
        tensor([4, 5, 1, 2, 3])
    """
    mlx_array = input._mlx_array

    if dims is None:
        # Flatten, roll, reshape back
        original_shape = mlx_array.shape
        flat = mlx_array.reshape(-1)
        rolled = mx.roll(flat, shifts)
        mlx_result = rolled.reshape(original_shape)
    else:
        # Roll along specified dimension(s)
        if isinstance(shifts, int):
            shifts = (shifts,)
        if isinstance(dims, int):
            dims = (dims,)

        mlx_result = mlx_array
        for shift, dim in zip(shifts, dims):
            mlx_result = mx.roll(mlx_result, shift, axis=dim)

    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def flip(input: Tensor, dims: Union[int, Tuple[int, ...], List[int]]) -> Tensor:
    """
    Reverse the order of elements in a tensor along given dimensions.

    Args:
        input: Input tensor
        dims: Dimension(s) to flip

    Returns:
        Flipped tensor

    Example:
        >>> x = mlx_compat.tensor([[1, 2], [3, 4]])
        >>> mlx_compat.flip(x, [0])
        tensor([[3, 4], [1, 2]])
    """
    mlx_array = input._mlx_array

    # Normalize dims to tuple
    if isinstance(dims, int):
        dims = (dims,)
    else:
        dims = tuple(dims)

    # Flip along each dimension
    mlx_result = mlx_array
    for dim in dims:
        # Use slicing to flip: [::-1] along the dimension
        ndim = len(mlx_result.shape)
        dim = dim if dim >= 0 else ndim + dim
        slices = [slice(None)] * ndim
        slices[dim] = slice(None, None, -1)
        mlx_result = mlx_result[tuple(slices)]

    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def fliplr(input: Tensor) -> Tensor:
    """
    Flip tensor along the second dimension (left/right).

    Args:
        input: Input tensor (at least 2D)

    Returns:
        Flipped tensor
    """
    if input.ndim < 2:
        raise RuntimeError("fliplr requires tensor to be at least 2D")
    return flip(input, dims=1)


def flipud(input: Tensor) -> Tensor:
    """
    Flip tensor along the first dimension (up/down).

    Args:
        input: Input tensor (at least 1D)

    Returns:
        Flipped tensor
    """
    if input.ndim < 1:
        raise RuntimeError("flipud requires tensor to be at least 1D")
    return flip(input, dims=0)


def rot90(input: Tensor, k: int = 1, dims: Tuple[int, int] = (0, 1)) -> Tensor:
    """
    Rotate tensor by 90 degrees in the plane specified by dims.

    Rotation direction is from first to second axis for k > 0.

    Args:
        input: Input tensor (at least 2D)
        k: Number of 90-degree rotations
        dims: Plane of rotation as (dim1, dim2)

    Returns:
        Rotated tensor

    Example:
        >>> x = mlx_compat.tensor([[1, 2], [3, 4]])
        >>> mlx_compat.rot90(x, 1)
        tensor([[2, 4], [1, 3]])
    """
    mlx_array = input._mlx_array
    ndim = len(mlx_array.shape)

    # Normalize k to 0-3
    k = k % 4

    if k == 0:
        result = Tensor._from_mlx_array(mlx_array)
    else:
        dim1, dim2 = dims
        dim1 = dim1 if dim1 >= 0 else ndim + dim1
        dim2 = dim2 if dim2 >= 0 else ndim + dim2

        # Build transpose permutation to swap dim1 and dim2
        perm = list(range(ndim))
        perm[dim1], perm[dim2] = perm[dim2], perm[dim1]

        # Helper for flip
        def flip_axis(arr, axis):
            slices = [slice(None)] * ndim
            slices[axis] = slice(None, None, -1)
            return arr[tuple(slices)]

        if k == 1:
            # 90 degrees: transpose and flip dim1
            mlx_result = mx.transpose(mlx_array, perm)
            mlx_result = flip_axis(mlx_result, dim1)
        elif k == 2:
            # 180 degrees: flip both dims
            mlx_result = flip_axis(mlx_array, dim1)
            mlx_result = flip_axis(mlx_result, dim2)
        elif k == 3:
            # 270 degrees: transpose and flip dim2
            mlx_result = mx.transpose(mlx_array, perm)
            mlx_result = flip_axis(mlx_result, dim2)

        result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# Extended Shape Operations (Sprint 4)
# =============================================================================

def ravel(input: Tensor) -> Tensor:
    """
    Return a contiguous flattened tensor.

    Args:
        input: Input tensor

    Returns:
        1D tensor
    """
    mlx_result = input._mlx_array.reshape(-1)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def t(input: Tensor) -> Tensor:
    """
    Transpose a 2D tensor.

    Args:
        input: Input tensor (2D or less)

    Returns:
        Transposed tensor
    """
    if input.ndim > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but got {input.ndim}")

    if input.ndim < 2:
        return input  # No-op for 0D and 1D tensors

    mlx_result = mx.transpose(input._mlx_array)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def adjoint(input: Tensor) -> Tensor:
    """
    Return the conjugate transpose (Hermitian transpose).

    Args:
        input: Input tensor

    Returns:
        Conjugate transposed tensor
    """
    # Transpose last two dimensions
    ndim = input.ndim
    if ndim < 2:
        # For scalars/1D, just return as-is (consistent with PyTorch)
        mlx_result = input._mlx_array
    else:
        # Build permutation to swap last two dimensions
        perm = list(range(ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        mlx_result = mx.transpose(input._mlx_array, perm)
        # Conjugate (no-op for real tensors, MLX handles complex)
        mlx_result = mx.conj(mlx_result)

    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def moveaxis(input: Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> Tensor:
    """
    Move axes of a tensor to new positions.

    Args:
        input: Input tensor
        source: Original positions of axes to move
        destination: Destination positions

    Returns:
        Tensor with moved axes
    """
    mlx_result = mx.moveaxis(input._mlx_array, source, destination)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def swapaxes(input: Tensor, axis0: int, axis1: int) -> Tensor:
    """
    Interchange two axes of a tensor.

    Args:
        input: Input tensor
        axis0: First axis
        axis1: Second axis

    Returns:
        Tensor with swapped axes
    """
    mlx_result = mx.swapaxes(input._mlx_array, axis0, axis1)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    result._is_view = True
    result._base = input if input._base is None else input._base

    return result


def hstack(tensors: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
    """
    Stack tensors horizontally (column-wise).

    For 1D tensors: concatenate along axis 0
    For 2D+ tensors: concatenate along axis 1

    Args:
        tensors: Sequence of tensors to stack

    Returns:
        Horizontally stacked tensor
    """
    if not tensors:
        raise ValueError("hstack expects at least one tensor")

    # Check dimensions
    first_ndim = tensors[0].ndim

    if first_ndim == 0:
        # Scalars: stack them into 1D
        return stack(tensors, dim=0)
    elif first_ndim == 1:
        # 1D tensors: concatenate along axis 0
        return cat(tensors, dim=0)
    else:
        # 2D+ tensors: concatenate along axis 1
        return cat(tensors, dim=1)


def vstack(tensors: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
    """
    Stack tensors vertically (row-wise).

    For 1D tensors: first reshape to (1, N), then concatenate along axis 0
    For 2D+ tensors: concatenate along axis 0

    Args:
        tensors: Sequence of tensors to stack

    Returns:
        Vertically stacked tensor
    """
    if not tensors:
        raise ValueError("vstack expects at least one tensor")

    first_ndim = tensors[0].ndim

    if first_ndim == 0:
        # Scalars: reshape to (1,) and stack
        reshaped = [Tensor._from_mlx_array(t._mlx_array.reshape(1)) for t in tensors]
        return cat(reshaped, dim=0)
    elif first_ndim == 1:
        # 1D tensors: reshape to (1, N) then concatenate along axis 0
        reshaped = [Tensor._from_mlx_array(t._mlx_array.reshape(1, -1)) for t in tensors]
        return cat(reshaped, dim=0)
    else:
        # 2D+ tensors: concatenate along axis 0
        return cat(tensors, dim=0)


def dstack(tensors: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
    """
    Stack tensors depth-wise (along third axis).

    Args:
        tensors: Sequence of tensors to stack

    Returns:
        Depth-stacked tensor
    """
    if not tensors:
        raise ValueError("dstack expects at least one tensor")

    first_ndim = tensors[0].ndim

    if first_ndim == 1:
        # 1D tensors: reshape to (1, N, 1) then concatenate
        reshaped = [Tensor._from_mlx_array(t._mlx_array.reshape(1, -1, 1)) for t in tensors]
        return cat(reshaped, dim=2)
    elif first_ndim == 2:
        # 2D tensors: reshape to (H, W, 1) then concatenate
        reshaped = [Tensor._from_mlx_array(mx.expand_dims(t._mlx_array, axis=-1)) for t in tensors]
        return cat(reshaped, dim=2)
    else:
        # 3D+ tensors: concatenate along axis 2
        return cat(tensors, dim=2)


def column_stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
    """
    Stack 1D tensors as columns into a 2D tensor.

    Args:
        tensors: Sequence of tensors

    Returns:
        2D tensor with each input as a column
    """
    if not tensors:
        raise ValueError("column_stack expects at least one tensor")

    # For 1D tensors, reshape to column vectors
    processed = []
    for t in tensors:
        if t.ndim == 1:
            processed.append(Tensor._from_mlx_array(t._mlx_array.reshape(-1, 1)))
        else:
            processed.append(t)

    return cat(processed, dim=1)


def row_stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]]) -> Tensor:
    """
    Stack tensors row-wise (alias for vstack).

    Args:
        tensors: Sequence of tensors

    Returns:
        Vertically stacked tensor
    """
    return vstack(tensors)


def hsplit(tensor: Tensor, indices_or_sections: Union[int, List[int]]) -> Tuple[Tensor, ...]:
    """
    Split tensor horizontally (column-wise).

    Unlike split() which uses chunk size, hsplit uses number of sections.

    Args:
        tensor: Input tensor
        indices_or_sections: Number of sections or split indices

    Returns:
        Tuple of split tensors
    """
    if tensor.ndim == 1:
        return tensor_split(tensor, indices_or_sections, dim=0)
    else:
        return tensor_split(tensor, indices_or_sections, dim=1)


def vsplit(tensor: Tensor, indices_or_sections: Union[int, List[int]]) -> Tuple[Tensor, ...]:
    """
    Split tensor vertically (row-wise).

    Unlike split() which uses chunk size, vsplit uses number of sections.

    Args:
        tensor: Input tensor
        indices_or_sections: Number of sections or split indices

    Returns:
        Tuple of split tensors
    """
    return tensor_split(tensor, indices_or_sections, dim=0)


def dsplit(tensor: Tensor, indices_or_sections: Union[int, List[int]]) -> Tuple[Tensor, ...]:
    """
    Split tensor depth-wise (along third dimension).

    Unlike split() which uses chunk size, dsplit uses number of sections.

    Args:
        tensor: Input tensor
        indices_or_sections: Number of sections or split indices

    Returns:
        Tuple of split tensors
    """
    if tensor.ndim < 3:
        raise RuntimeError(f"dsplit requires tensor with at least 3 dimensions, got {tensor.ndim}")
    return tensor_split(tensor, indices_or_sections, dim=2)


def tensor_split(input: Tensor, indices_or_sections: Union[int, List[int], Tensor], dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Split a tensor into multiple sub-tensors.

    Args:
        input: Input tensor
        indices_or_sections: Number of sections or split indices
        dim: Dimension along which to split

    Returns:
        Tuple of split tensors
    """
    if isinstance(indices_or_sections, Tensor):
        indices_or_sections = indices_or_sections.tolist()

    if isinstance(indices_or_sections, int):
        # Split into N equal parts
        n = indices_or_sections
        dim_size = input.shape[dim]
        # Calculate split sizes
        base_size = dim_size // n
        remainder = dim_size % n
        # First 'remainder' sections get one extra element
        sizes = [base_size + (1 if i < remainder else 0) for i in range(n)]
        return split(input, sizes, dim=dim)
    else:
        # indices_or_sections is a list of indices
        indices = list(indices_or_sections)
        return split(input, indices, dim=dim)


def block_diag(*tensors: Tensor) -> Tensor:
    """
    Create a block diagonal matrix from provided tensors.

    Pure MLX implementation.

    Args:
        *tensors: 2D tensors to form blocks

    Returns:
        Block diagonal matrix
    """
    if not tensors:
        return Tensor._from_mlx_array(mx.array([]))

    # Ensure all inputs are 2D MLX arrays
    processed = []
    for t in tensors:
        arr = t._mlx_array
        if arr.ndim == 0:
            processed.append(mx.reshape(arr, (1, 1)))
        elif arr.ndim == 1:
            processed.append(mx.reshape(arr, (1, -1)))
        else:
            processed.append(arr)

    # Calculate total dimensions
    total_rows = sum(p.shape[0] for p in processed)
    total_cols = sum(p.shape[1] for p in processed)

    # Get common dtype
    dtype = processed[0].dtype

    # Build result by creating padded blocks and concatenating
    # For each block, pad with zeros to align with its position in the result
    row_offset = 0
    col_offset = 0
    padded_blocks = []

    for i, p in enumerate(processed):
        rows, cols = p.shape
        # Create left padding (zeros before this block's columns)
        left_pad = mx.zeros((rows, col_offset), dtype=dtype) if col_offset > 0 else None
        # Create right padding (zeros after this block's columns)
        right_cols = total_cols - col_offset - cols
        right_pad = mx.zeros((rows, right_cols), dtype=dtype) if right_cols > 0 else None

        # Concatenate horizontally: [left_zeros | block | right_zeros]
        row_parts = []
        if left_pad is not None:
            row_parts.append(left_pad)
        row_parts.append(p.astype(dtype))
        if right_pad is not None:
            row_parts.append(right_pad)

        if len(row_parts) == 1:
            padded_row = row_parts[0]
        else:
            padded_row = mx.concatenate(row_parts, axis=1)

        padded_blocks.append(padded_row)
        row_offset += rows
        col_offset += cols

    # Concatenate all padded blocks vertically
    result = mx.concatenate(padded_blocks, axis=0)
    result_tensor = Tensor._from_mlx_array(result)

    if any(t.requires_grad for t in tensors):
        result_tensor.requires_grad = True

    return result_tensor


def diag_embed(input: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    """
    Create a tensor whose diagonals of certain 2D planes are filled by input.

    Pure MLX implementation.

    Args:
        input: Input tensor (last dimension becomes diagonal)
        offset: Diagonal offset (0 = main diagonal)
        dim1: First dimension of 2D planes
        dim2: Second dimension of 2D planes

    Returns:
        Tensor with embedded diagonals
    """
    arr = input._mlx_array
    input_shape = arr.shape
    diag_size = input_shape[-1]

    # Calculate output size based on offset
    out_size = diag_size + abs(offset)

    # Build output shape
    out_shape = list(input_shape[:-1]) + [out_size, out_size]

    # Create output filled with zeros
    dtype = arr.dtype
    result = mx.zeros(out_shape, dtype=dtype)

    # For 1D input (simple case), directly build diagonal matrix
    if arr.ndim == 1:
        # Create diagonal indices
        diag_indices = mx.arange(diag_size)
        if offset >= 0:
            row_indices = diag_indices
            col_indices = diag_indices + offset
        else:
            row_indices = diag_indices - offset
            col_indices = diag_indices

        # Build diagonal matrix using scatter-like approach
        # Create a full index array and use broadcasting
        rows = mx.arange(out_size)[:, None]  # (out_size, 1)
        cols = mx.arange(out_size)[None, :]  # (1, out_size)

        # For each diagonal element, check if (row, col) matches
        for i in range(diag_size):
            r = int(row_indices[i].item())
            c = int(col_indices[i].item())
            mask = (rows == r) & (cols == c)
            result = mx.where(mask, arr[i], result)
    else:
        # For higher dimensional inputs, flatten batch dims, process, reshape
        batch_shape = input_shape[:-1]
        batch_size = 1
        for s in batch_shape:
            batch_size *= s

        # Reshape to (batch_size, diag_size)
        flat_input = mx.reshape(arr, (batch_size, diag_size))

        # Create result as (batch_size, out_size, out_size)
        flat_result = mx.zeros((batch_size, out_size, out_size), dtype=dtype)

        # Create diagonal indices
        diag_indices = mx.arange(diag_size)
        if offset >= 0:
            row_indices = diag_indices
            col_indices = diag_indices + offset
        else:
            row_indices = diag_indices - offset
            col_indices = diag_indices

        # Fill diagonals for each batch element
        rows = mx.arange(out_size)[:, None]  # (out_size, 1)
        cols = mx.arange(out_size)[None, :]  # (1, out_size)

        for b in range(batch_size):
            batch_result = mx.zeros((out_size, out_size), dtype=dtype)
            for i in range(diag_size):
                r = int(row_indices[i].item())
                c = int(col_indices[i].item())
                mask = (rows == r) & (cols == c)
                batch_result = mx.where(mask, flat_input[b, i], batch_result)
            # Update the batch slice
            flat_result = mx.concatenate([
                flat_result[:b],
                mx.expand_dims(batch_result, axis=0),
                flat_result[b+1:]
            ], axis=0) if batch_size > 1 else mx.expand_dims(batch_result, axis=0)

        # Reshape back to output shape
        result = mx.reshape(flat_result, out_shape)

    result_tensor = Tensor._from_mlx_array(result)

    if input.requires_grad:
        result_tensor.requires_grad = True

    return result_tensor


def diagflat(input: Tensor, offset: int = 0) -> Tensor:
    """
    Create a 2D tensor with the flattened input as a diagonal.

    Args:
        input: Input tensor (will be flattened)
        offset: Diagonal offset (0 = main diagonal)

    Returns:
        2D tensor with diagonal filled
    """
    # Flatten and use diag_embed
    flat = ravel(input)
    return diag_embed(flat, offset=offset)


# =============================================================================
# Additional shape operations
# =============================================================================

# Aliases for PyTorch compatibility
movedim = moveaxis
swapdims = swapaxes


def broadcast_tensors(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """
    Broadcast tensors to a common shape.

    Pure MLX implementation.

    Args:
        *tensors: Input tensors

    Returns:
        Tuple of broadcast tensors
    """
    if not tensors:
        return ()

    # Get shapes
    shapes = [t.shape for t in tensors]

    # Calculate broadcast shape
    max_dims = max(len(s) for s in shapes)
    padded_shapes = [([1] * (max_dims - len(s))) + list(s) for s in shapes]

    broadcast_shape = []
    for dim_sizes in zip(*padded_shapes):
        max_size = 1
        for size in dim_sizes:
            if size != 1:
                if max_size != 1 and max_size != size:
                    raise RuntimeError(f"Shape mismatch: cannot broadcast shapes")
                max_size = size
        broadcast_shape.append(max_size)

    # Broadcast each tensor
    results = []
    for tensor in tensors:
        result_array = mx.broadcast_to(tensor._mlx_array, broadcast_shape)
        result = Tensor._from_mlx_array(result_array)
        if tensor.requires_grad:
            result.requires_grad = True
        results.append(result)

    return tuple(results)


def unflatten(input: Tensor, dim: int, sizes: Tuple[int, ...]) -> Tensor:
    """
    Unflatten a dimension into multiple dimensions.

    Args:
        input: Input tensor
        dim: Dimension to unflatten
        sizes: New sizes for the unflattened dimensions

    Returns:
        Unflattened tensor
    """
    shape = list(input.shape)
    ndim = len(shape)

    # Normalize dimension
    if dim < 0:
        dim = ndim + dim

    # Validate
    if shape[dim] != -1 and shape[dim] != prod(sizes):
        if -1 not in sizes:
            raise RuntimeError(f"Unflatten size mismatch: {shape[dim]} vs {prod(sizes)}")

    # Build new shape
    new_shape = shape[:dim] + list(sizes) + shape[dim+1:]

    result_array = mx.reshape(input._mlx_array, new_shape)
    result = Tensor._from_mlx_array(result_array)

    if input.requires_grad:
        result.requires_grad = True

    return result


def prod(iterable):
    """Helper to compute product of iterable."""
    result = 1
    for x in iterable:
        if x != -1:
            result *= x
    return result


def concat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int = 0) -> Tensor:
    """
    Alias for cat.

    Concatenate tensors along a dimension.

    Args:
        tensors: Sequence of tensors to concatenate
        dim: Dimension to concatenate along

    Returns:
        Concatenated tensor
    """
    return cat(tensors, dim)


def combinations(input: Tensor, r: int = 2, with_replacement: bool = False) -> Tensor:
    """
    Compute r-combinations of input tensor elements.

    Pure MLX implementation (using itertools for index generation only).

    Args:
        input: 1-D input tensor
        r: Number of elements in each combination
        with_replacement: Allow repeating elements

    Returns:
        Tensor of combinations
    """
    from itertools import combinations as iter_combinations, combinations_with_replacement

    arr = input._mlx_array
    n = arr.shape[0]

    if with_replacement:
        combs = list(combinations_with_replacement(range(n), r))
    else:
        combs = list(iter_combinations(range(n), r))

    if not combs:
        # Return empty tensor with correct shape
        return Tensor._from_mlx_array(mx.zeros((0, r), dtype=arr.dtype))

    # Convert combination indices to MLX array and gather
    indices = mx.array(combs, dtype=mx.int32)  # Shape: (num_combs, r)

    # Gather values from input array using the indices
    result = mx.take(arr, indices.reshape(-1), axis=0)
    result = mx.reshape(result, (len(combs), r))

    return Tensor._from_mlx_array(result)


def split_with_sizes(tensor: Tensor, split_sizes: List[int], dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Split tensor according to specified sizes.

    Args:
        tensor: Tensor to split
        split_sizes: List of sizes for each split
        dim: Dimension to split along

    Returns:
        Tuple of tensors
    """
    return split(tensor, split_sizes, dim)


def unsafe_chunk(tensor: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Like chunk but may return less than `chunks` pieces.

    Args:
        tensor: Tensor to chunk
        chunks: Number of chunks
        dim: Dimension to chunk along

    Returns:
        Tuple of tensors
    """
    return chunk(tensor, chunks, dim)


def unsafe_split(tensor: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Like split but may return pieces of different sizes.

    Args:
        tensor: Tensor to split
        split_size: Size of each split
        dim: Dimension to split along

    Returns:
        Tuple of tensors
    """
    return split(tensor, split_size, dim)


def unsafe_split_with_sizes(tensor: Tensor, split_sizes: List[int], dim: int = 0) -> Tuple[Tensor, ...]:
    """
    Like split_with_sizes but may return empty tensors.

    Args:
        tensor: Tensor to split
        split_sizes: List of sizes for each split
        dim: Dimension to split along

    Returns:
        Tuple of tensors
    """
    return split(tensor, split_sizes, dim)


__all__ = [
    'cat', 'stack', 'split', 'chunk',
    'expand', 'repeat', 'tile', 'repeat_interleave',
    'gather', 'narrow', 'select', 'unbind',
    'roll', 'flip', 'fliplr', 'flipud', 'rot90',
    'ravel', 't', 'adjoint', 'moveaxis', 'swapaxes',
    'hstack', 'vstack', 'dstack', 'column_stack', 'row_stack',
    'hsplit', 'vsplit', 'dsplit', 'tensor_split',
    'block_diag', 'diag_embed', 'diagflat',
    # Additional shape ops
    'movedim', 'swapdims',
    'broadcast_tensors', 'unflatten', 'concat', 'combinations',
    'split_with_sizes', 'unsafe_chunk', 'unsafe_split', 'unsafe_split_with_sizes',
]
