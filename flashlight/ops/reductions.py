"""
Reduction Operations

Implements PyTorch-compatible reduction operations with MLX backend.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..autograd.context import is_grad_enabled
from ..autograd.function import MaxBackward, MeanBackward, SumBackward
from ..tensor import Tensor


def sum(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    """
    Sum of all elements or along dimensions.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    if dim is not None:
        # Convert dim to axis for MLX
        axis = dim if isinstance(dim, (tuple, list)) else dim
        mlx_result = mx.sum(input._mlx_array, axis=axis, keepdims=keepdim)
    else:
        mlx_result = mx.sum(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = SumBackward(input, dim, keepdim)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def mean(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    """
    Mean of all elements or along dimensions.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        mlx_result = mx.mean(input._mlx_array, axis=axis, keepdims=keepdim)
    else:
        mlx_result = mx.mean(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)

    # Autograd graph construction
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
        grad_fn = MeanBackward(input, dim, keepdim)
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def max(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Union[Tensor, "MaxResult"]:
    """
    Maximum value(s) of tensor.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        If dim is None: single tensor with max value
        If dim is a tuple: single tensor with max values (no indices for multi-axis)
        If dim is a single int: MaxResult with lazy .values and .indices properties
    """
    from ._reduction_results import MaxResult

    if dim is None:
        # Return single max value
        mlx_result = mx.max(input._mlx_array, keepdims=keepdim)
        result = Tensor._from_mlx_array(mlx_result)

        # Autograd graph construction
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True
            grad_fn = MaxBackward(input, dim=None, keepdim=keepdim)
            grad_fn.output_tensor = result
            result._grad_fn = grad_fn

        return result

    # Handle tuple axis - MLX argmax only supports single int axis
    if isinstance(dim, (tuple, list)):
        mlx_values = mx.max(input._mlx_array, axis=dim, keepdims=keepdim)
        result = Tensor._from_mlx_array(mlx_values)

        # Autograd graph construction for tuple dim
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True
            # For tuple dim, we still use MaxBackward but it will handle each dim
            grad_fn = MaxBackward(input, dim=dim, keepdim=keepdim)
            grad_fn.output_tensor = result
            result._grad_fn = grad_fn

        return result

    # Single axis - return lazy (values, indices) result with gradient tracking
    return MaxResult(input, dim, keepdim)


def min(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Union[Tensor, "MinResult"]:
    """
    Minimum value(s) of tensor.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        If dim is None: single tensor with min value
        If dim is a tuple: single tensor with min values (no indices for multi-axis)
        If dim is a single int: MinResult with lazy .values and .indices properties
    """
    from ._reduction_results import MinResult

    if dim is None:
        # Return single min value
        mlx_result = mx.min(input._mlx_array, keepdims=keepdim)
        result = Tensor._from_mlx_array(mlx_result)

        if input.requires_grad:
            result.requires_grad = True

        return result

    # Handle tuple axis - MLX argmin only supports single int axis
    if isinstance(dim, (tuple, list)):
        mlx_values = mx.min(input._mlx_array, axis=dim, keepdims=keepdim)
        result = Tensor._from_mlx_array(mlx_values)

        if input.requires_grad:
            result.requires_grad = True

        return result

    # Single axis - return lazy (values, indices) result
    return MinResult(input._mlx_array, dim, keepdim)


def argmax(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Indices of maximum values.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = flatten first)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor of indices
    """
    if dim is None:
        # Flatten and find argmax
        mlx_result = mx.argmax(input._mlx_array, keepdims=keepdim)
    else:
        mlx_result = mx.argmax(input._mlx_array, axis=dim, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)
    return result


def argmin(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Indices of minimum values.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = flatten first)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor of indices
    """
    if dim is None:
        # Flatten and find argmin
        mlx_result = mx.argmin(input._mlx_array, keepdims=keepdim)
    else:
        mlx_result = mx.argmin(input._mlx_array, axis=dim, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)
    return result


def var(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
) -> Tensor:
    """
    Variance of elements.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions
        unbiased: Whether to use Bessel's correction (ddof=1)

    Returns:
        Result tensor
    """
    ddof = 1 if unbiased else 0

    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        mlx_result = mx.var(input._mlx_array, axis=axis, keepdims=keepdim, ddof=ddof)
    else:
        mlx_result = mx.var(input._mlx_array, keepdims=keepdim, ddof=ddof)

    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def std(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
) -> Tensor:
    """
    Standard deviation of elements.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions
        unbiased: Whether to use Bessel's correction (ddof=1)

    Returns:
        Result tensor
    """
    # Optimized: compute sqrt(var) directly on MLX arrays without intermediate Tensor
    ddof = 1 if unbiased else 0

    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        variance = mx.var(input._mlx_array, axis=axis, keepdims=keepdim, ddof=ddof)
    else:
        variance = mx.var(input._mlx_array, keepdims=keepdim, ddof=ddof)

    mlx_result = mx.sqrt(variance)
    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def prod(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Product of all elements or along a dimension.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    if dim is not None:
        mlx_result = mx.prod(input._mlx_array, axis=dim, keepdims=keepdim)
    else:
        mlx_result = mx.prod(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)

    if input.requires_grad:
        result.requires_grad = True

    return result


def all(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Test if all elements are True.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Boolean tensor
    """
    if dim is not None:
        mlx_result = mx.all(input._mlx_array, axis=dim, keepdims=keepdim)
    else:
        mlx_result = mx.all(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)
    return result


def any(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """
    Test if any element is True.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Boolean tensor
    """
    if dim is not None:
        mlx_result = mx.any(input._mlx_array, axis=dim, keepdims=keepdim)
    else:
        mlx_result = mx.any(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)
    return result


def amax(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    """
    Maximum values along dimensions (without returning indices).

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor with maximum values
    """
    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        mlx_result = mx.max(input._mlx_array, axis=axis, keepdims=keepdim)
    else:
        mlx_result = mx.max(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def amin(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    """
    Minimum values along dimensions (without returning indices).

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor with minimum values
    """
    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        mlx_result = mx.min(input._mlx_array, axis=axis, keepdims=keepdim)
    else:
        mlx_result = mx.min(input._mlx_array, keepdims=keepdim)

    result = Tensor._from_mlx_array(mlx_result)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def aminmax(
    input: Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Compute minimum and maximum values along a dimension simultaneously.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tuple of (min_values, max_values)
    """
    min_val = amin(input, dim=dim, keepdim=keepdim)
    max_val = amax(input, dim=dim, keepdim=keepdim)
    return min_val, max_val


# =============================================================================
# Extended Reduction Operations (Sprint 5)
# =============================================================================


def median(
    input: Tensor, dim: Optional[int] = None, keepdim: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Compute median values.

    PyTorch behavior:
    - For even-length arrays, returns the LOWER of the two middle values (not average)
    - When dim is specified, also returns the index of the median value

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        If dim is None: single tensor with median value
        If dim is specified: tuple of (values, indices)
    """
    arr = input._mlx_array

    if dim is None:
        # Flatten and find the lower-middle element (PyTorch behavior)
        flat = arr.flatten()
        sorted_flat = mx.sort(flat)
        n = flat.shape[0]
        # For even n, PyTorch uses (n-1)//2 which gives the lower middle
        mid_idx = (n - 1) // 2
        result = Tensor._from_mlx_array(sorted_flat[mid_idx])

        if input.requires_grad:
            result.requires_grad = True

        return result
    else:
        ndim = arr.ndim
        dim = dim if dim >= 0 else ndim + dim

        # Sort along dimension
        sorted_arr = mx.sort(arr, axis=dim)
        sorted_indices = mx.argsort(arr, axis=dim)

        n = arr.shape[dim]
        mid_idx = (n - 1) // 2

        # Build slices to extract the mid_idx element along dim
        slices = [slice(None)] * ndim
        slices[dim] = mid_idx

        values_mlx = sorted_arr[tuple(slices)]
        indices_mlx = sorted_indices[tuple(slices)]

        if keepdim:
            values_mlx = mx.expand_dims(values_mlx, axis=dim)
            indices_mlx = mx.expand_dims(indices_mlx, axis=dim)

        values = Tensor._from_mlx_array(values_mlx)
        indices = Tensor._from_mlx_array(indices_mlx)

        if input.requires_grad:
            values.requires_grad = True

        return values, indices


def mode(input: Tensor, dim: int = -1, keepdim: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Compute mode (most common value) along a dimension.

    PyTorch behavior:
    - Returns the smallest value among those with the highest frequency (when tied)
    - For n >= 24: returns FIRST index of the mode value
    - For n < 24: returns LAST index of the mode value
      (This matches PyTorch's internal algorithm switching behavior)

    Args:
        input: Input tensor
        dim: Dimension to reduce
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tuple of (values, indices)
    """
    arr = input._mlx_array
    ndim = arr.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Sort the array and get sort indices to track original positions
    sorted_arr = mx.sort(arr, axis=dim)
    sorted_indices = mx.argsort(arr, axis=dim)

    n = arr.shape[dim]

    # PyTorch threshold: n >= 24 returns first index, n < 24 returns last index
    use_first_index = n >= 24

    # For mode, we need to count runs of consecutive equal values in sorted array
    # This is inherently sequential, but we can use MLX ops within a Python loop
    # over the outer dimensions

    # Move target dimension to last axis for easier processing
    sorted_arr = mx.moveaxis(sorted_arr, dim, -1)
    sorted_indices = mx.moveaxis(sorted_indices, dim, -1)

    outer_shape = sorted_arr.shape[:-1]

    # Flatten outer dimensions for iteration
    if outer_shape:
        flat_outer_size = 1
        for s in outer_shape:
            flat_outer_size *= s
        sorted_arr_flat = sorted_arr.reshape(flat_outer_size, n)
        sorted_indices_flat = sorted_indices.reshape(flat_outer_size, n)
    else:
        sorted_arr_flat = sorted_arr.reshape(1, n)
        sorted_indices_flat = sorted_indices.reshape(1, n)

    # Process each slice
    values_list = []
    indices_list = []

    for i in range(sorted_arr_flat.shape[0]):
        slice_sorted = sorted_arr_flat[i]
        slice_indices = sorted_indices_flat[i]

        # Find run lengths using diff
        # Where diff != 0, a new run starts
        # Use pure MLX operations as much as possible

        # Get values as Python list for sequential processing (required for run-length)
        slice_sorted_list = slice_sorted.tolist()
        slice_indices_list = slice_indices.tolist()

        best_value = slice_sorted_list[0]
        best_count = 1
        best_orig_idx = slice_indices_list[0]

        current_value = slice_sorted_list[0]
        current_count = 1
        current_first_idx = slice_indices_list[0]
        current_last_idx = slice_indices_list[0]

        for j in range(1, n):
            if slice_sorted_list[j] == current_value:
                current_count += 1
                current_last_idx = slice_indices_list[j]
            else:
                if current_count > best_count or (
                    current_count == best_count and current_value < best_value
                ):
                    best_value = current_value
                    best_count = current_count
                    best_orig_idx = current_first_idx if use_first_index else current_last_idx
                current_value = slice_sorted_list[j]
                current_count = 1
                current_first_idx = slice_indices_list[j]
                current_last_idx = slice_indices_list[j]

        # Check the last run
        if current_count > best_count or (
            current_count == best_count and current_value < best_value
        ):
            best_value = current_value
            best_orig_idx = current_first_idx if use_first_index else current_last_idx

        values_list.append(best_value)
        indices_list.append(best_orig_idx)

    # Convert back to MLX arrays
    values_mlx = mx.array(values_list, dtype=arr.dtype)
    indices_mlx = mx.array(indices_list, dtype=mx.int64)

    # Reshape to outer_shape
    if outer_shape:
        values_mlx = values_mlx.reshape(outer_shape)
        indices_mlx = indices_mlx.reshape(outer_shape)
    else:
        values_mlx = values_mlx.reshape(())
        indices_mlx = indices_mlx.reshape(())

    if keepdim:
        values_mlx = mx.expand_dims(values_mlx, axis=dim)
        indices_mlx = mx.expand_dims(indices_mlx, axis=dim)

    values = Tensor._from_mlx_array(values_mlx)
    indices = Tensor._from_mlx_array(indices_mlx)

    if input.requires_grad:
        values.requires_grad = True

    return values, indices


def quantile(
    input: Tensor, q: Union[float, Tensor], dim: Optional[int] = None, keepdim: bool = False
) -> Tensor:
    """
    Compute quantiles using linear interpolation.

    Args:
        input: Input tensor
        q: Quantile(s) to compute (0 to 1)
        dim: Dimension to reduce (None = flatten)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor with quantile values
    """
    arr = input._mlx_array

    if isinstance(q, Tensor):
        q_val = q._mlx_array
    else:
        q_val = mx.array(q, dtype=mx.float32)

    # Ensure q is at least 1D for consistent processing
    q_scalar = q_val.ndim == 0
    if q_scalar:
        q_val = q_val.reshape(1)

    if dim is None:
        # Flatten and compute quantile
        flat = arr.flatten()
        sorted_flat = mx.sort(flat)
        n = flat.shape[0]

        # Compute indices using linear interpolation
        # virtual_index = q * (n - 1)
        virtual_index = q_val * (n - 1)

        # Get floor and ceil indices
        lower_idx = mx.floor(virtual_index).astype(mx.int32)
        upper_idx = mx.minimum(lower_idx + 1, mx.array(n - 1, dtype=mx.int32))

        # Interpolation weight
        frac = virtual_index - lower_idx.astype(mx.float32)

        # Get values at indices and interpolate
        lower_vals = mx.take(sorted_flat, lower_idx)
        upper_vals = mx.take(sorted_flat, upper_idx)

        result_mlx = lower_vals + frac * (upper_vals - lower_vals)

        if q_scalar:
            result_mlx = result_mlx.squeeze()

        result = Tensor._from_mlx_array(result_mlx)
    else:
        ndim = arr.ndim
        dim = dim if dim >= 0 else ndim + dim

        # Sort along dimension
        sorted_arr = mx.sort(arr, axis=dim)
        n = arr.shape[dim]

        # For each quantile value, compute the result
        results = []
        for i in range(q_val.shape[0]):
            q_i = q_val[i]
            virtual_index = q_i * (n - 1)

            lower_idx = int(mx.floor(virtual_index).item())
            upper_idx = min(lower_idx + 1, n - 1)
            frac = float(virtual_index.item()) - lower_idx

            # Extract slices at lower and upper indices
            slices_lower = [slice(None)] * ndim
            slices_lower[dim] = lower_idx
            slices_upper = [slice(None)] * ndim
            slices_upper[dim] = upper_idx

            lower_vals = sorted_arr[tuple(slices_lower)]
            upper_vals = sorted_arr[tuple(slices_upper)]

            result_i = lower_vals + frac * (upper_vals - lower_vals)

            if keepdim:
                result_i = mx.expand_dims(result_i, axis=dim)

            results.append(result_i)

        if q_scalar:
            result_mlx = results[0]
        else:
            # Stack results along a new dimension at the front
            result_mlx = mx.stack(results, axis=0)

        result = Tensor._from_mlx_array(result_mlx)

    if input.requires_grad:
        result.requires_grad = True

    return result


def nanmean(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    """
    Mean ignoring NaN values.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    arr = input._mlx_array

    # Create mask for non-NaN values
    mask = mx.logical_not(mx.isnan(arr))

    # Replace NaN with 0 for sum, then divide by count of non-NaN values
    masked_arr = mx.where(mask, arr, mx.zeros_like(arr))

    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        total = mx.sum(masked_arr, axis=axis, keepdims=keepdim)
        count = mx.sum(mask.astype(arr.dtype), axis=axis, keepdims=keepdim)
    else:
        total = mx.sum(masked_arr, keepdims=keepdim)
        count = mx.sum(mask.astype(arr.dtype), keepdims=keepdim)

    result_mlx = total / count
    result = Tensor._from_mlx_array(result_mlx)

    if input.requires_grad:
        result.requires_grad = True

    return result


def nansum(
    input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> Tensor:
    """
    Sum ignoring NaN values.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    arr = input._mlx_array

    # Replace NaN with 0 for sum
    mask = mx.isnan(arr)
    masked_arr = mx.where(mask, mx.zeros_like(arr), arr)

    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        result_mlx = mx.sum(masked_arr, axis=axis, keepdims=keepdim)
    else:
        result_mlx = mx.sum(masked_arr, keepdims=keepdim)

    result = Tensor._from_mlx_array(result_mlx)

    if input.requires_grad:
        result.requires_grad = True

    return result


def std_mean(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute standard deviation and mean simultaneously.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions
        unbiased: Whether to use Bessel's correction

    Returns:
        Tuple of (std, mean)
    """
    std_val = std(input, dim=dim, keepdim=keepdim, unbiased=unbiased)
    mean_val = mean(input, dim=dim, keepdim=keepdim)
    return std_val, mean_val


def var_mean(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    unbiased: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute variance and mean simultaneously.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions
        unbiased: Whether to use Bessel's correction

    Returns:
        Tuple of (var, mean)
    """
    var_val = var(input, dim=dim, keepdim=keepdim, unbiased=unbiased)
    mean_val = mean(input, dim=dim, keepdim=keepdim)
    return var_val, mean_val


def cummax(input: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
    """
    Cumulative maximum along a dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to compute

    Returns:
        Tuple of (values, indices)
    """
    arr = input._mlx_array
    ndim = arr.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Use native MLX cummax for values
    values_mlx = mx.cummax(arr, axis=dim)

    # Compute indices using pure MLX operations
    # PyTorch returns the index where the maximum was LAST updated (i.e., when a new max was found)
    # This is different from "first occurrence of the current max value"

    n = arr.shape[dim]

    # Create position indices [0, 1, 2, ..., n-1] along target dimension
    shape = [1] * ndim
    shape[dim] = n
    pos_indices = mx.arange(n, dtype=mx.int32).reshape(shape)
    pos_indices = mx.broadcast_to(pos_indices, arr.shape)

    # Detect where a new maximum is achieved (arr[i] > cummax[i-1])
    # Shift cummax by 1 to compare with previous cummax
    # For position 0, it's always "new max" since there's no previous

    # Create shifted cummax (pad with -inf at the start)
    neg_inf = mx.array(float("-inf"), dtype=arr.dtype)

    # Build slices for shifting
    slices_before = [slice(None)] * ndim
    slices_before[dim] = slice(0, -1)

    slices_after = [slice(None)] * ndim
    slices_after[dim] = slice(1, None)

    # Get cummax shifted by 1 (previous cummax values)
    prev_cummax_inner = values_mlx[tuple(slices_before)]

    # Pad with -inf at the beginning
    pad_shape = list(arr.shape)
    pad_shape[dim] = 1
    neg_inf_pad = mx.full(pad_shape, float("-inf"), dtype=arr.dtype)

    prev_cummax = mx.concatenate([neg_inf_pad, prev_cummax_inner], axis=dim)

    # Where arr >= prev_cummax, we update the index (includes ties, like PyTorch)
    # Otherwise, we keep the previous index (need cumulative tracking)
    is_new_max = mx.greater_equal(arr, prev_cummax)

    # For indices: where is_new_max, use current position; otherwise use previous index
    # This requires a scan operation, which we implement using cummax on a modified array

    # Trick: Create an array where new_max positions have their index, others have -1
    # Then cummax this to propagate the latest index forward
    neg_one = mx.full(pos_indices.shape, -1, dtype=mx.int32)
    masked_indices = mx.where(is_new_max, pos_indices, neg_one)
    indices_mlx = mx.cummax(masked_indices, axis=dim)

    values = Tensor._from_mlx_array(values_mlx)
    indices = Tensor._from_mlx_array(indices_mlx)

    if input.requires_grad:
        values.requires_grad = True

    return values, indices


def cummin(input: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
    """
    Cumulative minimum along a dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to compute

    Returns:
        Tuple of (values, indices)
    """
    arr = input._mlx_array
    ndim = arr.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Use native MLX cummin for values
    values_mlx = mx.cummin(arr, axis=dim)

    # Compute indices using pure MLX operations
    # PyTorch returns the index where arr[i] <= cummin[i-1] (i.e., when equal OR strictly less)
    # This means the index updates when we see a value that could become/remain the min

    n = arr.shape[dim]

    # Create position indices [0, 1, 2, ..., n-1] along target dimension
    shape = [1] * ndim
    shape[dim] = n
    pos_indices = mx.arange(n, dtype=mx.int32).reshape(shape)
    pos_indices = mx.broadcast_to(pos_indices, arr.shape)

    # Detect where arr[i] <= cummin[i-1] (new minimum or tie with current minimum)
    # Shift cummin by 1 to compare with previous cummin
    # For position 0, it's always "new min" since there's no previous

    # Build slices for shifting
    slices_before = [slice(None)] * ndim
    slices_before[dim] = slice(0, -1)

    # Get cummin shifted by 1 (previous cummin values)
    prev_cummin_inner = values_mlx[tuple(slices_before)]

    # Pad with +inf at the beginning (so position 0 is always considered a "new" min)
    pad_shape = list(arr.shape)
    pad_shape[dim] = 1
    pos_inf_pad = mx.full(pad_shape, float("inf"), dtype=arr.dtype)

    prev_cummin = mx.concatenate([pos_inf_pad, prev_cummin_inner], axis=dim)

    # Where arr <= prev_cummin, we update the index (includes ties)
    is_new_min = mx.less_equal(arr, prev_cummin)

    # For indices: where is_new_min, use current position; otherwise use previous index
    # Trick: Create array where new_min positions have their index, others have -1
    # Then cummax this to propagate the latest index forward
    neg_one = mx.full(pos_indices.shape, -1, dtype=mx.int32)
    masked_indices = mx.where(is_new_min, pos_indices, neg_one)
    indices_mlx = mx.cummax(masked_indices, axis=dim)

    values = Tensor._from_mlx_array(values_mlx)
    indices = Tensor._from_mlx_array(indices_mlx)

    if input.requires_grad:
        values.requires_grad = True

    return values, indices


def logcumsumexp(input: Tensor, dim: int) -> Tensor:
    """
    Log of cumulative sum of exponentials along a dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to compute

    Returns:
        Result tensor
    """
    arr = input._mlx_array
    ndim = arr.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Use native MLX logcumsumexp
    result_mlx = mx.logcumsumexp(arr, axis=dim)
    result = Tensor._from_mlx_array(result_mlx)

    if input.requires_grad:
        result.requires_grad = True

    return result


def histc(input: Tensor, bins: int = 100, min: float = 0, max: float = 0) -> Tensor:
    """
    Compute histogram of a tensor.

    Args:
        input: Input tensor
        bins: Number of bins
        min: Lower bound of histogram (0 = auto)
        max: Upper bound of histogram (0 = auto)

    Returns:
        1D tensor with histogram counts
    """
    arr = input._mlx_array.flatten()

    if min == 0 and max == 0:
        min_val = float(mx.min(arr).item())
        max_val = float(mx.max(arr).item())
    else:
        min_val = min
        max_val = max

    # Compute bin edges
    bin_width = (max_val - min_val) / bins

    # Handle edge case where all values are the same
    if bin_width == 0:
        # All values in one bin
        hist = mx.zeros(bins, dtype=mx.float32)
        hist = hist.at[0].add(mx.array(arr.shape[0], dtype=mx.float32))
        return Tensor._from_mlx_array(hist)

    # Compute which bin each element falls into
    # bin_index = floor((x - min) / bin_width)
    # Clamp to [0, bins-1] to handle edge cases
    normalized = (arr.astype(mx.float32) - min_val) / bin_width
    bin_indices = mx.floor(normalized).astype(mx.int32)
    bin_indices = mx.clip(bin_indices, 0, bins - 1)

    # Count occurrences in each bin
    # Use a loop since MLX doesn't have bincount
    hist = mx.zeros(bins, dtype=mx.float32)
    for i in range(bins):
        count = mx.sum((bin_indices == i).astype(mx.float32))
        hist = hist.at[i].add(count)

    return Tensor._from_mlx_array(hist)
