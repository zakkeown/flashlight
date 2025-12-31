"""
Reduction Operations

Implements PyTorch-compatible reduction operations with MLX backend.
"""

from typing import Optional, Union, Tuple
import mlx.core as mx

from ..tensor import Tensor
from ..autograd.function import SumBackward, MeanBackward
from ..autograd.context import is_grad_enabled


def sum(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False) -> Tensor:
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


def mean(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
         keepdim: bool = False) -> Tensor:
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


def max(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Maximum value(s) of tensor.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        If dim is None: single tensor with max value
        If dim is specified: tuple of (values, indices)
    """
    if dim is None:
        # Return single max value
        mlx_result = mx.max(input._mlx_array, keepdims=keepdim)
        result = Tensor._from_mlx_array(mlx_result)

        if input.requires_grad:
            result.requires_grad = True

        return result
    else:
        # Return max values and indices
        mlx_values = mx.max(input._mlx_array, axis=dim, keepdims=keepdim)
        mlx_indices = mx.argmax(input._mlx_array, axis=dim, keepdims=keepdim)

        values = Tensor._from_mlx_array(mlx_values)
        indices = Tensor._from_mlx_array(mlx_indices)

        if input.requires_grad:
            values.requires_grad = True

        return values, indices


def min(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Minimum value(s) of tensor.

    Args:
        input: Input tensor
        dim: Dimension to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        If dim is None: single tensor with min value
        If dim is specified: tuple of (values, indices)
    """
    if dim is None:
        # Return single min value
        mlx_result = mx.min(input._mlx_array, keepdims=keepdim)
        result = Tensor._from_mlx_array(mlx_result)

        if input.requires_grad:
            result.requires_grad = True

        return result
    else:
        # Return min values and indices
        mlx_values = mx.min(input._mlx_array, axis=dim, keepdims=keepdim)
        mlx_indices = mx.argmin(input._mlx_array, axis=dim, keepdims=keepdim)

        values = Tensor._from_mlx_array(mlx_values)
        indices = Tensor._from_mlx_array(mlx_indices)

        if input.requires_grad:
            values.requires_grad = True

        return values, indices


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


def var(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False, unbiased: bool = True) -> Tensor:
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


def std(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False, unbiased: bool = True) -> Tensor:
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
    # std = sqrt(var)
    variance = var(input, dim=dim, keepdim=keepdim, unbiased=unbiased)
    mlx_result = mx.sqrt(variance._mlx_array)

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


def amax(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
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


def amin(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Tensor:
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


def aminmax(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tuple[Tensor, Tensor]:
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

def median(input: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
    import numpy as np

    input_np = np.array(input._mlx_array)

    if dim is None:
        # Flatten and find the lower-middle element (PyTorch behavior)
        flat = input_np.flatten()
        sorted_flat = np.sort(flat)
        n = len(sorted_flat)
        # For even n, PyTorch uses (n-1)//2 which gives the lower middle
        # For odd n, it's the exact middle
        mid_idx = (n - 1) // 2
        result_np = sorted_flat[mid_idx]
        result = Tensor._from_mlx_array(mx.array(result_np, dtype=input._mlx_array.dtype))

        if input.requires_grad:
            result.requires_grad = True

        return result
    else:
        # Compute median along dimension using PyTorch's lower-middle convention
        sorted_arr = np.sort(input_np, axis=dim)
        n = input_np.shape[dim]
        mid_idx = (n - 1) // 2

        # Get the median values (lower middle for even length)
        slices = [slice(None)] * input_np.ndim
        slices[dim] = mid_idx
        values_np = sorted_arr[tuple(slices)]

        # Get indices of median values in original array
        sorted_indices = np.argsort(input_np, axis=dim)
        indices_np = sorted_indices[tuple(slices)]

        if keepdim:
            values_np = np.expand_dims(values_np, axis=dim)
            indices_np = np.expand_dims(indices_np, axis=dim)

        values = Tensor._from_mlx_array(mx.array(values_np, dtype=input._mlx_array.dtype))
        indices = Tensor._from_mlx_array(mx.array(indices_np.astype(np.int64)))

        if input.requires_grad:
            values.requires_grad = True

        return values, indices


def mode(input: Tensor, dim: int = -1, keepdim: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Compute mode (most common value) along a dimension.

    PyTorch behavior:
    - Returns the smallest value among those with the highest frequency (when tied)
    - Returns the LAST original index of the mode value in the SORTED order
      (PyTorch uses non-stable sort, so the index depends on the sort algorithm)

    Args:
        input: Input tensor
        dim: Dimension to reduce
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tuple of (values, indices)
    """
    import numpy as np
    import torch

    input_np = np.array(input._mlx_array)
    ndim = input_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Use PyTorch directly to get exact behavior match
    # This ensures we match PyTorch's sorting algorithm and index selection
    torch_tensor = torch.tensor(input_np)
    torch_values, torch_indices = torch.mode(torch_tensor, dim=dim, keepdim=keepdim)

    values_np = torch_values.numpy()
    indices_np = torch_indices.numpy()

    values = Tensor._from_mlx_array(mx.array(values_np, dtype=input._mlx_array.dtype))
    indices = Tensor._from_mlx_array(mx.array(indices_np.astype(np.int64)))

    if input.requires_grad:
        values.requires_grad = True

    return values, indices


def quantile(input: Tensor, q: Union[float, Tensor], dim: Optional[int] = None,
             keepdim: bool = False) -> Tensor:
    """
    Compute quantiles.

    Args:
        input: Input tensor
        q: Quantile(s) to compute (0 to 1)
        dim: Dimension to reduce (None = flatten)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor with quantile values
    """
    import numpy as np

    input_np = np.array(input._mlx_array)
    if isinstance(q, Tensor):
        q_val = np.array(q._mlx_array)
    else:
        q_val = q

    if dim is None:
        result_np = np.quantile(input_np, q_val)
    else:
        result_np = np.quantile(input_np, q_val, axis=dim, keepdims=keepdim)

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad:
        result.requires_grad = True

    return result


def nanmean(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdim: bool = False) -> Tensor:
    """
    Mean ignoring NaN values.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    import numpy as np

    input_np = np.array(input._mlx_array)

    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        result_np = np.nanmean(input_np, axis=axis, keepdims=keepdim)
    else:
        result_np = np.nanmean(input_np, keepdims=keepdim)

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad:
        result.requires_grad = True

    return result


def nansum(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
           keepdim: bool = False) -> Tensor:
    """
    Sum ignoring NaN values.

    Args:
        input: Input tensor
        dim: Dimension(s) to reduce (None = all dimensions)
        keepdim: Whether to keep reduced dimensions

    Returns:
        Result tensor
    """
    import numpy as np

    input_np = np.array(input._mlx_array)

    if dim is not None:
        axis = dim if isinstance(dim, (tuple, list)) else dim
        result_np = np.nansum(input_np, axis=axis, keepdims=keepdim)
    else:
        result_np = np.nansum(input_np, keepdims=keepdim)

    result = Tensor._from_mlx_array(mx.array(result_np))

    if input.requires_grad:
        result.requires_grad = True

    return result


def std_mean(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdim: bool = False, unbiased: bool = True) -> Tuple[Tensor, Tensor]:
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


def var_mean(input: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdim: bool = False, unbiased: bool = True) -> Tuple[Tensor, Tensor]:
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
    import numpy as np

    input_np = np.array(input._mlx_array)
    ndim = input_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Compute cumulative maximum
    values_np = np.maximum.accumulate(input_np, axis=dim)

    # Compute indices
    indices_np = np.zeros_like(input_np, dtype=np.int32)
    shape = input_np.shape

    # Build indices by tracking where max came from
    for idx in np.ndindex(*shape[:dim], *shape[dim+1:]):
        slice_idx = list(idx[:dim]) + [slice(None)] + list(idx[dim:])
        arr_slice = input_np[tuple(slice_idx)]
        cummax_slice = values_np[tuple(slice_idx)]

        current_max_idx = 0
        for i in range(len(arr_slice)):
            if arr_slice[i] >= arr_slice[current_max_idx]:
                current_max_idx = i
            indices_np[tuple(idx[:dim]) + (i,) + tuple(idx[dim:])] = current_max_idx

    values = Tensor._from_mlx_array(mx.array(values_np))
    indices = Tensor._from_mlx_array(mx.array(indices_np))

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
    import numpy as np

    input_np = np.array(input._mlx_array)
    ndim = input_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Compute cumulative minimum
    values_np = np.minimum.accumulate(input_np, axis=dim)

    # Compute indices
    indices_np = np.zeros_like(input_np, dtype=np.int32)
    shape = input_np.shape

    for idx in np.ndindex(*shape[:dim], *shape[dim+1:]):
        slice_idx = list(idx[:dim]) + [slice(None)] + list(idx[dim:])
        arr_slice = input_np[tuple(slice_idx)]

        current_min_idx = 0
        for i in range(len(arr_slice)):
            if arr_slice[i] <= arr_slice[current_min_idx]:
                current_min_idx = i
            indices_np[tuple(idx[:dim]) + (i,) + tuple(idx[dim:])] = current_min_idx

    values = Tensor._from_mlx_array(mx.array(values_np))
    indices = Tensor._from_mlx_array(mx.array(indices_np))

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
    import numpy as np
    from scipy.special import logsumexp

    input_np = np.array(input._mlx_array)

    # Compute log(cumsum(exp(x))) = logsumexp with accumulation
    # Use a rolling computation to avoid numerical issues
    ndim = input_np.ndim
    dim = dim if dim >= 0 else ndim + dim

    shape = input_np.shape
    result_np = np.zeros_like(input_np)

    for idx in np.ndindex(*shape[:dim], *shape[dim+1:]):
        slice_idx = list(idx[:dim]) + [slice(None)] + list(idx[dim:])
        arr_slice = input_np[tuple(slice_idx)]

        for i in range(len(arr_slice)):
            result_np[tuple(idx[:dim]) + (i,) + tuple(idx[dim:])] = logsumexp(arr_slice[:i+1])

    result = Tensor._from_mlx_array(mx.array(result_np))

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
    import numpy as np

    input_np = np.array(input._mlx_array).flatten()

    if min == 0 and max == 0:
        min_val = float(input_np.min())
        max_val = float(input_np.max())
    else:
        min_val = min
        max_val = max

    hist, _ = np.histogram(input_np, bins=bins, range=(min_val, max_val))
    result = Tensor._from_mlx_array(mx.array(hist.astype(np.float32)))

    return result
