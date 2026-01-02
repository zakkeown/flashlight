"""
Pooling Operations

Implements pooling operations with PyTorch-compatible API.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as mxnn

from ..distributions._constants import PROB_EPSILON
from ..tensor import Tensor


def _pair(x):
    """Convert single value to pair."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _compute_pool_output_size(input_size, kernel_size, stride, padding, dilation, ceil_mode):
    """
    Compute pooling output size following PyTorch's formula.

    Args:
        input_size: Size of the input dimension
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling
        padding: Padding on each side
        dilation: Dilation factor
        ceil_mode: Whether to use ceiling division

    Returns:
        Output size for this dimension
    """
    # Effective kernel size with dilation
    effective_kernel = dilation * (kernel_size - 1) + 1

    # Padded input size
    padded_size = input_size + 2 * padding

    if ceil_mode:
        # Ceiling division formula
        output_size = math.ceil((padded_size - effective_kernel) / stride) + 1
        # Safety check: ensure last pooling starts inside the input
        if (output_size - 1) * stride >= input_size + padding:
            output_size -= 1
    else:
        # Floor division (standard)
        output_size = (padded_size - effective_kernel) // stride + 1

    return max(1, output_size)


# Cached pool object getters to avoid allocation overhead on every call
@lru_cache(maxsize=64)
def _get_max_pool2d(
    kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]
):
    """Get or create a cached MaxPool2d layer."""
    return mxnn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


@lru_cache(maxsize=64)
def _get_avg_pool2d(
    kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]
):
    """Get or create a cached AvgPool2d layer."""
    return mxnn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


@lru_cache(maxsize=64)
def _get_max_pool1d(kernel_size: int, stride: int, padding: int):
    """Get or create a cached MaxPool1d layer."""
    return mxnn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)


@lru_cache(maxsize=64)
def _get_avg_pool1d(kernel_size: int, stride: int, padding: int):
    """Get or create a cached AvgPool1d layer."""
    return mxnn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)


def max_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int], None] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """
    2D max pooling operation.

    Args:
        input: Input tensor of shape [N, C, H, W] (PyTorch format) or [N, H, W, C] in nhwc_mode
        kernel_size: Size of pooling window
        stride: Stride for pooling (default: kernel_size)
        padding: Padding to apply
        dilation: Dilation factor
        return_indices: Whether to return indices
        ceil_mode: Whether to use ceil for output size calculation

    Returns:
        Output tensor of shape [N, C, H_out, W_out] or [N, H_out, W_out, C] in nhwc_mode
        If return_indices is True, returns tuple of (output, indices)

    Note:
        MLX uses NHWC format internally. When nhwc_mode() is enabled, input/output
        stay in NHWC format to avoid redundant layout conversions.
    """
    if return_indices:
        from ..nn.functional import max_pool2d_with_indices

        return max_pool2d_with_indices(
            input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    from ..layout import Layout, is_nhwc_mode

    # Check if we're in NHWC-native mode
    nhwc_native = is_nhwc_mode()

    # Convert parameters
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    padding = _pair(padding)

    # Get input in NHWC format (required by MLX)
    # Use getattr for faster attribute access with default
    if nhwc_native and getattr(input, "_layout", None) == Layout.NHWC:
        # Input is already in NHWC - no conversion needed
        input_nhwc = input._mlx_array
        N, H, W, C = input_nhwc.shape
    else:
        # Convert input from NCHW to NHWC
        # input: [N, C, H, W] -> [N, H, W, C]
        input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])
        N, H, W, C = input_nhwc.shape

    # Handle ceil_mode by adding extra padding if needed
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    # Calculate expected output dimensions with ceil_mode
    H_out = _compute_pool_output_size(H, kH, sH, pH, 1, ceil_mode)
    W_out = _compute_pool_output_size(W, kW, sW, pW, 1, ceil_mode)
    H_out_floor = _compute_pool_output_size(H, kH, sH, pH, 1, False)
    W_out_floor = _compute_pool_output_size(W, kW, sW, pW, 1, False)

    # If ceil_mode requires extra outputs, add padding
    extra_pad_h = 0
    extra_pad_w = 0
    if ceil_mode and H_out > H_out_floor:
        extra_pad_h = max(0, (H_out - 1) * sH + kH - H - 2 * pH)
    if ceil_mode and W_out > W_out_floor:
        extra_pad_w = max(0, (W_out - 1) * sW + kW - W - 2 * pW)

    if extra_pad_h > 0 or extra_pad_w > 0:
        # For max pooling, pad with -inf so padded values never win
        input_nhwc = mx.pad(
            input_nhwc,
            [(0, 0), (pH, pH + extra_pad_h), (pW, pW + extra_pad_w), (0, 0)],
            constant_values=float('-inf')
        )
        # Use pool with padding=0 since we pre-padded
        pool = _get_max_pool2d(kernel_size, stride, (0, 0))
    else:
        # Get cached MLX pooling layer and apply
        pool = _get_max_pool2d(kernel_size, stride, padding)

    output_nhwc = pool(input_nhwc)

    # Determine output layout based on mode
    if nhwc_native:
        # Stay in NHWC - no conversion back
        result = Tensor._from_mlx_array(output_nhwc, layout=Layout.NHWC)
    else:
        # Convert back from NHWC to NCHW
        # [N, H, W, C] -> [N, C, H, W]
        output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])
        result = Tensor._from_mlx_array(output_nchw, layout=Layout.NCHW)

    # Handle autograd
    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        from ..autograd.function import MaxPool2dBackward

        result.requires_grad = True
        grad_fn = MaxPool2dBackward(
            input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nhwc_native=nhwc_native,
            input_nhwc=input_nhwc,
            output_nhwc=output_nhwc,
        )
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int], None] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    """
    2D average pooling operation.

    Args:
        input: Input tensor of shape [N, C, H, W] (PyTorch format) or [N, H, W, C] in nhwc_mode
        kernel_size: Size of pooling window
        stride: Stride for pooling (default: kernel_size)
        padding: Padding to apply
        ceil_mode: Whether to use ceil for output size calculation
        count_include_pad: Whether to include padding in average
        divisor_override: If specified, use this as the divisor instead of kernel area

    Returns:
        Output tensor of shape [N, C, H_out, W_out] or [N, H_out, W_out, C] in nhwc_mode

    Note:
        MLX uses NHWC format internally. When nhwc_mode() is enabled, input/output
        stay in NHWC format to avoid redundant layout conversions.
    """
    from ..layout import Layout, is_nhwc_mode

    # Check if we're in NHWC-native mode
    nhwc_native = is_nhwc_mode()

    # Convert parameters
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    padding = _pair(padding)

    # Get input in NHWC format (required by MLX)
    if nhwc_native and hasattr(input, "_layout") and input._layout == Layout.NHWC:
        # Input is already in NHWC - no conversion needed
        input_nhwc = input._mlx_array
        N, H, W, C = input_nhwc.shape
    else:
        # Convert input from NCHW to NHWC
        # input: [N, C, H, W] -> [N, H, W, C]
        input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])
        N, H, W, C = input_nhwc.shape

    # Handle ceil_mode by adding extra padding if needed
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    # Calculate expected output dimensions with ceil_mode
    H_out = _compute_pool_output_size(H, kH, sH, pH, 1, ceil_mode)
    W_out = _compute_pool_output_size(W, kW, sW, pW, 1, ceil_mode)
    H_out_floor = _compute_pool_output_size(H, kH, sH, pH, 1, False)
    W_out_floor = _compute_pool_output_size(W, kW, sW, pW, 1, False)

    # If ceil_mode requires extra outputs, add padding
    extra_pad_h = 0
    extra_pad_w = 0
    if ceil_mode and H_out > H_out_floor:
        extra_pad_h = max(0, (H_out - 1) * sH + kH - H - 2 * pH)
    if ceil_mode and W_out > W_out_floor:
        extra_pad_w = max(0, (W_out - 1) * sW + kW - W - 2 * pW)

    # For avg pooling with ceil_mode, we need to handle the count properly
    ceil_mode_active = extra_pad_h > 0 or extra_pad_w > 0

    # Handle count_include_pad=False with padding, OR ceil_mode with extra padding
    # Note: ceil_mode extra padding is NEVER included in count, even with count_include_pad=True
    # Regular padding is included only when count_include_pad=True
    if ceil_mode_active or (not count_include_pad and (padding[0] > 0 or padding[1] > 0)):
        # Use sum pooling approach: compute sum, then divide by actual counts
        # First, pad the input with zeros (including extra for ceil_mode)
        total_pad_h = padding[0] + extra_pad_h
        total_pad_w = padding[1] + extra_pad_w
        padded_input = mx.pad(
            input_nhwc, [(0, 0), (padding[0], total_pad_h), (padding[1], total_pad_w), (0, 0)]
        )

        # Create a mask: 1s for elements to include in average, 0s for excluded
        # When count_include_pad=True: include regular padding (1s), exclude ceil_mode extra (0s)
        # When count_include_pad=False: exclude all padding (0s)
        ones_mask = mx.ones((N, H, W, C))
        if count_include_pad:
            # Include regular padding in count, but not ceil_mode extra padding
            padded_mask = mx.pad(
                ones_mask, [(0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)],
                constant_values=1.0
            )
            # Add extra ceil_mode padding as 0s
            if extra_pad_h > 0 or extra_pad_w > 0:
                padded_mask = mx.pad(
                    padded_mask, [(0, 0), (0, extra_pad_h), (0, extra_pad_w), (0, 0)],
                    constant_values=0.0
                )
        else:
            # Exclude all padding from count
            padded_mask = mx.pad(
                ones_mask, [(0, 0), (padding[0], total_pad_h), (padding[1], total_pad_w), (0, 0)],
                constant_values=0.0
            )

        # Compute sum pooling (use avg pool and multiply by kernel area)
        pool_no_pad = _get_avg_pool2d(kernel_size, stride, (0, 0))
        sum_output = pool_no_pad(padded_input) * (kernel_size[0] * kernel_size[1])
        count_output = pool_no_pad(padded_mask) * (kernel_size[0] * kernel_size[1])

        # Divide sum by count to get average (avoiding division by zero)
        output_nhwc = sum_output / mx.maximum(count_output, PROB_EPSILON)
    else:
        # Standard case: count_include_pad=True with no ceil_mode and no special padding handling needed
        pool = _get_avg_pool2d(kernel_size, stride, padding)
        output_nhwc = pool(input_nhwc)

    if divisor_override is not None:
        # When divisor_override is set, adjust for custom divisor
        # MLX AvgPool2d divides by kernel_size, so we need to undo that and apply override
        kernel_area = kernel_size[0] * kernel_size[1]
        output_nhwc = output_nhwc * (kernel_area / divisor_override)

    # Determine output layout based on mode
    if nhwc_native:
        # Stay in NHWC - no conversion back
        result = Tensor._from_mlx_array(output_nhwc, layout=Layout.NHWC)
    else:
        # Convert back from NHWC to NCHW
        # [N, H, W, C] -> [N, C, H, W]
        output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])
        result = Tensor._from_mlx_array(output_nchw, layout=Layout.NCHW)

    # Handle autograd
    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        from ..autograd.function import AvgPool2dBackward

        result.requires_grad = True
        grad_fn = AvgPool2dBackward(
            input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            nhwc_native=nhwc_native,
            divisor_override=divisor_override,
            count_include_pad=count_include_pad,
            input_nhwc=input_nhwc,
        )
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def _single(x):
    """Convert single value to single."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def avg_pool1d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int], None] = None,
    padding: Union[int, Tuple[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """
    1D average pooling operation.

    Args:
        input: Input tensor of shape [N, C, L] (PyTorch format)
        kernel_size: Size of pooling window
        stride: Stride for pooling (default: kernel_size)
        padding: Padding to apply
        ceil_mode: Whether to use ceil for output size calculation
        count_include_pad: Whether to include padding in average

    Returns:
        Output tensor of shape [N, C, L_out]
    """
    kernel_size = _single(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _single(stride)
    padding = _single(padding)

    # Get input dimensions
    x = input._mlx_array
    N, C, L = x.shape

    # Calculate expected output dimension with ceil_mode
    L_out = _compute_pool_output_size(L, kernel_size, stride, padding, 1, ceil_mode)
    L_out_floor = _compute_pool_output_size(L, kernel_size, stride, padding, 1, False)

    # If ceil_mode requires extra outputs, add padding
    extra_pad = 0
    if ceil_mode and L_out > L_out_floor:
        extra_pad = max(0, (L_out - 1) * stride + kernel_size - L - 2 * padding)

    ceil_mode_active = extra_pad > 0

    # Convert to 2D: [N, C, L] -> [N, C, 1, L]
    input_4d = mx.expand_dims(x, axis=2)

    # Convert from NCHW to NHWC
    input_nhwc = mx.transpose(input_4d, [0, 2, 3, 1])

    # Handle count_include_pad=False with padding, OR ceil_mode with extra padding
    # Note: ceil_mode extra padding is NEVER included in count, even with count_include_pad=True
    if ceil_mode_active or (not count_include_pad and padding > 0):
        # Use sum pooling approach: compute sum, then divide by actual counts
        total_pad = padding + extra_pad
        padded_input = mx.pad(input_nhwc, [(0, 0), (0, 0), (padding, total_pad), (0, 0)])

        # Create a mask: 1s for elements to include in average, 0s for excluded
        ones_mask = mx.ones_like(input_nhwc)
        if count_include_pad:
            # Include regular padding in count, but not ceil_mode extra padding
            padded_mask = mx.pad(
                ones_mask, [(0, 0), (0, 0), (padding, padding), (0, 0)],
                constant_values=1.0
            )
            if extra_pad > 0:
                padded_mask = mx.pad(
                    padded_mask, [(0, 0), (0, 0), (0, extra_pad), (0, 0)],
                    constant_values=0.0
                )
        else:
            # Exclude all padding from count
            padded_mask = mx.pad(
                ones_mask, [(0, 0), (0, 0), (padding, total_pad), (0, 0)],
                constant_values=0.0
            )

        # Compute sum pooling
        pool_no_pad = mxnn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, 0))
        sum_output = pool_no_pad(padded_input) * kernel_size
        count_output = pool_no_pad(padded_mask) * kernel_size

        # Divide sum by count
        output_nhwc = sum_output / mx.maximum(count_output, PROB_EPSILON)
    else:
        # Standard case: count_include_pad=True with no ceil_mode
        pool = mxnn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding))
        output_nhwc = pool(input_nhwc)

    # Convert back and squeeze
    output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])
    output_3d = mx.squeeze(output_nchw, axis=2)

    result = Tensor._from_mlx_array(output_3d)

    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def max_pool1d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int], None] = None,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """
    1D max pooling operation.

    Args:
        input: Input tensor of shape [N, C, L] (PyTorch format)
        kernel_size: Size of pooling window
        stride: Stride for pooling (default: kernel_size)
        padding: Padding to apply
        dilation: Dilation factor
        return_indices: Whether to return indices
        ceil_mode: Whether to use ceil for output size calculation

    Returns:
        Output tensor of shape [N, C, L_out]
        If return_indices is True, returns tuple of (output, indices)
    """
    if return_indices:
        from ..nn.functional import max_pool1d_with_indices

        return max_pool1d_with_indices(
            input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    kernel_size = _single(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _single(stride)
    padding = _single(padding)

    # Get input dimensions
    x = input._mlx_array
    N, C, L = x.shape

    # Calculate expected output dimension with ceil_mode
    L_out = _compute_pool_output_size(L, kernel_size, stride, padding, 1, ceil_mode)
    L_out_floor = _compute_pool_output_size(L, kernel_size, stride, padding, 1, False)

    # If ceil_mode requires extra outputs, add padding
    extra_pad = 0
    if ceil_mode and L_out > L_out_floor:
        extra_pad = max(0, (L_out - 1) * stride + kernel_size - L - 2 * padding)

    # Convert to 2D: [N, C, L] -> [N, C, 1, L]
    input_4d = mx.expand_dims(x, axis=2)

    # Convert from NCHW to NHWC
    input_nhwc = mx.transpose(input_4d, [0, 2, 3, 1])

    if extra_pad > 0:
        # For max pooling, pad with -inf so padded values never win
        input_nhwc = mx.pad(
            input_nhwc,
            [(0, 0), (0, 0), (padding, padding + extra_pad), (0, 0)],
            constant_values=float('-inf')
        )
        # Use pool with padding=0 since we pre-padded
        pool = mxnn.MaxPool2d(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, 0))
    else:
        # Standard case
        pool = mxnn.MaxPool2d(kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding))

    output_nhwc = pool(input_nhwc)

    # Convert back and squeeze
    output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])
    output_3d = mx.squeeze(output_nchw, axis=2)

    result = Tensor._from_mlx_array(output_3d)

    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def adaptive_avg_pool1d(input: Tensor, output_size: int) -> Tensor:
    """
    1D adaptive average pooling.

    Args:
        input: Input tensor of shape [N, C, L]
        output_size: Target output size L_out

    Returns:
        Output tensor of shape [N, C, output_size]
    """
    L = input.shape[2]

    if output_size == L:
        return input

    # Compute kernel size and stride to achieve output_size
    kernel_size = (L + output_size - 1) // output_size
    stride = L // output_size

    return avg_pool1d(input, kernel_size=kernel_size, stride=stride, padding=0)


def adaptive_max_pool1d(input: Tensor, output_size: int) -> Tuple[Tensor, Tensor]:
    """
    1D adaptive max pooling.

    Args:
        input: Input tensor of shape [N, C, L]
        output_size: Target output size L_out

    Returns:
        Tuple of (output, indices) where:
        - output: Output tensor of shape [N, C, output_size]
        - indices: Indices tensor of shape [N, C, output_size] with flattened indices
    """
    x = input._mlx_array
    N, C, L = x.shape

    if output_size == L:
        # Identity case - indices are just 0, 1, 2, ...
        indices_array = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (N, C, L))
        return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))

    # Compute adaptive pooling regions
    outputs = []
    indices_list = []

    for i in range(output_size):
        # Compute start and end indices for this output position
        start = (i * L) // output_size
        end = ((i + 1) * L) // output_size

        # Extract the region and find max
        region = x[:, :, start:end]  # [N, C, region_size]
        max_vals = mx.max(region, axis=2, keepdims=True)  # [N, C, 1]
        outputs.append(max_vals)

        # Find the indices of max values (relative to region)
        region_argmax = mx.argmax(region, axis=2, keepdims=True)  # [N, C, 1]
        # Convert to absolute indices in the input
        abs_indices = region_argmax + start
        indices_list.append(abs_indices)

    # Concatenate outputs and indices
    output_array = mx.concatenate(outputs, axis=2)  # [N, C, output_size]
    indices_array = mx.concatenate(indices_list, axis=2)  # [N, C, output_size]

    output = Tensor._from_mlx_array(output_array)
    indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))

    # Handle autograd
    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output, indices


def _triple(x):
    """Convert single value to triple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


def avg_pool3d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int], None] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
) -> Tensor:
    """
    3D average pooling operation.

    Args:
        input: Input tensor of shape [N, C, D, H, W] (PyTorch format)
        kernel_size: Size of pooling window
        stride: Stride for pooling (default: kernel_size)
        padding: Padding to apply
        ceil_mode: Whether to use ceil for output size calculation
        count_include_pad: Whether to include padding in average
        divisor_override: If specified, used as divisor

    Returns:
        Output tensor of shape [N, C, D_out, H_out, W_out]
    """
    kernel_size = _triple(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _triple(stride)
    padding = _triple(padding)

    x = input._mlx_array
    N, C, D, H, W = x.shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    # Calculate output dimensions with ceil_mode
    D_out = _compute_pool_output_size(D, kD, sD, pD, 1, ceil_mode)
    H_out = _compute_pool_output_size(H, kH, sH, pH, 1, ceil_mode)
    W_out = _compute_pool_output_size(W, kW, sW, pW, 1, ceil_mode)
    D_out_floor = _compute_pool_output_size(D, kD, sD, pD, 1, False)
    H_out_floor = _compute_pool_output_size(H, kH, sH, pH, 1, False)
    W_out_floor = _compute_pool_output_size(W, kW, sW, pW, 1, False)

    # Calculate extra padding needed for ceil_mode
    extra_pad_d = 0
    extra_pad_h = 0
    extra_pad_w = 0
    if ceil_mode and D_out > D_out_floor:
        extra_pad_d = max(0, (D_out - 1) * sD + kD - D - 2 * pD)
    if ceil_mode and H_out > H_out_floor:
        extra_pad_h = max(0, (H_out - 1) * sH + kH - H - 2 * pH)
    if ceil_mode and W_out > W_out_floor:
        extra_pad_w = max(0, (W_out - 1) * sW + kW - W - 2 * pW)

    ceil_mode_active = extra_pad_d > 0 or extra_pad_h > 0 or extra_pad_w > 0

    # Handle count_include_pad=False with padding, OR ceil_mode with extra padding
    # Note: ceil_mode extra padding is NEVER included in count, even with count_include_pad=True
    if ceil_mode_active or (not count_include_pad and (pD > 0 or pH > 0 or pW > 0)):
        # Use sum/count approach for proper average calculation
        total_pad_d = pD + extra_pad_d
        total_pad_h = pH + extra_pad_h
        total_pad_w = pW + extra_pad_w

        # Pad with zeros
        x_padded = mx.pad(x, [(0, 0), (0, 0), (pD, total_pad_d), (pH, total_pad_h), (pW, total_pad_w)])

        # Create mask: 1s for elements to include in average, 0s for excluded
        ones = mx.ones((N, C, D, H, W))
        if count_include_pad:
            # Include regular padding in count, but not ceil_mode extra padding
            mask_padded = mx.pad(
                ones, [(0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)],
                constant_values=1.0
            )
            if extra_pad_d > 0 or extra_pad_h > 0 or extra_pad_w > 0:
                mask_padded = mx.pad(
                    mask_padded, [(0, 0), (0, 0), (0, extra_pad_d), (0, extra_pad_h), (0, extra_pad_w)],
                    constant_values=0.0
                )
        else:
            # Exclude all padding from count
            mask_padded = mx.pad(
                ones, [(0, 0), (0, 0), (pD, total_pad_d), (pH, total_pad_h), (pW, total_pad_w)],
                constant_values=0.0
            )

        # Convert to NDHWC
        x_ndhwc = mx.transpose(x_padded, [0, 2, 3, 4, 1])
        mask_ndhwc = mx.transpose(mask_padded, [0, 2, 3, 4, 1])

        # Apply pooling and compute sum/count
        pool_2d = mxnn.AvgPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=(0, 0))
        kernel_area_2d = kH * kW

        sum_outputs = []
        count_outputs = []
        for d_idx in range(D_out):
            d_start = d_idx * sD
            # Sum over depth kernel
            depth_slice = x_ndhwc[:, d_start:d_start + kD, :, :, :]
            mask_slice = mask_ndhwc[:, d_start:d_start + kD, :, :, :]
            depth_sum = mx.sum(depth_slice, axis=1)  # [N, H, W, C]
            depth_count = mx.sum(mask_slice, axis=1)  # [N, H, W, C]

            # Apply 2D pooling (which computes average, so multiply by area to get sum)
            pooled_sum = pool_2d(depth_sum) * kernel_area_2d
            pooled_count = pool_2d(depth_count) * kernel_area_2d
            sum_outputs.append(pooled_sum)
            count_outputs.append(pooled_count)

        sum_result = mx.stack(sum_outputs, axis=1)
        count_result = mx.stack(count_outputs, axis=1)
        result = sum_result / mx.maximum(count_result, PROB_EPSILON)
        result = mx.transpose(result, [0, 4, 1, 2, 3])
    else:
        # Standard case: count_include_pad=True with no ceil_mode and no special handling needed
        # Pad if needed
        if pD > 0 or pH > 0 or pW > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)])

        # Convert NCDHW to NDHWC
        x = mx.transpose(x, [0, 2, 3, 4, 1])

        # Apply pooling depth by depth using mxnn.AvgPool2d
        pool_2d = mxnn.AvgPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=(0, 0))

        outputs = []
        for d_idx in range(D_out):
            d_start = d_idx * sD
            depth_slice = x[:, d_start:d_start + kD, :, :, :]
            depth_avg = mx.mean(depth_slice, axis=1)  # [N, H, W, C]

            # Apply 2D pooling on H,W dimensions
            pooled = pool_2d(depth_avg)
            outputs.append(pooled)

        result = mx.stack(outputs, axis=1)  # [N, D_out, H_out, W_out, C]
        result = mx.transpose(result, [0, 4, 1, 2, 3])  # [N, C, D_out, H_out, W_out]

    if divisor_override is not None:
        # Adjust for custom divisor
        kernel_volume = kD * kH * kW
        result = result * (kernel_volume / divisor_override)

    out = Tensor._from_mlx_array(result)

    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        out.requires_grad = True

    return out


def max_pool3d(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int], None] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False,
) -> Tensor:
    """
    3D max pooling operation.

    Args:
        input: Input tensor of shape [N, C, D, H, W] (PyTorch format)
        kernel_size: Size of pooling window
        stride: Stride for pooling (default: kernel_size)
        padding: Padding to apply
        dilation: Dilation factor
        return_indices: Whether to return indices
        ceil_mode: Whether to use ceil for output size calculation

    Returns:
        Output tensor of shape [N, C, D_out, H_out, W_out]
        If return_indices is True, returns tuple of (output, indices)
    """
    if return_indices:
        from ..nn.functional import max_pool3d_with_indices

        return max_pool3d_with_indices(
            input,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    kernel_size = _triple(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _triple(stride)
    padding = _triple(padding)

    x = input._mlx_array
    N, C, D, H, W = x.shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding

    # Calculate output dimensions with ceil_mode
    D_out = _compute_pool_output_size(D, kD, sD, pD, 1, ceil_mode)
    H_out = _compute_pool_output_size(H, kH, sH, pH, 1, ceil_mode)
    W_out = _compute_pool_output_size(W, kW, sW, pW, 1, ceil_mode)
    D_out_floor = _compute_pool_output_size(D, kD, sD, pD, 1, False)
    H_out_floor = _compute_pool_output_size(H, kH, sH, pH, 1, False)
    W_out_floor = _compute_pool_output_size(W, kW, sW, pW, 1, False)

    # Calculate extra padding needed for ceil_mode
    extra_pad_d = 0
    extra_pad_h = 0
    extra_pad_w = 0
    if ceil_mode and D_out > D_out_floor:
        extra_pad_d = max(0, (D_out - 1) * sD + kD - D - 2 * pD)
    if ceil_mode and H_out > H_out_floor:
        extra_pad_h = max(0, (H_out - 1) * sH + kH - H - 2 * pH)
    if ceil_mode and W_out > W_out_floor:
        extra_pad_w = max(0, (W_out - 1) * sW + kW - W - 2 * pW)

    total_pad_d = pD + extra_pad_d
    total_pad_h = pH + extra_pad_h
    total_pad_w = pW + extra_pad_w

    # Pad if needed - for max pooling use -inf for extra ceil_mode padding
    if total_pad_d > 0 or total_pad_h > 0 or total_pad_w > 0:
        if extra_pad_d > 0 or extra_pad_h > 0 or extra_pad_w > 0:
            # For max pooling with ceil_mode extra padding, pad with -inf
            x = mx.pad(
                x,
                [(0, 0), (0, 0), (pD, total_pad_d), (pH, total_pad_h), (pW, total_pad_w)],
                constant_values=float('-inf')
            )
        else:
            # Standard padding (no extra ceil_mode padding)
            x = mx.pad(x, [(0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)])

    # Convert NCDHW to NDHWC
    x = mx.transpose(x, [0, 2, 3, 4, 1])

    # Apply pooling depth by depth using mxnn.MaxPool2d
    pool_2d = mxnn.MaxPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=(0, 0))

    outputs = []
    for d_idx in range(D_out):
        d_start = d_idx * sD
        depth_slice = x[:, d_start:d_start + kD, :, :, :]
        depth_max = mx.max(depth_slice, axis=1)  # [N, H, W, C]

        # Apply 2D pooling on H,W dimensions
        pooled = pool_2d(depth_max)
        outputs.append(pooled)

    result = mx.stack(outputs, axis=1)  # [N, D_out, H_out, W_out, C]
    result = mx.transpose(result, [0, 4, 1, 2, 3])  # [N, C, D_out, H_out, W_out]

    out = Tensor._from_mlx_array(result)

    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        out.requires_grad = True

    return out


def adaptive_avg_pool3d(input: Tensor, output_size: Union[int, Tuple[int, int, int]]) -> Tensor:
    """
    3D adaptive average pooling.

    Args:
        input: Input tensor of shape [N, C, D, H, W]
        output_size: Target output size (D_out, H_out, W_out)

    Returns:
        Output tensor of shape [N, C, D_out, H_out, W_out]
    """
    output_size = _triple(output_size)
    x = input._mlx_array
    N, C, D, H, W = x.shape
    out_D, out_H, out_W = output_size

    if out_D == D and out_H == H and out_W == W:
        return input
    elif out_D == 1 and out_H == 1 and out_W == 1:
        result = mx.mean(x, axis=(2, 3, 4), keepdims=True)
        return Tensor._from_mlx_array(result)

    # Calculate adaptive strides and kernel sizes
    stride_d = D // out_D
    stride_h = H // out_H
    stride_w = W // out_W
    kernel_d = D - (out_D - 1) * stride_d
    kernel_h = H - (out_H - 1) * stride_h
    kernel_w = W - (out_W - 1) * stride_w

    return avg_pool3d(
        input,
        kernel_size=(kernel_d, kernel_h, kernel_w),
        stride=(stride_d, stride_h, stride_w),
        padding=0,
    )


def adaptive_max_pool3d(
    input: Tensor, output_size: Union[int, Tuple[int, int, int]], return_indices: bool = False
):
    """
    3D adaptive max pooling.

    Args:
        input: Input tensor of shape [N, C, D, H, W]
        output_size: Target output size (D_out, H_out, W_out)
        return_indices: Whether to return indices of max values

    Returns:
        If return_indices is False: Output tensor of shape [N, C, D_out, H_out, W_out]
        If return_indices is True: Tuple of (output, indices)
    """
    output_size = _triple(output_size)
    x = input._mlx_array
    N, C, D, H, W = x.shape
    out_D, out_H, out_W = output_size

    if out_D == D and out_H == H and out_W == W:
        # Identity case
        if return_indices:
            # Create indices grid: flattened D*H*W indices
            d_indices = mx.arange(D).reshape(1, 1, D, 1, 1)
            h_indices = mx.arange(H).reshape(1, 1, 1, H, 1)
            w_indices = mx.arange(W).reshape(1, 1, 1, 1, W)
            indices_array = d_indices * (H * W) + h_indices * W + w_indices
            indices_array = mx.broadcast_to(indices_array, (N, C, D, H, W))
            return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))
        return input
    elif out_D == 1 and out_H == 1 and out_W == 1:
        result = mx.max(x, axis=(2, 3, 4), keepdims=True)
        if return_indices:
            # Flatten spatial dims and find argmax
            x_flat = mx.reshape(x, (N, C, -1))  # [N, C, D*H*W]
            indices = mx.argmax(x_flat, axis=2, keepdims=True)  # [N, C, 1]
            indices = mx.reshape(indices, (N, C, 1, 1, 1))  # [N, C, 1, 1, 1]
            return Tensor._from_mlx_array(result), Tensor._from_mlx_array(indices.astype(mx.int64))
        return Tensor._from_mlx_array(result)

    # Compute adaptive pooling regions
    outputs = []
    indices_list = []

    for di in range(out_D):
        d_start = (di * D) // out_D
        d_end = ((di + 1) * D) // out_D
        depth_outputs = []
        depth_indices = []

        for hi in range(out_H):
            h_start = (hi * H) // out_H
            h_end = ((hi + 1) * H) // out_H
            row_outputs = []
            row_indices = []

            for wi in range(out_W):
                w_start = (wi * W) // out_W
                w_end = ((wi + 1) * W) // out_W

                # Extract the region and find max
                region = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                max_vals = mx.max(region, axis=(2, 3, 4), keepdims=True)  # [N, C, 1, 1, 1]
                row_outputs.append(max_vals)

                if return_indices:
                    # Flatten region and find argmax
                    region_d = d_end - d_start
                    region_h = h_end - h_start
                    region_w = w_end - w_start
                    region_flat = mx.reshape(
                        region, (N, C, -1)
                    )  # [N, C, region_d * region_h * region_w]
                    local_argmax = mx.argmax(region_flat, axis=2)  # [N, C]

                    # Convert local index to (local_d, local_h, local_w)
                    local_d = local_argmax // (region_h * region_w)
                    remainder = local_argmax % (region_h * region_w)
                    local_h = remainder // region_w
                    local_w = remainder % region_w

                    # Convert to global flat index in original D*H*W space
                    global_d = d_start + local_d
                    global_h = h_start + local_h
                    global_w = w_start + local_w
                    global_idx = global_d * (H * W) + global_h * W + global_w  # [N, C]
                    global_idx = mx.reshape(global_idx, (N, C, 1, 1, 1))
                    row_indices.append(global_idx)

            # Concatenate along width dimension
            row_output = mx.concatenate(row_outputs, axis=4)  # [N, C, 1, 1, out_W]
            depth_outputs.append(row_output)
            if return_indices:
                row_idx = mx.concatenate(row_indices, axis=4)  # [N, C, 1, 1, out_W]
                depth_indices.append(row_idx)

        # Concatenate along height dimension
        depth_output = mx.concatenate(depth_outputs, axis=3)  # [N, C, 1, out_H, out_W]
        outputs.append(depth_output)
        if return_indices:
            depth_idx = mx.concatenate(depth_indices, axis=3)  # [N, C, 1, out_H, out_W]
            indices_list.append(depth_idx)

    # Concatenate along depth dimension
    output_array = mx.concatenate(outputs, axis=2)  # [N, C, out_D, out_H, out_W]
    result = Tensor._from_mlx_array(output_array)

    # Handle autograd
    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    if return_indices:
        indices_array = mx.concatenate(indices_list, axis=2)  # [N, C, out_D, out_H, out_W]
        indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))
        return result, indices

    return result


__all__ = [
    "max_pool2d",
    "avg_pool2d",
    "max_pool1d",
    "avg_pool1d",
    "max_pool3d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool3d",
]
