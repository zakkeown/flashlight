"""
Convolution Operations

Implements convolution operations with PyTorch-compatible API.
"""

from typing import Tuple, Union

import mlx.core as mx

from ..tensor import Tensor


def _pair(x):
    """Convert single value to pair."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    _cached_weight_mlx=None,
) -> Tensor:
    """
    2D convolution operation.

    Args:
        input: Input tensor of shape [N, C, H, W] (PyTorch format) or [N, H, W, C] in nhwc_mode
        weight: Weight tensor of shape [out_channels, in_channels, kH, kW]
        bias: Optional bias tensor of shape [out_channels]
        stride: Stride for convolution
        padding: Padding to apply
        dilation: Dilation factor
        groups: Number of groups for grouped convolution
        _cached_weight_mlx: Optional pre-transposed weight for performance (internal use)

    Returns:
        Output tensor of shape [N, out_channels, H_out, W_out] or [N, H_out, W_out, out_channels] in nhwc_mode

    Note:
        MLX uses NHWC format internally. When nhwc_mode() is enabled, input/output
        stay in NHWC format to avoid redundant layout conversions.
    """
    from ..layout import Layout, is_nhwc_mode

    # Check if we're in NHWC-native mode
    nhwc_native = is_nhwc_mode()

    # Get input in NHWC format (required by MLX)
    # Only skip conversion if input is explicitly marked as NHWC
    if nhwc_native and hasattr(input, "_layout") and input._layout == Layout.NHWC:
        # Input is already in NHWC - no conversion needed
        input_nhwc = input._mlx_array
    else:
        # Convert input from NCHW to NHWC
        # input: [N, C, H, W] -> [N, H, W, C]
        input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])

    # Use cached weight if provided, otherwise transpose
    # MLX expects: [C_out, KH, KW, C_in]
    if _cached_weight_mlx is not None:
        weight_mlx = _cached_weight_mlx
    else:
        # Convert weight from [out, in, kH, kW] to [out, kH, kW, in]
        weight_mlx = mx.transpose(weight._mlx_array, [0, 2, 3, 1])

    # Convert parameters to tuples
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # Perform convolution
    output_nhwc = mx.conv2d(
        input_nhwc, weight_mlx, stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    # Add bias if provided
    if bias is not None:
        # bias shape: [out_channels] -> need to broadcast to [N, H, W, out_channels]
        output_nhwc = output_nhwc + bias._mlx_array

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

    if is_grad_enabled() and (
        input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        from ..autograd.function import Conv2dBackward

        result.requires_grad = True
        grad_fn = Conv2dBackward(
            input,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            nhwc_native=nhwc_native,
        )
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


__all__ = ["conv2d"]
