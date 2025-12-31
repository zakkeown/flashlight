"""
Convolution Operations

Implements convolution operations with PyTorch-compatible API.
"""

import mlx.core as mx
from ..tensor import Tensor
from typing import Union, Tuple


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
    groups: int = 1
) -> Tensor:
    """
    2D convolution operation.

    Args:
        input: Input tensor of shape [N, C, H, W] (PyTorch format)
        weight: Weight tensor of shape [out_channels, in_channels, kH, kW]
        bias: Optional bias tensor of shape [out_channels]
        stride: Stride for convolution
        padding: Padding to apply
        dilation: Dilation factor
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape [N, out_channels, H_out, W_out]

    Note:
        MLX uses NHWC format, so we need to transpose from PyTorch's NCHW.
    """
    # Convert input from NCHW to NHWC
    # input: [N, C, H, W] -> [N, H, W, C]
    input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])

    # Convert weight from [out, in, kH, kW] to [out, kH, kW, in]
    # MLX expects: [C_out, KH, KW, C_in]
    weight_mlx = mx.transpose(weight._mlx_array, [0, 2, 3, 1])

    # Convert parameters to tuples
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # Perform convolution
    output_nhwc = mx.conv2d(
        input_nhwc,
        weight_mlx,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    # Add bias if provided
    if bias is not None:
        # bias shape: [out_channels] -> need to broadcast to [N, H, W, out_channels]
        output_nhwc = output_nhwc + bias._mlx_array

    # Convert back from NHWC to NCHW
    # [N, H, W, C] -> [N, C, H, W]
    output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])

    result = Tensor._from_mlx_array(output_nchw)

    # Handle autograd
    from ..autograd.context import is_grad_enabled
    if is_grad_enabled() and (input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)):
        result.requires_grad = True
        # TODO: Add backward function for conv2d

    return result


__all__ = ['conv2d']
