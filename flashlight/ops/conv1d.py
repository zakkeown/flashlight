"""
1D Convolution Operations

Implements 1D convolution operations with PyTorch-compatible API.
"""

from typing import Tuple, Union

import mlx.core as mx

from ..tensor import Tensor


def _single(x):
    """Ensure value is a single int (not tuple)."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """
    1D convolution operation.

    Args:
        input: Input tensor of shape [N, C_in, L] (PyTorch format)
        weight: Weight tensor of shape [C_out, C_in/groups, K]
        bias: Optional bias tensor of shape [C_out]
        stride: Stride for convolution
        padding: Padding to apply
        dilation: Dilation factor
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape [N, C_out, L_out]

    Note:
        MLX uses channels-last format, so we need to convert from PyTorch's NCL.
    """
    # Normalize parameters to single ints
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    # Convert input from NCL to NLC (channels-last)
    # input: [N, C, L] -> [N, L, C]
    input_nlc = mx.transpose(input._mlx_array, [0, 2, 1])

    # Convert weight from [C_out, C_in/groups, K] to [C_out, K, C_in/groups]
    weight_mlx = mx.transpose(weight._mlx_array, [0, 2, 1])

    # Perform 1D convolution via 2D convolution with height=1
    # Reshape to add H=1 dimension
    N, L, C_in = input_nlc.shape
    C_out, K, _ = weight_mlx.shape

    # Add H=1 dimension: [N, L, C] -> [N, 1, L, C]
    input_4d = mx.reshape(input_nlc, (N, 1, L, C_in))

    # Weight: [C_out, K, C_in] -> [C_out, 1, K, C_in]
    weight_4d = mx.reshape(weight_mlx, (C_out, 1, K, weight_mlx.shape[2]))

    # Perform 2D convolution with kernel_size=(1, K), stride=(1, stride), padding=(0, padding)
    # MLX conv2d padding format: single int or 2-tuple (H_pad, W_pad), NOT nested tuples
    output_4d = mx.conv2d(
        input_4d,
        weight_4d,
        stride=(1, stride),
        padding=(0, padding),
        dilation=(1, dilation),
        groups=groups,
    )

    # Remove H dimension: [N, 1, L_out, C_out] -> [N, L_out, C_out]
    output_nlc = mx.squeeze(output_4d, axis=1)

    # Add bias if provided
    if bias is not None:
        output_nlc = output_nlc + bias._mlx_array

    # Convert back from NLC to NCL
    # [N, L_out, C_out] -> [N, C_out, L_out]
    output_ncl = mx.transpose(output_nlc, [0, 2, 1])

    result = Tensor._from_mlx_array(output_ncl)

    # Handle autograd
    from ..autograd.context import is_grad_enabled

    if is_grad_enabled() and (
        input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        result.requires_grad = True

    return result


__all__ = ["conv1d"]
