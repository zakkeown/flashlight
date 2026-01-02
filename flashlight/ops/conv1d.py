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
        Uses MLX's native conv1d. MLX expects channels-last format (NLC),
        while PyTorch uses channels-first (NCL), so we transpose.
    """
    # Normalize parameters to single ints
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    # Convert input from NCL to NLC (channels-last)
    # PyTorch: [N, C_in, L] -> MLX: [N, L, C_in]
    input_nlc = mx.transpose(input._mlx_array, [0, 2, 1])

    # Convert weight from PyTorch format to MLX format
    # PyTorch: [C_out, C_in/groups, K] -> MLX: [C_out, K, C_in/groups]
    weight_mlx = mx.transpose(weight._mlx_array, [0, 2, 1])

    # Use native MLX conv1d
    output_nlc = mx.conv1d(
        input_nlc,
        weight_mlx,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Add bias if provided (broadcast over [N, L_out, C_out])
    if bias is not None:
        output_nlc = output_nlc + bias._mlx_array

    # Convert back from NLC to NCL
    # MLX: [N, L_out, C_out] -> PyTorch: [N, C_out, L_out]
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
