"""
3D Convolution Operations

Implements 3D convolution operations with PyTorch-compatible API.
"""

import mlx.core as mx
from ..tensor import Tensor
from typing import Union, Tuple


def _triple(x):
    """Convert single value to triple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1
) -> Tensor:
    """
    3D convolution operation.

    Args:
        input: Input tensor of shape [N, C_in, D, H, W] (PyTorch format)
        weight: Weight tensor of shape [C_out, C_in/groups, kD, kH, kW]
        bias: Optional bias tensor of shape [C_out]
        stride: Stride for convolution
        padding: Padding to apply
        dilation: Dilation factor
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape [N, C_out, D_out, H_out, W_out]

    Note:
        3D convolution is not directly supported in MLX. This implementation
        uses a loop over the depth dimension with 2D convolutions.
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    # Get input dimensions
    N, C_in, D_in, H_in, W_in = input.shape
    C_out, _, kD, kH, kW = weight.shape

    # Calculate output size
    D_out = (D_in + 2 * padding[0] - dilation[0] * (kD - 1) - 1) // stride[0] + 1
    H_out = (H_in + 2 * padding[1] - dilation[1] * (kH - 1) - 1) // stride[1] + 1
    W_out = (W_in + 2 * padding[2] - dilation[2] * (kW - 1) - 1) // stride[2] + 1

    # Pad input along depth dimension
    if padding[0] > 0:
        pad_config = [(0, 0), (0, 0), (padding[0], padding[0]), (0, 0), (0, 0)]
        input_padded = mx.pad(input._mlx_array, pad_config)
    else:
        input_padded = input._mlx_array

    # Convert input from NCDHW to NDHWC
    # [N, C, D, H, W] -> [N, D, H, W, C]
    input_ndhwc = mx.transpose(input_padded, [0, 2, 3, 4, 1])

    # Initialize output tensor
    outputs = []

    # Loop over depth kernel positions
    for d_out in range(D_out):
        d_start = d_out * stride[0]

        # Collect contributions from all kernel depth positions
        depth_sum = None
        for kd in range(kD):
            d_in = d_start + kd * dilation[0]

            # Extract 2D slice: [N, H, W, C]
            input_2d = input_ndhwc[:, d_in, :, :, :]

            # Get 2D kernel slice: [C_out, C_in/groups, kH, kW]
            weight_2d = weight._mlx_array[:, :, kd, :, :]

            # Convert weight to MLX format: [C_out, kH, kW, C_in/groups]
            weight_2d_mlx = mx.transpose(weight_2d, [0, 2, 3, 1])

            # Perform 2D convolution
            # MLX conv2d padding format: single int or 2-tuple (H_pad, W_pad), NOT nested tuples
            conv_out = mx.conv2d(
                mx.expand_dims(input_2d, axis=0) if input_2d.ndim == 3 else input_2d,
                weight_2d_mlx,
                stride=(stride[1], stride[2]),
                padding=(padding[1], padding[2]),
                dilation=(dilation[1], dilation[2]),
                groups=groups
            )

            if depth_sum is None:
                depth_sum = conv_out
            else:
                depth_sum = depth_sum + conv_out

        outputs.append(depth_sum)

    # Stack along depth dimension: list of [N, H_out, W_out, C_out] -> [N, D_out, H_out, W_out, C_out]
    output_ndhwc = mx.stack(outputs, axis=1)

    # Add bias if provided
    if bias is not None:
        # Broadcast bias to [1, 1, 1, 1, C_out]
        output_ndhwc = output_ndhwc + mx.reshape(bias._mlx_array, (1, 1, 1, 1, -1))

    # Convert back from NDHWC to NCDHW
    # [N, D, H, W, C] -> [N, C, D, H, W]
    output_ncdhw = mx.transpose(output_ndhwc, [0, 4, 1, 2, 3])

    result = Tensor._from_mlx_array(output_ncdhw)

    # Handle autograd
    from ..autograd.context import is_grad_enabled
    if is_grad_enabled() and (input.requires_grad or weight.requires_grad or
                               (bias is not None and bias.requires_grad)):
        result.requires_grad = True

    return result


def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Union[Tensor, None] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    output_padding: Union[int, Tuple[int, int, int]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1
) -> Tensor:
    """
    3D transposed convolution operation.

    Also known as fractionally-strided convolution or deconvolution (though not a true
    mathematical deconvolution). This is the gradient of conv3d with respect to its input.

    Args:
        input: Input tensor of shape [N, C_in, D, H, W] (PyTorch format)
        weight: Weight tensor of shape [C_in, C_out/groups, kD, kH, kW]
        bias: Optional bias tensor of shape [C_out]
        stride: Stride for convolution (default: 1)
        padding: Padding to apply (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of groups for grouped convolution (default: 1)
        dilation: Dilation factor (default: 1)

    Returns:
        Output tensor of shape [N, C_out, D_out, H_out, W_out]

    Note:
        MLX uses NDHWC format internally, so we transpose from PyTorch's NCDHW.
        Weight format is also transposed:
        - PyTorch: [C_in, C_out/groups, kD, kH, kW]
        - MLX: [C_out, kD, kH, kW, C_in]
    """
    stride = _triple(stride)
    padding = _triple(padding)
    output_padding = _triple(output_padding)
    dilation = _triple(dilation)

    # Convert input from NCDHW to NDHWC
    # input: [N, C_in, D, H, W] -> [N, D, H, W, C_in]
    input_ndhwc = mx.transpose(input._mlx_array, [0, 2, 3, 4, 1])

    # Convert weight from PyTorch format to MLX format
    # PyTorch: [C_in, C_out/groups, kD, kH, kW]
    # MLX expects: [C_out, kD, kH, kW, C_in]
    # So we need: [1, 2, 3, 4, 0] transpose
    weight_mlx = mx.transpose(weight._mlx_array, [1, 2, 3, 4, 0])

    # Perform transposed convolution using native MLX conv_transpose3d
    output_ndhwc = mx.conv_transpose3d(
        input_ndhwc,
        weight_mlx,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        groups=groups
    )

    # Add bias if provided
    if bias is not None:
        # bias shape: [C_out] -> need to broadcast to [N, D, H, W, C_out]
        output_ndhwc = output_ndhwc + bias._mlx_array

    # Convert back from NDHWC to NCDHW
    # [N, D, H, W, C_out] -> [N, C_out, D, H, W]
    output_ncdhw = mx.transpose(output_ndhwc, [0, 4, 1, 2, 3])

    result = Tensor._from_mlx_array(output_ncdhw)

    # Handle autograd
    from ..autograd.context import is_grad_enabled
    if is_grad_enabled() and (input.requires_grad or weight.requires_grad or
                               (bias is not None and bias.requires_grad)):
        result.requires_grad = True

    return result


__all__ = ['conv3d', 'conv_transpose3d']
