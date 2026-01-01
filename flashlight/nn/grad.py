"""
Gradient Computation Utilities

PyTorch-compatible torch.nn.grad module with gradient computation utilities.
These functions compute gradients of convolution operations w.r.t. inputs and weights.
"""

import mlx.core as mx
from typing import Union, Tuple
from ..tensor import Tensor

__all__ = [
    'conv1d_weight',
    'conv2d_weight',
    'conv3d_weight',
    'conv1d_input',
    'conv2d_input',
    'conv3d_input',
]


def _single(x):
    """Convert to single int."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def _pair(x):
    """Convert to pair of ints."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _triple(x):
    """Convert to triple of ints."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


def _flip(arr, axis):
    """Flip array along specified axis/axes (since MLX doesn't have mx.flip)."""
    if isinstance(axis, int):
        axis = (axis,)

    # Build slice for each dimension
    slices = [slice(None)] * arr.ndim
    for ax in axis:
        slices[ax] = slice(None, None, -1)

    return arr[tuple(slices)]


def conv1d_input(
    input_size: Tuple[int, ...],
    weight: Tensor,
    grad_output: Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """
    Compute the gradient of conv1d with respect to the input.

    This is equivalent to transposed convolution (deconvolution).

    Args:
        input_size: Size of the input tensor (N, C_in, L)
        weight: Weight tensor of shape (C_out, C_in/groups, K)
        grad_output: Gradient of the output (N, C_out, L_out)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation of the convolution
        groups: Number of groups

    Returns:
        Gradient with respect to input
    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    # Get dimensions
    N, C_in, L = input_size
    C_out, C_in_per_group, K = weight.shape
    L_out = grad_output.shape[2]

    # Calculate output_padding for transposed convolution
    output_padding = L - ((L_out - 1) * stride - 2 * padding + dilation * (K - 1) + 1)
    output_padding = max(0, output_padding)

    # grad_output: [N, C_out, L_out] -> [N, L_out, C_out] (channels-last)
    grad_output_nlc = mx.transpose(grad_output._mlx_array, [0, 2, 1])

    # Weight for conv_transpose:
    # weight shape for conv1d: [C_out, C_in/groups, K]
    # For transposed conv, we need to swap in/out channels: [C_in/groups, C_out/groups, K]
    # In MLX format for 1D (via 2D): [C_in/groups, 1, K, C_out/groups]
    weight_arr = weight._mlx_array

    if groups == 1:
        # Transpose: [C_out, C_in, K] -> [C_in, K, C_out]
        weight_transposed = mx.transpose(weight_arr, [1, 2, 0])
    else:
        # For grouped conv, we need to reorder the weight tensor
        # weight: [C_out, C_in_per_group, K]
        # Need: [C_in, K, C_out_per_group] per group, then concat
        c_out_per_group = C_out // groups
        weight_groups = []
        for g in range(groups):
            w_g = weight_arr[g * c_out_per_group:(g + 1) * c_out_per_group]  # [c_out_per_group, C_in_per_group, K]
            w_g_transposed = mx.transpose(w_g, [1, 2, 0])  # [C_in_per_group, K, c_out_per_group]
            weight_groups.append(w_g_transposed)
        weight_transposed = mx.concatenate(weight_groups, axis=0)  # [C_in, K, c_out_per_group]

    # Convert to 4D for conv2d: [N, L_out, C_out] -> [N, 1, L_out, C_out]
    grad_output_4d = mx.expand_dims(grad_output_nlc, axis=1)

    # Weight: [C_in, K, C_out_per_group] -> [C_in, 1, K, C_out_per_group]
    weight_4d = mx.expand_dims(weight_transposed, axis=1)

    # Use conv_transpose2d with output_padding
    output_4d = mx.conv_transpose2d(
        grad_output_4d,
        weight_4d,
        stride=(1, stride),
        padding=(0, padding),
        dilation=(1, dilation),
        output_padding=(0, output_padding),
        groups=groups
    )

    # Remove H dimension and transpose back: [N, 1, L, C_in] -> [N, C_in, L]
    output_3d = mx.squeeze(output_4d, axis=1)
    output_ncl = mx.transpose(output_3d, [0, 2, 1])

    # Trim or pad to match expected input size (shouldn't be needed with correct output_padding)
    if output_ncl.shape[2] > L:
        output_ncl = output_ncl[:, :, :L]
    elif output_ncl.shape[2] < L:
        pad_amount = L - output_ncl.shape[2]
        output_ncl = mx.pad(output_ncl, [(0, 0), (0, 0), (0, pad_amount)])

    return Tensor._from_mlx_array(output_ncl)


def conv1d_weight(
    input: Tensor,
    weight_size: Tuple[int, ...],
    grad_output: Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """
    Compute the gradient of conv1d with respect to the weight.

    Args:
        input: Input tensor of shape (N, C_in, L)
        weight_size: Size of the weight tensor (C_out, C_in/groups, K)
        grad_output: Gradient of the output (N, C_out, L_out)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation of the convolution
        groups: Number of groups

    Returns:
        Gradient with respect to weight
    """
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)

    C_out, C_in_per_group, K = weight_size
    N, C_in, L = input.shape
    _, _, L_out = grad_output.shape

    input_arr = input._mlx_array
    grad_output_arr = grad_output._mlx_array

    # Pad input if needed
    if padding > 0:
        input_arr = mx.pad(input_arr, [(0, 0), (0, 0), (padding, padding)])

    # For the weight gradient, we compute convolution between input and grad_output
    # Collect results for each kernel position k

    if groups == 1:
        # input: [N, C_in, L_padded] -> [C_in, N, L_padded]
        input_transposed = mx.transpose(input_arr, [1, 0, 2])

        # grad_output: [N, C_out, L_out] -> [C_out, N, L_out]
        grad_output_transposed = mx.transpose(grad_output_arr, [1, 0, 2])

        # Collect gradient slices for each k
        grad_weight_slices = []
        for k in range(K):
            k_pos = k * dilation
            valid_positions = []
            for l_out in range(L_out):
                input_pos = l_out * stride + k_pos
                if input_pos < input_arr.shape[2]:
                    valid_positions.append(input_pos)

            if len(valid_positions) == L_out:
                indices = mx.array(valid_positions)
                input_at_k = mx.take(input_transposed, indices, axis=2)
                input_flat = mx.reshape(input_at_k, (C_in, N * L_out))
                grad_flat = mx.reshape(grad_output_transposed, (C_out, N * L_out))
                gw_k = mx.matmul(grad_flat, mx.transpose(input_flat, [1, 0]))
                grad_weight_slices.append(gw_k)
            else:
                grad_weight_slices.append(mx.zeros((C_out, C_in), dtype=input_arr.dtype))

        # Stack along K dimension: list of [C_out, C_in] -> [K, C_out, C_in] -> [C_out, C_in, K]
        grad_weight = mx.stack(grad_weight_slices, axis=2)
    else:
        # Handle grouped convolution
        c_in_per_group = C_in // groups
        c_out_per_group = C_out // groups

        # Build result by groups
        group_results = []
        for g in range(groups):
            in_start = g * c_in_per_group
            in_end = (g + 1) * c_in_per_group
            out_start = g * c_out_per_group
            out_end = (g + 1) * c_out_per_group

            input_g = input_arr[:, in_start:in_end, :]
            grad_output_g = grad_output_arr[:, out_start:out_end, :]

            input_g_transposed = mx.transpose(input_g, [1, 0, 2])
            grad_output_g_transposed = mx.transpose(grad_output_g, [1, 0, 2])

            grad_weight_g_slices = []
            for k in range(K):
                k_pos = k * dilation
                valid_positions = []
                for l_out in range(L_out):
                    input_pos = l_out * stride + k_pos
                    if input_pos < input_arr.shape[2]:
                        valid_positions.append(input_pos)

                if len(valid_positions) == L_out:
                    indices = mx.array(valid_positions)
                    input_at_k = mx.take(input_g_transposed, indices, axis=2)
                    input_flat = mx.reshape(input_at_k, (c_in_per_group, N * L_out))
                    grad_flat = mx.reshape(grad_output_g_transposed, (c_out_per_group, N * L_out))
                    gw_k = mx.matmul(grad_flat, mx.transpose(input_flat, [1, 0]))
                    grad_weight_g_slices.append(gw_k)
                else:
                    grad_weight_g_slices.append(mx.zeros((c_out_per_group, c_in_per_group), dtype=input_arr.dtype))

            # Stack for this group: [c_out_per_group, c_in_per_group, K]
            grad_weight_g = mx.stack(grad_weight_g_slices, axis=2)
            group_results.append(grad_weight_g)

        # Concatenate groups along output channel dimension
        grad_weight = mx.concatenate(group_results, axis=0)

    return Tensor._from_mlx_array(grad_weight)


def conv2d_input(
    input_size: Tuple[int, ...],
    weight: Tensor,
    grad_output: Tensor,
    stride: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, int]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
) -> Tensor:
    """
    Compute the gradient of conv2d with respect to the input.

    This is equivalent to transposed convolution (deconvolution).

    Args:
        input_size: Size of the input tensor (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in/groups, kH, kW)
        grad_output: Gradient of the output (N, C_out, H_out, W_out)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation of the convolution
        groups: Number of groups

    Returns:
        Gradient with respect to input
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    N, C_in, H, W = input_size
    C_out, C_in_per_group, kH, kW = weight.shape
    H_out, W_out = grad_output.shape[2], grad_output.shape[3]

    # Calculate output_padding needed for transposed convolution
    # For conv: H_out = (H + 2*padding - dilation*(kH-1) - 1) / stride + 1
    # For conv_transpose: H = (H_out - 1) * stride - 2*padding + dilation*(kH-1) + 1 + output_padding
    # So: output_padding = H - ((H_out - 1) * stride - 2*padding + dilation*(kH-1) + 1)
    output_padding_h = H - ((H_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kH - 1) + 1)
    output_padding_w = W - ((W_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kW - 1) + 1)
    output_padding_h = max(0, output_padding_h)
    output_padding_w = max(0, output_padding_w)

    # grad_output: [N, C_out, H_out, W_out] -> [N, H_out, W_out, C_out] (NHWC)
    grad_output_nhwc = mx.transpose(grad_output._mlx_array, [0, 2, 3, 1])

    # Weight for conv_transpose:
    # weight shape for conv2d: [C_out, C_in/groups, kH, kW]
    # For transposed conv, we swap in/out channels: [C_in/groups, kH, kW, C_out/groups]
    weight_arr = weight._mlx_array

    if groups == 1:
        # Transpose: [C_out, C_in, kH, kW] -> [C_in, kH, kW, C_out]
        weight_transposed = mx.transpose(weight_arr, [1, 2, 3, 0])
    else:
        # For grouped conv, reorder weight tensor
        c_out_per_group = C_out // groups
        weight_groups = []
        for g in range(groups):
            w_g = weight_arr[g * c_out_per_group:(g + 1) * c_out_per_group]  # [c_out_per_group, C_in_per_group, kH, kW]
            w_g_transposed = mx.transpose(w_g, [1, 2, 3, 0])  # [C_in_per_group, kH, kW, c_out_per_group]
            weight_groups.append(w_g_transposed)
        weight_transposed = mx.concatenate(weight_groups, axis=0)  # [C_in, kH, kW, c_out_per_group]

    # Use conv_transpose2d with output_padding
    output_nhwc = mx.conv_transpose2d(
        grad_output_nhwc,
        weight_transposed,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=(output_padding_h, output_padding_w),
        groups=groups
    )

    # Handle any remaining size mismatch (shouldn't happen with correct output_padding)
    if output_nhwc.shape[1] < H:
        h_pad = H - output_nhwc.shape[1]
        output_nhwc = mx.pad(output_nhwc, [(0, 0), (0, h_pad), (0, 0), (0, 0)])
    elif output_nhwc.shape[1] > H:
        output_nhwc = output_nhwc[:, :H, :, :]

    if output_nhwc.shape[2] < W:
        w_pad = W - output_nhwc.shape[2]
        output_nhwc = mx.pad(output_nhwc, [(0, 0), (0, 0), (0, w_pad), (0, 0)])
    elif output_nhwc.shape[2] > W:
        output_nhwc = output_nhwc[:, :, :W, :]

    # Convert back to NCHW: [N, H, W, C_in] -> [N, C_in, H, W]
    output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])

    return Tensor._from_mlx_array(output_nchw)


def conv2d_weight(
    input: Tensor,
    weight_size: Tuple[int, ...],
    grad_output: Tensor,
    stride: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, int]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
) -> Tensor:
    """
    Compute the gradient of conv2d with respect to the weight.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight_size: Size of the weight tensor (C_out, C_in/groups, kH, kW)
        grad_output: Gradient of the output (N, C_out, H_out, W_out)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation of the convolution
        groups: Number of groups

    Returns:
        Gradient with respect to weight
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    C_out, C_in_per_group, kH, kW = weight_size
    N, C_in, H, W = input.shape
    _, _, H_out, W_out = grad_output.shape

    input_arr = input._mlx_array
    grad_output_arr = grad_output._mlx_array

    # Pad input if needed
    if padding[0] > 0 or padding[1] > 0:
        input_arr = mx.pad(input_arr, [
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1])
        ])

    if groups == 1:
        # Collect results in nested list, then reshape
        # grad_weight shape: [C_out, C_in, kH, kW]
        grad_weight_rows = []  # Will be [kH * kW] list of [C_out, C_in] matrices

        for kh in range(kH):
            for kw in range(kW):
                kh_pos = kh * dilation[0]
                kw_pos = kw * dilation[1]

                # Check validity and compute indices
                valid = True
                h_indices = []
                w_indices = []
                for h_out in range(H_out):
                    h_in = h_out * stride[0] + kh_pos
                    if h_in >= input_arr.shape[2]:
                        valid = False
                        break
                    h_indices.append(h_in)
                if not valid:
                    grad_weight_rows.append(mx.zeros((C_out, C_in), dtype=input_arr.dtype))
                    continue

                for w_out in range(W_out):
                    w_in = w_out * stride[1] + kw_pos
                    if w_in >= input_arr.shape[3]:
                        valid = False
                        break
                    w_indices.append(w_in)
                if not valid:
                    grad_weight_rows.append(mx.zeros((C_out, C_in), dtype=input_arr.dtype))
                    continue

                h_idx = mx.array(h_indices)
                w_idx = mx.array(w_indices)

                input_at_kh = mx.take(input_arr, h_idx, axis=2)
                input_at_khkw = mx.take(input_at_kh, w_idx, axis=3)

                input_flat = mx.transpose(input_at_khkw, [1, 0, 2, 3])
                input_flat = mx.reshape(input_flat, (C_in, N * H_out * W_out))

                grad_flat = mx.transpose(grad_output_arr, [1, 0, 2, 3])
                grad_flat = mx.reshape(grad_flat, (C_out, N * H_out * W_out))

                gw_khkw = mx.matmul(grad_flat, mx.transpose(input_flat, [1, 0]))
                grad_weight_rows.append(gw_khkw)

        # Stack and reshape: [kH*kW, C_out, C_in] -> [C_out, C_in, kH, kW]
        grad_weight_stacked = mx.stack(grad_weight_rows, axis=0)  # [kH*kW, C_out, C_in]
        grad_weight_stacked = mx.reshape(grad_weight_stacked, (kH, kW, C_out, C_in))  # [kH, kW, C_out, C_in]
        grad_weight = mx.transpose(grad_weight_stacked, [2, 3, 0, 1])  # [C_out, C_in, kH, kW]
    else:
        # Handle grouped convolution
        c_in_per_group = C_in // groups
        c_out_per_group = C_out // groups

        group_results = []
        for g in range(groups):
            in_start = g * c_in_per_group
            in_end = (g + 1) * c_in_per_group
            out_start = g * c_out_per_group
            out_end = (g + 1) * c_out_per_group

            input_g = input_arr[:, in_start:in_end, :, :]
            grad_output_g = grad_output_arr[:, out_start:out_end, :, :]

            grad_weight_g_rows = []
            for kh in range(kH):
                for kw in range(kW):
                    kh_pos = kh * dilation[0]
                    kw_pos = kw * dilation[1]

                    valid = True
                    h_indices = []
                    w_indices = []
                    for h_out in range(H_out):
                        h_in = h_out * stride[0] + kh_pos
                        if h_in >= input_arr.shape[2]:
                            valid = False
                            break
                        h_indices.append(h_in)
                    if not valid:
                        grad_weight_g_rows.append(mx.zeros((c_out_per_group, c_in_per_group), dtype=input_arr.dtype))
                        continue

                    for w_out in range(W_out):
                        w_in = w_out * stride[1] + kw_pos
                        if w_in >= input_arr.shape[3]:
                            valid = False
                            break
                        w_indices.append(w_in)
                    if not valid:
                        grad_weight_g_rows.append(mx.zeros((c_out_per_group, c_in_per_group), dtype=input_arr.dtype))
                        continue

                    h_idx = mx.array(h_indices)
                    w_idx = mx.array(w_indices)

                    input_at_kh = mx.take(input_g, h_idx, axis=2)
                    input_at_khkw = mx.take(input_at_kh, w_idx, axis=3)

                    input_flat = mx.transpose(input_at_khkw, [1, 0, 2, 3])
                    input_flat = mx.reshape(input_flat, (c_in_per_group, N * H_out * W_out))

                    grad_flat = mx.transpose(grad_output_g, [1, 0, 2, 3])
                    grad_flat = mx.reshape(grad_flat, (c_out_per_group, N * H_out * W_out))

                    gw_khkw = mx.matmul(grad_flat, mx.transpose(input_flat, [1, 0]))
                    grad_weight_g_rows.append(gw_khkw)

            # Stack and reshape for this group
            grad_weight_g_stacked = mx.stack(grad_weight_g_rows, axis=0)
            grad_weight_g_stacked = mx.reshape(grad_weight_g_stacked, (kH, kW, c_out_per_group, c_in_per_group))
            grad_weight_g = mx.transpose(grad_weight_g_stacked, [2, 3, 0, 1])
            group_results.append(grad_weight_g)

        # Concatenate groups along output channel dimension
        grad_weight = mx.concatenate(group_results, axis=0)

    return Tensor._from_mlx_array(grad_weight)


def conv3d_input(
    input_size: Tuple[int, ...],
    weight: Tensor,
    grad_output: Tensor,
    stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
    padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
    dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
    groups: int = 1,
) -> Tensor:
    """
    Compute the gradient of conv3d with respect to the input.

    This is equivalent to transposed 3D convolution.

    Args:
        input_size: Size of the input tensor (N, C_in, D, H, W)
        weight: Weight tensor of shape (C_out, C_in/groups, kD, kH, kW)
        grad_output: Gradient of the output (N, C_out, D_out, H_out, W_out)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation of the convolution
        groups: Number of groups

    Returns:
        Gradient with respect to input
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    N, C_in, D, H, W = input_size
    C_out, C_in_per_group, kD, kH, kW = weight.shape
    D_out, H_out, W_out = grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]

    # Calculate output_padding for transposed convolution
    output_padding_d = D - ((D_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kD - 1) + 1)
    output_padding_h = H - ((H_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kH - 1) + 1)
    output_padding_w = W - ((W_out - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kW - 1) + 1)
    output_padding_d = max(0, output_padding_d)
    output_padding_h = max(0, output_padding_h)
    output_padding_w = max(0, output_padding_w)

    # grad_output: [N, C_out, D_out, H_out, W_out] -> [N, D_out, H_out, W_out, C_out] (NDHWC)
    grad_output_ndhwc = mx.transpose(grad_output._mlx_array, [0, 2, 3, 4, 1])

    # Weight for conv_transpose:
    # weight shape for conv3d: [C_out, C_in/groups, kD, kH, kW]
    # For transposed conv, we swap in/out channels: [C_in/groups, kD, kH, kW, C_out/groups]
    weight_arr = weight._mlx_array

    if groups == 1:
        # Transpose: [C_out, C_in, kD, kH, kW] -> [C_in, kD, kH, kW, C_out]
        weight_transposed = mx.transpose(weight_arr, [1, 2, 3, 4, 0])
    else:
        # For grouped conv, reorder weight tensor
        c_out_per_group = C_out // groups
        weight_groups = []
        for g in range(groups):
            w_g = weight_arr[g * c_out_per_group:(g + 1) * c_out_per_group]
            w_g_transposed = mx.transpose(w_g, [1, 2, 3, 4, 0])
            weight_groups.append(w_g_transposed)
        weight_transposed = mx.concatenate(weight_groups, axis=0)

    # Use conv_transpose3d from MLX with output_padding
    output_ndhwc = mx.conv_transpose3d(
        grad_output_ndhwc,
        weight_transposed,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=(output_padding_d, output_padding_h, output_padding_w),
        groups=groups
    )

    # Handle any remaining output size matching
    if output_ndhwc.shape[1] < D:
        d_pad = D - output_ndhwc.shape[1]
        output_ndhwc = mx.pad(output_ndhwc, [(0, 0), (0, d_pad), (0, 0), (0, 0), (0, 0)])
    elif output_ndhwc.shape[1] > D:
        output_ndhwc = output_ndhwc[:, :D, :, :, :]

    if output_ndhwc.shape[2] < H:
        h_pad = H - output_ndhwc.shape[2]
        output_ndhwc = mx.pad(output_ndhwc, [(0, 0), (0, 0), (0, h_pad), (0, 0), (0, 0)])
    elif output_ndhwc.shape[2] > H:
        output_ndhwc = output_ndhwc[:, :, :H, :, :]

    if output_ndhwc.shape[3] < W:
        w_pad = W - output_ndhwc.shape[3]
        output_ndhwc = mx.pad(output_ndhwc, [(0, 0), (0, 0), (0, 0), (0, w_pad), (0, 0)])
    elif output_ndhwc.shape[3] > W:
        output_ndhwc = output_ndhwc[:, :, :, :W, :]

    # Convert back to NCDHW: [N, D, H, W, C_in] -> [N, C_in, D, H, W]
    output_ncdhw = mx.transpose(output_ndhwc, [0, 4, 1, 2, 3])

    return Tensor._from_mlx_array(output_ncdhw)


def conv3d_weight(
    input: Tensor,
    weight_size: Tuple[int, ...],
    grad_output: Tensor,
    stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
    padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
    dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
    groups: int = 1,
) -> Tensor:
    """
    Compute the gradient of conv3d with respect to the weight.

    Args:
        input: Input tensor of shape (N, C_in, D, H, W)
        weight_size: Size of the weight tensor (C_out, C_in/groups, kD, kH, kW)
        grad_output: Gradient of the output (N, C_out, D_out, H_out, W_out)
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation of the convolution
        groups: Number of groups

    Returns:
        Gradient with respect to weight
    """
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    C_out, C_in_per_group, kD, kH, kW = weight_size
    N, C_in, D, H, W = input.shape
    _, _, D_out, H_out, W_out = grad_output.shape

    input_arr = input._mlx_array
    grad_output_arr = grad_output._mlx_array

    # Pad input if needed
    if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
        input_arr = mx.pad(input_arr, [
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1]),
            (padding[2], padding[2])
        ])

    if groups == 1:
        # Collect results for all kernel positions
        grad_weight_rows = []

        for kd in range(kD):
            for kh in range(kH):
                for kw in range(kW):
                    kd_pos = kd * dilation[0]
                    kh_pos = kh * dilation[1]
                    kw_pos = kw * dilation[2]

                    valid = True
                    d_indices = []
                    h_indices = []
                    w_indices = []

                    for d_out in range(D_out):
                        d_in = d_out * stride[0] + kd_pos
                        if d_in >= input_arr.shape[2]:
                            valid = False
                            break
                        d_indices.append(d_in)
                    if not valid:
                        grad_weight_rows.append(mx.zeros((C_out, C_in), dtype=input_arr.dtype))
                        continue

                    for h_out in range(H_out):
                        h_in = h_out * stride[1] + kh_pos
                        if h_in >= input_arr.shape[3]:
                            valid = False
                            break
                        h_indices.append(h_in)
                    if not valid:
                        grad_weight_rows.append(mx.zeros((C_out, C_in), dtype=input_arr.dtype))
                        continue

                    for w_out in range(W_out):
                        w_in = w_out * stride[2] + kw_pos
                        if w_in >= input_arr.shape[4]:
                            valid = False
                            break
                        w_indices.append(w_in)
                    if not valid:
                        grad_weight_rows.append(mx.zeros((C_out, C_in), dtype=input_arr.dtype))
                        continue

                    d_idx = mx.array(d_indices)
                    h_idx = mx.array(h_indices)
                    w_idx = mx.array(w_indices)

                    input_at_kd = mx.take(input_arr, d_idx, axis=2)
                    input_at_kdkh = mx.take(input_at_kd, h_idx, axis=3)
                    input_at_kdkhkw = mx.take(input_at_kdkh, w_idx, axis=4)

                    input_flat = mx.transpose(input_at_kdkhkw, [1, 0, 2, 3, 4])
                    input_flat = mx.reshape(input_flat, (C_in, N * D_out * H_out * W_out))

                    grad_flat = mx.transpose(grad_output_arr, [1, 0, 2, 3, 4])
                    grad_flat = mx.reshape(grad_flat, (C_out, N * D_out * H_out * W_out))

                    gw_kdkhkw = mx.matmul(grad_flat, mx.transpose(input_flat, [1, 0]))
                    grad_weight_rows.append(gw_kdkhkw)

        # Stack and reshape: [kD*kH*kW, C_out, C_in] -> [C_out, C_in, kD, kH, kW]
        grad_weight_stacked = mx.stack(grad_weight_rows, axis=0)  # [kD*kH*kW, C_out, C_in]
        grad_weight_stacked = mx.reshape(grad_weight_stacked, (kD, kH, kW, C_out, C_in))
        grad_weight = mx.transpose(grad_weight_stacked, [3, 4, 0, 1, 2])  # [C_out, C_in, kD, kH, kW]
    else:
        # Handle grouped convolution
        c_in_per_group = C_in // groups
        c_out_per_group = C_out // groups

        group_results = []
        for g in range(groups):
            in_start = g * c_in_per_group
            in_end = (g + 1) * c_in_per_group
            out_start = g * c_out_per_group
            out_end = (g + 1) * c_out_per_group

            input_g = input_arr[:, in_start:in_end, :, :, :]
            grad_output_g = grad_output_arr[:, out_start:out_end, :, :, :]

            grad_weight_g_rows = []
            for kd in range(kD):
                for kh in range(kH):
                    for kw in range(kW):
                        kd_pos = kd * dilation[0]
                        kh_pos = kh * dilation[1]
                        kw_pos = kw * dilation[2]

                        valid = True
                        d_indices = []
                        h_indices = []
                        w_indices = []

                        for d_out in range(D_out):
                            d_in = d_out * stride[0] + kd_pos
                            if d_in >= input_arr.shape[2]:
                                valid = False
                                break
                            d_indices.append(d_in)
                        if not valid:
                            grad_weight_g_rows.append(mx.zeros((c_out_per_group, c_in_per_group), dtype=input_arr.dtype))
                            continue

                        for h_out in range(H_out):
                            h_in = h_out * stride[1] + kh_pos
                            if h_in >= input_arr.shape[3]:
                                valid = False
                                break
                            h_indices.append(h_in)
                        if not valid:
                            grad_weight_g_rows.append(mx.zeros((c_out_per_group, c_in_per_group), dtype=input_arr.dtype))
                            continue

                        for w_out in range(W_out):
                            w_in = w_out * stride[2] + kw_pos
                            if w_in >= input_arr.shape[4]:
                                valid = False
                                break
                            w_indices.append(w_in)
                        if not valid:
                            grad_weight_g_rows.append(mx.zeros((c_out_per_group, c_in_per_group), dtype=input_arr.dtype))
                            continue

                        d_idx = mx.array(d_indices)
                        h_idx = mx.array(h_indices)
                        w_idx = mx.array(w_indices)

                        input_at_kd = mx.take(input_g, d_idx, axis=2)
                        input_at_kdkh = mx.take(input_at_kd, h_idx, axis=3)
                        input_at_kdkhkw = mx.take(input_at_kdkh, w_idx, axis=4)

                        input_flat = mx.transpose(input_at_kdkhkw, [1, 0, 2, 3, 4])
                        input_flat = mx.reshape(input_flat, (c_in_per_group, N * D_out * H_out * W_out))

                        grad_flat = mx.transpose(grad_output_g, [1, 0, 2, 3, 4])
                        grad_flat = mx.reshape(grad_flat, (c_out_per_group, N * D_out * H_out * W_out))

                        gw_kdkhkw = mx.matmul(grad_flat, mx.transpose(input_flat, [1, 0]))
                        grad_weight_g_rows.append(gw_kdkhkw)

            # Stack and reshape for this group
            grad_weight_g_stacked = mx.stack(grad_weight_g_rows, axis=0)
            grad_weight_g_stacked = mx.reshape(grad_weight_g_stacked, (kD, kH, kW, c_out_per_group, c_in_per_group))
            grad_weight_g = mx.transpose(grad_weight_g_stacked, [3, 4, 0, 1, 2])
            group_results.append(grad_weight_g)

        # Concatenate groups along output channel dimension
        grad_weight = mx.concatenate(group_results, axis=0)

    return Tensor._from_mlx_array(grad_weight)
