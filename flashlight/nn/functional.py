"""
Neural Network Functional API

PyTorch-compatible torch.nn.functional module.

This module provides functional versions of neural network operations,
re-exporting from flashlight.ops and adding new functional implementations.
"""

import warnings
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as mxnn

from ..autograd.context import is_grad_enabled
from ..distributions._constants import PROB_EPSILON
from ..tensor import Tensor

# =============================================================================
# Internal Utilities
# =============================================================================


def _get_legacy_reduction(
    size_average: Optional[bool], reduce: Optional[bool], reduction: str
) -> str:
    """
    Convert deprecated size_average/reduce arguments to reduction string.

    PyTorch deprecated these in favor of the reduction parameter, but they
    still exist for backwards compatibility.

    Args:
        size_average: Deprecated. If True, losses are averaged; if False, summed.
        reduce: Deprecated. If False, returns unreduced loss.
        reduction: The new-style reduction parameter.

    Returns:
        The reduction string to use ('none', 'mean', or 'sum').
    """
    if size_average is not None or reduce is not None:
        warnings.warn(
            "size_average and reduce args will be deprecated, please use reduction instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True
        if reduce:
            return "mean" if size_average else "sum"
        return "none"
    return reduction


# =============================================================================
# Activation Functions (re-exported from ops.activations)
# =============================================================================

from ..ops.activations import (  # New activations
    celu,
    celu_,
    elu,
    gelu,
    glu,
    hardshrink,
    hardtanh,
    hardtanh_,
    leaky_relu,
    log_softmax,
    logsigmoid,
    prelu,
    relu,
    rrelu,
    rrelu_,
    selu,
    selu_,
    sigmoid,
    silu,
    softmax,
    softmin,
    softshrink,
    tanh,
    tanhshrink,
    threshold,
    threshold_,
)


def relu6(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply ReLU6 activation: min(max(0, x), 6).

    Args:
        input: Input tensor
        inplace: Ignored (MLX doesn't support inplace ops)

    Returns:
        Result tensor
    """
    result_array = mx.minimum(mx.maximum(input._mlx_array, 0), 6)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def hardswish(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply Hard Swish activation: x * ReLU6(x + 3) / 6.

    Args:
        input: Input tensor
        inplace: Ignored (MLX doesn't support inplace ops)

    Returns:
        Result tensor
    """
    result_array = input._mlx_array * mx.minimum(mx.maximum(input._mlx_array + 3, 0), 6) / 6
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply Hard Sigmoid activation: ReLU6(x + 3) / 6.

    Args:
        input: Input tensor
        inplace: Ignored (MLX doesn't support inplace ops)

    Returns:
        Result tensor
    """
    result_array = mx.minimum(mx.maximum(input._mlx_array + 3, 0), 6) / 6
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def softplus(input: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """
    Apply Softplus activation: (1/beta) * log(1 + exp(beta * x)).

    Args:
        input: Input tensor
        beta: Scaling parameter
        threshold: Values above this use linear function

    Returns:
        Result tensor
    """
    scaled = input._mlx_array * beta
    # Use linear function for large values to avoid overflow
    result_array = mx.where(scaled > threshold, input._mlx_array, mx.log(1 + mx.exp(scaled)) / beta)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def softsign(input: Tensor) -> Tensor:
    """
    Apply Softsign activation: x / (1 + |x|).

    Args:
        input: Input tensor

    Returns:
        Result tensor
    """
    result_array = input._mlx_array / (1 + mx.abs(input._mlx_array))
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def mish(input: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply Mish activation: x * tanh(softplus(x)).

    Args:
        input: Input tensor
        inplace: Ignored (MLX doesn't support inplace ops)

    Returns:
        Result tensor
    """
    softplus_x = mx.log(1 + mx.exp(input._mlx_array))
    result_array = input._mlx_array * mx.tanh(softplus_x)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# Swish is alias for SiLU
swish = silu


# =============================================================================
# Convolution Operations (re-exported from ops.convolution)
# =============================================================================

from ..ops.conv1d import conv1d
from ..ops.conv3d import conv3d
from ..ops.convolution import conv2d

# =============================================================================
# Transposed Convolution Operations
# =============================================================================


def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1,
) -> Tensor:
    """
    Apply 1D transposed convolution (deconvolution).

    Args:
        input: Input tensor of shape [N, C_in, L]
        weight: Weight tensor of shape [C_in, C_out/groups, K]
        bias: Optional bias tensor of shape [C_out]
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output (default: 0)
        groups: Number of blocked connections (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Output tensor of shape [N, C_out, L_out]

    Note:
        Output size is computed as:
        L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
    """
    # Convert to 2D for processing: [N, C, L] -> [N, C, 1, L]
    input_4d = mx.expand_dims(input._mlx_array, axis=2)

    # Convert from NCHW to NHWC: [N, C, 1, L] -> [N, 1, L, C]
    input_nhwc = mx.transpose(input_4d, [0, 2, 3, 1])

    # Weight: [C_in, C_out/groups, K] -> [C_out/groups, 1, K, C_in] for MLX
    weight_4d = mx.expand_dims(weight._mlx_array, axis=2)  # [C_in, C_out/g, 1, K]
    weight_transposed = mx.transpose(weight_4d, [1, 2, 3, 0])  # [C_out/g, 1, K, C_in]

    # Perform transposed convolution with dilation support
    # MLX conv_transpose2d supports dilation parameter
    output_nhwc = mx.conv_transpose2d(
        input_nhwc,
        weight_transposed,
        stride=(1, stride),
        padding=(0, padding),
        dilation=(1, dilation),
        groups=groups,
    )

    # Handle output padding
    if output_padding > 0:
        pad_config = [(0, 0), (0, 0), (0, output_padding), (0, 0)]
        output_nhwc = mx.pad(output_nhwc, pad_config)

    # Add bias
    if bias is not None:
        output_nhwc = output_nhwc + bias._mlx_array

    # Convert back to NCHW and squeeze: [N, 1, L_out, C] -> [N, C, L_out]
    output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])
    output_3d = mx.squeeze(output_nchw, axis=2)

    result = Tensor._from_mlx_array(output_3d)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    output_padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tensor:
    """
    Apply 2D transposed convolution (deconvolution).

    Args:
        input: Input tensor of shape [N, C_in, H, W]
        weight: Weight tensor of shape [C_in, C_out/groups, kH, kW]
        bias: Optional bias tensor of shape [C_out]
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output (default: 0)
        groups: Number of blocked connections (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Output tensor of shape [N, C_out, H_out, W_out]
    """
    # Normalize to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Convert input from NCHW to NHWC
    input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])

    # Weight shape: [C_in, C_out/groups, kH, kW]
    # MLX conv_transpose expects: [C_out/groups, kH, kW, C_in]
    weight_transposed = mx.transpose(weight._mlx_array, [1, 2, 3, 0])

    # Perform transposed convolution
    output_nhwc = mx.conv_transpose2d(
        input_nhwc,
        weight_transposed,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=output_padding,
        groups=groups,
    )

    # Add bias if provided
    if bias is not None:
        output_nhwc = output_nhwc + bias._mlx_array

    # Convert back to NCHW
    output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])

    result = Tensor._from_mlx_array(output_nchw)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    output_padding: Union[int, Tuple[int, int, int]] = 0,
    groups: int = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
) -> Tensor:
    """
    Apply 3D transposed convolution (deconvolution).

    Args:
        input: Input tensor of shape [N, C_in, D, H, W]
        weight: Weight tensor of shape [C_in, C_out/groups, kD, kH, kW]
        bias: Optional bias tensor of shape [C_out]
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output (default: 0)
        groups: Number of blocked connections (default: 1)
        dilation: Spacing between kernel elements (default: 1)

    Returns:
        Output tensor of shape [N, C_out, D_out, H_out, W_out]
    """
    from ..ops.conv3d import conv_transpose3d as _conv_transpose3d

    return _conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)


# =============================================================================
# Pooling Operations (re-exported from ops.pooling)
# =============================================================================

from ..ops.pooling import (
    adaptive_avg_pool1d,
    adaptive_avg_pool3d,
)
from ..ops.pooling import adaptive_max_pool1d as _adaptive_max_pool1d_with_indices
from ..ops.pooling import adaptive_max_pool3d as _adaptive_max_pool3d_with_indices
from ..ops.pooling import (
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    max_pool1d,
    max_pool2d,
    max_pool3d,
)


def adaptive_avg_pool2d(input: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """
    Apply 2D adaptive average pooling.

    Args:
        input: Input tensor of shape [N, C, H, W]
        output_size: Target output size (H_out, W_out)

    Returns:
        Output tensor of shape [N, C, H_out, W_out]
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Get input dimensions
    N, C, H, W = input.shape

    # Calculate kernel size and stride to achieve output size
    out_h, out_w = output_size
    kernel_h = H // out_h
    kernel_w = W // out_w
    stride_h = kernel_h
    stride_w = kernel_w

    # Handle case where output_size is (1, 1) - global average pooling
    if out_h == 1 and out_w == 1:
        # Global average pooling
        input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])
        result_nhwc = mx.mean(input_nhwc, axis=(1, 2), keepdims=True)
        result_nchw = mx.transpose(result_nhwc, [0, 3, 1, 2])
        result = Tensor._from_mlx_array(result_nchw)

        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    # Use avg_pool2d with calculated kernel and stride
    return avg_pool2d(input, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))


def adaptive_max_pool2d(
    input: Tensor, output_size: Union[int, Tuple[int, int]], return_indices: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Apply 2D adaptive max pooling.

    Args:
        input: Input tensor of shape [N, C, H, W]
        output_size: Target output size (H_out, W_out)
        return_indices: If True, return indices (not supported)

    Returns:
        Output tensor of shape [N, C, H_out, W_out]
    """
    if return_indices:
        raise NotImplementedError("return_indices is not supported")

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Get input dimensions
    N, C, H, W = input.shape

    # Calculate kernel size and stride to achieve output size
    out_h, out_w = output_size
    kernel_h = H // out_h
    kernel_w = W // out_w
    stride_h = kernel_h
    stride_w = kernel_w

    # Handle case where output_size is (1, 1) - global max pooling
    if out_h == 1 and out_w == 1:
        # Global max pooling
        input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])
        result_nhwc = mx.max(input_nhwc, axis=(1, 2), keepdims=True)
        result_nchw = mx.transpose(result_nhwc, [0, 3, 1, 2])
        result = Tensor._from_mlx_array(result_nchw)

        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    # Use max_pool2d with calculated kernel and stride
    return max_pool2d(input, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))


def adaptive_max_pool1d(
    input: Tensor, output_size: int, return_indices: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Apply 1D adaptive max pooling.

    Args:
        input: Input tensor of shape [N, C, L]
        output_size: Target output size L_out
        return_indices: If True, return indices of max values. Default: False

    Returns:
        If return_indices is False (default): Output tensor of shape [N, C, output_size]
        If return_indices is True: Tuple of (output, indices)
    """
    output, indices = _adaptive_max_pool1d_with_indices(input, output_size)
    if return_indices:
        return output, indices
    return output


def adaptive_max_pool3d(
    input: Tensor, output_size: Union[int, Tuple[int, int, int]], return_indices: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Apply 3D adaptive max pooling.

    Args:
        input: Input tensor of shape [N, C, D, H, W]
        output_size: Target output size (D_out, H_out, W_out) or single int for cube
        return_indices: If True, return indices of max values. Default: False

    Returns:
        If return_indices is False (default): Output tensor of shape [N, C, D_out, H_out, W_out]
        If return_indices is True: Tuple of (output, indices)
    """
    # Note: _adaptive_max_pool3d_with_indices doesn't support return_indices yet,
    # so we call it and only return output by default
    output = _adaptive_max_pool3d_with_indices(input, output_size)
    if return_indices:
        raise NotImplementedError("return_indices is not supported for adaptive_max_pool3d")
    return output


# =============================================================================
# LP Pooling Operations
# =============================================================================


def lp_pool1d(
    input: Tensor,
    norm_type: float,
    kernel_size: int,
    stride: Optional[int] = None,
    ceil_mode: bool = False,
) -> Tensor:
    """
    Apply 1D power-average pooling.

    The output is computed as:
        out = (sum(|x|^p))^(1/p)

    where the sum is over a sliding window of kernel_size.

    Args:
        input: Input tensor of shape [N, C, L]
        norm_type: The exponent value p (typically 1 or 2)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
        ceil_mode: If True, use ceil instead of floor for output size

    Returns:
        Output tensor of shape [N, C, L_out]
    """
    if stride is None:
        stride = kernel_size

    x = input._mlx_array
    N, C, L = x.shape

    # Take absolute value and raise to power p
    x_p = mx.power(mx.abs(x), norm_type)

    # Convert NCL to NLC (channel last) for MLX
    x_p = mx.transpose(x_p, [0, 2, 1])

    # Sum pooling using avg pooling scaled by kernel size
    pool = mxnn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    pooled = pool(x_p)

    # Multiply by kernel_size to get sum, then take p-th root
    pooled = pooled * kernel_size
    result = mx.power(pooled, 1.0 / norm_type)

    # Convert back: NLC -> NCL
    result = mx.transpose(result, [0, 2, 1])

    output = Tensor._from_mlx_array(result)

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output


def lp_pool2d(
    input: Tensor,
    norm_type: float,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    ceil_mode: bool = False,
) -> Tensor:
    """
    Apply 2D power-average pooling.

    The output is computed as:
        out = (sum(|x|^p))^(1/p)

    where the sum is over a sliding window of kernel_size.

    Args:
        input: Input tensor of shape [N, C, H, W]
        norm_type: The exponent value p (typically 1 or 2)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
        ceil_mode: If True, use ceil instead of floor for output size

    Returns:
        Output tensor of shape [N, C, H_out, W_out]
    """
    # Handle kernel_size tuple
    if isinstance(kernel_size, int):
        kH, kW = kernel_size, kernel_size
    else:
        kH, kW = kernel_size

    # Handle stride
    if stride is None:
        sH, sW = kH, kW
    elif isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = stride

    x = input._mlx_array

    # Take absolute value and raise to power p
    x_p = mx.power(mx.abs(x), norm_type)

    # Convert NCHW to NHWC
    x_p = mx.transpose(x_p, [0, 2, 3, 1])

    # Sum pooling using avg pooling scaled by kernel area
    pool = mxnn.AvgPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=0)
    pooled = pool(x_p)

    # Multiply by kernel area to get sum, then take p-th root
    kernel_area = kH * kW
    pooled = pooled * kernel_area
    result = mx.power(pooled, 1.0 / norm_type)

    # Convert back to NCHW
    result = mx.transpose(result, [0, 3, 1, 2])

    output = Tensor._from_mlx_array(result)

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output


# =============================================================================
# Linear Operations
# =============================================================================


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Apply linear transformation: y = xW^T + b.

    Args:
        input: Input tensor of shape (*, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)

    Returns:
        Output tensor of shape (*, out_features)
    """
    # input @ weight.T
    result_array = mx.matmul(input._mlx_array, mx.transpose(weight._mlx_array))

    if bias is not None:
        result_array = result_array + bias._mlx_array

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (
        input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        result.requires_grad = True

    return result


# =============================================================================
# Dropout Operations
# =============================================================================


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    """
    Apply dropout during training.

    Args:
        input: Input tensor
        p: Probability of element being zeroed
        training: Apply dropout if True
        inplace: Ignored (MLX doesn't support inplace ops)

    Returns:
        Result tensor
    """
    if not training or p == 0.0:
        return input

    if p == 1.0:
        return Tensor._from_mlx_array(mx.zeros_like(input._mlx_array))

    # Generate mask with probability (1-p) of keeping
    mask = mx.random.bernoulli(mx.array(1.0 - p), shape=input.shape)

    # Apply inverted dropout (scale by 1/(1-p))
    scale = 1.0 / (1.0 - p)
    result_array = input._mlx_array * mask * scale

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """
    Apply 2D dropout (drops entire channels).

    Args:
        input: Input tensor of shape (N, C, H, W)
        p: Probability of channel being zeroed
        training: Apply dropout if True
        inplace: Ignored

    Returns:
        Result tensor
    """
    if not training or p == 0.0:
        return input

    if p == 1.0:
        return Tensor._from_mlx_array(mx.zeros_like(input._mlx_array))

    # Generate per-channel mask
    N, C = input.shape[0], input.shape[1]
    mask = mx.random.bernoulli(mx.array(1.0 - p), shape=(N, C, 1, 1))

    scale = 1.0 / (1.0 - p)
    result_array = input._mlx_array * mask * scale

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def dropout1d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """
    Apply 1D dropout (drops entire channels).

    Args:
        input: Input tensor of shape (N, C, L)
        p: Probability of channel being zeroed
        training: Apply dropout if True
        inplace: Ignored

    Returns:
        Result tensor
    """
    if not training or p == 0.0:
        return input

    if p == 1.0:
        return Tensor._from_mlx_array(mx.zeros_like(input._mlx_array))

    # Generate per-channel mask (drop entire channel across all L positions)
    N, C = input.shape[0], input.shape[1]
    mask = mx.random.bernoulli(mx.array(1.0 - p), shape=(N, C, 1))

    scale = 1.0 / (1.0 - p)
    result_array = input._mlx_array * mask * scale

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def dropout3d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """
    Apply 3D dropout (drops entire channels).

    Args:
        input: Input tensor of shape (N, C, D, H, W)
        p: Probability of channel being zeroed
        training: Apply dropout if True
        inplace: Ignored

    Returns:
        Result tensor
    """
    if not training or p == 0.0:
        return input

    if p == 1.0:
        return Tensor._from_mlx_array(mx.zeros_like(input._mlx_array))

    # Generate per-channel mask (drop entire channel across all D, H, W)
    N, C = input.shape[0], input.shape[1]
    mask = mx.random.bernoulli(mx.array(1.0 - p), shape=(N, C, 1, 1, 1))

    scale = 1.0 / (1.0 - p)
    result_array = input._mlx_array * mask * scale

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    """
    Apply alpha dropout for SELU networks.

    Alpha dropout maintains the self-normalizing property of SELU networks
    by using a specific dropout formula that preserves mean and variance.

    Args:
        input: Input tensor
        p: Probability of element being zeroed
        training: Apply dropout if True
        inplace: Ignored

    Returns:
        Result tensor
    """
    if not training or p == 0.0:
        return input

    # SELU parameters
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale

    # Alpha dropout formula for self-normalizing networks
    # When an element is dropped, it's set to alpha_p instead of 0
    keep_prob = 1.0 - p

    # Compute scaling factors to maintain variance
    a = ((1.0 - p) * (1.0 + p * alpha_p**2)) ** (-0.5)
    b = -a * alpha_p * p

    # Generate mask
    mask = mx.random.bernoulli(mx.array(keep_prob), shape=input.shape)

    # Apply alpha dropout: kept elements are scaled, dropped elements become alpha_p
    x = input._mlx_array
    result_array = a * (mask * x + (1 - mask) * alpha_p) + b

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def feature_alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False
) -> Tensor:
    """
    Apply feature-wise alpha dropout for SELU networks.

    Drops entire feature maps/channels while maintaining self-normalizing property.

    Args:
        input: Input tensor of shape (N, C, ...)
        p: Probability of channel being zeroed
        training: Apply dropout if True
        inplace: Ignored

    Returns:
        Result tensor
    """
    if not training or p == 0.0:
        return input

    # SELU parameters
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale

    keep_prob = 1.0 - p

    # Compute scaling factors
    a = ((1.0 - p) * (1.0 + p * alpha_p**2)) ** (-0.5)
    b = -a * alpha_p * p

    # Generate per-channel mask
    N, C = input.shape[0], input.shape[1]
    spatial_dims = len(input.shape) - 2
    mask_shape = (N, C) + (1,) * spatial_dims
    mask = mx.random.bernoulli(mx.array(keep_prob), shape=mask_shape)

    # Apply feature-wise alpha dropout
    x = input._mlx_array
    result_array = a * (mask * x + (1 - mask) * alpha_p) + b

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# Normalization Operations
# =============================================================================


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """
    Apply batch normalization.

    Args:
        input: Input tensor of shape (N, C, ...)
        running_mean: Running mean tensor
        running_var: Running variance tensor
        weight: Learnable scale parameter (gamma)
        bias: Learnable shift parameter (beta)
        training: If True, use batch statistics
        momentum: Momentum for running stats update
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    if training:
        # Use batch statistics
        # For 2D: input shape is (N, C, H, W)
        ndim = len(input.shape)
        if ndim == 4:
            # NCHW format - normalize over N, H, W
            axes = (0, 2, 3)
        elif ndim == 3:
            # NCL format - normalize over N, L
            axes = (0, 2)
        else:
            # NC format - normalize over N
            axes = (0,)

        mean = mx.mean(input._mlx_array, axis=axes, keepdims=True)
        var = mx.var(input._mlx_array, axis=axes, keepdims=True)
    else:
        # Use running statistics
        if running_mean is None or running_var is None:
            raise ValueError("running_mean and running_var must be provided when not training")
        mean = running_mean._mlx_array
        var = running_var._mlx_array

        # Reshape for broadcasting
        shape = [1] * len(input.shape)
        shape[1] = -1  # Channel dimension
        mean = mx.reshape(mean, shape)
        var = mx.reshape(var, shape)

    # Normalize
    x_norm = (input._mlx_array - mean) / mx.sqrt(var + eps)

    # Apply affine transformation
    if weight is not None:
        shape = [1] * len(input.shape)
        shape[1] = -1
        gamma = mx.reshape(weight._mlx_array, shape)
        x_norm = x_norm * gamma

    if bias is not None:
        shape = [1] * len(input.shape)
        shape[1] = -1
        beta = mx.reshape(bias._mlx_array, shape)
        x_norm = x_norm + beta

    result = Tensor._from_mlx_array(x_norm)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def layer_norm(
    input: Tensor,
    normalized_shape: Union[int, List[int], Tuple[int, ...]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Apply layer normalization.

    Args:
        input: Input tensor
        normalized_shape: Shape of the normalization dimensions
        weight: Learnable scale parameter (gamma)
        bias: Learnable shift parameter (beta)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # Normalize over the last len(normalized_shape) dimensions
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))

    mean = mx.mean(input._mlx_array, axis=axes, keepdims=True)
    var = mx.var(input._mlx_array, axis=axes, keepdims=True)

    x_norm = (input._mlx_array - mean) / mx.sqrt(var + eps)

    if weight is not None:
        x_norm = x_norm * weight._mlx_array

    if bias is not None:
        x_norm = x_norm + bias._mlx_array

    result = Tensor._from_mlx_array(x_norm)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Apply group normalization.

    Args:
        input: Input tensor of shape (N, C, ...)
        num_groups: Number of groups to divide channels into
        weight: Learnable scale parameter
        bias: Learnable shift parameter
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    N, C = input.shape[0], input.shape[1]
    assert C % num_groups == 0, "num_channels must be divisible by num_groups"

    # Reshape to (N, G, C//G, ...)
    spatial_dims = input.shape[2:]
    x = mx.reshape(input._mlx_array, (N, num_groups, C // num_groups) + spatial_dims)

    # Normalize over C//G and spatial dimensions
    axes = tuple(range(2, len(x.shape)))
    mean = mx.mean(x, axis=axes, keepdims=True)
    var = mx.var(x, axis=axes, keepdims=True)

    x_norm = (x - mean) / mx.sqrt(var + eps)

    # Reshape back
    x_norm = mx.reshape(x_norm, input.shape)

    if weight is not None:
        shape = [1, C] + [1] * len(spatial_dims)
        x_norm = x_norm * mx.reshape(weight._mlx_array, shape)

    if bias is not None:
        shape = [1, C] + [1] * len(spatial_dims)
        x_norm = x_norm + mx.reshape(bias._mlx_array, shape)

    result = Tensor._from_mlx_array(x_norm)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def instance_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """
    Apply instance normalization.

    Args:
        input: Input tensor of shape (N, C, ...)
        running_mean: Running mean (unused, for API compatibility)
        running_var: Running variance (unused, for API compatibility)
        weight: Learnable scale parameter
        bias: Learnable shift parameter
        use_input_stats: Always True for instance norm
        momentum: Unused, for API compatibility
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    # Instance norm: normalize over spatial dimensions for each sample and channel
    ndim = len(input.shape)
    if ndim == 4:
        axes = (2, 3)  # H, W
    elif ndim == 3:
        axes = (2,)  # L
    else:
        raise ValueError("Instance norm requires 3D or 4D input")

    mean = mx.mean(input._mlx_array, axis=axes, keepdims=True)
    var = mx.var(input._mlx_array, axis=axes, keepdims=True)

    x_norm = (input._mlx_array - mean) / mx.sqrt(var + eps)

    if weight is not None:
        shape = [1, input.shape[1]] + [1] * (ndim - 2)
        x_norm = x_norm * mx.reshape(weight._mlx_array, shape)

    if bias is not None:
        shape = [1, input.shape[1]] + [1] * (ndim - 2)
        x_norm = x_norm + mx.reshape(bias._mlx_array, shape)

    result = Tensor._from_mlx_array(x_norm)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# Loss Functions
# =============================================================================


def mse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Mean squared error loss.

    Args:
        input: Predicted values
        target: Target values
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'
        weight: Optional weight tensor for each element

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    diff = input._mlx_array - target._mlx_array
    loss = diff * diff

    if weight is not None:
        loss = loss * weight._mlx_array

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Mean absolute error loss.

    Args:
        input: Predicted values
        target: Target values
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'
        weight: Optional weight tensor for each element

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    loss = mx.abs(input._mlx_array - target._mlx_array)

    if weight is not None:
        loss = loss * weight._mlx_array

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Cross entropy loss with logits.

    Args:
        input: Logits tensor of shape (N, C) or (N, C, d1, d2, ...)
        target: Target tensor of shape (N,) with class indices
        weight: Manual rescaling weight for each class
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        ignore_index: Target value to ignore
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'
        label_smoothing: Label smoothing factor

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    # Compute log_softmax
    log_probs = log_softmax(input, dim=1)

    # Apply label smoothing if needed
    if label_smoothing > 0:
        n_classes = input.shape[1]
        N = input.shape[0]

        # With label smoothing, the loss becomes:
        # (1 - label_smoothing) * nll_loss + label_smoothing * uniform_loss
        # where uniform_loss = -mean(log_probs)

        # Compute standard NLL loss component
        nll = nll_loss(
            log_probs, target, weight=weight, ignore_index=ignore_index, reduction="none"
        )

        # Compute smooth loss: -mean of all log probs (KL div to uniform)
        # This is equivalent to: -sum(log_probs) / n_classes for each sample
        smooth_loss = -mx.mean(log_probs._mlx_array, axis=1)

        # Handle ignore_index for smooth loss
        if ignore_index >= 0:
            mask = target._mlx_array != ignore_index
            smooth_loss = mx.where(mask, smooth_loss, mx.zeros_like(smooth_loss))

        # Combine: (1 - eps) * nll + eps * smooth_loss
        combined_loss = (1.0 - label_smoothing) * nll._mlx_array + label_smoothing * smooth_loss

        # Apply reduction
        if reduction == "none":
            return Tensor._from_mlx_array(combined_loss)
        elif reduction == "sum":
            return Tensor._from_mlx_array(mx.sum(combined_loss))
        else:  # mean
            if ignore_index >= 0:
                mask = target._mlx_array != ignore_index
                return Tensor._from_mlx_array(
                    mx.sum(combined_loss) / mx.sum(mask.astype(combined_loss.dtype))
                )
            else:
                return Tensor._from_mlx_array(mx.mean(combined_loss))

    # Use nll_loss on log_softmax output (no label smoothing)
    return nll_loss(
        log_probs, target, weight=weight, ignore_index=ignore_index, reduction=reduction
    )


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Negative log likelihood loss.

    Args:
        input: Log probabilities of shape (N, C)
        target: Target class indices of shape (N,)
        weight: Weight for each class
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        ignore_index: Target value to ignore
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    N = input.shape[0]
    C = input.shape[1]

    # Gather log probabilities for target classes
    # target shape: (N,) -> indices into dim 1
    target_array = target._mlx_array

    # Create indices for gather
    batch_indices = mx.arange(N)

    # Gather the log probabilities for each target class
    # input[i, target[i]] for each i
    log_probs_at_target = input._mlx_array[batch_indices, target_array]

    # Negative log likelihood
    nll = -log_probs_at_target

    # Handle ignore_index
    if ignore_index >= 0:
        mask = target_array != ignore_index
        nll = mx.where(mask, nll, mx.zeros_like(nll))

    # Apply class weights if provided
    if weight is not None:
        class_weights = weight._mlx_array[target_array]
        nll = nll * class_weights

    # Reduction
    if reduction == "mean":
        if ignore_index >= 0:
            n_valid = mx.sum(mask.astype(mx.float32))
            loss = mx.sum(nll) / mx.maximum(n_valid, mx.array(1.0))
        else:
            loss = mx.mean(nll)
    elif reduction == "sum":
        loss = mx.sum(nll)
    elif reduction == "none":
        loss = nll
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Binary cross entropy loss.

    Args:
        input: Probabilities in range [0, 1]
        target: Binary targets (0 or 1)
        weight: Manual rescaling weight
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    input_clamped = mx.clip(input._mlx_array, PROB_EPSILON, 1.0 - PROB_EPSILON)

    loss = -target._mlx_array * mx.log(input_clamped) - (1 - target._mlx_array) * mx.log(
        1 - input_clamped
    )

    if weight is not None:
        loss = loss * weight._mlx_array

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Binary cross entropy with logits (more numerically stable).

    Args:
        input: Logits (unnormalized scores)
        target: Binary targets (0 or 1)
        weight: Manual rescaling weight
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'
        pos_weight: Weight for positive class

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    # Numerically stable formulation:
    # max(x, 0) - x * z + log(1 + exp(-|x|))
    x = input._mlx_array
    z = target._mlx_array

    if pos_weight is not None:
        pw = pos_weight._mlx_array
        loss = (1 - z) * x + (1 + (pw - 1) * z) * (
            mx.maximum(-x, mx.array(0.0)) + mx.log(1 + mx.exp(-mx.abs(x)))
        )
    else:
        loss = mx.maximum(x, mx.array(0.0)) - x * z + mx.log(1 + mx.exp(-mx.abs(x)))

    if weight is not None:
        loss = loss * weight._mlx_array

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    """
    Smooth L1 loss (Huber loss).

    Args:
        input: Predicted values
        target: Target values
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'
        beta: Threshold for switching between L1 and L2

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    diff = mx.abs(input._mlx_array - target._mlx_array)
    loss = mx.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# RNN Functional Operations
# =============================================================================


def rnn_tanh_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Optional[Tensor] = None,
    b_hh: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply an Elman RNN cell with tanh non-linearity.

    Args:
        input: Input tensor of shape (batch, input_size)
        hx: Hidden state tensor of shape (batch, hidden_size)
        w_ih: Input-hidden weights of shape (hidden_size, input_size)
        w_hh: Hidden-hidden weights of shape (hidden_size, hidden_size)
        b_ih: Input-hidden bias of shape (hidden_size)
        b_hh: Hidden-hidden bias of shape (hidden_size)

    Returns:
        New hidden state of shape (batch, hidden_size)
    """
    igates = mx.matmul(input._mlx_array, w_ih._mlx_array.T)
    hgates = mx.matmul(hx._mlx_array, w_hh._mlx_array.T)

    if b_ih is not None:
        igates = igates + b_ih._mlx_array
    if b_hh is not None:
        hgates = hgates + b_hh._mlx_array

    hy = mx.tanh(igates + hgates)

    result = Tensor._from_mlx_array(hy)
    if is_grad_enabled() and (input.requires_grad or hx.requires_grad or w_ih.requires_grad):
        result.requires_grad = True
    return result


def rnn_relu_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Optional[Tensor] = None,
    b_hh: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply an Elman RNN cell with ReLU non-linearity.

    Args:
        input: Input tensor of shape (batch, input_size)
        hx: Hidden state tensor of shape (batch, hidden_size)
        w_ih: Input-hidden weights of shape (hidden_size, input_size)
        w_hh: Hidden-hidden weights of shape (hidden_size, hidden_size)
        b_ih: Input-hidden bias of shape (hidden_size)
        b_hh: Hidden-hidden bias of shape (hidden_size)

    Returns:
        New hidden state of shape (batch, hidden_size)
    """
    igates = mx.matmul(input._mlx_array, w_ih._mlx_array.T)
    hgates = mx.matmul(hx._mlx_array, w_hh._mlx_array.T)

    if b_ih is not None:
        igates = igates + b_ih._mlx_array
    if b_hh is not None:
        hgates = hgates + b_hh._mlx_array

    hy = mx.maximum(igates + hgates, 0)

    result = Tensor._from_mlx_array(hy)
    if is_grad_enabled() and (input.requires_grad or hx.requires_grad or w_ih.requires_grad):
        result.requires_grad = True
    return result


def lstm_cell(
    input: Tensor,
    hx: Tuple[Tensor, Tensor],
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Optional[Tensor] = None,
    b_hh: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Apply an LSTM cell.

    Args:
        input: Input tensor of shape (batch, input_size)
        hx: Tuple of (h, c) where h and c are of shape (batch, hidden_size)
        w_ih: Input-hidden weights of shape (4*hidden_size, input_size)
        w_hh: Hidden-hidden weights of shape (4*hidden_size, hidden_size)
        b_ih: Input-hidden bias of shape (4*hidden_size)
        b_hh: Hidden-hidden bias of shape (4*hidden_size)

    Returns:
        Tuple of (h', c') where h' and c' are of shape (batch, hidden_size)
    """
    h, c = hx

    igates = mx.matmul(input._mlx_array, w_ih._mlx_array.T)
    hgates = mx.matmul(h._mlx_array, w_hh._mlx_array.T)

    if b_ih is not None:
        igates = igates + b_ih._mlx_array
    if b_hh is not None:
        hgates = hgates + b_hh._mlx_array

    gates = igates + hgates

    # Split gates: input, forget, cell, output
    chunked = mx.split(gates, 4, axis=1)
    i_gate = mx.sigmoid(chunked[0])
    f_gate = mx.sigmoid(chunked[1])
    g_gate = mx.tanh(chunked[2])
    o_gate = mx.sigmoid(chunked[3])

    c_new = f_gate * c._mlx_array + i_gate * g_gate
    h_new = o_gate * mx.tanh(c_new)

    h_result = Tensor._from_mlx_array(h_new)
    c_result = Tensor._from_mlx_array(c_new)

    if is_grad_enabled() and (input.requires_grad or h.requires_grad or w_ih.requires_grad):
        h_result.requires_grad = True
        c_result.requires_grad = True

    return h_result, c_result


def gru_cell(
    input: Tensor,
    hx: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Optional[Tensor] = None,
    b_hh: Optional[Tensor] = None,
) -> Tensor:
    """
    Apply a GRU cell.

    Args:
        input: Input tensor of shape (batch, input_size)
        hx: Hidden state tensor of shape (batch, hidden_size)
        w_ih: Input-hidden weights of shape (3*hidden_size, input_size)
        w_hh: Hidden-hidden weights of shape (3*hidden_size, hidden_size)
        b_ih: Input-hidden bias of shape (3*hidden_size)
        b_hh: Hidden-hidden bias of shape (3*hidden_size)

    Returns:
        New hidden state of shape (batch, hidden_size)
    """
    igates = mx.matmul(input._mlx_array, w_ih._mlx_array.T)
    hgates = mx.matmul(hx._mlx_array, w_hh._mlx_array.T)

    if b_ih is not None:
        igates = igates + b_ih._mlx_array
    if b_hh is not None:
        hgates = hgates + b_hh._mlx_array

    # Split gates: reset, update, new
    i_r, i_z, i_n = mx.split(igates, 3, axis=1)
    h_r, h_z, h_n = mx.split(hgates, 3, axis=1)

    r = mx.sigmoid(i_r + h_r)  # reset gate
    z = mx.sigmoid(i_z + h_z)  # update gate
    n = mx.tanh(i_n + r * h_n)  # new gate

    hy = (1 - z) * n + z * hx._mlx_array

    result = Tensor._from_mlx_array(hy)
    if is_grad_enabled() and (input.requires_grad or hx.requires_grad or w_ih.requires_grad):
        result.requires_grad = True
    return result


# =============================================================================
# Embedding Operations
# =============================================================================


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    """
    Embedding lookup.

    Args:
        input: Tensor of indices
        weight: Embedding weight matrix of shape (num_embeddings, embedding_dim)
        padding_idx: Index to pad (ignored in forward)
        max_norm: Max norm for embeddings
        norm_type: Norm type for max_norm
        scale_grad_by_freq: Scale gradients by frequency
        sparse: Use sparse gradients (not supported)

    Returns:
        Embedded tensor
    """
    # sparse=True now supported via simulated sparse gradients
    indices = input._mlx_array.astype(mx.int32)
    result_array = weight._mlx_array[indices]

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and weight.requires_grad:
        from ..autograd.function import EmbeddingBackward

        num_embeddings, embedding_dim = weight.shape
        result.requires_grad = True
        grad_fn = EmbeddingBackward(
            weight, indices, num_embeddings, embedding_dim, padding_idx=padding_idx, sparse=sparse
        )
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


def embedding_bag(
    input: Tensor,
    weight: Tensor,
    offsets: Optional[Tensor] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    mode: str = "mean",
    sparse: bool = False,
    per_sample_weights: Optional[Tensor] = None,
    include_last_offset: bool = False,
    padding_idx: Optional[int] = None,
) -> Tensor:
    """
    Compute sums or means of 'bags' of embeddings.

    Args:
        input: Tensor of indices. If 2D, each row is a bag. If 1D, use offsets.
        weight: Embedding weight matrix of shape (num_embeddings, embedding_dim)
        offsets: Only used when input is 1D. offsets[i] is the starting position for bag i.
        max_norm: Max norm for embeddings
        norm_type: Norm type for max_norm
        scale_grad_by_freq: Scale gradients by frequency (not supported)
        mode: 'sum', 'mean', or 'max'
        sparse: Use sparse gradients. When True, gradients are computed only for
                accessed indices, which is more memory-efficient for large vocabularies.
                Note: MLX doesn't have native sparse tensor support, so this uses a
                simulated sparse approach that still produces dense gradient tensors
                but computes them more efficiently for sparse access patterns.
        per_sample_weights: Weights for weighted aggregation
        include_last_offset: If True, offsets includes the size of indices as last element
        padding_idx: Index to ignore in aggregation

    Returns:
        Aggregated embeddings of shape (num_bags, embedding_dim)

    Example:
        >>> weight = torch.randn(10, 3)
        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        >>> offsets = torch.tensor([0, 4])
        >>> F.embedding_bag(input, weight, offsets)  # shape: (2, 3)
    """
    if scale_grad_by_freq:
        warnings.warn("scale_grad_by_freq is not supported in MLX")
    if mode not in ["sum", "mean", "max"]:
        raise ValueError(f"mode must be 'sum', 'mean', or 'max', got '{mode}'")

    weight_data = weight._mlx_array
    indices = input._mlx_array

    # Validate weight shape
    if weight_data.ndim != 2:
        raise ValueError(
            f"weight must be a 2D tensor (num_embeddings, embedding_dim), got {weight_data.ndim}D"
        )

    # Apply max_norm if specified
    if max_norm is not None:
        norms = mx.linalg.norm(weight_data, ord=norm_type, axis=1, keepdims=True)
        scale = mx.minimum(max_norm / (norms + 1e-8), mx.array(1.0))
        weight_data = weight_data * scale

    embedding_dim = weight_data.shape[1]

    # If input is 2D, treat each row as a bag
    if indices.ndim == 2:
        batch_size, bag_size = indices.shape

        # Lookup all embeddings
        flat_indices = indices.flatten().astype(mx.int32)
        embeddings = mx.take(weight_data, flat_indices, axis=0)
        embeddings = embeddings.reshape((batch_size, bag_size, embedding_dim))

        # Handle padding_idx - mask out padding indices
        if padding_idx is not None:
            mask = (indices != padding_idx).astype(mx.float32)
            mask = mx.expand_dims(mask, axis=-1)
            embeddings = embeddings * mask

        # Apply per_sample_weights if provided
        if per_sample_weights is not None:
            psw = per_sample_weights._mlx_array
            embeddings = embeddings * mx.expand_dims(psw, axis=-1)

        # Aggregate
        if mode == "sum":
            result_data = mx.sum(embeddings, axis=1)
        elif mode == "mean":
            if padding_idx is not None:
                # Mean over non-padding elements
                count = mx.sum(mask, axis=1)
                count = mx.maximum(count, 1.0)  # Avoid division by zero
                result_data = mx.sum(embeddings, axis=1) / count
            else:
                result_data = mx.mean(embeddings, axis=1)
        else:  # max
            result_data = mx.max(embeddings, axis=1)

        is_2d_input = True
        # bag_size is already set from indices.shape

    else:
        # 1D input with offsets
        if offsets is None:
            raise ValueError("offsets is required when input is 1D")

        offsets_data = offsets._mlx_array.astype(mx.int32)
        indices = indices.astype(mx.int32)

        # Determine bag boundaries
        if include_last_offset:
            # offsets already includes final boundary, so num_bags is len(offsets) - 1
            bag_boundaries = offsets_data
            num_bags = len(offsets_data) - 1
        else:
            # Add length of input as final boundary
            bag_boundaries = mx.concatenate([offsets_data, mx.array([len(indices)])])
            num_bags = len(offsets_data)

        # Process each bag
        results = []
        for i in range(num_bags):
            start = int(bag_boundaries[i])
            end = int(bag_boundaries[i + 1])

            if start >= end:
                # Empty bag
                results.append(mx.zeros((embedding_dim,)))
            else:
                bag_indices = indices[start:end]
                bag_embeddings = mx.take(weight_data, bag_indices, axis=0)

                # Handle padding_idx
                if padding_idx is not None:
                    mask = (bag_indices != padding_idx).astype(mx.float32)
                    mask = mx.expand_dims(mask, axis=-1)
                    bag_embeddings = bag_embeddings * mask

                # Apply per_sample_weights if provided
                if per_sample_weights is not None:
                    psw = per_sample_weights._mlx_array[start:end]
                    bag_embeddings = bag_embeddings * mx.expand_dims(psw, axis=-1)

                # Aggregate
                if mode == "sum":
                    results.append(mx.sum(bag_embeddings, axis=0))
                elif mode == "mean":
                    if padding_idx is not None:
                        count = mx.sum(mask)
                        count = mx.maximum(count, 1.0)
                        results.append(mx.sum(bag_embeddings, axis=0) / count)
                    else:
                        results.append(mx.mean(bag_embeddings, axis=0))
                else:  # max
                    results.append(mx.max(bag_embeddings, axis=0))

        result_data = mx.stack(results, axis=0)
        is_2d_input = False
        bag_size = None

    result = Tensor._from_mlx_array(result_data)

    if is_grad_enabled() and weight.requires_grad:
        from ..autograd.function import EmbeddingBagBackward

        num_embeddings, embedding_dim = weight.shape
        result.requires_grad = True

        # Store indices and offsets for backward pass
        indices_for_backward = input._mlx_array
        offsets_for_backward = offsets._mlx_array if offsets is not None else None
        psw_for_backward = per_sample_weights._mlx_array if per_sample_weights is not None else None

        grad_fn = EmbeddingBagBackward(
            weight,
            indices_for_backward,
            offsets_for_backward,
            num_embeddings,
            embedding_dim,
            mode=mode,
            padding_idx=padding_idx,
            sparse=sparse,
            per_sample_weights=psw_for_backward,
            include_last_offset=include_last_offset,
            is_2d_input=is_2d_input,
            bag_size=bag_size,
        )
        grad_fn.output_tensor = result
        result._grad_fn = grad_fn

    return result


# =============================================================================
# Padding Operations
# =============================================================================


def _pad_reflect(x: "mx.array", pad_pairs: list) -> "mx.array":
    """
    Apply reflect padding to a tensor.

    Reflect padding mirrors the tensor at the edge, excluding the edge value itself.
    For example, padding [1,2,3,4] by 2 on left gives [3,2,1,2,3,4].
    """
    result = x
    for dim, (before, after) in enumerate(pad_pairs):
        if before == 0 and after == 0:
            continue

        size = result.shape[dim]

        # Build the padded array by concatenating reflected slices
        parts = []

        # Left/before padding: reflect from index 1 to before+1 (exclusive of edge)
        if before > 0:
            # Take slice [1:before+1] and reverse it
            indices = list(range(1, min(before + 1, size)))[::-1]
            if indices:
                # Use gather along the dimension
                slices = [slice(None)] * result.ndim
                for idx in indices:
                    slices[dim] = slice(idx, idx + 1)
                    parts.append(result[tuple(slices)])
            # Handle case where before > size - 1 (need to wrap)
            if before > size - 1:
                # Repeat the reflection pattern
                remaining = before - len(indices)
                full_reflect = list(range(1, size))[::-1] + list(range(1, size))
                for i in range(remaining):
                    idx = full_reflect[i % len(full_reflect)]
                    slices[dim] = slice(idx, idx + 1)
                    parts.append(result[tuple(slices)])

        # Original tensor
        parts.append(result)

        # Right/after padding: reflect from index size-2 down
        if after > 0:
            indices = list(range(size - 2, max(size - 2 - after, -1), -1))
            if indices:
                slices = [slice(None)] * result.ndim
                for idx in indices:
                    if idx >= 0:
                        slices[dim] = slice(idx, idx + 1)
                        parts.append(result[tuple(slices)])

        result = mx.concatenate(parts, axis=dim)

    return result


def _pad_replicate(x: "mx.array", pad_pairs: list) -> "mx.array":
    """
    Apply replicate (edge) padding to a tensor.

    Replicate padding copies the edge value.
    For example, padding [1,2,3,4] by 2 on left gives [1,1,1,2,3,4].
    """
    result = x
    for dim, (before, after) in enumerate(pad_pairs):
        if before == 0 and after == 0:
            continue

        parts = []

        # Left/before padding: repeat first element
        if before > 0:
            slices = [slice(None)] * result.ndim
            slices[dim] = slice(0, 1)
            edge = result[tuple(slices)]
            # Repeat 'before' times along this dimension
            repeats = [1] * result.ndim
            repeats[dim] = before
            parts.append(mx.tile(edge, repeats))

        # Original tensor
        parts.append(result)

        # Right/after padding: repeat last element
        if after > 0:
            slices = [slice(None)] * result.ndim
            slices[dim] = slice(-1, None)
            edge = result[tuple(slices)]
            repeats = [1] * result.ndim
            repeats[dim] = after
            parts.append(mx.tile(edge, repeats))

        result = mx.concatenate(parts, axis=dim)

    return result


def _pad_circular(x: "mx.array", pad_pairs: list) -> "mx.array":
    """
    Apply circular (wrap) padding to a tensor.

    Circular padding wraps the tensor around.
    For example, padding [1,2,3,4] by 2 on left gives [3,4,1,2,3,4].
    """
    result = x
    for dim, (before, after) in enumerate(pad_pairs):
        if before == 0 and after == 0:
            continue

        size = result.shape[dim]
        parts = []

        # Left/before padding: take from end of tensor
        if before > 0:
            slices = [slice(None)] * result.ndim
            # Take last 'before' elements
            start_idx = size - before
            if start_idx < 0:
                # Need to wrap multiple times
                times = (before // size) + 1
                expanded = mx.concatenate([result] * times, axis=dim)
                new_size = expanded.shape[dim]
                slices[dim] = slice(new_size - before, new_size)
                parts.append(expanded[tuple(slices)])
            else:
                slices[dim] = slice(start_idx, size)
                parts.append(result[tuple(slices)])

        # Original tensor
        parts.append(result)

        # Right/after padding: take from beginning of tensor
        if after > 0:
            slices = [slice(None)] * result.ndim
            if after > size:
                # Need to wrap multiple times
                times = (after // size) + 1
                expanded = mx.concatenate([result] * times, axis=dim)
                slices[dim] = slice(0, after)
                parts.append(expanded[tuple(slices)])
            else:
                slices[dim] = slice(0, after)
                parts.append(result[tuple(slices)])

        result = mx.concatenate(parts, axis=dim)

    return result


def pad(
    input: Tensor, pad: Tuple[int, ...], mode: str = "constant", value: Optional[float] = None
) -> Tensor:
    """
    Pad a tensor.

    Args:
        input: Input tensor
        pad: Padding sizes (left, right, top, bottom, ...)
        mode: 'constant', 'reflect', 'replicate', or 'circular'
        value: Fill value for constant padding (default: 0.0)

    Returns:
        Padded tensor
    """
    # Handle None default like PyTorch
    if value is None:
        value = 0.0
    # Convert PyTorch pad format to MLX format
    # PyTorch: (left, right, top, bottom, ...) starting from last dim
    # MLX: ((before_0, after_0), (before_1, after_1), ...)

    ndim = len(input.shape)
    pad_pairs = [(0, 0)] * ndim

    # PyTorch pad is in reverse order (last dim first)
    for i in range(0, len(pad), 2):
        dim_idx = ndim - 1 - (i // 2)
        if dim_idx >= 0:
            pad_pairs[dim_idx] = (pad[i], pad[i + 1])

    if mode == "constant":
        result_array = mx.pad(input._mlx_array, pad_pairs, constant_values=value)
    elif mode == "reflect":
        result_array = _pad_reflect(input._mlx_array, pad_pairs)
    elif mode == "replicate":
        result_array = _pad_replicate(input._mlx_array, pad_pairs)
    elif mode == "circular":
        result_array = _pad_circular(input._mlx_array, pad_pairs)
    else:
        raise ValueError(f"Invalid padding mode: {mode}")

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    """
    Rearrange elements in a tensor of shape (*, C*r^2, H, W) to (*, C, H*r, W*r).

    Useful for efficient sub-pixel convolution (depth to space).

    Args:
        input: Input tensor of shape (N, C*r^2, H, W)
        upscale_factor: Factor to increase spatial resolution (r)

    Returns:
        Output tensor of shape (N, C, H*r, W*r)
    """
    r = upscale_factor
    N, C_in, H, W = input.shape

    if C_in % (r * r) != 0:
        raise ValueError(
            f"Number of channels ({C_in}) must be divisible by upscale_factor^2 ({r*r})"
        )

    C_out = C_in // (r * r)
    x = input._mlx_array

    # Reshape: (N, C*r^2, H, W) -> (N, C, r, r, H, W)
    x = mx.reshape(x, (N, C_out, r, r, H, W))

    # Permute: (N, C, r, r, H, W) -> (N, C, H, r, W, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))

    # Reshape: (N, C, H, r, W, r) -> (N, C, H*r, W*r)
    x = mx.reshape(x, (N, C_out, H * r, W * r))

    result = Tensor._from_mlx_array(x)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def pixel_unshuffle(input: Tensor, downscale_factor: int) -> Tensor:
    """
    Rearrange elements in a tensor of shape (*, C, H*r, W*r) to (*, C*r^2, H, W).

    Inverse of pixel_shuffle (space to depth).

    Args:
        input: Input tensor of shape (N, C, H*r, W*r)
        downscale_factor: Factor to decrease spatial resolution (r)

    Returns:
        Output tensor of shape (N, C*r^2, H, W)
    """
    r = downscale_factor
    N, C, H, W = input.shape

    if H % r != 0 or W % r != 0:
        raise ValueError(
            f"Height ({H}) and Width ({W}) must be divisible by downscale_factor ({r})"
        )

    H_out = H // r
    W_out = W // r
    C_out = C * r * r
    x = input._mlx_array

    # Reshape: (N, C, H*r, W*r) -> (N, C, H, r, W, r)
    x = mx.reshape(x, (N, C, H_out, r, W_out, r))

    # Permute: (N, C, H, r, W, r) -> (N, C, r, r, H, W)
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))

    # Reshape: (N, C, r, r, H, W) -> (N, C*r^2, H, W)
    x = mx.reshape(x, (N, C_out, H_out, W_out))

    result = Tensor._from_mlx_array(x)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def channel_shuffle(input: Tensor, groups: int) -> Tensor:
    """
    Shuffle channels for grouped convolutions.

    Divides the channels into groups and interleaves them.

    Args:
        input: Input tensor of shape (N, C, ...)
        groups: Number of groups to shuffle

    Returns:
        Output tensor with shuffled channels
    """
    N, C = input.shape[0], input.shape[1]

    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    channels_per_group = C // groups
    x = input._mlx_array

    # Get spatial dimensions
    spatial_shape = input.shape[2:]

    # Reshape: (N, C, ...) -> (N, groups, channels_per_group, ...)
    new_shape = (N, groups, channels_per_group) + tuple(spatial_shape)
    x = mx.reshape(x, new_shape)

    # Transpose groups and channels_per_group
    # (N, groups, channels_per_group, ...) -> (N, channels_per_group, groups, ...)
    perm = [0, 2, 1] + list(range(3, len(new_shape)))
    x = mx.transpose(x, perm)

    # Reshape back: (N, channels_per_group, groups, ...) -> (N, C, ...)
    final_shape = (N, C) + tuple(spatial_shape)
    x = mx.reshape(x, final_shape)

    result = Tensor._from_mlx_array(x)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# Interpolation Operations
# =============================================================================


def interpolate(
    input: Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
) -> Tensor:
    """
    Upsample/downsample input tensor.

    Args:
        input: Input tensor of shape (N, C, ...) where ... is spatial dimensions
        size: Output spatial size
        scale_factor: Multiplier for spatial size
        mode: 'nearest', 'nearest-exact', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'
        align_corners: Alignment mode for certain modes
        recompute_scale_factor: Recompute scale factor
        antialias: Apply antialiasing when downsampling (not implemented)

    Returns:
        Interpolated tensor
    """
    if antialias:
        warnings.warn("antialias is not supported and will be ignored")
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be specified")

    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be specified")

    # Get input spatial dimensions
    ndim = len(input.shape)
    spatial_dims = input.shape[2:]

    if size is not None:
        if isinstance(size, int):
            target_size = (size,) * len(spatial_dims)
        else:
            target_size = size
    else:
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * len(spatial_dims)
        target_size = tuple(int(s * f) for s, f in zip(spatial_dims, scale_factor))

    if mode == "nearest":
        # Simple nearest neighbor interpolation
        result_array = _nearest_interpolate(input._mlx_array, target_size)
    elif mode == "bilinear":
        # Bilinear interpolation for 4D tensors
        # Default align_corners to False if not specified
        if align_corners is None:
            align_corners = False
        result_array = _bilinear_interpolate(input._mlx_array, target_size, align_corners)
    elif mode == "linear":
        # Linear is 1D interpolation - need 3D input
        if len(input.shape) != 3:
            raise ValueError("Linear interpolation requires 3D input (N, C, L)")
        if align_corners is None:
            align_corners = False
        result_array = _linear_interpolate(input._mlx_array, target_size, align_corners)
    elif mode == "bicubic":
        # Bicubic interpolation for 4D tensors
        if len(input.shape) != 4:
            raise ValueError("Bicubic interpolation requires 4D input (N, C, H, W)")
        if align_corners is None:
            align_corners = False
        result_array = _bicubic_interpolate(input._mlx_array, target_size, align_corners)
    elif mode == "trilinear":
        # Trilinear interpolation for 5D tensors
        if len(input.shape) != 5:
            raise ValueError("Trilinear interpolation requires 5D input (N, C, D, H, W)")
        if align_corners is None:
            align_corners = False
        result_array = _trilinear_interpolate(input._mlx_array, target_size, align_corners)
    elif mode == "area":
        # Area-based interpolation (adaptive average pooling approach)
        result_array = _area_interpolate(input._mlx_array, target_size)
    elif mode == "nearest-exact":
        # Nearest-exact uses more mathematically consistent indexing
        result_array = _nearest_exact_interpolate(input._mlx_array, target_size)
    else:
        raise ValueError(f"Unknown interpolation mode '{mode}'")

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def _nearest_interpolate(x, target_size):
    """Helper for nearest neighbor interpolation."""
    ndim = len(x.shape)

    if ndim == 3:  # NCL (1D)
        N, C, L = x.shape
        target_l = target_size[0]

        # Create coordinate grid
        l_indices = mx.floor(mx.arange(target_l) * (L / target_l)).astype(mx.int32)

        # Gather values
        result = x[:, :, l_indices]

        return result

    elif ndim == 4:  # NCHW (2D)
        N, C, H, W = x.shape
        target_h, target_w = target_size

        # Create coordinate grids
        h_indices = mx.floor(mx.arange(target_h) * (H / target_h)).astype(mx.int32)
        w_indices = mx.floor(mx.arange(target_w) * (W / target_w)).astype(mx.int32)

        # Gather values
        x_h = x[:, :, h_indices, :]
        result = x_h[:, :, :, w_indices]

        return result

    elif ndim == 5:  # NCDHW (3D)
        N, C, D, H, W = x.shape
        target_d, target_h, target_w = target_size

        # Create coordinate grids
        d_indices = mx.floor(mx.arange(target_d) * (D / target_d)).astype(mx.int32)
        h_indices = mx.floor(mx.arange(target_h) * (H / target_h)).astype(mx.int32)
        w_indices = mx.floor(mx.arange(target_w) * (W / target_w)).astype(mx.int32)

        # Gather values - index each dimension sequentially
        x_d = x[:, :, d_indices, :, :]
        x_dh = x_d[:, :, :, h_indices, :]
        result = x_dh[:, :, :, :, w_indices]

        return result

    else:
        raise NotImplementedError(f"Nearest interpolation not supported for {ndim}D tensors")


def _nearest_exact_interpolate(x, target_size):
    """Helper for nearest-exact interpolation.

    Uses more mathematically consistent indexing than standard nearest:
    index = floor((i + 0.5) * (input_size / output_size))

    This matches PyTorch's 'nearest-exact' mode.
    """
    ndim = len(x.shape)

    if ndim == 3:  # NCL (1D)
        N, C, L = x.shape
        target_l = target_size[0]

        # Create coordinate grid with 0.5 offset for exact nearest
        scale = L / target_l
        l_indices = mx.floor((mx.arange(target_l) + 0.5) * scale).astype(mx.int32)
        l_indices = mx.clip(l_indices, 0, L - 1)

        # Gather values
        result = x[:, :, l_indices]

        return result

    elif ndim == 4:  # NCHW (2D)
        N, C, H, W = x.shape
        target_h, target_w = target_size

        # Create coordinate grids with 0.5 offset
        h_scale = H / target_h
        w_scale = W / target_w
        h_indices = mx.floor((mx.arange(target_h) + 0.5) * h_scale).astype(mx.int32)
        w_indices = mx.floor((mx.arange(target_w) + 0.5) * w_scale).astype(mx.int32)
        h_indices = mx.clip(h_indices, 0, H - 1)
        w_indices = mx.clip(w_indices, 0, W - 1)

        # Gather values
        x_h = x[:, :, h_indices, :]
        result = x_h[:, :, :, w_indices]

        return result

    elif ndim == 5:  # NCDHW (3D)
        N, C, D, H, W = x.shape
        target_d, target_h, target_w = target_size

        # Create coordinate grids with 0.5 offset
        d_scale = D / target_d
        h_scale = H / target_h
        w_scale = W / target_w
        d_indices = mx.floor((mx.arange(target_d) + 0.5) * d_scale).astype(mx.int32)
        h_indices = mx.floor((mx.arange(target_h) + 0.5) * h_scale).astype(mx.int32)
        w_indices = mx.floor((mx.arange(target_w) + 0.5) * w_scale).astype(mx.int32)
        d_indices = mx.clip(d_indices, 0, D - 1)
        h_indices = mx.clip(h_indices, 0, H - 1)
        w_indices = mx.clip(w_indices, 0, W - 1)

        # Gather values - index each dimension sequentially
        x_d = x[:, :, d_indices, :, :]
        x_dh = x_d[:, :, :, h_indices, :]
        result = x_dh[:, :, :, :, w_indices]

        return result

    else:
        raise NotImplementedError(f"Nearest-exact interpolation not supported for {ndim}D tensors")


def _bilinear_interpolate(x, target_size, align_corners=False):
    """Helper for bilinear interpolation."""
    if len(x.shape) != 4:
        raise NotImplementedError("Only 4D tensors supported for bilinear interpolation")

    N, C, H, W = x.shape
    target_h, target_w = target_size

    # Create coordinate grids for target
    # Map target coordinates to source coordinates
    y_coords = mx.arange(target_h, dtype=mx.float32)
    x_coords = mx.arange(target_w, dtype=mx.float32)

    if align_corners:
        # Corner-aligned: map [0, target_size-1] to [0, input_size-1]
        if target_h > 1:
            y_src = y_coords * ((H - 1) / (target_h - 1))
        else:
            y_src = mx.zeros_like(y_coords)

        if target_w > 1:
            x_src = x_coords * ((W - 1) / (target_w - 1))
        else:
            x_src = mx.zeros_like(x_coords)
    else:
        # Center-aligned: pixels are centers of cells
        y_scale = H / target_h
        x_scale = W / target_w
        y_src = (y_coords + 0.5) * y_scale - 0.5
        x_src = (x_coords + 0.5) * x_scale - 0.5

    # Clamp to valid range
    y_src = mx.clip(y_src, 0, H - 1)
    x_src = mx.clip(x_src, 0, W - 1)

    # Get integer and fractional parts
    y0 = mx.floor(y_src).astype(mx.int32)
    x0 = mx.floor(x_src).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H - 1)
    x1 = mx.minimum(x0 + 1, W - 1)

    # Fractional parts for interpolation weights
    fy = y_src - y0.astype(mx.float32)
    fx = x_src - x0.astype(mx.float32)

    # Reshape for broadcasting: fy is (target_h,), fx is (target_w,)
    fy = mx.reshape(fy, (1, 1, target_h, 1))
    fx = mx.reshape(fx, (1, 1, 1, target_w))

    # Gather the 4 corner values
    # x is (N, C, H, W), we need to index with y0, y1, x0, x1
    v00 = x[:, :, y0, :][:, :, :, x0]  # top-left
    v01 = x[:, :, y0, :][:, :, :, x1]  # top-right
    v10 = x[:, :, y1, :][:, :, :, x0]  # bottom-left
    v11 = x[:, :, y1, :][:, :, :, x1]  # bottom-right

    # Bilinear interpolation
    result = v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx + v10 * fy * (1 - fx) + v11 * fy * fx

    return result


def _linear_interpolate(x, target_size, align_corners=False):
    """Helper for 1D linear interpolation."""
    if len(x.shape) != 3:
        raise NotImplementedError("Only 3D tensors supported for linear interpolation")

    N, C, L = x.shape
    target_l = target_size[0]

    # Create coordinate grid
    coords = mx.arange(target_l, dtype=mx.float32)

    if align_corners:
        if target_l > 1:
            src = coords * ((L - 1) / (target_l - 1))
        else:
            src = mx.zeros_like(coords)
    else:
        scale = L / target_l
        src = (coords + 0.5) * scale - 0.5

    # Clamp to valid range
    src = mx.clip(src, 0, L - 1)

    # Get integer and fractional parts
    idx0 = mx.floor(src).astype(mx.int32)
    idx1 = mx.minimum(idx0 + 1, L - 1)
    frac = src - idx0.astype(mx.float32)

    # Reshape for broadcasting
    frac = mx.reshape(frac, (1, 1, target_l))

    # Gather values
    v0 = x[:, :, idx0]
    v1 = x[:, :, idx1]

    # Linear interpolation
    result = v0 * (1 - frac) + v1 * frac
    return result


def _bicubic_kernel(x):
    """Compute bicubic interpolation kernel weights (Keys' cubic)."""
    # Keys' cubic kernel with a=-0.5 (matches PyTorch default)
    a = -0.5
    absx = mx.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx

    # For |x| <= 1
    w1 = (a + 2) * absx3 - (a + 3) * absx2 + 1
    # For 1 < |x| < 2
    w2 = a * absx3 - 5 * a * absx2 + 8 * a * absx - 4 * a

    mask1 = absx <= 1
    mask2 = (absx > 1) & (absx < 2)

    return mx.where(mask1, w1, mx.where(mask2, w2, mx.zeros_like(x)))


def _bicubic_interpolate(x, target_size, align_corners=False):
    """Helper for bicubic interpolation."""
    if len(x.shape) != 4:
        raise NotImplementedError("Only 4D tensors supported for bicubic interpolation")

    N, C, H, W = x.shape
    target_h, target_w = target_size

    # Create coordinate grids
    y_coords = mx.arange(target_h, dtype=mx.float32)
    x_coords = mx.arange(target_w, dtype=mx.float32)

    if align_corners:
        if target_h > 1:
            y_src = y_coords * ((H - 1) / (target_h - 1))
        else:
            y_src = mx.zeros_like(y_coords)
        if target_w > 1:
            x_src = x_coords * ((W - 1) / (target_w - 1))
        else:
            x_src = mx.zeros_like(x_coords)
    else:
        y_scale = H / target_h
        x_scale = W / target_w
        y_src = (y_coords + 0.5) * y_scale - 0.5
        x_src = (x_coords + 0.5) * x_scale - 0.5

    # Get base indices and fractional parts
    y_base = mx.floor(y_src).astype(mx.int32)
    x_base = mx.floor(x_src).astype(mx.int32)
    y_frac = y_src - y_base.astype(mx.float32)
    x_frac = x_src - x_base.astype(mx.float32)

    # Pad input for boundary handling
    x_padded = mx.pad(x, [(0, 0), (0, 0), (1, 2), (1, 2)], mode="edge")

    # Adjust base indices for padding
    y_base = y_base + 1
    x_base = x_base + 1

    # Initialize result
    result = mx.zeros((N, C, target_h, target_w), dtype=x.dtype)

    # 4x4 kernel
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            yi = mx.clip(y_base + dy, 0, H + 2)
            xi = mx.clip(x_base + dx, 0, W + 2)

            # Compute kernel weights
            wy = _bicubic_kernel(y_frac - dy)
            wx = _bicubic_kernel(x_frac - dx)

            # Reshape for broadcasting
            wy = mx.reshape(wy, (1, 1, target_h, 1))
            wx = mx.reshape(wx, (1, 1, 1, target_w))

            # Gather and accumulate
            values = x_padded[:, :, yi, :][:, :, :, xi]
            result = result + values * wy * wx

    return result


def _trilinear_interpolate(x, target_size, align_corners=False):
    """Helper for trilinear interpolation."""
    if len(x.shape) != 5:
        raise NotImplementedError("Only 5D tensors supported for trilinear interpolation")

    N, C, D, H, W = x.shape
    target_d, target_h, target_w = target_size

    # Create coordinate grids
    d_coords = mx.arange(target_d, dtype=mx.float32)
    h_coords = mx.arange(target_h, dtype=mx.float32)
    w_coords = mx.arange(target_w, dtype=mx.float32)

    if align_corners:
        if target_d > 1:
            d_src = d_coords * ((D - 1) / (target_d - 1))
        else:
            d_src = mx.zeros_like(d_coords)
        if target_h > 1:
            h_src = h_coords * ((H - 1) / (target_h - 1))
        else:
            h_src = mx.zeros_like(h_coords)
        if target_w > 1:
            w_src = w_coords * ((W - 1) / (target_w - 1))
        else:
            w_src = mx.zeros_like(w_coords)
    else:
        d_scale = D / target_d
        h_scale = H / target_h
        w_scale = W / target_w
        d_src = (d_coords + 0.5) * d_scale - 0.5
        h_src = (h_coords + 0.5) * h_scale - 0.5
        w_src = (w_coords + 0.5) * w_scale - 0.5

    # Clamp coordinates
    d_src = mx.clip(d_src, 0, D - 1)
    h_src = mx.clip(h_src, 0, H - 1)
    w_src = mx.clip(w_src, 0, W - 1)

    # Get integer and fractional parts
    d0 = mx.floor(d_src).astype(mx.int32)
    h0 = mx.floor(h_src).astype(mx.int32)
    w0 = mx.floor(w_src).astype(mx.int32)
    d1 = mx.minimum(d0 + 1, D - 1)
    h1 = mx.minimum(h0 + 1, H - 1)
    w1 = mx.minimum(w0 + 1, W - 1)

    fd = d_src - d0.astype(mx.float32)
    fh = h_src - h0.astype(mx.float32)
    fw = w_src - w0.astype(mx.float32)

    # Reshape for broadcasting: fd is (target_d,), fh is (target_h,), fw is (target_w,)
    fd = mx.reshape(fd, (1, 1, target_d, 1, 1))
    fh = mx.reshape(fh, (1, 1, 1, target_h, 1))
    fw = mx.reshape(fw, (1, 1, 1, 1, target_w))

    # Gather the 8 corner values
    v000 = x[:, :, d0, :, :][:, :, :, h0, :][:, :, :, :, w0]
    v001 = x[:, :, d0, :, :][:, :, :, h0, :][:, :, :, :, w1]
    v010 = x[:, :, d0, :, :][:, :, :, h1, :][:, :, :, :, w0]
    v011 = x[:, :, d0, :, :][:, :, :, h1, :][:, :, :, :, w1]
    v100 = x[:, :, d1, :, :][:, :, :, h0, :][:, :, :, :, w0]
    v101 = x[:, :, d1, :, :][:, :, :, h0, :][:, :, :, :, w1]
    v110 = x[:, :, d1, :, :][:, :, :, h1, :][:, :, :, :, w0]
    v111 = x[:, :, d1, :, :][:, :, :, h1, :][:, :, :, :, w1]

    # Trilinear interpolation
    result = (
        v000 * (1 - fd) * (1 - fh) * (1 - fw)
        + v001 * (1 - fd) * (1 - fh) * fw
        + v010 * (1 - fd) * fh * (1 - fw)
        + v011 * (1 - fd) * fh * fw
        + v100 * fd * (1 - fh) * (1 - fw)
        + v101 * fd * (1 - fh) * fw
        + v110 * fd * fh * (1 - fw)
        + v111 * fd * fh * fw
    )

    return result


def _area_interpolate(x, target_size):
    """Helper for area interpolation (adaptive average pooling approach)."""
    ndim = len(x.shape)

    if ndim == 4:  # 4D: NCHW
        N, C, H, W = x.shape
        target_h, target_w = target_size

        # Collect all mean values and stack at the end
        rows = []
        for i in range(target_h):
            row = []
            for j in range(target_w):
                # Compute source region boundaries
                h_start = int(i * H / target_h)
                h_end = int((i + 1) * H / target_h)
                w_start = int(j * W / target_w)
                w_end = int((j + 1) * W / target_w)

                # Handle edge case where start == end
                if h_end <= h_start:
                    h_end = h_start + 1
                if w_end <= w_start:
                    w_end = w_start + 1

                # Extract region and compute mean
                region = x[:, :, h_start:h_end, w_start:w_end]
                mean_val = mx.mean(region, axis=(2, 3), keepdims=True)
                row.append(mean_val)
            rows.append(mx.concatenate(row, axis=3))
        result = mx.concatenate(rows, axis=2)
        return result

    elif ndim == 3:  # 3D: NCL
        N, C, L = x.shape
        target_l = target_size[0]

        cols = []
        for i in range(target_l):
            l_start = int(i * L / target_l)
            l_end = int((i + 1) * L / target_l)
            if l_end <= l_start:
                l_end = l_start + 1
            region = x[:, :, l_start:l_end]
            mean_val = mx.mean(region, axis=2, keepdims=True)
            cols.append(mean_val)
        result = mx.concatenate(cols, axis=2)
        return result

    elif ndim == 5:  # 5D: NCDHW
        N, C, D, H, W = x.shape
        target_d, target_h, target_w = target_size

        # Collect all mean values and stack at the end
        depth_slices = []
        for d in range(target_d):
            d_start = int(d * D / target_d)
            d_end = int((d + 1) * D / target_d)
            if d_end <= d_start:
                d_end = d_start + 1

            rows = []
            for i in range(target_h):
                row = []
                for j in range(target_w):
                    # Compute source region boundaries
                    h_start = int(i * H / target_h)
                    h_end = int((i + 1) * H / target_h)
                    w_start = int(j * W / target_w)
                    w_end = int((j + 1) * W / target_w)

                    # Handle edge case where start == end
                    if h_end <= h_start:
                        h_end = h_start + 1
                    if w_end <= w_start:
                        w_end = w_start + 1

                    # Extract region and compute mean
                    region = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    mean_val = mx.mean(region, axis=(2, 3, 4), keepdims=True)
                    row.append(mean_val)
                rows.append(mx.concatenate(row, axis=4))
            depth_slices.append(mx.concatenate(rows, axis=3))
        result = mx.concatenate(depth_slices, axis=2)
        return result

    else:
        raise NotImplementedError(f"Area interpolation not supported for {ndim}D tensors")


def upsample(
    input: Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """
    Upsample input tensor (deprecated, use interpolate instead).

    This function is deprecated. Use F.interpolate instead.

    Args:
        input: Input tensor
        size: Output spatial size
        scale_factor: Multiplier for spatial size
        mode: Upsampling mode ('nearest', 'linear', 'bilinear', 'trilinear')
        align_corners: Alignment mode

    Returns:
        Upsampled tensor
    """
    warnings.warn(
        "nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return interpolate(
        input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )


def upsample_nearest(
    input: Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
) -> Tensor:
    """
    Upsample input using nearest neighbor interpolation (deprecated).

    This function is deprecated. Use F.interpolate with mode='nearest' instead.

    Args:
        input: Input tensor
        size: Output spatial size
        scale_factor: Multiplier for spatial size

    Returns:
        Upsampled tensor
    """
    warnings.warn(
        "nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")


def upsample_bilinear(
    input: Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
) -> Tensor:
    """
    Upsample input using bilinear interpolation (deprecated).

    This function is deprecated. Use F.interpolate with mode='bilinear' instead.

    Args:
        input: Input tensor of shape (N, C, H, W)
        size: Output spatial size (H_out, W_out)
        scale_factor: Multiplier for spatial size

    Returns:
        Upsampled tensor
    """
    warnings.warn(
        "nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # PyTorch's upsample_bilinear uses align_corners=True by default
    return interpolate(
        input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
    )


# Re-export scaled_dot_product_attention from attention layers
from .layers.attention import scaled_dot_product_attention

# =============================================================================
# Other Utility Functions
# =============================================================================


def normalize(
    input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out: Optional[Tensor] = None
) -> Tensor:
    """
    Normalize input along a dimension.

    Args:
        input: Input tensor
        p: Norm degree
        dim: Dimension to normalize over
        eps: Small constant for numerical stability
        out: Output tensor (not supported, for API compatibility)

    Returns:
        Normalized tensor
    """
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    if p == 2.0:
        norm = mx.sqrt(mx.sum(input._mlx_array**2, axis=dim, keepdims=True) + eps)
    elif p == 1.0:
        norm = mx.sum(mx.abs(input._mlx_array), axis=dim, keepdims=True) + eps
    else:
        norm = (
            mx.power(
                mx.sum(mx.power(mx.abs(input._mlx_array), p), axis=dim, keepdims=True), 1.0 / p
            )
            + eps
        )

    result_array = input._mlx_array / norm
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def one_hot(input: Tensor, num_classes: int = -1) -> Tensor:
    """
    Create one-hot encoding.

    Args:
        input: Tensor of class indices
        num_classes: Total number of classes

    Returns:
        One-hot encoded tensor
    """
    if num_classes < 0:
        num_classes = int(mx.max(input._mlx_array).item()) + 1

    result_array = mx.eye(num_classes)[input._mlx_array.astype(mx.int32)]
    return Tensor._from_mlx_array(result_array)


# =============================================================================
# Distance Functions
# =============================================================================


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """
    Compute cosine similarity between x1 and x2 along a dimension.

    Args:
        x1: First input tensor
        x2: Second input tensor
        dim: Dimension along which to compute similarity
        eps: Small constant for numerical stability

    Returns:
        Cosine similarity tensor

    Example:
        >>> x1 = flashlight.randn(10, 128)
        >>> x2 = flashlight.randn(10, 128)
        >>> sim = F.cosine_similarity(x1, x2)
        >>> sim.shape
        (10,)
    """
    # Compute dot product along dimension
    dot = mx.sum(x1._mlx_array * x2._mlx_array, axis=dim)

    # Compute norms
    norm1 = mx.sqrt(mx.sum(x1._mlx_array**2, axis=dim) + eps)
    norm2 = mx.sqrt(mx.sum(x2._mlx_array**2, axis=dim) + eps)

    # Cosine similarity
    result_array = dot / (norm1 * norm2)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (x1.requires_grad or x2.requires_grad):
        result.requires_grad = True

    return result


def pairwise_distance(
    x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
) -> Tensor:
    """
    Compute pairwise distance between x1 and x2.

    Args:
        x1: First input tensor of shape (N, D)
        x2: Second input tensor of shape (N, D)
        p: Norm degree (default: 2 for Euclidean)
        eps: Small constant for numerical stability
        keepdim: Whether to keep the dimension

    Returns:
        Distance tensor of shape (N,) or (N, 1) if keepdim

    Example:
        >>> x1 = flashlight.randn(10, 128)
        >>> x2 = flashlight.randn(10, 128)
        >>> dist = F.pairwise_distance(x1, x2)
    """
    diff = x1._mlx_array - x2._mlx_array

    if p == 2.0:
        # Euclidean distance
        dist = mx.sqrt(mx.sum(diff**2, axis=-1) + eps)
    elif p == 1.0:
        # Manhattan distance
        dist = mx.sum(mx.abs(diff), axis=-1)
    else:
        # General p-norm
        dist = mx.power(mx.sum(mx.power(mx.abs(diff), p), axis=-1) + eps, 1.0 / p)

    if keepdim:
        dist = mx.expand_dims(dist, axis=-1)

    result = Tensor._from_mlx_array(dist)

    if is_grad_enabled() and (x1.requires_grad or x2.requires_grad):
        result.requires_grad = True

    return result


def pdist(input: Tensor, p: float = 2.0) -> Tensor:
    """
    Compute pairwise distances between all row vectors in input.

    Args:
        input: Input tensor of shape (N, D)
        p: Norm degree

    Returns:
        Condensed distance matrix of shape (N*(N-1)/2,)
    """
    n = input.shape[0]
    if n < 2:
        return Tensor._from_mlx_array(mx.array([]))

    x = input._mlx_array

    # Compute all pairwise distances
    # Use broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
    diff = mx.expand_dims(x, axis=1) - mx.expand_dims(x, axis=0)

    if p == 2.0:
        distances = mx.sqrt(mx.sum(diff**2, axis=-1))
    elif p == 1.0:
        distances = mx.sum(mx.abs(diff), axis=-1)
    else:
        distances = mx.power(mx.sum(mx.power(mx.abs(diff), p), axis=-1), 1.0 / p)

    # Extract upper triangle (excluding diagonal)
    # Create indices for upper triangle
    indices_i = []
    indices_j = []
    for i in range(n):
        for j in range(i + 1, n):
            indices_i.append(i)
            indices_j.append(j)

    if len(indices_i) == 0:
        return Tensor._from_mlx_array(mx.array([]))

    result_list = []
    for i, j in zip(indices_i, indices_j):
        result_list.append(distances[i, j])

    result_array = mx.stack(result_list) if result_list else mx.array([])
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def cdist(
    x1: Tensor,
    x2: Tensor,
    p: float = 2.0,
    compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
) -> Tensor:
    """
    Compute pairwise distances between two sets of vectors.

    Args:
        x1: First input tensor of shape (B, P, M) or (P, M)
        x2: Second input tensor of shape (B, R, M) or (R, M)
        p: Norm degree
        compute_mode: Computation mode (ignored, for API compatibility)

    Returns:
        Distance matrix of shape (B, P, R) or (P, R)

    Example:
        >>> x1 = flashlight.randn(10, 128)  # 10 vectors of dim 128
        >>> x2 = flashlight.randn(20, 128)  # 20 vectors of dim 128
        >>> dist = F.cdist(x1, x2)
        >>> dist.shape
        (10, 20)
    """
    # Handle batched and non-batched cases
    x1_arr = x1._mlx_array
    x2_arr = x2._mlx_array

    if len(x1_arr.shape) == 2:
        # (P, M) case
        x1_arr = mx.expand_dims(x1_arr, axis=1)  # (P, 1, M)
        x2_arr = mx.expand_dims(x2_arr, axis=0)  # (1, R, M)

        diff = x1_arr - x2_arr  # (P, R, M)

        if p == 2.0:
            distances = mx.sqrt(mx.sum(diff**2, axis=-1))
        elif p == 1.0:
            distances = mx.sum(mx.abs(diff), axis=-1)
        else:
            distances = mx.power(mx.sum(mx.power(mx.abs(diff), p), axis=-1), 1.0 / p)
    else:
        # (B, P, M) case
        x1_arr = mx.expand_dims(x1_arr, axis=2)  # (B, P, 1, M)
        x2_arr = mx.expand_dims(x2_arr, axis=1)  # (B, 1, R, M)

        diff = x1_arr - x2_arr  # (B, P, R, M)

        if p == 2.0:
            distances = mx.sqrt(mx.sum(diff**2, axis=-1))
        elif p == 1.0:
            distances = mx.sum(mx.abs(diff), axis=-1)
        else:
            distances = mx.power(mx.sum(mx.power(mx.abs(diff), p), axis=-1), 1.0 / p)

    result = Tensor._from_mlx_array(distances)

    if is_grad_enabled() and (x1.requires_grad or x2.requires_grad):
        result.requires_grad = True

    return result


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Triplet margin loss for metric learning.

    Loss = max(d(a, p) - d(a, n) + margin, 0)

    Args:
        anchor: Anchor samples (N, D)
        positive: Positive samples (N, D)
        negative: Negative samples (N, D)
        margin: Margin value
        p: Norm degree for distance
        eps: Small constant for stability
        swap: Use distance swap (harder mining)
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor

    Example:
        >>> anchor = flashlight.randn(10, 128)
        >>> positive = flashlight.randn(10, 128)
        >>> negative = flashlight.randn(10, 128)
        >>> loss = F.triplet_margin_loss(anchor, positive, negative)
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    # Compute distances
    d_ap = pairwise_distance(anchor, positive, p=p, eps=eps)
    d_an = pairwise_distance(anchor, negative, p=p, eps=eps)

    if swap:
        # Use harder negative: min(d(a, n), d(p, n))
        d_pn = pairwise_distance(positive, negative, p=p, eps=eps)
        d_an = Tensor._from_mlx_array(mx.minimum(d_an._mlx_array, d_pn._mlx_array))

    # Triplet loss
    loss = mx.maximum(d_ap._mlx_array - d_an._mlx_array + margin, 0.0)

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and (
        anchor.requires_grad or positive.requires_grad or negative.requires_grad
    ):
        result.requires_grad = True

    return result


def margin_ranking_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Margin ranking loss.

    Loss = max(0, -y * (x1 - x2) + margin)

    Args:
        input1: First input tensor
        input2: Second input tensor
        target: Target tensor with values 1 or -1
        margin: Margin value
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    loss = mx.maximum(0.0, -target._mlx_array * (input1._mlx_array - input2._mlx_array) + margin)

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and (input1.requires_grad or input2.requires_grad):
        result.requires_grad = True

    return result


def hinge_embedding_loss(
    input: Tensor,
    target: Tensor,
    margin: float = 1.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Hinge embedding loss.

    Loss = x if y == 1
         = max(0, margin - x) if y == -1

    Args:
        input: Input tensor
        target: Target tensor with values 1 or -1
        margin: Margin for negative samples
        size_average: Deprecated (use reduction). If True, average loss; if False, sum.
        reduce: Deprecated (use reduction). If False, return unreduced loss.
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)
    # For y == 1: loss = x
    # For y == -1: loss = max(0, margin - x)
    loss = mx.where(
        target._mlx_array == 1, input._mlx_array, mx.maximum(0.0, margin - input._mlx_array)
    )

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def huber_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    delta: float = 1.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Huber loss (smooth L1 loss variant).

    loss = 0.5 * (x - y)^2                  if |x - y| < delta
         = delta * (|x - y| - 0.5 * delta)  otherwise

    Args:
        input: Predicted values
        target: Target values
        reduction: 'none', 'mean', or 'sum'
        delta: Threshold for switching between quadratic and linear loss
        weight: Manual rescaling weight for each element

    Returns:
        Loss tensor
    """
    diff = mx.abs(input._mlx_array - target._mlx_array)
    loss = mx.where(diff < delta, 0.5 * diff * diff, delta * (diff - 0.5 * delta))

    if weight is not None:
        loss = loss * weight._mlx_array

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def kl_div(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    """
    Kullback-Leibler divergence loss.

    KL(target || input) = target * (log(target) - input)  if log_target=False
                        = exp(target) * (target - input)  if log_target=True

    Note: input should be log-probabilities, target should be probabilities
    (or log-probabilities if log_target=True).

    Args:
        input: Log-probabilities
        target: Target probabilities (or log-probs if log_target=True)
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        reduction: 'none', 'mean', 'batchmean', or 'sum'
        log_target: If True, target is in log-space

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    if log_target:
        # target is log-probabilities
        loss = mx.exp(target._mlx_array) * (target._mlx_array - input._mlx_array)
    else:
        # target is probabilities
        # Use xlogy pattern: x * log(x) returns 0 when x=0, avoiding log(0) issues
        target_arr = target._mlx_array
        xlogy_term = mx.where(
            target_arr == 0, mx.zeros_like(target_arr), target_arr * mx.log(target_arr)
        )
        loss = xlogy_term - target_arr * input._mlx_array

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "batchmean":
        # Divide by batch size only
        loss = mx.sum(loss) / input.shape[0]
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Soft margin loss (logistic loss).

    loss = log(1 + exp(-y * x))

    Args:
        input: Input tensor
        target: Target tensor with values 1 or -1
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    # log(1 + exp(-y * x)) = softplus(-y * x)
    import mlx.nn as nn

    loss = nn.softplus(-target._mlx_array * input._mlx_array)

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: int = 0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Cosine embedding loss.

    loss = 1 - cos(x1, x2)                    if y = 1
         = max(0, cos(x1, x2) - margin)       if y = -1

    Args:
        input1: First input tensor (N, D)
        input2: Second input tensor (N, D)
        target: Target tensor with values 1 or -1 (N,)
        margin: Margin for negative pairs
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    # Compute cosine similarity
    cos_sim = cosine_similarity(input1, input2, dim=1)._mlx_array

    # Compute loss based on target
    loss = mx.where(
        target._mlx_array == 1, 1 - cos_sim, mx.maximum(mx.array(0.0), cos_sim - margin)
    )

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and (input1.requires_grad or input2.requires_grad):
        result.requires_grad = True

    return result


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    """
    Gaussian negative log-likelihood loss.

    loss = 0.5 * (log(var) + (input - target)^2 / var)

    If full=True, adds the constant term 0.5 * log(2 * pi).

    Args:
        input: Predicted mean
        target: Target values
        var: Predicted variance (must be positive)
        full: Include constant term
        eps: Small value for numerical stability
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    # Ensure variance is positive
    var_arr = mx.maximum(var._mlx_array, eps)

    diff = input._mlx_array - target._mlx_array
    loss = 0.5 * (mx.log(var_arr) + diff * diff / var_arr)

    if full:
        import math

        loss = loss + 0.5 * math.log(2 * math.pi)

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and (input.requires_grad or var.requires_grad):
        result.requires_grad = True

    return result


def poisson_nll_loss(
    input: Tensor,
    target: Tensor,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    eps: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Poisson negative log-likelihood loss.

    If log_input=True:
        loss = exp(input) - target * input
    If log_input=False:
        loss = input - target * log(input + eps)

    Args:
        input: Predicted rate (or log-rate if log_input=True)
        target: Target counts
        log_input: If True, input is log(lambda)
        full: Include Stirling approximation term
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        eps: Small value for numerical stability
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    if log_input:
        loss = mx.exp(input._mlx_array) - target._mlx_array * input._mlx_array
    else:
        loss = input._mlx_array - target._mlx_array * mx.log(input._mlx_array + eps)

    if full:
        # Stirling approximation: log(n!) ≈ n*log(n) - n + 0.5*log(2*pi*n)
        import math

        target_arr = target._mlx_array
        approx = (
            target_arr * mx.log(mx.maximum(target_arr, 1))
            - target_arr
            + 0.5 * mx.log(2 * math.pi * mx.maximum(target_arr, 1))
        )
        loss = loss + approx

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Multi-class hinge loss (SVM loss).

    loss(x, y) = sum_j(max(0, margin - x[y] + x[j])^p) / C
    where j != y

    Args:
        input: Input tensor (N, C)
        target: Target class indices (N,)
        p: Exponent (1 or 2)
        margin: Margin value
        weight: Manual weight for each class
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    N, C = input.shape
    x = input._mlx_array
    y = target._mlx_array

    # Get the correct class scores
    batch_indices = mx.arange(N)
    correct_scores = x[batch_indices, y]  # (N,)
    correct_scores = mx.expand_dims(correct_scores, axis=1)  # (N, 1)

    # Compute margins
    margins = mx.maximum(mx.array(0.0), margin - correct_scores + x)  # (N, C)

    # Zero out the correct class
    one_hot_target = mx.zeros((N, C))
    one_hot_target = mx.where(mx.arange(C) == mx.expand_dims(y, axis=1), mx.array(0.0), margins)

    # Apply exponent
    if p == 2:
        one_hot_target = one_hot_target * one_hot_target

    # Apply class weights if provided
    if weight is not None:
        one_hot_target = one_hot_target * weight._mlx_array

    # Sum over classes and divide by total number of classes
    loss = mx.sum(one_hot_target, axis=1) / C

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def multilabel_soft_margin_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Multi-label one-versus-all loss using soft margin.

    loss = -1/C * sum_c(y_c * log(sigmoid(x_c)) + (1 - y_c) * log(1 - sigmoid(x_c)))

    This is equivalent to binary cross entropy with logits applied independently
    to each class.

    Args:
        input: Input tensor (N, C)
        target: Target tensor (N, C) with values 0 or 1
        weight: Manual weight for each class (C,)
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    reduction = _get_legacy_reduction(size_average, reduce, reduction)

    import mlx.nn as nn

    x = input._mlx_array
    y = target._mlx_array

    # Binary cross entropy with logits for each class
    # loss = -y * log(sigmoid(x)) - (1-y) * log(1 - sigmoid(x))
    # Numerically stable: max(x, 0) - x*y + log(1 + exp(-|x|))
    loss = mx.maximum(x, 0) - x * y + nn.softplus(-mx.abs(x))

    if weight is not None:
        loss = loss * weight._mlx_array

    # Apply reduction - mean/sum over ALL elements (not just classes)
    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)
    elif reduction != "none":
        raise ValueError(f"Invalid reduction: {reduction}")

    result = Tensor._from_mlx_array(loss)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


# =============================================================================
# Additional Functional APIs
# =============================================================================


def gumbel_softmax(
    logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1
) -> Tensor:
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    The Gumbel-Softmax distribution is a continuous distribution that approximates
    samples from a categorical distribution. It's useful for learning discrete
    latent variables with gradient descent.

    Args:
        logits: [..., num_features] unnormalized log probabilities
        tau: Temperature parameter (lower = closer to one-hot)
        hard: If True, discretizes to one-hot vectors but keeps gradients
        eps: Small constant for numerical stability
        dim: Dimension along which to apply softmax

    Returns:
        Sampled tensor of same shape as logits

    Example:
        >>> logits = torch.randn(10, 5)
        >>> soft = F.gumbel_softmax(logits, tau=0.5)
        >>> hard = F.gumbel_softmax(logits, tau=0.5, hard=True)
    """
    # Sample Gumbel noise
    U = mx.random.uniform(shape=logits._mlx_array.shape, dtype=logits._mlx_array.dtype)
    gumbel = -mx.log(-mx.log(U + eps) + eps)

    # Add Gumbel noise and apply softmax with temperature
    y = logits._mlx_array + gumbel
    y_soft = mx.softmax(y / tau, axis=dim)

    if hard:
        # Discretize but keep gradients through softmax
        index = mx.argmax(y_soft, axis=dim, keepdims=True)
        y_hard = mx.zeros_like(y_soft)
        # One-hot encoding
        y_hard = mx.take_along_axis(mx.ones_like(y_hard), index, axis=dim)
        # Straight-through estimator: forward pass uses hard, backward uses soft
        y = y_hard - mx.stop_gradient(y_soft) + y_soft
    else:
        y = y_soft

    result = Tensor._from_mlx_array(y)

    if is_grad_enabled() and logits.requires_grad:
        result.requires_grad = True

    return result


def bilinear(
    input1: Tensor, input2: Tensor, weight: Tensor, bias: Optional[Tensor] = None
) -> Tensor:
    """
    Apply a bilinear transformation to the input tensors.

    Computes: y = x1^T A x2 + b

    Args:
        input1: First input of shape (N, *, in1_features)
        input2: Second input of shape (N, *, in2_features)
        weight: Weight tensor of shape (out_features, in1_features, in2_features)
        bias: Optional bias of shape (out_features,)

    Returns:
        Output of shape (N, *, out_features)
    """
    x1 = input1._mlx_array
    x2 = input2._mlx_array
    w = weight._mlx_array

    # x1: [..., in1], x2: [..., in2], w: [out, in1, in2]
    # Result: x1^T @ W @ x2 for each output feature
    # Using einsum: 'bi,oij,bj->bo' for batch case

    # Get shapes
    batch_shape = x1.shape[:-1]
    in1 = x1.shape[-1]
    in2 = x2.shape[-1]
    out_features = w.shape[0]

    # Reshape for batch processing
    x1_flat = mx.reshape(x1, (-1, in1))  # [B, in1]
    x2_flat = mx.reshape(x2, (-1, in2))  # [B, in2]

    # Compute bilinear: for each output feature o: x1^T @ W[o] @ x2
    # x1_flat: [B, in1], w: [out, in1, in2], x2_flat: [B, in2]
    # Result: result[b, o] = sum_i sum_j x1[b, i] * w[o, i, j] * x2[b, j]

    # Expand to 4D for broadcasting: x1[B, 1, in1, 1], w[1, out, in1, in2], x2[B, 1, 1, in2]
    x1_4d = x1_flat[:, None, :, None]  # [B, 1, in1, 1]
    w_4d = w[None, :, :, :]  # [1, out, in1, in2]
    x2_4d = x2_flat[:, None, None, :]  # [B, 1, 1, in2]

    # Multiply all three and sum over in1 and in2 dimensions
    product = x1_4d * w_4d * x2_4d  # [B, out, in1, in2]
    result = mx.sum(product, axis=(2, 3))  # [B, out]

    if bias is not None:
        result = result + bias._mlx_array

    # Reshape back to original batch shape
    result = mx.reshape(result, batch_shape + (out_features,))

    output = Tensor._from_mlx_array(result)

    if is_grad_enabled() and (input1.requires_grad or input2.requires_grad or weight.requires_grad):
        output.requires_grad = True

    return output


def affine_grid(theta: Tensor, size: List[int], align_corners: Optional[bool] = None) -> Tensor:
    """
    Generate a 2D or 3D affine transformation grid for grid_sample.

    Args:
        theta: Affine transformation matrix of shape (N, 2, 3) for 2D or (N, 3, 4) for 3D
        size: Target output size (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        align_corners: If True, align corners of input and output grids

    Returns:
        Grid tensor of shape (N, H, W, 2) for 2D or (N, D, H, W, 3) for 3D
    """
    # Default align_corners to True if not specified
    if align_corners is None:
        align_corners = True
    N = size[0]

    if len(size) == 4:
        # 2D case
        _, _, H, W = size

        if align_corners:
            y_range = mx.linspace(-1, 1, H)
            x_range = mx.linspace(-1, 1, W)
        else:
            y_range = mx.linspace(-1 + 1 / H, 1 - 1 / H, H)
            x_range = mx.linspace(-1 + 1 / W, 1 - 1 / W, W)

        # Create meshgrid
        grid_y, grid_x = mx.meshgrid(y_range, x_range, indexing="ij")
        grid = mx.stack([grid_x, grid_y, mx.ones_like(grid_x)], axis=-1)  # [H, W, 3]
        grid = mx.reshape(grid, (H * W, 3))  # [H*W, 3]

        # Apply affine transformation: grid @ theta^T
        # theta: [N, 2, 3]
        # grid: [H*W, 3]
        # result: [N, H*W, 2]
        theta_np = theta._mlx_array
        result = mx.matmul(grid, mx.transpose(theta_np, (0, 2, 1)))  # [N, H*W, 2]
        result = mx.reshape(result, (N, H, W, 2))

    elif len(size) == 5:
        # 3D case
        _, _, D, H, W = size

        if align_corners:
            z_range = mx.linspace(-1, 1, D)
            y_range = mx.linspace(-1, 1, H)
            x_range = mx.linspace(-1, 1, W)
        else:
            z_range = mx.linspace(-1 + 1 / D, 1 - 1 / D, D)
            y_range = mx.linspace(-1 + 1 / H, 1 - 1 / H, H)
            x_range = mx.linspace(-1 + 1 / W, 1 - 1 / W, W)

        grid_z, grid_y, grid_x = mx.meshgrid(z_range, y_range, x_range, indexing="ij")
        grid = mx.stack([grid_x, grid_y, grid_z, mx.ones_like(grid_x)], axis=-1)
        grid = mx.reshape(grid, (D * H * W, 4))

        theta_np = theta._mlx_array
        result = mx.matmul(grid, mx.transpose(theta_np, (0, 2, 1)))
        result = mx.reshape(result, (N, D, H, W, 3))

    else:
        raise ValueError(f"Expected size to have 4 or 5 elements, got {len(size)}")

    output = Tensor._from_mlx_array(result)

    if is_grad_enabled() and theta.requires_grad:
        output.requires_grad = True

    return output


def local_response_norm(
    input: Tensor, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0
) -> Tensor:
    """
    Apply local response normalization over an input signal.

    LRN normalizes over local input regions. The formula is:
        b_c = a_c / (k + alpha * sum(a_j^2))^beta

    where the sum is over the local region centered at channel c.

    Args:
        input: Input tensor of shape (N, C, ...)
        size: Number of channels to normalize across (should be odd)
        alpha: Multiplicative factor
        beta: Exponent
        k: Additive factor

    Returns:
        Normalized tensor of same shape as input
    """
    x = input._mlx_array
    dim = x.ndim

    if dim < 3:
        raise ValueError("Expected 3D or higher tensor")

    # Compute squared values
    x_sq = mx.square(x)

    # Create averaging kernel
    half_size = size // 2
    C = x.shape[1]

    # Compute sum of squares over local regions using padding and slicing
    # Pad the channel dimension
    if dim == 4:  # NCHW
        padded = mx.pad(x_sq, [(0, 0), (half_size, half_size), (0, 0), (0, 0)])
    elif dim == 3:  # NCL
        padded = mx.pad(x_sq, [(0, 0), (half_size, half_size), (0, 0)])
    else:
        padded = mx.pad(x_sq, [(0, 0), (half_size, half_size)] + [(0, 0)] * (dim - 2))

    # Sum over the window - use list comprehension for better accuracy
    slices_list = []
    for i in range(size):
        slices = [slice(None)] * dim
        slices[1] = slice(i, i + C)
        slices_list.append(padded[tuple(slices)])

    # Stack and sum
    sum_sq = mx.sum(mx.stack(slices_list, axis=0), axis=0)

    # Normalization: x / (k + (alpha/size) * sum)^beta
    # Note: PyTorch divides alpha by size (see PyTorch docs)
    norm = mx.power(k + (alpha / size) * sum_sq, beta)
    result = x / norm

    output = Tensor._from_mlx_array(result)

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output


# =============================================================================
# Fold/Unfold Operations
# =============================================================================


def unfold(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tensor:
    """
    Extract sliding local blocks from a batched input tensor.

    This operation extracts patches from the input tensor using a sliding window
    and returns them as columns in the output tensor.

    Args:
        input: Input tensor of shape (N, C, H, W)
        kernel_size: Size of the sliding blocks (kH, kW)
        dilation: Dilation of the sliding blocks (default: 1)
        padding: Padding applied to input (default: 0)
        stride: Stride of the sliding blocks (default: 1)

    Returns:
        Output tensor of shape (N, C * kH * kW, L)
        where L = ((H + 2*padH - dilH*(kH-1) - 1) / strH + 1) *
                  ((W + 2*padW - dilW*(kW-1) - 1) / strW + 1)

    Example:
        >>> x = torch.randn(1, 3, 10, 12)
        >>> output = F.unfold(x, kernel_size=(3, 3))
        >>> output.shape
        torch.Size([1, 27, 80])
    """
    # Handle int inputs
    if isinstance(kernel_size, int):
        kH, kW = kernel_size, kernel_size
    else:
        kH, kW = kernel_size

    if isinstance(dilation, int):
        dH, dW = dilation, dilation
    else:
        dH, dW = dilation

    if isinstance(padding, int):
        padH, padW = padding, padding
    else:
        padH, padW = padding

    if isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = stride

    x = input._mlx_array
    N, C, H, W = x.shape

    # Apply padding if needed
    if padH > 0 or padW > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (padH, padH), (padW, padW)])
        H = H + 2 * padH
        W = W + 2 * padW

    # Calculate output spatial dimensions
    H_out = (H - dH * (kH - 1) - 1) // sH + 1
    W_out = (W - dW * (kW - 1) - 1) // sW + 1
    L = H_out * W_out

    # Extract patches organized by channel first, then kernel position
    # Output should be [C0_kH0_kW0, C0_kH0_kW1, ..., C0_kH(kH-1)_kW(kW-1), C1_kH0_kW0, ...]
    patches = []
    for c in range(C):
        for i in range(kH):
            for j in range(kW):
                # Calculate the slice positions with dilation
                h_start = i * dH
                w_start = j * dW
                # Extract all valid positions for this kernel element and channel
                # x is [N, C, H, W]
                patch = x[
                    :,
                    c : c + 1,
                    h_start : h_start + H_out * sH : sH,
                    w_start : w_start + W_out * sW : sW,
                ]
                # Reshape to [N, L]
                patch = mx.reshape(patch, (N, H_out * W_out))
                patches.append(patch)

    # Stack patches: [C*kH*kW, N, L] -> [N, C*kH*kW, L]
    result = mx.stack(patches, axis=1)  # [N, C*kH*kW, L]

    output = Tensor._from_mlx_array(result)

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output


def fold(
    input: Tensor,
    output_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tensor:
    """
    Combine an array of sliding local blocks into a large containing tensor.

    This is the inverse operation of unfold. It sums overlapping values.

    Args:
        input: Input tensor of shape (N, C * kH * kW, L)
        output_size: Shape of the spatial dimensions of the output (H, W)
        kernel_size: Size of the sliding blocks (kH, kW)
        dilation: Dilation of the sliding blocks (default: 1)
        padding: Padding applied to original input (default: 0)
        stride: Stride of the sliding blocks (default: 1)

    Returns:
        Output tensor of shape (N, C, H, W)

    Example:
        >>> x = torch.randn(1, 27, 80)  # 27 = 3 * 3 * 3 (C * kH * kW)
        >>> output = F.fold(x, output_size=(10, 12), kernel_size=(3, 3))
        >>> output.shape
        torch.Size([1, 3, 10, 12])
    """
    # Handle int inputs
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    else:
        H_out, W_out = output_size

    if isinstance(kernel_size, int):
        kH, kW = kernel_size, kernel_size
    else:
        kH, kW = kernel_size

    if isinstance(dilation, int):
        dH, dW = dilation, dilation
    else:
        dH, dW = dilation

    if isinstance(padding, int):
        padH, padW = padding, padding
    else:
        padH, padW = padding

    if isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = stride

    x = input._mlx_array
    N, C_k, L = x.shape

    # Calculate padded dimensions
    H_padded = H_out + 2 * padH
    W_padded = W_out + 2 * padW

    # Calculate L dimensions
    L_H = (H_padded - dH * (kH - 1) - 1) // sH + 1
    L_W = (W_padded - dW * (kW - 1) - 1) // sW + 1

    # Infer C from C_k and kernel size
    C = C_k // (kH * kW)

    # Reshape input: [N, C*kH*kW, L] -> [N, C, kH, kW, L_H, L_W]
    x = mx.reshape(x, (N, C, kH, kW, L_H, L_W))

    # Initialize output (padded) in NCHW format
    # We'll accumulate by creating sparse arrays and summing
    output = mx.zeros((N, C, H_padded, W_padded))

    # For each kernel position, create a sparse contribution array and add it
    for ki in range(kH):
        for kj in range(kW):
            # Get the contribution for this kernel position: [N, C, L_H, L_W]
            contrib = x[:, :, ki, kj, :, :]

            # Calculate target positions with dilation
            h_start = ki * dH
            w_start = kj * dW

            # Create a zeros array and place the contribution at strided positions
            # For non-unit stride, we need to expand and interleave with zeros
            if sH == 1 and sW == 1:
                # Simple case: place directly
                h_end = h_start + L_H
                w_end = w_start + L_W
                # Create a mask-like operation by padding the contribution
                padded = mx.pad(
                    contrib,
                    [
                        (0, 0),  # N
                        (0, 0),  # C
                        (h_start, H_padded - h_end),  # H
                        (w_start, W_padded - w_end),  # W
                    ],
                )
                output = output + padded
            else:
                # For stride > 1, we need to expand with zeros between elements
                # Create expanded array with zeros
                expanded = mx.zeros((N, C, L_H * sH, L_W * sW))
                # Place values at strided positions (every sH rows, every sW cols)
                # This is done by reshaping and broadcasting
                # contrib: [N, C, L_H, L_W]
                # We want to place at positions [::sH, ::sW] in expanded

                # Create the expanded version by tiling and masking
                # Simpler approach: iterate (acceptable for small kernels)
                for li in range(L_H):
                    for lj in range(L_W):
                        val = contrib[:, :, li : li + 1, lj : lj + 1]  # [N, C, 1, 1]
                        h_pos = li * sH
                        w_pos = lj * sW
                        # Create padded single-element array
                        single_padded = mx.pad(
                            val,
                            [
                                (0, 0),  # N
                                (0, 0),  # C
                                (h_start + h_pos, H_padded - h_start - h_pos - 1),  # H
                                (w_start + w_pos, W_padded - w_start - w_pos - 1),  # W
                            ],
                        )
                        output = output + single_padded

    # Remove padding
    if padH > 0 or padW > 0:
        output = output[:, :, padH : H_padded - padH, padW : W_padded - padW]

    output_tensor = Tensor._from_mlx_array(output)

    if is_grad_enabled() and input.requires_grad:
        output_tensor.requires_grad = True

    return output_tensor


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Activations
    "relu",
    "relu6",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "silu",
    "swish",
    "leaky_relu",
    "elu",
    "softplus",
    "softsign",
    "hardswish",
    "hardsigmoid",
    "mish",
    "celu",
    "celu_",
    "selu",
    "selu_",
    "hardtanh",
    "hardtanh_",
    "hardshrink",
    "softshrink",
    "tanhshrink",
    "threshold",
    "threshold_",
    "glu",
    "logsigmoid",
    "prelu",
    "softmin",
    "rrelu",
    "rrelu_",
    # Convolution
    "conv2d",
    "conv1d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "conv_tbc",
    # Pooling
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "lp_pool1d",
    "lp_pool2d",
    "lp_pool3d",
    # Pooling with indices
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "adaptive_max_pool1d_with_indices",
    "adaptive_max_pool2d_with_indices",
    "adaptive_max_pool3d_with_indices",
    "fractional_max_pool2d",
    "fractional_max_pool2d_with_indices",
    "fractional_max_pool3d",
    "fractional_max_pool3d_with_indices",
    "max_unpool1d",
    "max_unpool2d",
    "max_unpool3d",
    # Linear
    "linear",
    # RNN cells
    "rnn_tanh_cell",
    "rnn_relu_cell",
    "lstm_cell",
    "gru_cell",
    # Dropout
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "alpha_dropout",
    "feature_alpha_dropout",
    # Normalization
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    # Loss functions
    "mse_loss",
    "l1_loss",
    "cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "smooth_l1_loss",
    "triplet_margin_loss",
    "margin_ranking_loss",
    "hinge_embedding_loss",
    "huber_loss",
    "kl_div",
    "soft_margin_loss",
    "cosine_embedding_loss",
    "gaussian_nll_loss",
    "poisson_nll_loss",
    "multi_margin_loss",
    "multilabel_soft_margin_loss",
    "ctc_loss",
    # Embedding
    "embedding",
    "embedding_bag",
    # Padding
    "pad",
    # Fold/Unfold
    "fold",
    "unfold",
    # Pixel operations
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    "native_channel_shuffle",
    # Interpolation
    "interpolate",
    "upsample",
    "upsample_nearest",
    "upsample_bilinear",
    # Attention
    "scaled_dot_product_attention",
    # Utilities
    "normalize",
    "one_hot",
    # Distance functions
    "cosine_similarity",
    "pairwise_distance",
    "pdist",
    "cdist",
    # Additional functional APIs
    "gumbel_softmax",
    "bilinear",
    "affine_grid",
    "local_response_norm",
    # In-place activations
    "relu_",
    "leaky_relu_",
    "elu_",
    # Grid sampling
    "grid_sample",
    # RMS normalization
    "rms_norm",
    # Additional losses
    "triplet_margin_with_distance_loss",
    "multilabel_margin_loss",
    # Multi-head attention forward
    "multi_head_attention_forward",
    # Type re-exports for compatibility
    "Callable",
    "Optional",
    "Union",
    "Tensor",
    "DType",
    "TYPE_CHECKING",
    "BroadcastingList1",
    "BroadcastingList2",
    "BroadcastingList3",
    "GRID_SAMPLE_INTERPOLATION_MODES",
    "GRID_SAMPLE_PADDING_MODES",
    "reproducibility_notes",
    "sparse_support_notes",
    "tf32_notes",
    # Torch function dispatch stubs
    "has_torch_function",
    "has_torch_function_unary",
    "has_torch_function_variadic",
    "handle_torch_function",
    "boolean_dispatch",
    "assert_int_or_pair",
    # Module re-exports
    "grad",
    "importlib",
    "math",
    "np",
    "warnings",
    "torch",
]


# =============================================================================
# In-place Activation Functions
# =============================================================================


def relu_(input: Tensor) -> Tensor:
    """In-place ReLU activation."""
    input._mlx_array = mx.maximum(input._mlx_array, 0)
    return input


def leaky_relu_(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    """In-place Leaky ReLU activation."""
    input._mlx_array = mx.where(
        input._mlx_array > 0, input._mlx_array, input._mlx_array * negative_slope
    )
    return input


def elu_(input: Tensor, alpha: float = 1.0) -> Tensor:
    """In-place ELU activation."""
    input._mlx_array = mx.where(
        input._mlx_array > 0, input._mlx_array, alpha * (mx.exp(input._mlx_array) - 1)
    )
    return input


# =============================================================================
# Grid Sampling
# =============================================================================


def _reflect_coord(coord, dim_size, align_corners):
    """Reflect floating-point coordinate into valid range using periodic reflection.

    Args:
        coord: Coordinate array (can be any shape), already unnormalized to pixel space
        dim_size: Size of the dimension
        align_corners: Whether align_corners is True

    Returns:
        Reflected coordinate in valid range
    """
    if dim_size == 1:
        # Edge case: single pixel, always return 0
        return mx.zeros(coord.shape, dtype=coord.dtype)

    if align_corners:
        # Valid range is [0, dim_size - 1]
        lower = 0.0
        upper = float(dim_size - 1)
    else:
        # Valid range is [-0.5, dim_size - 0.5]
        lower = -0.5
        upper = float(dim_size) - 0.5

    period = upper - lower
    if period <= 0:
        return mx.zeros(coord.shape, dtype=coord.dtype)

    # Shift to [0, period] range
    coord_shifted = coord - lower

    # Use modulo with double period for proper reflection
    double_period = 2 * period
    coord_mod = coord_shifted % double_period
    coord_mod = mx.where(coord_mod < 0, coord_mod + double_period, coord_mod)

    # Reflect: if in second half of double period, reflect back
    coord_reflected = mx.where(coord_mod > period, double_period - coord_mod, coord_mod)

    # Shift back
    result = coord_reflected + lower

    return result


def _reflect_index(idx, dim_size):
    """Reflect integer index into valid range [0, dim_size-1] using periodic reflection.

    This is used for nearest neighbor mode where we have integer indices.

    Args:
        idx: Index array (can be any shape)
        dim_size: Size of the dimension

    Returns:
        Reflected index in valid range
    """
    if dim_size == 1:
        # Edge case: single pixel, always return 0
        return mx.zeros(idx.shape, dtype=mx.int32)
    period = 2 * (dim_size - 1)
    # Handle negative values properly with modulo
    idx_pos = idx.astype(mx.float32)
    idx_mod = idx_pos % period
    idx_mod = mx.where(idx_mod < 0, idx_mod + period, idx_mod)
    # Now idx_mod is in [0, period)
    # Reflect: if idx_mod >= dim_size, then idx_ref = period - idx_mod
    idx_ref = mx.where(idx_mod >= dim_size, period - idx_mod, idx_mod)
    return mx.clip(idx_ref.astype(mx.int32), 0, dim_size - 1)


def _sample_with_indices(x, iy, ix, valid_mask, padding_mode):
    """Sample from input tensor using indices.

    Uses functional operations instead of in-place assignment.

    Args:
        x: Input tensor (N, C, H_in, W_in)
        iy: Y indices (N, H_out, W_out)
        ix: X indices (N, H_out, W_out)
        valid_mask: Boolean mask for valid samples (N, H_out, W_out)
        padding_mode: Padding mode string

    Returns:
        Sampled tensor (N, C, H_out, W_out)
    """
    N, C, H_in, W_in = x.shape
    H_out, W_out = iy.shape[1], iy.shape[2]

    # Compute flat indices for gather: idx = n * (H_in * W_in) + iy * W_in + ix
    # First, create batch indices
    batch_idx = mx.arange(N)[:, None, None]  # (N, 1, 1)
    batch_idx = mx.broadcast_to(batch_idx, (N, H_out, W_out))

    # Compute linear indices into the spatial dimensions
    spatial_idx = iy * W_in + ix  # (N, H_out, W_out)

    # For each channel, we need to gather from x[n, c, :, :] reshaped as (N, H_in*W_in)
    # Then reshape to (N, H_out, W_out)

    # Reshape x to (N, C, H_in * W_in)
    x_flat = mx.reshape(x, (N, C, H_in * W_in))

    # We need to gather for each batch element
    # spatial_idx is (N, H_out, W_out), we need to use it to index into x_flat
    # Result should be (N, C, H_out, W_out)

    # Flatten spatial_idx to (N, H_out * W_out)
    spatial_idx_flat = mx.reshape(spatial_idx, (N, H_out * W_out))

    # Gather for each batch and channel
    # Using advanced indexing: for each n, gather x_flat[n, :, spatial_idx_flat[n]]
    output_list = []
    for n in range(N):
        # x_flat[n] is (C, H_in * W_in)
        # spatial_idx_flat[n] is (H_out * W_out,)
        # We want to gather along axis=1, resulting in (C, H_out * W_out)
        gathered = mx.take(x_flat[n], spatial_idx_flat[n], axis=1)
        output_list.append(gathered)

    # Stack along batch dimension: (N, C, H_out * W_out)
    output = mx.stack(output_list, axis=0)

    # Reshape to (N, C, H_out, W_out)
    output = mx.reshape(output, (N, C, H_out, W_out))

    # Apply valid mask for 'zeros' padding mode
    if padding_mode == "zeros":
        # valid_mask is (N, H_out, W_out), broadcast to (N, C, H_out, W_out)
        valid_mask_expanded = mx.expand_dims(valid_mask, axis=1)  # (N, 1, H_out, W_out)
        valid_mask_expanded = mx.broadcast_to(
            valid_mask_expanded.astype(x.dtype), (N, C, H_out, W_out)
        )
        output = output * valid_mask_expanded

    return output


def _trilinear_sample(x, grid_x, grid_y, grid_z, padding_mode, align_corners=False):
    """Perform trilinear sampling for 3D grid_sample using functional operations.

    Args:
        x: Input tensor (N, C, D_in, H_in, W_in)
        grid_x: X (width) coordinates (N, D_out, H_out, W_out) - already unnormalized to pixel space
        grid_y: Y (height) coordinates (N, D_out, H_out, W_out) - already unnormalized to pixel space
        grid_z: Z (depth) coordinates (N, D_out, H_out, W_out) - already unnormalized to pixel space
        padding_mode: Padding mode string
        align_corners: Whether align_corners is True

    Returns:
        Sampled tensor (N, C, D_out, H_out, W_out)
    """
    N, C, D_in, H_in, W_in = x.shape
    D_out, H_out, W_out = grid_x.shape[1], grid_x.shape[2], grid_x.shape[3]

    # For reflection mode, reflect the floating-point coordinates first
    if padding_mode == "reflection":
        grid_x = _reflect_coord(grid_x, W_in, align_corners)
        grid_y = _reflect_coord(grid_y, H_in, align_corners)
        grid_z = _reflect_coord(grid_z, D_in, align_corners)

    # Get corner coordinates for trilinear interpolation (8 corners of a cube)
    # tnw = top-north-west (z=floor, y=floor, x=floor)
    ix0 = mx.floor(grid_x).astype(mx.int32)  # x floor
    iy0 = mx.floor(grid_y).astype(mx.int32)  # y floor
    iz0 = mx.floor(grid_z).astype(mx.int32)  # z floor
    ix1 = ix0 + 1  # x ceil
    iy1 = iy0 + 1  # y ceil
    iz1 = iz0 + 1  # z ceil

    # Get interpolation weights
    wx = grid_x - mx.floor(grid_x)  # fractional x
    wy = grid_y - mx.floor(grid_y)  # fractional y
    wz = grid_z - mx.floor(grid_z)  # fractional z

    # Handle boundary conditions for zeros and border modes
    def get_valid_and_clipped(idx, dim_size):
        if padding_mode == "zeros":
            valid = (idx >= 0) & (idx < dim_size)
            clipped = mx.clip(idx, 0, dim_size - 1)
            return valid.astype(mx.float32), clipped
        else:  # border or reflection (reflection already handled above)
            return mx.ones(idx.shape, dtype=mx.float32), mx.clip(idx, 0, dim_size - 1)

    v_x0, ix0_c = get_valid_and_clipped(ix0, W_in)
    v_y0, iy0_c = get_valid_and_clipped(iy0, H_in)
    v_z0, iz0_c = get_valid_and_clipped(iz0, D_in)
    v_x1, ix1_c = get_valid_and_clipped(ix1, W_in)
    v_y1, iy1_c = get_valid_and_clipped(iy1, H_in)
    v_z1, iz1_c = get_valid_and_clipped(iz1, D_in)

    # Reshape x to (N, C, D_in * H_in * W_in) for gather operations
    x_flat = mx.reshape(x, (N, C, D_in * H_in * W_in))

    # Compute flat indices for all 8 corners
    def compute_flat_idx_3d(iz, iy, ix):
        return iz * (H_in * W_in) + iy * W_in + ix

    # 8 corners: tnw, tne, tsw, tse, bnw, bne, bsw, bse
    # t = top (z=iz0), b = bottom (z=iz1)
    # n = north (y=iy0), s = south (y=iy1)
    # w = west (x=ix0), e = east (x=ix1)
    idx_tnw = compute_flat_idx_3d(iz0_c, iy0_c, ix0_c)
    idx_tne = compute_flat_idx_3d(iz0_c, iy0_c, ix1_c)
    idx_tsw = compute_flat_idx_3d(iz0_c, iy1_c, ix0_c)
    idx_tse = compute_flat_idx_3d(iz0_c, iy1_c, ix1_c)
    idx_bnw = compute_flat_idx_3d(iz1_c, iy0_c, ix0_c)
    idx_bne = compute_flat_idx_3d(iz1_c, iy0_c, ix1_c)
    idx_bsw = compute_flat_idx_3d(iz1_c, iy1_c, ix0_c)
    idx_bse = compute_flat_idx_3d(iz1_c, iy1_c, ix1_c)

    # Flatten indices for gather: (N, D_out * H_out * W_out)
    out_size = D_out * H_out * W_out
    idx_tnw_flat = mx.reshape(idx_tnw, (N, out_size))
    idx_tne_flat = mx.reshape(idx_tne, (N, out_size))
    idx_tsw_flat = mx.reshape(idx_tsw, (N, out_size))
    idx_tse_flat = mx.reshape(idx_tse, (N, out_size))
    idx_bnw_flat = mx.reshape(idx_bnw, (N, out_size))
    idx_bne_flat = mx.reshape(idx_bne, (N, out_size))
    idx_bsw_flat = mx.reshape(idx_bsw, (N, out_size))
    idx_bse_flat = mx.reshape(idx_bse, (N, out_size))

    # Gather values for all 8 corners
    def gather_batch(x_flat, idx_flat):
        output_list = []
        for n in range(N):
            gathered = mx.take(x_flat[n], idx_flat[n], axis=1)
            output_list.append(gathered)
        return mx.stack(output_list, axis=0)

    val_tnw = gather_batch(x_flat, idx_tnw_flat)  # (N, C, out_size)
    val_tne = gather_batch(x_flat, idx_tne_flat)
    val_tsw = gather_batch(x_flat, idx_tsw_flat)
    val_tse = gather_batch(x_flat, idx_tse_flat)
    val_bnw = gather_batch(x_flat, idx_bnw_flat)
    val_bne = gather_batch(x_flat, idx_bne_flat)
    val_bsw = gather_batch(x_flat, idx_bsw_flat)
    val_bse = gather_batch(x_flat, idx_bse_flat)

    # Reshape to (N, C, D_out, H_out, W_out)
    out_shape = (N, C, D_out, H_out, W_out)
    val_tnw = mx.reshape(val_tnw, out_shape)
    val_tne = mx.reshape(val_tne, out_shape)
    val_tsw = mx.reshape(val_tsw, out_shape)
    val_tse = mx.reshape(val_tse, out_shape)
    val_bnw = mx.reshape(val_bnw, out_shape)
    val_bne = mx.reshape(val_bne, out_shape)
    val_bsw = mx.reshape(val_bsw, out_shape)
    val_bse = mx.reshape(val_bse, out_shape)

    # Compute trilinear interpolation weights
    # The weight for each corner is the volume of the opposite sub-cube
    # w_tnw = (1-wx)*(1-wy)*(1-wz), etc.
    w_tnw = (1 - wx) * (1 - wy) * (1 - wz)
    w_tne = wx * (1 - wy) * (1 - wz)
    w_tsw = (1 - wx) * wy * (1 - wz)
    w_tse = wx * wy * (1 - wz)
    w_bnw = (1 - wx) * (1 - wy) * wz
    w_bne = wx * (1 - wy) * wz
    w_bsw = (1 - wx) * wy * wz
    w_bse = wx * wy * wz

    # Apply validity masks (for zeros padding mode)
    m_tnw = v_x0 * v_y0 * v_z0
    m_tne = v_x1 * v_y0 * v_z0
    m_tsw = v_x0 * v_y1 * v_z0
    m_tse = v_x1 * v_y1 * v_z0
    m_bnw = v_x0 * v_y0 * v_z1
    m_bne = v_x1 * v_y0 * v_z1
    m_bsw = v_x0 * v_y1 * v_z1
    m_bse = v_x1 * v_y1 * v_z1

    # Expand weights and masks to match val dimensions (N, C, D_out, H_out, W_out)
    # Weights/masks are (N, D_out, H_out, W_out), need to add channel dim
    w_tnw = mx.expand_dims(w_tnw, axis=1)
    w_tne = mx.expand_dims(w_tne, axis=1)
    w_tsw = mx.expand_dims(w_tsw, axis=1)
    w_tse = mx.expand_dims(w_tse, axis=1)
    w_bnw = mx.expand_dims(w_bnw, axis=1)
    w_bne = mx.expand_dims(w_bne, axis=1)
    w_bsw = mx.expand_dims(w_bsw, axis=1)
    w_bse = mx.expand_dims(w_bse, axis=1)

    m_tnw = mx.expand_dims(m_tnw, axis=1)
    m_tne = mx.expand_dims(m_tne, axis=1)
    m_tsw = mx.expand_dims(m_tsw, axis=1)
    m_tse = mx.expand_dims(m_tse, axis=1)
    m_bnw = mx.expand_dims(m_bnw, axis=1)
    m_bne = mx.expand_dims(m_bne, axis=1)
    m_bsw = mx.expand_dims(m_bsw, axis=1)
    m_bse = mx.expand_dims(m_bse, axis=1)

    # Compute trilinear interpolation
    output = (
        val_tnw * w_tnw * m_tnw
        + val_tne * w_tne * m_tne
        + val_tsw * w_tsw * m_tsw
        + val_tse * w_tse * m_tse
        + val_bnw * w_bnw * m_bnw
        + val_bne * w_bne * m_bne
        + val_bsw * w_bsw * m_bsw
        + val_bse * w_bse * m_bse
    )

    return output


def _sample_with_indices_3d(x, iz, iy, ix, valid_mask, padding_mode):
    """Sample from 3D input tensor using indices (for nearest neighbor).

    Uses functional operations instead of in-place assignment.

    Args:
        x: Input tensor (N, C, D_in, H_in, W_in)
        iz: Z indices (N, D_out, H_out, W_out)
        iy: Y indices (N, D_out, H_out, W_out)
        ix: X indices (N, D_out, H_out, W_out)
        valid_mask: Boolean mask for valid samples (N, D_out, H_out, W_out)
        padding_mode: Padding mode string

    Returns:
        Sampled tensor (N, C, D_out, H_out, W_out)
    """
    N, C, D_in, H_in, W_in = x.shape
    D_out, H_out, W_out = iz.shape[1], iz.shape[2], iz.shape[3]

    # Compute linear indices into the spatial dimensions
    spatial_idx = iz * (H_in * W_in) + iy * W_in + ix  # (N, D_out, H_out, W_out)

    # Reshape x to (N, C, D_in * H_in * W_in)
    x_flat = mx.reshape(x, (N, C, D_in * H_in * W_in))

    # Flatten spatial_idx to (N, D_out * H_out * W_out)
    out_size = D_out * H_out * W_out
    spatial_idx_flat = mx.reshape(spatial_idx, (N, out_size))

    # Gather for each batch and channel
    output_list = []
    for n in range(N):
        gathered = mx.take(x_flat[n], spatial_idx_flat[n], axis=1)
        output_list.append(gathered)

    # Stack along batch dimension: (N, C, D_out * H_out * W_out)
    output = mx.stack(output_list, axis=0)

    # Reshape to (N, C, D_out, H_out, W_out)
    output = mx.reshape(output, (N, C, D_out, H_out, W_out))

    # Apply valid mask for 'zeros' padding mode
    if padding_mode == "zeros":
        # valid_mask is (N, D_out, H_out, W_out), broadcast to (N, C, D_out, H_out, W_out)
        valid_mask_expanded = mx.expand_dims(valid_mask, axis=1)  # (N, 1, D_out, H_out, W_out)
        valid_mask_expanded = mx.broadcast_to(
            valid_mask_expanded.astype(x.dtype), (N, C, D_out, H_out, W_out)
        )
        output = output * valid_mask_expanded

    return output


def _bilinear_sample(x, grid_x, grid_y, padding_mode, align_corners=False):
    """Perform bilinear sampling using functional operations.

    Args:
        x: Input tensor (N, C, H_in, W_in)
        grid_x: X coordinates (N, H_out, W_out) - already unnormalized to pixel space
        grid_y: Y coordinates (N, H_out, W_out) - already unnormalized to pixel space
        padding_mode: Padding mode string
        align_corners: Whether align_corners is True

    Returns:
        Sampled tensor (N, C, H_out, W_out)
    """
    N, C, H_in, W_in = x.shape
    H_out, W_out = grid_x.shape[1], grid_x.shape[2]

    # For reflection mode, reflect the floating-point coordinates first
    if padding_mode == "reflection":
        grid_x = _reflect_coord(grid_x, W_in, align_corners)
        grid_y = _reflect_coord(grid_y, H_in, align_corners)

    # Get corner coordinates for bilinear interpolation
    ix0 = mx.floor(grid_x).astype(mx.int32)
    iy0 = mx.floor(grid_y).astype(mx.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Get interpolation weights
    wx = grid_x - mx.floor(grid_x)
    wy = grid_y - mx.floor(grid_y)

    # Handle boundary conditions for zeros and border modes
    def get_valid_and_clipped(idx, dim_size):
        if padding_mode == "zeros":
            valid = (idx >= 0) & (idx < dim_size)
            clipped = mx.clip(idx, 0, dim_size - 1)
            return valid.astype(mx.float32), clipped
        else:  # border or reflection (reflection already handled above)
            return mx.ones(idx.shape, dtype=mx.float32), mx.clip(idx, 0, dim_size - 1)

    v_x0, ix0_c = get_valid_and_clipped(ix0, W_in)
    v_y0, iy0_c = get_valid_and_clipped(iy0, H_in)
    v_x1, ix1_c = get_valid_and_clipped(ix1, W_in)
    v_y1, iy1_c = get_valid_and_clipped(iy1, H_in)

    # Reshape x to (N, C, H_in * W_in) for gather operations
    x_flat = mx.reshape(x, (N, C, H_in * W_in))

    # Compute flat indices for all four corners
    def compute_flat_idx(iy, ix):
        return iy * W_in + ix

    idx_00 = compute_flat_idx(iy0_c, ix0_c)  # top-left
    idx_01 = compute_flat_idx(iy0_c, ix1_c)  # top-right
    idx_10 = compute_flat_idx(iy1_c, ix0_c)  # bottom-left
    idx_11 = compute_flat_idx(iy1_c, ix1_c)  # bottom-right

    # Flatten indices for gather
    idx_00_flat = mx.reshape(idx_00, (N, H_out * W_out))
    idx_01_flat = mx.reshape(idx_01, (N, H_out * W_out))
    idx_10_flat = mx.reshape(idx_10, (N, H_out * W_out))
    idx_11_flat = mx.reshape(idx_11, (N, H_out * W_out))

    # Gather values for all four corners
    def gather_batch(x_flat, idx_flat):
        output_list = []
        for n in range(N):
            gathered = mx.take(x_flat[n], idx_flat[n], axis=1)
            output_list.append(gathered)
        return mx.stack(output_list, axis=0)

    val_00 = gather_batch(x_flat, idx_00_flat)  # (N, C, H_out * W_out)
    val_01 = gather_batch(x_flat, idx_01_flat)
    val_10 = gather_batch(x_flat, idx_10_flat)
    val_11 = gather_batch(x_flat, idx_11_flat)

    # Reshape to (N, C, H_out, W_out)
    val_00 = mx.reshape(val_00, (N, C, H_out, W_out))
    val_01 = mx.reshape(val_01, (N, C, H_out, W_out))
    val_10 = mx.reshape(val_10, (N, C, H_out, W_out))
    val_11 = mx.reshape(val_11, (N, C, H_out, W_out))

    # Compute weights for bilinear interpolation
    # w_00 = (1 - wx) * (1 - wy), w_01 = wx * (1 - wy)
    # w_10 = (1 - wx) * wy, w_11 = wx * wy
    w_00 = (1 - wx) * (1 - wy)
    w_01 = wx * (1 - wy)
    w_10 = (1 - wx) * wy
    w_11 = wx * wy

    # Apply validity masks
    m_00 = v_x0 * v_y0
    m_01 = v_x1 * v_y0
    m_10 = v_x0 * v_y1
    m_11 = v_x1 * v_y1

    # Expand weights and masks to match val dimensions (N, C, H_out, W_out)
    w_00 = mx.expand_dims(w_00, axis=1)
    w_01 = mx.expand_dims(w_01, axis=1)
    w_10 = mx.expand_dims(w_10, axis=1)
    w_11 = mx.expand_dims(w_11, axis=1)

    m_00 = mx.expand_dims(m_00, axis=1)
    m_01 = mx.expand_dims(m_01, axis=1)
    m_10 = mx.expand_dims(m_10, axis=1)
    m_11 = mx.expand_dims(m_11, axis=1)

    # Compute bilinear interpolation
    output = (
        val_00 * w_00 * m_00 + val_01 * w_01 * m_01 + val_10 * w_10 * m_10 + val_11 * w_11 * m_11
    )

    return output


def _bicubic_sample(x, grid_x, grid_y, padding_mode, align_corners=False):
    """Perform bicubic sampling using functional operations.

    Uses cubic convolution algorithm with a 4x4 neighborhood of pixels.
    Based on Keys' cubic kernel with a=-0.5 (matches PyTorch default).

    Args:
        x: Input tensor (N, C, H_in, W_in)
        grid_x: X coordinates (N, H_out, W_out) - already unnormalized to pixel space
        grid_y: Y coordinates (N, H_out, W_out) - already unnormalized to pixel space
        padding_mode: Padding mode string
        align_corners: Whether align_corners is True

    Returns:
        Sampled tensor (N, C, H_out, W_out)
    """
    N, C, H_in, W_in = x.shape
    H_out, W_out = grid_x.shape[1], grid_x.shape[2]

    # For reflection mode, reflect the floating-point coordinates first
    if padding_mode == "reflection":
        grid_x = _reflect_coord(grid_x, W_in, align_corners)
        grid_y = _reflect_coord(grid_y, H_in, align_corners)

    # Get base integer coordinates (floor)
    ix_base = mx.floor(grid_x).astype(mx.int32)
    iy_base = mx.floor(grid_y).astype(mx.int32)

    # Get fractional parts for kernel weights
    fx = grid_x - mx.floor(grid_x)
    fy = grid_y - mx.floor(grid_y)

    # Reshape x to (N, C, H_in * W_in) for gather operations
    x_flat = mx.reshape(x, (N, C, H_in * W_in))

    # Helper for gathering values at a flat index
    def gather_batch(x_flat, idx_flat):
        output_list = []
        for n in range(N):
            gathered = mx.take(x_flat[n], idx_flat[n], axis=1)
            output_list.append(gathered)
        return mx.stack(output_list, axis=0)

    # Helper for handling boundary conditions
    def get_valid_and_clipped(idx, dim_size):
        if padding_mode == "zeros":
            valid = (idx >= 0) & (idx < dim_size)
            clipped = mx.clip(idx, 0, dim_size - 1)
            return valid.astype(mx.float32), clipped
        else:  # border or reflection (reflection already handled above)
            return mx.ones(idx.shape, dtype=mx.float32), mx.clip(idx, 0, dim_size - 1)

    # Initialize output accumulator
    output = mx.zeros((N, C, H_out, W_out), dtype=x.dtype)

    # Bicubic uses 4x4 neighborhood: offsets -1, 0, 1, 2 relative to base
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            # Compute indices for this offset
            ix = ix_base + dx
            iy = iy_base + dy

            # Get validity mask and clipped indices
            v_x, ix_c = get_valid_and_clipped(ix, W_in)
            v_y, iy_c = get_valid_and_clipped(iy, H_in)

            # Compute flat indices
            idx_flat = iy_c * W_in + ix_c
            idx_flat = mx.reshape(idx_flat, (N, H_out * W_out))

            # Gather values
            val = gather_batch(x_flat, idx_flat)
            val = mx.reshape(val, (N, C, H_out, W_out))

            # Compute bicubic kernel weights
            # Keys' cubic with a=-0.5
            wx = _bicubic_kernel(fx - dx)
            wy = _bicubic_kernel(fy - dy)

            # Expand weights for broadcasting
            wx = mx.expand_dims(wx, axis=1)  # (N, 1, H_out, W_out)
            wy = mx.expand_dims(wy, axis=1)  # (N, 1, H_out, W_out)

            # Validity mask
            mask = v_x * v_y
            mask = mx.expand_dims(mask, axis=1)  # (N, 1, H_out, W_out)

            # Accumulate weighted contribution
            output = output + val * wx * wy * mask

    return output


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Sample input using grid coordinates.

    Args:
        input: Input tensor (N, C, H_in, W_in) or (N, C, D, H, W)
        grid: Grid tensor (N, H_out, W_out, 2) or (N, D_out, H_out, W_out, 3)
        mode: Interpolation mode - 'bilinear', 'nearest', 'bicubic'
        padding_mode: How to handle out-of-bounds - 'zeros', 'border', 'reflection'
        align_corners: If True, align corner pixels

    Returns:
        Sampled tensor of shape (N, C, H_out, W_out) or (N, C, D_out, H_out, W_out)
    """
    # Default align_corners to False if not specified
    if align_corners is None:
        align_corners = False
    x = input._mlx_array
    g = grid._mlx_array

    N, C = x.shape[:2]
    spatial_dims = x.ndim - 2

    if spatial_dims == 2:
        # 2D case
        H_in, W_in = x.shape[2], x.shape[3]
        H_out, W_out = g.shape[1], g.shape[2]

        # Unnormalize grid coordinates from [-1, 1] to pixel coordinates
        if align_corners:
            grid_x = (g[..., 0] + 1) * (W_in - 1) / 2
            grid_y = (g[..., 1] + 1) * (H_in - 1) / 2
        else:
            grid_x = ((g[..., 0] + 1) * W_in - 1) / 2
            grid_y = ((g[..., 1] + 1) * H_in - 1) / 2

        if mode == "nearest":
            # For reflection mode, reflect coordinates first, then round
            if padding_mode == "reflection":
                grid_x = _reflect_coord(grid_x, W_in, align_corners)
                grid_y = _reflect_coord(grid_y, H_in, align_corners)

            # Round to nearest pixel
            ix = mx.round(grid_x).astype(mx.int32)
            iy = mx.round(grid_y).astype(mx.int32)

            # Handle padding mode
            if padding_mode == "zeros":
                valid = (ix >= 0) & (ix < W_in) & (iy >= 0) & (iy < H_in)
                ix = mx.clip(ix, 0, W_in - 1)
                iy = mx.clip(iy, 0, H_in - 1)
            else:  # border or reflection (reflection already handled above)
                ix = mx.clip(ix, 0, W_in - 1)
                iy = mx.clip(iy, 0, H_in - 1)
                valid = mx.ones(ix.shape, dtype=mx.bool_)

            # Use functional sampling instead of in-place assignment
            output = _sample_with_indices(x, iy, ix, valid, padding_mode)

        elif mode == "bilinear":
            # Use vectorized bilinear sampling
            output = _bilinear_sample(x, grid_x, grid_y, padding_mode, align_corners)

        else:  # bicubic
            # Use proper bicubic sampling with 4x4 neighborhood
            output = _bicubic_sample(x, grid_x, grid_y, padding_mode, align_corners)

    else:
        # 3D case
        D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        D_out, H_out, W_out = g.shape[1], g.shape[2], g.shape[3]

        # PyTorch grid format for 3D is (N, D_out, H_out, W_out, 3)
        # where the last dim is (x, y, z) = (W, H, D)
        # Unnormalize grid coordinates from [-1, 1] to pixel coordinates
        if align_corners:
            grid_x = (g[..., 0] + 1) * (W_in - 1) / 2  # x -> width
            grid_y = (g[..., 1] + 1) * (H_in - 1) / 2  # y -> height
            grid_z = (g[..., 2] + 1) * (D_in - 1) / 2  # z -> depth
        else:
            grid_x = ((g[..., 0] + 1) * W_in - 1) / 2
            grid_y = ((g[..., 1] + 1) * H_in - 1) / 2
            grid_z = ((g[..., 2] + 1) * D_in - 1) / 2

        if mode == "nearest":
            # For reflection mode, reflect coordinates first, then round
            if padding_mode == "reflection":
                grid_x = _reflect_coord(grid_x, W_in, align_corners)
                grid_y = _reflect_coord(grid_y, H_in, align_corners)
                grid_z = _reflect_coord(grid_z, D_in, align_corners)

            # Round to nearest pixel
            ix = mx.round(grid_x).astype(mx.int32)
            iy = mx.round(grid_y).astype(mx.int32)
            iz = mx.round(grid_z).astype(mx.int32)

            # Handle padding mode
            if padding_mode == "zeros":
                valid = (ix >= 0) & (ix < W_in) & (iy >= 0) & (iy < H_in) & (iz >= 0) & (iz < D_in)
                ix = mx.clip(ix, 0, W_in - 1)
                iy = mx.clip(iy, 0, H_in - 1)
                iz = mx.clip(iz, 0, D_in - 1)
            else:  # border or reflection (reflection already handled above)
                ix = mx.clip(ix, 0, W_in - 1)
                iy = mx.clip(iy, 0, H_in - 1)
                iz = mx.clip(iz, 0, D_in - 1)
                valid = mx.ones(ix.shape, dtype=mx.bool_)

            # Use functional 3D sampling
            output = _sample_with_indices_3d(x, iz, iy, ix, valid, padding_mode)

        elif mode == "bilinear":
            # Use trilinear sampling (called 'bilinear' in PyTorch for consistency)
            output = _trilinear_sample(x, grid_x, grid_y, grid_z, padding_mode, align_corners)

        else:  # bicubic not supported for 3D
            raise ValueError("bicubic interpolation only supports 4D input (2D spatial)")

    result = Tensor._from_mlx_array(output)
    if is_grad_enabled() and (input.requires_grad or grid.requires_grad):
        result.requires_grad = True
    return result


# =============================================================================
# RMS Normalization
# =============================================================================


def rms_norm(
    input: Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[Tensor] = None,
    eps: Optional[float] = None,
) -> Tensor:
    """Apply Root Mean Square Layer Normalization.

    Args:
        input: Input tensor
        normalized_shape: Shape of the last dimensions to normalize over
        weight: Optional learnable scaling parameter
        eps: Small value for numerical stability

    Returns:
        Normalized tensor
    """
    # Default eps to 1e-6 if not specified (PyTorch's default)
    if eps is None:
        eps = 1e-6
    x = input._mlx_array

    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    # Compute RMS over the normalized dimensions
    dims = list(range(-len(normalized_shape), 0))
    rms = mx.sqrt(mx.mean(x * x, axis=dims, keepdims=True) + eps)

    output = x / rms

    if weight is not None:
        output = output * weight._mlx_array

    result = Tensor._from_mlx_array(output)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


# =============================================================================
# Additional Loss Functions
# =============================================================================


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    distance_function: callable = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Triplet margin loss with custom distance function.

    Args:
        anchor: Anchor samples
        positive: Positive samples
        negative: Negative samples
        distance_function: Custom distance function (default: pairwise_distance)
        margin: Margin value
        swap: Whether to use the swap strategy
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss value
    """
    if distance_function is None:

        def distance_function(x, y):
            return pairwise_distance(x, y)

    d_ap = distance_function(anchor, positive)
    d_an = distance_function(anchor, negative)

    if swap:
        d_pn = distance_function(positive, negative)
        d_an = mx.minimum(d_an._mlx_array, d_pn._mlx_array)
        d_an = Tensor._from_mlx_array(d_an)

    loss = mx.maximum(d_ap._mlx_array - d_an._mlx_array + margin, 0)

    if reduction == "mean":
        loss = mx.mean(loss)
    elif reduction == "sum":
        loss = mx.sum(loss)

    result = Tensor._from_mlx_array(loss)
    if is_grad_enabled():
        result.requires_grad = True
    return result


def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    """Multi-label margin loss.

    Args:
        input: Input predictions (N, C)
        target: Target labels (N, C) with positive labels set to class indices
                and rest set to -1
        size_average: Deprecated (use reduction)
        reduce: Deprecated (use reduction)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss value
    """
    # Handle deprecated arguments
    if size_average is not None or reduce is not None:
        import warnings

        warnings.warn(
            "size_average and reduce args will be deprecated, please use reduction instead"
        )
        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True
        if reduce:
            reduction = "mean" if size_average else "sum"
        else:
            reduction = "none"

    x = input._mlx_array
    y = target._mlx_array.astype(mx.int32)
    N, C = x.shape

    # Create masks for positive labels (>= 0) and negative labels (< 0 marks padding)
    positive_mask = (y >= 0).astype(x.dtype)  # (N, C) - 1 where we have valid labels

    # For each sample and each position, get the class index
    # Convert target indices to one-hot-like positive class indicators
    # y[n, j] gives the positive class index at position j
    # We need to create a mask showing which classes are positive

    # Build positive class mask for each sample
    # For each sample n, classes in y[n, :] (where y >= 0) are positive
    class_indices = mx.arange(C)  # (C,)

    # For each sample, for each valid target entry, mark that class as positive
    # y shape: (N, C), contains class indices or -1
    # We want: positive_class_mask (N, C) where [n, c] = 1 if class c is a positive class for sample n

    # Expand for broadcasting: y (N, C, 1), class_indices (C,)
    y_expanded = mx.expand_dims(y, axis=2)  # (N, C, 1)
    # Check which classes match any of the target entries
    matches = y_expanded == class_indices  # (N, C, C)
    # A class is positive if it matches any valid target entry
    # Also need to mask out invalid entries (y < 0)
    valid_entries = mx.expand_dims(y >= 0, axis=2)  # (N, C, 1)
    valid_matches = matches & valid_entries  # (N, C, C)
    positive_class_mask = mx.any(valid_matches, axis=1).astype(x.dtype)  # (N, C)

    # Negative class mask
    negative_class_mask = 1.0 - positive_class_mask  # (N, C)

    # For each valid positive entry j, we need to compare x[pos_class] vs all negative classes
    # Count valid positive entries per sample
    num_positive_entries = mx.sum(positive_mask, axis=1, keepdims=True)  # (N, 1)

    # Get scores for positive classes
    # positive_class_mask (N, C) tells us which classes are positive
    # For each sample n, we want x[n, pos_classes]

    # Broadcast approach: compute pairwise differences
    # x_pos: scores at positive classes, x_neg: scores at negative classes
    # Loss = sum over (pos_entry j) sum over (neg_class k): max(0, 1 - x[pos_class[j]] + x[neg_class[k]])

    # Simpler vectorized approach:
    # For each (n, pos_class), compute margin against all negative classes
    # Then sum, weighted by how many times that pos_class appears in targets

    # Positive class scores: (N, C) masked
    pos_scores = x * positive_class_mask  # (N, C), 0 where not positive
    neg_scores = x * negative_class_mask  # (N, C), 0 where not negative

    # For margin loss: for each positive class p and negative class n:
    # loss += max(0, 1 - x[p] + x[n])
    # Expand for pairwise: pos_scores (N, C, 1), neg_scores (N, 1, C)
    pos_expanded = mx.expand_dims(pos_scores, axis=2)  # (N, C, 1)
    neg_expanded = mx.expand_dims(neg_scores, axis=1)  # (N, 1, C)

    # Compute margins
    margins = 1.0 - pos_expanded + neg_expanded  # (N, C, C)
    margins = mx.maximum(margins, 0)  # hinge

    # Mask: only count where pos_class_mask (axis 1) and neg_class_mask (axis 2)
    pos_mask_expanded = mx.expand_dims(positive_class_mask, axis=2)  # (N, C, 1)
    neg_mask_expanded = mx.expand_dims(negative_class_mask, axis=1)  # (N, 1, C)
    valid_pairs = pos_mask_expanded * neg_mask_expanded  # (N, C, C)

    # Weight by how many times each positive class appears in target
    # Count occurrences of each class in target
    # For each class c, count how many positions j have y[j] == c (and y[j] >= 0)
    pos_class_counts = mx.sum(valid_matches.astype(x.dtype), axis=1)  # (N, C)
    pos_class_counts_expanded = mx.expand_dims(pos_class_counts, axis=2)  # (N, C, 1)

    # Weighted margins
    weighted_margins = margins * valid_pairs * pos_class_counts_expanded

    # Sum over all pairs for each sample, then divide by C
    sample_losses = mx.sum(weighted_margins, axis=(1, 2)) / C  # (N,)

    if reduction == "mean":
        loss = mx.mean(sample_losses)
    elif reduction == "sum":
        loss = mx.sum(sample_losses)
    else:
        loss = sample_losses

    result = Tensor._from_mlx_array(loss)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


# =============================================================================
# Multi-head Attention Forward
# =============================================================================


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Multi-head attention forward pass (functional version).

    This is a simplified implementation that delegates to scaled_dot_product_attention.
    """
    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    head_dim = embed_dim // num_heads

    # Project queries, keys, values
    if use_separate_proj_weight:
        q = linear(
            query, q_proj_weight, in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        )
        k = linear(
            key,
            k_proj_weight,
            in_proj_bias[embed_dim : 2 * embed_dim] if in_proj_bias is not None else None,
        )
        v = linear(
            value,
            v_proj_weight,
            in_proj_bias[2 * embed_dim :] if in_proj_bias is not None else None,
        )
    else:
        # Combined projection
        qkv_same = query is key and key is value
        if qkv_same:
            # Self-attention
            qkv = linear(query, in_proj_weight, in_proj_bias)
            qkv = qkv.reshape(tgt_len, bsz, 3, num_heads, head_dim)
            q = qkv[:, :, 0, :, :].reshape(tgt_len, bsz, embed_dim)
            k = qkv[:, :, 1, :, :].reshape(src_len, bsz, embed_dim)
            v = qkv[:, :, 2, :, :].reshape(src_len, bsz, embed_dim)
        else:
            # Encoder-decoder attention
            q = linear(
                query,
                Tensor._from_mlx_array(in_proj_weight._mlx_array[:embed_dim]),
                (
                    Tensor._from_mlx_array(in_proj_bias._mlx_array[:embed_dim])
                    if in_proj_bias
                    else None
                ),
            )
            k = linear(
                key,
                Tensor._from_mlx_array(in_proj_weight._mlx_array[embed_dim : 2 * embed_dim]),
                (
                    Tensor._from_mlx_array(in_proj_bias._mlx_array[embed_dim : 2 * embed_dim])
                    if in_proj_bias
                    else None
                ),
            )
            v = linear(
                value,
                Tensor._from_mlx_array(in_proj_weight._mlx_array[2 * embed_dim :]),
                (
                    Tensor._from_mlx_array(in_proj_bias._mlx_array[2 * embed_dim :])
                    if in_proj_bias
                    else None
                ),
            )

    # Reshape for multi-head attention: (L, N, E) -> (N, num_heads, L, head_dim)
    q = q.reshape(tgt_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    k = k.reshape(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    v = v.reshape(src_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)

    # Use scaled dot product attention
    dropout_p_val = dropout_p if training else 0.0
    attn_output = scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p_val, is_causal=is_causal
    )

    # Reshape back: (N, num_heads, L, head_dim) -> (L, N, E)
    attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)

    # Output projection
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    attn_weights = None
    if need_weights:
        # Compute attention weights for return
        scale = 1.0 / (head_dim**0.5)
        attn_weights = mx.matmul(q._mlx_array, mx.transpose(k._mlx_array, (0, 1, 3, 2))) * scale
        attn_weights = mx.softmax(attn_weights, axis=-1)
        if average_attn_weights:
            attn_weights = mx.mean(attn_weights, axis=1)
        attn_weights = Tensor._from_mlx_array(attn_weights)

    return attn_output, attn_weights


# =============================================================================
# Type Re-exports for Compatibility
# =============================================================================

import importlib
import math
import warnings

# These are re-exported from typing for PyTorch API compatibility
from typing import Callable, Optional, Union

# Re-export for compatibility
TYPE_CHECKING = False  # We don't use type checking at runtime

# DType is a type alias - re-export from our dtype module
from ..dtype import DType

# Tensor re-export
from ..tensor import Tensor

# Constants matching PyTorch's internal constants
BroadcastingList1 = list
BroadcastingList2 = list
BroadcastingList3 = list

GRID_SAMPLE_INTERPOLATION_MODES = {"bilinear": 0, "nearest": 1, "bicubic": 2}
GRID_SAMPLE_PADDING_MODES = {"zeros": 0, "border": 1, "reflection": 2}

# Documentation notes (used for docstrings)
reproducibility_notes = {"note": "Operations are generally deterministic in MLX"}
sparse_support_notes = {"note": "Sparse tensor support is limited in MLX"}
tf32_notes = {"note": "TF32 is not applicable to MLX (Apple Silicon)"}

# Module re-exports
from . import grad

torch = None  # Placeholder - user should import flashlight instead


# =============================================================================
# Torch Function Dispatch (Compatibility Stubs)
# =============================================================================


def has_torch_function(*args) -> bool:
    """Check if any argument has a __torch_function__ method.

    In flashlight, we don't use torch function dispatch, so this always returns False.
    """
    return False


def has_torch_function_unary(arg) -> bool:
    """Check if argument has a __torch_function__ method (unary version).

    In flashlight, we don't use torch function dispatch, so this always returns False.
    """
    return False


def has_torch_function_variadic(*args) -> bool:
    """Check if any argument has a __torch_function__ method (variadic version).

    In flashlight, we don't use torch function dispatch, so this always returns False.
    """
    return False


def handle_torch_function(public_api, relevant_args, *args, **kwargs):
    """Handle torch function dispatch.

    In flashlight, this just calls the public API directly since we don't
    support __torch_function__ dispatch.
    """
    return public_api(*args, **kwargs)


def boolean_dispatch(
    arg=None,
    arg_name=None,
    arg_index=None,
    default=None,
    if_true=None,
    if_false=None,
    module_name=None,
    func_name=None,
):
    """Boolean dispatch helper.

    This is used internally in PyTorch for legacy API compatibility.
    We return a simple dispatcher.
    """

    def dispatcher(*args, **kwargs):
        # Get the boolean argument
        if arg_name is not None and arg_name in kwargs:
            cond = kwargs[arg_name]
        elif arg_index is not None and len(args) > arg_index:
            cond = args[arg_index]
        else:
            cond = default

        if cond:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    return dispatcher


def assert_int_or_pair(arg, arg_name, message):
    """Assert that argument is an int or pair of ints.

    Args:
        arg: The argument to check
        arg_name: Name of the argument for error messages
        message: Error message if assertion fails
    """
    if isinstance(arg, (int, tuple, list)):
        if isinstance(arg, (tuple, list)):
            if len(arg) not in (1, 2):
                raise ValueError(message)
            if not all(isinstance(x, int) for x in arg):
                raise ValueError(message)
    else:
        raise ValueError(message)


# =============================================================================
# Pooling Functions with Indices
# =============================================================================


def _to_single(x):
    """Convert value to single int."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def _to_pair(x):
    """Convert value to pair."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _to_triple(x):
    """Convert value to triple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


def max_pool1d_with_indices(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Max pool 1D with indices.

    Returns the max pooled output and flattened indices into the input tensor.
    """
    kernel_size = _to_single(kernel_size)
    stride = _to_single(stride) if stride is not None else kernel_size
    padding = _to_single(padding)
    dilation = _to_single(dilation)

    x = input._mlx_array
    N, C, L = x.shape

    # Apply padding if needed
    if padding > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (padding, padding)])

    L_padded = L + 2 * padding
    L_out = (L_padded - dilation * (kernel_size - 1) - 1) // stride + 1

    outputs = []
    indices_list = []

    for i in range(L_out):
        start = i * stride
        # Extract window with dilation
        window_indices = [start + j * dilation for j in range(kernel_size)]
        window = mx.stack([x[:, :, idx] for idx in window_indices], axis=-1)  # [N, C, kernel_size]

        max_vals = mx.max(window, axis=-1, keepdims=True)  # [N, C, 1]
        local_argmax = mx.argmax(window, axis=-1, keepdims=True)  # [N, C, 1]

        # Convert to global index (in padded space, then adjust for padding)
        global_idx = start + local_argmax * dilation - padding  # Adjust for padding
        outputs.append(max_vals)
        indices_list.append(global_idx)

    output_array = mx.concatenate(outputs, axis=-1)  # [N, C, L_out]
    indices_array = mx.concatenate(indices_list, axis=-1)  # [N, C, L_out]

    output = Tensor._from_mlx_array(output_array)
    indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output, indices


def max_pool2d_with_indices(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Max pool 2D with indices.

    Returns the max pooled output and flattened indices into the H*W dimensions.
    """
    kH, kW = _to_pair(kernel_size)
    sH, sW = _to_pair(stride) if stride is not None else (kH, kW)
    pH, pW = _to_pair(padding)
    dH, dW = _to_pair(dilation)

    x = input._mlx_array
    N, C, H, W = x.shape

    # Apply padding if needed
    if pH > 0 or pW > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (pH, pH), (pW, pW)])

    H_padded = H + 2 * pH
    W_padded = W + 2 * pW
    H_out = (H_padded - dH * (kH - 1) - 1) // sH + 1
    W_out = (W_padded - dW * (kW - 1) - 1) // sW + 1

    outputs = []
    indices_list = []

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sH
            w_start = j * sW

            # Extract window with dilation
            window_values = []
            window_positions = []
            for kh in range(kH):
                for kw in range(kW):
                    h_idx = h_start + kh * dH
                    w_idx = w_start + kw * dW
                    window_values.append(x[:, :, h_idx, w_idx])
                    # Compute flattened index in original (non-padded) space
                    orig_h = h_idx - pH
                    orig_w = w_idx - pW
                    flat_idx = orig_h * W + orig_w
                    window_positions.append(flat_idx)

            window = mx.stack(window_values, axis=-1)  # [N, C, kH*kW]
            max_vals = mx.max(window, axis=-1, keepdims=True)  # [N, C, 1]
            local_argmax = mx.argmax(window, axis=-1)  # [N, C]

            # Map local argmax to global flat index
            positions_array = mx.array(window_positions)  # [kH*kW]
            global_indices = positions_array[local_argmax]  # [N, C]

            outputs.append(max_vals)
            indices_list.append(mx.expand_dims(global_indices, axis=-1))

    # Reshape outputs
    output_array = mx.concatenate(outputs, axis=-1)  # [N, C, H_out*W_out]
    output_array = mx.reshape(output_array, (N, C, H_out, W_out))

    indices_array = mx.concatenate(indices_list, axis=-1)  # [N, C, H_out*W_out]
    indices_array = mx.reshape(indices_array, (N, C, H_out, W_out))

    output = Tensor._from_mlx_array(output_array)
    indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output, indices


def max_pool3d_with_indices(
    input: Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Max pool 3D with indices.

    Returns the max pooled output and flattened indices into the D*H*W dimensions.
    """
    kD, kH, kW = _to_triple(kernel_size)
    sD, sH, sW = _to_triple(stride) if stride is not None else (kD, kH, kW)
    pD, pH, pW = _to_triple(padding)
    dD, dH, dW = _to_triple(dilation)

    x = input._mlx_array
    N, C, D, H, W = x.shape

    # Apply padding if needed
    if pD > 0 or pH > 0 or pW > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (pD, pD), (pH, pH), (pW, pW)])

    D_padded = D + 2 * pD
    H_padded = H + 2 * pH
    W_padded = W + 2 * pW
    D_out = (D_padded - dD * (kD - 1) - 1) // sD + 1
    H_out = (H_padded - dH * (kH - 1) - 1) // sH + 1
    W_out = (W_padded - dW * (kW - 1) - 1) // sW + 1

    outputs = []
    indices_list = []

    for d in range(D_out):
        for i in range(H_out):
            for j in range(W_out):
                d_start = d * sD
                h_start = i * sH
                w_start = j * sW

                # Extract window with dilation
                window_values = []
                window_positions = []
                for kd in range(kD):
                    for kh in range(kH):
                        for kw in range(kW):
                            d_idx = d_start + kd * dD
                            h_idx = h_start + kh * dH
                            w_idx = w_start + kw * dW
                            window_values.append(x[:, :, d_idx, h_idx, w_idx])
                            # Compute flattened index in original (non-padded) space
                            orig_d = d_idx - pD
                            orig_h = h_idx - pH
                            orig_w = w_idx - pW
                            flat_idx = orig_d * H * W + orig_h * W + orig_w
                            window_positions.append(flat_idx)

                window = mx.stack(window_values, axis=-1)  # [N, C, kD*kH*kW]
                max_vals = mx.max(window, axis=-1, keepdims=True)  # [N, C, 1]
                local_argmax = mx.argmax(window, axis=-1)  # [N, C]

                # Map local argmax to global flat index
                positions_array = mx.array(window_positions)  # [kD*kH*kW]
                global_indices = positions_array[local_argmax]  # [N, C]

                outputs.append(max_vals)
                indices_list.append(mx.expand_dims(global_indices, axis=-1))

    # Reshape outputs
    output_array = mx.concatenate(outputs, axis=-1)  # [N, C, D_out*H_out*W_out]
    output_array = mx.reshape(output_array, (N, C, D_out, H_out, W_out))

    indices_array = mx.concatenate(indices_list, axis=-1)  # [N, C, D_out*H_out*W_out]
    indices_array = mx.reshape(indices_array, (N, C, D_out, H_out, W_out))

    output = Tensor._from_mlx_array(output_array)
    indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output, indices


def adaptive_max_pool1d_with_indices(input: Tensor, output_size, return_indices: bool = False):
    """Adaptive max pool 1D with indices.

    Returns the adaptively max pooled output and indices into the input tensor.
    """
    # Use the ops implementation which now returns (output, indices)
    from ..ops.pooling import adaptive_max_pool1d as _adaptive_max_pool1d

    return _adaptive_max_pool1d(input, output_size)


def adaptive_max_pool2d_with_indices(input: Tensor, output_size, return_indices: bool = False):
    """Adaptive max pool 2D with indices.

    Returns the adaptively max pooled output and flattened indices into H*W.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    x = input._mlx_array
    N, C, H, W = x.shape
    out_H, out_W = output_size

    if out_H == H and out_W == W:
        # Identity case
        indices_array = mx.broadcast_to(mx.arange(H * W).reshape(1, 1, H, W), (N, C, H, W))
        return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))

    outputs = []
    indices_list = []

    for i in range(out_H):
        for j in range(out_W):
            # Compute start and end indices for this output position
            h_start = (i * H) // out_H
            h_end = ((i + 1) * H) // out_H
            w_start = (j * W) // out_W
            w_end = ((j + 1) * W) // out_W

            # Extract the region
            region = x[:, :, h_start:h_end, w_start:w_end]  # [N, C, region_h, region_w]
            region_h = h_end - h_start
            region_w = w_end - w_start

            # Reshape to [N, C, region_h*region_w] for argmax
            region_flat = mx.reshape(region, (N, C, region_h * region_w))
            max_vals = mx.max(region_flat, axis=-1, keepdims=True)  # [N, C, 1]
            local_argmax = mx.argmax(region_flat, axis=-1)  # [N, C]

            # Convert local argmax to global flat index in H*W
            local_h = local_argmax // region_w
            local_w = local_argmax % region_w
            global_h = h_start + local_h
            global_w = w_start + local_w
            global_idx = global_h * W + global_w

            outputs.append(max_vals)
            indices_list.append(mx.expand_dims(global_idx, axis=-1))

    # Reshape outputs
    output_array = mx.concatenate(outputs, axis=-1)  # [N, C, out_H*out_W]
    output_array = mx.reshape(output_array, (N, C, out_H, out_W))

    indices_array = mx.concatenate(indices_list, axis=-1)  # [N, C, out_H*out_W]
    indices_array = mx.reshape(indices_array, (N, C, out_H, out_W))

    output = Tensor._from_mlx_array(output_array)
    indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output, indices


def adaptive_max_pool3d_with_indices(input: Tensor, output_size, return_indices: bool = False):
    """Adaptive max pool 3D with indices.

    Returns the adaptively max pooled output and flattened indices into D*H*W.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    x = input._mlx_array
    N, C, D, H, W = x.shape
    out_D, out_H, out_W = output_size

    if out_D == D and out_H == H and out_W == W:
        # Identity case
        indices_array = mx.broadcast_to(
            mx.arange(D * H * W).reshape(1, 1, D, H, W), (N, C, D, H, W)
        )
        return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))

    outputs = []
    indices_list = []

    for d in range(out_D):
        for i in range(out_H):
            for j in range(out_W):
                # Compute start and end indices for this output position
                d_start = (d * D) // out_D
                d_end = ((d + 1) * D) // out_D
                h_start = (i * H) // out_H
                h_end = ((i + 1) * H) // out_H
                w_start = (j * W) // out_W
                w_end = ((j + 1) * W) // out_W

                # Extract the region
                region = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                region_d = d_end - d_start
                region_h = h_end - h_start
                region_w = w_end - w_start

                # Reshape to [N, C, region_d*region_h*region_w] for argmax
                region_flat = mx.reshape(region, (N, C, region_d * region_h * region_w))
                max_vals = mx.max(region_flat, axis=-1, keepdims=True)  # [N, C, 1]
                local_argmax = mx.argmax(region_flat, axis=-1)  # [N, C]

                # Convert local argmax to global flat index in D*H*W
                local_d = local_argmax // (region_h * region_w)
                local_hw = local_argmax % (region_h * region_w)
                local_h = local_hw // region_w
                local_w = local_hw % region_w

                global_d = d_start + local_d
                global_h = h_start + local_h
                global_w = w_start + local_w
                global_idx = global_d * H * W + global_h * W + global_w

                outputs.append(max_vals)
                indices_list.append(mx.expand_dims(global_idx, axis=-1))

    # Reshape outputs
    output_array = mx.concatenate(outputs, axis=-1)  # [N, C, out_D*out_H*out_W]
    output_array = mx.reshape(output_array, (N, C, out_D, out_H, out_W))

    indices_array = mx.concatenate(indices_list, axis=-1)  # [N, C, out_D*out_H*out_W]
    indices_array = mx.reshape(indices_array, (N, C, out_D, out_H, out_W))

    output = Tensor._from_mlx_array(output_array)
    indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))

    if is_grad_enabled() and input.requires_grad:
        output.requires_grad = True

    return output, indices


def fractional_max_pool2d(*args, **kwargs):
    """Fractional max pool 2D."""
    # Parse arguments
    input = args[0] if args else kwargs.get("input")
    kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
    output_size = args[2] if len(args) > 2 else kwargs.get("output_size", None)
    output_ratio = args[3] if len(args) > 3 else kwargs.get("output_ratio", None)
    return_indices = args[4] if len(args) > 4 else kwargs.get("return_indices", False)
    _random_samples = args[5] if len(args) > 5 else kwargs.get("_random_samples", None)

    from .layers.pooling import FractionalMaxPool2d

    pool = FractionalMaxPool2d(
        kernel_size, output_size, output_ratio, return_indices, _random_samples
    )
    return pool(input)


def fractional_max_pool2d_with_indices(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
):
    """Fractional max pool 2D with indices."""
    return fractional_max_pool2d(
        input, kernel_size, output_size, output_ratio, True, _random_samples
    )


def fractional_max_pool3d(*args, **kwargs):
    """Fractional max pool 3D."""
    # Parse arguments
    input = args[0] if args else kwargs.get("input")
    kernel_size = args[1] if len(args) > 1 else kwargs.get("kernel_size")
    output_size = args[2] if len(args) > 2 else kwargs.get("output_size", None)
    output_ratio = args[3] if len(args) > 3 else kwargs.get("output_ratio", None)
    return_indices = args[4] if len(args) > 4 else kwargs.get("return_indices", False)
    _random_samples = args[5] if len(args) > 5 else kwargs.get("_random_samples", None)

    from .layers.pooling import FractionalMaxPool3d

    pool = FractionalMaxPool3d(
        kernel_size, output_size, output_ratio, return_indices, _random_samples
    )
    return pool(input)


def fractional_max_pool3d_with_indices(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
):
    """Fractional max pool 3D with indices."""
    return fractional_max_pool3d(
        input, kernel_size, output_size, output_ratio, True, _random_samples
    )


def max_unpool1d(
    input: Tensor, indices: Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    """Partial inverse of max_pool1d."""
    from .layers.pooling import MaxUnpool1d

    unpool = MaxUnpool1d(kernel_size, stride, padding)
    return unpool(input, indices, output_size)


def max_unpool2d(
    input: Tensor, indices: Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    """Partial inverse of max_pool2d."""
    from .layers.pooling import MaxUnpool2d

    unpool = MaxUnpool2d(kernel_size, stride, padding)
    return unpool(input, indices, output_size)


def max_unpool3d(
    input: Tensor, indices: Tensor, kernel_size, stride=None, padding=0, output_size=None
):
    """Partial inverse of max_pool3d."""
    from .layers.pooling import MaxUnpool3d

    unpool = MaxUnpool3d(kernel_size, stride, padding)
    return unpool(input, indices, output_size)


def lp_pool3d(input: Tensor, norm_type: float, kernel_size, stride=None, ceil_mode=False):
    """3D power-average pooling."""
    from .layers.pooling import LPPool3d

    pool = LPPool3d(norm_type, kernel_size, stride, ceil_mode)
    return pool(input)


# =============================================================================
# Other Missing Functions
# =============================================================================


def conv_tbc(input: Tensor, weight: Tensor, bias: Optional[Tensor], pad: int = 0):
    """1D convolution with time-batch-channel layout.

    This is used by fairseq. The input is (T, B, C) instead of (B, C, T).
    """
    # Transpose to (B, C_in, T) for standard conv1d
    x = input.permute(1, 2, 0)
    # Transpose weight from (kW, C_in, C_out) to (C_out, C_in, kW)
    w = weight.permute(2, 1, 0)
    # Apply conv1d
    output = conv1d(x, w, bias, padding=pad)
    # Transpose back to (T, B, C)
    return output.permute(2, 0, 1)


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    """CTC (Connectionist Temporal Classification) loss.

    Simplified implementation - for full functionality, use the CTCLoss module.
    """
    from .losses import CTCLoss

    loss_fn = CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    return loss_fn(log_probs, targets, input_lengths, target_lengths)


def native_channel_shuffle(input: Tensor, groups: int) -> Tensor:
    """Channel shuffle operation.

    This is the native implementation of channel_shuffle.
    """
    return channel_shuffle(input, groups)
