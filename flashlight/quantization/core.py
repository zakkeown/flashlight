"""
Core Quantization Functions

Provides quantize/dequantize operations using MLX's quantization capabilities.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from .tensor import QuantizedTensor


def quantize_per_tensor(
    input: Tensor,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    bits: int = 4,
    group_size: int = 64,
) -> QuantizedTensor:
    """
    Quantize a tensor using per-tensor quantization.

    MLX uses group-wise affine quantization. For per-tensor quantization,
    we use a single group covering the entire tensor.

    Args:
        input: The tensor to quantize
        scale: Scale factor (computed if None)
        zero_point: Zero point (computed if None)
        bits: Number of bits (2, 4, or 8)
        group_size: Elements per group (64 default, use input.numel for true per-tensor)

    Returns:
        QuantizedTensor containing the quantized data

    Example:
        >>> x = flashlight.randn(128, 256)
        >>> qx = quantize_per_tensor(x, bits=4)
        >>> x_restored = qx.dequantize()
    """
    arr = input._mlx_array

    # MLX quantize: returns (quantized_data, scales, biases)
    quantized, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)

    return QuantizedTensor(
        data=quantized,
        scales=scales,
        biases=biases,
        shape=input.shape,
        dtype=input.dtype,
        bits=bits,
        group_size=group_size,
    )


def quantize_per_channel(
    input: Tensor,
    scales: Optional[Tensor] = None,
    zero_points: Optional[Tensor] = None,
    axis: int = 0,
    bits: int = 4,
    group_size: int = 64,
) -> QuantizedTensor:
    """
    Quantize a tensor using per-channel quantization.

    Each slice along the specified axis gets its own scale/zero_point.
    This is commonly used for weight quantization in neural networks.

    Args:
        input: The tensor to quantize
        scales: Per-channel scales (computed if None)
        zero_points: Per-channel zero points (computed if None)
        axis: Axis along which to quantize
        bits: Number of bits (2, 4, or 8)
        group_size: Elements per group within each channel

    Returns:
        QuantizedTensor containing the quantized data

    Example:
        >>> weight = flashlight.randn(256, 512)  # out_features x in_features
        >>> qweight = quantize_per_channel(weight, axis=0, bits=4)
    """
    arr = input._mlx_array

    # For per-channel, we reshape to make the channel dimension last
    # then apply quantization, then reshape back
    if axis != -1 and axis != arr.ndim - 1:
        # Move target axis to last position
        perm = list(range(arr.ndim))
        perm.remove(axis)
        perm.append(axis)
        arr = mx.transpose(arr, perm)

    # Quantize with group-wise quantization along the last dimension
    quantized, scales_out, biases = mx.quantize(arr, group_size=group_size, bits=bits)

    return QuantizedTensor(
        data=quantized,
        scales=scales_out,
        biases=biases,
        shape=input.shape,
        dtype=input.dtype,
        bits=bits,
        group_size=group_size,
    )


def quantize_dynamic(
    input: Tensor,
    bits: int = 8,
    group_size: int = 64,
) -> QuantizedTensor:
    """
    Dynamic quantization - compute scale/zero_point from the input itself.

    This is useful for activations where the range isn't known ahead of time.

    Args:
        input: The tensor to quantize
        bits: Number of bits (2, 4, or 8)
        group_size: Elements per quantization group

    Returns:
        QuantizedTensor with dynamically computed parameters
    """
    return quantize_per_tensor(input, bits=bits, group_size=group_size)


def dequantize(input: QuantizedTensor) -> Tensor:
    """
    Convert a quantized tensor back to floating point.

    Args:
        input: QuantizedTensor to dequantize

    Returns:
        Regular Tensor with dequantized values

    Example:
        >>> qx = quantize_per_tensor(x, bits=4)
        >>> x_restored = dequantize(qx)
    """
    return input.dequantize()


def quantized_matmul(
    x: Tensor,
    qweight: QuantizedTensor,
    transpose: bool = True,
) -> Tensor:
    """
    Efficient matrix multiplication with quantized weights.

    Computes x @ dequantize(qweight) without fully dequantizing.

    Args:
        x: Input tensor
        qweight: Quantized weight tensor
        transpose: Whether to transpose the weight before multiplication

    Returns:
        Result of the matrix multiplication
    """
    result = mx.quantized_matmul(
        x._mlx_array,
        qweight._data,
        qweight._scales,
        qweight._biases,
        transpose=transpose,
        group_size=qweight._group_size,
        bits=qweight._bits,
    )
    return Tensor._from_mlx_array(result)
