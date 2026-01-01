"""
Convolutional Layers

Implements convolutional neural network layers.
"""

import mlx.core as mx
from ..module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from ...ops.convolution import conv2d
from ...ops.conv1d import conv1d
from ...ops.conv3d import conv3d, conv_transpose3d
from typing import Union, Tuple, Optional, Any
import math


def _single(x):
    """Ensure value is a single int."""
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def _pair(x):
    """Convert single value to pair."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def _triple(x):
    """Convert single value to triple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


def _apply_padding_mode_1d(input_array, padding: int, padding_mode: str):
    """
    Apply padding to 1D input using the specified padding mode.

    Args:
        input_array: MLX array of shape [N, C, L]
        padding: Amount of padding to apply on each side
        padding_mode: One of 'zeros', 'reflect', 'replicate', 'circular'

    Returns:
        Padded MLX array of shape [N, C, L + 2*padding]
    """
    if padding == 0:
        return input_array

    if padding_mode == 'zeros':
        # Zero padding is handled by the conv operation itself
        return input_array
    elif padding_mode == 'reflect':
        # Reflect padding: [a, b, c, d] with pad=2 -> [c, b, a, b, c, d, c, b]
        # Use slicing to reflect
        N, C, L = input_array.shape
        if padding >= L:
            raise ValueError(f"Padding size {padding} should be less than input size {L} for reflect mode")
        left_pad = mx.flip(input_array[:, :, 1:padding+1], axis=2)
        right_pad = mx.flip(input_array[:, :, -(padding+1):-1], axis=2)
        return mx.concatenate([left_pad, input_array, right_pad], axis=2)
    elif padding_mode == 'replicate':
        # Replicate padding: [a, b, c, d] with pad=2 -> [a, a, a, b, c, d, d, d]
        N, C, L = input_array.shape
        left_pad = mx.broadcast_to(input_array[:, :, :1], (N, C, padding))
        right_pad = mx.broadcast_to(input_array[:, :, -1:], (N, C, padding))
        return mx.concatenate([left_pad, input_array, right_pad], axis=2)
    elif padding_mode == 'circular':
        # Circular padding: [a, b, c, d] with pad=2 -> [c, d, a, b, c, d, a, b]
        N, C, L = input_array.shape
        left_pad = input_array[:, :, -padding:]
        right_pad = input_array[:, :, :padding]
        return mx.concatenate([left_pad, input_array, right_pad], axis=2)
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")


def _apply_padding_mode_2d(input_array, padding: Tuple[int, int], padding_mode: str):
    """
    Apply padding to 2D input using the specified padding mode.

    Args:
        input_array: MLX array of shape [N, C, H, W]
        padding: Tuple of (pad_h, pad_w) for padding on each side
        padding_mode: One of 'zeros', 'reflect', 'replicate', 'circular'

    Returns:
        Padded MLX array of shape [N, C, H + 2*pad_h, W + 2*pad_w]
    """
    pad_h, pad_w = padding
    if pad_h == 0 and pad_w == 0:
        return input_array

    if padding_mode == 'zeros':
        return input_array
    elif padding_mode == 'reflect':
        N, C, H, W = input_array.shape
        if pad_h >= H or pad_w >= W:
            raise ValueError(f"Padding ({pad_h}, {pad_w}) should be less than input size ({H}, {W}) for reflect mode")
        result = input_array
        # Pad height dimension
        if pad_h > 0:
            top_pad = mx.flip(result[:, :, 1:pad_h+1, :], axis=2)
            bottom_pad = mx.flip(result[:, :, -(pad_h+1):-1, :], axis=2)
            result = mx.concatenate([top_pad, result, bottom_pad], axis=2)
        # Pad width dimension
        if pad_w > 0:
            left_pad = mx.flip(result[:, :, :, 1:pad_w+1], axis=3)
            right_pad = mx.flip(result[:, :, :, -(pad_w+1):-1], axis=3)
            result = mx.concatenate([left_pad, result, right_pad], axis=3)
        return result
    elif padding_mode == 'replicate':
        N, C, H, W = input_array.shape
        result = input_array
        # Pad height dimension
        if pad_h > 0:
            top_pad = mx.broadcast_to(result[:, :, :1, :], (N, C, pad_h, W))
            bottom_pad = mx.broadcast_to(result[:, :, -1:, :], (N, C, pad_h, W))
            result = mx.concatenate([top_pad, result, bottom_pad], axis=2)
        # Pad width dimension (now with updated H dimension)
        if pad_w > 0:
            new_H = H + 2 * pad_h if pad_h > 0 else H
            left_pad = mx.broadcast_to(result[:, :, :, :1], (N, C, new_H, pad_w))
            right_pad = mx.broadcast_to(result[:, :, :, -1:], (N, C, new_H, pad_w))
            result = mx.concatenate([left_pad, result, right_pad], axis=3)
        return result
    elif padding_mode == 'circular':
        N, C, H, W = input_array.shape
        result = input_array
        # Pad height dimension
        if pad_h > 0:
            top_pad = result[:, :, -pad_h:, :]
            bottom_pad = result[:, :, :pad_h, :]
            result = mx.concatenate([top_pad, result, bottom_pad], axis=2)
        # Pad width dimension
        if pad_w > 0:
            left_pad = result[:, :, :, -pad_w:]
            right_pad = result[:, :, :, :pad_w]
            result = mx.concatenate([left_pad, result, right_pad], axis=3)
        return result
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")


def _apply_padding_mode_3d(input_array, padding: Tuple[int, int, int], padding_mode: str):
    """
    Apply padding to 3D input using the specified padding mode.

    Args:
        input_array: MLX array of shape [N, C, D, H, W]
        padding: Tuple of (pad_d, pad_h, pad_w) for padding on each side
        padding_mode: One of 'zeros', 'reflect', 'replicate', 'circular'

    Returns:
        Padded MLX array of shape [N, C, D + 2*pad_d, H + 2*pad_h, W + 2*pad_w]
    """
    pad_d, pad_h, pad_w = padding
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return input_array

    if padding_mode == 'zeros':
        return input_array
    elif padding_mode == 'reflect':
        N, C, D, H, W = input_array.shape
        if pad_d >= D or pad_h >= H or pad_w >= W:
            raise ValueError(f"Padding ({pad_d}, {pad_h}, {pad_w}) should be less than input size ({D}, {H}, {W}) for reflect mode")
        result = input_array
        # Pad depth dimension
        if pad_d > 0:
            front_pad = mx.flip(result[:, :, 1:pad_d+1, :, :], axis=2)
            back_pad = mx.flip(result[:, :, -(pad_d+1):-1, :, :], axis=2)
            result = mx.concatenate([front_pad, result, back_pad], axis=2)
        # Pad height dimension
        if pad_h > 0:
            top_pad = mx.flip(result[:, :, :, 1:pad_h+1, :], axis=3)
            bottom_pad = mx.flip(result[:, :, :, -(pad_h+1):-1, :], axis=3)
            result = mx.concatenate([top_pad, result, bottom_pad], axis=3)
        # Pad width dimension
        if pad_w > 0:
            left_pad = mx.flip(result[:, :, :, :, 1:pad_w+1], axis=4)
            right_pad = mx.flip(result[:, :, :, :, -(pad_w+1):-1], axis=4)
            result = mx.concatenate([left_pad, result, right_pad], axis=4)
        return result
    elif padding_mode == 'replicate':
        N, C, D, H, W = input_array.shape
        result = input_array
        # Pad depth dimension
        if pad_d > 0:
            front_pad = mx.broadcast_to(result[:, :, :1, :, :], (N, C, pad_d, H, W))
            back_pad = mx.broadcast_to(result[:, :, -1:, :, :], (N, C, pad_d, H, W))
            result = mx.concatenate([front_pad, result, back_pad], axis=2)
        # Pad height dimension
        if pad_h > 0:
            new_D = D + 2 * pad_d if pad_d > 0 else D
            top_pad = mx.broadcast_to(result[:, :, :, :1, :], (N, C, new_D, pad_h, W))
            bottom_pad = mx.broadcast_to(result[:, :, :, -1:, :], (N, C, new_D, pad_h, W))
            result = mx.concatenate([top_pad, result, bottom_pad], axis=3)
        # Pad width dimension
        if pad_w > 0:
            new_D = D + 2 * pad_d if pad_d > 0 else D
            new_H = H + 2 * pad_h if pad_h > 0 else H
            left_pad = mx.broadcast_to(result[:, :, :, :, :1], (N, C, new_D, new_H, pad_w))
            right_pad = mx.broadcast_to(result[:, :, :, :, -1:], (N, C, new_D, new_H, pad_w))
            result = mx.concatenate([left_pad, result, right_pad], axis=4)
        return result
    elif padding_mode == 'circular':
        N, C, D, H, W = input_array.shape
        result = input_array
        # Pad depth dimension
        if pad_d > 0:
            front_pad = result[:, :, -pad_d:, :, :]
            back_pad = result[:, :, :pad_d, :, :]
            result = mx.concatenate([front_pad, result, back_pad], axis=2)
        # Pad height dimension
        if pad_h > 0:
            top_pad = result[:, :, :, -pad_h:, :]
            bottom_pad = result[:, :, :, :pad_h, :]
            result = mx.concatenate([top_pad, result, bottom_pad], axis=3)
        # Pad width dimension
        if pad_w > 0:
            left_pad = result[:, :, :, :, -pad_w:]
            right_pad = result[:, :, :, :, :pad_w]
            result = mx.concatenate([left_pad, result, right_pad], axis=4)
        return result
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")


class Conv1d(Module):
    """
    1D Convolutional layer.

    Applies a 1D convolution over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to add learnable bias (default: True)

    Shape:
        - Input: [N, in_channels, L]
        - Output: [N, out_channels, L_out]

    Example:
        >>> conv = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        >>> x = flashlight.randn(4, 3, 100)
        >>> output = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility
        if padding_mode not in ('zeros', 'reflect', 'replicate', 'circular'):
            raise ValueError(f"padding_mode must be one of 'zeros', 'reflect', 'replicate', 'circular', got '{padding_mode}'")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Validate groups
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # Initialize weight: [out_channels, in_channels/groups, kernel_size]
        self.weight = Parameter(
            Tensor._from_mlx_array(
                mx.zeros((out_channels, in_channels // groups, self.kernel_size))
            )
        )

        # Initialize bias
        if bias:
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(out_channels)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        fan_in = self.in_channels * self.kernel_size
        bound = math.sqrt(1.0 / fan_in)

        self.weight._mlx_array = mx.random.uniform(
            low=-bound, high=bound, shape=self.weight.shape
        )

        if self.bias is not None:
            self.bias._mlx_array = mx.random.uniform(
                low=-bound, high=bound, shape=self.bias.shape
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply 1D convolution.

        Args:
            input: Input tensor of shape [N, in_channels, L]

        Returns:
            Output tensor of shape [N, out_channels, L_out]
        """
        # Apply non-zero padding mode if needed
        if self.padding_mode != 'zeros' and self.padding > 0:
            # Manually pad the input and use padding=0 for convolution
            padded_input = Tensor._from_mlx_array(
                _apply_padding_mode_1d(input._mlx_array, self.padding, self.padding_mode)
            )
            padded_input.requires_grad = input.requires_grad
            return conv1d(
                padded_input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=0,  # Padding already applied
                dilation=self.dilation,
                groups=self.groups
            )
        else:
            return conv1d(
                input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )

    def extra_repr(self) -> str:
        s = (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        return s


class Conv2d(Module):
    """
    2D Convolutional layer.

    Applies a 2D convolution over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to add learnable bias (default: True)

    Shape:
        - Input: [N, in_channels, H, W]
        - Output: [N, out_channels, H_out, W_out]

    Example:
        >>> conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        >>> x = flashlight.randn(4, 3, 32, 32)
        >>> output = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility
        if padding_mode not in ('zeros', 'reflect', 'replicate', 'circular'):
            raise ValueError(f"padding_mode must be one of 'zeros', 'reflect', 'replicate', 'circular', got '{padding_mode}'")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Validate groups
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # Initialize weight: [out_channels, in_channels/groups, kH, kW]
        self.weight = Parameter(
            Tensor._from_mlx_array(
                mx.zeros((out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1]))
            )
        )

        # Initialize bias
        if bias:
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(out_channels)))
        else:
            self.bias = None

        # Weight transpose caching for performance
        # MLX uses [out, kH, kW, in] format, PyTorch uses [out, in, kH, kW]
        self._cached_weight_mlx = None
        self._cached_weight_id = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = math.sqrt(1.0 / fan_in)

        self.weight._mlx_array = mx.random.uniform(
            low=-bound, high=bound, shape=self.weight.shape
        )

        if self.bias is not None:
            self.bias._mlx_array = mx.random.uniform(
                low=-bound, high=bound, shape=self.bias.shape
            )
        # Invalidate weight cache after reset
        self._cached_weight_mlx = None
        self._cached_weight_id = None

    def _get_weight_mlx(self):
        """Get weight in MLX format [out, kH, kW, in], cached for performance."""
        # Check if weight has changed by comparing array id
        current_id = id(self.weight._mlx_array)
        if self._cached_weight_mlx is None or self._cached_weight_id != current_id:
            # Convert from [out, in, kH, kW] to [out, kH, kW, in]
            self._cached_weight_mlx = mx.transpose(self.weight._mlx_array, [0, 2, 3, 1])
            self._cached_weight_id = current_id
        return self._cached_weight_mlx

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply 2D convolution.

        Args:
            input: Input tensor of shape [N, in_channels, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H_out, W_out]
        """
        # Apply non-zero padding mode if needed
        if self.padding_mode != 'zeros' and (self.padding[0] > 0 or self.padding[1] > 0):
            # Manually pad the input and use padding=0 for convolution
            padded_input = Tensor._from_mlx_array(
                _apply_padding_mode_2d(input._mlx_array, self.padding, self.padding_mode)
            )
            padded_input.requires_grad = input.requires_grad
            return conv2d(
                padded_input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=(0, 0),  # Padding already applied
                dilation=self.dilation,
                groups=self.groups,
                _cached_weight_mlx=self._get_weight_mlx()
            )
        else:
            return conv2d(
                input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                _cached_weight_mlx=self._get_weight_mlx()
            )

    def extra_repr(self) -> str:
        s = (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        return s


class Conv3d(Module):
    """
    3D Convolutional layer.

    Applies a 3D convolution over an input signal composed of several input planes.

    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to add learnable bias (default: True)

    Shape:
        - Input: [N, in_channels, D, H, W]
        - Output: [N, out_channels, D_out, H_out, W_out]

    Example:
        >>> conv = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        >>> x = flashlight.randn(4, 3, 16, 32, 32)
        >>> output = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility
        if padding_mode not in ('zeros', 'reflect', 'replicate', 'circular'):
            raise ValueError(f"padding_mode must be one of 'zeros', 'reflect', 'replicate', 'circular', got '{padding_mode}'")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Validate groups
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # Initialize weight: [out_channels, in_channels/groups, kD, kH, kW]
        self.weight = Parameter(
            Tensor._from_mlx_array(
                mx.zeros((out_channels, in_channels // groups,
                          self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
            )
        )

        # Initialize bias
        if bias:
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(out_channels)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = math.sqrt(1.0 / fan_in)

        self.weight._mlx_array = mx.random.uniform(
            low=-bound, high=bound, shape=self.weight.shape
        )

        if self.bias is not None:
            self.bias._mlx_array = mx.random.uniform(
                low=-bound, high=bound, shape=self.bias.shape
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply 3D convolution.

        Args:
            input: Input tensor of shape [N, in_channels, D, H, W]

        Returns:
            Output tensor of shape [N, out_channels, D_out, H_out, W_out]
        """
        # Apply non-zero padding mode if needed
        if self.padding_mode != 'zeros' and (self.padding[0] > 0 or self.padding[1] > 0 or self.padding[2] > 0):
            # Manually pad the input and use padding=0 for convolution
            padded_input = Tensor._from_mlx_array(
                _apply_padding_mode_3d(input._mlx_array, self.padding, self.padding_mode)
            )
            padded_input.requires_grad = input.requires_grad
            return conv3d(
                padded_input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=(0, 0, 0),  # Padding already applied
                dilation=self.dilation,
                groups=self.groups
            )
        else:
            return conv3d(
                input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )

    def extra_repr(self) -> str:
        s = (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        return s


class ConvTranspose1d(Module):
    """
    1D Transposed Convolutional layer (deconvolution).

    Applies a 1D transposed convolution over an input signal.

    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        bias: Whether to add learnable bias (default: True)
        dilation: Spacing between kernel elements (default: 1)

    Shape:
        - Input: [N, in_channels, L]
        - Output: [N, out_channels, L_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # Only 'zeros' is supported for transposed convolutions (same as PyTorch)
        if padding_mode != 'zeros':
            raise ValueError("Only 'zeros' padding mode is supported for ConvTranspose1d")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.output_padding = _single(output_padding)
        self.dilation = _single(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # Weight shape for transpose conv: [in_channels, out_channels/groups, kernel_size]
        self.weight = Parameter(
            Tensor._from_mlx_array(
                mx.zeros((in_channels, out_channels // groups, self.kernel_size))
            )
        )

        if bias:
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(out_channels)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        fan_in = self.out_channels * self.kernel_size
        bound = math.sqrt(1.0 / fan_in)

        self.weight._mlx_array = mx.random.uniform(
            low=-bound, high=bound, shape=self.weight.shape
        )

        if self.bias is not None:
            self.bias._mlx_array = mx.random.uniform(
                low=-bound, high=bound, shape=self.bias.shape
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply transposed 1D convolution.

        Args:
            input: Input tensor of shape [N, in_channels, L]

        Returns:
            Output tensor of shape [N, out_channels, L_out]

        Note:
            Output size is computed as:
            L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1
        """
        # Convert to 2D for processing: [N, C, L] -> [N, C, 1, L]
        input_4d = mx.expand_dims(input._mlx_array, axis=2)

        # Convert from NCHW to NHWC: [N, C, 1, L] -> [N, 1, L, C]
        input_nhwc = mx.transpose(input_4d, [0, 2, 3, 1])

        # Weight: [in_channels, out_channels/groups, kernel_size]
        # Need [out_channels, 1, kernel_size, in_channels] for MLX
        weight_4d = mx.expand_dims(self.weight._mlx_array, axis=2)  # [in, out/g, 1, k]
        weight_transposed = mx.transpose(weight_4d, [1, 2, 3, 0])  # [out/g, 1, k, in]

        # Perform transposed convolution with dilation support
        output_nhwc = mx.conv_transpose2d(
            input_nhwc,
            weight_transposed,
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, self.dilation),
            groups=self.groups
        )

        # Handle output padding
        if self.output_padding > 0:
            pad_config = [(0, 0), (0, 0), (0, self.output_padding), (0, 0)]
            output_nhwc = mx.pad(output_nhwc, pad_config)

        # Add bias
        if self.bias is not None:
            output_nhwc = output_nhwc + self.bias._mlx_array

        # Convert back to NCHW and squeeze: [N, 1, L_out, C] -> [N, C, L_out]
        output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])
        output_3d = mx.squeeze(output_nchw, axis=2)

        result = Tensor._from_mlx_array(output_3d)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        s = (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        return s


class ConvTranspose2d(Module):
    """
    2D Transposed Convolutional layer (deconvolution).

    Applies a 2D transposed convolution over an input image.

    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to add learnable bias (default: True)

    Shape:
        - Input: [N, in_channels, H, W]
        - Output: [N, out_channels, H_out, W_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # Only 'zeros' is supported for transposed convolutions (same as PyTorch)
        if padding_mode != 'zeros':
            raise ValueError("Only 'zeros' padding mode is supported for ConvTranspose2d")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Validate
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # Note: weight shape for transpose conv is [in_channels, out_channels/groups, kH, kW]
        self.weight = Parameter(
            Tensor._from_mlx_array(
                mx.zeros((in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1]))
            )
        )

        if bias:
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(out_channels)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        fan_in = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = math.sqrt(1.0 / fan_in)

        self.weight._mlx_array = mx.random.uniform(
            low=-bound, high=bound, shape=self.weight.shape
        )

        if self.bias is not None:
            self.bias._mlx_array = mx.random.uniform(
                low=-bound, high=bound, shape=self.bias.shape
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply transposed 2D convolution.

        Args:
            input: Input tensor of shape [N, in_channels, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H_out, W_out]
        """
        # Convert input from NCHW to NHWC
        input_nhwc = mx.transpose(input._mlx_array, [0, 2, 3, 1])

        # Weight shape: [in_channels, out_channels/groups, kH, kW]
        # MLX conv_transpose expects: [out_channels, kH, kW, in_channels]
        # So we need to transpose appropriately
        weight_transposed = mx.transpose(self.weight._mlx_array, [1, 2, 3, 0])

        # Perform transposed convolution
        output_nhwc = mx.conv_transpose2d(
            input_nhwc,
            weight_transposed,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

        # Add output padding if specified
        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            # Pad on the right and bottom
            pad_config = [
                (0, 0),  # N
                (0, self.output_padding[0]),  # H
                (0, self.output_padding[1]),  # W
                (0, 0)   # C
            ]
            output_nhwc = mx.pad(output_nhwc, pad_config)

        # Add bias if provided
        if self.bias is not None:
            output_nhwc = output_nhwc + self.bias._mlx_array

        # Convert back to NCHW
        output_nchw = mx.transpose(output_nhwc, [0, 3, 1, 2])

        result = Tensor._from_mlx_array(output_nchw)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


class ConvTranspose3d(Module):
    """
    3D Transposed Convolutional layer (deconvolution).

    Applies a 3D transposed convolution over an input volume.

    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        bias: Whether to add learnable bias (default: True)
        dilation: Spacing between kernel elements (default: 1)

    Shape:
        - Input: [N, in_channels, D, H, W]
        - Output: [N, out_channels, D_out, H_out, W_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = 'zeros',
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # Only 'zeros' is supported for transposed convolutions (same as PyTorch)
        if padding_mode != 'zeros':
            raise ValueError("Only 'zeros' padding mode is supported for ConvTranspose3d")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        # Weight shape: [in_channels, out_channels/groups, kD, kH, kW]
        self.weight = Parameter(
            Tensor._from_mlx_array(
                mx.zeros((in_channels, out_channels // groups,
                          self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))
            )
        )

        if bias:
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(out_channels)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        fan_in = self.out_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = math.sqrt(1.0 / fan_in)

        self.weight._mlx_array = mx.random.uniform(
            low=-bound, high=bound, shape=self.weight.shape
        )

        if self.bias is not None:
            self.bias._mlx_array = mx.random.uniform(
                low=-bound, high=bound, shape=self.bias.shape
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply transposed 3D convolution.

        Args:
            input: Input tensor of shape [N, in_channels, D, H, W]

        Returns:
            Output tensor of shape [N, out_channels, D_out, H_out, W_out]
        """
        return conv_transpose3d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation
        )

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']
