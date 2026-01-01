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
        >>> x = mlx_compat.randn(4, 3, 100)
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
        # device, dtype, and padding_mode are accepted for PyTorch compatibility
        # padding_mode other than 'zeros' is not currently supported
        if padding_mode != 'zeros':
            import warnings
            warnings.warn(f"padding_mode='{padding_mode}' is not supported in MLX, using 'zeros'")
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
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


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
        >>> x = mlx_compat.randn(4, 3, 32, 32)
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
        # device, dtype, and padding_mode are accepted for PyTorch compatibility
        # padding_mode other than 'zeros' is not currently supported
        if padding_mode != 'zeros':
            import warnings
            warnings.warn(f"padding_mode='{padding_mode}' is not supported in MLX, using 'zeros'")
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
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


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
        >>> x = mlx_compat.randn(4, 3, 16, 32, 32)
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
        # device, dtype, and padding_mode are accepted for PyTorch compatibility
        # padding_mode other than 'zeros' is not currently supported
        if padding_mode != 'zeros':
            import warnings
            warnings.warn(f"padding_mode='{padding_mode}' is not supported in MLX, using 'zeros'")
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
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


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
        if padding_mode != 'zeros':
            import warnings
            warnings.warn(f"padding_mode='{padding_mode}' is not supported in MLX, using 'zeros'")
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
        """
        # Convert to 2D for processing: [N, C, L] -> [N, C, 1, L]
        input_4d = mx.expand_dims(input._mlx_array, axis=2)

        # Convert from NCHW to NHWC: [N, C, 1, L] -> [N, 1, L, C]
        input_nhwc = mx.transpose(input_4d, [0, 2, 3, 1])

        # Weight: [in_channels, out_channels/groups, kernel_size]
        # Need [out_channels, 1, kernel_size, in_channels] for MLX
        weight_4d = mx.expand_dims(self.weight._mlx_array, axis=2)  # [in, out/g, 1, k]
        weight_transposed = mx.transpose(weight_4d, [1, 2, 3, 0])  # [out/g, 1, k, in]

        # Perform transposed convolution
        output_nhwc = mx.conv_transpose2d(
            input_nhwc,
            weight_transposed,
            stride=(1, self.stride),
            padding=(0, self.padding),
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
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


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
        # device, dtype, and padding_mode are accepted for PyTorch compatibility
        # padding_mode other than 'zeros' is not currently supported
        if padding_mode != 'zeros':
            import warnings
            warnings.warn(f"padding_mode='{padding_mode}' is not supported in MLX, using 'zeros'")
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
        if padding_mode != 'zeros':
            import warnings
            warnings.warn(f"padding_mode='{padding_mode}' is not supported in MLX, using 'zeros'")
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
