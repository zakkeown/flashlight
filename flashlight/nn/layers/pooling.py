"""
Pooling Layers

Implements pooling layers for neural networks.
"""

from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as mxnn

from ...ops.pooling import avg_pool2d, max_pool2d
from ...tensor import Tensor
from ..module import Module


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


class MaxPool2d(Module):
    """
    2D Max Pooling layer.

    Applies a 2D max pooling over an input signal.

    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply (default: 0)
        dilation: Dilation factor (default: 1)
        return_indices: Whether to return indices (not supported)
        ceil_mode: Whether to use ceil for output size (default: False)

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H_out, W_out]

    Example:
        >>> pool = nn.MaxPool2d(kernel_size=2, stride=2)
        >>> x = flashlight.randn(4, 64, 32, 32)
        >>> output = pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        """
        Apply max pooling.

        Args:
            input: Input tensor of shape [N, C, H, W]

        Returns:
            If return_indices is False: Output tensor of shape [N, C, H_out, W_out]
            If return_indices is True: Tuple of (output, indices)
        """
        if self.return_indices:
            from ..functional import max_pool2d_with_indices

            return max_pool2d_with_indices(
                input,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )
        else:
            return max_pool2d(
                input,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                return_indices=self.return_indices,
                ceil_mode=self.ceil_mode,
            )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, " f"stride={self.stride}, " f"padding={self.padding}"
        )


class AvgPool2d(Module):
    """
    2D Average Pooling layer.

    Applies a 2D average pooling over an input signal.

    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply (default: 0)
        ceil_mode: Whether to use ceil for output size (default: False)
        count_include_pad: Whether to include padding in average (default: True)

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H_out, W_out]

    Example:
        >>> pool = nn.AvgPool2d(kernel_size=2, stride=2)
        >>> x = flashlight.randn(4, 64, 32, 32)
        >>> output = pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = None,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply average pooling.

        Args:
            input: Input tensor of shape [N, C, H, W]

        Returns:
            Output tensor of shape [N, C, H_out, W_out]
        """
        return avg_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, " f"stride={self.stride}, " f"padding={self.padding}"
        )


class MaxPool1d(Module):
    """
    1D Max Pooling layer.

    Applies a 1D max pooling over an input signal.

    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply (default: 0)
        dilation: Dilation factor (default: 1)
        return_indices: Whether to return indices (not supported)
        ceil_mode: Whether to use ceil for output size (default: False)

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L_out]
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        """Apply 1D max pooling.

        Returns:
            If return_indices is False: Output tensor of shape [N, C, L_out]
            If return_indices is True: Tuple of (output, indices)
        """
        # _single returns int or tuple, normalize to int for 1D operations
        ks = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        st = self.stride[0] if isinstance(self.stride, tuple) else self.stride
        pd = self.padding[0] if isinstance(self.padding, tuple) else self.padding
        dl = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation

        if self.return_indices:
            from ..functional import max_pool1d_with_indices

            return max_pool1d_with_indices(
                input, kernel_size=ks, stride=st, padding=pd, dilation=dl, ceil_mode=self.ceil_mode
            )
        else:
            # MLX expects NLC format (channels last)
            x = input._mlx_array
            x = mx.transpose(x, [0, 2, 1])  # NCL -> NLC

            # Use MLX MaxPool1d
            pool = mxnn.MaxPool1d(
                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
            )
            result = pool(x)

            # Convert back: NLC -> NCL
            result = mx.transpose(result, [0, 2, 1])

            return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool1d(Module):
    """
    1D Average Pooling layer.

    Applies a 1D average pooling over an input signal.

    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply (default: 0)
        ceil_mode: Whether to use ceil for output size (default: False)
        count_include_pad: Whether to include padding in average (default: True)

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L_out]
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        """Apply 1D average pooling."""
        # MLX expects NLC format (channels last)
        x = input._mlx_array
        x = mx.transpose(x, [0, 2, 1])  # NCL -> NLC

        # Use MLX AvgPool1d
        pool = mxnn.AvgPool1d(
            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        result = pool(x)

        # Convert back: NLC -> NCL
        result = mx.transpose(result, [0, 2, 1])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class MaxPool3d(Module):
    """
    3D Max Pooling layer.

    Applies a 3D max pooling over an input signal.

    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply (default: 0)
        dilation: Dilation factor (default: 1)
        return_indices: Whether to return indices (not supported)
        ceil_mode: Whether to use ceil for output size (default: False)

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D_out, H_out, W_out]
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], None] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) if stride is not None else self.kernel_size
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        """Apply 3D max pooling using loop over depth.

        Returns:
            If return_indices is False: Output tensor of shape [N, C, D_out, H_out, W_out]
            If return_indices is True: Tuple of (output, indices)
        """
        if self.return_indices:
            from ..functional import max_pool3d_with_indices

            return max_pool3d_with_indices(
                input,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )

        x = input._mlx_array
        N, C, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding

        # Pad along depth dimension if needed
        if pD > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pD, pD), (0, 0), (0, 0)])

        # Convert NCDHW to NDHWC
        x = mx.transpose(x, [0, 2, 3, 4, 1])

        # Calculate output depth
        D_padded = D + 2 * pD
        D_out = (D_padded - kD) // sD + 1

        # Create MLX 2D max pooling layer
        pool_2d = mxnn.MaxPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=(pH, pW))

        outputs = []
        for d_out in range(D_out):
            d_start = d_out * sD
            # Get depth slice and apply max over kernel depth
            depth_slice = x[:, d_start : d_start + kD, :, :, :]
            depth_max = mx.max(depth_slice, axis=1)  # [N, H, W, C]

            # Apply 2D pooling
            pooled = pool_2d(depth_max)
            outputs.append(pooled)

        # Stack along depth dimension
        result = mx.stack(outputs, axis=1)  # [N, D_out, H_out, W_out, C]

        # Convert back to NCDHW
        result = mx.transpose(result, [0, 4, 1, 2, 3])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool3d(Module):
    """
    3D Average Pooling layer.

    Applies a 3D average pooling over an input signal.

    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to apply (default: 0)
        ceil_mode: Whether to use ceil for output size (default: False)
        count_include_pad: Whether to include padding in average (default: True)

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D_out, H_out, W_out]
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], None] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = None,
    ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) if stride is not None else self.kernel_size
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        """Apply 3D average pooling using loop over depth."""
        x = input._mlx_array
        N, C, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding

        # Pad along depth dimension if needed
        if pD > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pD, pD), (0, 0), (0, 0)])

        # Convert NCDHW to NDHWC
        x = mx.transpose(x, [0, 2, 3, 4, 1])

        # Calculate output depth
        D_padded = D + 2 * pD
        D_out = (D_padded - kD) // sD + 1

        # Create MLX 2D average pooling layer
        pool_2d = mxnn.AvgPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=(pH, pW))

        outputs = []
        for d_out in range(D_out):
            d_start = d_out * sD
            # Get depth slice and apply mean over kernel depth
            depth_slice = x[:, d_start : d_start + kD, :, :, :]
            depth_avg = mx.mean(depth_slice, axis=1)  # [N, H, W, C]

            # Apply 2D pooling
            pooled = pool_2d(depth_avg)
            outputs.append(pooled)

        # Stack along depth dimension
        result = mx.stack(outputs, axis=1)  # [N, D_out, H_out, W_out, C]

        # Convert back to NCDHW
        result = mx.transpose(result, [0, 4, 1, 2, 3])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool1d(Module):
    """
    1D Adaptive Average Pooling layer.

    Produces output of specified size regardless of input size.

    Args:
        output_size: Target output size

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, output_size]
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = _single(output_size)

    def forward(self, input: Tensor) -> Tensor:
        """Apply adaptive average pooling."""
        x = input._mlx_array
        N, C, L = x.shape
        out_size = self.output_size

        if out_size == L:
            return input
        elif out_size == 1:
            result = mx.mean(x, axis=2, keepdims=True)
        else:
            # Calculate adaptive kernel and stride
            stride = L // out_size
            kernel_size = L - (out_size - 1) * stride

            # Convert NCL to NLC (channel last) for MLX
            x_nlc = mx.transpose(x, [0, 2, 1])

            # Use MLX AvgPool1d
            pool = mxnn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
            result = pool(x_nlc)

            # Convert back to NCL
            result = mx.transpose(result, [0, 2, 1])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
    """
    2D Adaptive Average Pooling layer.

    Produces output of specified size regardless of input size.

    Args:
        output_size: Target output size (H, W) or single int for square

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, output_size[0], output_size[1]]
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = _pair(output_size)

    def forward(self, input: Tensor) -> Tensor:
        """Apply adaptive average pooling."""
        x = input._mlx_array
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        if out_H == H and out_W == W:
            return input
        elif out_H == 1 and out_W == 1:
            result = mx.mean(x, axis=(2, 3), keepdims=True)
        else:
            # Calculate adaptive kernel and stride
            stride_h = H // out_H
            stride_w = W // out_W
            kernel_h = H - (out_H - 1) * stride_h
            kernel_w = W - (out_W - 1) * stride_w

            # Convert NCHW to NHWC
            x_nhwc = mx.transpose(x, [0, 2, 3, 1])

            # Use MLX AvgPool2d
            pool = mxnn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0
            )
            result = pool(x_nhwc)

            # Convert back to NCHW
            result = mx.transpose(result, [0, 3, 1, 2])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveAvgPool3d(Module):
    """
    3D Adaptive Average Pooling layer.

    Produces output of specified size regardless of input size.

    Args:
        output_size: Target output size (D, H, W) or single int for cube

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, output_size[0], output_size[1], output_size[2]]
    """

    def __init__(self, output_size: Union[int, Tuple[int, int, int]]):
        super().__init__()
        self.output_size = _triple(output_size)

    def forward(self, input: Tensor) -> Tensor:
        """Apply adaptive average pooling."""
        x = input._mlx_array
        N, C, D, H, W = x.shape
        out_D, out_H, out_W = self.output_size

        if out_D == D and out_H == H and out_W == W:
            return input
        elif out_D == 1 and out_H == 1 and out_W == 1:
            result = mx.mean(x, axis=(2, 3, 4), keepdims=True)
        else:
            # Simple implementation using reshape + mean for specific output sizes
            # This handles common cases like global pooling (1,1,1) efficiently
            # For arbitrary sizes, we compute adaptive strides

            # Convert NCDHW to NDHWC
            x_ndhwc = mx.transpose(x, [0, 2, 3, 4, 1])

            # Split and pool each dimension
            stride_d = D // out_D
            stride_h = H // out_H
            stride_w = W // out_W
            kernel_h = H - (out_H - 1) * stride_h
            kernel_w = W - (out_W - 1) * stride_w

            # Create MLX 2D avg pool for H,W pooling
            pool_2d = mxnn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0
            )

            outputs = []
            for d in range(out_D):
                d_start = d * stride_d
                d_end = D if d == out_D - 1 else (d + 1) * stride_d
                slice_d = x_ndhwc[:, d_start:d_end, :, :, :]
                pooled_d = mx.mean(slice_d, axis=1, keepdims=True)  # Pool depth

                # Pool H and W using mxnn.AvgPool2d
                pooled_hw = pool_2d(mx.squeeze(pooled_d, axis=1))
                outputs.append(mx.expand_dims(pooled_hw, axis=1))

            result = mx.concatenate(outputs, axis=1)
            # Convert back to NCDHW
            result = mx.transpose(result, [0, 4, 1, 2, 3])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool1d(Module):
    """
    1D Adaptive Max Pooling layer.

    Produces output of specified size regardless of input size.

    Args:
        output_size: Target output size
        return_indices: Whether to return indices of max values

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, output_size]
        - Indices (if return_indices=True): [N, C, output_size]
    """

    def __init__(self, output_size: int, return_indices: bool = False):
        super().__init__()
        self.output_size = _single(output_size)
        self.return_indices = return_indices

    def forward(self, input: Tensor):
        """Apply adaptive max pooling.

        Returns:
            If return_indices is False: Output tensor of shape [N, C, output_size]
            If return_indices is True: Tuple of (output, indices)
        """
        x = input._mlx_array
        N, C, L = x.shape
        out_size = self.output_size

        if out_size == L:
            # Identity case
            if self.return_indices:
                indices_array = mx.broadcast_to(mx.arange(L).reshape(1, 1, L), (N, C, L))
                return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))
            return input
        elif out_size == 1:
            result = mx.max(x, axis=2, keepdims=True)
            if self.return_indices:
                indices = mx.argmax(x, axis=2, keepdims=True)
                return Tensor._from_mlx_array(result), Tensor._from_mlx_array(
                    indices.astype(mx.int64)
                )
            return Tensor._from_mlx_array(result)
        else:
            # Compute adaptive pooling regions
            outputs = []
            indices_list = []

            for i in range(out_size):
                # Compute start and end indices for this output position
                start = (i * L) // out_size
                end = ((i + 1) * L) // out_size

                # Extract the region and find max
                region = x[:, :, start:end]  # [N, C, region_size]
                max_vals = mx.max(region, axis=2, keepdims=True)  # [N, C, 1]
                outputs.append(max_vals)

                if self.return_indices:
                    # Find the indices of max values (relative to region)
                    region_argmax = mx.argmax(region, axis=2, keepdims=True)  # [N, C, 1]
                    # Convert to absolute indices in the input
                    abs_indices = region_argmax + start
                    indices_list.append(abs_indices)

            # Concatenate outputs
            output_array = mx.concatenate(outputs, axis=2)  # [N, C, output_size]
            result = Tensor._from_mlx_array(output_array)

            if self.return_indices:
                indices_array = mx.concatenate(indices_list, axis=2)  # [N, C, output_size]
                indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))
                return result, indices

            return result

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, return_indices={self.return_indices}"


class AdaptiveMaxPool2d(Module):
    """
    2D Adaptive Max Pooling layer.

    Produces output of specified size regardless of input size.

    Args:
        output_size: Target output size (H, W) or single int for square
        return_indices: Whether to return indices of max values

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, output_size[0], output_size[1]]
        - Indices (if return_indices=True): [N, C, output_size[0], output_size[1]]
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]], return_indices: bool = False):
        super().__init__()
        self.output_size = _pair(output_size)
        self.return_indices = return_indices

    def forward(self, input: Tensor):
        """Apply adaptive max pooling.

        Returns:
            If return_indices is False: Output tensor of shape [N, C, out_H, out_W]
            If return_indices is True: Tuple of (output, indices)
        """
        x = input._mlx_array
        N, C, H, W = x.shape
        out_H, out_W = self.output_size

        if out_H == H and out_W == W:
            # Identity case
            if self.return_indices:
                # Create indices grid: flattened H*W indices
                h_indices = mx.arange(H).reshape(1, 1, H, 1)
                w_indices = mx.arange(W).reshape(1, 1, 1, W)
                indices_array = h_indices * W + w_indices
                indices_array = mx.broadcast_to(indices_array, (N, C, H, W))
                return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))
            return input
        elif out_H == 1 and out_W == 1:
            result = mx.max(x, axis=(2, 3), keepdims=True)
            if self.return_indices:
                # Flatten spatial dims and find argmax
                x_flat = mx.reshape(x, (N, C, -1))  # [N, C, H*W]
                indices = mx.argmax(x_flat, axis=2, keepdims=True)  # [N, C, 1]
                indices = mx.reshape(indices, (N, C, 1, 1))  # [N, C, 1, 1]
                return Tensor._from_mlx_array(result), Tensor._from_mlx_array(
                    indices.astype(mx.int64)
                )
            return Tensor._from_mlx_array(result)
        else:
            # Compute adaptive pooling regions
            outputs = []
            indices_list = []

            for i in range(out_H):
                h_start = (i * H) // out_H
                h_end = ((i + 1) * H) // out_H
                row_outputs = []
                row_indices = []

                for j in range(out_W):
                    w_start = (j * W) // out_W
                    w_end = ((j + 1) * W) // out_W

                    # Extract the region and find max
                    region = x[:, :, h_start:h_end, w_start:w_end]  # [N, C, region_h, region_w]
                    max_vals = mx.max(region, axis=(2, 3), keepdims=True)  # [N, C, 1, 1]
                    row_outputs.append(max_vals)

                    if self.return_indices:
                        # Flatten region and find argmax
                        region_h = h_end - h_start
                        region_w = w_end - w_start
                        region_flat = mx.reshape(region, (N, C, -1))  # [N, C, region_h * region_w]
                        local_argmax = mx.argmax(region_flat, axis=2)  # [N, C]

                        # Convert local index to (local_h, local_w)
                        local_h = local_argmax // region_w
                        local_w = local_argmax % region_w

                        # Convert to global flat index in original H*W space
                        global_h = h_start + local_h
                        global_w = w_start + local_w
                        global_idx = global_h * W + global_w  # [N, C]
                        global_idx = mx.reshape(global_idx, (N, C, 1, 1))
                        row_indices.append(global_idx)

                # Concatenate along width dimension
                row_output = mx.concatenate(row_outputs, axis=3)  # [N, C, 1, out_W]
                outputs.append(row_output)
                if self.return_indices:
                    row_idx = mx.concatenate(row_indices, axis=3)  # [N, C, 1, out_W]
                    indices_list.append(row_idx)

            # Concatenate along height dimension
            output_array = mx.concatenate(outputs, axis=2)  # [N, C, out_H, out_W]
            result = Tensor._from_mlx_array(output_array)

            if self.return_indices:
                indices_array = mx.concatenate(indices_list, axis=2)  # [N, C, out_H, out_W]
                indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))
                return result, indices

            return result

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, return_indices={self.return_indices}"


class AdaptiveMaxPool3d(Module):
    """
    3D Adaptive Max Pooling layer.

    Produces output of specified size regardless of input size.

    Args:
        output_size: Target output size (D, H, W) or single int for cube
        return_indices: Whether to return indices of max values

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, output_size[0], output_size[1], output_size[2]]
        - Indices (if return_indices=True): [N, C, output_size[0], output_size[1], output_size[2]]
    """

    def __init__(self, output_size: Union[int, Tuple[int, int, int]], return_indices: bool = False):
        super().__init__()
        self.output_size = _triple(output_size)
        self.return_indices = return_indices

    def forward(self, input: Tensor):
        """Apply adaptive max pooling.

        Returns:
            If return_indices is False: Output tensor of shape [N, C, out_D, out_H, out_W]
            If return_indices is True: Tuple of (output, indices)
        """
        x = input._mlx_array
        N, C, D, H, W = x.shape
        out_D, out_H, out_W = self.output_size

        if out_D == D and out_H == H and out_W == W:
            # Identity case
            if self.return_indices:
                # Create indices grid: flattened D*H*W indices
                d_indices = mx.arange(D).reshape(1, 1, D, 1, 1)
                h_indices = mx.arange(H).reshape(1, 1, 1, H, 1)
                w_indices = mx.arange(W).reshape(1, 1, 1, 1, W)
                indices_array = d_indices * (H * W) + h_indices * W + w_indices
                indices_array = mx.broadcast_to(indices_array, (N, C, D, H, W))
                return input, Tensor._from_mlx_array(indices_array.astype(mx.int64))
            return input
        elif out_D == 1 and out_H == 1 and out_W == 1:
            result = mx.max(x, axis=(2, 3, 4), keepdims=True)
            if self.return_indices:
                # Flatten spatial dims and find argmax
                x_flat = mx.reshape(x, (N, C, -1))  # [N, C, D*H*W]
                indices = mx.argmax(x_flat, axis=2, keepdims=True)  # [N, C, 1]
                indices = mx.reshape(indices, (N, C, 1, 1, 1))  # [N, C, 1, 1, 1]
                return Tensor._from_mlx_array(result), Tensor._from_mlx_array(
                    indices.astype(mx.int64)
                )
            return Tensor._from_mlx_array(result)
        else:
            # Compute adaptive pooling regions
            outputs = []
            indices_list = []

            for di in range(out_D):
                d_start = (di * D) // out_D
                d_end = ((di + 1) * D) // out_D
                depth_outputs = []
                depth_indices = []

                for hi in range(out_H):
                    h_start = (hi * H) // out_H
                    h_end = ((hi + 1) * H) // out_H
                    row_outputs = []
                    row_indices = []

                    for wi in range(out_W):
                        w_start = (wi * W) // out_W
                        w_end = ((wi + 1) * W) // out_W

                        # Extract the region and find max
                        region = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        max_vals = mx.max(region, axis=(2, 3, 4), keepdims=True)  # [N, C, 1, 1, 1]
                        row_outputs.append(max_vals)

                        if self.return_indices:
                            # Flatten region and find argmax
                            region_d = d_end - d_start
                            region_h = h_end - h_start
                            region_w = w_end - w_start
                            region_flat = mx.reshape(
                                region, (N, C, -1)
                            )  # [N, C, region_d * region_h * region_w]
                            local_argmax = mx.argmax(region_flat, axis=2)  # [N, C]

                            # Convert local index to (local_d, local_h, local_w)
                            local_d = local_argmax // (region_h * region_w)
                            remainder = local_argmax % (region_h * region_w)
                            local_h = remainder // region_w
                            local_w = remainder % region_w

                            # Convert to global flat index in original D*H*W space
                            global_d = d_start + local_d
                            global_h = h_start + local_h
                            global_w = w_start + local_w
                            global_idx = global_d * (H * W) + global_h * W + global_w  # [N, C]
                            global_idx = mx.reshape(global_idx, (N, C, 1, 1, 1))
                            row_indices.append(global_idx)

                    # Concatenate along width dimension
                    row_output = mx.concatenate(row_outputs, axis=4)  # [N, C, 1, 1, out_W]
                    depth_outputs.append(row_output)
                    if self.return_indices:
                        row_idx = mx.concatenate(row_indices, axis=4)  # [N, C, 1, 1, out_W]
                        depth_indices.append(row_idx)

                # Concatenate along height dimension
                depth_output = mx.concatenate(depth_outputs, axis=3)  # [N, C, 1, out_H, out_W]
                outputs.append(depth_output)
                if self.return_indices:
                    depth_idx = mx.concatenate(depth_indices, axis=3)  # [N, C, 1, out_H, out_W]
                    indices_list.append(depth_idx)

            # Concatenate along depth dimension
            output_array = mx.concatenate(outputs, axis=2)  # [N, C, out_D, out_H, out_W]
            result = Tensor._from_mlx_array(output_array)

            if self.return_indices:
                indices_array = mx.concatenate(indices_list, axis=2)  # [N, C, out_D, out_H, out_W]
                indices = Tensor._from_mlx_array(indices_array.astype(mx.int64))
                return result, indices

            return result

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}, return_indices={self.return_indices}"


class LPPool1d(Module):
    """
    Apply 1D power-average pooling.

    The output is computed as:
        out = (sum(|x|^p))^(1/p)

    Args:
        norm_type: The exponent value p (typically 1 or 2)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
        ceil_mode: If True, use ceil instead of floor for output size

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L_out]

    Example:
        >>> pool = nn.LPPool1d(norm_type=2, kernel_size=3)
        >>> x = flashlight.randn(1, 16, 50)
        >>> output = pool(x)
    """

    def __init__(
        self, norm_type: float, kernel_size: int, stride: int = None, ceil_mode: bool = False
    ):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Apply 1D LP pooling."""
        x = input._mlx_array
        N, C, L = x.shape

        # Take absolute value and raise to power p
        x_p = mx.power(mx.abs(x), self.norm_type)

        # Convert NCL to NLC (channel last) for MLX
        x_p = mx.transpose(x_p, [0, 2, 1])

        # Sum pooling using avg pooling scaled by kernel size
        pool = mxnn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=0)
        pooled = pool(x_p)

        # Multiply by kernel_size to get sum, then take p-th root
        pooled = pooled * self.kernel_size
        result = mx.power(pooled, 1.0 / self.norm_type)

        # Convert back: NLC -> NCL
        result = mx.transpose(result, [0, 2, 1])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}"


class LPPool2d(Module):
    """
    Apply 2D power-average pooling.

    The output is computed as:
        out = (sum(|x|^p))^(1/p)

    Args:
        norm_type: The exponent value p (typically 1 or 2)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
        ceil_mode: If True, use ceil instead of floor for output size

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H_out, W_out]

    Example:
        >>> pool = nn.LPPool2d(norm_type=2, kernel_size=3)
        >>> x = flashlight.randn(1, 16, 50, 50)
        >>> output = pool(x)
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Apply 2D LP pooling."""
        x = input._mlx_array
        kH, kW = self.kernel_size
        sH, sW = self.stride

        # Take absolute value and raise to power p
        x_p = mx.power(mx.abs(x), self.norm_type)

        # Convert NCHW to NHWC
        x_p = mx.transpose(x_p, [0, 2, 3, 1])

        # Sum pooling using avg pooling scaled by kernel area
        pool = mxnn.AvgPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=0)
        pooled = pool(x_p)

        # Multiply by kernel area to get sum, then take p-th root
        kernel_area = kH * kW
        pooled = pooled * kernel_area
        result = mx.power(pooled, 1.0 / self.norm_type)

        # Convert back to NCHW
        result = mx.transpose(result, [0, 3, 1, 2])

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}"


class LPPool3d(Module):
    """
    Apply 3D power-average pooling.

    The output is computed as:
        out = (sum(|x|^p))^(1/p)

    Args:
        norm_type: The exponent value p (typically 1 or 2)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
        ceil_mode: If True, use ceil instead of floor for output size

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D_out, H_out, W_out]
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], None] = None,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) if stride is not None else self.kernel_size
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        """Apply 3D LP pooling."""
        x = input._mlx_array
        N, C, D, H, W = x.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride

        # Calculate output size
        D_out = (D - kD) // sD + 1
        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1

        # Take absolute value and raise to power p
        x_p = mx.power(mx.abs(x), self.norm_type)

        # 3D LP pooling using slice-based approach with MLX
        # Process each depth slice and combine
        output_slices = []
        for d in range(D_out):
            d_start = d * sD
            # Extract a 2D+depth slice: shape (N, C, kD, H, W)
            depth_slice = x_p[:, :, d_start : d_start + kD, :, :]

            # Sum over the depth kernel dimension
            depth_sum = mx.sum(depth_slice, axis=2)  # Shape: (N, C, H, W)

            # Convert to NHWC for 2D pooling
            depth_sum_nhwc = mx.transpose(depth_sum, [0, 2, 3, 1])

            # Apply 2D avg pooling and scale
            pool = mxnn.AvgPool2d(kernel_size=(kH, kW), stride=(sH, sW), padding=0)
            pooled = pool(depth_sum_nhwc)  # Shape: (N, H_out, W_out, C)

            # Multiply by 2D kernel area to get sum (depth already summed)
            kernel_area_2d = kH * kW
            pooled = pooled * kernel_area_2d

            # Convert back to NCHW
            pooled = mx.transpose(pooled, [0, 3, 1, 2])  # Shape: (N, C, H_out, W_out)

            output_slices.append(mx.expand_dims(pooled, axis=2))

        # Stack along depth dimension
        output = mx.concatenate(output_slices, axis=2)  # Shape: (N, C, D_out, H_out, W_out)

        # Take p-th root
        result = mx.power(output, 1.0 / self.norm_type)

        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}"


class FractionalMaxPool2d(Module):
    """
    Applies 2D fractional max pooling over an input signal.

    Fractional max pooling randomly selects pooling regions, providing a form
    of stochastic pooling that can improve generalization.

    Args:
        kernel_size: Size of the pooling region
        output_size: Target output size (H_out, W_out)
        output_ratio: Target output ratio relative to input
        return_indices: Whether to return indices (default: False)
        _random_samples: Optional random samples for reproducibility

    Note:
        Either output_size or output_ratio must be specified, but not both.

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        output_size: Union[int, Tuple[int, int], None] = None,
        output_ratio: Union[float, Tuple[float, float], None] = None,
        return_indices: bool = False,
        _random_samples=None,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.output_size = _pair(output_size) if output_size is not None else None
        self.output_ratio = _pair(output_ratio) if output_ratio is not None else None
        self.return_indices = return_indices
        self._random_samples = _random_samples

        if output_size is None and output_ratio is None:
            raise ValueError("Either output_size or output_ratio must be defined")
        if output_size is not None and output_ratio is not None:
            raise ValueError("Only one of output_size or output_ratio should be defined")

    def forward(self, input: Tensor):
        """Apply fractional max pooling."""
        x = input._mlx_array
        N, C, H, W = x.shape
        kH, kW = self.kernel_size

        # Determine output size
        if self.output_size is not None:
            H_out, W_out = self.output_size
        else:
            H_out = int(H * self.output_ratio[0])
            W_out = int(W * self.output_ratio[1])

        # Generate random pooling regions using MLX random
        # Use uniform distribution and sort for region boundaries
        if H_out > 1 and H - kH > 0:
            h_random = mx.random.uniform(shape=(H_out - 1,))
            h_boundaries = mx.sort(h_random * (H - kH))
            h_regions = mx.concatenate(
                [mx.array([0.0]), h_boundaries, mx.array([float(H - kH + 1)])]
            )
        else:
            h_regions = mx.array([0.0, float(H - kH + 1)])

        if W_out > 1 and W - kW > 0:
            w_random = mx.random.uniform(shape=(W_out - 1,))
            w_boundaries = mx.sort(w_random * (W - kW))
            w_regions = mx.concatenate(
                [mx.array([0.0]), w_boundaries, mx.array([float(W - kW + 1)])]
            )
        else:
            w_regions = mx.array([0.0, float(W - kW + 1)])

        # Convert regions to Python for indexing
        h_regions_list = [int(v) for v in h_regions.tolist()]
        w_regions_list = [int(v) for v in w_regions.tolist()]

        # Build output using MLX operations
        output_rows = []
        indices_rows = [] if self.return_indices else None

        for i in range(H_out):
            output_cols = []
            indices_cols = [] if self.return_indices else None

            h_start = h_regions_list[i]
            h_end = min(h_regions_list[i + 1] + kH - 1, H)

            for j in range(W_out):
                w_start = w_regions_list[j]
                w_end = min(w_regions_list[j + 1] + kW - 1, W)

                window = x[:, :, h_start:h_end, w_start:w_end]
                # Max over spatial dimensions
                max_val = mx.max(window, axis=(2, 3), keepdims=True)
                output_cols.append(max_val)

                if self.return_indices:
                    # Flatten window and find argmax
                    window_flat = mx.reshape(window, (N, C, -1))
                    flat_idx = mx.argmax(window_flat, axis=2, keepdims=True)
                    # Convert to global index
                    global_idx = flat_idx + h_start * W + w_start
                    indices_cols.append(global_idx)

            # Concatenate columns
            row_output = mx.concatenate(output_cols, axis=3)
            output_rows.append(row_output)
            if self.return_indices:
                row_indices = mx.concatenate(indices_cols, axis=2)
                indices_rows.append(mx.expand_dims(row_indices, axis=2))

        # Concatenate rows
        output = mx.concatenate(output_rows, axis=2)
        result = Tensor._from_mlx_array(output)

        if self.return_indices:
            indices = mx.concatenate(indices_rows, axis=2)
            idx_tensor = Tensor._from_mlx_array(indices)
            return result, idx_tensor
        return result

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, output_size={self.output_size}, output_ratio={self.output_ratio}"


class FractionalMaxPool3d(Module):
    """
    Applies 3D fractional max pooling over an input signal.

    Args:
        kernel_size: Size of the pooling region
        output_size: Target output size (D_out, H_out, W_out)
        output_ratio: Target output ratio relative to input
        return_indices: Whether to return indices (default: False)
        _random_samples: Optional random samples for reproducibility

    Shape:
        - Input: (N, C, D_in, H_in, W_in)
        - Output: (N, C, D_out, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        output_size: Union[int, Tuple[int, int, int], None] = None,
        output_ratio: Union[float, Tuple[float, float, float], None] = None,
        return_indices: bool = False,
        _random_samples=None,
    ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.output_size = _triple(output_size) if output_size is not None else None
        self.output_ratio = _triple(output_ratio) if output_ratio is not None else None
        self.return_indices = return_indices
        self._random_samples = _random_samples

        if output_size is None and output_ratio is None:
            raise ValueError("Either output_size or output_ratio must be defined")
        if output_size is not None and output_ratio is not None:
            raise ValueError("Only one of output_size or output_ratio should be defined")

    def forward(self, input: Tensor):
        """Apply fractional max pooling 3D."""
        x = input._mlx_array
        N, C, D, H, W = x.shape
        kD, kH, kW = self.kernel_size

        if self.output_size is not None:
            D_out, H_out, W_out = self.output_size
        else:
            D_out = int(D * self.output_ratio[0])
            H_out = int(H * self.output_ratio[1])
            W_out = int(W * self.output_ratio[2])

        # Compute uniform grid step sizes
        d_step = D / D_out
        h_step = H / H_out
        w_step = W / W_out

        # Build output using MLX operations slice by slice
        output_depth = []
        for di in range(D_out):
            output_height = []
            d_start = int(di * d_step)
            d_end = min(int((di + 1) * d_step) + kD - 1, D)

            for hi in range(H_out):
                output_width = []
                h_start = int(hi * h_step)
                h_end = min(int((hi + 1) * h_step) + kH - 1, H)

                for wi in range(W_out):
                    w_start = int(wi * w_step)
                    w_end = min(int((wi + 1) * w_step) + kW - 1, W)

                    window = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    # Max over all spatial dimensions, keep dims for concatenation
                    max_val = mx.max(window, axis=(2, 3, 4), keepdims=True)
                    output_width.append(max_val)

                # Concatenate along W dimension
                row_output = mx.concatenate(output_width, axis=4)
                output_height.append(row_output)

            # Concatenate along H dimension
            height_output = mx.concatenate(output_height, axis=3)
            output_depth.append(height_output)

        # Concatenate along D dimension
        output = mx.concatenate(output_depth, axis=2)

        return Tensor._from_mlx_array(output)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, output_size={self.output_size}, output_ratio={self.output_ratio}"


class MaxUnpool1d(Module):
    """
    Computes a partial inverse of MaxPool1d.

    MaxPool1d is not fully invertible, since the non-maximal values are lost.
    MaxUnpool1d takes the output of MaxPool1d including the indices of the
    maximal values and computes a partial inverse.

    Args:
        kernel_size: Size of max pooling window
        stride: Stride of the max pooling window (default: kernel_size)
        padding: Padding that was added to the input

    Shape:
        - Input: (N, C, L_in)
        - Output: (N, C, L_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], None] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        self.padding = _single(padding)

    def forward(self, input: Tensor, indices: Tensor, output_size=None):
        """Apply max unpooling."""
        x = input._mlx_array
        idx = indices._mlx_array.astype(mx.int32)
        N, C, L_in = x.shape

        if output_size is not None:
            L_out = output_size[-1]
        else:
            L_out = (L_in - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Use one-hot scatter approach
        # Reshape to (N*C, L_in)
        x_flat = mx.reshape(x, (N * C, L_in))
        idx_flat = mx.reshape(idx, (N * C, L_in))

        # For each position in the flattened input, create one-hot vector
        # and multiply by value, then sum
        # one_hot shape: (N*C, L_in, L_out)
        # We can use broadcasting: indices (N*C, L_in, 1) == arange (L_out,)
        arange = mx.arange(L_out)  # (L_out,)
        idx_expanded = mx.expand_dims(idx_flat, axis=2)  # (N*C, L_in, 1)

        # Create mask where each position matches its target index
        one_hot = (idx_expanded == arange).astype(x.dtype)  # (N*C, L_in, L_out)

        # Multiply by values and sum over L_in dimension
        x_expanded = mx.expand_dims(x_flat, axis=2)  # (N*C, L_in, 1)
        scattered = one_hot * x_expanded  # (N*C, L_in, L_out)
        output_flat = mx.sum(scattered, axis=1)  # (N*C, L_out)

        # Reshape back to (N, C, L_out)
        output = mx.reshape(output_flat, (N, C, L_out))

        return Tensor._from_mlx_array(output)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class MaxUnpool2d(Module):
    """
    Computes a partial inverse of MaxPool2d.

    Args:
        kernel_size: Size of max pooling window
        stride: Stride of the max pooling window (default: kernel_size)
        padding: Padding that was added to the input

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)

    def forward(self, input: Tensor, indices: Tensor, output_size=None):
        """Apply max unpooling 2D."""
        x = input._mlx_array
        idx = indices._mlx_array.astype(mx.int32)
        N, C, H_in, W_in = x.shape

        if output_size is not None:
            H_out, W_out = output_size[-2], output_size[-1]
        else:
            H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        spatial_out = H_out * W_out
        spatial_in = H_in * W_in

        # Flatten spatial dimensions: (N, C, H_in, W_in) -> (N*C, H_in*W_in)
        x_flat = mx.reshape(x, (N * C, spatial_in))
        idx_flat = mx.reshape(idx, (N * C, spatial_in))

        # Use one-hot scatter approach
        arange = mx.arange(spatial_out)  # (H_out * W_out,)
        idx_expanded = mx.expand_dims(idx_flat, axis=2)  # (N*C, spatial_in, 1)

        # Create mask where each position matches its target index
        one_hot = (idx_expanded == arange).astype(x.dtype)  # (N*C, spatial_in, spatial_out)

        # Multiply by values and sum over spatial_in dimension
        x_expanded = mx.expand_dims(x_flat, axis=2)  # (N*C, spatial_in, 1)
        scattered = one_hot * x_expanded  # (N*C, spatial_in, spatial_out)
        output_flat = mx.sum(scattered, axis=1)  # (N*C, spatial_out)

        # Reshape back to (N, C, H_out, W_out)
        output = mx.reshape(output_flat, (N, C, H_out, W_out))

        return Tensor._from_mlx_array(output)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class MaxUnpool3d(Module):
    """
    Computes a partial inverse of MaxPool3d.

    Args:
        kernel_size: Size of max pooling window
        stride: Stride of the max pooling window (default: kernel_size)
        padding: Padding that was added to the input

    Shape:
        - Input: (N, C, D_in, H_in, W_in)
        - Output: (N, C, D_out, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], None] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) if stride is not None else self.kernel_size
        self.padding = _triple(padding)

    def forward(self, input: Tensor, indices: Tensor, output_size=None):
        """Apply max unpooling 3D."""
        x = input._mlx_array
        idx = indices._mlx_array.astype(mx.int32)
        N, C, D_in, H_in, W_in = x.shape

        if output_size is not None:
            D_out, H_out, W_out = output_size[-3], output_size[-2], output_size[-1]
        else:
            D_out = (D_in - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
            H_out = (H_in - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
            W_out = (W_in - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]

        spatial_out = D_out * H_out * W_out
        spatial_in = D_in * H_in * W_in

        # Flatten spatial dimensions: (N, C, D_in, H_in, W_in) -> (N*C, spatial_in)
        x_flat = mx.reshape(x, (N * C, spatial_in))
        idx_flat = mx.reshape(idx, (N * C, spatial_in))

        # Use one-hot scatter approach
        arange = mx.arange(spatial_out)  # (D_out * H_out * W_out,)
        idx_expanded = mx.expand_dims(idx_flat, axis=2)  # (N*C, spatial_in, 1)

        # Create mask where each position matches its target index
        one_hot = (idx_expanded == arange).astype(x.dtype)  # (N*C, spatial_in, spatial_out)

        # Multiply by values and sum over spatial_in dimension
        x_expanded = mx.expand_dims(x_flat, axis=2)  # (N*C, spatial_in, 1)
        scattered = one_hot * x_expanded  # (N*C, spatial_in, spatial_out)
        output_flat = mx.sum(scattered, axis=1)  # (N*C, spatial_out)

        # Reshape back to (N, C, D_out, H_out, W_out)
        output = mx.reshape(output_flat, (N, C, D_out, H_out, W_out))

        return Tensor._from_mlx_array(output)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


__all__ = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
]
