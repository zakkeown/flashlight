"""
Padding Layers

Implements padding layers for neural networks.
"""

import mlx.core as mx
from ..module import Module
from ...tensor import Tensor
from typing import Union, Tuple


def _flip_axis(arr, axis):
    """
    Flip array along a specific axis using slicing.

    MLX doesn't have a flip function, so we implement it using slicing [::-1].
    """
    ndim = len(arr.shape)
    axis = axis if axis >= 0 else ndim + axis
    slices = [slice(None)] * ndim
    slices[axis] = slice(None, None, -1)
    return arr[tuple(slices)]


class ZeroPad1d(Module):
    """
    Pads the input tensor with zeros on both sides in 1D.

    Args:
        padding: Size of padding (left, right) or single int for both sides

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L + padding_left + padding_right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply zero padding."""
        x = input._mlx_array
        # Pad format: [(before, after) for each dimension]
        pad_config = [(0, 0), (0, 0), self.padding]
        result = mx.pad(x, pad_config, constant_values=0)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ZeroPad2d(Module):
    """
    Pads the input tensor with zeros on all sides in 2D.

    Args:
        padding: Size of padding (left, right, top, bottom) or single int for all sides

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply zero padding."""
        x = input._mlx_array
        left, right, top, bottom = self.padding
        # Pad format: [(before, after) for each dimension]
        pad_config = [(0, 0), (0, 0), (top, bottom), (left, right)]
        result = mx.pad(x, pad_config, constant_values=0)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ZeroPad3d(Module):
    """
    Pads the input tensor with zeros on all sides in 3D.

    Args:
        padding: Size of padding (left, right, top, bottom, front, back) or single int

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D + front + back, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding,) * 6
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply zero padding."""
        x = input._mlx_array
        left, right, top, bottom, front, back = self.padding
        # Pad format: [(before, after) for each dimension]
        pad_config = [(0, 0), (0, 0), (front, back), (top, bottom), (left, right)]
        result = mx.pad(x, pad_config, constant_values=0)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ConstantPad1d(Module):
    """
    Pads the input tensor with a constant value in 1D.

    Args:
        padding: Size of padding (left, right) or single int for both sides
        value: Constant value for padding

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L + padding_left + padding_right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int]], value: float):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        """Apply constant padding."""
        x = input._mlx_array
        pad_config = [(0, 0), (0, 0), self.padding]
        result = mx.pad(x, pad_config, constant_values=self.value)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'padding={self.padding}, value={self.value}'


class ConstantPad2d(Module):
    """
    Pads the input tensor with a constant value in 2D.

    Args:
        padding: Size of padding (left, right, top, bottom) or single int for all sides
        value: Constant value for padding

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]], value: float):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = tuple(padding)
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        """Apply constant padding."""
        x = input._mlx_array
        left, right, top, bottom = self.padding
        pad_config = [(0, 0), (0, 0), (top, bottom), (left, right)]
        result = mx.pad(x, pad_config, constant_values=self.value)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'padding={self.padding}, value={self.value}'


class ConstantPad3d(Module):
    """
    Pads the input tensor with a constant value in 3D.

    Args:
        padding: Size of padding (left, right, top, bottom, front, back) or single int
        value: Constant value for padding

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D + front + back, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]], value: float):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding,) * 6
        else:
            self.padding = tuple(padding)
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        """Apply constant padding."""
        x = input._mlx_array
        left, right, top, bottom, front, back = self.padding
        pad_config = [(0, 0), (0, 0), (front, back), (top, bottom), (left, right)]
        result = mx.pad(x, pad_config, constant_values=self.value)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'padding={self.padding}, value={self.value}'


class ReflectionPad1d(Module):
    """
    Pads the input tensor using reflection of the input boundary.

    Args:
        padding: Size of padding (left, right) or single int for both sides

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L + padding_left + padding_right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply reflection padding."""
        x = input._mlx_array
        left, right = self.padding

        # Build padded tensor manually for reflection
        parts = []
        if left > 0:
            # Reflect left: x[:, :, 1:left+1] reversed
            left_pad = _flip_axis(x[:, :, 1:left + 1], axis=2)
            parts.append(left_pad)
        parts.append(x)
        if right > 0:
            # Reflect right: x[:, :, -(right+1):-1] reversed
            right_pad = _flip_axis(x[:, :, -(right + 1):-1], axis=2)
            parts.append(right_pad)

        result = mx.concatenate(parts, axis=2)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReflectionPad2d(Module):
    """
    Pads the input tensor using reflection of the input boundary.

    Args:
        padding: Size of padding (left, right, top, bottom) or single int for all sides

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply reflection padding."""
        x = input._mlx_array
        left, right, top, bottom = self.padding

        # Pad width first
        if left > 0 or right > 0:
            parts_w = []
            if left > 0:
                left_pad = _flip_axis(x[:, :, :, 1:left + 1], axis=3)
                parts_w.append(left_pad)
            parts_w.append(x)
            if right > 0:
                right_pad = _flip_axis(x[:, :, :, -(right + 1):-1], axis=3)
                parts_w.append(right_pad)
            x = mx.concatenate(parts_w, axis=3)

        # Then pad height
        if top > 0 or bottom > 0:
            parts_h = []
            if top > 0:
                top_pad = _flip_axis(x[:, :, 1:top + 1, :], axis=2)
                parts_h.append(top_pad)
            parts_h.append(x)
            if bottom > 0:
                bottom_pad = _flip_axis(x[:, :, -(bottom + 1):-1, :], axis=2)
                parts_h.append(bottom_pad)
            x = mx.concatenate(parts_h, axis=2)

        return Tensor._from_mlx_array(x)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReplicationPad1d(Module):
    """
    Pads the input tensor using replication of the input boundary.

    Args:
        padding: Size of padding (left, right) or single int for both sides

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L + padding_left + padding_right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply replication padding."""
        x = input._mlx_array
        left, right = self.padding

        parts = []
        if left > 0:
            # Replicate first element
            left_pad = mx.broadcast_to(x[:, :, :1], (x.shape[0], x.shape[1], left))
            parts.append(left_pad)
        parts.append(x)
        if right > 0:
            # Replicate last element
            right_pad = mx.broadcast_to(x[:, :, -1:], (x.shape[0], x.shape[1], right))
            parts.append(right_pad)

        result = mx.concatenate(parts, axis=2)
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReplicationPad2d(Module):
    """
    Pads the input tensor using replication of the input boundary.

    Args:
        padding: Size of padding (left, right, top, bottom) or single int for all sides

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply replication padding."""
        x = input._mlx_array
        left, right, top, bottom = self.padding
        N, C, H, W = x.shape

        # Pad width first
        if left > 0 or right > 0:
            parts_w = []
            if left > 0:
                left_pad = mx.broadcast_to(x[:, :, :, :1], (N, C, H, left))
                parts_w.append(left_pad)
            parts_w.append(x)
            if right > 0:
                right_pad = mx.broadcast_to(x[:, :, :, -1:], (N, C, H, right))
                parts_w.append(right_pad)
            x = mx.concatenate(parts_w, axis=3)

        # Update W after width padding
        W_new = x.shape[3]

        # Then pad height
        if top > 0 or bottom > 0:
            parts_h = []
            if top > 0:
                top_pad = mx.broadcast_to(x[:, :, :1, :], (N, C, top, W_new))
                parts_h.append(top_pad)
            parts_h.append(x)
            if bottom > 0:
                bottom_pad = mx.broadcast_to(x[:, :, -1:, :], (N, C, bottom, W_new))
                parts_h.append(bottom_pad)
            x = mx.concatenate(parts_h, axis=2)

        return Tensor._from_mlx_array(x)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReplicationPad3d(Module):
    """
    Pads the input tensor using replication of the input boundary.

    Args:
        padding: Size of padding (left, right, top, bottom, front, back) or single int

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D + front + back, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding,) * 6
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply replication padding."""
        x = input._mlx_array
        left, right, top, bottom, front, back = self.padding
        N, C, D, H, W = x.shape

        # Pad width first
        if left > 0 or right > 0:
            parts = []
            if left > 0:
                parts.append(mx.broadcast_to(x[:, :, :, :, :1], (N, C, D, H, left)))
            parts.append(x)
            if right > 0:
                parts.append(mx.broadcast_to(x[:, :, :, :, -1:], (N, C, D, H, right)))
            x = mx.concatenate(parts, axis=4)
            W = x.shape[4]

        # Pad height
        if top > 0 or bottom > 0:
            parts = []
            if top > 0:
                parts.append(mx.broadcast_to(x[:, :, :, :1, :], (N, C, D, top, x.shape[4])))
            parts.append(x)
            if bottom > 0:
                parts.append(mx.broadcast_to(x[:, :, :, -1:, :], (N, C, D, bottom, x.shape[4])))
            x = mx.concatenate(parts, axis=3)

        # Pad depth
        if front > 0 or back > 0:
            parts = []
            if front > 0:
                parts.append(mx.broadcast_to(x[:, :, :1, :, :], (N, C, front, x.shape[3], x.shape[4])))
            parts.append(x)
            if back > 0:
                parts.append(mx.broadcast_to(x[:, :, -1:, :, :], (N, C, back, x.shape[3], x.shape[4])))
            x = mx.concatenate(parts, axis=2)

        return Tensor._from_mlx_array(x)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class CircularPad1d(Module):
    """
    Pads the input tensor using circular padding of the input boundary.

    Args:
        padding: Size of padding (left, right) or single int for both sides

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply circular padding."""
        x = input._mlx_array
        left, right = self.padding

        parts = []
        if left > 0:
            parts.append(x[:, :, -left:])
        parts.append(x)
        if right > 0:
            parts.append(x[:, :, :right])

        result = mx.concatenate(parts, axis=2) if len(parts) > 1 else x
        return Tensor._from_mlx_array(result)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class CircularPad2d(Module):
    """
    Pads the input tensor using circular padding of the input boundary.

    Args:
        padding: Size of padding (left, right, top, bottom) or single int

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply circular padding."""
        x = input._mlx_array
        left, right, top, bottom = self.padding

        # Pad width first
        if left > 0 or right > 0:
            parts = []
            if left > 0:
                parts.append(x[:, :, :, -left:])
            parts.append(x)
            if right > 0:
                parts.append(x[:, :, :, :right])
            x = mx.concatenate(parts, axis=3)

        # Then pad height
        if top > 0 or bottom > 0:
            parts = []
            if top > 0:
                parts.append(x[:, :, -top:, :])
            parts.append(x)
            if bottom > 0:
                parts.append(x[:, :, :bottom, :])
            x = mx.concatenate(parts, axis=2)

        return Tensor._from_mlx_array(x)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class CircularPad3d(Module):
    """
    Pads the input tensor using circular padding of the input boundary.

    Args:
        padding: Size of padding (left, right, top, bottom, front, back) or single int

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D + front + back, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding,) * 6
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply circular padding."""
        x = input._mlx_array
        left, right, top, bottom, front, back = self.padding

        # Pad width first
        if left > 0 or right > 0:
            parts = []
            if left > 0:
                parts.append(x[:, :, :, :, -left:])
            parts.append(x)
            if right > 0:
                parts.append(x[:, :, :, :, :right])
            x = mx.concatenate(parts, axis=4)

        # Pad height
        if top > 0 or bottom > 0:
            parts = []
            if top > 0:
                parts.append(x[:, :, :, -top:, :])
            parts.append(x)
            if bottom > 0:
                parts.append(x[:, :, :, :bottom, :])
            x = mx.concatenate(parts, axis=3)

        # Pad depth
        if front > 0 or back > 0:
            parts = []
            if front > 0:
                parts.append(x[:, :, -front:, :, :])
            parts.append(x)
            if back > 0:
                parts.append(x[:, :, :back, :, :])
            x = mx.concatenate(parts, axis=2)

        return Tensor._from_mlx_array(x)

    def extra_repr(self) -> str:
        return f'{self.padding}'


class ReflectionPad3d(Module):
    """
    Pads the input tensor using reflection of the input boundary.

    Args:
        padding: Size of padding (left, right, top, bottom, front, back) or single int

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D + front + back, H + top + bottom, W + left + right]
    """

    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding,) * 6
        else:
            self.padding = tuple(padding)

    def forward(self, input: Tensor) -> Tensor:
        """Apply reflection padding."""
        import numpy as np

        x = np.array(input._mlx_array)
        left, right, top, bottom, front, back = self.padding

        # Use numpy pad with reflect mode
        pad_width = [(0, 0), (0, 0), (front, back), (top, bottom), (left, right)]
        result = np.pad(x, pad_width, mode='reflect')

        return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))

    def extra_repr(self) -> str:
        return f'{self.padding}'


__all__ = [
    'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
    'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d',
    'CircularPad1d', 'CircularPad2d', 'CircularPad3d',
]
