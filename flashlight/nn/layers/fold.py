"""
Fold and Unfold Layers

Implements nn.Fold and nn.Unfold for extracting and combining sliding local blocks.
"""

from typing import Union, Tuple

from ..module import Module
from ...tensor import Tensor
from ..functional import fold, unfold


class Unfold(Module):
    """
    Extract sliding local blocks from a batched input tensor.

    This layer extracts patches from the input tensor using a sliding window
    and returns them as columns in the output tensor. Also known as im2col.

    Args:
        kernel_size: Size of the sliding blocks (kH, kW)
        dilation: Dilation of the sliding blocks (default: 1)
        padding: Padding applied to input (default: 0)
        stride: Stride of the sliding blocks (default: 1)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C * kH * kW, L)
          where L = ((H + 2*padH - dilH*(kH-1) - 1) / strH + 1) *
                    ((W + 2*padW - dilW*(kW-1) - 1) / strW + 1)

    Example:
        >>> unfold = nn.Unfold(kernel_size=(3, 3))
        >>> x = torch.randn(1, 3, 10, 12)
        >>> output = unfold(x)
        >>> output.shape
        torch.Size([1, 27, 80])
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

    def extra_repr(self) -> str:
        return (
            f'kernel_size={self.kernel_size}, dilation={self.dilation}, '
            f'padding={self.padding}, stride={self.stride}'
        )


class Fold(Module):
    """
    Combine an array of sliding local blocks into a large containing tensor.

    This layer is the inverse operation of Unfold. It sums overlapping values.
    Also known as col2im.

    Args:
        output_size: Shape of the spatial dimensions of the output (H, W)
        kernel_size: Size of the sliding blocks (kH, kW)
        dilation: Dilation of the sliding blocks (default: 1)
        padding: Padding applied to original input (default: 0)
        stride: Stride of the sliding blocks (default: 1)

    Shape:
        - Input: (N, C * kH * kW, L)
        - Output: (N, C, H, W)

    Example:
        >>> fold = nn.Fold(output_size=(10, 12), kernel_size=(3, 3))
        >>> x = torch.randn(1, 27, 80)  # 27 = 3 * 3 * 3 (C * kH * kW)
        >>> output = fold(x)
        >>> output.shape
        torch.Size([1, 3, 10, 12])
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1
    ):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return fold(
            input,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

    def extra_repr(self) -> str:
        return (
            f'output_size={self.output_size}, kernel_size={self.kernel_size}, '
            f'dilation={self.dilation}, padding={self.padding}, stride={self.stride}'
        )


__all__ = ['Fold', 'Unfold']
