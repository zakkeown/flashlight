"""
Upsampling and Pixel Shuffle Layers

Implements nn.Upsample, nn.UpsamplingNearest2d, nn.UpsamplingBilinear2d,
nn.PixelShuffle, and nn.PixelUnshuffle.
"""

from typing import Any, Optional, Tuple, Union

from ...tensor import Tensor
from ..functional import interpolate, pixel_shuffle, pixel_unshuffle
from ..module import Module


class Upsample(Module):
    """
    Upsamples a given multi-channel input.

    The input is expected to be of shape (N, C, ...) where '...' denotes
    additional spatial dimensions.

    Args:
        size: Output spatial sizes
        scale_factor: Multiplier for spatial size
        mode: Upsampling algorithm: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
        align_corners: If True, align corner pixels of input and output tensors
        recompute_scale_factor: Recompute scale factor for use in interpolation

    Shape:
        - Input: (N, C, ...) where ... is spatial dimensions
        - Output: (N, C, ...) with upsampled spatial dimensions
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input: Tensor) -> Tensor:
        return interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    def extra_repr(self) -> str:
        if self.size is not None:
            return f"size={self.size}, mode={self.mode}"
        return f"scale_factor={self.scale_factor}, mode={self.mode}"


class UpsamplingNearest2d(Module):
    """
    Apply 2D nearest neighbor upsampling.

    Args:
        size: Output spatial sizes (H_out, W_out)
        scale_factor: Multiplier for spatial size

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, input: Tensor) -> Tensor:
        return interpolate(input, size=self.size, scale_factor=self.scale_factor, mode="nearest")

    def extra_repr(self) -> str:
        if self.size is not None:
            return f"size={self.size}"
        return f"scale_factor={self.scale_factor}"


class UpsamplingBilinear2d(Module):
    """
    Apply 2D bilinear upsampling.

    Args:
        size: Output spatial sizes (H_out, W_out)
        scale_factor: Multiplier for spatial size

    Shape:
        - Input: (N, C, H_in, W_in)
        - Output: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, input: Tensor) -> Tensor:
        # PyTorch's UpsamplingBilinear2d uses align_corners=True by default
        return interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True,
        )

    def extra_repr(self) -> str:
        if self.size is not None:
            return f"size={self.size}"
        return f"scale_factor={self.scale_factor}"


class PixelShuffle(Module):
    """
    Rearrange elements from depth to space (sub-pixel convolution).

    Rearranges elements in a tensor of shape (*, C*r^2, H, W) to a tensor
    of shape (*, C, H*r, W*r) where r is the upscale_factor.

    This is useful for implementing efficient sub-pixel convolution with
    a stride of 1/r.

    Args:
        upscale_factor: Factor to increase spatial resolution by

    Shape:
        - Input: (N, C*upscale_factor^2, H, W)
        - Output: (N, C, H*upscale_factor, W*upscale_factor)

    Example:
        >>> ps = nn.PixelShuffle(2)
        >>> x = torch.randn(1, 8, 4, 4)  # 8 = 2 * 2^2
        >>> ps(x).shape
        torch.Size([1, 2, 8, 8])
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self) -> str:
        return f"upscale_factor={self.upscale_factor}"


class PixelUnshuffle(Module):
    """
    Rearrange elements from space to depth (inverse of PixelShuffle).

    Rearranges elements in a tensor of shape (*, C, H*r, W*r) to a tensor
    of shape (*, C*r^2, H, W) where r is the downscale_factor.

    Args:
        downscale_factor: Factor to decrease spatial resolution by

    Shape:
        - Input: (N, C, H*downscale_factor, W*downscale_factor)
        - Output: (N, C*downscale_factor^2, H, W)

    Example:
        >>> pu = nn.PixelUnshuffle(2)
        >>> x = torch.randn(1, 2, 8, 8)
        >>> pu(x).shape
        torch.Size([1, 8, 4, 4])
    """

    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return f"downscale_factor={self.downscale_factor}"


__all__ = [
    "Upsample",
    "UpsamplingNearest2d",
    "UpsamplingBilinear2d",
    "PixelShuffle",
    "PixelUnshuffle",
]
