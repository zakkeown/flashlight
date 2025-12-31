"""
Dropout Layer

Implements dropout regularization.
"""

import mlx.core as mx
from ..module import Module
from ...tensor import Tensor


class Dropout(Module):
    """
    During training, randomly zeroes some elements of the input tensor
    with probability p using samples from a Bernoulli distribution.

    Args:
        p: Probability of an element to be zeroed (default: 0.5)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*) any shape
        - Output: (*) same shape as input

    Example:
        >>> m = nn.Dropout(p=0.2)
        >>> x = mlx_compat.randn(20, 16)
        >>> output = m(x)
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply dropout.

        During training, randomly zero elements and scale by 1/(1-p).
        During evaluation, return input unchanged.
        """
        if not self.training or self.p == 0:
            return input

        # Generate dropout mask
        # Bernoulli(1-p): probability (1-p) of keeping
        keep_prob = 1 - self.p
        mask_array = mx.random.bernoulli(keep_prob, input.shape)

        # Apply mask and scale
        # Scale by 1/(1-p) to maintain expected value
        output_array = (input._mlx_array * mask_array) / keep_prob

        result = Tensor._from_mlx_array(output_array)

        # Preserve gradient tracking
        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True
            # Note: Dropout gradient is the same mask scaled by 1/(1-p)
            # This is handled correctly by the multiplication autograd

        return result

    def extra_repr(self) -> str:
        return f'p={self.p}'


class Dropout1d(Module):
    """
    Randomly zero out entire channels (1D feature maps).

    Args:
        p: Probability of a channel to be zeroed (default: 0.5)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (N, C, L) or (C, L)
        - Output: Same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input

        # Get shape info
        shape = input.shape
        if len(shape) == 2:
            # (C, L) -> mask per channel
            mask_shape = (shape[0], 1)
        else:
            # (N, C, L) -> mask per channel per batch
            mask_shape = (shape[0], shape[1], 1)

        keep_prob = 1 - self.p
        mask_array = mx.random.bernoulli(keep_prob, mask_shape)
        mask_array = mx.broadcast_to(mask_array, shape)
        output_array = (input._mlx_array * mask_array) / keep_prob

        result = Tensor._from_mlx_array(output_array)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return f'p={self.p}'


class Dropout2d(Module):
    """
    Randomly zero out entire channels (2D feature maps).

    Args:
        p: Probability of a channel to be zeroed (default: 0.5)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (N, C, H, W) or (C, H, W)
        - Output: Same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input

        shape = input.shape
        if len(shape) == 3:
            # (C, H, W) -> mask per channel
            mask_shape = (shape[0], 1, 1)
        else:
            # (N, C, H, W) -> mask per channel per batch
            mask_shape = (shape[0], shape[1], 1, 1)

        keep_prob = 1 - self.p
        mask_array = mx.random.bernoulli(keep_prob, mask_shape)
        mask_array = mx.broadcast_to(mask_array, shape)
        output_array = (input._mlx_array * mask_array) / keep_prob

        result = Tensor._from_mlx_array(output_array)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return f'p={self.p}'


class Dropout3d(Module):
    """
    Randomly zero out entire channels (3D feature maps).

    Args:
        p: Probability of a channel to be zeroed (default: 0.5)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (N, C, D, H, W) or (C, D, H, W)
        - Output: Same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input

        shape = input.shape
        if len(shape) == 4:
            # (C, D, H, W) -> mask per channel
            mask_shape = (shape[0], 1, 1, 1)
        else:
            # (N, C, D, H, W) -> mask per channel per batch
            mask_shape = (shape[0], shape[1], 1, 1, 1)

        keep_prob = 1 - self.p
        mask_array = mx.random.bernoulli(keep_prob, mask_shape)
        mask_array = mx.broadcast_to(mask_array, shape)
        output_array = (input._mlx_array * mask_array) / keep_prob

        result = Tensor._from_mlx_array(output_array)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return f'p={self.p}'


class AlphaDropout(Module):
    """
    Applies Alpha Dropout over the input.

    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property of inputs. For SELU activation functions.

    Args:
        p: Probability of an element to be dropped (default: 0.5)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (*) any shape
        - Output: (*) same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input

        # Alpha dropout parameters for SELU
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale

        keep_prob = 1 - self.p

        # Compute affine transformation parameters
        a = ((1 - self.p) * (1 + self.p * alpha_p ** 2)) ** -0.5
        b = -a * alpha_p * self.p

        # Create mask
        mask = mx.random.bernoulli(keep_prob, input.shape)

        # Apply alpha dropout transformation
        x = input._mlx_array
        output_array = mask * x + (1 - mask) * alpha_p
        output_array = a * output_array + b

        result = Tensor._from_mlx_array(output_array)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return f'p={self.p}'


class FeatureAlphaDropout(Module):
    """
    Randomly masks out entire channels with Alpha Dropout.

    Args:
        p: Probability of a channel to be zeroed (default: 0.5)
        inplace: Unused (for PyTorch compatibility)

    Shape:
        - Input: (N, C, *) where * means any number of additional dimensions
        - Output: Same shape as input
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input

        shape = input.shape
        # Mask shape: one value per channel
        mask_shape = list(shape)
        for i in range(2, len(shape)):
            mask_shape[i] = 1
        mask_shape = tuple(mask_shape)

        # Alpha dropout parameters
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale

        keep_prob = 1 - self.p

        a = ((1 - self.p) * (1 + self.p * alpha_p ** 2)) ** -0.5
        b = -a * alpha_p * self.p

        mask = mx.random.bernoulli(keep_prob, mask_shape)
        mask = mx.broadcast_to(mask, shape)

        x = input._mlx_array
        output_array = mask * x + (1 - mask) * alpha_p
        output_array = a * output_array + b

        result = Tensor._from_mlx_array(output_array)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return f'p={self.p}'


__all__ = ['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout']
