"""
Normalization Layers

Implements normalization layers for neural networks.
"""

import mlx.core as mx
from ..module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from typing import Optional, Any, Union, List


class BatchNorm2d(Module):
    """
    2D Batch Normalization.

    Applies Batch Normalization over a 4D input (N, C, H, W).

    Args:
        num_features: Number of features (channels)
        eps: Value added to denominator for numerical stability (default: 1e-5)
        momentum: Value used for running mean/var computation (default: 0.1)
        affine: Whether to learn affine parameters (default: True)
        track_running_stats: Whether to track running statistics (default: True)

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H, W]

    Example:
        >>> bn = nn.BatchNorm2d(64)
        >>> x = mlx_compat.randn(4, 64, 32, 32)
        >>> output = bn(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        # MLX uses unified memory and default dtype
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_features)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_features)))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            # These are not parameters (no gradients)
            self.running_mean = Tensor._from_mlx_array(mx.zeros(num_features))
            self.running_var = Tensor._from_mlx_array(mx.ones(num_features))
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        # Cache for reshaped weight/bias (avoid reshape on every forward)
        # Use a dict to store cache to avoid Module's __setattr__ intercepting Tensor values
        self._affine_cache = {}

    def _get_affine_params(self, is_nhwc: bool):
        """Get cached reshaped weight and bias for given layout."""
        if not self.affine:
            return None, None

        weight_id = id(self.weight._mlx_array)
        bias_id = id(self.bias._mlx_array)

        cache = self._affine_cache
        # Check if cache is valid
        if cache.get('weight_id') != weight_id or cache.get('bias_id') != bias_id:
            # Cache invalidated - recompute both layouts
            cache['weight_nchw'] = self.weight.reshape(1, -1, 1, 1)
            cache['weight_nhwc'] = self.weight.reshape(1, 1, 1, -1)
            cache['bias_nchw'] = self.bias.reshape(1, -1, 1, 1)
            cache['bias_nhwc'] = self.bias.reshape(1, 1, 1, -1)
            cache['weight_id'] = weight_id
            cache['bias_id'] = bias_id

        if is_nhwc:
            return cache['weight_nhwc'], cache['bias_nhwc']
        else:
            return cache['weight_nchw'], cache['bias_nchw']

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply batch normalization.

        Args:
            input: Input tensor of shape [N, C, H, W] (NCHW) or [N, H, W, C] (NHWC)

        Returns:
            Normalized tensor of same shape as input
        """
        from ...layout import is_nhwc_mode, Layout

        # Check if input is in NHWC layout
        is_nhwc = (
            is_nhwc_mode() and
            hasattr(input, '_layout') and
            input._layout == Layout.NHWC
        )

        if is_nhwc:
            # NHWC: [N, H, W, C] - normalize over N, H, W (dims 0, 1, 2)
            reduce_dims = [0, 1, 2]
            broadcast_shape = (1, 1, 1, -1)  # [1, 1, 1, C]
        else:
            # NCHW: [N, C, H, W] - normalize over N, H, W (dims 0, 2, 3)
            reduce_dims = [0, 2, 3]
            broadcast_shape = (1, -1, 1, 1)  # [1, C, 1, 1]

        if self.training:
            # Compute batch statistics
            batch_mean = input.mean(dim=reduce_dims, keepdim=False)  # [C]
            batch_var = input.var(dim=reduce_dims, keepdim=False, unbiased=False)  # [C]

            # Update running statistics
            if self.track_running_stats:
                momentum = self.momentum
                self.running_mean._mlx_array = (
                    (1 - momentum) * self.running_mean._mlx_array +
                    momentum * batch_mean._mlx_array
                )
                self.running_var._mlx_array = (
                    (1 - momentum) * self.running_var._mlx_array +
                    momentum * batch_var._mlx_array
                )
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                # If not tracking, compute batch stats even in eval mode
                mean = input.mean(dim=reduce_dims, keepdim=False)
                var = input.var(dim=reduce_dims, keepdim=False, unbiased=False)

        # Normalize: (x - mean) / sqrt(var + eps)
        # Reshape mean and var for broadcasting
        mean_reshaped = mean.reshape(*broadcast_shape)
        var_reshaped = var.reshape(*broadcast_shape)

        normalized = (input - mean_reshaped) / Tensor._from_mlx_array(
            mx.sqrt(var_reshaped._mlx_array + self.eps)
        )

        # Apply affine transformation using cached reshaped weight/bias
        if self.affine:
            weight_reshaped, bias_reshaped = self._get_affine_params(is_nhwc)
            output = normalized * weight_reshaped + bias_reshaped
        else:
            output = normalized

        # Preserve layout in output
        if is_nhwc:
            output._layout = Layout.NHWC

        return output

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )


class LayerNorm(Module):
    """
    Layer Normalization.

    Applies Layer Normalization over the last dimension(s).

    Args:
        normalized_shape: Input shape from an expected input
        eps: Value added to denominator for numerical stability (default: 1e-5)
        elementwise_affine: Whether to learn affine parameters (default: True)
        bias: Whether to include a bias term (default: True)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [..., normalized_shape]
        - Output: [..., normalized_shape]

    Example:
        >>> ln = nn.LayerNorm(128)
        >>> x = mlx_compat.randn(4, 10, 128)
        >>> output = ln(x)
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(normalized_shape)))
            if bias:
                self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(normalized_shape)))
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply layer normalization.

        Args:
            input: Input tensor

        Returns:
            Normalized tensor
        """
        # Compute mean and variance over the last dimension
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        normalized = (input - mean) / Tensor._from_mlx_array(
            mx.sqrt(var._mlx_array + self.eps)
        )

        # Apply affine transformation
        if self.elementwise_affine:
            output = normalized * self.weight + self.bias
        else:
            output = normalized

        return output

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class BatchNorm1d(Module):
    """
    1D Batch Normalization.

    Applies Batch Normalization over a 2D or 3D input (N, C) or (N, C, L).

    Args:
        num_features: Number of features (channels)
        eps: Value added to denominator for numerical stability (default: 1e-5)
        momentum: Value used for running mean/var computation (default: 0.1)
        affine: Whether to learn affine parameters (default: True)
        track_running_stats: Whether to track running statistics (default: True)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [N, C] or [N, C, L]
        - Output: Same as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_features)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_features)))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor._from_mlx_array(mx.zeros(num_features))
            self.running_var = Tensor._from_mlx_array(mx.ones(num_features))
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply batch normalization."""
        is_3d = input._mlx_array.ndim == 3

        if self.training:
            if is_3d:
                # [N, C, L] -> mean over N, L
                batch_mean = input.mean(dim=[0, 2], keepdim=False)
                batch_var = input.var(dim=[0, 2], keepdim=False, unbiased=False)
            else:
                # [N, C] -> mean over N
                batch_mean = input.mean(dim=0, keepdim=False)
                batch_var = input.var(dim=0, keepdim=False, unbiased=False)

            if self.track_running_stats:
                momentum = self.momentum
                self.running_mean._mlx_array = (
                    (1 - momentum) * self.running_mean._mlx_array +
                    momentum * batch_mean._mlx_array
                )
                self.running_var._mlx_array = (
                    (1 - momentum) * self.running_var._mlx_array +
                    momentum * batch_var._mlx_array
                )
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                if is_3d:
                    mean = input.mean(dim=[0, 2], keepdim=False)
                    var = input.var(dim=[0, 2], keepdim=False, unbiased=False)
                else:
                    mean = input.mean(dim=0, keepdim=False)
                    var = input.var(dim=0, keepdim=False, unbiased=False)

        # Reshape for broadcasting
        if is_3d:
            mean_reshaped = mean.reshape(1, -1, 1)
            var_reshaped = var.reshape(1, -1, 1)
        else:
            mean_reshaped = mean.reshape(1, -1)
            var_reshaped = var.reshape(1, -1)

        normalized = (input - mean_reshaped) / Tensor._from_mlx_array(
            mx.sqrt(var_reshaped._mlx_array + self.eps)
        )

        if self.affine:
            if is_3d:
                weight_reshaped = self.weight.reshape(1, -1, 1)
                bias_reshaped = self.bias.reshape(1, -1, 1)
            else:
                weight_reshaped = self.weight.reshape(1, -1)
                bias_reshaped = self.bias.reshape(1, -1)
            output = normalized * weight_reshaped + bias_reshaped
        else:
            output = normalized

        return output

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )


class BatchNorm3d(Module):
    """
    3D Batch Normalization.

    Applies Batch Normalization over a 5D input (N, C, D, H, W).

    Args:
        num_features: Number of features (channels)
        eps: Value added to denominator for numerical stability (default: 1e-5)
        momentum: Value used for running mean/var computation (default: 0.1)
        affine: Whether to learn affine parameters (default: True)
        track_running_stats: Whether to track running statistics (default: True)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D, H, W]
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_features)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_features)))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor._from_mlx_array(mx.zeros(num_features))
            self.running_var = Tensor._from_mlx_array(mx.ones(num_features))
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply batch normalization."""
        if self.training:
            # [N, C, D, H, W] -> mean over N, D, H, W
            batch_mean = input.mean(dim=[0, 2, 3, 4], keepdim=False)
            batch_var = input.var(dim=[0, 2, 3, 4], keepdim=False, unbiased=False)

            if self.track_running_stats:
                momentum = self.momentum
                self.running_mean._mlx_array = (
                    (1 - momentum) * self.running_mean._mlx_array +
                    momentum * batch_mean._mlx_array
                )
                self.running_var._mlx_array = (
                    (1 - momentum) * self.running_var._mlx_array +
                    momentum * batch_var._mlx_array
                )
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = input.mean(dim=[0, 2, 3, 4], keepdim=False)
                var = input.var(dim=[0, 2, 3, 4], keepdim=False, unbiased=False)

        # Reshape for broadcasting: [1, C, 1, 1, 1]
        mean_reshaped = mean.reshape(1, -1, 1, 1, 1)
        var_reshaped = var.reshape(1, -1, 1, 1, 1)

        normalized = (input - mean_reshaped) / Tensor._from_mlx_array(
            mx.sqrt(var_reshaped._mlx_array + self.eps)
        )

        if self.affine:
            weight_reshaped = self.weight.reshape(1, -1, 1, 1, 1)
            bias_reshaped = self.bias.reshape(1, -1, 1, 1, 1)
            output = normalized * weight_reshaped + bias_reshaped
        else:
            output = normalized

        return output

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )


class GroupNorm(Module):
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group.

    Args:
        num_groups: Number of groups to divide the channels into
        num_channels: Number of channels
        eps: Value added to denominator for numerical stability (default: 1e-5)
        affine: Whether to learn affine parameters (default: True)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [N, C, ...]
        - Output: [N, C, ...]
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_channels)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_channels)))
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply group normalization."""
        x = input._mlx_array
        shape = x.shape
        N = shape[0]
        C = shape[1]
        G = self.num_groups

        # Reshape to [N, G, C//G, ...]
        new_shape = (N, G, C // G) + shape[2:]
        x = mx.reshape(x, new_shape)

        # Compute mean and var over group dimensions (all dims except N and G)
        axes = tuple(range(2, len(new_shape)))
        mean = mx.mean(x, axis=axes, keepdims=True)
        var = mx.var(x, axis=axes, keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back to original shape
        x = mx.reshape(x, shape)

        result = Tensor._from_mlx_array(x)

        # Apply affine transformation
        if self.affine:
            # Reshape weight and bias for broadcasting
            ndim = len(shape)
            view_shape = (1, C) + (1,) * (ndim - 2)
            weight = self.weight.reshape(*view_shape)
            bias = self.bias.reshape(*view_shape)
            result = result * weight + bias

        return result

    def extra_repr(self) -> str:
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}'


class InstanceNorm1d(Module):
    """
    1D Instance Normalization.

    Normalizes each sample independently across spatial dimensions.

    Args:
        num_features: Number of features (channels)
        eps: Value added to denominator for numerical stability (default: 1e-5)
        momentum: Momentum for running stats (default: 0.1)
        affine: Whether to learn affine parameters (default: False)
        track_running_stats: Whether to track running statistics (default: False)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [N, C, L]
        - Output: [N, C, L]
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_features)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_features)))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor._from_mlx_array(mx.zeros(num_features))
            self.running_var = Tensor._from_mlx_array(mx.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply instance normalization."""
        x = input._mlx_array
        # [N, C, L] -> normalize over L for each (N, C) pair
        mean = mx.mean(x, axis=2, keepdims=True)
        var = mx.var(x, axis=2, keepdims=True)

        x = (x - mean) / mx.sqrt(var + self.eps)

        result = Tensor._from_mlx_array(x)

        if self.affine:
            weight = self.weight.reshape(1, -1, 1)
            bias = self.bias.reshape(1, -1, 1)
            result = result * weight + bias

        return result

    def extra_repr(self) -> str:
        return f'{self.num_features}, eps={self.eps}, affine={self.affine}'


class InstanceNorm2d(Module):
    """
    2D Instance Normalization.

    Normalizes each sample independently across spatial dimensions.

    Args:
        num_features: Number of features (channels)
        eps: Value added to denominator for numerical stability (default: 1e-5)
        momentum: Momentum for running stats (default: 0.1)
        affine: Whether to learn affine parameters (default: False)
        track_running_stats: Whether to track running statistics (default: False)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [N, C, H, W]
        - Output: [N, C, H, W]
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_features)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_features)))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor._from_mlx_array(mx.zeros(num_features))
            self.running_var = Tensor._from_mlx_array(mx.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply instance normalization."""
        x = input._mlx_array
        # [N, C, H, W] -> normalize over H, W for each (N, C) pair
        mean = mx.mean(x, axis=(2, 3), keepdims=True)
        var = mx.var(x, axis=(2, 3), keepdims=True)

        x = (x - mean) / mx.sqrt(var + self.eps)

        result = Tensor._from_mlx_array(x)

        if self.affine:
            weight = self.weight.reshape(1, -1, 1, 1)
            bias = self.bias.reshape(1, -1, 1, 1)
            result = result * weight + bias

        return result

    def extra_repr(self) -> str:
        return f'{self.num_features}, eps={self.eps}, affine={self.affine}'


class InstanceNorm3d(Module):
    """
    3D Instance Normalization.

    Normalizes each sample independently across spatial dimensions.

    Args:
        num_features: Number of features (channels)
        eps: Value added to denominator for numerical stability (default: 1e-5)
        momentum: Momentum for running stats (default: 0.1)
        affine: Whether to learn affine parameters (default: False)
        track_running_stats: Whether to track running statistics (default: False)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [N, C, D, H, W]
        - Output: [N, C, D, H, W]
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(num_features)))
            self.bias = Parameter(Tensor._from_mlx_array(mx.zeros(num_features)))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor._from_mlx_array(mx.zeros(num_features))
            self.running_var = Tensor._from_mlx_array(mx.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply instance normalization."""
        x = input._mlx_array
        # [N, C, D, H, W] -> normalize over D, H, W for each (N, C) pair
        mean = mx.mean(x, axis=(2, 3, 4), keepdims=True)
        var = mx.var(x, axis=(2, 3, 4), keepdims=True)

        x = (x - mean) / mx.sqrt(var + self.eps)

        result = Tensor._from_mlx_array(x)

        if self.affine:
            weight = self.weight.reshape(1, -1, 1, 1, 1)
            bias = self.bias.reshape(1, -1, 1, 1, 1)
            result = result * weight + bias

        return result

    def extra_repr(self) -> str:
        return f'{self.num_features}, eps={self.eps}, affine={self.affine}'


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization.

    A simplification of LayerNorm that does not re-center the activations.

    Args:
        normalized_shape: Input shape from an expected input
        eps: Value added to denominator for numerical stability (default: 1e-6)
        elementwise_affine: Whether to learn affine parameters (default: True)

    Shape:
        - Input: [..., normalized_shape]
        - Output: [..., normalized_shape]
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = None,
        elementwise_affine: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        # PyTorch defaults eps to None, then uses 1e-6 internally
        self.eps = eps if eps is not None else 1e-6
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(Tensor._from_mlx_array(mx.ones(normalized_shape)))
        else:
            self.weight = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply RMS normalization."""
        x = input._mlx_array

        # Compute RMS over the last dimension(s)
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))
        rms = mx.sqrt(mx.mean(x ** 2, axis=axes, keepdims=True) + self.eps)

        # Normalize
        x = x / rms

        result = Tensor._from_mlx_array(x)

        if self.elementwise_affine:
            result = result * self.weight

        return result

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class LocalResponseNorm(Module):
    """
    Local Response Normalization.

    Applies local response normalization over an input signal.

    Args:
        size: Amount of neighboring channels to normalize across
        alpha: Multiplicative factor (default: 1e-4)
        beta: Exponent (default: 0.75)
        k: Additive factor (default: 1)

    Shape:
        - Input: (N, C, ...)
        - Output: (N, C, ...)
    """

    def __init__(
        self,
        size: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0
    ):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """Apply local response normalization."""
        x = input._mlx_array
        shape = x.shape
        ndim = len(shape)

        # Square the input
        x_sq = mx.square(x)

        # Compute sum over local region in channel dimension
        # For simplicity, we use a sliding window approach
        C = shape[1]
        half_n = self.size // 2

        # Pad channels
        if ndim == 3:
            # [N, C, L]
            x_sq_padded = mx.pad(x_sq, [(0, 0), (half_n, half_n), (0, 0)])
        elif ndim == 4:
            # [N, C, H, W]
            x_sq_padded = mx.pad(x_sq, [(0, 0), (half_n, half_n), (0, 0), (0, 0)])
        elif ndim == 5:
            # [N, C, D, H, W]
            x_sq_padded = mx.pad(x_sq, [(0, 0), (half_n, half_n), (0, 0), (0, 0), (0, 0)])
        else:
            raise ValueError(f"LocalResponseNorm expects 3D, 4D or 5D input, got {ndim}D")

        # Sum over local window - use list comprehension for better accuracy
        if ndim == 3:
            slices_list = [x_sq_padded[:, i:i+C, :] for i in range(self.size)]
        elif ndim == 4:
            slices_list = [x_sq_padded[:, i:i+C, :, :] for i in range(self.size)]
        else:  # ndim == 5
            slices_list = [x_sq_padded[:, i:i+C, :, :, :] for i in range(self.size)]

        # Stack and sum
        local_sum = mx.sum(mx.stack(slices_list, axis=0), axis=0)

        # Apply normalization using mx.power for better precision
        # Note: PyTorch divides alpha by size (see PyTorch docs)
        norm_factor = mx.power(self.k + (self.alpha / self.size) * local_sum, self.beta)
        output = x / norm_factor

        result = Tensor._from_mlx_array(output)

        from ...autograd.context import is_grad_enabled
        if is_grad_enabled() and input.requires_grad:
            result.requires_grad = True

        return result

    def extra_repr(self) -> str:
        return f'{self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}'


class SyncBatchNorm(BatchNorm2d):
    """
    Synchronized Batch Normalization.

    This is a stub implementation that behaves like regular BatchNorm.
    MLX runs on single GPU, so there is no multi-GPU synchronization needed.

    Unlike BatchNorm2d, SyncBatchNorm can handle inputs of any dimension >= 2:
    - 2D input (N, C): BatchNorm1d style
    - 3D input (N, C, L): BatchNorm1d style
    - 4D input (N, C, H, W): BatchNorm2d style
    - 5D input (N, C, D, H, W): BatchNorm3d style

    Args:
        Same as BatchNorm2d
        process_group: Process group for synchronization (ignored in MLX)

    Note:
        This class is provided for API compatibility with PyTorch.
        It behaves identically to regular BatchNorm since MLX uses unified memory
        and doesn't support multi-GPU training.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group=None,
        device=None,
        dtype=None
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        # process_group ignored - MLX uses unified memory on single GPU
        self.process_group = process_group

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply batch normalization for any input dimension >= 2.

        Args:
            input: Input tensor of shape [N, C, ...] where ... can be any number of dims

        Returns:
            Normalized tensor of same shape as input
        """
        ndim = input.ndim
        if ndim < 2:
            raise ValueError(f"Expected at least 2D input, got {ndim}D")

        # Determine which dimensions to reduce over (all except batch and channel)
        # For input [N, C, ...], reduce over dims [0, 2, 3, ...]
        reduce_dims = [0] + list(range(2, ndim))

        if self.training:
            # Compute batch statistics over N and spatial dimensions
            batch_mean = input.mean(dim=reduce_dims, keepdim=False)  # [C]
            batch_var = input.var(dim=reduce_dims, keepdim=False, unbiased=False)  # [C]

            # Update running statistics
            if self.track_running_stats:
                momentum = self.momentum
                self.running_mean._mlx_array = (
                    (1 - momentum) * self.running_mean._mlx_array +
                    momentum * batch_mean._mlx_array
                )
                self.running_var._mlx_array = (
                    (1 - momentum) * self.running_var._mlx_array +
                    momentum * batch_var._mlx_array
                )
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = input.mean(dim=reduce_dims, keepdim=False)
                var = input.var(dim=reduce_dims, keepdim=False, unbiased=False)

        # Reshape mean and var to [1, C, 1, 1, ...] for broadcasting
        # Shape should be [1, C] followed by (ndim - 2) ones
        broadcast_shape = [1, -1] + [1] * (ndim - 2)
        mean_reshaped = mean.reshape(*broadcast_shape)
        var_reshaped = var.reshape(*broadcast_shape)

        normalized = (input - mean_reshaped) / Tensor._from_mlx_array(
            mx.sqrt(var_reshaped._mlx_array + self.eps)
        )

        # Apply affine transformation
        if self.affine:
            weight_reshaped = self.weight.reshape(*broadcast_shape)
            bias_reshaped = self.bias.reshape(*broadcast_shape)
            output = normalized * weight_reshaped + bias_reshaped
        else:
            output = normalized

        return output

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """
        Convert BatchNorm layers to SyncBatchNorm.

        This is a no-op in MLX since there's no multi-GPU.
        """
        return module


class CrossMapLRN2d(Module):
    """
    Cross-channel Local Response Normalization.

    Applies cross-channel local response normalization over a 4D input signal.
    This is a variant of LocalResponseNorm that normalizes across channels
    rather than within a spatial neighborhood.

    Args:
        size: Amount of neighboring channels to use for normalization
        alpha: Multiplicative factor (default: 0.0001)
        beta: Exponent (default: 0.75)
        k: Additive factor (default: 1)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) (same shape as input)

    Example:
        >>> lrn = nn.CrossMapLRN2d(5, alpha=1e-4, beta=0.75, k=1)
        >>> x = torch.randn(1, 8, 32, 32)
        >>> output = lrn(x)
    """

    def __init__(
        self,
        size: int,
        alpha: float = 0.0001,
        beta: float = 0.75,
        k: float = 1
    ):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        """Apply cross-channel LRN.

        This implementation matches PyTorch's exact algorithm which uses a sliding
        window approach with specific edge handling.
        """
        x = input._mlx_array
        N, C, H, W = x.shape

        # Square the input
        x_sq = mx.square(x)

        # PyTorch's algorithm uses a specific sliding window approach
        pre_pad = int((self.size - 1) / 2 + 1)
        pre_pad_crop = min(pre_pad, C)

        # Use cumulative sum for efficient sliding window computation
        # Pad with zeros at the beginning for the cumsum difference trick
        zeros_pad = mx.zeros((N, 1, H, W), dtype=x.dtype)
        x_sq_padded = mx.concatenate([zeros_pad, x_sq], axis=1)  # Shape: (N, C+1, H, W)
        cumsum = mx.cumsum(x_sq_padded, axis=1)  # Shape: (N, C+1, H, W)

        # Build scale using vectorized operations
        # For each channel c, we need sum of x_sq in a window around c
        # The window is asymmetric based on PyTorch's algorithm

        # Create index arrays for the sliding window bounds
        # For channel c: window starts at max(0, c - pre_pad + 1) and ends at min(C, c + pre_pad)
        c_indices = mx.arange(C)

        # Compute window bounds for each channel
        # PyTorch's window: channels [c - floor((size-1)/2), c + ceil((size-1)/2)]
        # But with the specific pre_pad logic
        half_size = self.size // 2

        # Build the scale tensor channel by channel using vectorized slicing
        # For efficiency, we compute all channels at once using the cumsum trick
        scale_parts = []
        for c in range(C):
            # Determine window for this channel based on PyTorch's algorithm
            if c == 0:
                # First channel: sum channels [0:pre_pad_crop]
                window_sum = cumsum[:, pre_pad_crop, :, :] - cumsum[:, 0, :, :]
            else:
                # Start from previous and adjust
                start_idx = max(0, c - half_size)
                end_idx = min(C, c + half_size + 1)
                window_sum = cumsum[:, end_idx, :, :] - cumsum[:, start_idx, :, :]
            scale_parts.append(mx.expand_dims(window_sum, axis=1))

        scale = mx.concatenate(scale_parts, axis=1)

        # Apply scaling: scale = scale * (alpha / size) + k
        scale = scale * (self.alpha / self.size) + self.k

        # Compute output: x * scale^(-beta)
        output = x * mx.power(scale, -self.beta)

        return Tensor._from_mlx_array(output)

    def extra_repr(self) -> str:
        return f'size={self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}'


__all__ = [
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'LayerNorm', 'GroupNorm',
    'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'RMSNorm', 'LocalResponseNorm', 'SyncBatchNorm',
    'CrossMapLRN2d',
]
