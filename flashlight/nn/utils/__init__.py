"""
Neural Network Utilities

PyTorch-compatible torch.nn.utils module providing various utilities
for neural network training and manipulation.
"""

from typing import Iterable, List, Optional, Union
import math

from ...tensor import Tensor
from ..module import Module


def clip_grad_norm_(
    parameters: Union[Tensor, Iterable[Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> Tensor:
    """
    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Args:
        parameters: An iterable of Tensors or a single Tensor
        max_norm: Max norm of the gradients
        norm_type: Type of the used p-norm. Can be 'inf' for infinity norm.
        error_if_nonfinite: If True, raise error if total norm is nan/inf
        foreach: Use foreach implementation (ignored in MLX)

    Returns:
        Total norm of the parameter gradients (as a single Tensor)
    """
    import mlx.core as mx

    if isinstance(parameters, Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return Tensor(mx.array(0.0))

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        norms = [mx.abs(g._mlx_array).max() for g in grads]
        total_norm = Tensor(mx.max(mx.stack(norms)))
    else:
        total_norm_sq = sum(mx.power(mx.abs(g._mlx_array), norm_type).sum() for g in grads)
        total_norm = Tensor(mx.power(total_norm_sq, 1.0 / norm_type))

    total_norm_val = float(total_norm.item()) if hasattr(total_norm, 'item') else float(total_norm._mlx_array.item())
    if error_if_nonfinite and (math.isnan(total_norm_val) or math.isinf(total_norm_val)):
        raise RuntimeError(
            f'The total norm of order {norm_type} is non-finite, so it cannot be clipped.'
        )

    clip_coef = max_norm / (total_norm_val + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g._mlx_array = g._mlx_array * clip_coef

    return total_norm


def clip_grad_value_(
    parameters: Union[Tensor, Iterable[Tensor]],
    clip_value: float,
    foreach: Optional[bool] = None,
) -> None:
    """
    Clips gradient of an iterable of parameters at specified value.

    Args:
        parameters: An iterable of Tensors or a single Tensor
        clip_value: Maximum allowed value of the gradients (clamps to [-clip_value, clip_value])
        foreach: Use foreach implementation (ignored in MLX)
    """
    import mlx.core as mx

    if isinstance(parameters, Tensor):
        parameters = [parameters]

    clip_value = float(clip_value)

    for p in parameters:
        if p.grad is not None:
            p.grad._mlx_array = mx.clip(p.grad._mlx_array, -clip_value, clip_value)


def get_total_norm(
    tensors: Iterable[Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> Tensor:
    """
    Compute the total norm of tensors.

    Args:
        tensors: An iterable of Tensors
        norm_type: Type of the used p-norm
        error_if_nonfinite: If True, raise error if total norm is nan/inf
        foreach: Use foreach implementation (ignored in MLX)

    Returns:
        Total norm as a Tensor
    """
    import mlx.core as mx

    tensors = list(tensors)
    if len(tensors) == 0:
        return Tensor(mx.array(0.0))

    norm_type = float(norm_type)

    if norm_type == float('inf'):
        norms = [mx.abs(t._mlx_array).max() for t in tensors]
        total_norm = Tensor(mx.max(mx.stack(norms)))
    else:
        total_norm_sq = sum(mx.power(mx.abs(t._mlx_array), norm_type).sum() for t in tensors)
        total_norm = Tensor(mx.power(total_norm_sq, 1.0 / norm_type))

    if error_if_nonfinite:
        total_norm_val = float(total_norm.item()) if hasattr(total_norm, 'item') else float(total_norm._mlx_array.item())
        if math.isnan(total_norm_val) or math.isinf(total_norm_val):
            raise RuntimeError(f'The total norm is non-finite.')

    return total_norm


def parameters_to_vector(parameters: Iterable[Tensor]) -> Tensor:
    """
    Convert parameters to a single vector.

    Args:
        parameters: An iterable of Tensors

    Returns:
        A 1D Tensor containing all parameter values
    """
    import mlx.core as mx

    vec = []
    for param in parameters:
        vec.append(param.flatten()._mlx_array)

    return Tensor(mx.concatenate(vec))


def vector_to_parameters(vec: Tensor, parameters: Iterable[Tensor]) -> None:
    """
    Convert a vector back to parameters.

    Args:
        vec: A 1D Tensor containing all parameter values
        parameters: An iterable of Tensors to fill with values from vec
    """
    pointer = 0
    for param in parameters:
        num_param = param.numel if isinstance(param.numel, int) else param.numel()
        param._mlx_array = vec._mlx_array[pointer:pointer + num_param].reshape(param.shape)
        pointer += num_param


def weight_norm(module: Module, name: str = 'weight', dim: int = 0) -> Module:
    """
    Apply weight normalization to a parameter in the given module.

    Weight normalization is a reparametrization that decouples the magnitude
    of weight vectors from their direction:
        weight = g * (v / ||v||)

    This can accelerate convergence of stochastic gradient descent.

    Args:
        module: Module containing the parameter to apply weight norm to
        name: Name of the weight parameter (default: 'weight')
        dim: Dimension over which to compute the norm (default: 0)

    Returns:
        The module with weight normalization applied

    Example:
        >>> layer = nn.Linear(20, 40)
        >>> layer = nn.utils.weight_norm(layer)
    """
    from .parametrizations import weight_norm as _weight_norm
    return _weight_norm(module, name, dim)


def remove_weight_norm(module: Module, name: str = 'weight') -> Module:
    """
    Remove weight normalization from a module.

    Args:
        module: Module to remove weight normalization from
        name: Name of the weight parameter (default: 'weight')

    Returns:
        The module with weight normalization removed
    """
    from .parametrizations import remove_weight_norm as _remove_weight_norm
    return _remove_weight_norm(module, name)


def spectral_norm(module: Module, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12, dim: Optional[int] = None) -> Module:
    """
    Apply spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of GANs by constraining
    the Lipschitz constant of the discriminator. It normalizes the weight
    matrix by its spectral norm (largest singular value).

    Args:
        module: Module containing the parameter to apply spectral norm to
        name: Name of the weight parameter (default: 'weight')
        n_power_iterations: Number of power iterations for estimation (default: 1)
        eps: Epsilon for numerical stability (default: 1e-12)
        dim: Dimension over which to compute spectral norm (default: 0)

    Returns:
        The module with spectral normalization applied

    Example:
        >>> layer = nn.Linear(20, 40)
        >>> layer = nn.utils.spectral_norm(layer)
    """
    from .parametrizations import spectral_norm as _spectral_norm
    return _spectral_norm(module, name, n_power_iterations, eps, dim)


def remove_spectral_norm(module: Module, name: str = 'weight') -> Module:
    """
    Remove spectral normalization from a module.

    Args:
        module: Module to remove spectral normalization from
        name: Name of the weight parameter (default: 'weight')

    Returns:
        The module with spectral normalization removed
    """
    from .parametrizations import remove_spectral_norm as _remove_spectral_norm
    return _remove_spectral_norm(module, name)


def skip_init(module_cls, *args, **kwargs):
    """
    Create a module instance without initializing parameters.

    Args:
        module_cls: The module class to instantiate
        *args: Positional arguments for the module
        **kwargs: Keyword arguments for the module

    Returns:
        An uninitialized module instance
    """
    # In MLX, we just create the module normally
    # since lazy initialization is handled differently
    return module_cls(*args, **kwargs)


# Aliases for compatibility
clip_grad_norm = clip_grad_norm_


def clip_grads_with_norm_(
    parameters: Union[Tensor, Iterable[Tensor]],
    max_norm: float,
    total_norm: Tensor,
    foreach: Optional[bool] = None,
) -> None:
    """
    Clip gradients given a pre-computed total norm.

    Args:
        parameters: An iterable of Tensors or a single Tensor
        max_norm: Max norm of the gradients
        total_norm: Pre-computed total norm
        foreach: Use foreach implementation (ignored in MLX)
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    total_norm_val = float(total_norm.item()) if hasattr(total_norm, 'item') else float(total_norm._mlx_array.item())
    clip_coef = max_norm / (total_norm_val + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad._mlx_array = p.grad._mlx_array * clip_coef


def convert_conv2d_weight_memory_format(module: Module, memory_format) -> Module:
    """
    Convert Conv2d weight memory format.

    Note: MLX uses a different memory layout. This is a no-op stub.

    Args:
        module: Module to convert
        memory_format: Target memory format

    Returns:
        The module (unchanged in MLX)
    """
    return module


def convert_conv3d_weight_memory_format(module: Module, memory_format) -> Module:
    """
    Convert Conv3d weight memory format.

    Note: MLX uses a different memory layout. This is a no-op stub.

    Args:
        module: Module to convert
        memory_format: Target memory format

    Returns:
        The module (unchanged in MLX)
    """
    return module


def fuse_conv_bn_eval(conv: Module, bn: Module, transpose: bool = False) -> Module:
    """
    Fuse Conv and BatchNorm modules for evaluation.

    This combines the convolution weights with BatchNorm parameters for more
    efficient inference. The fused convolution produces the same output as
    running conv followed by bn in eval mode.

    Args:
        conv: Convolution module (Conv1d, Conv2d, or Conv3d)
        bn: BatchNorm module (BatchNorm1d, BatchNorm2d, or BatchNorm3d)
        transpose: Whether this is a transposed convolution

    Returns:
        Fused convolution module with updated weights and bias

    Note:
        The BatchNorm must be in eval mode (bn.training = False) for correct results.
        After fusion, the BatchNorm layer should be removed from the model.

    Formula:
        W_fused = W * (gamma / sqrt(running_var + eps))
        b_fused = gamma * (b - running_mean) / sqrt(running_var + eps) + beta
    """
    import mlx.core as mx
    import copy

    # Get conv parameters
    conv_w = conv.weight._mlx_array
    conv_b = conv.bias._mlx_array if conv.bias is not None else None

    # Get bn parameters
    bn_rm = bn.running_mean._mlx_array if bn.running_mean is not None else mx.zeros(bn.num_features)
    bn_rv = bn.running_var._mlx_array if bn.running_var is not None else mx.ones(bn.num_features)
    bn_eps = bn.eps
    bn_w = bn.weight._mlx_array if bn.weight is not None else mx.ones(bn.num_features)
    bn_b = bn.bias._mlx_array if bn.bias is not None else mx.zeros(bn.num_features)

    # Fuse the weights
    fused_w, fused_b = fuse_conv_bn_weights(
        conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=transpose
    )

    # Create a copy of the conv layer with fused weights
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight._mlx_array = fused_w

    # Set the bias (create one if it didn't exist)
    if fused_conv.bias is None:
        from ..parameter import Parameter
        fused_conv.bias = Parameter(Tensor._from_mlx_array(fused_b))
    else:
        fused_conv.bias._mlx_array = fused_b

    # Invalidate any weight cache in the conv layer
    if hasattr(fused_conv, '_cached_weight_mlx'):
        fused_conv._cached_weight_mlx = None
        fused_conv._cached_weight_id = None

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose: bool = False):
    """
    Fuse Conv and BatchNorm weights.

    Combines convolution weights with BatchNorm parameters to produce fused
    weights and bias that give equivalent output.

    Args:
        conv_w: Convolution weight tensor (MLX array)
                Shape: [out_channels, in_channels/groups, *kernel_size] for normal conv
                Shape: [in_channels, out_channels/groups, *kernel_size] for transpose conv
        conv_b: Convolution bias tensor or None (MLX array)
                Shape: [out_channels]
        bn_rm: BatchNorm running mean (MLX array)
               Shape: [num_features]
        bn_rv: BatchNorm running variance (MLX array)
               Shape: [num_features]
        bn_eps: BatchNorm epsilon value (float)
        bn_w: BatchNorm weight (gamma) or None (MLX array)
              Shape: [num_features]
        bn_b: BatchNorm bias (beta) or None (MLX array)
              Shape: [num_features]
        transpose: Whether this is a transposed convolution

    Returns:
        Tuple of (fused_weight, fused_bias) as MLX arrays

    Formula:
        W_fused = W * (gamma / sqrt(running_var + eps))
        b_fused = gamma * (b - running_mean) / sqrt(running_var + eps) + beta
    """
    import mlx.core as mx

    # Compute the scale factor: gamma / sqrt(var + eps)
    # Shape: [out_channels] for normal conv, [in_channels] for transpose
    std = mx.sqrt(bn_rv + bn_eps)
    scale = bn_w / std

    # Handle missing conv bias
    if conv_b is None:
        conv_b = mx.zeros(bn_rm.shape)

    if transpose:
        # For transposed conv: weight shape is [in_channels, out_channels/groups, *kernel_size]
        # The output channels are at dimension 1, but BN normalizes over in_channels (dim 0)
        # Actually, for ConvTranspose, output channels = weight.shape[1] * groups
        # BN is applied to output, so we scale by output channels

        # Reshape scale for broadcasting: [1, out_channels, 1, 1, ...]
        # Number of spatial dims = weight.ndim - 2
        num_spatial_dims = conv_w.ndim - 2
        scale_shape = (1, -1) + (1,) * num_spatial_dims
        scale_reshaped = mx.reshape(scale, scale_shape)

        # Fuse weight: multiply each output channel by its scale
        fused_w = conv_w * scale_reshaped
    else:
        # For normal conv: weight shape is [out_channels, in_channels/groups, *kernel_size]
        # Reshape scale for broadcasting: [out_channels, 1, 1, 1, ...]
        num_spatial_dims = conv_w.ndim - 2
        scale_shape = (-1,) + (1,) * (num_spatial_dims + 1)
        scale_reshaped = mx.reshape(scale, scale_shape)

        # Fuse weight: multiply each output channel by its scale
        fused_w = conv_w * scale_reshaped

    # Fuse bias: gamma * (b - mean) / std + beta
    # = gamma * b / std - gamma * mean / std + beta
    # = b * scale - mean * scale + beta
    fused_b = (conv_b - bn_rm) * scale + bn_b

    return fused_w, fused_b


def fuse_linear_bn_eval(linear: Module, bn: Module) -> Module:
    """
    Fuse Linear and BatchNorm modules for evaluation.

    This combines the linear layer weights with BatchNorm parameters for more
    efficient inference. The fused linear layer produces the same output as
    running linear followed by bn in eval mode.

    Args:
        linear: Linear module
        bn: BatchNorm module (BatchNorm1d)

    Returns:
        Fused linear module with updated weights and bias

    Note:
        The BatchNorm must be in eval mode (bn.training = False) for correct results.
        After fusion, the BatchNorm layer should be removed from the model.
        This is typically used when Linear is followed by BatchNorm1d.

    Formula:
        W_fused = W * (gamma / sqrt(running_var + eps))
        b_fused = gamma * (b - running_mean) / sqrt(running_var + eps) + beta
    """
    import mlx.core as mx
    import copy

    # Get linear parameters
    linear_w = linear.weight._mlx_array
    linear_b = linear.bias._mlx_array if linear.bias is not None else None

    # Get bn parameters
    bn_rm = bn.running_mean._mlx_array if bn.running_mean is not None else mx.zeros(bn.num_features)
    bn_rv = bn.running_var._mlx_array if bn.running_var is not None else mx.ones(bn.num_features)
    bn_eps = bn.eps
    bn_w = bn.weight._mlx_array if bn.weight is not None else mx.ones(bn.num_features)
    bn_b = bn.bias._mlx_array if bn.bias is not None else mx.zeros(bn.num_features)

    # Fuse the weights
    fused_w, fused_b = fuse_linear_bn_weights(
        linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b
    )

    # Create a copy of the linear layer with fused weights
    fused_linear = copy.deepcopy(linear)
    fused_linear.weight._mlx_array = fused_w

    # Set the bias (create one if it didn't exist)
    if fused_linear.bias is None:
        from ..parameter import Parameter
        fused_linear.bias = Parameter(Tensor._from_mlx_array(fused_b))
    else:
        fused_linear.bias._mlx_array = fused_b

    # Invalidate any weight cache in the linear layer
    if hasattr(fused_linear, '_weight_cache'):
        fused_linear._weight_cache = {}

    return fused_linear


def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """
    Fuse Linear and BatchNorm weights.

    Combines linear layer weights with BatchNorm parameters to produce fused
    weights and bias that give equivalent output.

    Args:
        linear_w: Linear weight tensor (MLX array)
                  Shape: [out_features, in_features]
        linear_b: Linear bias tensor or None (MLX array)
                  Shape: [out_features]
        bn_rm: BatchNorm running mean (MLX array)
               Shape: [num_features] (should equal out_features)
        bn_rv: BatchNorm running variance (MLX array)
               Shape: [num_features]
        bn_eps: BatchNorm epsilon value (float)
        bn_w: BatchNorm weight (gamma) or None (MLX array)
              Shape: [num_features]
        bn_b: BatchNorm bias (beta) or None (MLX array)
              Shape: [num_features]

    Returns:
        Tuple of (fused_weight, fused_bias) as MLX arrays

    Formula:
        W_fused = W * (gamma / sqrt(running_var + eps)).reshape(-1, 1)
        b_fused = gamma * (b - running_mean) / sqrt(running_var + eps) + beta

    Note:
        For Linear layer with weight [out_features, in_features]:
        - Each row corresponds to one output feature
        - BatchNorm normalizes each output feature independently
        - So we scale each row of weights by the corresponding scale factor
    """
    import mlx.core as mx

    # Compute the scale factor: gamma / sqrt(var + eps)
    # Shape: [out_features]
    std = mx.sqrt(bn_rv + bn_eps)
    scale = bn_w / std

    # Handle missing linear bias
    if linear_b is None:
        linear_b = mx.zeros(bn_rm.shape)

    # Fuse weight: multiply each output feature (row) by its scale
    # Linear weight shape: [out_features, in_features]
    # scale shape: [out_features] -> reshape to [out_features, 1] for broadcasting
    scale_reshaped = mx.reshape(scale, (-1, 1))
    fused_w = linear_w * scale_reshaped

    # Fuse bias: gamma * (b - mean) / std + beta
    # = (b - mean) * scale + beta
    fused_b = (linear_b - bn_rm) * scale + bn_b

    return fused_w, fused_b


# Import submodules
from . import rnn
from . import parametrizations
from . import stateless


__all__ = [
    'clip_grad_norm_',
    'clip_grad_norm',
    'clip_grad_value_',
    'clip_grads_with_norm_',
    'get_total_norm',
    'parameters_to_vector',
    'vector_to_parameters',
    'weight_norm',
    'remove_weight_norm',
    'spectral_norm',
    'remove_spectral_norm',
    'skip_init',
    'convert_conv2d_weight_memory_format',
    'convert_conv3d_weight_memory_format',
    'fuse_conv_bn_eval',
    'fuse_conv_bn_weights',
    'fuse_linear_bn_eval',
    'fuse_linear_bn_weights',
    'rnn',
    'parametrizations',
    'stateless',
]
