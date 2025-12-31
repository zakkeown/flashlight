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

    Note: This is a simplified stub. Full weight normalization requires
    hooks which are not fully implemented in MLX.

    Args:
        module: Module containing the parameter to apply weight norm to
        name: Name of the weight parameter
        dim: Dimension over which to compute the norm

    Returns:
        The module with weight normalization applied
    """
    import warnings
    warnings.warn(
        "weight_norm is a stub in MLX - weight normalization is not fully supported",
        UserWarning
    )
    return module


def remove_weight_norm(module: Module, name: str = 'weight') -> Module:
    """
    Remove weight normalization from a module.

    Args:
        module: Module to remove weight normalization from
        name: Name of the weight parameter

    Returns:
        The module with weight normalization removed
    """
    return module


def spectral_norm(module: Module, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12, dim: Optional[int] = None) -> Module:
    """
    Apply spectral normalization to a parameter in the given module.

    Note: This is a simplified stub. Full spectral normalization requires
    hooks which are not fully implemented in MLX.

    Args:
        module: Module containing the parameter to apply spectral norm to
        name: Name of the weight parameter
        n_power_iterations: Number of power iterations for estimation
        eps: Epsilon for numerical stability
        dim: Dimension over which to compute spectral norm

    Returns:
        The module with spectral normalization applied
    """
    import warnings
    warnings.warn(
        "spectral_norm is a stub in MLX - spectral normalization is not fully supported",
        UserWarning
    )
    return module


def remove_spectral_norm(module: Module, name: str = 'weight') -> Module:
    """
    Remove spectral normalization from a module.

    Args:
        module: Module to remove spectral normalization from
        name: Name of the weight parameter

    Returns:
        The module with spectral normalization removed
    """
    return module


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

    Note: This is a stub in MLX. Returns the conv module unchanged.

    Args:
        conv: Convolution module
        bn: BatchNorm module
        transpose: Whether this is a transposed convolution

    Returns:
        Fused module (conv unchanged in MLX)
    """
    import warnings
    warnings.warn(
        "fuse_conv_bn_eval is a stub in MLX - conv-bn fusion is not implemented",
        UserWarning
    )
    return conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose: bool = False):
    """
    Fuse Conv and BatchNorm weights.

    Note: This is a stub in MLX.

    Returns:
        Original conv weights and bias unchanged
    """
    import warnings
    warnings.warn(
        "fuse_conv_bn_weights is a stub in MLX - conv-bn fusion is not implemented",
        UserWarning
    )
    return conv_w, conv_b


def fuse_linear_bn_eval(linear: Module, bn: Module) -> Module:
    """
    Fuse Linear and BatchNorm modules for evaluation.

    Note: This is a stub in MLX. Returns the linear module unchanged.

    Args:
        linear: Linear module
        bn: BatchNorm module

    Returns:
        Fused module (linear unchanged in MLX)
    """
    import warnings
    warnings.warn(
        "fuse_linear_bn_eval is a stub in MLX - linear-bn fusion is not implemented",
        UserWarning
    )
    return linear


def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """
    Fuse Linear and BatchNorm weights.

    Note: This is a stub in MLX.

    Returns:
        Original linear weights and bias unchanged
    """
    import warnings
    warnings.warn(
        "fuse_linear_bn_weights is a stub in MLX - linear-bn fusion is not implemented",
        UserWarning
    )
    return linear_w, linear_b


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
