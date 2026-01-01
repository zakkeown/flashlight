"""
Mobile optimization utilities.

Provides optimization functions for deploying models on mobile/edge devices.
MLX is already optimized for Apple Silicon, so these provide API compatibility
while applying MLX-specific optimizations where applicable.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import warnings


class MobileOptimizerType(Enum):
    """Types of mobile optimizations available."""

    CONV_BN_FUSION = "conv_bn_fusion"
    INSERT_FOLD_PREPACK_OPS = "insert_fold_prepack_ops"
    REMOVE_DROPOUT = "remove_dropout"
    FUSE_ADD_RELU = "fuse_add_relu"
    FUSE_HARDSWISH = "fuse_hardswish"
    FUSE_CLAMP_MIN_MAX = "fuse_clamp_min_max"
    HOIST_CONV_PACKED_PARAMS = "hoist_conv_packed_params"


class LintCode(Enum):
    """Lint codes for mobile module analysis."""

    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4


def optimize_for_mobile(
    script_module: Any,
    optimization_blocklist: Optional[Set[MobileOptimizerType]] = None,
    preserved_methods: Optional[List[str]] = None,
    backend: str = "CPU",
) -> Any:
    """
    Optimize a model for mobile/edge deployment.

    For MLX, this applies optimizations suitable for Apple Silicon inference.
    Since MLX already uses Metal for acceleration, many PyTorch mobile
    optimizations are not needed. This function focuses on:

    1. Removing dropout layers (inference mode)
    2. Fusing batch normalization with convolutions
    3. Converting to inference mode

    Args:
        script_module: The module to optimize. Can be an nn.Module.
        optimization_blocklist: Set of optimizations to skip.
        preserved_methods: Methods to preserve during optimization.
        backend: Target backend (ignored, MLX uses Metal).

    Returns:
        Optimized module ready for inference.

    Example:
        >>> model = MyModel()
        >>> model.eval()
        >>> optimized = optimize_for_mobile(model)
    """
    if optimization_blocklist is None:
        optimization_blocklist = set()

    # Import here to avoid circular imports
    import mlx_compat.nn as nn

    # Clone the module for optimization
    optimized = _clone_module(script_module)

    # Set to eval mode
    if hasattr(optimized, "eval"):
        optimized.eval()

    # Apply optimizations
    if MobileOptimizerType.REMOVE_DROPOUT not in optimization_blocklist:
        _remove_dropout(optimized)

    if MobileOptimizerType.CONV_BN_FUSION not in optimization_blocklist:
        _fuse_conv_bn(optimized)

    return optimized


def _clone_module(module: Any) -> Any:
    """Clone a module for optimization."""
    import copy

    try:
        return copy.deepcopy(module)
    except Exception:
        # If deepcopy fails, return the original
        warnings.warn(
            "Could not clone module for optimization, modifying in place",
            UserWarning,
        )
        return module


def _remove_dropout(module: Any) -> None:
    """Remove dropout layers from a module."""
    import mlx_compat.nn as nn

    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            # Replace with identity
            setattr(module, name, nn.Identity())
        else:
            _remove_dropout(child)


def _fuse_conv_bn(module: Any) -> None:
    """
    Fuse Conv + BatchNorm layers.

    This optimization folds batch normalization parameters into the
    preceding convolution, eliminating the batch norm computation.
    """
    import mlx_compat.nn as nn
    import mlx.core as mx

    children = list(module.named_children())

    for i, (name, child) in enumerate(children):
        # Check if this is a conv followed by batch norm
        if i + 1 < len(children):
            next_name, next_child = children[i + 1]

            is_conv = isinstance(child, (nn.Conv1d, nn.Conv2d))
            is_bn = isinstance(next_child, (nn.BatchNorm1d, nn.BatchNorm2d))

            if is_conv and is_bn:
                # Fuse the layers
                _fuse_conv_bn_pair(child, next_child)
                # Replace batch norm with identity
                setattr(module, next_name, nn.Identity())

        # Recurse into children
        _fuse_conv_bn(child)


def _fuse_conv_bn_pair(conv: Any, bn: Any) -> None:
    """
    Fuse a single conv-bn pair.

    The fused weight and bias are computed as:
    w_fused = w * (gamma / sqrt(var + eps))
    b_fused = (b - mean) * (gamma / sqrt(var + eps)) + beta
    """
    import mlx.core as mx

    # Get batch norm parameters
    gamma = bn.weight._mlx_array if hasattr(bn.weight, "_mlx_array") else bn.weight
    beta = bn.bias._mlx_array if hasattr(bn.bias, "_mlx_array") else bn.bias
    mean = bn.running_mean._mlx_array if hasattr(bn.running_mean, "_mlx_array") else bn.running_mean
    var = bn.running_var._mlx_array if hasattr(bn.running_var, "_mlx_array") else bn.running_var
    eps = bn.eps

    # Compute scale factor
    scale = gamma / mx.sqrt(var + eps)

    # Get conv parameters
    w = conv.weight._mlx_array if hasattr(conv.weight, "_mlx_array") else conv.weight

    # Fuse weight: reshape scale for broadcasting
    # Conv weight shape: [out_channels, kH, kW, in_channels] for MLX
    scale_shape = [scale.shape[0]] + [1] * (len(w.shape) - 1)
    w_fused = w * scale.reshape(scale_shape)

    # Fuse bias
    if conv.bias is not None:
        b = conv.bias._mlx_array if hasattr(conv.bias, "_mlx_array") else conv.bias
        b_fused = (b - mean) * scale + beta
    else:
        b_fused = -mean * scale + beta

    # Update conv parameters
    import mlx_compat

    conv.weight = mlx_compat.nn.Parameter(mlx_compat.tensor(w_fused))
    conv.bias = mlx_compat.nn.Parameter(mlx_compat.tensor(b_fused))


def generate_mobile_module_lints(module: Any) -> List[Dict[str, Any]]:
    """
    Generate lint warnings for mobile deployment.

    Analyzes a module and returns a list of potential issues that
    may affect mobile deployment.

    Args:
        module: The module to analyze.

    Returns:
        List of lint dictionaries with 'code', 'message', and 'name' keys.
    """
    import mlx_compat.nn as nn

    lints = []

    def check_module(name: str, mod: Any) -> None:
        # Check for dropout
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            lints.append({
                "code": LintCode.DROPOUT,
                "name": name,
                "message": f"Dropout layer '{name}' found. Consider removing for inference.",
            })

        # Check for batch norm in training mode
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if mod.training:
                lints.append({
                    "code": LintCode.BATCHNORM,
                    "name": name,
                    "message": f"BatchNorm '{name}' is in training mode. Call model.eval() for inference.",
                })

        # Check for requires_grad on parameters
        for param_name, param in mod.named_parameters():
            if hasattr(param, "requires_grad") and param.requires_grad:
                lints.append({
                    "code": LintCode.REQUIRES_GRAD,
                    "name": f"{name}.{param_name}",
                    "message": f"Parameter '{name}.{param_name}' has requires_grad=True.",
                })
                break  # Only report once per module

    # Walk the module tree
    for name, child in module.named_modules():
        check_module(name if name else "root", child)

    return lints
