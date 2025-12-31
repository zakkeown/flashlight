"""
Parametrizations

PyTorch-compatible torch.nn.utils.parametrizations module.
Provides utilities for parametrizing module parameters.
"""

import warnings
from typing import Optional

from ..module import Module


def orthogonal(
    module: Module,
    name: str = 'weight',
    orthogonal_map: Optional[str] = None,
    *,
    use_trivialization: bool = True,
) -> Module:
    """
    Apply orthogonal parametrization to a matrix parameter.

    Note: This is a stub in MLX.

    Args:
        module: Module containing the parameter
        name: Name of the parameter
        orthogonal_map: Type of orthogonal map to use
        use_trivialization: Whether to use trivialization

    Returns:
        The module with orthogonal parametrization
    """
    warnings.warn(
        "orthogonal parametrization is a stub in MLX",
        UserWarning
    )
    return module


def spectral_norm(
    module: Module,
    name: str = 'weight',
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> Module:
    """
    Apply spectral normalization to a parameter.

    Note: This is a stub in MLX.

    Args:
        module: Module containing the parameter
        name: Name of the parameter
        n_power_iterations: Number of power iterations
        eps: Epsilon for numerical stability
        dim: Dimension over which to compute spectral norm

    Returns:
        The module with spectral normalization
    """
    warnings.warn(
        "spectral_norm parametrization is a stub in MLX",
        UserWarning
    )
    return module


def weight_norm(
    module: Module,
    name: str = 'weight',
    dim: int = 0,
) -> Module:
    """
    Apply weight normalization to a parameter.

    Note: This is a stub in MLX.

    Args:
        module: Module containing the parameter
        name: Name of the parameter
        dim: Dimension over which to compute norm

    Returns:
        The module with weight normalization
    """
    warnings.warn(
        "weight_norm parametrization is a stub in MLX",
        UserWarning
    )
    return module


__all__ = [
    'orthogonal',
    'spectral_norm',
    'weight_norm',
]
