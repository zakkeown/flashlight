"""
Parametrizations

PyTorch-compatible torch.nn.utils.parametrizations module.
Provides utilities for parametrizing module parameters.

Weight Normalization:
    Reparametrizes weight as: weight = g * (v / ||v||)
    where g is a learned scalar and v is the unnormalized weight.

Spectral Normalization:
    Normalizes weight by its spectral norm (largest singular value):
    weight = weight / sigma(weight)
    Uses power iteration to estimate sigma.
"""

import math
import warnings
from typing import Optional

import mlx.core as mx

from ...tensor import Tensor
from ..module import Module
from ..parameter import Parameter


class _WeightNorm:
    """
    Helper class to apply weight normalization reparametrization.

    Weight normalization reparametrizes the weight tensor as:
        weight = g * (v / ||v||)
    where g is a scalar magnitude and v is the direction vector.
    """

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim

    @staticmethod
    def compute_weight(v: mx.array, g: mx.array, dim: int) -> mx.array:
        """Compute the weight from v (direction) and g (magnitude)."""
        # Compute norm over all dimensions except dim
        ndim = v.ndim
        if dim < 0:
            dim = ndim + dim

        # Build reduction axes (all except dim)
        axes = [i for i in range(ndim) if i != dim]

        # Compute L2 norm
        if axes:
            norm = mx.sqrt(mx.sum(v**2, axis=axes, keepdims=True))
        else:
            norm = mx.sqrt(mx.sum(v**2))

        # Normalize and scale
        return g * (v / (norm + 1e-12))

    def apply(self, module: Module, inputs) -> None:
        """Pre-forward hook to compute and set the normalized weight."""
        v = getattr(module, self.name + "_v")
        g = getattr(module, self.name + "_g")

        # Compute normalized weight
        weight = self.compute_weight(v._mlx_array, g._mlx_array, self.dim)

        # Set the weight parameter
        setattr(module, self.name, Parameter(Tensor._from_mlx_array(weight)))


class _SpectralNorm:
    """
    Helper class to apply spectral normalization.

    Spectral normalization normalizes the weight by its spectral norm
    (largest singular value), estimated via power iteration.
    """

    def __init__(self, name: str, n_power_iterations: int, dim: int, eps: float):
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps

    @staticmethod
    def reshape_weight_to_matrix(weight: mx.array, dim: int) -> mx.array:
        """Reshape weight tensor to 2D matrix for spectral norm computation."""
        if dim != 0:
            # Permute to move dim to position 0
            perm = [dim] + [i for i in range(weight.ndim) if i != dim]
            weight = mx.transpose(weight, perm)

        # Reshape to 2D: (dim_size, everything_else)
        height = weight.shape[0]
        return mx.reshape(weight, (height, -1))

    def compute_weight(self, weight: mx.array, u: mx.array, v: mx.array) -> tuple:
        """
        Compute spectral normalized weight using power iteration.

        Returns: (normalized_weight, updated_u, updated_v)
        """
        weight_mat = self.reshape_weight_to_matrix(weight, self.dim)

        # Power iteration
        u_new = u
        v_new = v
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v_new = mx.matmul(weight_mat.T, u_new)
            v_new = v_new / (mx.sqrt(mx.sum(v_new**2)) + self.eps)
            # u = W v / ||W v||
            u_new = mx.matmul(weight_mat, v_new)
            u_new = u_new / (mx.sqrt(mx.sum(u_new**2)) + self.eps)

        # Compute spectral norm: sigma = u^T W v
        sigma = mx.sum(u_new * mx.matmul(weight_mat, v_new))

        # Normalize weight
        weight_normalized = weight / (sigma + self.eps)

        return weight_normalized, u_new, v_new

    def apply(self, module: Module, inputs) -> None:
        """Pre-forward hook to compute and set the spectrally normalized weight."""
        weight_orig = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # Compute normalized weight
        weight_normalized, u_new, v_new = self.compute_weight(
            weight_orig._mlx_array, u._mlx_array, v._mlx_array
        )

        # Update u and v for next iteration (only during training)
        if module.training:
            module._buffers[self.name + "_u"] = Tensor._from_mlx_array(u_new)
            module._buffers[self.name + "_v"] = Tensor._from_mlx_array(v_new)

        # Set the weight parameter
        setattr(module, self.name, Parameter(Tensor._from_mlx_array(weight_normalized)))


def orthogonal(
    module: Module,
    name: str = "weight",
    orthogonal_map: Optional[str] = None,
    *,
    use_trivialization: bool = True,
) -> Module:
    """
    Apply orthogonal parametrization to a matrix parameter.

    Note: This is a stub in MLX. Full orthogonal parametrization requires
    matrix exponential or Cayley transform which are complex to implement.

    Args:
        module: Module containing the parameter
        name: Name of the parameter
        orthogonal_map: Type of orthogonal map to use
        use_trivialization: Whether to use trivialization

    Returns:
        The module with orthogonal parametrization
    """
    warnings.warn("orthogonal parametrization is a stub in MLX", UserWarning)
    return module


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> Module:
    """
    Apply spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of GANs by constraining
    the Lipschitz constant of the discriminator. It normalizes the weight
    matrix by its spectral norm (largest singular value).

    The spectral norm is estimated using power iteration.

    Args:
        module: Module containing the parameter to apply spectral norm to
        name: Name of the weight parameter (default: 'weight')
        n_power_iterations: Number of power iterations for estimating spectral norm.
                           More iterations = more accurate but slower. (default: 1)
        eps: Epsilon for numerical stability (default: 1e-12)
        dim: Dimension corresponding to the output of the module. For Conv2d,
             this is typically 0. If None, defaults to 0. (default: None)

    Returns:
        The module with spectral normalization applied

    Example:
        >>> layer = nn.Linear(20, 40)
        >>> layer = spectral_norm(layer)
        >>> # Now layer.weight is spectrally normalized during forward pass
    """
    if dim is None:
        dim = 0

    weight = getattr(module, name)
    if weight is None:
        raise ValueError(f"Module {module} has no parameter or buffer named {name}")

    weight_array = weight._mlx_array
    height = weight_array.shape[dim] if dim < weight_array.ndim else weight_array.shape[0]
    weight_mat = _SpectralNorm.reshape_weight_to_matrix(weight_array, dim)
    width = weight_mat.shape[1]

    # Initialize u and v vectors randomly
    u = mx.random.normal(shape=(height,))
    u = u / (mx.sqrt(mx.sum(u**2)) + eps)
    v = mx.random.normal(shape=(width,))
    v = v / (mx.sqrt(mx.sum(v**2)) + eps)

    # Register the original weight and u, v vectors
    delattr(module, name)
    module.register_parameter(name + "_orig", Parameter(weight))
    module.register_buffer(name + "_u", Tensor._from_mlx_array(u))
    module.register_buffer(name + "_v", Tensor._from_mlx_array(v))

    # Create the spectral norm helper
    sn = _SpectralNorm(name, n_power_iterations, dim, eps)

    # Store reference for removal
    if not hasattr(module, "_spectral_norm_hooks"):
        module._spectral_norm_hooks = {}

    # Register pre-forward hook
    handle = module.register_forward_pre_hook(sn.apply)
    module._spectral_norm_hooks[name] = (handle, sn)

    # Compute initial normalized weight
    sn.apply(module, None)

    return module


def weight_norm(
    module: Module,
    name: str = "weight",
    dim: int = 0,
) -> Module:
    """
    Apply weight normalization to a parameter in the given module.

    Weight normalization is a reparametrization that decouples the magnitude
    of weight vectors from their direction:
        weight = g * (v / ||v||)

    This can accelerate convergence of stochastic gradient descent.

    Args:
        module: Module containing the parameter to apply weight norm to
        name: Name of the weight parameter (default: 'weight')
        dim: Dimension over which to compute the norm (default: 0).
             For 2D weights (Linear), dim=0 normalizes each output neuron.
             For 4D weights (Conv2d), dim=0 normalizes each output channel.

    Returns:
        The module with weight normalization applied

    Example:
        >>> layer = nn.Linear(20, 40)
        >>> layer = weight_norm(layer)
        >>> # Now layer.weight is computed from layer.weight_v and layer.weight_g
    """
    weight = getattr(module, name)
    if weight is None:
        raise ValueError(f"Module {module} has no parameter or buffer named {name}")

    weight_array = weight._mlx_array
    ndim = weight_array.ndim

    # Normalize dim
    if dim < 0:
        dim = ndim + dim

    # Compute initial norm
    axes = [i for i in range(ndim) if i != dim]
    if axes:
        norm = mx.sqrt(mx.sum(weight_array**2, axis=axes, keepdims=True))
    else:
        norm = mx.sqrt(mx.sum(weight_array**2, keepdims=True))

    # g is the magnitude (shape matches weight along dim, 1 elsewhere)
    g = norm

    # v is the direction (same shape as weight)
    v = weight_array

    # Remove original weight and register new parameters
    delattr(module, name)
    module.register_parameter(name + "_g", Parameter(Tensor._from_mlx_array(g)))
    module.register_parameter(name + "_v", Parameter(Tensor._from_mlx_array(v)))

    # Create the weight norm helper
    wn = _WeightNorm(name, dim)

    # Store reference for removal
    if not hasattr(module, "_weight_norm_hooks"):
        module._weight_norm_hooks = {}

    # Register pre-forward hook
    handle = module.register_forward_pre_hook(wn.apply)
    module._weight_norm_hooks[name] = (handle, wn)

    # Compute initial normalized weight
    wn.apply(module, None)

    return module


def remove_spectral_norm(module: Module, name: str = "weight") -> Module:
    """
    Remove spectral normalization from a module.

    Args:
        module: Module to remove spectral normalization from
        name: Name of the weight parameter (default: 'weight')

    Returns:
        The module with spectral normalization removed
    """
    if not hasattr(module, "_spectral_norm_hooks") or name not in module._spectral_norm_hooks:
        raise ValueError(f"spectral_norm not applied to {name}")

    # Get the current weight value
    weight = getattr(module, name)

    # Remove hook
    handle, _ = module._spectral_norm_hooks[name]
    handle.remove()
    del module._spectral_norm_hooks[name]

    # Remove the split parameters
    delattr(module, name + "_orig")
    if hasattr(module, "_buffers"):
        if name + "_u" in module._buffers:
            del module._buffers[name + "_u"]
        if name + "_v" in module._buffers:
            del module._buffers[name + "_v"]

    # Restore original weight
    module.register_parameter(name, Parameter(weight))

    return module


def remove_weight_norm(module: Module, name: str = "weight") -> Module:
    """
    Remove weight normalization from a module.

    Args:
        module: Module to remove weight normalization from
        name: Name of the weight parameter (default: 'weight')

    Returns:
        The module with weight normalization removed
    """
    if not hasattr(module, "_weight_norm_hooks") or name not in module._weight_norm_hooks:
        raise ValueError(f"weight_norm not applied to {name}")

    # Get the current weight value
    weight = getattr(module, name)

    # Remove hook
    handle, _ = module._weight_norm_hooks[name]
    handle.remove()
    del module._weight_norm_hooks[name]

    # Remove the split parameters
    delattr(module, name + "_g")
    delattr(module, name + "_v")

    # Restore combined weight
    module.register_parameter(name, Parameter(weight))

    return module


__all__ = [
    "orthogonal",
    "spectral_norm",
    "weight_norm",
    "remove_spectral_norm",
    "remove_weight_norm",
]
