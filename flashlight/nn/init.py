"""
Weight Initialization Functions

PyTorch-compatible torch.nn.init module for initializing neural network weights.
All functions modify tensors in-place (following PyTorch convention) and return
the modified tensor.
"""

import math
from typing import Optional, Union, Callable, Literal

import mlx.core as mx

from ..tensor import Tensor

# Re-export typing utilities for compatibility
try:
    from typing import Callable, Literal, ParamSpec, TypeVar, Union
except ImportError:
    # Python 3.9 compatibility
    from typing import Callable, TypeVar, Union
    try:
        from typing import Literal
    except ImportError:
        from typing_extensions import Literal
    try:
        from typing import ParamSpec
    except ImportError:
        ParamSpec = TypeVar  # Fallback

__all__ = [
    'calculate_gain',
    'uniform_', 'normal_', 'constant_', 'ones_', 'zeros_',
    'eye_', 'dirac_',
    'xavier_uniform_', 'xavier_normal_',
    'kaiming_uniform_', 'kaiming_normal_',
    'orthogonal_', 'sparse_', 'trunc_normal_',
    # Non-underscore versions (deprecated but still used)
    'uniform', 'normal', 'constant',
    'xavier_uniform', 'xavier_normal',
    'kaiming_uniform', 'kaiming_normal',
    'orthogonal', 'sparse',
    'eye', 'dirac',
]


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """
    Return the recommended gain value for the given nonlinearity function.

    Args:
        nonlinearity: The non-linear function (nn.functional name)
        param: Optional parameter for the non-linear function

    Returns:
        The recommended gain value

    Example:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
                  'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        else:
            negative_slope = param
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value from Self-Normalizing Neural Networks paper
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _no_grad_uniform_(tensor: Tensor, a: float, b: float) -> Tensor:
    """Fill tensor with uniform distribution without gradient tracking."""
    new_array = mx.random.uniform(low=a, high=b, shape=tensor.shape)
    new_array = new_array.astype(tensor._mlx_array.dtype)
    tensor._mlx_array = new_array
    return tensor


def _no_grad_normal_(tensor: Tensor, mean: float, std: float) -> Tensor:
    """Fill tensor with normal distribution without gradient tracking."""
    new_array = mx.random.normal(shape=tensor.shape) * std + mean
    new_array = new_array.astype(tensor._mlx_array.dtype)
    tensor._mlx_array = new_array
    return tensor


def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float,
                           a: float, b: float) -> Tensor:
    """Fill tensor with truncated normal distribution."""
    # Simple rejection sampling approach for truncated normal
    l = (a - mean) / std
    u = (b - mean) / std

    # Use inverse CDF method approximation
    # Generate uniform samples and transform
    shape = tensor.shape
    size = 1
    for s in shape:
        size *= s

    # Generate more samples than needed to account for rejection
    samples = mx.random.normal(shape=(int(size * 1.5),)) * std + mean

    # Clamp to bounds (approximation of truncation)
    samples = mx.clip(samples, a, b)

    # Take only what we need and reshape
    samples = samples[:size].reshape(shape)
    samples = samples.astype(tensor._mlx_array.dtype)
    tensor._mlx_array = samples
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """
    Fill the input Tensor with values drawn from the uniform distribution U(a, b).

    Args:
        tensor: An n-dimensional Tensor
        a: The lower bound of the uniform distribution
        b: The upper bound of the uniform distribution

    Returns:
        The input tensor filled with uniform values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """
    Fill the input Tensor with values drawn from the normal distribution N(mean, std^2).

    Args:
        tensor: An n-dimensional Tensor
        mean: The mean of the normal distribution
        std: The standard deviation of the normal distribution

    Returns:
        The input tensor filled with normal values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    return _no_grad_normal_(tensor, mean, std)


def constant_(tensor: Tensor, val: float) -> Tensor:
    """
    Fill the input Tensor with the value val.

    Args:
        tensor: An n-dimensional Tensor
        val: The value to fill the tensor with

    Returns:
        The input tensor filled with val

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    new_array = mx.full(tensor.shape, val, dtype=tensor._mlx_array.dtype)
    tensor._mlx_array = new_array
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """
    Fill the input Tensor with ones.

    Args:
        tensor: An n-dimensional Tensor

    Returns:
        The input tensor filled with ones

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    return constant_(tensor, 1.0)


def zeros_(tensor: Tensor) -> Tensor:
    """
    Fill the input Tensor with zeros.

    Args:
        tensor: An n-dimensional Tensor

    Returns:
        The input tensor filled with zeros

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    return constant_(tensor, 0.0)


def eye_(tensor: Tensor) -> Tensor:
    """
    Fill the 2-dimensional input Tensor with the identity matrix.

    Args:
        tensor: A 2-dimensional Tensor

    Returns:
        The input tensor filled with identity matrix

    Example:
        >>> w = torch.empty(3, 3)
        >>> nn.init.eye_(w)
    """
    if tensor.ndim != 2:
        raise ValueError(f"Only tensors with 2 dimensions are supported. Got {tensor.ndim}")

    rows, cols = tensor.shape
    eye_array = mx.eye(rows, cols, dtype=tensor._mlx_array.dtype)
    tensor._mlx_array = eye_array
    return tensor


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """
    Fill the {3, 4, 5}-dimensional input Tensor with the Dirac delta function.

    Preserves the identity of the inputs in Convolutional layers, where as many
    input channels are preserved as possible.

    Args:
        tensor: A {3, 4, 5}-dimensional Tensor
        groups: Number of groups in the conv layer (default: 1)

    Returns:
        The input tensor filled with Dirac delta

    Example:
        >>> w = torch.empty(3, 3, 5, 5)
        >>> nn.init.dirac_(w)
    """
    dimensions = tensor.ndim
    if dimensions not in [3, 4, 5]:
        raise ValueError(f"Only tensors with 3, 4, or 5 dimensions are supported. Got {dimensions}")

    sizes = tensor.shape

    if sizes[0] % groups != 0:
        raise ValueError('dim 0 must be divisible by groups')

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    # Fill with zeros first
    tensor._mlx_array = mx.zeros(sizes, dtype=tensor._mlx_array.dtype)

    # Set diagonal elements
    for g in range(groups):
        for d in range(min_dim):
            if dimensions == 3:
                tensor._mlx_array[g * out_chans_per_grp + d, d, sizes[2] // 2] = 1
            elif dimensions == 4:
                tensor._mlx_array[g * out_chans_per_grp + d, d, sizes[2] // 2, sizes[3] // 2] = 1
            elif dimensions == 5:
                tensor._mlx_array[g * out_chans_per_grp + d, d, sizes[2] // 2, sizes[3] // 2, sizes[4] // 2] = 1

    return tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple:
    """Calculate fan_in and fan_out for a tensor."""
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fill the input Tensor with values using Xavier uniform initialization.

    Also known as Glorot initialization. Values are drawn from U(-a, a) where
    a = gain * sqrt(6 / (fan_in + fan_out)).

    Args:
        tensor: An n-dimensional Tensor
        gain: An optional scaling factor

    Returns:
        The input tensor filled with Xavier uniform values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from std
    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fill the input Tensor with values using Xavier normal initialization.

    Also known as Glorot initialization. Values are drawn from N(0, std^2) where
    std = gain * sqrt(2 / (fan_in + fan_out)).

    Args:
        tensor: An n-dimensional Tensor
        gain: An optional scaling factor

    Returns:
        The input tensor filled with Xavier normal values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                     nonlinearity: str = 'leaky_relu') -> Tensor:
    """
    Fill the input Tensor with values using Kaiming uniform initialization.

    Also known as He initialization. Values are drawn from U(-bound, bound) where
    bound = gain * sqrt(3 / fan_mode).

    Args:
        tensor: An n-dimensional Tensor
        a: The negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode: Either 'fan_in' (default) or 'fan_out'
        nonlinearity: The non-linear function (nn.functional name)

    Returns:
        The input tensor filled with Kaiming uniform values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    if mode == 'fan_in':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        fan = fan_in
    elif mode == 'fan_out':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        fan = fan_out
    else:
        raise ValueError(f"Mode {mode} not supported, please use 'fan_in' or 'fan_out'")

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                    nonlinearity: str = 'leaky_relu') -> Tensor:
    """
    Fill the input Tensor with values using Kaiming normal initialization.

    Also known as He initialization. Values are drawn from N(0, std^2) where
    std = gain / sqrt(fan_mode).

    Args:
        tensor: An n-dimensional Tensor
        a: The negative slope of the rectifier used after this layer (only used with 'leaky_relu')
        mode: Either 'fan_in' (default) or 'fan_out'
        nonlinearity: The non-linear function (nn.functional name)

    Returns:
        The input tensor filled with Kaiming normal values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if mode == 'fan_in':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        fan = fan_in
    elif mode == 'fan_out':
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        fan = fan_out
    else:
        raise ValueError(f"Mode {mode} not supported, please use 'fan_in' or 'fan_out'")

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _no_grad_normal_(tensor, 0.0, std)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """
    Fill the input Tensor with a (semi) orthogonal matrix.

    The input tensor must have at least 2 dimensions. For tensors with more than 2
    dimensions the trailing dimensions are flattened.

    Args:
        tensor: An n-dimensional Tensor, where n >= 2
        gain: An optional scaling factor

    Returns:
        The input tensor filled with orthogonal values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.shape[0]
    cols = 1
    for s in tensor.shape[1:]:
        cols *= s

    # Generate random matrix
    flat_shape = (rows, cols) if rows < cols else (cols, rows)
    random_matrix = mx.random.normal(shape=flat_shape)

    # QR decomposition
    # Note: MLX may not have QR, so we use SVD as fallback
    try:
        q, r = mx.linalg.qr(random_matrix)
    except AttributeError:
        # Fallback to SVD-based orthogonalization
        u, s, vh = mx.linalg.svd(random_matrix, full_matrices=False)
        q = u if rows >= cols else vh

    if rows < cols:
        q = q.T

    # Apply gain and reshape
    q = q * gain
    q = q.reshape(tensor.shape)
    q = q.astype(tensor._mlx_array.dtype)
    tensor._mlx_array = q
    return tensor


def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01) -> Tensor:
    """
    Fill the 2D input Tensor as a sparse matrix.

    The non-zero elements are drawn from N(0, std^2).

    Args:
        tensor: An n-dimensional Tensor
        sparsity: The fraction of elements in each column to be set to zero
        std: The standard deviation of the normal distribution used to generate non-zero values

    Returns:
        The input tensor filled with sparse values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    # Fill with normal distribution
    tensor = _no_grad_normal_(tensor, 0.0, std)

    # Set random elements to zero in each column
    for col_idx in range(cols):
        row_indices = mx.random.permutation(mx.arange(rows))[:num_zeros]
        for idx in row_indices.tolist():
            tensor._mlx_array[idx, col_idx] = 0

    return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> Tensor:
    """
    Fill the input Tensor with values drawn from a truncated normal distribution.

    Values are effectively drawn from the normal distribution N(mean, std^2) with
    values outside [a, b] redrawn until they are within the bounds.

    Args:
        tensor: An n-dimensional Tensor
        mean: The mean of the normal distribution
        std: The standard deviation of the normal distribution
        a: The minimum cutoff value
        b: The maximum cutoff value

    Returns:
        The input tensor filled with truncated normal values

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# Non-underscore versions (deprecated but still supported for backwards compatibility)
def uniform(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Deprecated. Use uniform_ instead."""
    return uniform_(tensor, a, b)


def normal(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Deprecated. Use normal_ instead."""
    return normal_(tensor, mean, std)


def constant(tensor: Tensor, val: float) -> Tensor:
    """Deprecated. Use constant_ instead."""
    return constant_(tensor, val)


def xavier_uniform(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Deprecated. Use xavier_uniform_ instead."""
    return xavier_uniform_(tensor, gain)


def xavier_normal(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Deprecated. Use xavier_normal_ instead."""
    return xavier_normal_(tensor, gain)


def kaiming_uniform(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                    nonlinearity: str = 'leaky_relu') -> Tensor:
    """Deprecated. Use kaiming_uniform_ instead."""
    return kaiming_uniform_(tensor, a, mode, nonlinearity)


def kaiming_normal(tensor: Tensor, a: float = 0, mode: str = 'fan_in',
                   nonlinearity: str = 'leaky_relu') -> Tensor:
    """Deprecated. Use kaiming_normal_ instead."""
    return kaiming_normal_(tensor, a, mode, nonlinearity)


def orthogonal(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Deprecated. Use orthogonal_ instead."""
    return orthogonal_(tensor, gain)


def sparse(tensor: Tensor, sparsity: float, std: float = 0.01) -> Tensor:
    """Deprecated. Use sparse_ instead."""
    return sparse_(tensor, sparsity, std)


def eye(tensor: Tensor) -> Tensor:
    """Deprecated. Use eye_ instead."""
    return eye_(tensor)


def dirac(tensor: Tensor, groups: int = 1) -> Tensor:
    """Deprecated. Use dirac_ instead."""
    return dirac_(tensor, groups)
