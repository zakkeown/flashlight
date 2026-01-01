"""
Tensor creation factory functions.

Provides PyTorch-compatible tensor creation functions that wrap MLX operations.

Reference:
- pytorch-mlx-porting-docs/02-OPERATORS/operator-reference/tensor-creation.md
- pytorch-mlx-porting-docs/08-PORTING-GUIDE/mlx-mapping.md (lines 96-135)
"""

from typing import Union, Optional, Tuple, Sequence

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

from .tensor import Tensor
from .dtype import DType, get_dtype, get_default_dtype
from .device import Device, get_default_device


# ==================== Helper Function ====================

def _parse_shape(*size: Union[int, Tuple[int, ...], Sequence[int]]) -> Tuple[int, ...]:
    """
    Parse shape specification (handles both *args and tuple).

    Args:
        *size: Either individual dimensions or a tuple/list of dimensions

    Returns:
        Tuple of integers representing shape
    """
    if len(size) == 1:
        if isinstance(size[0], (tuple, list)):
            return tuple(size[0])
        elif isinstance(size[0], int):
            return (size[0],)
        else:
            raise TypeError(f"Invalid shape type: {type(size[0])}")
    else:
        return tuple(size)


# ==================== Constant Fill Operations ====================

def zeros(
    *size: Union[int, Tuple[int, ...]],
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor filled with zeros.

    Args:
        *size: Shape of the tensor (e.g., 3, 4 or (3, 4))
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with zeros

    Example:
        >>> zeros(3, 4)
        >>> zeros((2, 3, 4), dtype='int32')
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    shape = _parse_shape(*size)
    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    mlx_array = mx.zeros(shape, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def ones(
    *size: Union[int, Tuple[int, ...]],
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor filled with ones.

    Args:
        *size: Shape of the tensor
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with ones

    Example:
        >>> ones(3, 4)
        >>> ones((5,), dtype='int64')
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    shape = _parse_shape(*size)
    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    mlx_array = mx.ones(shape, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def full(
    size: Union[Tuple[int, ...], Sequence[int]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor filled with a specified value.

    Args:
        size: Shape of the tensor (tuple)
        fill_value: The value to fill the tensor with
        dtype: Data type (default: inferred from fill_value)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor filled with fill_value

    Example:
        >>> full((3, 4), 7)
        >>> full((2, 3), 3.14, dtype='float16')
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    shape = tuple(size)
    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    mlx_array = mx.full(shape, fill_value, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def empty(
    *size: Union[int, Tuple[int, ...]],
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create an uninitialized tensor (returns zeros in MLX).

    Note: MLX doesn't support truly uninitialized arrays for safety.
          This function returns a zero-filled tensor for compatibility.

    Args:
        *size: Shape of the tensor
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor (zero-filled, not truly empty)

    Example:
        >>> empty(3, 4)  # Returns zeros in MLX
    """
    import warnings
    warnings.warn(
        "MLX does not support uninitialized arrays. "
        "Returning zero-filled tensor instead.",
        UserWarning,
        stacklevel=2
    )
    return zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)


# ==================== -like Variants ====================

def zeros_like(
    input: Tensor,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor of zeros with the same shape as input."""
    dtype = get_dtype(dtype) if dtype else input.dtype
    return zeros(*input.shape, dtype=dtype, device=device, requires_grad=requires_grad)


def ones_like(
    input: Tensor,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor of ones with the same shape as input."""
    dtype = get_dtype(dtype) if dtype else input.dtype
    return ones(*input.shape, dtype=dtype, device=device, requires_grad=requires_grad)


def full_like(
    input: Tensor,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with fill_value, with the same shape as input."""
    dtype = get_dtype(dtype) if dtype else input.dtype
    return full(input.shape, fill_value, dtype=dtype, device=device, requires_grad=requires_grad)


def empty_like(
    input: Tensor,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create an uninitialized tensor (zeros in MLX) with the same shape as input."""
    dtype = get_dtype(dtype) if dtype else input.dtype
    return empty(*input.shape, dtype=dtype, device=device, requires_grad=requires_grad)


# ==================== Sequence Operations ====================

def arange(
    start: Union[int, float] = 0,
    end: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a 1-D tensor with evenly spaced values.

    Args:
        start: Starting value (or end if only one argument)
        end: End value (exclusive)
        step: Step size
        dtype: Data type
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        1-D tensor with range values

    Example:
        >>> arange(10)  # 0 to 9
        >>> arange(2, 10, 2)  # 2, 4, 6, 8
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Handle single argument case: arange(10) -> arange(0, 10, 1)
    if end is None:
        end = start
        start = 0

    if dtype is None:
        # Infer dtype from inputs
        if isinstance(start, float) or isinstance(end, float) or isinstance(step, float):
            dtype = get_default_dtype()
        else:
            from . import dtype as dtype_module
            dtype = dtype_module.int32
    else:
        dtype = get_dtype(dtype)

    mlx_array = mx.arange(start, end, step, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def linspace(
    start: Union[int, float],
    end: Union[int, float],
    steps: int = 100,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a 1-D tensor with evenly spaced values (inclusive).

    Args:
        start: Starting value
        end: End value (inclusive)
        steps: Number of values to generate
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        1-D tensor with linearly spaced values

    Example:
        >>> linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    dtype = get_dtype(dtype) if dtype else get_default_dtype()
    mlx_array = mx.linspace(start, end, steps, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def logspace(
    start: Union[int, float],
    end: Union[int, float],
    steps: int = 100,
    base: float = 10.0,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a 1-D tensor with logarithmically spaced values.

    Values are base^linspace(start, end, steps).

    Args:
        start: Starting exponent
        end: End exponent
        steps: Number of values
        base: Base of the logarithm
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        1-D tensor with logarithmically spaced values

    Example:
        >>> logspace(0, 3, 4)  # [10^0, 10^1, 10^2, 10^3] = [1, 10, 100, 1000]
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    # MLX doesn't have logspace, so implement it
    linear = mx.linspace(start, end, steps, dtype=dtype._mlx_dtype)
    mlx_array = mx.power(base, linear)

    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


# ==================== Identity/Diagonal Operations ====================

def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a 2-D identity matrix.

    Args:
        n: Number of rows
        m: Number of columns (default: n)
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        2-D identity tensor

    Example:
        >>> eye(3)  # 3x3 identity
        >>> eye(3, 5)  # 3x5 matrix with 1s on diagonal
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    if m is None:
        m = n

    dtype = get_dtype(dtype) if dtype else get_default_dtype()
    mlx_array = mx.eye(n, m, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


# ==================== Random Operations ====================

def randn(
    *size: Union[int, Tuple[int, ...]],
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with values from standard normal distribution N(0, 1).

    Args:
        *size: Shape of the tensor
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random normal values

    Example:
        >>> randn(3, 4)
        >>> randn((2, 3, 4), dtype='float16')
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    shape = _parse_shape(*size)
    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    mlx_array = mx.random.normal(shape, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def rand(
    *size: Union[int, Tuple[int, ...]],
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with values from uniform distribution [0, 1).

    Args:
        *size: Shape of the tensor
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random uniform values

    Example:
        >>> rand(3, 4)
        >>> rand((5,), dtype='float32')
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    shape = _parse_shape(*size)
    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    mlx_array = mx.random.uniform(shape=shape, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def randint(
    low: int,
    high: Optional[int] = None,
    size: Union[Tuple[int, ...], Sequence[int]] = (),
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with random integers.

    Args:
        low: Lowest integer (or high if high is None)
        high: Highest integer (exclusive)
        size: Shape of the tensor
        dtype: Data type (default: int64)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random integers

    Example:
        >>> randint(0, 10, (3, 4))  # Random integers in [0, 10)
        >>> randint(10, size=(5,))  # Random integers in [0, 10)
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Handle PyTorch API: randint(high, size) vs randint(low, high, size)
    if high is None:
        high = low
        low = 0

    if dtype is None:
        from . import dtype as dtype_module
        dtype = dtype_module.int64
    else:
        dtype = get_dtype(dtype)

    shape = tuple(size) if size else ()
    mlx_array = mx.random.randint(low, high, shape, dtype=dtype._mlx_dtype)
    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


# ==================== From Existing Data ====================

def tensor(
    data,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor from data (list, numpy array, etc.).

    Args:
        data: Input data
        dtype: Data type
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        New tensor

    Example:
        >>> tensor([1, 2, 3])
        >>> tensor([[1, 2], [3, 4]], dtype='float32')
    """
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def from_numpy(
    ndarray,
    *,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor from a NumPy array.

    Args:
        ndarray: NumPy array
        requires_grad: Whether to track gradients

    Returns:
        Tensor with data from NumPy array

    Example:
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3])
        >>> t = from_numpy(arr)
    """
    return Tensor(ndarray, requires_grad=requires_grad)


def clone(input: Tensor) -> Tensor:
    """
    Create a copy of the tensor.

    Args:
        input: Tensor to clone

    Returns:
        New tensor with copied data
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # MLX arrays are immutable, so we can just copy the reference
    # But we create a new Tensor object for proper semantics
    mlx_array = mx.array(input._mlx_array)
    return Tensor._from_mlx_array(mlx_array, requires_grad=input.requires_grad)


# ==================== Grid Operations ====================

def meshgrid(
    *tensors: Tensor,
    indexing: str = None
) -> Tuple[Tensor, ...]:
    """
    Create coordinate grids from coordinate vectors.

    Args:
        *tensors: 1-D tensors representing coordinates
        indexing: 'xy' for Cartesian, 'ij' for matrix indexing (default: 'ij')

    Returns:
        Tuple of tensors with shape (N1, N2, ...) where Ni is len(tensors[i])

    Example:
        >>> x = mlx_compat.arange(3)
        >>> y = mlx_compat.arange(4)
        >>> X, Y = mlx_compat.meshgrid(x, y)
        >>> X.shape, Y.shape
        ((3, 4), (3, 4))
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    if len(tensors) == 0:
        return ()

    mlx_arrays = [t._mlx_array for t in tensors]

    # MLX meshgrid uses 'ij' indexing by default
    # PyTorch default is None (which behaves like 'ij' but warns)
    if indexing is None:
        indexing = 'ij'  # Default behavior
    grids = mx.meshgrid(*mlx_arrays, indexing=indexing)

    results = tuple(Tensor._from_mlx_array(g) for g in grids)

    # Propagate requires_grad
    if any(t.requires_grad for t in tensors):
        for r in results:
            r.requires_grad = True

    return results


def cartesian_prod(*tensors: Tensor) -> Tensor:
    """
    Compute the Cartesian product of given sequence of tensors.

    Args:
        *tensors: 1-D tensors

    Returns:
        2-D tensor with Cartesian product
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    if len(tensors) == 0:
        return Tensor._from_mlx_array(mx.array([]))

    if len(tensors) == 1:
        return tensors[0].unsqueeze(1)

    grids = meshgrid(*tensors, indexing='ij')
    # Stack and reshape to (N, len(tensors))
    stacked = mx.stack([g._mlx_array.reshape(-1) for g in grids], axis=1)
    return Tensor._from_mlx_array(stacked)


# ==================== Random -like Variants ====================

def rand_like(
    input: Tensor,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with uniform random values [0, 1) with same shape as input.

    Args:
        input: Tensor whose shape to use
        dtype: Data type (default: same as input)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random uniform values
    """
    dtype = get_dtype(dtype) if dtype else input.dtype
    return rand(*input.shape, dtype=dtype, device=device, requires_grad=requires_grad)


def randn_like(
    input: Tensor,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with standard normal random values with same shape as input.

    Args:
        input: Tensor whose shape to use
        dtype: Data type (default: same as input)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random normal values
    """
    dtype = get_dtype(dtype) if dtype else input.dtype
    return randn(*input.shape, dtype=dtype, device=device, requires_grad=requires_grad)


def randint_like(
    input: Tensor,
    low: int,
    high: Optional[int] = None,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with random integers with same shape as input.

    Args:
        input: Tensor whose shape to use
        low: Lowest integer (or high if high is None)
        high: Highest integer (exclusive)
        dtype: Data type (default: int64)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with random integers
    """
    if dtype is None:
        from . import dtype as dtype_module
        dtype = dtype_module.int64
    else:
        dtype = get_dtype(dtype)
    return randint(low, high, input.shape, dtype=dtype, device=device, requires_grad=requires_grad)


def randperm(
    n: int,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a random permutation of integers from 0 to n-1.

    Args:
        n: Upper bound (exclusive)
        dtype: Data type (default: int64)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        1-D tensor with random permutation
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    if dtype is None:
        from . import dtype as dtype_module
        dtype = dtype_module.int64
    else:
        dtype = get_dtype(dtype)

    # Create range and shuffle using random.permutation
    indices = mx.array(list(range(n)), dtype=dtype._mlx_dtype)
    # Use Fisher-Yates shuffle via random sampling
    perm = mx.random.permutation(n)
    mlx_array = mx.take(indices, perm)

    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


# ==================== Additional Creation Functions ====================

def as_tensor(
    data,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
) -> Tensor:
    """
    Convert data to a Tensor, sharing memory when possible.

    Unlike tensor(), this tries to avoid copying data when the input
    is already an array-like with the same dtype.

    Args:
        data: Input data (list, numpy array, tensor, etc.)
        dtype: Data type
        device: Device (compatibility)

    Returns:
        Tensor containing the data
    """
    # If already a Tensor with matching dtype, return as-is
    if isinstance(data, Tensor):
        if dtype is None:
            return data
        target_dtype = get_dtype(dtype)
        if data.dtype == target_dtype:
            return data
        # Need to convert dtype
        return Tensor(data._mlx_array, dtype=dtype)

    return tensor(data, dtype=dtype, device=device)


def scalar_tensor(
    value: Union[int, float],
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor containing a single scalar value.

    Args:
        value: The scalar value
        dtype: Data type
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        0-D tensor containing the scalar
    """
    return tensor(value, dtype=dtype, device=device, requires_grad=requires_grad)


def normal(
    mean: Union[float, Tensor] = 0.0,
    std: Union[float, Tensor] = 1.0,
    size: Optional[Union[Tuple[int, ...], Sequence[int]]] = None,
    *,
    dtype: Optional[Union[DType, str]] = None,
    device: Optional[Union[Device, str]] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor with values from a normal distribution N(mean, std).

    Args:
        mean: Mean of the distribution (scalar or tensor)
        std: Standard deviation of the distribution (scalar or tensor)
        size: Shape of the output tensor (required if mean/std are scalars)
        dtype: Data type (default: float32)
        device: Device (compatibility)
        requires_grad: Whether to track gradients

    Returns:
        Tensor with normally distributed values
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Determine shape
    if size is not None:
        shape = tuple(size)
    elif isinstance(mean, Tensor):
        shape = mean.shape
    elif isinstance(std, Tensor):
        shape = std.shape
    else:
        raise ValueError("size must be specified when mean and std are scalars")

    dtype = get_dtype(dtype) if dtype else get_default_dtype()

    # Generate standard normal, then scale and shift
    z = mx.random.normal(shape, dtype=dtype._mlx_dtype)

    # Apply mean and std
    if isinstance(mean, Tensor):
        mean_val = mean._mlx_array
    else:
        mean_val = mean

    if isinstance(std, Tensor):
        std_val = std._mlx_array
    else:
        std_val = std

    mlx_array = z * std_val + mean_val

    return Tensor._from_mlx_array(mlx_array, requires_grad=requires_grad)


def bernoulli(
    input: Tensor,
    *,
    generator=None,
) -> Tensor:
    """
    Draw binary random numbers (0 or 1) from Bernoulli distributions.

    Each element of the output is drawn from a Bernoulli distribution
    with probability given by the corresponding element of input.

    Args:
        input: Tensor of probabilities (values in [0, 1])
        generator: Random number generator (ignored, for compatibility)

    Returns:
        Tensor with values 0 or 1
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Generate uniform random and compare to probabilities
    probs = input._mlx_array
    uniform = mx.random.uniform(shape=probs.shape)
    result = mx.less(uniform, probs).astype(probs.dtype)

    return Tensor._from_mlx_array(result)


def multinomial(
    input: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator=None,
) -> Tensor:
    """
    Draw samples from multinomial distributions.

    Args:
        input: Tensor of probabilities (can be unnormalized)
        num_samples: Number of samples to draw
        replacement: Whether to draw with replacement
        generator: Random number generator (ignored, for compatibility)

    Returns:
        Tensor of sampled indices
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    probs = input._mlx_array

    # Handle batched input
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, num_classes = probs.shape

    # Normalize probabilities
    probs = probs / mx.sum(probs, axis=-1, keepdims=True)

    results = []
    for b in range(batch_size):
        batch_probs = probs[b]

        if replacement:
            # With replacement: use categorical sampling
            # Implement via inverse CDF sampling
            cumsum = mx.cumsum(batch_probs)
            uniform = mx.random.uniform(shape=(num_samples,))
            # Binary search for each sample
            samples = mx.sum(mx.expand_dims(uniform, 1) >= cumsum, axis=1)
            samples = mx.clip(samples, 0, num_classes - 1)
        else:
            # Without replacement: sequential sampling
            if num_samples > num_classes:
                raise ValueError(
                    f"Cannot sample {num_samples} without replacement from {num_classes} classes"
                )
            # Use Gumbel-max trick for sampling without replacement
            gumbel = -mx.log(-mx.log(mx.random.uniform(shape=(num_classes,))))
            log_probs = mx.log(batch_probs + 1e-10)
            perturbed = log_probs + gumbel
            samples = mx.argsort(perturbed, axis=-1)[-num_samples:][::-1]

        results.append(samples)

    result = mx.stack(results, axis=0)

    if squeeze_output:
        result = result.squeeze(0)

    # Convert to int64
    from . import dtype as dtype_module
    result = result.astype(dtype_module.int64._mlx_dtype)

    return Tensor._from_mlx_array(result)


def poisson(
    input: Tensor,
    *,
    generator=None,
) -> Tensor:
    """
    Draw samples from Poisson distributions.

    Args:
        input: Tensor of rate parameters (lambda)
        generator: Random number generator (ignored, for compatibility)

    Returns:
        Tensor with Poisson-distributed samples
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not available")

    # Pure MLX implementation of Poisson sampling
    # Uses inverse transform method with rejection for large lambda
    rates = input._mlx_array.astype(mx.float32)  # Ensure float type

    # For small lambda (< 30), use Knuth's algorithm (inverse transform)
    # For large lambda (>= 30), use normal approximation
    threshold = 30.0

    # Handle both small and large lambda cases
    small_lambda_mask = rates < threshold

    # Initialize result
    result = mx.zeros(rates.shape, dtype=mx.int32)

    # Small lambda: Knuth's algorithm
    # P(X >= k) = exp(-lambda) * sum_{i=0}^{k-1} lambda^i / i!
    # Generate U ~ Uniform(0,1), find smallest k such that prod_{i=1}^{k} U_i < exp(-lambda)
    if mx.any(small_lambda_mask):
        small_rates = mx.where(small_lambda_mask, rates, mx.ones(rates.shape, dtype=mx.float32))
        L = mx.exp(-small_rates)
        k = mx.zeros(small_rates.shape, dtype=mx.int32)
        p = mx.ones(small_rates.shape, dtype=mx.float32)

        # Iterate up to a maximum (for numerical stability)
        max_iter = 200
        for _ in range(max_iter):
            u = mx.random.uniform(shape=small_rates.shape)
            p = p * u
            # Increment k where p >= L
            increment = mx.where(p >= L, mx.ones(k.shape, dtype=mx.int32), mx.zeros(k.shape, dtype=mx.int32))
            k = k + increment
            # Check if all done
            if mx.all(p < L):
                break

        result = mx.where(small_lambda_mask, k, result)

    # Large lambda: Normal approximation with rounding
    # Poisson(lambda) ~ N(lambda, lambda) for large lambda
    if mx.any(~small_lambda_mask):
        large_rates = mx.where(~small_lambda_mask, rates, mx.ones(rates.shape, dtype=mx.float32))
        # Sample from normal distribution
        normal_samples = mx.random.normal(shape=large_rates.shape)
        # Scale and shift: X ~ lambda + sqrt(lambda) * Z
        approx_samples = large_rates + mx.sqrt(large_rates) * normal_samples
        # Round to nearest non-negative integer
        approx_samples = mx.maximum(mx.round(approx_samples), mx.zeros(approx_samples.shape, dtype=mx.float32))
        result = mx.where(~small_lambda_mask, approx_samples.astype(mx.int32), result)

    return Tensor._from_mlx_array(result)


__all__ = [
    # Constant fill
    'zeros', 'ones', 'full', 'empty',
    # -like variants
    'zeros_like', 'ones_like', 'full_like', 'empty_like',
    'rand_like', 'randn_like', 'randint_like',
    # Sequences
    'arange', 'linspace', 'logspace',
    # Identity
    'eye',
    # Random
    'randn', 'rand', 'randint', 'randperm',
    'normal', 'bernoulli', 'multinomial', 'poisson',
    # From data
    'tensor', 'from_numpy', 'clone',
    'as_tensor', 'scalar_tensor',
    # Grid operations
    'meshgrid', 'cartesian_prod',
]
