"""
Data type mappings between PyTorch and MLX.

Maps PyTorch dtypes to MLX dtypes, handling incompatibilities
(e.g., MLX doesn't support float64).

Reference:
- pytorch-mlx-porting-docs/01-FOUNDATIONS/type-system.md
- pytorch-mlx-porting-docs/08-PORTING-GUIDE/mlx-mapping.md (lines 137-150)
"""

import warnings
from typing import Optional

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class DType:
    """
    PyTorch-compatible dtype wrapper for MLX dtypes.

    Provides a PyTorch-like interface while mapping to MLX dtypes internally.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a dtype.

        Args:
            mlx_dtype: The underlying MLX dtype
            name: String name (e.g., 'float32')
            itemsize: Size in bytes
            is_floating: Whether this is a floating point type
            is_complex: Whether this is a complex type
            is_signed: Whether this is a signed type
        """
        # Parse arguments - support both positional and keyword args
        mlx_dtype = args[0] if args else kwargs.get('mlx_dtype')
        name = args[1] if len(args) > 1 else kwargs.get('name')
        itemsize = args[2] if len(args) > 2 else kwargs.get('itemsize')
        is_floating = args[3] if len(args) > 3 else kwargs.get('is_floating', False)
        is_complex = args[4] if len(args) > 4 else kwargs.get('is_complex', False)
        is_signed = args[5] if len(args) > 5 else kwargs.get('is_signed', True)

        self._mlx_dtype = mlx_dtype
        self.name = name
        self.itemsize = itemsize
        self.is_floating_point = is_floating
        self.is_complex = is_complex
        self.is_signed = is_signed

    def __repr__(self):
        return f"mlx_compat.{self.name}"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)


# Define all supported dtypes
# Reference: mlx-mapping.md lines 139-150

if MLX_AVAILABLE:
    # Floating point types
    float32 = DType(mx.float32, 'float32', 4, is_floating=True)
    float16 = DType(mx.float16, 'float16', 2, is_floating=True)
    bfloat16 = DType(mx.bfloat16, 'bfloat16', 2, is_floating=True)

    # Integer types (signed)
    int8 = DType(mx.int8, 'int8', 1)
    int16 = DType(mx.int16, 'int16', 2)
    int32 = DType(mx.int32, 'int32', 4)
    int64 = DType(mx.int64, 'int64', 8)

    # Integer types (unsigned)
    uint8 = DType(mx.uint8, 'uint8', 1, is_signed=False)
    uint16 = DType(mx.uint16, 'uint16', 2, is_signed=False)
    uint32 = DType(mx.uint32, 'uint32', 4, is_signed=False)
    uint64 = DType(mx.uint64, 'uint64', 8, is_signed=False)

    # Boolean
    bool = DType(mx.bool_, 'bool', 1, is_signed=False)

    # Complex types (if available in MLX)
    try:
        complex64 = DType(mx.complex64, 'complex64', 8, is_complex=True)
    except AttributeError:
        complex64 = None

    # Aliases (PyTorch compatibility)
    float = float32
    half = float16
    int = int32
    long = int64
    short = int16
    byte = int8  # Note: PyTorch byte is uint8, but we map to int8
else:
    # Stub dtypes when MLX is not available
    float32 = None
    float16 = None
    bfloat16 = None
    int8 = None
    int16 = None
    int32 = None
    int64 = None
    uint8 = None
    uint16 = None
    uint32 = None
    uint64 = None
    bool = None
    complex64 = None

    # Aliases
    float = None
    half = None
    int = None
    long = None
    short = None
    byte = None


# Unsupported PyTorch dtypes (MLX limitations)
# Reference: mlx-mapping.md line 142
class UnsupportedDType(DType):
    """Dtype that exists in PyTorch but not in MLX."""

    def __init__(self, name: str, fallback: Optional[DType], itemsize: int,
                 is_floating: bool = False, is_complex: bool = False):
        super().__init__(None, name, itemsize, is_floating, is_complex)
        self.fallback = fallback

    def _raise_unsupported(self):
        msg = f"{self.name} is not supported in MLX."
        if self.fallback:
            msg += f" Consider using {self.fallback.name} instead."
        raise TypeError(msg)


# Float64 - Not supported in MLX (use float32 as fallback)
float64 = UnsupportedDType('float64', float32, 8, is_floating=True)
double = float64  # Alias

# Complex128 - Not supported in MLX
complex128 = UnsupportedDType('complex128', complex64, 16, is_complex=True)


def _warn_unsupported(dtype_name: str, fallback_name: str):
    """Warn about unsupported dtype and suggest fallback."""
    warnings.warn(
        f"{dtype_name} is not supported in MLX. "
        f"Automatically converting to {fallback_name}. "
        f"This may result in reduced precision.",
        UserWarning,
        stacklevel=3
    )


def get_dtype(dtype) -> Optional[DType]:
    """
    Convert various dtype representations to mlx_compat.DType.

    Args:
        dtype: Can be:
            - mlx_compat.DType
            - MLX dtype (mx.float32, etc.)
            - String ('float32', 'int64', etc.)
            - None (returns default float32)

    Returns:
        Corresponding mlx_compat.DType or None
    """
    if dtype is None:
        return float32

    if isinstance(dtype, DType):
        return dtype

    # String lookup
    if isinstance(dtype, str):
        dtype_map = {
            'float32': float32, 'float': float32,
            'float64': float64, 'double': float64,
            'float16': float16, 'half': float16,
            'bfloat16': bfloat16,
            'int8': int8,
            'int16': int16, 'short': int16,
            'int32': int32, 'int': int32,
            'int64': int64, 'long': int64,
            'uint8': uint8, 'byte': uint8,
            'uint16': uint16,
            'uint32': uint32,
            'uint64': uint64,
            'bool': bool,
            'complex64': complex64,
            'complex128': complex128,
        }
        result = dtype_map.get(dtype)
        if result is None:
            raise ValueError(f"Unknown dtype: {dtype}")

        # Warn about unsupported types
        if isinstance(result, UnsupportedDType):
            if result.fallback:
                _warn_unsupported(result.name, result.fallback.name)
                return result.fallback
            else:
                result._raise_unsupported()

        return result

    # MLX dtype lookup
    if MLX_AVAILABLE:
        mlx_dtype_map = {
            mx.float32: float32,
            mx.float16: float16,
            mx.bfloat16: bfloat16,
            mx.int8: int8,
            mx.int16: int16,
            mx.int32: int32,
            mx.int64: int64,
            mx.uint8: uint8,
            mx.uint16: uint16,
            mx.uint32: uint32,
            mx.uint64: uint64,
            mx.bool_: bool,
        }
        if complex64:
            mlx_dtype_map[mx.complex64] = complex64

        if dtype in mlx_dtype_map:
            return mlx_dtype_map[dtype]

    raise ValueError(f"Cannot convert {dtype} to mlx_compat dtype")


# Default dtype management
_default_dtype = float32


def set_default_dtype(dtype):
    """
    Set the default floating point dtype.

    Args:
        dtype: New default dtype (must be a floating point type)
    """
    global _default_dtype
    dtype = get_dtype(dtype)

    if not dtype.is_floating_point:
        raise TypeError(f"Default dtype must be floating point, got {dtype}")

    _default_dtype = dtype


def get_default_dtype() -> DType:
    """Get the current default floating point dtype."""
    return _default_dtype


def torch_dtype_to_numpy(dtype):
    """
    Convert a torch/mlx_compat dtype to numpy dtype.

    Args:
        dtype: A DType or string dtype name

    Returns:
        Corresponding numpy dtype
    """
    import numpy as np

    # Handle DType objects
    if isinstance(dtype, DType):
        dtype = dtype.name

    # String lookup
    dtype_map = {
        'float32': np.float32,
        'float': np.float32,
        'float64': np.float64,
        'double': np.float64,
        'float16': np.float16,
        'half': np.float16,
        'bfloat16': np.float32,  # numpy doesn't have bfloat16, use float32
        'int8': np.int8,
        'int16': np.int16,
        'short': np.int16,
        'int32': np.int32,
        'int': np.int32,
        'int64': np.int64,
        'long': np.int64,
        'uint8': np.uint8,
        'byte': np.uint8,
        'uint16': np.uint16,
        'uint32': np.uint32,
        'uint64': np.uint64,
        'bool': np.bool_,
        'complex64': np.complex64,
        'complex128': np.complex128,
    }

    if isinstance(dtype, str):
        result = dtype_map.get(dtype)
        if result is None:
            raise ValueError(f"Unknown dtype: {dtype}")
        return result

    # If it's already a numpy dtype, return it
    if hasattr(dtype, 'name') and dtype.name in dtype_map:
        return dtype_map[dtype.name]

    raise ValueError(f"Cannot convert {dtype} to numpy dtype")


def numpy_to_mlx_dtype(np_dtype, return_raw=True, warn_on_downgrade=True):
    """
    Convert a numpy dtype to MLX dtype.

    Args:
        np_dtype: A numpy dtype
        return_raw: If True, return the raw MLX dtype (mx.float32, etc.)
                   If False, return the DType wrapper
        warn_on_downgrade: If True, emit a warning when float64 is downgraded to float32

    Returns:
        Corresponding MLX dtype (raw or wrapped based on return_raw)
    """
    import numpy as np

    # Handle numpy dtype objects first to get the type
    original_dtype = np_dtype
    if hasattr(np_dtype, 'type'):
        np_dtype = np_dtype.type

    dtype_map = {
        np.float32: float32,
        np.float64: float32,  # MLX doesn't support float64, use float32
        np.float16: float16,
        np.int8: int8,
        np.int16: int16,
        np.int32: int32,
        np.int64: int64,
        np.uint8: uint8,
        np.uint16: uint16,
        np.uint32: uint32,
        np.uint64: uint64,
        np.bool_: bool,
    }

    result = dtype_map.get(np_dtype)
    if result is None:
        raise ValueError(f"Cannot convert {original_dtype} to MLX dtype")

    # Warn when float64 is silently downgraded to float32
    if warn_on_downgrade and np_dtype == np.float64:
        warnings.warn(
            "float64 is not supported in MLX. Automatically converting to float32. "
            "This may result in reduced precision.",
            UserWarning,
            stacklevel=3
        )

    # Return raw MLX dtype or wrapped DType
    if return_raw and hasattr(result, '_mlx_dtype'):
        return result._mlx_dtype
    return result


__all__ = [
    'DType',
    # Floating point
    'float32', 'float16', 'bfloat16', 'float64',
    'float', 'half', 'double',
    # Integers
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'short', 'int', 'long', 'byte',
    # Boolean
    'bool',
    # Complex
    'complex64', 'complex128',
    # Functions
    'get_dtype', 'set_default_dtype', 'get_default_dtype',
    'torch_dtype_to_numpy', 'numpy_to_mlx_dtype',
]
