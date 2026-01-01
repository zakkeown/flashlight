"""
Layout management for NHWC-native mode optimization.

This module provides layout tracking and conversion utilities to minimize
redundant NCHW <-> NHWC conversions in spatial operations.

Usage:
    import flashlight

    # Default: NCHW mode (PyTorch compatibility)
    output = model(input)  # Converts at every spatial layer

    # Optimized: NHWC mode
    with flashlight.nhwc_mode():
        output = model(input)  # No intermediate conversions
    # Output automatically converted to NCHW when leaving context
"""

from enum import Enum, auto
from typing import Optional, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from .tensor import Tensor


class Layout(Enum):
    """Memory layout for tensor data."""
    NCHW = auto()    # PyTorch default: [N, C, H, W] - channels first (2D)
    NHWC = auto()    # MLX native: [N, H, W, C] - channels last (2D)
    NCL = auto()     # 1D channels-first: [N, C, L]
    NLC = auto()     # 1D channels-last: [N, L, C]
    NCDHW = auto()   # 3D channels-first: [N, C, D, H, W]
    NDHWC = auto()   # 3D channels-last: [N, D, H, W, C]
    CONTIGUOUS = auto()  # Non-spatial tensor (no specific layout)


# Thread-local storage for NHWC mode
_layout_mode = threading.local()


def is_nhwc_mode() -> bool:
    """Check if NHWC-native mode is enabled."""
    return getattr(_layout_mode, 'enabled', False)


def _set_nhwc_mode(mode: bool) -> bool:
    """Internal: Set NHWC mode and return previous state."""
    prev = is_nhwc_mode()
    _layout_mode.enabled = mode
    return prev


class nhwc_mode:
    """
    Context manager that enables NHWC-native mode.

    When enabled, spatial tensors are kept in NHWC format throughout
    operations, only converting at API boundaries when needed.

    This can significantly reduce overhead in models with many
    consecutive spatial operations (Conv2d, BatchNorm, Pooling).

    Example:
        >>> import flashlight
        >>> model = flashlight.nn.Sequential(
        ...     flashlight.nn.Conv2d(3, 64, 3),
        ...     flashlight.nn.BatchNorm2d(64),
        ...     flashlight.nn.ReLU(),
        ...     flashlight.nn.MaxPool2d(2),
        ... )
        >>> with flashlight.nhwc_mode():
        ...     # All spatial ops use NHWC internally - no intermediate conversions
        ...     output = model(input)
        >>> # Output is automatically in NCHW format after context exits

    Can also be used as a decorator:
        >>> @flashlight.nhwc_mode()
        ... def forward_pass(model, x):
        ...     return model(x)
    """

    def __init__(self):
        self.prev = None

    def __enter__(self):
        self.prev = _set_nhwc_mode(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_nhwc_mode(self.prev)
        return False

    def __call__(self, func):
        """Allow using as a decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class nchw_mode:
    """
    Context manager that forces NCHW mode (PyTorch compatibility).

    This is the default mode, but can be used to explicitly ensure
    NCHW format when nested inside an nhwc_mode context.

    Example:
        >>> with flashlight.nhwc_mode():
        ...     # Operations here use NHWC
        ...     with flashlight.nchw_mode():
        ...         # Operations here use NCHW
        ...         pass
        ...     # Back to NHWC
    """

    def __init__(self):
        self.prev = None

    def __enter__(self):
        self.prev = _set_nhwc_mode(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_nhwc_mode(self.prev)
        return False


# Transpose permutation maps for layout conversions
_LAYOUT_TRANSPOSE = {
    # 2D (4D tensor): NCHW <-> NHWC
    (Layout.NCHW, Layout.NHWC): [0, 2, 3, 1],
    (Layout.NHWC, Layout.NCHW): [0, 3, 1, 2],
    # 1D (3D tensor): NCL <-> NLC
    (Layout.NCL, Layout.NLC): [0, 2, 1],
    (Layout.NLC, Layout.NCL): [0, 2, 1],
    # 3D (5D tensor): NCDHW <-> NDHWC
    (Layout.NCDHW, Layout.NDHWC): [0, 2, 3, 4, 1],
    (Layout.NDHWC, Layout.NCDHW): [0, 4, 1, 2, 3],
}


def infer_layout(tensor: 'Tensor') -> Layout:
    """
    Infer the layout of a tensor based on its explicit _layout attribute.

    For tensors without explicit layout, assumes NCHW (PyTorch default).
    Only tensors that have gone through layout-aware operations in nhwc_mode
    will have explicit NHWC layout set.

    Args:
        tensor: The tensor to infer layout for

    Returns:
        The inferred Layout enum value
    """
    # If tensor has explicit layout, use it
    if hasattr(tensor, '_layout') and tensor._layout is not None:
        return tensor._layout

    # Default: assume NCHW format (PyTorch convention)
    # User-created tensors are always NCHW
    ndim = tensor.ndim
    if ndim == 4:
        return Layout.NCHW
    elif ndim == 3:
        return Layout.NCL
    elif ndim == 5:
        return Layout.NCDHW

    return Layout.CONTIGUOUS


def convert_layout(tensor: 'Tensor', target_layout: Layout) -> 'Tensor':
    """
    Convert tensor to target layout.

    Args:
        tensor: The tensor to convert
        target_layout: The target layout

    Returns:
        Tensor in target layout (same tensor if already in correct layout)
    """
    import mlx.core as mx
    from .tensor import Tensor

    current = infer_layout(tensor)

    # Already in correct layout or non-spatial
    if current == target_layout or current == Layout.CONTIGUOUS:
        return tensor

    key = (current, target_layout)
    if key not in _LAYOUT_TRANSPOSE:
        raise ValueError(f"Cannot convert from {current} to {target_layout}")

    perm = _LAYOUT_TRANSPOSE[key]
    result_array = mx.transpose(tensor._mlx_array, perm)

    result = Tensor._from_mlx_array(
        result_array,
        requires_grad=tensor.requires_grad,
    )
    result._layout = target_layout

    return result


def ensure_layout(tensor: 'Tensor', layout: Layout) -> 'Tensor':
    """
    Ensure tensor has specified layout, converting if necessary.

    Args:
        tensor: The tensor to check/convert
        layout: The required layout

    Returns:
        Tensor in the specified layout
    """
    current = infer_layout(tensor)
    if current == layout or current == Layout.CONTIGUOUS:
        return tensor
    return convert_layout(tensor, layout)


def ensure_nhwc(tensor: 'Tensor') -> 'Tensor':
    """
    Ensure 4D tensor is in NHWC layout.

    Args:
        tensor: A 4D tensor

    Returns:
        Tensor in NHWC layout
    """
    return ensure_layout(tensor, Layout.NHWC)


def ensure_nchw(tensor: 'Tensor') -> 'Tensor':
    """
    Ensure 4D tensor is in NCHW layout.

    Args:
        tensor: A 4D tensor

    Returns:
        Tensor in NCHW layout
    """
    return ensure_layout(tensor, Layout.NCHW)


def get_output_layout() -> Layout:
    """
    Get the appropriate output layout based on current mode.

    Returns:
        Layout.NHWC if in nhwc_mode, Layout.NCHW otherwise
    """
    return Layout.NHWC if is_nhwc_mode() else Layout.NCHW


def get_output_layout_1d() -> Layout:
    """Get the appropriate 1D output layout based on current mode."""
    return Layout.NLC if is_nhwc_mode() else Layout.NCL


def get_output_layout_3d() -> Layout:
    """Get the appropriate 3D output layout based on current mode."""
    return Layout.NDHWC if is_nhwc_mode() else Layout.NCDHW
