"""
Device management compatibility layer.

MLX uses unified memory, so device management is primarily a compatibility shim
to match PyTorch's API. All tensors are automatically on the Metal device
with unified CPU/GPU memory access.

Reference:
- pytorch-mlx-porting-docs/08-PORTING-GUIDE/mlx-mapping.md (lines 44-68)
- pytorch-mlx-porting-docs/01-FOUNDATIONS/type-system.md
"""

import warnings
from typing import Union, Optional


class Device:
    """
    PyTorch-compatible device wrapper.

    In MLX, all arrays use unified memory and are accessible from both
    CPU and Metal GPU. Device specification is primarily for API compatibility.

    Examples:
        >>> device = Device('cpu')
        >>> device = Device('cuda')  # Accepted for compatibility
        >>> device = Device('mps')   # Metal Performance Shaders
    """

    def __init__(self, device: Union[str, 'Device', int]):
        """
        Create a device object.

        Args:
            device: Can be:
                - String: 'cpu', 'cuda', 'cuda:0', 'mps', 'metal'
                - Device: Another device object
                - int: CUDA device index (e.g., 0, 1)
        """
        if isinstance(device, Device):
            self.type = device.type
            self.index = device.index
            return

        if isinstance(device, int):
            # CUDA device index
            self.type = 'cuda'
            self.index = device
            self._warn_unified_memory()
            return

        # Parse string
        device_str = str(device).lower()

        if ':' in device_str:
            # e.g., 'cuda:0', 'cpu:0'
            parts = device_str.split(':')
            self.type = parts[0]
            try:
                self.index = int(parts[1])
            except (ValueError, IndexError):
                self.index = None
        else:
            self.type = device_str
            self.index = None

        # Normalize device types
        if self.type in ('cuda', 'gpu'):
            self.type = 'cuda'
            self._warn_unified_memory()
        elif self.type in ('mps', 'metal'):
            self.type = 'mps'  # Metal Performance Shaders
        elif self.type == 'cpu':
            self.type = 'cpu'
        else:
            raise ValueError(
                f"Invalid device type: {device}. "
                f"Supported: 'cpu', 'cuda', 'mps', 'metal'"
            )

    def _warn_unified_memory(self):
        """Warn about MLX unified memory (one-time warning)."""
        if not hasattr(Device, '_warned_unified'):
            warnings.warn(
                "MLX uses unified memory. Device specification ('cpu', 'cuda', 'mps') "
                "is for API compatibility only. All tensors are accessible from both "
                "CPU and Metal GPU without explicit transfers.",
                UserWarning,
                stacklevel=4
            )
            Device._warned_unified = True

    def __repr__(self):
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self):
        return hash((self.type, self.index))


# Default devices
_default_device = Device('cpu')


def get_default_device() -> Device:
    """Get the current default device."""
    return _default_device


def set_default_device(device: Union[str, Device]):
    """
    Set the default device for tensor creation.

    Args:
        device: Device specification
    """
    global _default_device
    _default_device = Device(device)


# Convenience functions (PyTorch compatibility)
def current_device() -> int:
    """
    Get current CUDA device index.

    In MLX with unified memory, this always returns 0 for compatibility.
    """
    return 0


def device_count() -> int:
    """
    Get number of Metal devices.

    Returns 1 if Metal is available, 0 otherwise.
    """
    try:
        import mlx.core as mx
        # If MLX imports successfully, Metal is available
        return 1
    except ImportError:
        return 0


def is_available() -> bool:
    """
    Check if Metal/MLX is available.

    Returns:
        True if MLX can be imported and Metal is available
    """
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


def synchronize(device: Optional[Union[str, Device]] = None):
    """
    Synchronize device operations (for PyTorch compatibility).

    In MLX, this is handled automatically. This function is a no-op
    but provided for API compatibility.

    Args:
        device: Device to synchronize (ignored in MLX)
    """
    # MLX handles synchronization automatically
    # This is a no-op for compatibility
    pass


__all__ = [
    'Device',
    'get_default_device',
    'set_default_device',
    'current_device',
    'device_count',
    'is_available',
    'synchronize',
]
