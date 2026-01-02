"""
CUDA Compatibility Module

Provides a compatibility shim for code that references torch.cuda.
Since MLX uses Apple's unified memory architecture, there is no separate
CUDA device. All tensors are accessible from both CPU and GPU automatically.

This module provides stub implementations that allow PyTorch code using
torch.cuda to run without modification, though actual CUDA operations
will not be performed.
"""

import warnings
from typing import Any, Optional, Union

# Track if we've warned about CUDA compatibility
_warned_cuda_compat = False


def _warn_cuda_compat():
    """Emit one-time warning about CUDA compatibility."""
    global _warned_cuda_compat
    if not _warned_cuda_compat:
        warnings.warn(
            "flashlight.cuda is a compatibility shim. MLX uses unified memory "
            "and does not have separate CUDA devices. CUDA operations are no-ops.",
            UserWarning,
            stacklevel=3,
        )
        _warned_cuda_compat = True


def is_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        Always False, as MLX uses unified memory instead of CUDA.
    """
    return False


def device_count() -> int:
    """
    Get number of CUDA devices.

    Returns:
        Always 0, as MLX uses unified memory instead of CUDA.
    """
    return 0


def current_device() -> int:
    """
    Get current CUDA device index.

    Returns:
        Always 0 for API compatibility, though no CUDA devices exist.
    """
    return 0


def set_device(device: Union[int, str]) -> None:
    """
    Set current CUDA device.

    Args:
        device: Device index or device string.

    Note:
        This is a no-op in MLX as it uses unified memory.
    """
    _warn_cuda_compat()


def synchronize(device: Optional[int] = None) -> None:
    """
    Synchronize CUDA device.

    Args:
        device: Device to synchronize (ignored).

    Note:
        MLX handles synchronization automatically. This is a no-op.
    """
    pass


def get_device_name(device: Optional[int] = None) -> str:
    """
    Get name of CUDA device.

    Args:
        device: Device index (ignored).

    Returns:
        A string indicating MLX unified memory.
    """
    return "MLX Unified Memory (Apple Silicon)"


def get_device_capability(device: Optional[int] = None) -> tuple:
    """
    Get CUDA compute capability.

    Args:
        device: Device index (ignored).

    Returns:
        (0, 0) as MLX doesn't have CUDA compute capability.
    """
    return (0, 0)


def get_device_properties(device: Optional[int] = None) -> Any:
    """
    Get CUDA device properties.

    Args:
        device: Device index (ignored).

    Returns:
        A simple object with placeholder properties.
    """

    class DeviceProperties:
        name = "MLX Unified Memory"
        major = 0
        minor = 0
        total_memory = 0
        multi_processor_count = 0

    return DeviceProperties()


def memory_allocated(device: Optional[int] = None) -> int:
    """
    Get current GPU memory usage.

    Args:
        device: Device index (ignored).

    Returns:
        0 as MLX uses unified memory with automatic management.
    """
    return 0


def max_memory_allocated(device: Optional[int] = None) -> int:
    """
    Get maximum GPU memory usage.

    Args:
        device: Device index (ignored).

    Returns:
        0 as MLX uses unified memory with automatic management.
    """
    return 0


def memory_reserved(device: Optional[int] = None) -> int:
    """
    Get current GPU memory reserved.

    Args:
        device: Device index (ignored).

    Returns:
        0 as MLX uses unified memory with automatic management.
    """
    return 0


def max_memory_reserved(device: Optional[int] = None) -> int:
    """
    Get maximum GPU memory reserved.

    Args:
        device: Device index (ignored).

    Returns:
        0 as MLX uses unified memory with automatic management.
    """
    return 0


def empty_cache() -> None:
    """
    Release unoccupied cached memory.

    Note:
        This is a no-op in MLX as memory is managed automatically.
    """
    pass


def reset_peak_memory_stats(device: Optional[int] = None) -> None:
    """
    Reset peak memory statistics.

    Args:
        device: Device index (ignored).

    Note:
        This is a no-op in MLX.
    """
    pass


def memory_summary(device: Optional[int] = None, abbreviated: bool = False) -> str:
    """
    Get a human-readable memory summary.

    Args:
        device: Device index (ignored).
        abbreviated: Whether to use abbreviated format.

    Returns:
        A string indicating MLX memory management.
    """
    return (
        "MLX Memory Summary\n"
        "==================\n"
        "MLX uses Apple's unified memory architecture.\n"
        "Memory is shared between CPU and GPU automatically.\n"
        "No explicit CUDA memory management is needed.\n"
    )


class Stream:
    """Compatibility shim for CUDA streams."""

    def __init__(self, device: Optional[int] = None, **kwargs):
        self.device = device

    def synchronize(self):
        """No-op synchronization."""
        pass

    def wait_stream(self, stream: "Stream"):
        """No-op wait."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Event:
    """Compatibility shim for CUDA events."""

    def __init__(self, enable_timing: bool = False, blocking: bool = False):
        self.enable_timing = enable_timing
        self.blocking = blocking

    def record(self, stream: Optional[Stream] = None):
        """No-op record."""
        pass

    def synchronize(self):
        """No-op synchronization."""
        pass

    def wait(self, stream: Optional[Stream] = None):
        """No-op wait."""
        pass

    def elapsed_time(self, end_event: "Event") -> float:
        """Return 0.0 as no actual timing is done."""
        return 0.0


def current_stream(device: Optional[int] = None) -> Stream:
    """
    Get current CUDA stream.

    Args:
        device: Device index (ignored).

    Returns:
        A compatibility Stream object.
    """
    return Stream(device)


def default_stream(device: Optional[int] = None) -> Stream:
    """
    Get default CUDA stream.

    Args:
        device: Device index (ignored).

    Returns:
        A compatibility Stream object.
    """
    return Stream(device)


def stream(stream: Optional[Stream] = None):
    """
    Context manager for CUDA stream.

    Args:
        stream: Stream to use (ignored).

    Returns:
        A no-op context manager.
    """

    class StreamContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return StreamContext()


# Amp (Automatic Mixed Precision) compatibility
class amp:
    """Compatibility shim for CUDA AMP."""

    @staticmethod
    def autocast(device_type: str = "cuda", enabled: bool = True, **kwargs):
        """Return a no-op context manager."""

        class AutocastContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return AutocastContext()

    class GradScaler:
        """Compatibility shim for GradScaler."""

        def __init__(self, **kwargs):
            pass

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

        def get_scale(self):
            return 1.0

        def is_enabled(self):
            return False
