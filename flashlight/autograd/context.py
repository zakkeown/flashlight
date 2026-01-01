"""
Autograd Context Managers

Implements PyTorch-compatible context managers for controlling gradient tracking:
- no_grad(): Disables gradient tracking
- enable_grad(): Enables gradient tracking
- set_grad_enabled(): Conditionally enables/disables gradients
"""

import threading
from typing import Optional

# Thread-local storage for gradient tracking state
_grad_enabled = threading.local()


def is_grad_enabled() -> bool:
    """
    Returns True if gradient tracking is currently enabled.

    Returns:
        bool: Whether gradients are being tracked
    """
    if not hasattr(_grad_enabled, "enabled"):
        _grad_enabled.enabled = True  # Enabled by default
    return _grad_enabled.enabled


def _set_grad_enabled(mode: bool) -> bool:
    """
    Internal function to set gradient tracking mode.

    Args:
        mode: True to enable gradients, False to disable

    Returns:
        Previous gradient tracking mode
    """
    prev = is_grad_enabled()
    _grad_enabled.enabled = mode
    return prev


class no_grad:
    """
    Context manager that disables gradient tracking.

    Operations performed within this context will not be tracked
    for automatic differentiation.

    Example:
        >>> x = flashlight.tensor([1., 2., 3.], requires_grad=True)
        >>> with flashlight.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
    """

    def __init__(self):
        self.prev = None

    def __enter__(self):
        self.prev = _set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled(self.prev)
        return False

    def __call__(self, func):
        """Allow using as a decorator."""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class enable_grad:
    """
    Context manager that enables gradient tracking.

    This is useful when you want to enable gradients within a no_grad() context.

    Example:
        >>> x = flashlight.tensor([1., 2., 3.], requires_grad=True)
        >>> with flashlight.no_grad():
        ...     with flashlight.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
    """

    def __init__(self, *args, **kwargs):
        # Accept *args, **kwargs to match PyTorch's signature
        self.prev = None

    def __enter__(self):
        self.prev = _set_grad_enabled(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled(self.prev)
        return False

    def __call__(self, func):
        """Allow using as a decorator."""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class set_grad_enabled:
    """
    Context manager that conditionally enables or disables gradient tracking.

    Args:
        mode: If True, enable gradients. If False, disable gradients.

    Example:
        >>> x = flashlight.tensor([1., 2., 3.], requires_grad=True)
        >>> with flashlight.set_grad_enabled(False):
        ...     y = x * 2
        >>> y.requires_grad
        False
    """

    def __init__(self, mode: bool):
        self.mode = mode
        self.prev = None

    def __enter__(self):
        self.prev = _set_grad_enabled(self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_enabled(self.prev)
        return False

    def __call__(self, func):
        """Allow using as a decorator."""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
