"""
Autograd anomaly detection mode.

Provides context managers for detecting NaN/Inf gradients and other
anomalies during backpropagation.
"""

import threading
import warnings

# Thread-local storage for anomaly detection state
_anomaly_state = threading.local()


def is_anomaly_enabled() -> bool:
    """Check if anomaly detection mode is enabled."""
    return getattr(_anomaly_state, "enabled", False)


def is_anomaly_check_nan_enabled() -> bool:
    """Check if NaN checking is enabled in anomaly mode."""
    return getattr(_anomaly_state, "check_nan", True)


class detect_anomaly:
    """
    Context manager for enabling anomaly detection during backward pass.

    When enabled, this will check for NaN/Inf values in gradients and raise
    an error with information about which operation produced them.

    Args:
        check_nan: If True (default), check for NaN/Inf values in gradients.

    Example:
        >>> with flashlight.autograd.detect_anomaly():
        ...     loss = model(input)
        ...     loss.backward()  # Will raise if NaN/Inf detected
    """

    def __init__(self, check_nan: bool = True):
        self.check_nan = check_nan

    def __enter__(self):
        self.prev_enabled = is_anomaly_enabled()
        self.prev_check_nan = is_anomaly_check_nan_enabled()
        _anomaly_state.enabled = True
        _anomaly_state.check_nan = self.check_nan
        if not self.prev_enabled:
            warnings.warn(
                "Anomaly Detection enabled. "
                "This mode will slow down backward passes and is intended for debugging only.",
                stacklevel=2,
            )
        return self

    def __exit__(self, *args):
        _anomaly_state.enabled = self.prev_enabled
        _anomaly_state.check_nan = self.prev_check_nan


class set_detect_anomaly:
    """
    Context manager that sets anomaly detection on or off.

    Unlike detect_anomaly, this does not issue a warning.

    Args:
        mode: Whether to enable anomaly detection.
        check_nan: If True (default), check for NaN/Inf values.

    Example:
        >>> with flashlight.autograd.set_detect_anomaly(True):
        ...     # anomaly detection enabled here
        ...     pass
    """

    def __init__(self, mode: bool, check_nan: bool = True):
        self.mode = mode
        self.check_nan = check_nan

    def __enter__(self):
        self.prev_enabled = is_anomaly_enabled()
        self.prev_check_nan = is_anomaly_check_nan_enabled()
        _anomaly_state.enabled = self.mode
        _anomaly_state.check_nan = self.check_nan
        return self

    def __exit__(self, *args):
        _anomaly_state.enabled = self.prev_enabled
        _anomaly_state.check_nan = self.prev_check_nan


__all__ = [
    "detect_anomaly",
    "set_detect_anomaly",
    "is_anomaly_enabled",
    "is_anomaly_check_nan_enabled",
]
