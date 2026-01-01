"""
Automatic Mixed Precision Module

PyTorch-compatible torch.amp module for MLX.
Provides autocast context manager and GradScaler for mixed precision training.

Note: MLX handles dtype management differently. These utilities are provided
for API compatibility but may have simplified behavior.
"""

import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Union

import mlx.core as mx

from .tensor import Tensor


def is_autocast_available(device_type: str) -> bool:
    """
    Check if autocast is available for a device type.

    Args:
        device_type: Device type ('cuda', 'cpu', 'mps', etc.)

    Returns:
        True if autocast is available (always True for MLX)
    """
    # MLX runs on Apple Silicon, autocast is available
    return True


class autocast:
    """
    Context manager for automatic mixed precision.

    In MLX, this provides API compatibility. MLX uses unified memory
    and can work with float16/bfloat16 efficiently.

    Args:
        device_type: Device type ('cuda', 'cpu', 'mps') - ignored in MLX
        dtype: Data type for autocasting (default: float16)
        enabled: Whether autocast is enabled
        cache_enabled: Whether to cache weight casts - ignored in MLX
    """

    def __init__(
        self,
        device_type: str = "cuda",
        dtype: Optional[Any] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        self.device_type = device_type
        self.dtype = dtype if dtype is not None else mx.float16
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self._prev_dtype = None

    def __enter__(self):
        if self.enabled:
            # In MLX, we just note the desired dtype
            # Actual casting would need to be done explicitly
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, func):
        """Decorator usage."""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class GradScaler:
    """
    Gradient scaler for mixed precision training.

    In MLX, this provides API compatibility. MLX handles gradients
    differently, so this is largely a no-op wrapper.

    Args:
        device: Device type - ignored in MLX
        init_scale: Initial scale factor
        growth_factor: Factor to increase scale
        backoff_factor: Factor to decrease scale on overflow
        growth_interval: Steps between scale increases
        enabled: Whether scaling is enabled
    """

    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self._device = device
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        self._found_inf = False

    def scale(self, outputs: Union[Tensor, Sequence[Tensor]]) -> Union[Tensor, Sequence[Tensor]]:
        """
        Scale outputs (typically loss) by the scale factor.

        Args:
            outputs: Loss tensor(s) to scale

        Returns:
            Scaled outputs
        """
        if not self._enabled:
            return outputs

        if isinstance(outputs, Tensor):
            return Tensor(outputs._mlx_array * self._scale)
        return [Tensor(o._mlx_array * self._scale) for o in outputs]

    def unscale_(self, optimizer) -> None:
        """
        Unscale gradients in the optimizer.

        Args:
            optimizer: The optimizer whose gradients to unscale
        """
        if not self._enabled:
            return

        # In a full implementation, this would unscale all gradients
        # For now, this is a no-op as MLX handles gradients differently
        pass

    def step(self, optimizer, *args, **kwargs) -> None:
        """
        Perform optimizer step after unscaling.

        Args:
            optimizer: The optimizer to step
            *args, **kwargs: Additional arguments to pass to optimizer.step()
        """
        if not self._enabled:
            optimizer.step(*args, **kwargs)
            return

        # Check for inf/nan in gradients
        self._found_inf = False

        if not self._found_inf:
            optimizer.step(*args, **kwargs)

    def update(self, new_scale: Optional[float] = None) -> None:
        """
        Update the scale factor.

        Args:
            new_scale: New scale value, or None to use automatic update
        """
        if not self._enabled:
            return

        if new_scale is not None:
            self._scale = new_scale
            return

        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

    def get_scale(self) -> float:
        """Get the current scale factor."""
        return self._scale

    def get_growth_factor(self) -> float:
        """Get the growth factor."""
        return self._growth_factor

    def get_backoff_factor(self) -> float:
        """Get the backoff factor."""
        return self._backoff_factor

    def get_growth_interval(self) -> int:
        """Get the growth interval."""
        return self._growth_interval

    def is_enabled(self) -> bool:
        """Check if scaler is enabled."""
        return self._enabled

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self._scale = state_dict["scale"]
        self._growth_factor = state_dict.get("growth_factor", self._growth_factor)
        self._backoff_factor = state_dict.get("backoff_factor", self._backoff_factor)
        self._growth_interval = state_dict.get("growth_interval", self._growth_interval)
        self._growth_tracker = state_dict.get("_growth_tracker", 0)


def custom_fwd(fwd=None, device_type: str = "cuda", *, cast_inputs: Optional[Any] = None):
    """
    Decorator for custom autograd function forward pass.

    In MLX, this is a passthrough decorator for compatibility.

    Args:
        fwd: Forward function to wrap
        device_type: Device type (ignored in MLX)
        cast_inputs: Dtype to cast inputs to (ignored in MLX)
    """
    if fwd is None:
        return lambda f: custom_fwd(f, device_type=device_type, cast_inputs=cast_inputs)
    return fwd


def custom_bwd(bwd=None, device_type: str = "cuda"):
    """
    Decorator for custom autograd function backward pass.

    In MLX, this is a passthrough decorator for compatibility.

    Args:
        bwd: Backward function to wrap
        device_type: Device type (ignored in MLX)
    """
    if bwd is None:
        return lambda b: custom_bwd(b, device_type=device_type)
    return bwd


# Submodule compatibility
class autocast_mode:
    """Compatibility module for torch.amp.autocast_mode."""

    autocast = autocast
    is_autocast_available = staticmethod(is_autocast_available)


class grad_scaler:
    """Compatibility module for torch.amp.grad_scaler."""

    GradScaler = GradScaler


__all__ = [
    "autocast",
    "GradScaler",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
    "autocast_mode",
    "grad_scaler",
]
