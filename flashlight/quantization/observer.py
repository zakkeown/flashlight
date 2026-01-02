"""
Quantization Observers

Observers collect statistics from tensors for computing quantization parameters.
They are used during calibration (PTQ) and training (QAT).
"""

from typing import Optional, Tuple

import mlx.core as mx

from ..nn.module import Module
from ..tensor import Tensor


class ObserverBase(Module):
    """
    Base class for quantization observers.

    Observers track tensor statistics to determine optimal quantization parameters.
    """

    def __init__(self, bits: int = 8, group_size: int = 64):
        super().__init__()
        self.bits = bits
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        """Observe the tensor and return it unchanged."""
        raise NotImplementedError

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        """Calculate quantization parameters (scale, zero_point)."""
        raise NotImplementedError

    def reset_min_max_vals(self) -> None:
        """Reset observation state."""
        pass


class MinMaxObserver(ObserverBase):
    """
    Observer that tracks min/max values for computing scale and zero_point.

    This is the simplest observer - it just tracks the global min and max
    of all observed values.

    Args:
        bits: Number of bits for quantization (determines qmin/qmax)
        group_size: Group size for quantization
        symmetric: If True, use symmetric quantization around zero

    Example:
        >>> observer = MinMaxObserver(bits=8)
        >>> observer(x)  # Observe tensor x
        >>> observer(y)  # Observe tensor y
        >>> scale, zero_point = observer.calculate_qparams()
    """

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 64,
        symmetric: bool = False,
    ):
        super().__init__(bits=bits, group_size=group_size)
        self.symmetric = symmetric
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def forward(self, x: Tensor) -> Tensor:
        """Observe min/max of the input tensor."""
        x_min = float(x.min().item())
        x_max = float(x.max().item())

        if self.min_val is None:
            self.min_val = x_min
        else:
            self.min_val = min(self.min_val, x_min)

        if self.max_val is None:
            self.max_val = x_max
        else:
            self.max_val = max(self.max_val, x_max)

        return x

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        """
        Calculate scale and zero_point from observed min/max.

        Returns:
            Tuple of (scale, zero_point) tensors
        """
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has not seen any data. Call forward() first.")

        # Quantization range
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1

        min_val = self.min_val
        max_val = self.max_val

        if self.symmetric:
            # Symmetric quantization: zero_point = 0
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / qmax if qmax != 0 else 1.0
            zero_point = 0
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / (qmax - qmin) if (qmax - qmin) != 0 else 1.0
            zero_point = qmin - round(min_val / scale) if scale != 0 else 0
            zero_point = max(qmin, min(qmax, zero_point))

        from ..factories import tensor

        return tensor([scale]), tensor([zero_point])

    def reset_min_max_vals(self) -> None:
        """Reset tracked min/max values."""
        self.min_val = None
        self.max_val = None


class MovingAverageMinMaxObserver(ObserverBase):
    """
    Observer that uses exponential moving average for min/max.

    This is better for training (QAT) where statistics may shift.

    Args:
        averaging_constant: EMA factor (default 0.01)
        bits: Number of bits for quantization
        group_size: Group size for quantization
        symmetric: If True, use symmetric quantization
    """

    def __init__(
        self,
        averaging_constant: float = 0.01,
        bits: int = 8,
        group_size: int = 64,
        symmetric: bool = False,
    ):
        super().__init__(bits=bits, group_size=group_size)
        self.averaging_constant = averaging_constant
        self.symmetric = symmetric
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def forward(self, x: Tensor) -> Tensor:
        """Update min/max with exponential moving average."""
        x_min = float(x.min().item())
        x_max = float(x.max().item())

        if self.min_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            # EMA update
            self.min_val = self.min_val * (1 - self.averaging_constant) + x_min * self.averaging_constant
            self.max_val = self.max_val * (1 - self.averaging_constant) + x_max * self.averaging_constant

        return x

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        """Calculate scale and zero_point from EMA min/max."""
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has not seen any data.")

        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1

        if self.symmetric:
            max_abs = max(abs(self.min_val), abs(self.max_val))
            scale = max_abs / qmax if qmax != 0 else 1.0
            zero_point = 0
        else:
            scale = (self.max_val - self.min_val) / (qmax - qmin) if (qmax - qmin) != 0 else 1.0
            zero_point = qmin - round(self.min_val / scale) if scale != 0 else 0
            zero_point = max(qmin, min(qmax, zero_point))

        from ..factories import tensor

        return tensor([scale]), tensor([zero_point])

    def reset_min_max_vals(self) -> None:
        """Reset EMA state."""
        self.min_val = None
        self.max_val = None


class PerChannelMinMaxObserver(ObserverBase):
    """
    Observer that tracks per-channel min/max for weight quantization.

    Args:
        ch_axis: Channel axis (typically 0 for weights)
        bits: Number of bits for quantization
        group_size: Group size for quantization
        symmetric: If True, use symmetric quantization
    """

    def __init__(
        self,
        ch_axis: int = 0,
        bits: int = 8,
        group_size: int = 64,
        symmetric: bool = True,
    ):
        super().__init__(bits=bits, group_size=group_size)
        self.ch_axis = ch_axis
        self.symmetric = symmetric
        self.min_vals: Optional[Tensor] = None
        self.max_vals: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """Track per-channel min/max."""
        # Compute min/max along all axes except the channel axis
        axes = list(range(x.ndim))
        axes.remove(self.ch_axis)

        if axes:
            # Reduce along all axes except ch_axis one at a time
            x_min = x
            x_max = x
            for ax in sorted(axes, reverse=True):  # Reverse to preserve indices
                x_min = x_min.min(dim=ax).values
                x_max = x_max.max(dim=ax).values
        else:
            from ..factories import zeros_like
            x_min = x + zeros_like(x)  # Clone
            x_max = x + zeros_like(x)

        if self.min_vals is None:
            self.min_vals = x_min
            self.max_vals = x_max
        else:
            # Element-wise min/max
            self.min_vals = Tensor._from_mlx_array(
                mx.minimum(self.min_vals._mlx_array, x_min._mlx_array)
            )
            self.max_vals = Tensor._from_mlx_array(
                mx.maximum(self.max_vals._mlx_array, x_max._mlx_array)
            )

        return x

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        """Calculate per-channel scale and zero_point."""
        if self.min_vals is None or self.max_vals is None:
            raise RuntimeError("Observer has not seen any data.")

        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1

        min_arr = self.min_vals._mlx_array
        max_arr = self.max_vals._mlx_array

        if self.symmetric:
            max_abs = mx.maximum(mx.abs(min_arr), mx.abs(max_arr))
            scale = max_abs / qmax
            zero_point = mx.zeros(scale.shape, dtype=mx.int32)
        else:
            scale = (max_arr - min_arr) / (qmax - qmin)
            zero_point = mx.round(qmin - min_arr / scale).astype(mx.int32)
            zero_point = mx.clip(zero_point, qmin, qmax)

        return (
            Tensor._from_mlx_array(scale),
            Tensor._from_mlx_array(zero_point),
        )

    def reset_min_max_vals(self) -> None:
        """Reset per-channel min/max values."""
        self.min_vals = None
        self.max_vals = None


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """
    Per-channel observer with exponential moving average.

    Combines per-channel tracking with EMA for QAT.
    """

    def __init__(
        self,
        averaging_constant: float = 0.01,
        ch_axis: int = 0,
        bits: int = 8,
        group_size: int = 64,
        symmetric: bool = True,
    ):
        super().__init__(ch_axis=ch_axis, bits=bits, group_size=group_size, symmetric=symmetric)
        self.averaging_constant = averaging_constant

    def forward(self, x: Tensor) -> Tensor:
        """Update per-channel min/max with EMA."""
        axes = list(range(x.ndim))
        axes.remove(self.ch_axis)

        if axes:
            # Reduce along all axes except ch_axis one at a time
            x_min = x
            x_max = x
            for ax in sorted(axes, reverse=True):  # Reverse to preserve indices
                x_min = x_min.min(dim=ax).values
                x_max = x_max.max(dim=ax).values
        else:
            from ..factories import zeros_like
            x_min = x + zeros_like(x)  # Clone
            x_max = x + zeros_like(x)

        if self.min_vals is None:
            self.min_vals = x_min
            self.max_vals = x_max
        else:
            # EMA update
            c = self.averaging_constant
            self.min_vals = Tensor._from_mlx_array(
                self.min_vals._mlx_array * (1 - c) + x_min._mlx_array * c
            )
            self.max_vals = Tensor._from_mlx_array(
                self.max_vals._mlx_array * (1 - c) + x_max._mlx_array * c
            )

        return x


class HistogramObserver(ObserverBase):
    """
    Observer that builds a histogram for more accurate quantization.

    Uses histogram data to find optimal scale/zero_point that minimizes
    quantization error.

    Args:
        bins: Number of histogram bins
        bits: Number of bits for quantization
        group_size: Group size for quantization
    """

    def __init__(
        self,
        bins: int = 2048,
        bits: int = 8,
        group_size: int = 64,
    ):
        super().__init__(bits=bits, group_size=group_size)
        self.bins = bins
        self.histogram: Optional[Tensor] = None
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def forward(self, x: Tensor) -> Tensor:
        """Accumulate histogram of values."""
        x_min = float(x.min().item())
        x_max = float(x.max().item())

        if self.min_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            self.min_val = min(self.min_val, x_min)
            self.max_val = max(self.max_val, x_max)

        # Compute histogram
        # Note: MLX doesn't have histogram, so we'll use a simple bin counting approach
        flat = x.flatten()._mlx_array
        bin_edges = mx.linspace(self.min_val, self.max_val, self.bins + 1)
        bin_indices = mx.searchsorted(bin_edges[1:-1], flat)
        bin_indices = mx.clip(bin_indices, 0, self.bins - 1)

        # Count occurrences
        new_hist = mx.zeros((self.bins,), dtype=mx.int32)
        for i in range(self.bins):
            new_hist = mx.where(
                mx.arange(self.bins) == i,
                new_hist + mx.sum(bin_indices == i),
                new_hist,
            )

        if self.histogram is None:
            self.histogram = Tensor._from_mlx_array(new_hist.astype(mx.float32))
        else:
            self.histogram = Tensor._from_mlx_array(
                self.histogram._mlx_array + new_hist.astype(mx.float32)
            )

        return x

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        """
        Calculate optimal scale/zero_point from histogram.

        Uses a simple min/max approach for now. Could be extended
        to use entropy-based or MSE-minimizing approaches.
        """
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has not seen any data.")

        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1

        scale = (self.max_val - self.min_val) / (qmax - qmin) if (qmax - qmin) != 0 else 1.0
        zero_point = qmin - round(self.min_val / scale) if scale != 0 else 0
        zero_point = max(qmin, min(qmax, zero_point))

        from ..factories import tensor

        return tensor([scale]), tensor([zero_point])

    def reset_min_max_vals(self) -> None:
        """Reset histogram and min/max values."""
        self.histogram = None
        self.min_val = None
        self.max_val = None


class NoopObserver(ObserverBase):
    """
    Observer that does nothing - for layers that shouldn't be observed.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self) -> Tuple[Tensor, Tensor]:
        from ..factories import tensor

        return tensor([1.0]), tensor([0])
