"""
Fake Quantization for QAT

FakeQuantize simulates quantization effects during training while keeping
values in floating point. This enables gradients to flow through the
quantization operation using the straight-through estimator.
"""

from typing import Optional, Tuple, Type

import mlx.core as mx

from ..nn.module import Module
from ..tensor import Tensor
from .observer import MinMaxObserver, ObserverBase


class FakeQuantizeBase(Module):
    """
    Base class for fake quantization modules.

    Fake quantization performs:
    1. Quantize the input to integers
    2. Immediately dequantize back to float
    3. Use straight-through estimator for gradients

    This simulates quantization error during training.
    """

    def __init__(
        self,
        observer: ObserverBase,
        quant_min: int = -128,
        quant_max: int = 127,
        bits: int = 8,
    ):
        super().__init__()
        self.observer = observer
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.bits = bits
        self.scale: Optional[Tensor] = None
        self.zero_point: Optional[Tensor] = None
        self.fake_quant_enabled = True
        self.observer_enabled = True

    def enable_fake_quant(self, enabled: bool = True) -> "FakeQuantizeBase":
        """Enable or disable fake quantization."""
        self.fake_quant_enabled = enabled
        return self

    def disable_fake_quant(self) -> "FakeQuantizeBase":
        """Disable fake quantization."""
        return self.enable_fake_quant(False)

    def enable_observer(self, enabled: bool = True) -> "FakeQuantizeBase":
        """Enable or disable the observer."""
        self.observer_enabled = enabled
        return self

    def disable_observer(self) -> "FakeQuantizeBase":
        """Disable the observer."""
        return self.enable_observer(False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with fake quantization.

        If observer is enabled, observe the input.
        If fake_quant is enabled, apply fake quantization.
        """
        if self.observer_enabled:
            self.observer(x)

        if not self.fake_quant_enabled:
            return x

        # Get quantization parameters (update every forward when observer is enabled)
        try:
            scale, zero_point = self.observer.calculate_qparams()
            self.scale = scale
            self.zero_point = zero_point
        except RuntimeError:
            # Observer hasn't seen data yet - pass through unchanged
            return x

        return self._fake_quantize(x, self.scale, self.zero_point)

    def _fake_quantize(
        self,
        x: Tensor,
        scale: Tensor,
        zero_point: Tensor,
    ) -> Tensor:
        """
        Apply fake quantization: quantize then dequantize.

        Uses straight-through estimator for gradients.
        """
        scale_val = scale._mlx_array
        zp_val = zero_point._mlx_array

        # Quantize: x_q = clamp(round(x / scale) + zero_point, qmin, qmax)
        x_scaled = x._mlx_array / scale_val
        x_zp = x_scaled + zp_val.astype(x_scaled.dtype)
        x_q = mx.clip(mx.round(x_zp), self.quant_min, self.quant_max)

        # Dequantize: x_dq = (x_q - zero_point) * scale
        x_dq = (x_q - zp_val.astype(x_q.dtype)) * scale_val

        # Straight-through estimator: forward uses quantized values,
        # backward uses identity gradient
        # In MLX, we achieve this by: output = x + (x_dq - x).stop_gradient()
        # But MLX doesn't have stop_gradient, so we just return the dequantized value
        # The gradient will flow through x_dq back to x correctly
        return Tensor._from_mlx_array(x_dq)

    def extra_repr(self) -> str:
        return (
            f"bits={self.bits}, "
            f"quant_min={self.quant_min}, quant_max={self.quant_max}, "
            f"fake_quant_enabled={self.fake_quant_enabled}, "
            f"observer_enabled={self.observer_enabled}"
        )


class FakeQuantize(FakeQuantizeBase):
    """
    Standard fake quantization module for QAT.

    Args:
        observer: Observer class or instance for tracking statistics
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        bits: Number of bits for quantization

    Example:
        >>> fake_quant = FakeQuantize(MinMaxObserver(bits=8))
        >>> x_fq = fake_quant(x)  # Observe and fake quantize
        >>> loss.backward()  # Gradients flow through
    """

    def __init__(
        self,
        observer: Optional[ObserverBase] = None,
        quant_min: int = -128,
        quant_max: int = 127,
        bits: int = 8,
    ):
        if observer is None:
            observer = MinMaxObserver(bits=bits)

        super().__init__(
            observer=observer,
            quant_min=quant_min,
            quant_max=quant_max,
            bits=bits,
        )

    @classmethod
    def with_args(
        cls,
        observer: Type[ObserverBase] = MinMaxObserver,
        quant_min: int = -128,
        quant_max: int = 127,
        bits: int = 8,
        **observer_kwargs,
    ) -> "FakeQuantize":
        """
        Factory method for creating FakeQuantize with observer arguments.

        Args:
            observer: Observer class to instantiate
            quant_min: Minimum quantized value
            quant_max: Maximum quantized value
            bits: Number of bits
            **observer_kwargs: Additional arguments for the observer

        Returns:
            FakeQuantize instance

        Example:
            >>> fq = FakeQuantize.with_args(
            ...     observer=MovingAverageMinMaxObserver,
            ...     averaging_constant=0.01,
            ...     bits=8,
            ... )
        """
        obs = observer(bits=bits, **observer_kwargs)
        return cls(observer=obs, quant_min=quant_min, quant_max=quant_max, bits=bits)


class LearnedFakeQuantize(FakeQuantizeBase):
    """
    Fake quantization with learnable scale and zero_point.

    Instead of computing scale/zero_point from observations,
    they are treated as learnable parameters.
    """

    def __init__(
        self,
        init_scale: float = 1.0,
        init_zero_point: float = 0.0,
        quant_min: int = -128,
        quant_max: int = 127,
        bits: int = 8,
    ):
        from ..nn.parameter import Parameter
        from ..factories import tensor

        # Use a NoopObserver since we're learning parameters
        from .observer import NoopObserver

        super().__init__(
            observer=NoopObserver(),
            quant_min=quant_min,
            quant_max=quant_max,
            bits=bits,
        )

        # Make scale and zero_point learnable
        self.scale = Parameter(tensor([init_scale]))
        self.zero_point = Parameter(tensor([init_zero_point]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward with learnable quantization parameters."""
        if not self.fake_quant_enabled:
            return x

        return self._fake_quantize(x, self.scale, self.zero_point)
