"""
Quantization Configuration

QConfig specifies how to quantize weights and activations.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type

from .fake_quantize import FakeQuantize
from .observer import (
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
)


@dataclass
class QConfig:
    """
    Configuration for quantization.

    Specifies observer/fake_quantize factories for weights and activations.

    Attributes:
        activation: Factory for activation observer/fake_quantize
        weight: Factory for weight observer/fake_quantize

    Example:
        >>> qconfig = QConfig(
        ...     activation=MinMaxObserver.with_args(bits=8),
        ...     weight=PerChannelMinMaxObserver.with_args(bits=8),
        ... )
    """

    activation: Optional[Callable[[], ObserverBase]] = None
    weight: Optional[Callable[[], ObserverBase]] = None

    @classmethod
    def from_observers(
        cls,
        activation_observer: Type[ObserverBase] = MinMaxObserver,
        weight_observer: Type[ObserverBase] = PerChannelMinMaxObserver,
        bits: int = 8,
        **kwargs,
    ) -> "QConfig":
        """
        Create QConfig from observer classes.

        Args:
            activation_observer: Observer class for activations
            weight_observer: Observer class for weights
            bits: Number of bits for quantization
            **kwargs: Additional arguments for observers

        Returns:
            QConfig instance
        """
        return cls(
            activation=lambda: activation_observer(bits=bits, **kwargs),
            weight=lambda: weight_observer(bits=bits, symmetric=True, **kwargs),
        )


def _activation_observer_factory(bits: int = 8):
    """Factory for default activation observer."""
    return MinMaxObserver(bits=bits)


def _weight_observer_factory(bits: int = 8):
    """Factory for default weight observer."""
    return PerChannelMinMaxObserver(bits=bits, symmetric=True)


def _qat_activation_factory(bits: int = 8):
    """Factory for QAT activation fake quantization."""
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver(bits=bits),
        bits=bits,
    )


def _qat_weight_factory(bits: int = 8):
    """Factory for QAT weight fake quantization."""
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver(bits=bits, symmetric=True),
        bits=bits,
    )


# Default QConfig for post-training quantization (PTQ)
default_qconfig = QConfig(
    activation=lambda: _activation_observer_factory(8),
    weight=lambda: _weight_observer_factory(8),
)

# Default QConfig for quantization-aware training (QAT)
default_qat_qconfig = QConfig(
    activation=lambda: _qat_activation_factory(8),
    weight=lambda: _qat_weight_factory(8),
)

# Symmetric quantization config
symmetric_qconfig = QConfig(
    activation=lambda: MinMaxObserver(bits=8, symmetric=True),
    weight=lambda: PerChannelMinMaxObserver(bits=8, symmetric=True),
)

# 4-bit quantization config (for more aggressive compression)
int4_qconfig = QConfig(
    activation=lambda: MinMaxObserver(bits=4),
    weight=lambda: PerChannelMinMaxObserver(bits=4, symmetric=True),
)

# Dynamic quantization (activations not quantized)
dynamic_qconfig = QConfig(
    activation=None,
    weight=lambda: MinMaxObserver(bits=8, symmetric=True),
)


def get_default_qconfig(backend: str = "mlx") -> QConfig:
    """
    Get the default QConfig for a backend.

    Args:
        backend: Backend name (only "mlx" supported)

    Returns:
        Default QConfig for the backend
    """
    return default_qconfig


def get_default_qat_qconfig(backend: str = "mlx") -> QConfig:
    """
    Get the default QAT QConfig for a backend.

    Args:
        backend: Backend name (only "mlx" supported)

    Returns:
        Default QAT QConfig for the backend
    """
    return default_qat_qconfig
