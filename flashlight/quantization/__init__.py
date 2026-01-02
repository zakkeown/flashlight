"""
Quantization Module

Provides PyTorch-compatible quantization APIs mapped to MLX's quantization capabilities.
MLX supports group-wise quantization for efficient inference on Apple Silicon.

Key components:
- quantize_per_tensor / quantize_per_channel: Quantization functions
- QuantizedTensor: Tensor with quantized storage
- Observers: MinMaxObserver, MovingAverageMinMaxObserver for calibration
- FakeQuantize: For quantization-aware training (QAT)
- prepare / convert / prepare_qat: Model transformation utilities
"""

from .core import (
    dequantize,
    quantize_dynamic,
    quantize_per_channel,
    quantize_per_tensor,
)
from .fake_quantize import FakeQuantize, FakeQuantizeBase, LearnedFakeQuantize
from .observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    NoopObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
)
from .qconfig import QConfig, default_qat_qconfig, default_qconfig
from .stubs import DeQuantStub, QuantStub
from .tensor import QuantizedTensor
from .utils import convert, prepare, prepare_qat

__all__ = [
    # Core functions
    "quantize_per_tensor",
    "quantize_per_channel",
    "quantize_dynamic",
    "dequantize",
    # Tensor
    "QuantizedTensor",
    # Observers
    "ObserverBase",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "PerChannelMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "HistogramObserver",
    "NoopObserver",
    # FakeQuantize
    "FakeQuantizeBase",
    "FakeQuantize",
    "LearnedFakeQuantize",
    # Stubs
    "QuantStub",
    "DeQuantStub",
    # QConfig
    "QConfig",
    "default_qconfig",
    "default_qat_qconfig",
    # Utils
    "prepare",
    "convert",
    "prepare_qat",
]
