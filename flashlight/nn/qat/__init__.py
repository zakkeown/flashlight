"""
Quantization Aware Training (QAT)

PyTorch-compatible torch.nn.qat module stub.
MLX does not currently support quantization-aware training, so this module
provides stub implementations for API compatibility.
"""

import warnings

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Linear",
    "Embedding",
    "EmbeddingBag",
]


def _qat_not_supported():
    """Raise warning about QAT not being supported."""
    warnings.warn(
        "Quantization-aware training is not supported in MLX. "
        "QAT modules will behave like their non-quantized counterparts.",
        UserWarning,
    )


# Import regular layers as QAT versions (they'll just work without quantization)
from ..layers.conv import Conv1d as _Conv1d
from ..layers.conv import Conv2d as _Conv2d
from ..layers.conv import Conv3d as _Conv3d
from ..layers.embedding import Embedding as _Embedding
from ..layers.embedding import EmbeddingBag as _EmbeddingBag
from ..layers.linear import Linear as _Linear


class Conv1d(_Conv1d):
    """QAT Conv1d stub - behaves like regular Conv1d in MLX."""

    def __init__(self, *args, qconfig=None, **kwargs):
        _qat_not_supported()
        super().__init__(*args, **kwargs)
        self.qconfig = qconfig


class Conv2d(_Conv2d):
    """QAT Conv2d stub - behaves like regular Conv2d in MLX."""

    def __init__(self, *args, qconfig=None, **kwargs):
        _qat_not_supported()
        super().__init__(*args, **kwargs)
        self.qconfig = qconfig


class Conv3d(_Conv3d):
    """QAT Conv3d stub - behaves like regular Conv3d in MLX."""

    def __init__(self, *args, qconfig=None, **kwargs):
        _qat_not_supported()
        super().__init__(*args, **kwargs)
        self.qconfig = qconfig


class Linear(_Linear):
    """QAT Linear stub - behaves like regular Linear in MLX."""

    def __init__(self, *args, qconfig=None, **kwargs):
        _qat_not_supported()
        super().__init__(*args, **kwargs)
        self.qconfig = qconfig


class Embedding(_Embedding):
    """QAT Embedding stub - behaves like regular Embedding in MLX."""

    def __init__(self, *args, qconfig=None, **kwargs):
        _qat_not_supported()
        super().__init__(*args, **kwargs)
        self.qconfig = qconfig


class EmbeddingBag(_EmbeddingBag):
    """QAT EmbeddingBag stub - behaves like regular EmbeddingBag in MLX."""

    def __init__(self, *args, qconfig=None, **kwargs):
        _qat_not_supported()
        super().__init__(*args, **kwargs)
        self.qconfig = qconfig
