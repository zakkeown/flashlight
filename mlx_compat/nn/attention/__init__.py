"""
Attention Mechanisms

PyTorch-compatible torch.nn.attention module for attention-related utilities.
Re-exports attention layer implementations for convenience.
"""

from contextlib import contextmanager
from enum import IntEnum
from typing import List, Union

from ..layers.attention import (
    MultiheadAttention,
    scaled_dot_product_attention,
)


class SDPBackend(IntEnum):
    """
    Enum for scaled dot product attention backend selection.

    MLX uses its own optimized backend, so these are provided for API compatibility.
    All backends effectively use the same MLX implementation.
    """
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2
    CUDNN_ATTENTION = 3
    OVERRIDEABLE = 4


# Warning flag for unfused kernels (not applicable to MLX)
WARN_FOR_UNFUSED_KERNELS = False


@contextmanager
def sdpa_kernel(backends: Union[SDPBackend, List[SDPBackend]], set_priority: bool = False):
    """
    Context manager to select SDPA backends.

    In MLX, this is a no-op since MLX uses its own optimized implementation.
    Provided for PyTorch API compatibility.

    Args:
        backends: Backend or list of backends to enable
        set_priority: Whether to set backend priority (not used in MLX)

    Yields:
        None
    """
    # MLX uses its own implementation, so this is a no-op
    yield


__all__ = [
    'MultiheadAttention',
    'scaled_dot_product_attention',
    'SDPBackend',
    'sdpa_kernel',
    'WARN_FOR_UNFUSED_KERNELS',
]
