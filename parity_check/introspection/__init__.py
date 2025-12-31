"""
API introspection utilities for PyTorch and mlx_compat.
"""

from .pytorch_api import enumerate_pytorch_api, get_pytorch_api_info
from .mlx_api import enumerate_mlx_api, get_mlx_api_info
from .signature import extract_signature, compare_signatures
from .source_parser import (
    get_source_signature,
    extract_module_signatures,
    PYTORCH_REFERENCE_ROOT,
)
from .stub_parser import (
    get_stub_signature,
    get_stub_signature_for_module,
    get_all_stub_signatures,
    TORCH_STUBS_AVAILABLE,
)

__all__ = [
    "enumerate_pytorch_api",
    "get_pytorch_api_info",
    "enumerate_mlx_api",
    "get_mlx_api_info",
    "extract_signature",
    "compare_signatures",
    "get_source_signature",
    "extract_module_signatures",
    "PYTORCH_REFERENCE_ROOT",
    "get_stub_signature",
    "get_stub_signature_for_module",
    "get_all_stub_signatures",
    "TORCH_STUBS_AVAILABLE",
]
