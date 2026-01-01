"""
Configuration for the parity check system.
"""

from typing import Dict, List
import sys

# PyTorch modules to check for API parity
PYTORCH_MODULES: List[str] = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.init",
    "torch.nn.attention",
    "torch.nn.parameter",
    "torch.nn.modules",
    "torch.nn.utils",
    "torch.nn.qat",
    "torch.nn.quantizable",
    "torch.optim",
    "torch.optim.lr_scheduler",
    "torch.linalg",
    "torch.autograd",
    "torch.utils.data",
    "torch.fft",
    "torch.special",
    "torch.distributions",
    "torch.amp",
    "torch.nn.grad",
    "torch.random",
    "torch.signal.windows",
    "torch.distributions.constraints",
    "torch.distributions.transforms",
    "torch.nn.utils.rnn",
    "torch.nn.utils.parametrizations",
    "torch.nn.utils.parametrize",
    "torch.nn.utils.stateless",
]

# Mapping from PyTorch modules to flashlight modules
MODULE_MAPPING: Dict[str, str] = {
    "torch": "flashlight",
    "torch.nn": "flashlight.nn",
    "torch.nn.functional": "flashlight.nn.functional",
    "torch.nn.init": "flashlight.nn.init",
    "torch.nn.attention": "flashlight.nn.attention",
    "torch.nn.parameter": "flashlight.nn.parameter",
    "torch.nn.modules": "flashlight.nn.modules",
    "torch.nn.utils": "flashlight.nn.utils",
    "torch.nn.qat": "flashlight.nn.qat",
    "torch.nn.quantizable": "flashlight.nn.quantizable",
    "torch.optim": "flashlight.optim",
    "torch.optim.lr_scheduler": "flashlight.optim.lr_scheduler",
    "torch.linalg": "flashlight.linalg",
    "torch.autograd": "flashlight.autograd",
    "torch.utils.data": "flashlight.data",
    "torch.fft": "flashlight.fft",
    "torch.special": "flashlight.special",
    "torch.distributions": "flashlight.distributions",
    "torch.amp": "flashlight.amp",
    "torch.nn.grad": "flashlight.nn.grad",
    "torch.random": "flashlight.random",
    "torch.signal.windows": "flashlight.signal.windows",
    "torch.distributions.constraints": "flashlight.distributions.constraints",
    "torch.distributions.transforms": "flashlight.distributions.transforms",
    "torch.nn.utils.rnn": "flashlight.nn.utils.rnn",
    "torch.nn.utils.parametrizations": "flashlight.nn.utils.parametrizations",
    "torch.nn.utils.parametrize": "flashlight.nn.utils.parametrize",
    "torch.nn.utils.stateless": "flashlight.nn.utils.stateless",
}

# API types we care about
API_TYPES = ["function", "class", "module"]


def get_pytorch_version() -> str:
    """Get the installed PyTorch version."""
    try:
        import torch
        return torch.__version__.split("+")[0]
    except ImportError:
        return "unknown"


def get_flashlight_version() -> str:
    """Get the installed flashlight version."""
    try:
        import flashlight
        return flashlight.__version__
    except ImportError:
        return "unknown"


# Version-specific APIs (added in specific PyTorch versions)
# Format: {"full.api.path": "minimum_version"}
VERSION_SPECIFIC_APIS: Dict[str, str] = {
    "torch.compile": "2.0.0",
    "torch.nn.RMSNorm": "2.4.0",
}

# Deprecated APIs that should be ignored
DEPRECATED_APIS: Dict[str, str] = {
    "torch.nn.Container": "1.8.0",
}


def should_check_api(module: str, api: str) -> bool:
    """
    Determine if an API should be checked based on PyTorch version.

    Args:
        module: The module name (e.g., "torch.nn")
        api: The API name (e.g., "Linear")

    Returns:
        True if the API should be checked, False otherwise
    """
    from packaging import version

    full_name = f"{module}.{api}"
    pytorch_version = version.parse(get_pytorch_version())

    # Check if this API was added in a newer version
    if full_name in VERSION_SPECIFIC_APIS:
        required_version = version.parse(VERSION_SPECIFIC_APIS[full_name])
        if pytorch_version < required_version:
            return False

    # Check if this API was deprecated
    if full_name in DEPRECATED_APIS:
        deprecated_in = version.parse(DEPRECATED_APIS[full_name])
        if pytorch_version >= deprecated_in:
            return False

    return True
