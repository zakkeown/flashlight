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

# Mapping from PyTorch modules to mlx_compat modules
MODULE_MAPPING: Dict[str, str] = {
    "torch": "mlx_compat",
    "torch.nn": "mlx_compat.nn",
    "torch.nn.functional": "mlx_compat.nn.functional",
    "torch.nn.init": "mlx_compat.nn.init",
    "torch.nn.attention": "mlx_compat.nn.attention",
    "torch.nn.parameter": "mlx_compat.nn.parameter",
    "torch.nn.modules": "mlx_compat.nn.modules",
    "torch.nn.utils": "mlx_compat.nn.utils",
    "torch.nn.qat": "mlx_compat.nn.qat",
    "torch.nn.quantizable": "mlx_compat.nn.quantizable",
    "torch.optim": "mlx_compat.optim",
    "torch.optim.lr_scheduler": "mlx_compat.optim.lr_scheduler",
    "torch.linalg": "mlx_compat.linalg",
    "torch.autograd": "mlx_compat.autograd",
    "torch.utils.data": "mlx_compat.data",
    "torch.fft": "mlx_compat.fft",
    "torch.special": "mlx_compat.special",
    "torch.distributions": "mlx_compat.distributions",
    "torch.amp": "mlx_compat.amp",
    "torch.nn.grad": "mlx_compat.nn.grad",
    "torch.random": "mlx_compat.random",
    "torch.signal.windows": "mlx_compat.signal.windows",
    "torch.distributions.constraints": "mlx_compat.distributions.constraints",
    "torch.distributions.transforms": "mlx_compat.distributions.transforms",
    "torch.nn.utils.rnn": "mlx_compat.nn.utils.rnn",
    "torch.nn.utils.parametrizations": "mlx_compat.nn.utils.parametrizations",
    "torch.nn.utils.parametrize": "mlx_compat.nn.utils.parametrize",
    "torch.nn.utils.stateless": "mlx_compat.nn.utils.stateless",
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


def get_mlx_compat_version() -> str:
    """Get the installed mlx_compat version."""
    try:
        import mlx_compat
        return mlx_compat.__version__
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
