"""
Autograd System

Implements tape-based automatic differentiation for flashlight tensors.
"""

from .anomaly_mode import detect_anomaly, set_detect_anomaly
from .context import enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from .engine import _get_create_graph, backward, zero_grad, zero_grad_all
from .function import Function, GradientFunction
from .grad import grad

# Variable is a deprecated alias for Tensor (for backwards compatibility)
# In modern PyTorch, Variable and Tensor are the same thing
from ..tensor import Tensor as Variable

__all__ = [
    "Function",
    "GradientFunction",
    "Variable",
    "grad",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "backward",
    "zero_grad",
    "zero_grad_all",
    "detect_anomaly",
    "set_detect_anomaly",
    "_get_create_graph",
]
