"""
Autograd System

Implements tape-based automatic differentiation for flashlight tensors.
"""

from .context import enable_grad, is_grad_enabled, no_grad, set_grad_enabled
from .engine import backward, zero_grad, zero_grad_all
from .function import Function, GradientFunction

__all__ = [
    "Function",
    "GradientFunction",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "backward",
    "zero_grad",
    "zero_grad_all",
]
