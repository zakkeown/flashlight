"""
Autograd System

Implements tape-based automatic differentiation for flashlight tensors.
"""

from .function import Function, GradientFunction
from .context import no_grad, enable_grad, set_grad_enabled, is_grad_enabled
from .engine import backward, zero_grad, zero_grad_all

__all__ = [
    'Function',
    'GradientFunction',
    'no_grad',
    'enable_grad',
    'set_grad_enabled',
    'is_grad_enabled',
    'backward',
    'zero_grad',
    'zero_grad_all',
]
