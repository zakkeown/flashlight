"""
Stateless Module Utilities

PyTorch-compatible torch.nn.utils.stateless module.
Provides utilities for running modules with different parameters.
"""

import warnings
from typing import Any, Dict

from ...tensor import Tensor
from ..module import Module


def functional_call(
    module: Module,
    parameters_and_buffers: Dict[str, Any],
    args=None,
    kwargs=None,
    tie_weights: bool = True,
    strict: bool = False,
) -> Any:
    """
    Perform a functional call on a module with replacement parameters.

    Note: This is a simplified implementation in MLX.

    Args:
        module: The module to call
        parameters_and_buffers: Dict of replacement parameters
        args: Positional arguments for the module
        kwargs: Keyword arguments for the module
        tie_weights: Whether to tie weights
        strict: Whether to require all parameters to be replaced

    Returns:
        Output of the module
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    # Store original parameters
    original_params = {}
    for name, param in module.named_parameters():
        original_params[name] = param

    # Replace parameters
    try:
        for name, value in parameters_and_buffers.items():
            parts = name.split(".")
            obj = module
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        # Call the module
        return module(*args, **kwargs)
    finally:
        # Restore original parameters
        for name, param in original_params.items():
            parts = name.split(".")
            obj = module
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], param)


__all__ = [
    "functional_call",
]
