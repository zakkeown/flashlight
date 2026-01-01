"""
Parametrization Utilities

PyTorch-compatible torch.nn.utils.parametrize module.
Provides utilities for module parametrization.
"""

import warnings
from typing import Optional, Union, Sequence
from contextlib import contextmanager

from ..module import Module
from ...tensor import Tensor


class ParametrizationList(Module):
    """
    A sequential container that holds and manages parametrizations.

    This is used internally to store parametrizations applied to a module's parameter.
    """

    def __init__(
        self,
        modules: Sequence[Module],
        original: Union[Tensor, "Parameter"],
        unsafe: bool = False,
    ):
        super().__init__()
        self.original = original
        self._modules_list = list(modules)
        self.unsafe = unsafe

    def forward(self, x):
        for module in self._modules_list:
            x = module(x)
        return x

    def right_inverse(self, x):
        """Compute the right inverse through all parametrizations."""
        for module in reversed(self._modules_list):
            if hasattr(module, 'right_inverse'):
                x = module.right_inverse(x)
        return x


def is_parametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    """
    Check if a module has parametrizations.

    Args:
        module: The module to check
        tensor_name: If provided, check if this specific tensor is parametrized

    Returns:
        True if the module (or tensor) is parametrized
    """
    if not hasattr(module, 'parametrizations'):
        return False

    if tensor_name is None:
        # Check if any parametrizations exist
        return len(module.parametrizations) > 0

    return tensor_name in module.parametrizations


def register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False,
) -> Module:
    """
    Register a parametrization to a tensor in a module.

    Args:
        module: The module containing the tensor to parametrize
        tensor_name: Name of the tensor to parametrize
        parametrization: The parametrization module
        unsafe: If True, skip validity checks

    Returns:
        The module with parametrization registered
    """
    if not hasattr(module, 'parametrizations'):
        module.parametrizations = {}

    if tensor_name not in module.parametrizations:
        # Get the original tensor
        original = getattr(module, tensor_name)
        module.parametrizations[tensor_name] = ParametrizationList(
            [parametrization], original, unsafe=unsafe
        )
    else:
        # Add to existing parametrization list
        module.parametrizations[tensor_name]._modules_list.append(parametrization)

    return module


def remove_parametrizations(
    module: Module,
    tensor_name: str,
    leave_parametrized: bool = True,
) -> Module:
    """
    Remove parametrizations from a tensor.

    Args:
        module: The module with parametrizations
        tensor_name: Name of the tensor to remove parametrizations from
        leave_parametrized: If True, replace tensor with parametrized value

    Returns:
        The module with parametrizations removed
    """
    if not hasattr(module, 'parametrizations'):
        return module

    if tensor_name not in module.parametrizations:
        return module

    param_list = module.parametrizations[tensor_name]

    if leave_parametrized:
        # Set the tensor to its current parametrized value
        parametrized_value = param_list(param_list.original)
        setattr(module, tensor_name, parametrized_value)
    else:
        # Restore the original tensor
        setattr(module, tensor_name, param_list.original)

    del module.parametrizations[tensor_name]

    if len(module.parametrizations) == 0:
        delattr(module, 'parametrizations')

    return module


@contextmanager
def cached():
    """
    Context manager to enable caching of parametrizations.

    In MLX, this is a no-op since we don't have the same caching mechanism.
    """
    yield


def type_before_parametrizations(module: Module) -> type:
    """
    Get the type of a module before any parametrizations were applied.

    Args:
        module: The module to check

    Returns:
        The original type of the module
    """
    return type(module)


def transfer_parametrizations_and_params(
    from_module: Module,
    to_module: Module,
    tensor_name: Optional[str] = None,
) -> Module:
    """
    Transfer parametrizations from one module to another.

    Args:
        from_module: Source module with parametrizations
        to_module: Target module to transfer to
        tensor_name: If provided, only transfer this tensor's parametrizations

    Returns:
        The target module with transferred parametrizations
    """
    if not hasattr(from_module, 'parametrizations'):
        return to_module

    if tensor_name is not None:
        if tensor_name in from_module.parametrizations:
            if not hasattr(to_module, 'parametrizations'):
                to_module.parametrizations = {}
            to_module.parametrizations[tensor_name] = from_module.parametrizations[tensor_name]
    else:
        to_module.parametrizations = from_module.parametrizations.copy()

    return to_module


__all__ = [
    'ParametrizationList',
    'is_parametrized',
    'register_parametrization',
    'remove_parametrizations',
    'cached',
    'type_before_parametrizations',
    'transfer_parametrizations_and_params',
]
