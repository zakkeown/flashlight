"""
Hook management utilities.

Implements PyTorch-compatible hook utilities for MLX.
"""

from collections import OrderedDict
import weakref
from typing import Any, Optional, Callable, Dict, List, Union, Tuple


__all__ = ["RemovableHandle", "unserializable_hook", "warn_if_has_hooks", "BackwardHook"]


class RemovableHandle:
    """
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict: A dictionary of hooks, indexed by hook id.
        extra_dict: An additional dictionary or list of dictionaries whose keys
            will be deleted when the same keys are removed from hooks_dict.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: Tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        """Remove the hook from the module."""
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (
                self.hooks_dict_ref(),
                self.id,
                tuple(ref() for ref in self.extra_dict_ref),
            )

    def __setstate__(self, state) -> None:
        if state[0] is None:
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2] if d is not None)

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


def unserializable_hook(f: Callable) -> Callable:
    """
    Mark a function as an unserializable hook with this decorator.

    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.

    Args:
        f: The hook function to mark.

    Returns:
        The same function with the unserializable marker.

    Example:
        >>> @unserializable_hook
        ... def my_hook(grad):
        ...     return grad * 2
    """
    f.__mlx_unserializable__ = True
    return f


def warn_if_has_hooks(tensor) -> None:
    """
    Warn if tensor has backward hooks that may not be serialized.

    Args:
        tensor: Tensor to check for hooks.
    """
    import warnings

    if hasattr(tensor, "_backward_hooks") and tensor._backward_hooks:
        for k in tensor._backward_hooks:
            hook = tensor._backward_hooks[k]
            if not hasattr(hook, "__mlx_unserializable__"):
                warnings.warn(
                    f"backward hook {repr(hook)} on tensor will not be "
                    "serialized. If this is expected, you can decorate the "
                    "function with @mlx_compat.utils.hooks.unserializable_hook "
                    "to suppress this warning",
                    stacklevel=2,
                )


class BackwardHook:
    """
    A wrapper class to implement nn.Module backward hooks.

    Handles:
    - Ignoring non-Tensor inputs and replacing them with None
    - Generating proper nodes to capture gradients
    - Linking output gradients with input gradients
    - Calling user hooks once both are available

    Note: MLX's autograd model differs from PyTorch's, so this is a
    simplified compatibility layer.

    Args:
        module: The module this hook is attached to.
        user_hooks: Dictionary of user backward hooks.
        user_pre_hooks: Dictionary of user backward pre-hooks.
    """

    def __init__(
        self,
        module: Any,
        user_hooks: Dict[int, Callable],
        user_pre_hooks: Dict[int, Callable],
    ) -> None:
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks
        self.module = module

        self.grad_outputs: Optional[Tuple] = None
        self.n_outputs: int = -1
        self.output_tensors_index: Optional[List[int]] = None
        self.n_inputs: int = -1
        self.input_tensors_index: Optional[List[int]] = None

    def _pack_with_none(
        self, indices: List[int], values: Tuple, size: int
    ) -> Tuple[Any, ...]:
        """Pack values at indices into list of given size, filling with None."""
        res: List[Any] = [None] * size
        for idx, val in zip(indices, values):
            res[idx] = val
        return tuple(res)

    def _unpack_none(self, indices: List[int], values: Tuple) -> Tuple[Any, ...]:
        """Unpack values at specified indices."""
        return tuple(values[idx] for idx in indices)

    def setup_input_hook(self, args: Tuple) -> Tuple:
        """
        Set up hook for input gradients.

        Args:
            args: The input arguments to the forward pass.

        Returns:
            The same args, unchanged.
        """
        from ..tensor import Tensor

        tensors_idx: List[int] = []
        tensors: List[Any] = []

        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                tensors_idx.append(i)
                tensors.append(arg)

        self.n_inputs = len(args)
        self.input_tensors_index = tensors_idx if tensors_idx else None

        return args

    def setup_output_hook(self, args: Union[Tuple, Any]) -> Union[Tuple, Any]:
        """
        Set up hook for output gradients.

        Args:
            args: The output(s) from the forward pass.

        Returns:
            The same args, unchanged.
        """
        from ..tensor import Tensor

        is_tuple = isinstance(args, tuple)
        if not is_tuple:
            args = (args,)

        tensors_idx: List[int] = []
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                tensors_idx.append(i)

        self.n_outputs = len(args)
        self.output_tensors_index = tensors_idx if tensors_idx else None

        if not is_tuple:
            return args[0]
        return args

    def __call__(
        self,
        grad_inputs: Tuple[Any, ...],
        grad_outputs: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        """
        Execute the backward hooks.

        Args:
            grad_inputs: Gradients with respect to module inputs.
            grad_outputs: Gradients with respect to module outputs.

        Returns:
            Modified grad_inputs after hook processing.
        """
        # Execute pre-hooks
        for hook in self.user_pre_hooks.values():
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                grad_inputs = result

        # Execute hooks
        for hook in self.user_hooks.values():
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                grad_inputs = result

        return grad_inputs
