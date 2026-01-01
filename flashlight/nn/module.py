"""
Neural Network Module Base Class

Implements PyTorch-compatible nn.Module for building neural networks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Set, Union

if TYPE_CHECKING:
    from ..tensor import Tensor
    from .parameter import Parameter

import mlx.core as mx


class RemovableHandle:
    """
    A handle which provides the capability to remove a hook.

    Returned by register_forward_hook and register_forward_pre_hook.
    """

    def __init__(self, hooks_dict: OrderedDict, id: int):
        self.hooks_dict = hooks_dict
        self.id = id

    def remove(self) -> None:
        """Remove the hook from the module."""
        if self.id in self.hooks_dict:
            del self.hooks_dict[self.id]


class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class. Modules can contain other modules,
    allowing to nest them in a tree structure.

    Example:
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(10, 5)
        ...
        ...     def forward(self, x):
        ...         return self.linear(x)
    """

    def __init__(self, *args, **kwargs):
        """Initialize the module."""
        # Use OrderedDict to maintain parameter/module order
        self._parameters: OrderedDict[str, "Parameter"] = OrderedDict()
        self._modules: OrderedDict[str, "Module"] = OrderedDict()
        self._buffers: OrderedDict[str, "Tensor"] = OrderedDict()
        self.training: bool = True
        # Hook registries
        self._forward_hooks: OrderedDict[int, Callable] = OrderedDict()
        self._forward_pre_hooks: OrderedDict[int, Callable] = OrderedDict()
        self._hook_id_counter: int = 0

    def forward(self, *args, **kwargs):
        """
        Define the forward pass computation.

        Should be overridden by all subclasses.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Output tensor(s)
        """
        raise NotImplementedError(
            f"Module {self.__class__.__name__} is missing the required 'forward' method"
        )

    def __call__(self, *args, **kwargs):
        """
        Call forward method with hook support.

        This allows the module to be called like a function.
        Hooks are called before and after forward().
        """
        # Run forward pre-hooks
        for hook in self._forward_pre_hooks.values():
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result

        # Call forward
        output = self.forward(*args, **kwargs)

        # Run forward hooks
        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, output)
            if hook_result is not None:
                output = hook_result

        return output

    def __setattr__(self, name: str, value) -> None:
        """
        Override setattr to register parameters and modules automatically.

        When you assign a Parameter or Module to an attribute, it's automatically
        registered with the parent module.
        """
        # Avoid recursion for internal attributes
        if name in (
            "_parameters",
            "_modules",
            "_buffers",
            "training",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_hook_id_counter",
        ):
            object.__setattr__(self, name, value)
            return

        # Check if we're initialized (have _parameters dict)
        if not hasattr(self, "_parameters"):
            object.__setattr__(self, name, value)
            return

        # Import here to avoid circular dependency
        from ..tensor import Tensor
        from .parameter import Parameter

        # Remove from old location if it exists
        if name in self._parameters:
            del self._parameters[name]
        if name in self._modules:
            del self._modules[name]
        if name in self._buffers:
            del self._buffers[name]

        # Register in appropriate dict
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and name != "grad":
            # Tensors that aren't parameters are registered as buffers
            # (e.g., running mean/var in BatchNorm)
            self._buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        """
        Get attribute, checking parameters and modules first.
        """
        if "_parameters" in self.__dict__:
            parameters = self.__dict__["_parameters"]
            if name in parameters:
                return parameters[name]

        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]

        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                return buffers[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """Delete attribute from parameters or modules."""
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        elif name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        """
        Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name: Name of the child module
            module: Child module to add
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        self._modules[name] = module

    def register_buffer(self, name: str, tensor: Optional["Tensor"]) -> None:
        """
        Add a buffer to the module.

        Buffers are tensors that should be saved and restored in state_dict,
        but not trained by the optimizer.

        Args:
            name: Name of the buffer
            tensor: Tensor to register as buffer
        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        self._buffers[name] = tensor

    def register_parameter(self, name: str, param: Optional["Parameter"]) -> None:
        """
        Add a parameter to the module.

        Args:
            name: Name of the parameter
            param: Parameter to register, or None
        """
        from .parameter import Parameter

        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter, got {type(param).__name__}")
        else:
            self._parameters[name] = param

    def register_forward_hook(self, hook: Callable) -> RemovableHandle:
        """
        Register a forward hook on the module.

        The hook will be called every time after forward() has computed an output.
        It should have the following signature:
            hook(module, input, output) -> None or modified output

        Args:
            hook: The user defined hook to be registered

        Returns:
            A handle that can be used to remove the added hook by calling handle.remove()
        """
        handle = RemovableHandle(self._forward_hooks, self._hook_id_counter)
        self._forward_hooks[self._hook_id_counter] = hook
        self._hook_id_counter += 1
        return handle

    def register_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        """
        Register a forward pre-hook on the module.

        The hook will be called every time before forward() is invoked.
        It should have the following signature:
            hook(module, input) -> None or modified input

        Args:
            hook: The user defined hook to be registered

        Returns:
            A handle that can be used to remove the added hook by calling handle.remove()
        """
        handle = RemovableHandle(self._forward_pre_hooks, self._hook_id_counter)
        self._forward_pre_hooks[self._hook_id_counter] = hook
        self._hook_id_counter += 1
        return handle

    def parameters(self, recurse: bool = True) -> Iterator["Parameter"]:
        """
        Return an iterator over module parameters.

        Args:
            recurse: If True, yields parameters of this module and all submodules.
                    Otherwise, yields only parameters that are direct members.

        Yields:
            Parameter: Module parameters

        Example:
            >>> for param in model.parameters():
            ...     print(param.shape)
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, "Parameter"]]:
        """
        Return an iterator over module parameters, yielding both name and parameter.

        Args:
            prefix: Prefix to prepend to all parameter names
            recurse: If True, yields parameters of this module and all submodules

        Yields:
            (str, Parameter): Tuple of parameter name and parameter
        """
        for name, param in self._parameters.items():
            yield (prefix + ("." if prefix else "") + name, param)

        if recurse:
            for name, module in self._modules.items():
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_parameters(prefix=submodule_prefix, recurse=recurse)

    def children(self) -> Iterator["Module"]:
        """
        Return an iterator over immediate child modules.

        Yields:
            Module: Child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        """
        Return an iterator over immediate child modules, yielding name and module.

        Yields:
            (str, Module): Tuple of module name and module
        """
        for name, module in self._modules.items():
            yield name, module

    def modules(self) -> Iterator["Module"]:
        """
        Return an iterator over all modules in the network.

        Yields:
            Module: A module in the network

        Note:
            Duplicate modules are returned only once.
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(
        self, memo: Optional[Set[int]] = None, prefix: str = ""
    ) -> Iterator[tuple[str, "Module"]]:
        """
        Return an iterator over all modules in the network, yielding name and module.

        Args:
            memo: Set of module ids to avoid duplicates
            prefix: Prefix to prepend to all module names

        Yields:
            (str, Module): Tuple of module name and module
        """
        if memo is None:
            memo = set()

        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self

            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(memo, submodule_prefix)

    def train(self, mode: bool = True) -> "Module":
        """
        Set the module in training mode.

        This affects certain modules like Dropout and BatchNorm.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)

        Returns:
            self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """
        Set the module in evaluation mode.

        This is equivalent to self.train(False).

        Returns:
            self
        """
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> "Module":
        """
        Change if autograd should record operations on parameters.

        Args:
            requires_grad: Whether to require gradients

        Returns:
            self
        """
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Set gradients of all model parameters to zero.

        Args:
            set_to_none: If True, set gradients to None instead of zero
                        (more efficient, default True)
        """
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def apply(self, fn: Callable[["Module"], None]) -> "Module":
        """
        Apply a function to every submodule (as returned by .children())
        as well as self.

        Typical use includes initializing the parameters of a model.

        Args:
            fn: Function to be applied to each submodule

        Returns:
            self
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def to(self, *args, **kwargs) -> "Module":
        """
        Move and/or cast the parameters and buffers.

        Args:
            device: Target device (for compatibility, ignored in MLX)
            dtype: Target dtype

        Returns:
            self
        """
        # Extract dtype if provided
        dtype = None
        for arg in args:
            if hasattr(arg, "_mlx_dtype"):  # It's a DType
                dtype = arg
                break

        if "dtype" in kwargs:
            dtype = kwargs["dtype"]

        # Convert all parameters and buffers
        if dtype is not None:
            for param in self.parameters():
                param.data = param.data.to(dtype=dtype)

            for name, buf in self._buffers.items():
                self._buffers[name] = buf.to(dtype=dtype)

        return self

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Return a dictionary containing the module's state.

        Args:
            destination: Dictionary to store state in
            prefix: Prefix for parameter names

        Returns:
            dict: State dictionary
        """
        if destination is None:
            destination = OrderedDict()

        # Save parameters
        for name, param in self._parameters.items():
            destination[prefix + name] = param.data

        # Save buffers
        for name, buf in self._buffers.items():
            destination[prefix + name] = buf

        # Recursively save child modules
        for name, module in self._modules.items():
            module.state_dict(destination, prefix + name + ".")

        return destination

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Load parameters and buffers from state_dict.

        Args:
            state_dict: Dictionary containing parameters and buffers
            strict: Whether to strictly enforce that keys match
        """
        if strict:
            # Check for unexpected keys
            expected_keys = set(self.state_dict().keys())
            provided_keys = set(state_dict.keys())
            unexpected = provided_keys - expected_keys
            if unexpected:
                raise RuntimeError(f"Unexpected keys in state_dict: {unexpected}")

        # Load parameters
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data = state_dict[name]
            elif strict:
                raise KeyError(f"Missing key in state_dict: {name}")

        # Load buffers
        for name, buf in self._buffers.items():
            if name in state_dict:
                self._buffers[name] = state_dict[name]
            elif strict:
                raise KeyError(f"Missing key in state_dict: {name}")

        # Recursively load child modules
        for name, module in self._modules.items():
            # Filter state dict for this module
            prefix = name + "."
            module_state = {
                k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
            }
            if module_state or not strict:
                module.load_state_dict(module_state, strict=strict)

    def __repr__(self) -> str:
        """String representation of the module."""
        # We treat the extra repr like the original nn.Module
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = self._add_indent(mod_str, 2)
            child_lines.append(f"({key}): {mod_str}")

        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.

        To print customized extra information, you should override this method.

        Returns:
            str: Extra information string
        """
        return ""

    @staticmethod
    def _add_indent(s: str, num_spaces: int) -> str:
        """Add indentation to a string."""
        lines = s.split("\n")
        if len(lines) == 1:
            return s
        first = lines[0]
        rest = [" " * num_spaces + line for line in lines[1:]]
        return "\n".join([first] + rest)


__all__ = ["Module"]
