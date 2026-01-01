"""
Container Modules

Implements PyTorch-compatible container modules:
- Sequential: Sequential container of modules
- ModuleList: List of modules
- ModuleDict: Dictionary of modules
"""

from collections import OrderedDict
from typing import Dict, Iterable, Iterator, Optional, Union

from ..tensor import Tensor
from .module import Module

# Module types that benefit from NHWC layout (spatial operations)
_SPATIAL_LAYER_NAMES = frozenset(
    {
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        "GroupNorm",
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "AdaptiveMaxPool1d",
        "AdaptiveMaxPool2d",
        "AdaptiveMaxPool3d",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
        "ZeroPad1d",
        "ZeroPad2d",
        "ReflectionPad1d",
        "ReflectionPad2d",
        "ReplicationPad1d",
        "ReplicationPad2d",
        "ConstantPad1d",
        "ConstantPad2d",
        "ConstantPad3d",
    }
)


def _is_spatial_layer(module: Module) -> bool:
    """Check if module is a spatial layer that benefits from NHWC mode."""
    return type(module).__name__ in _SPATIAL_LAYER_NAMES


def _has_consecutive_spatial_layers(modules) -> bool:
    """
    Check if there are 2+ consecutive spatial layers.

    This is the threshold where NHWC mode becomes beneficial - avoiding
    at least one intermediate NCHW<->NHWC conversion.
    """
    consecutive = 0
    max_consecutive = 0
    for module in modules:
        if _is_spatial_layer(module):
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            # Non-spatial layers like ReLU, Dropout don't break the chain
            # because they're layout-agnostic
            layer_name = type(module).__name__
            layout_agnostic = layer_name in {
                "ReLU",
                "LeakyReLU",
                "PReLU",
                "ELU",
                "SELU",
                "GELU",
                "Sigmoid",
                "Tanh",
                "Softmax",
                "LogSoftmax",
                "Dropout",
                "Dropout2d",
                "Dropout3d",
                "Identity",
                "Flatten",
            }
            if not layout_agnostic:
                consecutive = 0
    return max_consecutive >= 2


class Sequential(Module):
    """
    Sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an OrderedDict of modules can be passed in.

    The forward() method of Sequential accepts any input and forwards it to
    the first module it contains. It then "chains" outputs to inputs
    sequentially for each subsequent module, finally returning the output
    of the last module.

    Example:
        >>> # Using Sequential to create a small model
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 5)
        ... )
        >>> x = flashlight.randn(3, 10)
        >>> output = model(x)
        >>> print(output.shape)  # (3, 5)

        >>> # Using Sequential with OrderedDict
        >>> model = nn.Sequential(OrderedDict([
        ...     ('fc1', nn.Linear(10, 20)),
        ...     ('relu1', nn.ReLU()),
        ...     ('fc2', nn.Linear(20, 5))
        ... ]))
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # Cache whether this Sequential benefits from NHWC mode
        self._use_nhwc_optimization = None

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self._modules)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return list(iterator.values())[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules, idx)

    def _get_key_by_idx(self, idx):
        """Get the key at idx-th position"""
        size = len(self._modules)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return list(self._modules.keys())[idx]

    def __setitem__(self, idx, module):
        key = self._get_key_by_idx(idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_key_by_idx(idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def _should_use_nhwc(self) -> bool:
        """Check if this Sequential should use NHWC optimization."""
        if self._use_nhwc_optimization is None:
            self._use_nhwc_optimization = _has_consecutive_spatial_layers(self._modules.values())
        return self._use_nhwc_optimization

    def forward(self, input: Tensor) -> Tensor:
        # Use NHWC optimization for sequences with consecutive spatial layers
        # and 4D input (spatial data)
        if input.ndim == 4 and self._should_use_nhwc():
            from ..layout import ensure_nchw, is_nhwc_mode, nhwc_mode

            # Only apply optimization if we're not already in NHWC mode
            if not is_nhwc_mode():
                with nhwc_mode():
                    for module in self._modules.values():
                        input = module(input)
                # Ensure output is in NCHW format for PyTorch compatibility
                return ensure_nchw(input)

        # Standard forward pass
        for module in self._modules.values():
            input = module(input)
        return input

    def append(self, module: Module) -> "Sequential":
        """
        Append a module to the end of the Sequential.

        Args:
            module: Module to append

        Returns:
            This Sequential for chaining
        """
        self.add_module(str(len(self)), module)
        self._use_nhwc_optimization = None  # Invalidate cache
        return self

    def extend(self, modules: Iterable[Module]) -> "Sequential":
        """
        Append modules from an iterable to the end of the Sequential.

        Args:
            modules: Iterable of modules to append

        Returns:
            This Sequential for chaining
        """
        for module in modules:
            self.append(module)
        # Cache already invalidated by append
        return self


class ModuleList(Module):
    """
    Holds modules in a list.

    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Args:
        modules: An iterable of modules to add

    Example:
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linears = nn.ModuleList([
        ...             nn.Linear(10, 10) for i in range(10)
        ...         ])
        ...
        ...     def forward(self, x):
        ...         # ModuleList can act as an iterable, or be indexed
        ...         for i, l in enumerate(self.linears):
        ...             x = self.linears[i // 2](x) + l(x)
        ...         return x
    """

    def __init__(self, modules: Optional[Iterable[Module]] = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = idx.__index__()
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # Re-index remaining modules
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module: Module) -> "ModuleList":
        """
        Append a module to the end of the list.

        Args:
            module: Module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        """
        Append modules from a Python iterable to the end of the list.

        Args:
            modules: Iterable of modules to append
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend expects an iterable")
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def insert(self, index: int, module: Module) -> None:
        """
        Insert a module before a given index in the list.

        Args:
            index: Index to insert before
            module: Module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module


class ModuleDict(Module):
    """
    Holds modules in a dictionary.

    ModuleDict can be indexed like a regular Python dictionary, but modules it
    contains are properly registered, and will be visible by all Module methods.

    ModuleDict is an ordered dictionary that respects the order of insertion.

    Args:
        modules: A mapping (dictionary) of (string: module) or an iterable of
            key-value pairs of type (string, module)

    Example:
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.choices = nn.ModuleDict({
        ...             'conv': nn.Linear(10, 10),
        ...             'pool': nn.Linear(10, 10)
        ...         })
        ...         self.activations = nn.ModuleDict({
        ...             'relu': nn.ReLU(),
        ...             'sigmoid': nn.Sigmoid()
        ...         })
        ...
        ...     def forward(self, x, choice, act):
        ...         x = self.choices[choice](x)
        ...         x = self.activations[act](x)
        ...         return x
    """

    def __init__(self, modules: Optional[Union[Dict[str, Module], Iterable]] = None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self) -> Iterator[str]:
        """Return an iterator over module keys."""
        return self._modules.keys()

    def items(self) -> Iterator:
        """Return an iterator over (key, module) pairs."""
        return self._modules.items()

    def values(self) -> Iterator[Module]:
        """Return an iterator over modules."""
        return self._modules.values()

    def update(self, modules: Union[Dict[str, Module], Iterable]) -> None:
        """
        Update the ModuleDict with key-value pairs from a mapping or iterable.

        Args:
            modules: A mapping (dictionary) or iterable of key-value pairs
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleDict.update expects an iterable")

        if isinstance(modules, dict):
            for key, module in modules.items():
                self.add_module(key, module)
        else:
            for key, module in modules:
                self.add_module(key, module)

    def pop(self, key: str) -> Module:
        """
        Remove and return the module with the given key.

        Args:
            key: Key of the module to remove

        Returns:
            The removed module
        """
        module = self._modules[key]
        del self._modules[key]
        return module


class ParameterList(Module):
    """
    Holds parameters in a list.

    ParameterList can be indexed like a regular Python list, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Args:
        parameters: An iterable of parameters to add

    Example:
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.params = nn.ParameterList([
        ...             nn.Parameter(torch.randn(10, 10)) for i in range(10)
        ...         ])
        ...
        ...     def forward(self, x):
        ...         for param in self.params:
        ...             x = x @ param
        ...         return x
    """

    def __init__(self, values: Optional[Iterable["Parameter"]] = None):
        super().__init__()
        if values is not None:
            self.extend(values)

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of parameters"""
        idx = idx.__index__()
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ParameterList(list(self._parameters.values())[idx])
        else:
            return self._parameters[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, param):
        idx = self._get_abs_string_index(idx)
        # Use setattr which triggers Module's __setattr__ to register parameter
        setattr(self, idx, param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def append(self, parameter: "Parameter") -> "ParameterList":
        """
        Append a parameter to the end of the list.

        Args:
            parameter: Parameter to append
        """
        setattr(self, str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable["Parameter"]) -> "ParameterList":
        """
        Append parameters from a Python iterable to the end of the list.

        Args:
            parameters: Iterable of parameters to append
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParameterList.extend expects an iterable")
        offset = len(self)
        for i, param in enumerate(parameters):
            setattr(self, str(offset + i), param)
        return self


class ParameterDict(Module):
    """
    Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters
    it contains are properly registered, and will be visible by all Module methods.

    Args:
        parameters: A mapping of (string: parameter) or an iterable of
            key-value pairs of type (string, parameter)

    Example:
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.params = nn.ParameterDict({
        ...             'left': nn.Parameter(torch.randn(5, 10)),
        ...             'right': nn.Parameter(torch.randn(5, 10))
        ...         })
        ...
        ...     def forward(self, x, choice):
        ...         return x @ self.params[choice]
    """

    def __init__(self, parameters: Optional[Union[Dict[str, "Parameter"], Iterable]] = None):
        super().__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> "Parameter":
        return self._parameters[key]

    def __setitem__(self, key: str, parameter: "Parameter") -> None:
        setattr(self, key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def keys(self):
        """Return an iterable of the ParameterDict keys."""
        return self._parameters.keys()

    def values(self):
        """Return an iterable of the ParameterDict values."""
        return self._parameters.values()

    def items(self):
        """Return an iterable of the ParameterDict key/value pairs."""
        return self._parameters.items()

    def update(self, parameters) -> None:
        """
        Update the ParameterDict with the key-value pairs from a mapping or an iterable.

        Args:
            parameters: A mapping of (string: parameter) or an iterable of
                key-value pairs of type (string, parameter)
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParameterDict.update expects an iterable")

        if isinstance(parameters, dict):
            for key, param in parameters.items():
                setattr(self, key, param)
        else:
            for key, param in parameters:
                setattr(self, key, param)

    def pop(self, key: str) -> "Parameter":
        """
        Remove and return the parameter with the given key.

        Args:
            key: Key of the parameter to remove

        Returns:
            The removed parameter
        """
        param = self._parameters[key]
        del self._parameters[key]
        return param


class Container(Module):
    """
    Deprecated base class for container modules.

    This class is deprecated in favor of Module. It exists only for backwards
    compatibility with old code that may still reference it.

    Warning:
        This class is deprecated. Use Module directly as a base class instead.
    """

    def __init__(self, **kwargs):
        import warnings

        warnings.warn(
            "nn.Container is deprecated. Use nn.Module instead.", DeprecationWarning, stacklevel=2
        )
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


__all__ = ["Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict", "Container"]
