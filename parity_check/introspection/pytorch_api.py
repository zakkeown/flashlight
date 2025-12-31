"""
PyTorch API introspection utilities.

Extracts the public API from PyTorch modules using __all__ and dir() introspection.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Set

from .signature import extract_signature


def classify_api_type(obj: Any) -> str:
    """
    Classify an API object into a type category.

    Args:
        obj: The object to classify

    Returns:
        One of: "function", "class", "module", "constant", "unknown"
    """
    if inspect.ismodule(obj):
        return "module"
    elif inspect.isclass(obj):
        return "class"
    elif inspect.isfunction(obj) or inspect.isbuiltin(obj) or callable(obj):
        return "function"
    else:
        return "constant"


def get_public_names(module) -> List[str]:
    """
    Get the public API names from a module.

    Uses __all__ if defined, otherwise filters dir() to exclude private names.

    Args:
        module: The module to inspect

    Returns:
        List of public API names
    """
    if hasattr(module, "__all__"):
        return list(module.__all__)

    # Fall back to dir() filtering
    return [name for name in dir(module) if not name.startswith("_")]


def get_pytorch_api_info(obj: Any, name: str) -> Dict[str, Any]:
    """
    Extract detailed information about a PyTorch API.

    Args:
        obj: The API object
        name: The name of the API

    Returns:
        Dictionary with API information
    """
    api_type = classify_api_type(obj)

    # Get docstring safely (some objects have property descriptors for __doc__)
    try:
        docstring = obj.__doc__
        if docstring and isinstance(docstring, str):
            docstring = docstring[:200]
        else:
            docstring = None
    except Exception:
        docstring = None

    info = {
        "name": name,
        "type": api_type,
        "module": getattr(obj, "__module__", None),
        "docstring": docstring,
    }

    # Extract signature for callables
    if api_type in ("function", "class"):
        info["signature"] = extract_signature(obj)

    return info


def enumerate_pytorch_api(modules: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Enumerate the public API of PyTorch modules.

    Args:
        modules: List of module names to inspect. If None, uses default list.

    Returns:
        Dictionary mapping module names to their API dictionaries.
        Format: {module_name: {api_name: api_info}}
    """
    from ..config import PYTORCH_MODULES

    if modules is None:
        modules = PYTORCH_MODULES

    result = {}

    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
            continue

        apis = {}
        public_names = get_public_names(module)

        for name in public_names:
            try:
                obj = getattr(module, name)
            except AttributeError:
                continue

            # Skip if it's a submodule that we'll enumerate separately
            if inspect.ismodule(obj) and f"{module_name}.{name}" in modules:
                continue

            apis[name] = get_pytorch_api_info(obj, name)

        result[module_name] = apis

    return result


def get_pytorch_nn_classes() -> Set[str]:
    """
    Get all nn.Module subclasses from torch.nn.

    Returns:
        Set of class names that are nn.Module subclasses
    """
    import torch.nn as nn

    classes = set()
    for name in dir(nn):
        if name.startswith("_"):
            continue
        obj = getattr(nn, name, None)
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            classes.add(name)

    return classes


def get_pytorch_optim_classes() -> Set[str]:
    """
    Get all optimizer classes from torch.optim.

    Returns:
        Set of class names that are Optimizer subclasses
    """
    import torch.optim as optim

    classes = set()
    for name in dir(optim):
        if name.startswith("_"):
            continue
        obj = getattr(optim, name, None)
        if inspect.isclass(obj) and issubclass(obj, optim.Optimizer):
            classes.add(name)

    return classes
