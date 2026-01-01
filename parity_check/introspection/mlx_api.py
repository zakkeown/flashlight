"""
flashlight API introspection utilities.

Extracts the public API from flashlight modules for comparison with PyTorch.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional

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


def get_mlx_api_info(obj: Any, name: str) -> Dict[str, Any]:
    """
    Extract detailed information about an flashlight API.

    Args:
        obj: The API object
        name: The name of the API

    Returns:
        Dictionary with API information
    """
    api_type = classify_api_type(obj)

    info = {
        "name": name,
        "type": api_type,
        "module": getattr(obj, "__module__", None),
        "docstring": obj.__doc__[:200] if obj.__doc__ else None,
    }

    # Extract signature for callables
    if api_type in ("function", "class"):
        info["signature"] = extract_signature(obj)

    return info


def enumerate_mlx_api(modules: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Enumerate the public API of flashlight modules.

    Args:
        modules: List of module names to inspect. If None, uses default list.

    Returns:
        Dictionary mapping module names to their API dictionaries.
        Format: {module_name: {api_name: api_info}}

    Note:
        The returned module names use the PyTorch naming convention (e.g., "torch.nn")
        for easy comparison, even though the actual modules are from flashlight.
    """
    from ..config import MODULE_MAPPING, PYTORCH_MODULES

    if modules is None:
        modules = PYTORCH_MODULES

    result = {}

    for pytorch_module_name in modules:
        # Map PyTorch module name to flashlight module name
        mlx_module_name = MODULE_MAPPING.get(pytorch_module_name)
        if mlx_module_name is None:
            continue

        try:
            module = importlib.import_module(mlx_module_name)
        except ImportError as e:
            # Module not implemented yet - return empty dict
            result[pytorch_module_name] = {}
            continue

        apis = {}
        public_names = get_public_names(module)

        for name in public_names:
            try:
                obj = getattr(module, name)
            except AttributeError:
                continue

            # Skip if it's a submodule that we'll enumerate separately
            # (matches PyTorch behavior in pytorch_api.py)
            if inspect.ismodule(obj) and f"{pytorch_module_name}.{name}" in modules:
                continue

            apis[name] = get_mlx_api_info(obj, name)

        # Use PyTorch module name as key for comparison
        result[pytorch_module_name] = apis

    return result
