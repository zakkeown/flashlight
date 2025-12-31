"""
Parser for PyTorch .pyi stub files to extract function signatures.

This module parses the generated .pyi stub files from installed PyTorch
to extract function signatures for C++ builtins that can't be introspected
via inspect.signature().
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to find installed PyTorch stubs
try:
    import torch
    TORCH_INSTALL_PATH = Path(torch.__file__).parent
    TORCH_STUBS_AVAILABLE = True
except ImportError:
    TORCH_INSTALL_PATH = None
    TORCH_STUBS_AVAILABLE = False


def get_stub_paths() -> Dict[str, Path]:
    """
    Get paths to PyTorch stub files.

    Returns:
        Dictionary mapping module names to stub file paths
    """
    if not TORCH_STUBS_AVAILABLE:
        return {}

    stubs = {}

    # Main variable functions stub
    var_funcs = TORCH_INSTALL_PATH / "_C" / "_VariableFunctions.pyi"
    if var_funcs.exists():
        stubs["torch._C._VariableFunctions"] = var_funcs

    # Main _C module stub
    c_init = TORCH_INSTALL_PATH / "_C" / "__init__.pyi"
    if c_init.exists():
        stubs["torch._C"] = c_init

    # nn module stubs
    nn_pyi = TORCH_INSTALL_PATH / "_C" / "_nn.pyi"
    if nn_pyi.exists():
        stubs["torch._C._nn"] = nn_pyi

    return stubs


def parse_stub_file(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse a .pyi stub file to extract function signatures.

    Handles overloaded functions by taking the most complete signature
    (the one with the most parameters or most specific types).

    Args:
        filepath: Path to the .pyi file

    Returns:
        Dictionary mapping function names to signature info
    """
    signatures = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
    except Exception as e:
        return signatures

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            sig = _parse_stub_function(node)

            # For overloaded functions, keep the signature with most parameters
            # or the first one if they have the same count
            if name in signatures:
                existing = signatures[name]
                existing_params = len(existing.get("parameters", []))
                new_params = len(sig.get("parameters", []))

                # Prefer signature with more parameters or with defaults
                if new_params > existing_params:
                    signatures[name] = sig
                elif new_params == existing_params:
                    # Check if new one has more defaults
                    new_defaults = sum(1 for p in sig["parameters"] if p.get("default"))
                    old_defaults = sum(1 for p in existing["parameters"] if p.get("default"))
                    if new_defaults > old_defaults:
                        signatures[name] = sig
            else:
                signatures[name] = sig

    return signatures


def _parse_stub_function(node: ast.FunctionDef) -> Dict[str, Any]:
    """
    Parse a function definition from a stub file.

    Args:
        node: AST FunctionDef node

    Returns:
        Signature dictionary
    """
    parameters = []

    # Parse positional-only args (before /)
    posonlyargs = getattr(node.args, 'posonlyargs', [])
    for arg in posonlyargs:
        parameters.append({
            "name": arg.arg,
            "kind": "POSITIONAL_ONLY",
            "default": None,
            "annotation": _get_annotation_str(arg.annotation),
        })

    # Parse regular args
    num_defaults = len(node.args.defaults)
    num_args = len(node.args.args)
    default_offset = num_args - num_defaults

    for i, arg in enumerate(node.args.args):
        # Skip 'self' and 'cls'
        if arg.arg in ('self', 'cls'):
            continue

        default = None
        if i >= default_offset:
            default_node = node.args.defaults[i - default_offset]
            default = _get_default_str(default_node)

        parameters.append({
            "name": arg.arg,
            "kind": "POSITIONAL_OR_KEYWORD",
            "default": default,
            "annotation": _get_annotation_str(arg.annotation),
        })

    # Parse *args
    if node.args.vararg:
        parameters.append({
            "name": node.args.vararg.arg,
            "kind": "VAR_POSITIONAL",
            "default": None,
            "annotation": _get_annotation_str(node.args.vararg.annotation),
        })

    # Parse keyword-only args
    for i, arg in enumerate(node.args.kwonlyargs):
        default = None
        if i < len(node.args.kw_defaults) and node.args.kw_defaults[i] is not None:
            default = _get_default_str(node.args.kw_defaults[i])

        parameters.append({
            "name": arg.arg,
            "kind": "KEYWORD_ONLY",
            "default": default,
            "annotation": _get_annotation_str(arg.annotation),
        })

    # Parse **kwargs
    if node.args.kwarg:
        parameters.append({
            "name": node.args.kwarg.arg,
            "kind": "VAR_KEYWORD",
            "default": None,
            "annotation": _get_annotation_str(node.args.kwarg.annotation),
        })

    return {
        "parameters": parameters,
        "return_annotation": _get_annotation_str(node.returns),
        "extractable": True,
        "source": "stub",
    }


def _get_annotation_str(annotation: Optional[ast.AST]) -> Optional[str]:
    """Convert an AST annotation node to a string."""
    if annotation is None:
        return None
    try:
        return ast.unparse(annotation)
    except Exception:
        return None


def _get_default_str(default: ast.AST) -> str:
    """Convert an AST default value node to a string."""
    try:
        return ast.unparse(default)
    except Exception:
        return repr(default)


# Cache for parsed stubs
_stub_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}


def get_stub_signature(func_name: str) -> Optional[Dict[str, Any]]:
    """
    Get signature for a function from PyTorch stubs.

    Args:
        func_name: Function name (e.g., 'add', 'matmul')

    Returns:
        Signature dictionary or None if not found
    """
    global _stub_cache

    if not TORCH_STUBS_AVAILABLE:
        return None

    # Parse stubs on first access
    if not _stub_cache:
        stub_paths = get_stub_paths()
        for module_name, path in stub_paths.items():
            _stub_cache[module_name] = parse_stub_file(path)

    # Search in all parsed stubs
    for module_name, sigs in _stub_cache.items():
        if func_name in sigs:
            return sigs[func_name]

    return None


def get_all_stub_signatures() -> Dict[str, Dict[str, Any]]:
    """
    Get all function signatures from PyTorch stubs.

    Returns:
        Dictionary mapping function names to signature info
    """
    global _stub_cache

    if not TORCH_STUBS_AVAILABLE:
        return {}

    # Parse stubs on first access
    if not _stub_cache:
        stub_paths = get_stub_paths()
        for module_name, path in stub_paths.items():
            _stub_cache[module_name] = parse_stub_file(path)

    # Merge all signatures
    all_sigs = {}
    for module_name, sigs in _stub_cache.items():
        all_sigs.update(sigs)

    return all_sigs


def clear_cache():
    """Clear the stub cache."""
    global _stub_cache
    _stub_cache = {}


# Module to stub file mapping for specific APIs
MODULE_STUB_MAPPING = {
    "torch": "torch._C._VariableFunctions",
    "torch.nn.functional": "torch._C._nn",
}


def get_stub_signature_for_module(module: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Get signature for a specific module.function from stubs.

    Args:
        module: Module name (e.g., 'torch', 'torch.nn.functional')
        name: Function name (e.g., 'add', 'relu')

    Returns:
        Signature dictionary or None
    """
    global _stub_cache

    if not TORCH_STUBS_AVAILABLE:
        return None

    # Parse stubs on first access
    if not _stub_cache:
        stub_paths = get_stub_paths()
        for module_name, path in stub_paths.items():
            _stub_cache[module_name] = parse_stub_file(path)

    # Try module-specific stub first
    stub_module = MODULE_STUB_MAPPING.get(module)
    if stub_module and stub_module in _stub_cache:
        if name in _stub_cache[stub_module]:
            return _stub_cache[stub_module][name]

    # Fall back to searching all stubs
    return get_stub_signature(name)
