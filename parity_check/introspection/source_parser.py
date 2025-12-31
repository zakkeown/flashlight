"""
AST-based signature extraction from PyTorch reference source.

This module parses the reference PyTorch source code to extract function and class
signatures, bypassing the limitation that C++ builtins can't be introspected via
inspect.signature().
"""

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path to the reference PyTorch source
PYTORCH_REFERENCE_ROOT = Path(__file__).parent.parent.parent / "reference" / "pytorch"


def get_source_file(module: str) -> Optional[Path]:
    """
    Get the source file path for a PyTorch module.

    Args:
        module: Module name like 'torch', 'torch.nn', 'torch.optim'

    Returns:
        Path to the source file, or None if not found
    """
    # Map module name to file path
    parts = module.split(".")

    if parts[0] != "torch":
        return None

    if len(parts) == 1:
        # torch -> torch/__init__.py
        return PYTORCH_REFERENCE_ROOT / "torch" / "__init__.py"

    # torch.nn -> torch/nn/__init__.py or torch/nn.py
    subpath = "/".join(parts[1:])
    init_path = PYTORCH_REFERENCE_ROOT / "torch" / subpath / "__init__.py"
    if init_path.exists():
        return init_path

    module_path = PYTORCH_REFERENCE_ROOT / "torch" / f"{subpath}.py"
    if module_path.exists():
        return module_path

    return None


def parse_function_signature(node: ast.FunctionDef) -> Dict[str, Any]:
    """
    Parse a function definition node to extract signature info.

    Args:
        node: AST FunctionDef node

    Returns:
        Dictionary with signature information matching extract_signature format
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
        "source": "ast",
    }


def _get_annotation_str(annotation: Optional[ast.AST]) -> Optional[str]:
    """Convert an AST annotation node to a string."""
    if annotation is None:
        return None
    return ast.unparse(annotation)


def _get_default_str(default: ast.AST) -> str:
    """Convert an AST default value node to a string."""
    try:
        return ast.unparse(default)
    except Exception:
        return repr(default)


def parse_class_init(node: ast.ClassDef) -> Optional[Dict[str, Any]]:
    """
    Parse a class definition to extract __init__ signature.

    Args:
        node: AST ClassDef node

    Returns:
        Dictionary with __init__ signature, or None if no __init__
    """
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return parse_function_signature(item)
    return None


def extract_signatures_from_file(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """
    Extract all function and class signatures from a Python source file.

    Args:
        filepath: Path to the Python source file

    Returns:
        Dictionary mapping names to signature info
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
            # Top-level functions
            if not node.name.startswith('_') or node.name == '__init__':
                signatures[node.name] = parse_function_signature(node)

        elif isinstance(node, ast.ClassDef):
            # Classes - extract __init__ signature
            if not node.name.startswith('_'):
                init_sig = parse_class_init(node)
                if init_sig:
                    signatures[node.name] = init_sig

    return signatures


def extract_module_signatures(module: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract signatures for a module from reference source.

    This searches the reference source directory structure to find all
    relevant source files for the module.

    Args:
        module: Module name like 'torch.optim'

    Returns:
        Dictionary mapping API names to signature info
    """
    signatures = {}
    parts = module.split(".")

    if parts[0] != "torch":
        return signatures

    # Build path to module directory/file
    if len(parts) == 1:
        # torch module - __init__.py
        base_dir = PYTORCH_REFERENCE_ROOT / "torch"
        if (base_dir / "__init__.py").exists():
            signatures.update(extract_signatures_from_file(base_dir / "__init__.py"))
    else:
        submodule = "/".join(parts[1:])
        base_dir = PYTORCH_REFERENCE_ROOT / "torch" / submodule

        # Check if it's a package or module
        if base_dir.is_dir():
            # Package - check __init__.py and all .py files
            init_file = base_dir / "__init__.py"
            if init_file.exists():
                signatures.update(extract_signatures_from_file(init_file))

            # Also scan individual files in the package
            for py_file in base_dir.glob("*.py"):
                if py_file.name != "__init__.py":
                    file_sigs = extract_signatures_from_file(py_file)
                    # Prefix with module structure for debugging
                    for name, sig in file_sigs.items():
                        sig["source_file"] = str(py_file.relative_to(PYTORCH_REFERENCE_ROOT))
                        signatures[name] = sig
        else:
            # Single module file
            module_file = PYTORCH_REFERENCE_ROOT / "torch" / f"{submodule}.py"
            if module_file.exists():
                signatures.update(extract_signatures_from_file(module_file))

    return signatures


def get_reference_signature(module: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific signature from the reference source.

    Args:
        module: Module name like 'torch.optim'
        name: API name like 'Adam'

    Returns:
        Signature info dictionary, or None if not found
    """
    sigs = extract_module_signatures(module)
    return sigs.get(name)


# Mapping from module.name to source file for common APIs
# This helps when the API is defined in a submodule but exported at a higher level
SOURCE_LOCATIONS = {
    # Optimizers
    "torch.optim.Adam": "torch/optim/adam.py",
    "torch.optim.AdamW": "torch/optim/adamw.py",
    "torch.optim.SGD": "torch/optim/sgd.py",
    "torch.optim.RMSprop": "torch/optim/rmsprop.py",
    "torch.optim.Adagrad": "torch/optim/adagrad.py",
    "torch.optim.Adadelta": "torch/optim/adadelta.py",
    "torch.optim.Adamax": "torch/optim/adamax.py",
    "torch.optim.ASGD": "torch/optim/asgd.py",
    "torch.optim.Rprop": "torch/optim/rprop.py",
    "torch.optim.NAdam": "torch/optim/nadam.py",
    "torch.optim.RAdam": "torch/optim/radam.py",
    "torch.optim.Adafactor": "torch/optim/_adafactor.py",
    # Transformers
    "torch.nn.Transformer": "torch/nn/modules/transformer.py",
    "torch.nn.TransformerEncoder": "torch/nn/modules/transformer.py",
    "torch.nn.TransformerDecoder": "torch/nn/modules/transformer.py",
    "torch.nn.TransformerEncoderLayer": "torch/nn/modules/transformer.py",
    "torch.nn.TransformerDecoderLayer": "torch/nn/modules/transformer.py",
}

# Mapping from public module to source directories
# torch.nn exports classes from torch/nn/modules/
MODULE_SOURCE_DIRS = {
    "torch.nn": [
        "torch/nn/modules",
        "torch/nn",
    ],
    "torch.nn.functional": [
        "torch/nn/functional.py",
    ],
    "torch.utils.data": [
        "torch/utils/data",
    ],
}


def get_source_signature(module: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Get signature from reference source, using SOURCE_LOCATIONS mapping.

    Falls back to installed PyTorch stub files for C++ builtins.

    Args:
        module: Module name
        name: API name

    Returns:
        Signature dictionary or None
    """
    full_name = f"{module}.{name}"

    # Check explicit mapping first
    if full_name in SOURCE_LOCATIONS:
        source_path = PYTORCH_REFERENCE_ROOT / SOURCE_LOCATIONS[full_name]
        if source_path.exists():
            sigs = extract_signatures_from_file(source_path)
            if name in sigs:
                return sigs[name]

    # Try module extraction from reference source
    ref_sig = get_reference_signature(module, name)
    if ref_sig is not None:
        return ref_sig

    # Fall back to stub files for C++ builtins
    try:
        from .stub_parser import get_stub_signature_for_module
        stub_sig = get_stub_signature_for_module(module, name)
        if stub_sig is not None:
            return stub_sig
    except ImportError:
        pass

    return None
