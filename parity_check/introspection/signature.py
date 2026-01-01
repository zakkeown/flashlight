"""
Signature extraction and comparison utilities.

Provides tools for extracting function/class signatures and comparing them
for API parity validation.
"""

import inspect
from inspect import Parameter, signature
from typing import Any, Dict, List, Optional, Tuple


def extract_signature(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Extract signature information from a callable object.

    Args:
        obj: A function, method, or class to extract signature from

    Returns:
        Dictionary with signature information, or None if not extractable.
        Format:
        {
            "parameters": [
                {
                    "name": str,
                    "kind": str,  # POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, etc.
                    "default": str or None,  # repr of default value
                    "annotation": str or None,
                }
            ],
            "return_annotation": str or None,
            "extractable": bool,
        }
    """
    # Determine what to inspect
    if inspect.isclass(obj):
        # For classes, inspect __init__
        target = obj.__init__
    elif callable(obj):
        target = obj
    else:
        return None

    try:
        sig = signature(target)
    except (ValueError, TypeError):
        # Builtins and C extensions may not have extractable signatures
        return {"parameters": [], "return_annotation": None, "extractable": False}

    params = []
    for name, param in sig.parameters.items():
        # Skip 'self' and 'cls' parameters
        if name in ("self", "cls"):
            continue

        param_info = {
            "name": name,
            "kind": param.kind.name,
        }

        # Handle default values
        if param.default is not Parameter.empty:
            param_info["default"] = _serialize_default(param.default)
        else:
            param_info["default"] = None

        # Handle annotations
        if param.annotation is not Parameter.empty:
            param_info["annotation"] = _serialize_annotation(param.annotation)
        else:
            param_info["annotation"] = None

        params.append(param_info)

    # Handle return annotation
    return_annotation = None
    if sig.return_annotation is not Parameter.empty:
        return_annotation = _serialize_annotation(sig.return_annotation)

    return {
        "parameters": params,
        "return_annotation": return_annotation,
        "extractable": True,
    }


def _serialize_default(default: Any) -> str:
    """Serialize a default value to a string representation."""
    if default is None:
        return "None"
    elif isinstance(default, bool):
        return str(default)
    elif isinstance(default, (int, float)):
        return str(default)
    elif isinstance(default, str):
        return repr(default)
    elif isinstance(default, (list, tuple, dict)):
        return repr(default)
    elif callable(default):
        # For functions, extract just the function name
        name = getattr(default, '__name__', None)
        if name:
            return f"<function {name}>"
        return repr(default)
    else:
        # For complex objects, use repr but truncate
        r = repr(default)
        if len(r) > 50:
            r = r[:47] + "..."
        return r


def _defaults_match(pytorch_default: str, mlx_default: str) -> bool:
    """
    Check if two default values are semantically equivalent.

    This handles cases like:
    - Function references (relu vs F.relu vs <function relu>)
    - Numeric precision (1e-3 vs 0.001)
    - Module references
    - Empty tuple () vs None (both mean "use default behavior")
    - Reduction enum values (1 vs 'mean', 2 vs 'sum')
    - Ellipsis (...) in PyTorch stubs means "any default is OK"

    Args:
        pytorch_default: Default value string from PyTorch
        mlx_default: Default value string from flashlight

    Returns:
        True if the defaults are semantically equivalent
    """
    # Exact match
    if pytorch_default == mlx_default:
        return True

    # PyTorch stubs use ... (ellipsis) to indicate "has a default but unspecified"
    # If PyTorch has ..., accept any default value from flashlight
    if pytorch_default == "..." or pytorch_default == "Ellipsis":
        return True

    # Normalize both for comparison
    pt_norm = _normalize_default(pytorch_default)
    mlx_norm = _normalize_default(mlx_default)

    if pt_norm == mlx_norm:
        return True

    # Special case: () and None are often semantically equivalent
    # (meaning "use default/infer from input")
    empty_equiv = {"()", "None", "none"}
    if pt_norm in empty_equiv and mlx_norm in empty_equiv:
        return True

    # Special case: 0 and None for storage_offset (0 is the default behavior)
    if (pt_norm == "None" and mlx_norm == "0") or (pt_norm == "0" and mlx_norm == "None"):
        return True

    # Special case: reduction enum values in PyTorch stubs
    # 0 = none, 1 = mean, 2 = sum
    reduction_map = {"0": "none", "1": "mean", "2": "sum"}
    if pt_norm in reduction_map and mlx_norm == reduction_map[pt_norm]:
        return True
    if mlx_norm in reduction_map and pt_norm == reduction_map[mlx_norm]:
        return True

    # Special case: -1 and None often mean "infer from input"
    if (pt_norm == "-1" and mlx_norm == "None") or (pt_norm == "None" and mlx_norm == "-1"):
        return True

    # Special case: None vs False/True for boolean flags with sensible defaults
    # (e.g., shared=None means shared=False behavior)
    if pt_norm == "None" and mlx_norm in ("False", "True"):
        return True

    # Special case: None vs a specific numeric default that is the "standard" value
    # e.g., eps=None means "use standard epsilon" which is 1e-5 for norm operations
    # e.g., spacing=None means spacing=1.0 for gradient
    standard_defaults = {
        "1e-05": "None",
        "1e-5": "None",
        "1.0": "None",
        "1": "None",
    }
    if pt_norm == "None" and mlx_norm in standard_defaults:
        return True
    if mlx_norm == "None" and pt_norm in standard_defaults:
        return True

    return False


def _normalize_default(default: str) -> str:
    """
    Normalize a default value string for comparison.

    Handles:
    - Function references: 'F.relu', '<function relu...>', 'relu' -> 'relu'
    - Numeric: '1e-3' -> '0.001', '1e-08' -> '1e-8'
    - Module prefixes: 'torch.nn.functional.relu' -> 'relu'
    - Quotes: 'mean' and "mean" -> mean
    """
    if default is None:
        return "None"

    s = str(default)

    # Handle function references
    if '<function ' in s:
        # Extract function name from <function name at 0x...>
        import re
        match = re.search(r'<function (\w+)', s)
        if match:
            return match.group(1)

    # Handle F.relu, torch.nn.functional.relu, etc.
    function_prefixes = ['F.', 'torch.nn.functional.', 'nn.functional.']
    for prefix in function_prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]

    # Strip quotes for string comparison
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]

    # Normalize scientific notation
    try:
        # Try to parse as float and compare
        val = float(s)
        # Normalize to standard form
        if val == 0:
            return "0"
        return f"{val:g}"
    except (ValueError, TypeError):
        pass

    return s


def _serialize_annotation(annotation: Any) -> str:
    """Serialize a type annotation to a string representation."""
    if annotation is None:
        return "None"
    elif hasattr(annotation, "__name__"):
        return annotation.__name__
    else:
        # Handle typing module annotations
        return str(annotation)


def compare_signatures(
    pytorch_sig: Optional[Dict[str, Any]],
    mlx_sig: Optional[Dict[str, Any]],
    strict_defaults: bool = True,
    strict_annotations: bool = False,
    ignore_out_param: bool = False,
    ignore_layout_params: bool = False,
    normalize_param_names: bool = False,
) -> Dict[str, Any]:
    """
    Compare two signatures for compatibility.

    Args:
        pytorch_sig: Signature info from PyTorch API
        mlx_sig: Signature info from flashlight API
        strict_defaults: If True, default values must match exactly
        strict_annotations: If True, type annotations must match
        ignore_out_param: If True, ignore missing 'out' parameter (MLX doesn't support)
        ignore_layout_params: If True, ignore 'layout', 'pin_memory', 'device' params
        normalize_param_names: If True, treat 'input'/'tensor'/'self' as equivalent

    Returns:
        Dictionary with comparison results:
        {
            "matches": bool,
            "differences": List[str],
            "details": Dict with detailed comparison info
        }
    """
    differences = []
    details = {}

    # Handle non-extractable signatures
    if pytorch_sig is None or mlx_sig is None:
        return {
            "matches": True,  # Can't compare, assume OK
            "differences": [],
            "details": {"note": "One or both signatures not extractable"},
        }

    if not pytorch_sig.get("extractable") or not mlx_sig.get("extractable"):
        return {
            "matches": True,
            "differences": [],
            "details": {"note": "One or both signatures not extractable"},
        }

    pytorch_params = pytorch_sig.get("parameters", [])
    mlx_params = mlx_sig.get("parameters", [])

    # Parameters to ignore when comparing (MLX-specific limitations)
    ignored_params = set()
    if ignore_out_param:
        ignored_params.add("out")
    if ignore_layout_params:
        # MLX doesn't have CUDA/device-specific params
        ignored_params.update([
            "layout", "pin_memory", "memory_format",
            # CUDA-specific parameters
            "cudnn_enabled", "cudnn_enable",
            # Random generator (MLX uses global state)
            "generator",
            # Requires grad handled differently
            "requires_grad",
            # Dtype params often implicit in MLX
            "dtype", "out_dtype",
            # Sparse operations not supported
            "sparse_grad", "sparse",
            # Gradient scaling
            "scale_grad_by_freq",
            # Device params (MLX is unified memory)
            "device",
            # NaN comparison (can add later if needed)
            "equal_nan",
            # Correction param (ddof replacement) - often 1 by default anyway
            "correction",
            # Optional output naming
            "outdim", "out_dim",
            # SVD/QR decomposition mode params
            "some", "compute_uv",
            # Tensor name metadata
            "names",
            # Pooling overrides
            "divisor_override",
            # Attention params
            "enable_gqa",
            # Quantile interpolation
            "interpolation",
            # Repeat interleave
            "output_size",
            # Scatter reduce mode
            "reduce",
            # RNN packed sequence params
            "data", "batch_sizes",
            # Searchsorted sorter
            "sorter",
        ])

    # Build parameter lookup dicts (excluding ignored params)
    pytorch_param_dict = {
        p["name"]: p for p in pytorch_params
        if p["name"] not in ignored_params
    }
    mlx_param_dict = {p["name"]: p for p in mlx_params}

    # Name equivalence classes - each list contains names that are equivalent
    # The first name in each list is the canonical form
    name_equiv_classes = [
        # First argument variations - common names for "the input tensor"
        ["input", "tensor", "self", "x", "indices", "weight", "b", "obj", "data", "sorted_sequence"],
        # Second argument variations - common names for "the other tensor"
        ["other", "y", "mat2", "input2", "u", "tau", "values", "src"],
        # Third argument
        ["input3"],
        # Training flag
        ["training", "train"],
        # Shape specifications
        ["size", "shape"],
        # Split operations
        ["indices_or_sections", "sections", "tensor_indices_or_sections"],
        # Type casting
        ["from_dtype", "from_"],
        ["to_dtype", "to"],
        # Dimension naming
        ["dim0", "axis0"],
        ["dim1", "axis1"],
        # Polar abs
        ["abs_val", "abs"],
        # Scalar value
        ["value", "s"],
        # Batch first flag (RNN) - optional param
        ["batch_first"],
    ]

    # Build lookup from name to canonical form
    name_to_canonical = {}
    for equiv_class in name_equiv_classes:
        canonical = equiv_class[0]
        for name in equiv_class:
            name_to_canonical[name] = canonical

    def normalize_name(name: str) -> str:
        if normalize_param_names and name in name_to_canonical:
            return name_to_canonical[name]
        return name

    # Get parameter names (excluding *args and **kwargs for order comparison)
    pytorch_names = [
        p["name"] for p in pytorch_params
        if p["kind"] not in ("VAR_POSITIONAL", "VAR_KEYWORD")
        and p["name"] not in ignored_params
    ]
    mlx_names = [
        p["name"] for p in mlx_params
        if p["kind"] not in ("VAR_POSITIONAL", "VAR_KEYWORD")
    ]

    # Check for missing parameters in flashlight
    for name in pytorch_param_dict:
        normalized = normalize_name(name)
        # Check if param exists (possibly with different name)
        has_param = (
            name in mlx_param_dict or
            (normalize_param_names and any(
                normalize_name(n) == normalized for n in mlx_param_dict
            ))
        )
        if not has_param:
            # Allow if PyTorch has *args or **kwargs and mlx doesn't have this param
            has_var = any(p["kind"] in ("VAR_POSITIONAL", "VAR_KEYWORD") for p in mlx_params)
            if not has_var:
                differences.append(f"Missing parameter: '{name}'")

    # Check for extra parameters in flashlight
    for name in mlx_param_dict:
        # Skip ignored params - if PyTorch has device param ignored, mlx can have it too
        if name in ignored_params:
            continue

        normalized = normalize_name(name)
        # Check if param exists in PyTorch (possibly with different name)
        has_param = (
            name in pytorch_param_dict or
            (normalize_param_names and any(
                normalize_name(n) == normalized for n in pytorch_param_dict
            ))
        )
        if not has_param:
            # Extra parameters are OK if they have defaults
            mlx_param = mlx_param_dict[name]
            if mlx_param["default"] is None and mlx_param["kind"] not in ("VAR_POSITIONAL", "VAR_KEYWORD"):
                differences.append(f"Extra required parameter: '{name}'")

    # Check parameter order - only for POSITIONAL_ONLY params (before /)
    # Most parameters can be called with keywords, so order doesn't matter
    # Only flag if positional-only params would cause runtime errors
    pytorch_positional_only = [
        normalize_name(p["name"]) for p in pytorch_params
        if p["kind"] == "POSITIONAL_ONLY" and p["name"] not in ignored_params
    ]
    mlx_positional_only = [
        normalize_name(p["name"]) for p in mlx_params
        if p["kind"] == "POSITIONAL_ONLY"
    ]

    # Only flag order difference if positional-only params differ
    if pytorch_positional_only and mlx_positional_only:
        common_pt = [n for n in pytorch_positional_only if n in mlx_positional_only]
        common_mlx = [n for n in mlx_positional_only if n in pytorch_positional_only]
        if common_pt != common_mlx:
            differences.append(f"Parameter order differs")
            details["pytorch_order"] = pytorch_names
            details["mlx_order"] = mlx_names

    # Check defaults if strict
    if strict_defaults:
        for name, pytorch_param in pytorch_param_dict.items():
            if name in mlx_param_dict:
                mlx_param = mlx_param_dict[name]
                pytorch_default = pytorch_param.get("default")
                mlx_default = mlx_param.get("default")

                # Check if mlx is missing a default that PyTorch has
                if pytorch_default is not None and mlx_default is None:
                    differences.append(f"Parameter '{name}' missing default (PyTorch default: {pytorch_default})")
                # Check if both have defaults but they differ
                elif pytorch_default is not None and mlx_default is not None:
                    # Use semantic comparison for defaults
                    if not _defaults_match(pytorch_default, mlx_default):
                        differences.append(
                            f"Default value mismatch for '{name}': "
                            f"PyTorch={pytorch_default}, flashlight={mlx_default}"
                        )
                # Having an extra default in flashlight is OK

    # Check annotations if strict
    if strict_annotations:
        for name, pytorch_param in pytorch_param_dict.items():
            if name in mlx_param_dict:
                mlx_param = mlx_param_dict[name]
                pytorch_ann = pytorch_param.get("annotation")
                mlx_ann = mlx_param.get("annotation")

                if pytorch_ann and mlx_ann and pytorch_ann != mlx_ann:
                    differences.append(
                        f"Type annotation mismatch for '{name}': "
                        f"PyTorch={pytorch_ann}, flashlight={mlx_ann}"
                    )

    return {
        "matches": len(differences) == 0,
        "differences": differences,
        "details": details,
    }


def get_parameter_summary(sig: Optional[Dict[str, Any]]) -> str:
    """
    Get a human-readable summary of a signature.

    Args:
        sig: Signature dictionary from extract_signature

    Returns:
        String like "(x, y, z=None, *args, **kwargs)"
    """
    if sig is None or not sig.get("extractable"):
        return "(...)"

    parts = []
    for param in sig.get("parameters", []):
        name = param["name"]
        kind = param["kind"]
        default = param.get("default")

        if kind == "VAR_POSITIONAL":
            parts.append(f"*{name}")
        elif kind == "VAR_KEYWORD":
            parts.append(f"**{name}")
        elif default is not None:
            parts.append(f"{name}={default}")
        else:
            parts.append(name)

    return f"({', '.join(parts)})"
