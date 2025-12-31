"""
Exclusions system for API parity checking.

Provides utilities for loading and applying API exclusions - APIs that are
intentionally not implemented in mlx_compat (e.g., CUDA-specific APIs).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Signature exclusions - APIs where signature mismatch is acceptable
_SIGNATURE_EXCLUSIONS: Dict[str, Dict[str, str]] = {}


def load_exclusions(extra_file: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Load all exclusion files and merge them.

    Args:
        extra_file: Optional path to an additional exclusion YAML file

    Returns:
        Dictionary mapping module names to API exclusion dicts.
        Format: {module_name: {api_name: reason}}
    """
    exclusions: Dict[str, Dict[str, str]] = {}
    exclusion_dir = Path(__file__).parent

    # Load all YAML files in the exclusions directory (except signature exclusions)
    for yaml_file in exclusion_dir.glob("*.yaml"):
        # Skip signature exclusions - those are handled separately
        if yaml_file.name == "signature_exclusions.yaml":
            continue
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}
            for module, apis in data.items():
                if module not in exclusions:
                    exclusions[module] = {}
                if isinstance(apis, dict):
                    exclusions[module].update(apis)

    # Load extra file if provided
    if extra_file:
        extra_path = Path(extra_file)
        if extra_path.exists():
            with open(extra_path) as f:
                data = yaml.safe_load(f) or {}
                for module, apis in data.items():
                    if module not in exclusions:
                        exclusions[module] = {}
                    if isinstance(apis, dict):
                        exclusions[module].update(apis)

    return exclusions


def is_excluded(
    module_name: str,
    api_name: str,
    exclusions: Dict[str, Dict[str, str]],
) -> Tuple[bool, str]:
    """
    Check if an API is excluded from parity checking.

    Args:
        module_name: The module name (e.g., "torch.nn")
        api_name: The API name (e.g., "Linear")
        exclusions: The exclusions dictionary

    Returns:
        Tuple of (is_excluded, reason)
    """
    module_exclusions = exclusions.get(module_name, {})
    if api_name in module_exclusions:
        return True, module_exclusions[api_name]
    return False, ""


def get_exclusion_summary(exclusions: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    """
    Get a summary of exclusions by module.

    Args:
        exclusions: The exclusions dictionary

    Returns:
        Dictionary mapping module names to exclusion counts
    """
    return {module: len(apis) for module, apis in exclusions.items()}


def load_signature_exclusions() -> Dict[str, Dict[str, str]]:
    """
    Load signature-specific exclusions.

    These are APIs where signature mismatches are acceptable (e.g., when
    our explicit signature is better than PyTorch's *args/**kwargs wrapper).

    Returns:
        Dictionary mapping module names to API exclusion dicts.
        Format: {module_name: {api_name: reason}}
    """
    global _SIGNATURE_EXCLUSIONS

    if _SIGNATURE_EXCLUSIONS:
        return _SIGNATURE_EXCLUSIONS

    exclusion_dir = Path(__file__).parent
    sig_file = exclusion_dir / "signature_exclusions.yaml"

    if sig_file.exists():
        with open(sig_file) as f:
            _SIGNATURE_EXCLUSIONS = yaml.safe_load(f) or {}

    return _SIGNATURE_EXCLUSIONS


def is_signature_excluded(
    module_name: str,
    api_name: str,
) -> Tuple[bool, str]:
    """
    Check if an API's signature mismatch should be excluded.

    Args:
        module_name: The module name (e.g., "torch.nn.functional")
        api_name: The API name (e.g., "max_pool2d")

    Returns:
        Tuple of (is_excluded, reason)
    """
    exclusions = load_signature_exclusions()
    module_exclusions = exclusions.get(module_name, {})
    if api_name in module_exclusions:
        return True, module_exclusions[api_name]
    return False, ""


# Numerical exclusions - APIs that cannot be numerically compared
_NUMERICAL_EXCLUSIONS: Dict[str, Dict[str, str]] = {}


def load_numerical_exclusions() -> Dict[str, Dict[str, str]]:
    """
    Load numerical parity exclusions.

    These are APIs that cannot be meaningfully compared for numerical
    equivalence (e.g., random generators, context managers, etc.).

    Returns:
        Dictionary mapping module names to API exclusion dicts.
        Format: {module_name: {api_name: reason}}
    """
    global _NUMERICAL_EXCLUSIONS

    if _NUMERICAL_EXCLUSIONS:
        return _NUMERICAL_EXCLUSIONS

    exclusion_dir = Path(__file__).parent
    num_file = exclusion_dir / "numerical_exclusions.yaml"

    if num_file.exists():
        with open(num_file) as f:
            _NUMERICAL_EXCLUSIONS = yaml.safe_load(f) or {}

    return _NUMERICAL_EXCLUSIONS


def is_numerical_excluded(
    module_name: str,
    api_name: str,
) -> Tuple[bool, str]:
    """
    Check if an API should be excluded from numerical parity testing.

    Args:
        module_name: The module name (e.g., "torch.nn")
        api_name: The API name (e.g., "Dropout")

    Returns:
        Tuple of (is_excluded, reason)
    """
    exclusions = load_numerical_exclusions()
    module_exclusions = exclusions.get(module_name, {})

    # Handle case where module_exclusions is None (empty YAML value)
    if module_exclusions is None:
        return False, ""

    # Check for wildcard exclusion (entire module)
    if "*" in module_exclusions:
        return True, module_exclusions["*"]

    # Check for specific API exclusion
    if api_name in module_exclusions:
        return True, module_exclusions[api_name]

    return False, ""
