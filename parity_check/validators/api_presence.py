"""
API presence validator.

Validates that expected PyTorch APIs exist in mlx_compat.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..exclusions import is_excluded


@dataclass
class PresenceValidationResult:
    """Result of API presence validation."""

    present: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[Dict[str, Any]] = field(default_factory=list)
    excluded: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_checked(self) -> int:
        """Total number of APIs checked (excluding excluded ones)."""
        return len(self.present) + len(self.missing)

    @property
    def coverage_percentage(self) -> float:
        """Percentage of APIs that are implemented."""
        if self.total_checked == 0:
            return 100.0
        return (len(self.present) / self.total_checked) * 100

    @property
    def is_complete(self) -> bool:
        """Returns True if all non-excluded APIs are implemented."""
        return len(self.missing) == 0


class APIPresenceValidator:
    """
    Validates that expected PyTorch APIs exist in mlx_compat.

    Args:
        pytorch_apis: Dictionary of PyTorch APIs by module
        mlx_apis: Dictionary of mlx_compat APIs by module
        exclusions: Dictionary of excluded APIs by module
    """

    def __init__(
        self,
        pytorch_apis: Dict[str, Dict[str, Any]],
        mlx_apis: Dict[str, Dict[str, Any]],
        exclusions: Dict[str, Dict[str, str]],
    ):
        self.pytorch_apis = pytorch_apis
        self.mlx_apis = mlx_apis
        self.exclusions = exclusions

    def validate(self) -> PresenceValidationResult:
        """
        Validate API presence.

        Returns:
            PresenceValidationResult with present, missing, and excluded APIs
        """
        present = []
        missing = []
        excluded = []

        for module, apis in self.pytorch_apis.items():
            mlx_module_apis = self.mlx_apis.get(module, {})

            for api_name, api_info in apis.items():
                # Check if excluded
                is_excl, reason = is_excluded(module, api_name, self.exclusions)

                if is_excl:
                    excluded.append({
                        "module": module,
                        "api": api_name,
                        "reason": reason,
                        "type": api_info.get("type", "unknown"),
                    })
                elif api_name in mlx_module_apis:
                    present.append({
                        "module": module,
                        "api": api_name,
                        "type": api_info.get("type", "unknown"),
                    })
                else:
                    missing.append({
                        "module": module,
                        "api": api_name,
                        "type": api_info.get("type", "unknown"),
                        "pytorch_module": api_info.get("module"),
                    })

        return PresenceValidationResult(
            present=present,
            missing=missing,
            excluded=excluded,
        )

    def validate_module(self, module: str) -> PresenceValidationResult:
        """
        Validate API presence for a single module.

        Args:
            module: Module name to validate

        Returns:
            PresenceValidationResult for the specified module
        """
        pytorch_apis = {module: self.pytorch_apis.get(module, {})}
        mlx_apis = {module: self.mlx_apis.get(module, {})}

        validator = APIPresenceValidator(pytorch_apis, mlx_apis, self.exclusions)
        return validator.validate()
