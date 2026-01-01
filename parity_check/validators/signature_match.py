"""
Signature matching validator.

Validates that flashlight API signatures match PyTorch signatures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..exclusions import is_signature_excluded
from ..introspection.signature import compare_signatures, get_parameter_summary
from ..introspection.source_parser import get_source_signature, PYTORCH_REFERENCE_ROOT


@dataclass
class SignatureValidationResult:
    """Result of signature validation."""

    matches: List[Dict[str, Any]] = field(default_factory=list)
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_compared(self) -> int:
        """Total number of signatures compared."""
        return len(self.matches) + len(self.mismatches)

    @property
    def match_percentage(self) -> float:
        """Percentage of signatures that match."""
        if self.total_compared == 0:
            return 100.0
        return (len(self.matches) / self.total_compared) * 100

    @property
    def is_complete(self) -> bool:
        """Returns True if all signatures match."""
        return len(self.mismatches) == 0


class SignatureValidator:
    """
    Validates that flashlight API signatures match PyTorch.

    Args:
        pytorch_apis: Dictionary of PyTorch APIs by module
        mlx_apis: Dictionary of flashlight APIs by module
        strict_defaults: If True, default values must match
        strict_annotations: If True, type annotations must match
        use_source_fallback: If True, try AST-based source extraction when
                            inspect.signature() fails (requires reference source)
        ignore_out_param: If True, ignore missing 'out' parameter (MLX limitation)
        ignore_layout_params: If True, ignore 'layout', 'pin_memory' params (MLX limitation)
        normalize_param_names: If True, treat 'input'/'tensor' as equivalent
    """

    def __init__(
        self,
        pytorch_apis: Dict[str, Dict[str, Any]],
        mlx_apis: Dict[str, Dict[str, Any]],
        strict_defaults: bool = True,
        strict_annotations: bool = False,
        use_source_fallback: bool = True,
        ignore_out_param: bool = False,
        ignore_layout_params: bool = False,
        normalize_param_names: bool = False,
    ):
        self.pytorch_apis = pytorch_apis
        self.mlx_apis = mlx_apis
        self.strict_defaults = strict_defaults
        self.strict_annotations = strict_annotations
        self.use_source_fallback = use_source_fallback
        self.ignore_out_param = ignore_out_param
        self.ignore_layout_params = ignore_layout_params
        self.normalize_param_names = normalize_param_names
        self._source_available = PYTORCH_REFERENCE_ROOT.exists()

    def _get_source_signature(self, module: str, api_name: str) -> Optional[Dict[str, Any]]:
        """
        Try to get signature from reference PyTorch source.

        Args:
            module: Module name (e.g., 'torch.optim')
            api_name: API name (e.g., 'Adam')

        Returns:
            Signature dictionary or None if not found
        """
        if not self.use_source_fallback or not self._source_available:
            return None

        try:
            return get_source_signature(module, api_name)
        except Exception:
            return None

    def validate(self) -> SignatureValidationResult:
        """
        Validate signatures for all common APIs.

        Returns:
            SignatureValidationResult with matches, mismatches, and skipped APIs
        """
        matches = []
        mismatches = []
        skipped = []

        for module, apis in self.pytorch_apis.items():
            mlx_module_apis = self.mlx_apis.get(module, {})

            for api_name, pytorch_info in apis.items():
                # Skip if not present in flashlight
                if api_name not in mlx_module_apis:
                    continue

                mlx_info = mlx_module_apis[api_name]

                pytorch_sig = pytorch_info.get("signature")
                mlx_sig = mlx_info.get("signature")

                # Try source-based fallback if PyTorch signature not extractable
                pytorch_not_extractable = (
                    pytorch_sig is None or not pytorch_sig.get("extractable")
                )
                if pytorch_not_extractable:
                    source_sig = self._get_source_signature(module, api_name)
                    if source_sig:
                        pytorch_sig = source_sig

                # Skip if either signature is still not extractable
                if pytorch_sig is None or mlx_sig is None:
                    skipped.append({
                        "module": module,
                        "api": api_name,
                        "reason": "Signature not extractable",
                    })
                    continue

                if not pytorch_sig.get("extractable") or not mlx_sig.get("extractable"):
                    skipped.append({
                        "module": module,
                        "api": api_name,
                        "reason": "Signature not extractable (builtin)",
                    })
                    continue

                # Check if signature mismatch is excluded
                sig_excluded, exclusion_reason = is_signature_excluded(module, api_name)

                # Compare signatures
                comparison = compare_signatures(
                    pytorch_sig,
                    mlx_sig,
                    strict_defaults=self.strict_defaults,
                    strict_annotations=self.strict_annotations,
                    ignore_out_param=self.ignore_out_param,
                    ignore_layout_params=self.ignore_layout_params,
                    normalize_param_names=self.normalize_param_names,
                )

                # Determine source type for reporting
                source_type = pytorch_sig.get("source", "inspect")

                if comparison["matches"]:
                    matches.append({
                        "module": module,
                        "api": api_name,
                        "source": source_type,
                    })
                elif sig_excluded:
                    # Treat excluded signature mismatches as matches
                    matches.append({
                        "module": module,
                        "api": api_name,
                        "note": f"Signature exclusion: {exclusion_reason}",
                        "source": source_type,
                    })
                else:
                    mismatches.append({
                        "module": module,
                        "api": api_name,
                        "differences": comparison["differences"],
                        "pytorch_signature": get_parameter_summary(pytorch_sig),
                        "mlx_signature": get_parameter_summary(mlx_sig),
                        "details": comparison.get("details", {}),
                        "source": source_type,
                    })

        return SignatureValidationResult(
            matches=matches,
            mismatches=mismatches,
            skipped=skipped,
        )

    def validate_api(self, module: str, api_name: str) -> Dict[str, Any]:
        """
        Validate signature for a single API.

        Args:
            module: Module name
            api_name: API name

        Returns:
            Dictionary with validation results
        """
        pytorch_info = self.pytorch_apis.get(module, {}).get(api_name)
        mlx_info = self.mlx_apis.get(module, {}).get(api_name)

        if pytorch_info is None:
            return {"error": f"API {module}.{api_name} not found in PyTorch"}
        if mlx_info is None:
            return {"error": f"API {module}.{api_name} not found in flashlight"}

        pytorch_sig = pytorch_info.get("signature")
        mlx_sig = mlx_info.get("signature")

        comparison = compare_signatures(
            pytorch_sig,
            mlx_sig,
            strict_defaults=self.strict_defaults,
            strict_annotations=self.strict_annotations,
        )

        return {
            "module": module,
            "api": api_name,
            "matches": comparison["matches"],
            "differences": comparison["differences"],
            "pytorch_signature": get_parameter_summary(pytorch_sig),
            "mlx_signature": get_parameter_summary(mlx_sig),
        }
