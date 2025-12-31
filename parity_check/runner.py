"""
Main runner for parity validation.

Orchestrates the introspection, validation, and reporting pipeline.
"""

from typing import Dict, List, Optional

from .config import (
    MODULE_MAPPING,
    PYTORCH_MODULES,
    get_mlx_compat_version,
    get_pytorch_version,
)
from .exclusions import load_exclusions
from .introspection import enumerate_mlx_api, enumerate_pytorch_api
from .reports import ParityReport
from .validators import (
    APIPresenceValidator,
    BehavioralParityValidator,
    NumericalParityValidator,
    SignatureValidator,
)


class ParityRunner:
    """
    Main runner for API parity validation.

    Args:
        modules: List of PyTorch modules to check. If None, uses default list.
        extra_exclusions_file: Path to additional exclusions YAML file.
        strict_defaults: Whether to strictly check default values.
        strict_annotations: Whether to strictly check type annotations.
        ignore_out_param: Ignore missing 'out=' parameter (MLX limitation). Default True.
        ignore_layout_params: Ignore 'layout', 'pin_memory' params (MLX limitation). Default True.
        normalize_param_names: Treat 'input'/'tensor' as equivalent. Default True.
    """

    def __init__(
        self,
        modules: Optional[List[str]] = None,
        extra_exclusions_file: Optional[str] = None,
        strict_defaults: bool = True,
        strict_annotations: bool = False,
        ignore_out_param: bool = True,
        ignore_layout_params: bool = True,
        normalize_param_names: bool = True,
        # Numerical parity options
        run_numerical: bool = False,
        numerical_rtol: float = 1e-5,
        numerical_atol: float = 1e-6,
        # Behavioral parity options
        run_behavioral: bool = False,
        behavioral_categories: Optional[List[str]] = None,
    ):
        self.modules = modules or PYTORCH_MODULES
        self.extra_exclusions_file = extra_exclusions_file
        self.strict_defaults = strict_defaults
        self.strict_annotations = strict_annotations
        self.ignore_out_param = ignore_out_param
        self.ignore_layout_params = ignore_layout_params
        self.normalize_param_names = normalize_param_names

        # Numerical parity settings
        self.run_numerical = run_numerical
        self.numerical_rtol = numerical_rtol
        self.numerical_atol = numerical_atol

        # Behavioral parity settings
        self.run_behavioral = run_behavioral
        self.behavioral_categories = behavioral_categories

        # These will be populated during run()
        self.pytorch_apis: Dict = {}
        self.mlx_apis: Dict = {}
        self.exclusions: Dict = {}

    def run(self) -> ParityReport:
        """
        Run the full parity validation pipeline.

        Returns:
            ParityReport with all validation results
        """
        # Load exclusions
        self.exclusions = load_exclusions(self.extra_exclusions_file)

        # Enumerate APIs
        self.pytorch_apis = enumerate_pytorch_api(self.modules)
        self.mlx_apis = enumerate_mlx_api(self.modules)

        # Run presence validation
        presence_validator = APIPresenceValidator(
            self.pytorch_apis,
            self.mlx_apis,
            self.exclusions,
        )
        presence_result = presence_validator.validate()

        # Run signature validation
        signature_validator = SignatureValidator(
            self.pytorch_apis,
            self.mlx_apis,
            strict_defaults=self.strict_defaults,
            strict_annotations=self.strict_annotations,
            ignore_out_param=self.ignore_out_param,
            ignore_layout_params=self.ignore_layout_params,
            normalize_param_names=self.normalize_param_names,
        )
        signature_result = signature_validator.validate()

        # Calculate module stats
        module_stats = self._calculate_module_stats(presence_result)

        # Run numerical validation if requested
        numerical_result = None
        if self.run_numerical:
            numerical_validator = NumericalParityValidator(
                self.pytorch_apis,
                self.mlx_apis,
                rtol=self.numerical_rtol,
                atol=self.numerical_atol,
            )
            numerical_result = numerical_validator.validate()

        # Run behavioral validation if requested
        behavioral_result = None
        if self.run_behavioral:
            behavioral_validator = BehavioralParityValidator(
                categories=self.behavioral_categories,
                seed=42,
            )
            behavioral_result = behavioral_validator.validate()

        # Build report
        report = ParityReport(
            pytorch_version=get_pytorch_version(),
            mlx_compat_version=get_mlx_compat_version(),
            # Presence stats
            total_pytorch_apis=sum(len(apis) for apis in self.pytorch_apis.values()),
            implemented_apis=len(presence_result.present),
            missing_apis=len(presence_result.missing),
            excluded_apis=len(presence_result.excluded),
            # Signature stats
            signature_matches=len(signature_result.matches),
            signature_mismatches=len(signature_result.mismatches),
            signature_skipped=len(signature_result.skipped),
            # Numerical stats
            numerical_matches=len(numerical_result.matches) if numerical_result else 0,
            numerical_mismatches=len(numerical_result.mismatches) if numerical_result else 0,
            numerical_skipped=len(numerical_result.skipped) if numerical_result else 0,
            numerical_errors=len(numerical_result.errors) if numerical_result else 0,
            # Behavioral stats
            behavioral_passed=len(behavioral_result.passed) if behavioral_result else 0,
            behavioral_failed=len(behavioral_result.failed) if behavioral_result else 0,
            behavioral_skipped=len(behavioral_result.skipped) if behavioral_result else 0,
            behavioral_errors=len(behavioral_result.errors) if behavioral_result else 0,
            # Details
            missing_api_list=presence_result.missing,
            excluded_api_list=presence_result.excluded,
            signature_mismatch_list=signature_result.mismatches,
            numerical_mismatch_list=numerical_result.mismatches if numerical_result else [],
            numerical_error_list=numerical_result.errors if numerical_result else [],
            behavioral_failed_list=[
                {
                    "category": r.category,
                    "test": r.test_name,
                    "error": r.error,
                    "details": r.details,
                }
                for r in behavioral_result.failed
            ] if behavioral_result else [],
            behavioral_by_category=behavioral_result.by_category() if behavioral_result else {},
            # Module breakdown
            module_stats=module_stats,
        )

        return report

    def _calculate_module_stats(self, presence_result) -> Dict[str, Dict[str, int]]:
        """Calculate per-module statistics."""
        stats = {}

        for module in self.modules:
            total = len(self.pytorch_apis.get(module, {}))
            implemented = sum(
                1 for api in presence_result.present if api["module"] == module
            )
            missing = sum(
                1 for api in presence_result.missing if api["module"] == module
            )
            excluded = sum(
                1 for api in presence_result.excluded if api["module"] == module
            )

            stats[module] = {
                "total": total,
                "implemented": implemented,
                "missing": missing,
                "excluded": excluded,
            }

        return stats

    def get_missing_by_type(self) -> Dict[str, List[str]]:
        """
        Get missing APIs grouped by type.

        Returns:
            Dictionary mapping type names to lists of missing API names
        """
        presence_validator = APIPresenceValidator(
            self.pytorch_apis,
            self.mlx_apis,
            self.exclusions,
        )
        result = presence_validator.validate()

        by_type: Dict[str, List[str]] = {}
        for api in result.missing:
            api_type = api.get("type", "unknown")
            if api_type not in by_type:
                by_type[api_type] = []
            by_type[api_type].append(f"{api['module']}.{api['api']}")

        return by_type

    def get_missing_by_module(self) -> Dict[str, List[str]]:
        """
        Get missing APIs grouped by module.

        Returns:
            Dictionary mapping module names to lists of missing API names
        """
        presence_validator = APIPresenceValidator(
            self.pytorch_apis,
            self.mlx_apis,
            self.exclusions,
        )
        result = presence_validator.validate()

        by_module: Dict[str, List[str]] = {}
        for api in result.missing:
            module = api["module"]
            if module not in by_module:
                by_module[module] = []
            by_module[module].append(api["api"])

        return by_module
