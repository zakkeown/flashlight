"""
Report formatters for parity check results.

Provides console, JSON, and Markdown output formats.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class ParityReport:
    """Complete parity validation report."""

    pytorch_version: str
    mlx_compat_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # API presence stats
    total_pytorch_apis: int = 0
    implemented_apis: int = 0
    missing_apis: int = 0
    excluded_apis: int = 0

    # Signature validation stats
    signature_matches: int = 0
    signature_mismatches: int = 0
    signature_skipped: int = 0

    # Numerical parity stats
    numerical_matches: int = 0
    numerical_mismatches: int = 0
    numerical_skipped: int = 0
    numerical_errors: int = 0

    # Behavioral parity stats
    behavioral_passed: int = 0
    behavioral_failed: int = 0
    behavioral_skipped: int = 0
    behavioral_errors: int = 0

    # Detailed lists
    missing_api_list: List[Dict[str, Any]] = field(default_factory=list)
    excluded_api_list: List[Dict[str, Any]] = field(default_factory=list)
    signature_mismatch_list: List[Dict[str, Any]] = field(default_factory=list)
    numerical_mismatch_list: List[Dict[str, Any]] = field(default_factory=list)
    numerical_error_list: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_failed_list: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Per-module breakdown
    module_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def coverage_percentage(self) -> float:
        """Calculate API coverage percentage."""
        applicable = self.total_pytorch_apis - self.excluded_apis
        if applicable == 0:
            return 100.0
        return (self.implemented_apis / applicable) * 100

    @property
    def signature_match_percentage(self) -> float:
        """Calculate signature match percentage."""
        total = self.signature_matches + self.signature_mismatches
        if total == 0:
            return 100.0
        return (self.signature_matches / total) * 100

    @property
    def numerical_match_percentage(self) -> float:
        """Calculate numerical match percentage."""
        total = self.numerical_matches + self.numerical_mismatches
        if total == 0:
            return 100.0
        return (self.numerical_matches / total) * 100

    @property
    def numerical_tested(self) -> bool:
        """Check if numerical testing was performed."""
        return (
            self.numerical_matches > 0
            or self.numerical_mismatches > 0
            or self.numerical_errors > 0
        )

    @property
    def behavioral_tested(self) -> bool:
        """Check if behavioral testing was performed."""
        return (
            self.behavioral_passed > 0
            or self.behavioral_failed > 0
            or self.behavioral_errors > 0
        )

    @property
    def behavioral_pass_percentage(self) -> float:
        """Calculate behavioral test pass percentage."""
        total = self.behavioral_passed + self.behavioral_failed
        if total == 0:
            return 100.0
        return (self.behavioral_passed / total) * 100


class ReportFormatter:
    """Format parity reports for different outputs."""

    def to_console(self, report: ParityReport) -> str:
        """
        Format report for console output.

        Args:
            report: The parity report

        Returns:
            Human-readable string for terminal display
        """
        lines = [
            "",
            "=" * 70,
            "  MLX-COMPAT API PARITY REPORT",
            "=" * 70,
            "",
            f"  PyTorch Version:      {report.pytorch_version}",
            f"  mlx_compat Version:   {report.mlx_compat_version}",
            f"  Generated:            {report.timestamp}",
            "",
            "-" * 70,
            "  API COVERAGE",
            "-" * 70,
            "",
            f"  Total PyTorch APIs:     {report.total_pytorch_apis:>6}",
            f"  Implemented:            {report.implemented_apis:>6}",
            f"  Missing:                {report.missing_apis:>6}",
            f"  Excluded (intentional): {report.excluded_apis:>6}",
            "",
            f"  Coverage:               {report.coverage_percentage:>6.1f}%",
            "",
        ]

        # Per-module breakdown
        if report.module_stats:
            lines.extend([
                "-" * 70,
                "  COVERAGE BY MODULE",
                "-" * 70,
                "",
            ])
            for module, stats in sorted(report.module_stats.items()):
                total = stats.get("total", 0) - stats.get("excluded", 0)
                implemented = stats.get("implemented", 0)
                pct = (implemented / total * 100) if total > 0 else 100.0
                lines.append(f"  {module:30} {implemented:>4}/{total:<4} ({pct:>5.1f}%)")
            lines.append("")

        # Signature validation
        lines.extend([
            "-" * 70,
            "  SIGNATURE VALIDATION",
            "-" * 70,
            "",
            f"  Matching signatures:    {report.signature_matches:>6}",
            f"  Mismatched signatures:  {report.signature_mismatches:>6}",
            f"  Skipped (not extractable): {report.signature_skipped:>3}",
            "",
            f"  Match rate:             {report.signature_match_percentage:>6.1f}%",
            "",
        ])

        # Missing APIs (limited for console)
        if report.missing_api_list:
            lines.extend([
                "-" * 70,
                "  MISSING APIs (first 30)",
                "-" * 70,
                "",
            ])
            for api in report.missing_api_list[:30]:
                api_type = api.get("type", "")
                type_indicator = f"[{api_type}]" if api_type else ""
                lines.append(f"  - {api['module']}.{api['api']} {type_indicator}")

            if len(report.missing_api_list) > 30:
                lines.append(f"  ... and {len(report.missing_api_list) - 30} more")
            lines.append("")

        # Signature mismatches (limited for console)
        if report.signature_mismatch_list:
            lines.extend([
                "-" * 70,
                "  SIGNATURE MISMATCHES (first 15)",
                "-" * 70,
                "",
            ])
            for mismatch in report.signature_mismatch_list[:15]:
                lines.append(f"  {mismatch['module']}.{mismatch['api']}:")
                lines.append(f"    PyTorch: {mismatch.get('pytorch_signature', '?')}")
                lines.append(f"    MLX:     {mismatch.get('mlx_signature', '?')}")
                for diff in mismatch.get("differences", [])[:3]:
                    lines.append(f"      - {diff}")
                lines.append("")

            if len(report.signature_mismatch_list) > 15:
                lines.append(f"  ... and {len(report.signature_mismatch_list) - 15} more")
            lines.append("")

        # Numerical parity section (only if testing was performed)
        if report.numerical_tested:
            lines.extend([
                "-" * 70,
                "  NUMERICAL PARITY",
                "-" * 70,
                "",
                f"  Matching APIs:          {report.numerical_matches:>6}",
                f"  Mismatched APIs:        {report.numerical_mismatches:>6}",
                f"  Skipped (untestable):   {report.numerical_skipped:>6}",
                f"  Errors:                 {report.numerical_errors:>6}",
                "",
                f"  Match rate:             {report.numerical_match_percentage:>6.1f}%",
                "",
            ])

            # Numerical mismatches
            if report.numerical_mismatch_list:
                lines.extend([
                    "-" * 70,
                    "  NUMERICAL MISMATCHES (first 15)",
                    "-" * 70,
                    "",
                ])
                for mismatch in report.numerical_mismatch_list[:15]:
                    lines.append(f"  {mismatch.module}.{mismatch.api}:")
                    if mismatch.max_diff is not None:
                        lines.append(f"    Max diff: {mismatch.max_diff:.2e}")
                    if mismatch.pytorch_output_shape:
                        lines.append(f"    PT shape: {mismatch.pytorch_output_shape}")
                    if mismatch.mlx_output_shape:
                        lines.append(f"    MLX shape: {mismatch.mlx_output_shape}")
                    if mismatch.error:
                        lines.append(f"    Error: {mismatch.error}")
                    lines.append("")

                if len(report.numerical_mismatch_list) > 15:
                    lines.append(f"  ... and {len(report.numerical_mismatch_list) - 15} more")
                lines.append("")

            # Numerical errors
            if report.numerical_error_list:
                lines.extend([
                    "-" * 70,
                    f"  NUMERICAL ERRORS (all {len(report.numerical_error_list)})",
                    "-" * 70,
                    "",
                ])
                for error in report.numerical_error_list:
                    lines.append(f"  {error['module']}.{error['api']}:")
                    lines.append(f"    {error.get('error', 'Unknown error')}")
                    lines.append("")

        # Behavioral parity section (only if testing was performed)
        if report.behavioral_tested:
            lines.extend([
                "-" * 70,
                "  BEHAVIORAL PARITY",
                "-" * 70,
                "",
                f"  Passed tests:           {report.behavioral_passed:>6}",
                f"  Failed tests:           {report.behavioral_failed:>6}",
                f"  Skipped:                {report.behavioral_skipped:>6}",
                f"  Errors:                 {report.behavioral_errors:>6}",
                "",
                f"  Pass rate:              {report.behavioral_pass_percentage:>6.1f}%",
                "",
            ])

            # By-category breakdown
            if report.behavioral_by_category:
                lines.extend([
                    "-" * 70,
                    "  BEHAVIORAL BY CATEGORY",
                    "-" * 70,
                    "",
                ])
                for category, stats in sorted(report.behavioral_by_category.items()):
                    passed = stats.get("passed", 0)
                    failed = stats.get("failed", 0)
                    total = passed + failed
                    pct = (passed / total * 100) if total > 0 else 100.0
                    status = "PASS" if failed == 0 else "FAIL"
                    lines.append(f"  {category:25} {passed:>3}/{total:<3} ({pct:>5.1f}%) [{status}]")
                lines.append("")

            # Behavioral failures
            if report.behavioral_failed_list:
                lines.extend([
                    "-" * 70,
                    "  BEHAVIORAL FAILURES (first 15)",
                    "-" * 70,
                    "",
                ])
                for failure in report.behavioral_failed_list[:15]:
                    lines.append(f"  [{failure['category']}] {failure['test']}:")
                    if failure.get('error'):
                        lines.append(f"    Error: {failure['error']}")
                    lines.append("")

                if len(report.behavioral_failed_list) > 15:
                    lines.append(f"  ... and {len(report.behavioral_failed_list) - 15} more")
                lines.append("")

        lines.extend([
            "=" * 70,
            "",
        ])

        return "\n".join(lines)

    def to_json(self, report: ParityReport) -> str:
        """
        Format report as JSON.

        Args:
            report: The parity report

        Returns:
            JSON string
        """
        data = {
            "metadata": {
                "pytorch_version": report.pytorch_version,
                "mlx_compat_version": report.mlx_compat_version,
                "timestamp": report.timestamp,
            },
            "coverage": {
                "total_apis": report.total_pytorch_apis,
                "implemented": report.implemented_apis,
                "missing": report.missing_apis,
                "excluded": report.excluded_apis,
                "percentage": round(report.coverage_percentage, 2),
            },
            "signatures": {
                "matches": report.signature_matches,
                "mismatches": report.signature_mismatches,
                "skipped": report.signature_skipped,
                "percentage": round(report.signature_match_percentage, 2),
            },
            "module_stats": report.module_stats,
            "details": {
                "missing_apis": report.missing_api_list,
                "excluded_apis": report.excluded_api_list,
                "signature_mismatches": report.signature_mismatch_list,
            },
        }

        # Add numerical section if testing was performed
        if report.numerical_tested:
            data["numerical"] = {
                "matches": report.numerical_matches,
                "mismatches": report.numerical_mismatches,
                "skipped": report.numerical_skipped,
                "errors": report.numerical_errors,
                "percentage": round(report.numerical_match_percentage, 2),
            }
            data["details"]["numerical_mismatches"] = [
                {
                    "module": m.module,
                    "api": m.api,
                    "max_diff": m.max_diff,
                    "mean_diff": m.mean_diff,
                    "pytorch_shape": m.pytorch_output_shape,
                    "mlx_shape": m.mlx_output_shape,
                    "error": m.error,
                }
                for m in report.numerical_mismatch_list
            ]
            data["details"]["numerical_errors"] = report.numerical_error_list

        # Add behavioral section if testing was performed
        if report.behavioral_tested:
            data["behavioral"] = {
                "passed": report.behavioral_passed,
                "failed": report.behavioral_failed,
                "skipped": report.behavioral_skipped,
                "errors": report.behavioral_errors,
                "percentage": round(report.behavioral_pass_percentage, 2),
                "by_category": report.behavioral_by_category,
            }
            data["details"]["behavioral_failures"] = report.behavioral_failed_list

        return json.dumps(data, indent=2)

    def to_markdown(self, report: ParityReport) -> str:
        """
        Format report as Markdown.

        Args:
            report: The parity report

        Returns:
            Markdown string suitable for GitHub PRs
        """
        lines = [
            "# MLX-Compat API Parity Report",
            "",
            f"**PyTorch Version:** {report.pytorch_version}  ",
            f"**mlx_compat Version:** {report.mlx_compat_version}  ",
            f"**Generated:** {report.timestamp}",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|------:|",
            f"| Total PyTorch APIs | {report.total_pytorch_apis} |",
            f"| Implemented | {report.implemented_apis} |",
            f"| Missing | {report.missing_apis} |",
            f"| Excluded | {report.excluded_apis} |",
            f"| **Coverage** | **{report.coverage_percentage:.1f}%** |",
            "",
            "## Signature Validation",
            "",
            f"- Matching: {report.signature_matches}",
            f"- Mismatched: {report.signature_mismatches}",
            f"- Skipped: {report.signature_skipped}",
            f"- **Match rate: {report.signature_match_percentage:.1f}%**",
            "",
        ]

        # Module breakdown
        if report.module_stats:
            lines.extend([
                "## Coverage by Module",
                "",
                "| Module | Implemented | Total | Coverage |",
                "|--------|------------:|------:|---------:|",
            ])
            for module, stats in sorted(report.module_stats.items()):
                total = stats.get("total", 0) - stats.get("excluded", 0)
                implemented = stats.get("implemented", 0)
                pct = (implemented / total * 100) if total > 0 else 100.0
                lines.append(f"| `{module}` | {implemented} | {total} | {pct:.1f}% |")
            lines.append("")

        # Missing APIs
        if report.missing_api_list:
            lines.extend([
                "## Missing APIs",
                "",
                "<details>",
                "<summary>Click to expand (showing first 100)</summary>",
                "",
            ])
            for api in report.missing_api_list[:100]:
                lines.append(f"- `{api['module']}.{api['api']}`")
            if len(report.missing_api_list) > 100:
                lines.append(f"- ... and {len(report.missing_api_list) - 100} more")
            lines.extend(["", "</details>", ""])

        # Signature mismatches
        if report.signature_mismatch_list:
            lines.extend([
                "## Signature Mismatches",
                "",
                "<details>",
                "<summary>Click to expand (showing first 50)</summary>",
                "",
            ])
            for mismatch in report.signature_mismatch_list[:50]:
                lines.append(f"### `{mismatch['module']}.{mismatch['api']}`")
                lines.append("")
                lines.append(f"- **PyTorch:** `{mismatch.get('pytorch_signature', '?')}`")
                lines.append(f"- **MLX:** `{mismatch.get('mlx_signature', '?')}`")
                lines.append("- Differences:")
                for diff in mismatch.get("differences", []):
                    lines.append(f"  - {diff}")
                lines.append("")
            if len(report.signature_mismatch_list) > 50:
                lines.append(f"... and {len(report.signature_mismatch_list) - 50} more")
            lines.extend(["", "</details>", ""])

        # Numerical parity section
        if report.numerical_tested:
            lines.extend([
                "## Numerical Parity",
                "",
                f"- Matching: {report.numerical_matches}",
                f"- Mismatched: {report.numerical_mismatches}",
                f"- Skipped: {report.numerical_skipped}",
                f"- Errors: {report.numerical_errors}",
                f"- **Match rate: {report.numerical_match_percentage:.1f}%**",
                "",
            ])

            # Numerical mismatches
            if report.numerical_mismatch_list:
                lines.extend([
                    "### Numerical Mismatches",
                    "",
                    "<details>",
                    "<summary>Click to expand (showing first 50)</summary>",
                    "",
                    "| API | Max Diff | Shape |",
                    "|-----|----------|-------|",
                ])
                for mismatch in report.numerical_mismatch_list[:50]:
                    max_diff = f"{mismatch.max_diff:.2e}" if mismatch.max_diff else "N/A"
                    shape = str(mismatch.pytorch_output_shape) if mismatch.pytorch_output_shape else "N/A"
                    lines.append(f"| `{mismatch.module}.{mismatch.api}` | {max_diff} | {shape} |")
                if len(report.numerical_mismatch_list) > 50:
                    lines.append(f"| ... | {len(report.numerical_mismatch_list) - 50} more | |")
                lines.extend(["", "</details>", ""])

            # Numerical errors
            if report.numerical_error_list:
                lines.extend([
                    "### Numerical Errors",
                    "",
                    "<details>",
                    "<summary>Click to expand (showing first 30)</summary>",
                    "",
                ])
                for error in report.numerical_error_list[:30]:
                    lines.append(f"- `{error['module']}.{error['api']}`: {error.get('error', 'Unknown')}")
                if len(report.numerical_error_list) > 30:
                    lines.append(f"- ... and {len(report.numerical_error_list) - 30} more")
                lines.extend(["", "</details>", ""])

        # Behavioral parity section
        if report.behavioral_tested:
            lines.extend([
                "## Behavioral Parity",
                "",
                f"- Passed: {report.behavioral_passed}",
                f"- Failed: {report.behavioral_failed}",
                f"- Skipped: {report.behavioral_skipped}",
                f"- Errors: {report.behavioral_errors}",
                f"- **Pass rate: {report.behavioral_pass_percentage:.1f}%**",
                "",
            ])

            # By-category breakdown
            if report.behavioral_by_category:
                lines.extend([
                    "### Results by Category",
                    "",
                    "| Category | Passed | Failed | Pass Rate |",
                    "|----------|-------:|-------:|----------:|",
                ])
                for category, stats in sorted(report.behavioral_by_category.items()):
                    passed = stats.get("passed", 0)
                    failed = stats.get("failed", 0)
                    total = passed + failed
                    pct = (passed / total * 100) if total > 0 else 100.0
                    lines.append(f"| `{category}` | {passed} | {failed} | {pct:.1f}% |")
                lines.append("")

            # Behavioral failures
            if report.behavioral_failed_list:
                lines.extend([
                    "### Behavioral Failures",
                    "",
                    "<details>",
                    "<summary>Click to expand (showing first 30)</summary>",
                    "",
                ])
                for failure in report.behavioral_failed_list[:30]:
                    lines.append(f"- **[{failure['category']}] {failure['test']}**")
                    if failure.get('error'):
                        lines.append(f"  - Error: {failure['error']}")
                if len(report.behavioral_failed_list) > 30:
                    lines.append(f"- ... and {len(report.behavioral_failed_list) - 30} more")
                lines.extend(["", "</details>", ""])

        return "\n".join(lines)
