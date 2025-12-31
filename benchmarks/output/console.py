"""
Console output formatter for benchmark results.

Provides human-readable tables and summaries.
"""

import platform
import datetime
from typing import List, Optional

from benchmarks.core.config import BenchmarkResult, BenchmarkSummary, BenchmarkLevel


class ConsoleFormatter:
    """
    Formats benchmark results for console output.

    Produces human-readable tables with timing statistics,
    PyTorch comparisons, and summary information.
    """

    def __init__(
        self,
        verbose: bool = False,
        show_header: bool = True,
        width: int = 100,
    ):
        self.verbose = verbose
        self.show_header = show_header
        self.width = width

    def format_results(
        self,
        results: List[BenchmarkResult],
        summary: Optional[BenchmarkSummary] = None,
    ) -> str:
        """Format all results as a string."""
        lines = []

        if self.show_header:
            lines.append(self._format_header())

        # Group by level
        operators = [r for r in results if r.level == BenchmarkLevel.OPERATOR]
        layers = [r for r in results if r.level == BenchmarkLevel.LAYER]
        models = [r for r in results if r.level == BenchmarkLevel.MODEL]

        if operators:
            lines.append(self._format_section("OPERATOR BENCHMARKS", operators))

        if layers:
            lines.append(self._format_section("LAYER BENCHMARKS", layers))

        if models:
            lines.append(self._format_section("MODEL BENCHMARKS", models))

        if summary:
            lines.append(self._format_summary(summary))

        lines.append("=" * self.width)

        return "\n".join(lines)

    def _format_header(self) -> str:
        """Format the header with system information."""
        lines = []
        lines.append("=" * self.width)
        lines.append(self._center("mlx_compat Benchmarking Suite v1.0.0"))
        lines.append("=" * self.width)

        # System info
        system = platform.system()
        version = platform.release()
        machine = platform.machine()
        lines.append(f"Platform: {system} {version} ({machine})")

        # Try to get processor info
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                cpu = result.stdout.strip()
                lines.append(f"Processor: {cpu}")
        except Exception:
            pass

        # Framework versions
        try:
            import mlx
            lines.append(f"MLX Version: {mlx.__version__}")
        except (ImportError, AttributeError):
            pass

        try:
            import torch
            lines.append(f"PyTorch Version: {torch.__version__}")
        except ImportError:
            lines.append("PyTorch: Not available")

        lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * self.width)

        return "\n".join(lines)

    def _center(self, text: str) -> str:
        """Center text within width."""
        padding = (self.width - len(text)) // 2
        return " " * padding + text

    def _format_section(
        self,
        title: str,
        results: List[BenchmarkResult],
    ) -> str:
        """Format a section of benchmark results."""
        lines = []
        lines.append("")
        lines.append(title)
        lines.append("-" * self.width)

        # Determine if PyTorch comparison is available
        has_pytorch = any(r.pytorch_timing is not None for r in results)

        if has_pytorch:
            # Header with comparison
            header = f"{'Name':<20} {'Config':<25} {'mlx(ms)':<12} {'torch(ms)':<12} {'Speedup':<10} {'Throughput':<15}"
            lines.append(header)
            lines.append("-" * self.width)

            for result in results:
                lines.append(self._format_result_with_comparison(result))
        else:
            # Header without comparison
            header = f"{'Name':<20} {'Config':<30} {'Time (ms)':<15} {'Throughput':<20}"
            lines.append(header)
            lines.append("-" * self.width)

            for result in results:
                lines.append(self._format_result(result))

        return "\n".join(lines)

    def _format_result(self, result: BenchmarkResult) -> str:
        """Format a single result without PyTorch comparison."""
        if result.error:
            return f"{result.name:<20} ERROR: {result.error}"

        config_str = self._format_config(result.input_config)
        time_str = f"{result.mlx_timing.mean_ms:.3f} +/- {result.mlx_timing.std_ms:.3f}"
        throughput_str = f"{result.throughput:.2f} {result.throughput_unit}"

        return f"{result.name:<20} {config_str:<30} {time_str:<15} {throughput_str:<20}"

    def _format_result_with_comparison(self, result: BenchmarkResult) -> str:
        """Format a single result with PyTorch comparison."""
        if result.error:
            return f"{result.name:<20} ERROR: {result.error}"

        config_str = self._format_config(result.input_config)[:25]
        mlx_time = f"{result.mlx_timing.mean_ms:.2f}+/-{result.mlx_timing.std_ms:.2f}"

        if result.pytorch_timing:
            torch_time = f"{result.pytorch_timing.mean_ms:.2f}+/-{result.pytorch_timing.std_ms:.2f}"
        else:
            torch_time = "N/A"

        if result.comparison:
            speedup = result.comparison.relative_performance
        else:
            speedup = "N/A"

        throughput = f"{result.throughput:.1f} {result.throughput_unit}"

        return f"{result.name:<20} {config_str:<25} {mlx_time:<12} {torch_time:<12} {speedup:<10} {throughput:<15}"

    def _format_config(self, config: dict) -> str:
        """Format input configuration as compact string."""
        parts = []
        for key, value in config.items():
            if key == "batch":
                parts.insert(0, f"b={value}")
            elif isinstance(value, (list, tuple)):
                parts.append(f"{key}={list(value)}")
            else:
                # Shorten common keys
                short_key = key.replace("_", "")[:4]
                parts.append(f"{short_key}={value}")

        return ", ".join(parts[:4])  # Limit to 4 parts

    def _format_summary(self, summary: BenchmarkSummary) -> str:
        """Format the summary section."""
        lines = []
        lines.append("")
        lines.append("=" * self.width)
        lines.append("SUMMARY")
        lines.append("=" * self.width)

        lines.append(f"Total benchmarks: {summary.total_benchmarks}")

        # By level
        level_parts = []
        for level, count in summary.by_level.items():
            level_parts.append(f"{level}: {count}")
        lines.append("  " + ", ".join(level_parts))

        lines.append("")
        lines.append("mlx_compat vs PyTorch:")
        if summary.faster_count + summary.slower_count > 0:
            total = summary.faster_count + summary.slower_count
            faster_pct = 100 * summary.faster_count / total
            lines.append(f"  Faster: {summary.faster_count} ({faster_pct:.1f}%)")
            lines.append(f"  Slower: {summary.slower_count} ({100-faster_pct:.1f}%)")
            lines.append(f"  Average speedup: {summary.average_speedup:.2f}x")
            lines.append(f"  Geometric mean: {summary.geometric_mean_speedup:.2f}x")
        else:
            lines.append("  No comparison data available")

        lines.append("")
        total_parity = summary.numerical_parity_passed + summary.numerical_parity_failed
        if total_parity > 0:
            lines.append(f"Numerical parity: {summary.numerical_parity_passed}/{total_parity} passed")
            if summary.numerical_parity_failed > 0:
                lines.append(f"  Failed: {summary.numerical_parity_failed}")

        if summary.errors:
            lines.append("")
            lines.append(f"Errors: {len(summary.errors)}")
            for error in summary.errors[:5]:  # Show first 5 errors
                lines.append(f"  - {error}")
            if len(summary.errors) > 5:
                lines.append(f"  ... and {len(summary.errors) - 5} more")

        return "\n".join(lines)

    def print_results(
        self,
        results: List[BenchmarkResult],
        summary: Optional[BenchmarkSummary] = None,
    ) -> None:
        """Print formatted results to console."""
        print(self.format_results(results, summary))

    def print_progress(self, name: str, current: int, total: int) -> None:
        """Print progress update."""
        pct = 100 * current / total if total > 0 else 0
        print(f"[{current}/{total}] ({pct:.0f}%) Running: {name}")
