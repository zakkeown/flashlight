"""
JSON output formatter for benchmark results.

Provides machine-readable output for CI/automation.
"""

import json
import platform
import datetime
from typing import List, Dict, Any, Optional

from benchmarks.core.config import BenchmarkResult, BenchmarkSummary, BenchmarkLevel


class JSONFormatter:
    """
    Formats benchmark results as JSON.

    Produces structured JSON output suitable for:
    - CI/CD pipelines
    - Automated regression testing
    - Historical performance tracking
    - Data visualization tools
    """

    def __init__(self, indent: int = 2, include_raw_times: bool = False):
        """
        Initialize JSON formatter.

        Args:
            indent: JSON indentation level
            include_raw_times: Include all raw timing data (can be large)
        """
        self.indent = indent
        self.include_raw_times = include_raw_times

    def format_results(
        self,
        results: List[BenchmarkResult],
        summary: Optional[BenchmarkSummary] = None,
    ) -> str:
        """Format all results as JSON string."""
        output = self._build_output_dict(results, summary)
        return json.dumps(output, indent=self.indent)

    def _build_output_dict(
        self,
        results: List[BenchmarkResult],
        summary: Optional[BenchmarkSummary],
    ) -> Dict[str, Any]:
        """Build the output dictionary."""
        return {
            "metadata": self._build_metadata(),
            "results": self._build_results(results),
            "summary": summary.to_dict() if summary else None,
        }

    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata section."""
        metadata = {
            "benchmark_suite_version": "1.0.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "platform": {
                "os": platform.system(),
                "os_version": platform.release(),
                "architecture": platform.machine(),
            },
            "frameworks": {},
        }

        # Get processor info
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                metadata["platform"]["processor"] = result.stdout.strip()
        except Exception:
            pass

        # Framework versions
        try:
            import mlx
            metadata["frameworks"]["mlx"] = mlx.__version__
        except (ImportError, AttributeError):
            pass

        try:
            import mlx_compat
            metadata["frameworks"]["mlx_compat"] = getattr(mlx_compat, '__version__', 'unknown')
        except ImportError:
            pass

        try:
            import torch
            metadata["frameworks"]["pytorch"] = torch.__version__
            metadata["frameworks"]["pytorch_mps_available"] = torch.backends.mps.is_available()
        except ImportError:
            metadata["frameworks"]["pytorch"] = None

        return metadata

    def _build_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build results section grouped by level."""
        operators = []
        layers = []
        models = []

        for result in results:
            formatted = self._format_result(result)

            if result.level == BenchmarkLevel.OPERATOR:
                operators.append(formatted)
            elif result.level == BenchmarkLevel.LAYER:
                layers.append(formatted)
            elif result.level == BenchmarkLevel.MODEL:
                models.append(formatted)

        return {
            "operators": operators,
            "layers": layers,
            "models": models,
        }

    def _format_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Format a single benchmark result."""
        output = {
            "name": result.name,
            "input_config": result.input_config,
            "dtype": result.dtype,
            "timestamp": result.timestamp,
        }

        # MLX timing
        output["mlx_compat"] = {
            "mean_ms": round(result.mlx_timing.mean_ms, 4),
            "std_ms": round(result.mlx_timing.std_ms, 4),
            "min_ms": round(result.mlx_timing.min_ms, 4),
            "max_ms": round(result.mlx_timing.max_ms, 4),
            "median_ms": round(result.mlx_timing.median_ms, 4),
            "percentiles": {
                str(k): round(v, 4)
                for k, v in result.mlx_timing.percentiles.items()
            },
            "total_iterations": result.mlx_timing.total_iterations,
        }

        if self.include_raw_times:
            output["mlx_compat"]["raw_times_ms"] = [
                round(t, 4) for t in result.mlx_timing.times_ms
            ]

        # Throughput
        output["throughput"] = {
            "value": round(result.throughput, 4),
            "unit": result.throughput_unit,
        }

        # PyTorch timing
        if result.pytorch_timing is not None:
            output["pytorch"] = {
                "mean_ms": round(result.pytorch_timing.mean_ms, 4),
                "std_ms": round(result.pytorch_timing.std_ms, 4),
                "min_ms": round(result.pytorch_timing.min_ms, 4),
                "max_ms": round(result.pytorch_timing.max_ms, 4),
                "median_ms": round(result.pytorch_timing.median_ms, 4),
                "percentiles": {
                    str(k): round(v, 4)
                    for k, v in result.pytorch_timing.percentiles.items()
                },
            }

        # Comparison stats
        if result.comparison is not None:
            output["comparison"] = {
                "speedup": round(result.comparison.speedup, 4),
                "relative_performance": result.comparison.relative_performance,
                "numerical_match": result.comparison.numerical_match,
                "max_abs_diff": result.comparison.max_abs_diff,
            }

        # Memory stats
        if result.mlx_memory is not None:
            output["mlx_memory"] = {
                "peak_mb": round(result.mlx_memory.peak_mb, 2),
                "allocated_mb": round(result.mlx_memory.allocated_mb, 2),
                "delta_mb": round(result.mlx_memory.delta_mb, 2),
            }
            if result.mlx_memory.bandwidth_gbps is not None:
                output["mlx_memory"]["bandwidth_gbps"] = round(
                    result.mlx_memory.bandwidth_gbps, 2
                )

        # Error
        if result.error:
            output["error"] = result.error

        return output

    def save_results(
        self,
        results: List[BenchmarkResult],
        filepath: str,
        summary: Optional[BenchmarkSummary] = None,
    ) -> None:
        """Save results to a JSON file."""
        json_str = self.format_results(results, summary)
        with open(filepath, 'w') as f:
            f.write(json_str)

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from a JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def format_for_ci(
    results: List[BenchmarkResult],
    summary: Optional[BenchmarkSummary] = None,
) -> str:
    """
    Format results in a compact format suitable for CI logs.

    Returns a single-line JSON for each result.
    """
    lines = []

    for result in results:
        compact = {
            "name": result.name,
            "level": result.level.value,
            "mlx_ms": round(result.mlx_timing.mean_ms, 3),
        }

        if result.comparison:
            compact["speedup"] = round(result.comparison.speedup, 2)
            compact["match"] = result.comparison.numerical_match

        if result.error:
            compact["error"] = result.error

        lines.append(json.dumps(compact))

    return "\n".join(lines)
