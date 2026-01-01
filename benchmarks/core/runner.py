"""
Benchmark runner orchestrator.

Discovers, executes, and aggregates benchmark results across
all granularity levels (operator, layer, model).
"""

import fnmatch
import math
from typing import List, Dict, Optional, Callable, Any, Type
from dataclasses import dataclass

from benchmarks.core.config import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkLevel,
    BenchmarkSummary,
    TimingStats,
    ComparisonStats,
)
from benchmarks.core.timing import Timer, sync_mlx, sync_pytorch
from benchmarks.core.comparison import FrameworkComparator
from benchmarks.core.memory import MLXMemoryTracker, PyTorchMemoryTracker


@dataclass
class RegisteredBenchmark:
    """A registered benchmark with metadata."""
    name: str
    level: BenchmarkLevel
    benchmark_class: Type['BaseBenchmark']


class BenchmarkRegistry:
    """Registry for benchmark discovery and management."""

    def __init__(self):
        self._benchmarks: Dict[str, RegisteredBenchmark] = {}

    def register(
        self,
        name: str,
        level: BenchmarkLevel,
        benchmark_class: Type['BaseBenchmark'],
    ) -> None:
        """Register a benchmark class."""
        key = f"{level.value}:{name}"
        self._benchmarks[key] = RegisteredBenchmark(
            name=name,
            level=level,
            benchmark_class=benchmark_class,
        )

    def get_benchmarks(
        self,
        level: Optional[BenchmarkLevel] = None,
        filter_pattern: Optional[str] = None,
    ) -> List[RegisteredBenchmark]:
        """Get registered benchmarks, optionally filtered."""
        results = []

        for key, benchmark in self._benchmarks.items():
            # Filter by level
            if level is not None and level != BenchmarkLevel.ALL:
                if benchmark.level != level:
                    continue

            # Filter by name pattern
            if filter_pattern is not None:
                if not fnmatch.fnmatch(benchmark.name.lower(), filter_pattern.lower()):
                    continue

            results.append(benchmark)

        return results

    def list_benchmarks(self) -> List[Dict[str, str]]:
        """List all registered benchmarks."""
        return [
            {"name": b.name, "level": b.level.value}
            for b in self._benchmarks.values()
        ]


# Global registry
_registry = BenchmarkRegistry()


def register_benchmark(
    name: str,
    level: BenchmarkLevel,
) -> Callable[[Type['BaseBenchmark']], Type['BaseBenchmark']]:
    """Decorator to register a benchmark class."""
    def decorator(cls: Type['BaseBenchmark']) -> Type['BaseBenchmark']:
        _registry.register(name, level, cls)
        return cls
    return decorator


class BaseBenchmark:
    """
    Base class for all benchmarks.

    Subclasses should implement:
    - get_input_configs(): Return list of input configurations
    - create_mlx_inputs(config): Create flashlight inputs
    - create_pytorch_inputs(config, device): Create PyTorch inputs
    - mlx_operation(*inputs): The MLX operation to benchmark
    - pytorch_operation(*inputs): The PyTorch operation to benchmark
    """

    name: str = "base"
    level: BenchmarkLevel = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def get_input_configs(self) -> List[Dict[str, Any]]:
        """Return list of input configurations to benchmark."""
        raise NotImplementedError

    def create_mlx_inputs(self, config: Dict[str, Any]) -> tuple:
        """Create flashlight inputs for this config."""
        raise NotImplementedError

    def create_pytorch_inputs(
        self, config: Dict[str, Any], device: str
    ) -> tuple:
        """Create PyTorch inputs for this config."""
        raise NotImplementedError

    def mlx_operation(self, *inputs: Any) -> Any:
        """MLX operation to benchmark."""
        raise NotImplementedError

    def pytorch_operation(self, *inputs: Any) -> Any:
        """PyTorch operation to benchmark."""
        raise NotImplementedError

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> tuple:
        """
        Calculate throughput for this operation.

        Returns:
            Tuple of (throughput_value, throughput_unit)
        """
        # Default: operations per second
        if time_ms > 0:
            ops_per_sec = 1000.0 / time_ms
            return ops_per_sec, "ops/sec"
        return 0.0, "ops/sec"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        """Calculate total bytes transferred for bandwidth estimation."""
        return 0

    def run(self) -> List[BenchmarkResult]:
        """Run all configurations for this benchmark."""
        results = []

        for input_config in self.get_input_configs():
            try:
                result = self._run_single(input_config)
                results.append(result)
            except Exception as e:
                # Record error but continue
                results.append(BenchmarkResult(
                    name=self.name,
                    level=self.level,
                    input_config=input_config,
                    mlx_timing=TimingStats(
                        times_ms=[], mean_ms=0, std_ms=0,
                        min_ms=0, max_ms=0, median_ms=0,
                        percentiles={}, total_iterations=0,
                        warmup_iterations=0,
                    ),
                    throughput=0,
                    throughput_unit="ops/sec",
                    error=str(e),
                ))

        return results

    def _run_single(self, config: Dict[str, Any]) -> BenchmarkResult:
        """Run benchmark for a single configuration."""
        timer = Timer(
            warmup_iterations=self.config.warmup_iterations,
            benchmark_iterations=self.config.benchmark_iterations,
            num_trials=self.config.num_trials,
            sync_fn=sync_mlx,
        )

        # Create MLX inputs
        mlx_inputs = self.create_mlx_inputs(config)

        # Time MLX operation
        mlx_timing = timer.time_function(
            lambda: self.mlx_operation(*mlx_inputs)
        )

        # Calculate throughput
        throughput, throughput_unit = self.calculate_throughput(
            config, mlx_timing.mean_ms
        )

        # PyTorch comparison
        pytorch_timing = None
        comparison = None

        if self.config.compare_pytorch:
            try:
                import torch

                # Check device availability
                device = self.config.pytorch_device
                if device == "mps" and not torch.backends.mps.is_available():
                    device = "cpu"

                # Create PyTorch inputs
                pytorch_inputs = self.create_pytorch_inputs(config, device)

                # Time PyTorch operation
                timer.sync_fn = lambda: sync_pytorch(device)
                pytorch_timing = timer.time_function(
                    lambda: self.pytorch_operation(*pytorch_inputs)
                )

                # Calculate comparison
                speedup = pytorch_timing.mean_ms / mlx_timing.mean_ms if mlx_timing.mean_ms > 0 else 1.0

                if speedup >= 1.0:
                    relative_perf = f"{speedup:.2f}x faster"
                else:
                    pct_slower = (1.0 - speedup) * 100
                    relative_perf = f"{speedup:.2f}x ({pct_slower:.1f}% slower)"

                # Numerical comparison
                mlx_out = self.mlx_operation(*mlx_inputs)
                pytorch_out = self.pytorch_operation(*pytorch_inputs)
                numerical_match, max_diff = self._check_numerical(mlx_out, pytorch_out)

                comparison = ComparisonStats(
                    speedup=speedup,
                    relative_performance=relative_perf,
                    numerical_match=numerical_match,
                    max_abs_diff=max_diff,
                )

            except ImportError:
                pass  # PyTorch not available
            except Exception as e:
                # Log error but don't fail
                comparison = ComparisonStats(
                    speedup=0.0,
                    relative_performance=f"Error: {e}",
                    numerical_match=False,
                    max_abs_diff=float('inf'),
                )

        # Memory profiling
        mlx_memory = None
        if self.config.track_memory:
            tracker = MLXMemoryTracker()
            tensor_bytes = self.calculate_bytes(config)
            mlx_memory = tracker.profile_function(
                lambda: self.mlx_operation(*mlx_inputs),
                tensor_bytes=tensor_bytes if tensor_bytes > 0 else None,
                time_ms=mlx_timing.mean_ms,
            )

        return BenchmarkResult(
            name=self.name,
            level=self.level,
            input_config=config,
            mlx_timing=mlx_timing,
            throughput=throughput,
            throughput_unit=throughput_unit,
            pytorch_timing=pytorch_timing,
            comparison=comparison,
            mlx_memory=mlx_memory,
        )

    def _check_numerical(self, mlx_out: Any, pytorch_out: Any) -> tuple:
        """Check numerical parity between outputs."""
        import numpy as np

        try:
            # Convert to numpy
            if hasattr(mlx_out, 'numpy'):
                mlx_np = mlx_out.numpy()
            else:
                mlx_np = np.array(mlx_out)

            if hasattr(pytorch_out, 'detach'):
                pytorch_np = pytorch_out.detach().cpu().numpy()
            else:
                pytorch_np = np.array(pytorch_out)

            max_diff = float(np.max(np.abs(mlx_np - pytorch_np)))

            np.testing.assert_allclose(
                mlx_np, pytorch_np,
                rtol=self.config.rtol,
                atol=self.config.atol,
            )
            return True, max_diff

        except Exception:
            return False, float('inf')


class BenchmarkRunner:
    """
    Main orchestrator for running benchmarks.

    Example:
        config = BenchmarkConfig(compare_pytorch=True)
        runner = BenchmarkRunner(config)

        # Run all benchmarks
        results = runner.run(level=BenchmarkLevel.ALL)

        # Get summary
        summary = runner.summarize(results)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._benchmarks: List[BaseBenchmark] = []

    def add_benchmark(self, benchmark: BaseBenchmark) -> None:
        """Add a benchmark to run."""
        self._benchmarks.append(benchmark)

    def add_benchmarks(self, benchmarks: List[BaseBenchmark]) -> None:
        """Add multiple benchmarks to run."""
        self._benchmarks.extend(benchmarks)

    def discover_benchmarks(
        self,
        level: BenchmarkLevel = BenchmarkLevel.ALL,
        filter_pattern: Optional[str] = None,
    ) -> None:
        """
        Discover and add benchmarks from registered classes.

        Args:
            level: Benchmark level to discover
            filter_pattern: Glob pattern to filter by name
        """
        registered = _registry.get_benchmarks(level, filter_pattern)

        for reg in registered:
            benchmark = reg.benchmark_class(self.config)
            self._benchmarks.append(benchmark)

    def load_benchmark_modules(self) -> None:
        """Load benchmark modules to trigger registration."""
        # Import all benchmark modules to trigger registration
        try:
            from benchmarks import operators
            from benchmarks import layers
            from benchmarks import models
        except ImportError:
            pass

    def run(
        self,
        level: Optional[BenchmarkLevel] = None,
        filter_pattern: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[BenchmarkResult]:
        """
        Execute benchmarks.

        Args:
            level: Optional level filter
            filter_pattern: Optional name filter
            progress_callback: Callback(name, current, total) for progress

        Returns:
            List of benchmark results
        """
        all_results: List[BenchmarkResult] = []

        # Filter benchmarks if needed
        benchmarks = self._benchmarks
        if level is not None and level != BenchmarkLevel.ALL:
            benchmarks = [b for b in benchmarks if b.level == level]

        if filter_pattern is not None:
            benchmarks = [
                b for b in benchmarks
                if fnmatch.fnmatch(b.name.lower(), filter_pattern.lower())
            ]

        total = len(benchmarks)
        for i, benchmark in enumerate(benchmarks):
            if progress_callback:
                progress_callback(benchmark.name, i + 1, total)

            results = benchmark.run()
            all_results.extend(results)

        return all_results

    def list_benchmarks(self) -> List[Dict[str, str]]:
        """List all available benchmarks."""
        return [
            {"name": b.name, "level": b.level.value}
            for b in self._benchmarks
        ]

    def summarize(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """
        Summarize benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            BenchmarkSummary with aggregated statistics
        """
        total = len(results)
        by_level: Dict[str, int] = {}
        faster_count = 0
        slower_count = 0
        speedups: List[float] = []
        parity_passed = 0
        parity_failed = 0
        errors: List[str] = []

        for result in results:
            # Count by level
            level_str = result.level.value
            by_level[level_str] = by_level.get(level_str, 0) + 1

            # Track errors
            if result.error:
                errors.append(f"{result.name}: {result.error}")
                continue

            # Comparison stats
            if result.comparison:
                speedups.append(result.comparison.speedup)
                if result.comparison.speedup >= 1.0:
                    faster_count += 1
                else:
                    slower_count += 1

                if result.comparison.numerical_match:
                    parity_passed += 1
                else:
                    parity_failed += 1

        # Calculate averages
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0

        # Geometric mean for speedup
        if speedups:
            log_sum = sum(math.log(s) for s in speedups if s > 0)
            geo_mean = math.exp(log_sum / len(speedups))
        else:
            geo_mean = 1.0

        return BenchmarkSummary(
            total_benchmarks=total,
            by_level=by_level,
            faster_count=faster_count,
            slower_count=slower_count,
            average_speedup=avg_speedup,
            geometric_mean_speedup=geo_mean,
            numerical_parity_passed=parity_passed,
            numerical_parity_failed=parity_failed,
            errors=errors,
        )
