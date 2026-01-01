"""
High-precision timing utilities for benchmarking.

Features:
- Nanosecond precision using time.perf_counter_ns()
- Warmup iterations to eliminate JIT overhead
- Multiple trials for statistical reliability
- Proper synchronization for MLX and PyTorch
- Comprehensive statistics including percentiles
"""

import time
import statistics
import gc
from typing import Callable, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from benchmarks.core.config import TimingStats


def _force_eval(result: Any) -> None:
    """
    Force evaluation of MLX tensor results.

    MLX operations are lazy - this ensures the computation
    actually runs before we stop timing.
    """
    if result is None:
        return

    try:
        import mlx.core as mx

        # Handle flashlight tensors (have _mlx_array attribute)
        if hasattr(result, '_mlx_array'):
            mx.eval(result._mlx_array)
        # Handle raw MLX arrays
        elif isinstance(result, mx.array):
            mx.eval(result)
        # Handle tuples (e.g., from attention returning (output, weights))
        elif isinstance(result, tuple):
            for item in result:
                _force_eval(item)
    except ImportError:
        pass
    except Exception:
        pass


def sync_mlx() -> None:
    """
    Synchronize MLX operations.

    Forces evaluation of any pending computations to ensure
    accurate timing measurements.
    """
    try:
        import mlx.core as mx
        # mx.synchronize() ensures all pending Metal operations complete
        mx.synchronize()
    except ImportError:
        pass
    except AttributeError:
        # Fallback for older MLX versions
        try:
            import mlx.core as mx
            mx.eval(mx.array([0.0]))
        except Exception:
            pass


def sync_pytorch(device: str = "mps") -> None:
    """
    Synchronize PyTorch operations.

    Args:
        device: PyTorch device ("mps", "cuda", "cpu")
    """
    try:
        import torch
        if device == "mps":
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        # CPU doesn't need synchronization
    except (ImportError, AttributeError):
        pass


class Timer:
    """
    High-precision timer with warmup and statistical analysis.

    Provides accurate timing measurements for both MLX and PyTorch
    operations with proper synchronization and statistical analysis.

    Example:
        timer = Timer(warmup_iterations=10, benchmark_iterations=100)

        # Time MLX operation
        result = timer.time_function(mlx_matmul, a, b)
        print(f"Mean: {result.mean_ms:.3f} ms")

        # Time PyTorch operation
        timer.sync_fn = lambda: sync_pytorch("mps")
        result = timer.time_function(torch_matmul, a, b)
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        num_trials: int = 5,
        min_time_seconds: float = 0.1,
        percentiles: Optional[List[int]] = None,
        sync_fn: Optional[Callable[[], None]] = None,
        force_gc: bool = True,
    ):
        """
        Initialize timer.

        Args:
            warmup_iterations: Iterations before timing starts
            benchmark_iterations: Iterations to time per trial
            num_trials: Number of complete trials to run
            min_time_seconds: Minimum total benchmark time
            percentiles: Percentile values to calculate
            sync_fn: Synchronization function (default: sync_mlx)
            force_gc: Force garbage collection between trials
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.num_trials = num_trials
        self.min_time_seconds = min_time_seconds
        self.percentiles = percentiles or [50, 90, 95, 99]
        self.sync_fn = sync_fn or sync_mlx
        self.force_gc = force_gc

    def time_function(
        self,
        fn: Callable[..., Any],
        *args: Any,
        setup_fn: Optional[Callable[[], None]] = None,
        teardown_fn: Optional[Callable[[], None]] = None,
        **kwargs: Any,
    ) -> TimingStats:
        """
        Time a function with proper warmup and synchronization.

        Args:
            fn: Function to benchmark
            args: Positional arguments for fn
            setup_fn: Optional function called before each iteration
            teardown_fn: Optional function called after each iteration
            kwargs: Keyword arguments for fn

        Returns:
            TimingStats with comprehensive timing statistics
        """
        all_times: List[float] = []

        for trial in range(self.num_trials):
            # Force garbage collection between trials
            if self.force_gc:
                gc.collect()

            # Warmup phase
            for _ in range(self.warmup_iterations):
                if setup_fn:
                    setup_fn()
                _ = fn(*args, **kwargs)
                self.sync_fn()
                if teardown_fn:
                    teardown_fn()

            # Benchmark phase
            trial_times: List[float] = []
            for _ in range(self.benchmark_iterations):
                if setup_fn:
                    setup_fn()

                start = time.perf_counter_ns()
                result = fn(*args, **kwargs)
                # Force evaluation of the result for MLX tensors
                _force_eval(result)
                self.sync_fn()
                end = time.perf_counter_ns()

                if teardown_fn:
                    teardown_fn()

                # Convert nanoseconds to milliseconds
                trial_times.append((end - start) / 1e6)

            all_times.extend(trial_times)

        return self._compute_stats(all_times)

    def time_paired(
        self,
        fn_a: Callable[..., Any],
        fn_b: Callable[..., Any],
        args_a: Tuple[Any, ...],
        args_b: Tuple[Any, ...],
        sync_a: Optional[Callable[[], None]] = None,
        sync_b: Optional[Callable[[], None]] = None,
    ) -> Tuple[TimingStats, TimingStats]:
        """
        Time two functions with interleaved execution for fair comparison.

        Alternates between functions to minimize effects of thermal throttling
        and background processes.

        Args:
            fn_a: First function to benchmark
            fn_b: Second function to benchmark
            args_a: Arguments for first function
            args_b: Arguments for second function
            sync_a: Sync function for first (default: sync_mlx)
            sync_b: Sync function for second (default: sync_pytorch)

        Returns:
            Tuple of TimingStats for both functions
        """
        sync_a = sync_a or sync_mlx
        sync_b = sync_b or (lambda: sync_pytorch("mps"))

        times_a: List[float] = []
        times_b: List[float] = []

        for trial in range(self.num_trials):
            if self.force_gc:
                gc.collect()

            # Warmup both
            for _ in range(self.warmup_iterations):
                _ = fn_a(*args_a)
                sync_a()
                _ = fn_b(*args_b)
                sync_b()

            # Interleaved benchmark
            for _ in range(self.benchmark_iterations):
                # Time function A
                start = time.perf_counter_ns()
                _ = fn_a(*args_a)
                sync_a()
                end = time.perf_counter_ns()
                times_a.append((end - start) / 1e6)

                # Time function B
                start = time.perf_counter_ns()
                _ = fn_b(*args_b)
                sync_b()
                end = time.perf_counter_ns()
                times_b.append((end - start) / 1e6)

        return self._compute_stats(times_a), self._compute_stats(times_b)

    def _compute_stats(self, times: List[float]) -> TimingStats:
        """Compute comprehensive timing statistics."""
        if not times:
            return TimingStats(
                times_ms=[],
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                median_ms=0.0,
                percentiles={},
                total_iterations=0,
                warmup_iterations=self.warmup_iterations * self.num_trials,
            )

        return TimingStats(
            times_ms=times,
            mean_ms=statistics.mean(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_ms=min(times),
            max_ms=max(times),
            median_ms=statistics.median(times),
            percentiles=self._calculate_percentiles(times),
            total_iterations=len(times),
            warmup_iterations=self.warmup_iterations * self.num_trials,
        )

    def _calculate_percentiles(self, times: List[float]) -> Dict[int, float]:
        """Calculate percentile values from timing data."""
        sorted_times = sorted(times)
        n = len(sorted_times)
        result = {}

        for p in self.percentiles:
            # Linear interpolation for percentile
            idx = (n - 1) * p / 100.0
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            fraction = idx - lower

            value = sorted_times[lower] * (1 - fraction) + sorted_times[upper] * fraction
            result[p] = value

        return result


def benchmark_function(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs: Any,
) -> Tuple[float, float]:
    """
    Simple convenience function for quick benchmarking.

    Args:
        fn: Function to benchmark
        args: Positional arguments for fn
        warmup: Warmup iterations
        iterations: Benchmark iterations
        kwargs: Keyword arguments for fn

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    timer = Timer(
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
        num_trials=1,
    )
    result = timer.time_function(fn, *args, **kwargs)
    return result.mean_ms, result.std_ms


def quick_bench(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> float:
    """
    Quick single-value benchmark for testing.

    Args:
        fn: Function to benchmark
        args: Arguments for fn
        kwargs: Keyword arguments for fn

    Returns:
        Mean time in milliseconds
    """
    mean_ms, _ = benchmark_function(fn, *args, warmup=5, iterations=20, **kwargs)
    return mean_ms
