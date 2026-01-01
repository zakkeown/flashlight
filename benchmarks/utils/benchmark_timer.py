"""
Benchmark the Timer class performance.

Compares MLX Timer against PyTorch Timer for overhead and accuracy.
"""

import time
import numpy as np

import mlx.core as mx
import flashlight
from flashlight.utils.benchmark import Timer, Compare

try:
    import torch
    import torch.utils.benchmark as torch_benchmark
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def benchmark_timer_overhead():
    """Measure Timer overhead vs raw timing."""
    print("=" * 60)
    print("Timer Overhead Benchmark")
    print("=" * 60)

    # Measure raw timing overhead
    iterations = 1000

    # Raw time.perf_counter
    start = time.perf_counter()
    for _ in range(iterations):
        t1 = time.perf_counter()
        x = 1 + 1
        t2 = time.perf_counter()
    raw_total = time.perf_counter() - start
    raw_per_iteration = raw_total / iterations

    print(f"Raw time.perf_counter overhead: {raw_per_iteration * 1e6:.2f} us/iteration")

    # MLX Timer
    timer = Timer(stmt="x = 1 + 1")

    start = time.perf_counter()
    for _ in range(iterations):
        m = timer.timeit(1)
    mlx_timer_total = time.perf_counter() - start
    mlx_per_iteration = mlx_timer_total / iterations

    print(f"MLX Timer overhead: {mlx_per_iteration * 1e6:.2f} us/iteration")
    print(f"Overhead ratio: {mlx_per_iteration / raw_per_iteration:.1f}x")

    # MLX Timer with autorange
    start = time.perf_counter()
    m = timer.blocked_autorange(min_run_time=0.1)
    mlx_autorange_time = time.perf_counter() - start

    print(f"\nAutorange (0.1s min): {mlx_autorange_time:.3f}s total, {len(m.raw_times)} measurements")
    print(f"Median per-iteration: {m.median * 1e9:.2f} ns")


def benchmark_mlx_operations():
    """Benchmark common MLX operations."""
    print("\n" + "=" * 60)
    print("MLX Operations Benchmark")
    print("=" * 60)

    results = []

    # Matrix operations
    for size in [100, 500, 1000]:
        timer = Timer(
            stmt="mx.matmul(a, b)",
            setup=f"import mlx.core as mx; a = mx.random.normal(({size}, {size})); b = mx.random.normal(({size}, {size}))",
            label="matmul",
            sub_label=f"{size}x{size}",
        )
        results.append(timer.blocked_autorange(min_run_time=0.2))

    # Element-wise operations
    for size in [1000, 10000, 100000]:
        timer = Timer(
            stmt="mx.exp(a)",
            setup=f"import mlx.core as mx; a = mx.random.normal(({size},))",
            label="exp",
            sub_label=f"{size}",
        )
        results.append(timer.blocked_autorange(min_run_time=0.2))

    # Display results
    compare = Compare(results)
    compare.print()


def benchmark_vs_pytorch():
    """Compare Timer performance against PyTorch Timer."""
    if not HAS_TORCH:
        print("\n[Skipping PyTorch comparison - torch not available]")
        return

    print("\n" + "=" * 60)
    print("MLX Timer vs PyTorch Timer")
    print("=" * 60)

    # Simple arithmetic
    print("\nSimple arithmetic (x = a + b):")

    # MLX
    mlx_timer = Timer(
        stmt="c = a + b",
        setup="import mlx.core as mx; a = mx.ones((100, 100)); b = mx.ones((100, 100))",
        label="add",
        sub_label="MLX",
    )
    mlx_m = mlx_timer.blocked_autorange(min_run_time=0.2)

    # PyTorch
    torch_timer = torch_benchmark.Timer(
        stmt="c = a + b",
        setup="import torch; a = torch.ones((100, 100)); b = torch.ones((100, 100))",
        label="add",
        sub_label="PyTorch",
    )
    torch_m = torch_timer.blocked_autorange(min_run_time=0.2)

    print(f"  MLX:     {mlx_m.median * 1e6:.2f} us (median)")
    print(f"  PyTorch: {torch_m.median * 1e6:.2f} us (median)")

    # Matrix multiplication
    print("\nMatrix multiplication (1000x1000):")

    mlx_timer = Timer(
        stmt="mx.matmul(a, b); mx.synchronize()",
        setup="import mlx.core as mx; a = mx.random.normal((1000, 1000)); b = mx.random.normal((1000, 1000))",
        label="matmul",
        sub_label="MLX",
    )
    mlx_m = mlx_timer.blocked_autorange(min_run_time=0.5)

    # PyTorch on MPS (if available)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_timer = torch_benchmark.Timer(
        stmt="torch.matmul(a, b); torch.mps.synchronize() if torch.backends.mps.is_available() else None",
        setup=f"import torch; a = torch.randn((1000, 1000), device='{device}'); b = torch.randn((1000, 1000), device='{device}')",
        label="matmul",
        sub_label=f"PyTorch ({device})",
    )
    torch_m = torch_timer.blocked_autorange(min_run_time=0.5)

    print(f"  MLX:     {mlx_m.median * 1e3:.2f} ms (median)")
    print(f"  PyTorch: {torch_m.median * 1e3:.2f} ms (median)")
    print(f"  Speedup: {torch_m.median / mlx_m.median:.2f}x")


def benchmark_compare_display():
    """Benchmark Compare table generation speed."""
    print("\n" + "=" * 60)
    print("Compare Table Generation Benchmark")
    print("=" * 60)

    from flashlight.utils.benchmark import TaskSpec, Measurement

    # Generate many measurements
    measurements = []
    for i in range(100):
        spec = TaskSpec(
            stmt=f"op_{i}",
            label="benchmark",
            sub_label=f"op_{i}",
            description="test",
        )
        # Random times around 1ms
        times = list(np.random.normal(0.001, 0.0001, 10))
        measurements.append(Measurement(
            number_per_run=1000,
            raw_times=times,
            task_spec=spec,
        ))

    # Time table generation
    start = time.perf_counter()
    compare = Compare(measurements)
    output = str(compare)
    elapsed = time.perf_counter() - start

    print(f"Generated table for 100 measurements in {elapsed * 1000:.2f} ms")
    print(f"Table length: {len(output)} characters")
    print(f"Target: <100ms [{'PASS' if elapsed < 0.1 else 'FAIL'}]")


def main():
    """Run all benchmarks."""
    print("MLX Compat Utils Benchmark Suite")
    print("=" * 60)

    benchmark_timer_overhead()
    benchmark_mlx_operations()
    benchmark_vs_pytorch()
    benchmark_compare_display()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
