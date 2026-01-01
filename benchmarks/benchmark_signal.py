#!/usr/bin/env python3
"""
Signal Processing Benchmark - Window Functions

Compares flashlight.signal.windows vs torch.signal.windows performance.
"""

import sys
sys.path.insert(0, '..')

import time
import numpy as np

# MLX Compat
try:
    from flashlight.signal import windows as mlx_windows
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available")

# PyTorch
try:
    import torch
    import torch.signal.windows as torch_windows
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")


def benchmark_function(func, M, num_iterations=1000, warmup=100, **kwargs):
    """Benchmark a window function."""
    # Warmup
    for _ in range(warmup):
        _ = func(M, **kwargs)

    # Sync if MLX
    if MLX_AVAILABLE and hasattr(func, '__module__') and 'mlx' in func.__module__:
        mx.eval(func(M, **kwargs)._mlx_array)

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        result = func(M, **kwargs)

    # Sync for MLX
    if MLX_AVAILABLE and hasattr(result, '_mlx_array'):
        mx.eval(result._mlx_array)

    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_time_us = total_time_ms * 1000 / num_iterations

    return avg_time_us


def run_benchmark():
    """Run comprehensive window function benchmarks."""

    if not MLX_AVAILABLE:
        print("MLX not available. Skipping benchmarks.")
        return

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping benchmarks.")
        return

    print("=" * 80)
    print("Signal Window Functions Benchmark")
    print("MLX Compat vs PyTorch")
    print("=" * 80)
    print()

    # Window functions to benchmark (name, mlx_func, torch_func, kwargs)
    window_functions = [
        ("bartlett", mlx_windows.bartlett, torch_windows.bartlett, {}),
        ("blackman", mlx_windows.blackman, torch_windows.blackman, {}),
        ("cosine", mlx_windows.cosine, torch_windows.cosine, {}),
        ("exponential", mlx_windows.exponential, torch_windows.exponential, {"tau": 1.0}),
        ("gaussian", mlx_windows.gaussian, torch_windows.gaussian, {"std": 7.0}),
        ("general_hamming", mlx_windows.general_hamming, torch_windows.general_hamming, {"alpha": 0.54}),
        ("hamming", mlx_windows.hamming, torch_windows.hamming, {}),
        ("hann", mlx_windows.hann, torch_windows.hann, {}),
        ("kaiser", mlx_windows.kaiser, torch_windows.kaiser, {"beta": 12.0}),
        ("nuttall", mlx_windows.nuttall, torch_windows.nuttall, {}),
    ]

    # Window sizes to test
    sizes = [64, 256, 1024, 4096, 16384, 65536]

    # Results storage
    results = []

    print(f"{'Function':<18} {'Size':>8} {'MLX (us)':>12} {'PyTorch (us)':>12} {'Speedup':>10}")
    print("-" * 80)

    for name, mlx_func, torch_func, kwargs in window_functions:
        for M in sizes:
            # Adjust iterations based on size
            num_iterations = 1000 if M <= 4096 else 100

            mlx_time = benchmark_function(mlx_func, M, num_iterations=num_iterations, **kwargs)
            torch_time = benchmark_function(torch_func, M, num_iterations=num_iterations, **kwargs)

            speedup = torch_time / mlx_time if mlx_time > 0 else float('inf')

            results.append({
                'name': name,
                'size': M,
                'mlx_us': mlx_time,
                'torch_us': torch_time,
                'speedup': speedup
            })

            speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"
            print(f"{name:<18} {M:>8} {mlx_time:>12.2f} {torch_time:>12.2f} {speedup_str:>10}")

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Calculate overall statistics
    speedups = [r['speedup'] for r in results]
    avg_speedup = np.mean(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)

    faster_count = sum(1 for s in speedups if s >= 1.0)
    total_count = len(speedups)

    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Min Speedup: {min_speedup:.2f}x")
    print(f"Max Speedup: {max_speedup:.2f}x")
    print(f"MLX faster in {faster_count}/{total_count} benchmarks ({100*faster_count/total_count:.1f}%)")

    # Per-function summary
    print()
    print("Per-Function Average Speedup:")
    for name in set(r['name'] for r in results):
        func_results = [r for r in results if r['name'] == name]
        func_avg = np.mean([r['speedup'] for r in func_results])
        status = "FASTER" if func_avg >= 1.0 else "SLOWER"
        print(f"  {name:<18}: {func_avg:.2f}x ({status})")

    return results


if __name__ == "__main__":
    run_benchmark()
