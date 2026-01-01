"""
Collate Function Benchmarks

Compares MLX collate performance against PyTorch collate.
"""

import time
import sys
sys.path.insert(0, '../..')

import numpy as np

try:
    import flashlight
    from flashlight.data.dataloader import default_collate as mlx_collate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    from torch.utils.data._utils.collate import default_collate as torch_collate
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def benchmark_tensor_collate(batch_size, feature_dim, num_iterations=100):
    """Benchmark collating tensors."""
    results = {}

    # MLX
    if MLX_AVAILABLE:
        # Create samples
        samples = [flashlight.randn(feature_dim) for _ in range(batch_size)]

        # Warmup
        for _ in range(10):
            _ = mlx_collate(samples)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = mlx_collate(samples)
        elapsed = time.perf_counter() - start

        results['mlx'] = {
            'total_time': elapsed,
            'time_per_collate': elapsed / num_iterations,
            'collates_per_sec': num_iterations / elapsed,
        }

    # PyTorch
    if TORCH_AVAILABLE:
        # Create samples
        samples = [torch.randn(feature_dim) for _ in range(batch_size)]

        # Warmup
        for _ in range(10):
            _ = torch_collate(samples)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = torch_collate(samples)
        elapsed = time.perf_counter() - start

        results['torch'] = {
            'total_time': elapsed,
            'time_per_collate': elapsed / num_iterations,
            'collates_per_sec': num_iterations / elapsed,
        }

    return results


def benchmark_tuple_collate(batch_size, num_tensors, feature_dim, num_iterations=100):
    """Benchmark collating tuples of tensors."""
    results = {}

    # MLX
    if MLX_AVAILABLE:
        # Create samples
        samples = [
            tuple(flashlight.randn(feature_dim) for _ in range(num_tensors))
            for _ in range(batch_size)
        ]

        # Warmup
        for _ in range(10):
            _ = mlx_collate(samples)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = mlx_collate(samples)
        elapsed = time.perf_counter() - start

        results['mlx'] = {
            'total_time': elapsed,
            'time_per_collate': elapsed / num_iterations,
            'collates_per_sec': num_iterations / elapsed,
        }

    # PyTorch
    if TORCH_AVAILABLE:
        # Create samples
        samples = [
            tuple(torch.randn(feature_dim) for _ in range(num_tensors))
            for _ in range(batch_size)
        ]

        # Warmup
        for _ in range(10):
            _ = torch_collate(samples)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = torch_collate(samples)
        elapsed = time.perf_counter() - start

        results['torch'] = {
            'total_time': elapsed,
            'time_per_collate': elapsed / num_iterations,
            'collates_per_sec': num_iterations / elapsed,
        }

    return results


def benchmark_dict_collate(batch_size, num_keys, feature_dim, num_iterations=100):
    """Benchmark collating dicts of tensors."""
    results = {}

    keys = [f'key_{i}' for i in range(num_keys)]

    # MLX
    if MLX_AVAILABLE:
        # Create samples
        samples = [
            {k: flashlight.randn(feature_dim) for k in keys}
            for _ in range(batch_size)
        ]

        # Warmup
        for _ in range(10):
            _ = mlx_collate(samples)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = mlx_collate(samples)
        elapsed = time.perf_counter() - start

        results['mlx'] = {
            'total_time': elapsed,
            'time_per_collate': elapsed / num_iterations,
            'collates_per_sec': num_iterations / elapsed,
        }

    # PyTorch
    if TORCH_AVAILABLE:
        # Create samples
        samples = [
            {k: torch.randn(feature_dim) for k in keys}
            for _ in range(batch_size)
        ]

        # Warmup
        for _ in range(10):
            _ = torch_collate(samples)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = torch_collate(samples)
        elapsed = time.perf_counter() - start

        results['torch'] = {
            'total_time': elapsed,
            'time_per_collate': elapsed / num_iterations,
            'collates_per_sec': num_iterations / elapsed,
        }

    return results


def run_collate_benchmarks():
    """Run all collate benchmarks."""
    print("\n" + "=" * 70)
    print("Collate Function Benchmark: MLX vs PyTorch")
    print("=" * 70)

    # Tensor collation benchmarks
    print("\n--- Tensor Collation ---")
    for batch_size, feature_dim in [(32, 64), (64, 128), (128, 256), (256, 512)]:
        print(f"\nBatch: {batch_size}, Features: {feature_dim}")
        results = benchmark_tensor_collate(batch_size, feature_dim, num_iterations=100)

        if 'mlx' in results:
            print(f"  MLX:     {results['mlx']['time_per_collate']*1000:.3f} ms/collate")
        if 'torch' in results:
            print(f"  PyTorch: {results['torch']['time_per_collate']*1000:.3f} ms/collate")
        if 'mlx' in results and 'torch' in results:
            speedup = results['torch']['time_per_collate'] / results['mlx']['time_per_collate']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")

    # Tuple collation benchmarks
    print("\n--- Tuple Collation ---")
    for batch_size, num_tensors, feature_dim in [(32, 2, 64), (64, 3, 128), (128, 4, 256)]:
        print(f"\nBatch: {batch_size}, Tensors: {num_tensors}, Features: {feature_dim}")
        results = benchmark_tuple_collate(batch_size, num_tensors, feature_dim, num_iterations=100)

        if 'mlx' in results:
            print(f"  MLX:     {results['mlx']['time_per_collate']*1000:.3f} ms/collate")
        if 'torch' in results:
            print(f"  PyTorch: {results['torch']['time_per_collate']*1000:.3f} ms/collate")
        if 'mlx' in results and 'torch' in results:
            speedup = results['torch']['time_per_collate'] / results['mlx']['time_per_collate']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")

    # Dict collation benchmarks
    print("\n--- Dict Collation ---")
    for batch_size, num_keys, feature_dim in [(32, 3, 64), (64, 5, 128), (128, 10, 256)]:
        print(f"\nBatch: {batch_size}, Keys: {num_keys}, Features: {feature_dim}")
        results = benchmark_dict_collate(batch_size, num_keys, feature_dim, num_iterations=100)

        if 'mlx' in results:
            print(f"  MLX:     {results['mlx']['time_per_collate']*1000:.3f} ms/collate")
        if 'torch' in results:
            print(f"  PyTorch: {results['torch']['time_per_collate']*1000:.3f} ms/collate")
        if 'mlx' in results and 'torch' in results:
            speedup = results['torch']['time_per_collate'] / results['mlx']['time_per_collate']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")


if __name__ == '__main__':
    run_collate_benchmarks()
