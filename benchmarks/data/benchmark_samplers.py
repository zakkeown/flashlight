"""
Sampler Benchmarks

Compares MLX sampler performance against PyTorch samplers.
"""

import time
import sys
sys.path.insert(0, '../..')

import numpy as np

try:
    import flashlight
    from flashlight.data import (
        SequentialSampler, RandomSampler, WeightedRandomSampler, BatchSampler
    )
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    from torch.utils.data import (
        SequentialSampler as TorchSequentialSampler,
        RandomSampler as TorchRandomSampler,
        WeightedRandomSampler as TorchWeightedRandomSampler,
        BatchSampler as TorchBatchSampler,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def benchmark_random_sampler(dataset_size, num_iterations=10):
    """Benchmark RandomSampler shuffle speed."""
    results = {}

    # MLX
    if MLX_AVAILABLE:
        data = list(range(dataset_size))
        start = time.perf_counter()
        for _ in range(num_iterations):
            sampler = RandomSampler(data, replacement=False)
            _ = list(sampler)
        elapsed = time.perf_counter() - start
        results['mlx'] = {
            'total_time': elapsed,
            'time_per_shuffle': elapsed / num_iterations,
            'shuffles_per_sec': num_iterations / elapsed,
        }

    # PyTorch
    if TORCH_AVAILABLE:
        data = list(range(dataset_size))
        start = time.perf_counter()
        for _ in range(num_iterations):
            sampler = TorchRandomSampler(data, replacement=False)
            _ = list(sampler)
        elapsed = time.perf_counter() - start
        results['torch'] = {
            'total_time': elapsed,
            'time_per_shuffle': elapsed / num_iterations,
            'shuffles_per_sec': num_iterations / elapsed,
        }

    return results


def benchmark_weighted_sampler(num_weights, num_samples, num_iterations=10):
    """Benchmark WeightedRandomSampler speed."""
    results = {}

    # Create weights
    np_weights = np.random.rand(num_weights).astype(np.float32)

    # MLX
    if MLX_AVAILABLE:
        weights = list(np_weights)
        start = time.perf_counter()
        for _ in range(num_iterations):
            sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
            _ = list(sampler)
        elapsed = time.perf_counter() - start
        results['mlx'] = {
            'total_time': elapsed,
            'time_per_sample': elapsed / num_iterations,
            'samples_per_sec': (num_samples * num_iterations) / elapsed,
        }

    # PyTorch
    if TORCH_AVAILABLE:
        weights = torch.from_numpy(np_weights)
        start = time.perf_counter()
        for _ in range(num_iterations):
            sampler = TorchWeightedRandomSampler(weights, num_samples, replacement=True)
            _ = list(sampler)
        elapsed = time.perf_counter() - start
        results['torch'] = {
            'total_time': elapsed,
            'time_per_sample': elapsed / num_iterations,
            'samples_per_sec': (num_samples * num_iterations) / elapsed,
        }

    return results


def benchmark_batch_sampler(dataset_size, batch_size, num_iterations=10):
    """Benchmark BatchSampler speed."""
    results = {}

    # MLX
    if MLX_AVAILABLE:
        data = list(range(dataset_size))
        start = time.perf_counter()
        for _ in range(num_iterations):
            base_sampler = RandomSampler(data)
            batch_sampler = BatchSampler(base_sampler, batch_size, drop_last=False)
            _ = list(batch_sampler)
        elapsed = time.perf_counter() - start
        results['mlx'] = {
            'total_time': elapsed,
            'time_per_iteration': elapsed / num_iterations,
        }

    # PyTorch
    if TORCH_AVAILABLE:
        data = list(range(dataset_size))
        start = time.perf_counter()
        for _ in range(num_iterations):
            base_sampler = TorchRandomSampler(data)
            batch_sampler = TorchBatchSampler(base_sampler, batch_size, drop_last=False)
            _ = list(batch_sampler)
        elapsed = time.perf_counter() - start
        results['torch'] = {
            'total_time': elapsed,
            'time_per_iteration': elapsed / num_iterations,
        }

    return results


def run_sampler_benchmarks():
    """Run all sampler benchmarks."""
    print("\n" + "=" * 70)
    print("Sampler Benchmark: MLX vs PyTorch")
    print("=" * 70)

    # RandomSampler benchmarks
    print("\n--- RandomSampler (shuffle) ---")
    for dataset_size in [1000, 10000, 100000, 1000000]:
        print(f"\nDataset size: {dataset_size:,}")
        results = benchmark_random_sampler(dataset_size, num_iterations=10)

        if 'mlx' in results:
            print(f"  MLX:     {results['mlx']['time_per_shuffle']*1000:.2f} ms/shuffle")
        if 'torch' in results:
            print(f"  PyTorch: {results['torch']['time_per_shuffle']*1000:.2f} ms/shuffle")
        if 'mlx' in results and 'torch' in results:
            speedup = results['torch']['time_per_shuffle'] / results['mlx']['time_per_shuffle']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")

    # WeightedRandomSampler benchmarks
    print("\n--- WeightedRandomSampler ---")
    for num_weights, num_samples in [(100, 1000), (1000, 10000), (10000, 100000)]:
        print(f"\nWeights: {num_weights}, Samples: {num_samples}")
        results = benchmark_weighted_sampler(num_weights, num_samples, num_iterations=10)

        if 'mlx' in results:
            print(f"  MLX:     {results['mlx']['samples_per_sec']/1e6:.2f}M samples/sec")
        if 'torch' in results:
            print(f"  PyTorch: {results['torch']['samples_per_sec']/1e6:.2f}M samples/sec")
        if 'mlx' in results and 'torch' in results:
            speedup = results['mlx']['samples_per_sec'] / results['torch']['samples_per_sec']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")

    # BatchSampler benchmarks
    print("\n--- BatchSampler ---")
    for dataset_size, batch_size in [(10000, 32), (100000, 64), (1000000, 128)]:
        print(f"\nDataset: {dataset_size:,}, Batch size: {batch_size}")
        results = benchmark_batch_sampler(dataset_size, batch_size, num_iterations=10)

        if 'mlx' in results:
            print(f"  MLX:     {results['mlx']['time_per_iteration']*1000:.2f} ms/iteration")
        if 'torch' in results:
            print(f"  PyTorch: {results['torch']['time_per_iteration']*1000:.2f} ms/iteration")
        if 'mlx' in results and 'torch' in results:
            speedup = results['torch']['time_per_iteration'] / results['mlx']['time_per_iteration']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")


if __name__ == '__main__':
    run_sampler_benchmarks()
