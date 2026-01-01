"""
DataLoader Benchmarks

Compares MLX DataLoader performance against PyTorch DataLoader.
"""

import time
import sys
sys.path.insert(0, '../..')

import numpy as np

try:
    import flashlight
    from flashlight.data import TensorDataset, DataLoader
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    from torch.utils.data import TensorDataset as TorchTensorDataset
    from torch.utils.data import DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def benchmark_mlx_dataloader(dataset_size, feature_dim, batch_size, shuffle, num_epochs=3, warmup=1):
    """Benchmark MLX DataLoader."""
    if not MLX_AVAILABLE:
        return None

    # Create dataset
    x = flashlight.randn(dataset_size, feature_dim)
    y = flashlight.randint(0, 10, (dataset_size,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Warmup
    for _ in range(warmup):
        for batch in loader:
            pass

    # Benchmark
    start_time = time.perf_counter()
    total_samples = 0

    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            total_samples += batch_x.shape[0]

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    throughput = total_samples / elapsed

    return {
        'elapsed_time': elapsed,
        'total_samples': total_samples,
        'throughput': throughput,
        'samples_per_sec': throughput,
    }


def benchmark_pytorch_dataloader(dataset_size, feature_dim, batch_size, shuffle, num_epochs=3, warmup=1):
    """Benchmark PyTorch DataLoader."""
    if not TORCH_AVAILABLE:
        return None

    # Create dataset
    x = torch.randn(dataset_size, feature_dim)
    y = torch.randint(0, 10, (dataset_size,))
    dataset = TorchTensorDataset(x, y)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    # Warmup
    for _ in range(warmup):
        for batch in loader:
            pass

    # Benchmark
    start_time = time.perf_counter()
    total_samples = 0

    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            total_samples += batch_x.shape[0]

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    throughput = total_samples / elapsed

    return {
        'elapsed_time': elapsed,
        'total_samples': total_samples,
        'throughput': throughput,
        'samples_per_sec': throughput,
    }


def format_throughput(samples_per_sec):
    """Format throughput for display."""
    if samples_per_sec >= 1e6:
        return f"{samples_per_sec / 1e6:.2f}M samples/sec"
    elif samples_per_sec >= 1e3:
        return f"{samples_per_sec / 1e3:.2f}K samples/sec"
    return f"{samples_per_sec:.2f} samples/sec"


def run_dataloader_benchmarks():
    """Run all DataLoader benchmarks."""
    print("\n" + "=" * 70)
    print("DataLoader Benchmark: MLX vs PyTorch")
    print("=" * 70)

    configs = [
        # Small datasets
        {"dataset_size": 1000, "feature_dim": 64, "batch_size": 32, "shuffle": False, "label": "Small (1K), no shuffle"},
        {"dataset_size": 1000, "feature_dim": 64, "batch_size": 32, "shuffle": True, "label": "Small (1K), shuffle"},
        # Medium datasets
        {"dataset_size": 10000, "feature_dim": 64, "batch_size": 64, "shuffle": False, "label": "Medium (10K), no shuffle"},
        {"dataset_size": 10000, "feature_dim": 64, "batch_size": 64, "shuffle": True, "label": "Medium (10K), shuffle"},
        {"dataset_size": 50000, "feature_dim": 128, "batch_size": 64, "shuffle": False, "label": "Medium (50K), no shuffle"},
        {"dataset_size": 50000, "feature_dim": 128, "batch_size": 64, "shuffle": True, "label": "Medium (50K), shuffle"},
        # Large datasets
        {"dataset_size": 100000, "feature_dim": 128, "batch_size": 128, "shuffle": False, "label": "Large (100K), no shuffle"},
        {"dataset_size": 100000, "feature_dim": 128, "batch_size": 128, "shuffle": True, "label": "Large (100K), shuffle"},
        {"dataset_size": 500000, "feature_dim": 64, "batch_size": 256, "shuffle": False, "label": "Large (500K), no shuffle"},
        {"dataset_size": 500000, "feature_dim": 64, "batch_size": 256, "shuffle": True, "label": "Large (500K), shuffle"},
    ]

    results = []

    for config in configs:
        label = config.pop("label")
        print(f"\n{label}:")
        print(f"  Dataset: {config['dataset_size']} samples, {config['feature_dim']} features")
        print(f"  Batch size: {config['batch_size']}, Shuffle: {config['shuffle']}")

        mlx_result = benchmark_mlx_dataloader(**config)
        torch_result = benchmark_pytorch_dataloader(**config)

        if mlx_result:
            print(f"  MLX:     {format_throughput(mlx_result['samples_per_sec'])} ({mlx_result['elapsed_time']:.3f}s)")
        else:
            print(f"  MLX:     Not available")

        if torch_result:
            print(f"  PyTorch: {format_throughput(torch_result['samples_per_sec'])} ({torch_result['elapsed_time']:.3f}s)")
        else:
            print(f"  PyTorch: Not available")

        if mlx_result and torch_result:
            speedup = mlx_result['samples_per_sec'] / torch_result['samples_per_sec']
            if speedup >= 1.0:
                print(f"  Speedup: {speedup:.2f}x (MLX faster)")
            else:
                print(f"  Speedup: {1/speedup:.2f}x (PyTorch faster)")

        results.append({
            "label": label,
            "config": config,
            "mlx": mlx_result,
            "torch": torch_result,
        })

    return results


if __name__ == '__main__':
    run_dataloader_benchmarks()
