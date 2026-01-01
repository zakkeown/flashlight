"""
Benchmark gradient checkpointing.

Compares memory usage and speed with/without checkpointing,
and compares performance against PyTorch.
"""

import time
import gc

import mlx.core as mx
import mlx_compat
import mlx_compat.nn as nn
from mlx_compat.utils.checkpoint import checkpoint, checkpoint_sequential

try:
    import torch
    import torch.utils.checkpoint as torch_checkpoint
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_mlx_model(num_layers=8, hidden_size=512):
    """Create a simple MLP model for testing."""
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def create_torch_model(num_layers=8, hidden_size=512):
    """Create equivalent PyTorch model."""
    if not HAS_TORCH:
        return None

    layers = []
    for i in range(num_layers):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        if i < num_layers - 1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def benchmark_forward_pass():
    """Benchmark forward pass with and without checkpointing."""
    print("=" * 60)
    print("Forward Pass Benchmark")
    print("=" * 60)

    batch_size = 32
    hidden_size = 512
    num_layers = 8

    model = create_mlx_model(num_layers, hidden_size)

    # Create input
    x = mlx_compat.randn(batch_size, hidden_size)
    x.requires_grad = True

    # Warmup
    _ = model(x)
    mx.eval(_._mlx_array)

    # Without checkpoint
    gc.collect()
    start = time.perf_counter()
    for _ in range(100):
        out = model(x)
        mx.eval(out._mlx_array)
    normal_time = (time.perf_counter() - start) / 100

    print(f"Without checkpoint: {normal_time * 1000:.3f} ms/forward")

    # With checkpoint_sequential
    gc.collect()
    layers_list = list(model.children())

    start = time.perf_counter()
    for _ in range(100):
        out = checkpoint_sequential(layers_list, segments=4, input=x)
        mx.eval(out._mlx_array)
    checkpoint_time = (time.perf_counter() - start) / 100

    print(f"With checkpoint:    {checkpoint_time * 1000:.3f} ms/forward")
    print(f"Overhead:           {checkpoint_time / normal_time:.2f}x")


def benchmark_memory_usage():
    """Benchmark memory usage with and without checkpointing."""
    print("\n" + "=" * 60)
    print("Memory Usage Benchmark")
    print("=" * 60)

    batch_size = 64
    hidden_size = 1024
    num_layers = 16

    model = create_mlx_model(num_layers, hidden_size)

    # Create input
    x = mlx_compat.randn(batch_size, hidden_size)
    x.requires_grad = True

    # Force garbage collection
    gc.collect()
    mx.synchronize()

    # Get baseline memory
    baseline_memory = mx.metal.get_active_memory() if hasattr(mx.metal, 'get_active_memory') else 0

    # Without checkpoint
    out = model(x)
    mx.eval(out._mlx_array)

    normal_memory = mx.metal.get_active_memory() if hasattr(mx.metal, 'get_active_memory') else 0
    normal_diff = normal_memory - baseline_memory

    print(f"Without checkpoint: {normal_diff / 1024 / 1024:.2f} MB")

    # Clean up
    del out
    gc.collect()
    mx.synchronize()

    # With checkpoint
    x2 = mlx_compat.randn(batch_size, hidden_size)
    x2.requires_grad = True

    baseline_memory = mx.metal.get_active_memory() if hasattr(mx.metal, 'get_active_memory') else 0

    layers_list = list(model.children())
    out = checkpoint_sequential(layers_list, segments=4, input=x2)
    mx.eval(out._mlx_array)

    checkpoint_memory = mx.metal.get_active_memory() if hasattr(mx.metal, 'get_active_memory') else 0
    checkpoint_diff = checkpoint_memory - baseline_memory

    print(f"With checkpoint:    {checkpoint_diff / 1024 / 1024:.2f} MB")

    if normal_diff > 0 and checkpoint_diff > 0:
        reduction = (1 - checkpoint_diff / normal_diff) * 100
        print(f"Memory reduction:   {reduction:.1f}%")
    else:
        print("(Memory tracking not available on this system)")


def benchmark_vs_pytorch():
    """Compare checkpointing performance against PyTorch."""
    if not HAS_TORCH:
        print("\n[Skipping PyTorch comparison - torch not available]")
        return

    print("\n" + "=" * 60)
    print("MLX vs PyTorch Checkpoint Performance")
    print("=" * 60)

    batch_size = 32
    hidden_size = 512
    num_layers = 8
    num_iterations = 50

    # Create models
    mlx_model = create_mlx_model(num_layers, hidden_size)
    torch_model = create_torch_model(num_layers, hidden_size)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_model = torch_model.to(device)

    # MLX - without checkpoint
    mlx_x = mlx_compat.randn(batch_size, hidden_size)
    mlx_x.requires_grad = True

    # Warmup
    _ = mlx_model(mlx_x)
    mx.eval(_._mlx_array)

    start = time.perf_counter()
    for _ in range(num_iterations):
        out = mlx_model(mlx_x)
        mx.eval(out._mlx_array)
    mlx_normal_time = (time.perf_counter() - start) / num_iterations

    # MLX - with checkpoint
    layers_list = list(mlx_model.children())
    start = time.perf_counter()
    for _ in range(num_iterations):
        out = checkpoint_sequential(layers_list, segments=4, input=mlx_x)
        mx.eval(out._mlx_array)
    mlx_checkpoint_time = (time.perf_counter() - start) / num_iterations

    # PyTorch - without checkpoint
    torch_x = torch.randn(batch_size, hidden_size, device=device)
    torch_x.requires_grad = True

    # Warmup
    _ = torch_model(torch_x)
    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        out = torch_model(torch_x)
        if device == "mps":
            torch.mps.synchronize()
    torch_normal_time = (time.perf_counter() - start) / num_iterations

    # PyTorch - with checkpoint
    start = time.perf_counter()
    for _ in range(num_iterations):
        out = torch_checkpoint.checkpoint_sequential(
            torch_model, segments=4, input=torch_x, use_reentrant=True
        )
        if device == "mps":
            torch.mps.synchronize()
    torch_checkpoint_time = (time.perf_counter() - start) / num_iterations

    print("\nWithout Checkpoint:")
    print(f"  MLX:     {mlx_normal_time * 1000:.3f} ms")
    print(f"  PyTorch: {torch_normal_time * 1000:.3f} ms ({device})")
    print(f"  Ratio:   {torch_normal_time / mlx_normal_time:.2f}x")

    print("\nWith Checkpoint:")
    print(f"  MLX:     {mlx_checkpoint_time * 1000:.3f} ms")
    print(f"  PyTorch: {torch_checkpoint_time * 1000:.3f} ms ({device})")
    print(f"  Ratio:   {torch_checkpoint_time / mlx_checkpoint_time:.2f}x")

    print("\nCheckpoint Overhead:")
    print(f"  MLX:     {mlx_checkpoint_time / mlx_normal_time:.2f}x")
    print(f"  PyTorch: {torch_checkpoint_time / torch_normal_time:.2f}x")


def benchmark_different_segments():
    """Benchmark different number of checkpoint segments."""
    print("\n" + "=" * 60)
    print("Segment Count Benchmark")
    print("=" * 60)

    batch_size = 32
    hidden_size = 512
    num_layers = 16

    model = create_mlx_model(num_layers, hidden_size)
    layers_list = list(model.children())

    x = mlx_compat.randn(batch_size, hidden_size)
    x.requires_grad = True

    segment_counts = [1, 2, 4, 8, 16]

    print(f"\n{'Segments':<10} {'Time (ms)':<12} {'Relative':<10}")
    print("-" * 35)

    baseline = None
    for segments in segment_counts:
        # Warmup
        out = checkpoint_sequential(layers_list, segments=segments, input=x)
        mx.eval(out._mlx_array)

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            out = checkpoint_sequential(layers_list, segments=segments, input=x)
            mx.eval(out._mlx_array)
        elapsed = (time.perf_counter() - start) / 50

        if baseline is None:
            baseline = elapsed

        print(f"{segments:<10} {elapsed * 1000:<12.3f} {elapsed / baseline:<10.2f}x")


def main():
    """Run all checkpoint benchmarks."""
    print("MLX Compat Checkpoint Benchmark Suite")
    print("=" * 60)

    benchmark_forward_pass()
    benchmark_memory_usage()
    benchmark_vs_pytorch()
    benchmark_different_segments()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
