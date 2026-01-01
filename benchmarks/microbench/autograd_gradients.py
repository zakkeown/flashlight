"""
Autograd Gradient Benchmarks

Benchmarks comparing MLX Compat autograd gradient computation against PyTorch.
Tests both MPS and CPU backends for PyTorch.
"""

import time
import numpy as np
import sys
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

try:
    import mlx_compat
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX Compat not available")

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_MPS_AVAILABLE = False
    print("PyTorch not available")


@dataclass
class BenchmarkResult:
    name: str
    mlx_time_ms: float
    torch_cpu_time_ms: float
    torch_mps_time_ms: float
    shape: Tuple[int, ...]
    speedup_vs_cpu: float
    speedup_vs_mps: float


def warmup(func: Callable, iterations: int = 3) -> None:
    """Warmup function to avoid cold start."""
    for _ in range(iterations):
        func()


def benchmark(func: Callable, iterations: int = 100) -> float:
    """Benchmark a function and return average time in milliseconds."""
    # Sync before timing
    if MLX_AVAILABLE:
        mx.eval(mx.array([0]))
    if TORCH_AVAILABLE:
        if TORCH_MPS_AVAILABLE:
            torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    if MLX_AVAILABLE:
        mx.eval(mx.array([0]))
    if TORCH_AVAILABLE:
        if TORCH_MPS_AVAILABLE:
            torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def benchmark_max_backward(shape: Tuple[int, ...], iterations: int = 100) -> BenchmarkResult:
    """Benchmark max backward gradient computation."""
    data = np.random.randn(*shape).astype(np.float32)

    # MLX Compat
    def mlx_max_backward():
        x = mlx_compat.tensor(data, requires_grad=True)
        y = x.max()
        y.backward()
        return x.grad

    # PyTorch CPU
    def torch_cpu_backward():
        x = torch.tensor(data, requires_grad=True, device='cpu')
        y = x.max()
        y.backward()
        return x.grad

    # PyTorch MPS
    def torch_mps_backward():
        x = torch.tensor(data, requires_grad=True, device='mps')
        y = x.max()
        y.backward()
        torch.mps.synchronize()
        return x.grad

    warmup(mlx_max_backward)
    mlx_time = benchmark(mlx_max_backward, iterations)

    warmup(torch_cpu_backward)
    cpu_time = benchmark(torch_cpu_backward, iterations)

    if TORCH_MPS_AVAILABLE:
        warmup(torch_mps_backward)
        mps_time = benchmark(torch_mps_backward, iterations)
    else:
        mps_time = float('inf')

    return BenchmarkResult(
        name="max_backward",
        mlx_time_ms=mlx_time,
        torch_cpu_time_ms=cpu_time,
        torch_mps_time_ms=mps_time,
        shape=shape,
        speedup_vs_cpu=cpu_time / mlx_time,
        speedup_vs_mps=mps_time / mlx_time if mps_time != float('inf') else 0,
    )


def benchmark_matmul_backward(shapes: Tuple[Tuple[int, int], Tuple[int, int]], iterations: int = 100) -> BenchmarkResult:
    """Benchmark matmul backward gradient computation."""
    shape_a, shape_b = shapes
    data_a = np.random.randn(*shape_a).astype(np.float32)
    data_b = np.random.randn(*shape_b).astype(np.float32)

    # MLX Compat
    def mlx_matmul_backward():
        a = mlx_compat.tensor(data_a, requires_grad=True)
        b = mlx_compat.tensor(data_b, requires_grad=True)
        c = a @ b
        mlx_compat.sum(c).backward()
        return a.grad, b.grad

    # PyTorch CPU
    def torch_cpu_backward():
        a = torch.tensor(data_a, requires_grad=True, device='cpu')
        b = torch.tensor(data_b, requires_grad=True, device='cpu')
        c = a @ b
        c.sum().backward()
        return a.grad, b.grad

    # PyTorch MPS
    def torch_mps_backward():
        a = torch.tensor(data_a, requires_grad=True, device='mps')
        b = torch.tensor(data_b, requires_grad=True, device='mps')
        c = a @ b
        c.sum().backward()
        torch.mps.synchronize()
        return a.grad, b.grad

    warmup(mlx_matmul_backward)
    mlx_time = benchmark(mlx_matmul_backward, iterations)

    warmup(torch_cpu_backward)
    cpu_time = benchmark(torch_cpu_backward, iterations)

    if TORCH_MPS_AVAILABLE:
        warmup(torch_mps_backward)
        mps_time = benchmark(torch_mps_backward, iterations)
    else:
        mps_time = float('inf')

    return BenchmarkResult(
        name="matmul_backward",
        mlx_time_ms=mlx_time,
        torch_cpu_time_ms=cpu_time,
        torch_mps_time_ms=mps_time,
        shape=shape_a + shape_b,
        speedup_vs_cpu=cpu_time / mlx_time,
        speedup_vs_mps=mps_time / mlx_time if mps_time != float('inf') else 0,
    )


def benchmark_relu_backward(shape: Tuple[int, ...], iterations: int = 100) -> BenchmarkResult:
    """Benchmark ReLU backward gradient computation."""
    data = np.random.randn(*shape).astype(np.float32)

    # MLX Compat
    def mlx_relu_backward():
        x = mlx_compat.tensor(data, requires_grad=True)
        y = mlx_compat.relu(x)
        mlx_compat.sum(y).backward()
        return x.grad

    # PyTorch CPU
    def torch_cpu_backward():
        x = torch.tensor(data, requires_grad=True, device='cpu')
        y = torch.relu(x)
        y.sum().backward()
        return x.grad

    # PyTorch MPS
    def torch_mps_backward():
        x = torch.tensor(data, requires_grad=True, device='mps')
        y = torch.relu(x)
        y.sum().backward()
        torch.mps.synchronize()
        return x.grad

    warmup(mlx_relu_backward)
    mlx_time = benchmark(mlx_relu_backward, iterations)

    warmup(torch_cpu_backward)
    cpu_time = benchmark(torch_cpu_backward, iterations)

    if TORCH_MPS_AVAILABLE:
        warmup(torch_mps_backward)
        mps_time = benchmark(torch_mps_backward, iterations)
    else:
        mps_time = float('inf')

    return BenchmarkResult(
        name="relu_backward",
        mlx_time_ms=mlx_time,
        torch_cpu_time_ms=cpu_time,
        torch_mps_time_ms=mps_time,
        shape=shape,
        speedup_vs_cpu=cpu_time / mlx_time,
        speedup_vs_mps=mps_time / mlx_time if mps_time != float('inf') else 0,
    )


def benchmark_conv2d_backward(batch_channels_hw: Tuple[int, int, int, int], iterations: int = 50) -> BenchmarkResult:
    """Benchmark Conv2d backward gradient computation."""
    batch, in_ch, h, w = batch_channels_hw
    out_ch = in_ch * 2
    data = np.random.randn(batch, in_ch, h, w).astype(np.float32)

    # MLX Compat
    mlx_conv = mlx_compat.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def mlx_conv_backward():
        x = mlx_compat.tensor(data, requires_grad=True)
        y = mlx_conv(x)
        mlx_compat.sum(y).backward()
        return x.grad

    # PyTorch CPU
    torch_conv_cpu = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1).to('cpu')

    def torch_cpu_backward():
        x = torch.tensor(data, requires_grad=True, device='cpu')
        y = torch_conv_cpu(x)
        y.sum().backward()
        return x.grad

    # PyTorch MPS
    if TORCH_MPS_AVAILABLE:
        torch_conv_mps = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1).to('mps')

        def torch_mps_backward():
            x = torch.tensor(data, requires_grad=True, device='mps')
            y = torch_conv_mps(x)
            y.sum().backward()
            torch.mps.synchronize()
            return x.grad
    else:
        torch_mps_backward = lambda: None

    warmup(mlx_conv_backward)
    mlx_time = benchmark(mlx_conv_backward, iterations)

    warmup(torch_cpu_backward)
    cpu_time = benchmark(torch_cpu_backward, iterations)

    if TORCH_MPS_AVAILABLE:
        warmup(torch_mps_backward)
        mps_time = benchmark(torch_mps_backward, iterations)
    else:
        mps_time = float('inf')

    return BenchmarkResult(
        name="conv2d_backward",
        mlx_time_ms=mlx_time,
        torch_cpu_time_ms=cpu_time,
        torch_mps_time_ms=mps_time,
        shape=batch_channels_hw,
        speedup_vs_cpu=cpu_time / mlx_time,
        speedup_vs_mps=mps_time / mlx_time if mps_time != float('inf') else 0,
    )


def benchmark_chain_backward(shape: Tuple[int, ...], iterations: int = 100) -> BenchmarkResult:
    """Benchmark chained operations backward gradient computation."""
    data = np.random.randn(*shape).astype(np.float32)

    # MLX Compat
    def mlx_chain_backward():
        x = mlx_compat.tensor(data, requires_grad=True)
        y = mlx_compat.relu(mlx_compat.sigmoid(x * 2 + 1))
        mlx_compat.sum(y).backward()
        return x.grad

    # PyTorch CPU
    def torch_cpu_backward():
        x = torch.tensor(data, requires_grad=True, device='cpu')
        y = torch.relu(torch.sigmoid(x * 2 + 1))
        y.sum().backward()
        return x.grad

    # PyTorch MPS
    def torch_mps_backward():
        x = torch.tensor(data, requires_grad=True, device='mps')
        y = torch.relu(torch.sigmoid(x * 2 + 1))
        y.sum().backward()
        torch.mps.synchronize()
        return x.grad

    warmup(mlx_chain_backward)
    mlx_time = benchmark(mlx_chain_backward, iterations)

    warmup(torch_cpu_backward)
    cpu_time = benchmark(torch_cpu_backward, iterations)

    if TORCH_MPS_AVAILABLE:
        warmup(torch_mps_backward)
        mps_time = benchmark(torch_mps_backward, iterations)
    else:
        mps_time = float('inf')

    return BenchmarkResult(
        name="chain_backward",
        mlx_time_ms=mlx_time,
        torch_cpu_time_ms=cpu_time,
        torch_mps_time_ms=mps_time,
        shape=shape,
        speedup_vs_cpu=cpu_time / mlx_time,
        speedup_vs_mps=mps_time / mlx_time if mps_time != float('inf') else 0,
    )


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("AUTOGRAD GRADIENT BENCHMARKS: MLX Compat vs PyTorch")
    print("=" * 100)
    print(f"\n{'Operation':<20} {'Shape':<25} {'MLX (ms)':<12} {'CPU (ms)':<12} {'MPS (ms)':<12} {'vs CPU':<10} {'vs MPS':<10}")
    print("-" * 100)

    for r in results:
        shape_str = str(r.shape)[:22]
        mps_str = f"{r.torch_mps_time_ms:.3f}" if r.torch_mps_time_ms != float('inf') else "N/A"
        mps_speedup = f"{r.speedup_vs_mps:.2f}x" if r.speedup_vs_mps > 0 else "N/A"
        print(f"{r.name:<20} {shape_str:<25} {r.mlx_time_ms:<12.3f} {r.torch_cpu_time_ms:<12.3f} {mps_str:<12} {r.speedup_vs_cpu:.2f}x      {mps_speedup}")

    print("-" * 100)

    # Summary
    avg_cpu_speedup = sum(r.speedup_vs_cpu for r in results) / len(results)
    mps_results = [r for r in results if r.speedup_vs_mps > 0]
    avg_mps_speedup = sum(r.speedup_vs_mps for r in mps_results) / len(mps_results) if mps_results else 0

    print(f"\nSummary:")
    print(f"  Average speedup vs PyTorch CPU: {avg_cpu_speedup:.2f}x")
    if mps_results:
        print(f"  Average speedup vs PyTorch MPS: {avg_mps_speedup:.2f}x")
    print()


def run_benchmarks():
    """Run all benchmarks."""
    if not MLX_AVAILABLE or not TORCH_AVAILABLE:
        print("Both MLX Compat and PyTorch are required for benchmarks")
        return

    print("Running autograd gradient benchmarks...")
    print(f"PyTorch MPS available: {TORCH_MPS_AVAILABLE}")

    results = []

    # Max backward
    for shape in [(100, 100), (1000, 1000), (10000,)]:
        results.append(benchmark_max_backward(shape))

    # Matmul backward
    for shapes in [((64, 128), (128, 64)), ((256, 512), (512, 256)), ((1024, 1024), (1024, 1024))]:
        results.append(benchmark_matmul_backward(shapes))

    # ReLU backward
    for shape in [(1000, 1000), (10000,), (100, 100, 100)]:
        results.append(benchmark_relu_backward(shape))

    # Conv2d backward
    for batch_channels_hw in [(1, 3, 64, 64), (4, 16, 32, 32), (8, 32, 16, 16)]:
        results.append(benchmark_conv2d_backward(batch_channels_hw))

    # Chain backward
    for shape in [(1000, 1000), (10000,)]:
        results.append(benchmark_chain_backward(shape))

    print_results(results)


if __name__ == "__main__":
    run_benchmarks()
