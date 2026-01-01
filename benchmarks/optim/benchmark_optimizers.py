"""
Optimizer Benchmarks: MLX Compat vs PyTorch

Compares performance of MLX optimizers against PyTorch CPU and MPS backends.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import numpy as np
from typing import Dict, List, Tuple, Any

# MLX imports
try:
    import mlx.core as mx
    import flashlight
    import flashlight.nn as nn
    import flashlight.optim as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False


def create_mlx_params(sizes: List[Tuple[int, ...]], seed: int = 42) -> List[Any]:
    """Create MLX parameters with given sizes."""
    np.random.seed(seed)
    params = []
    for size in sizes:
        data = np.random.randn(*size).astype(np.float32)
        param = nn.Parameter(flashlight.tensor(data))
        param.grad = flashlight.tensor(np.random.randn(*size).astype(np.float32))
        params.append(param)
    return params


def create_torch_params(sizes: List[Tuple[int, ...]], device: str = 'cpu', seed: int = 42) -> List[torch.Tensor]:
    """Create PyTorch parameters with given sizes."""
    np.random.seed(seed)
    params = []
    for size in sizes:
        data = torch.from_numpy(np.random.randn(*size).astype(np.float32)).to(device)
        param = torch.nn.Parameter(data)
        param.grad = torch.from_numpy(np.random.randn(*size).astype(np.float32)).to(device)
        params.append(param)
    return params


def benchmark_mlx_optimizer(
    optimizer_class,
    params: List[Any],
    optimizer_kwargs: Dict,
    num_steps: int = 100,
    warmup_steps: int = 10
) -> float:
    """Benchmark MLX optimizer and return time per step in ms."""
    opt = optimizer_class(params, **optimizer_kwargs)

    # Warmup
    for _ in range(warmup_steps):
        opt.step()
        mx.eval([p._mlx_array for p in params])

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        opt.step()
        mx.eval([p._mlx_array for p in params])
    end = time.perf_counter()

    return (end - start) / num_steps * 1000  # ms per step


def benchmark_torch_optimizer(
    optimizer_class,
    params: List[torch.Tensor],
    optimizer_kwargs: Dict,
    device: str,
    num_steps: int = 100,
    warmup_steps: int = 10
) -> float:
    """Benchmark PyTorch optimizer and return time per step in ms."""
    opt = optimizer_class(params, **optimizer_kwargs)

    # Warmup
    for _ in range(warmup_steps):
        opt.step()
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        opt.step()
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_steps * 1000  # ms per step


def count_params(sizes: List[Tuple[int, ...]]) -> int:
    """Count total parameters."""
    return sum(np.prod(s) for s in sizes)


def run_optimizer_benchmark(
    mlx_opt_class,
    torch_opt_class,
    optimizer_kwargs: Dict,
    param_sizes: List[Tuple[int, ...]],
    num_steps: int = 100
) -> Dict[str, float]:
    """Run benchmark for a single optimizer configuration."""
    results = {}

    # MLX benchmark
    if MLX_AVAILABLE:
        mlx_params = create_mlx_params(param_sizes)
        results['mlx'] = benchmark_mlx_optimizer(mlx_opt_class, mlx_params, optimizer_kwargs, num_steps)

    # PyTorch CPU benchmark
    if TORCH_AVAILABLE:
        torch_params_cpu = create_torch_params(param_sizes, device='cpu')
        results['torch_cpu'] = benchmark_torch_optimizer(torch_opt_class, torch_params_cpu, optimizer_kwargs, 'cpu', num_steps)

    # PyTorch MPS benchmark
    if MPS_AVAILABLE:
        torch_params_mps = create_torch_params(param_sizes, device='mps')
        results['torch_mps'] = benchmark_torch_optimizer(torch_opt_class, torch_params_mps, optimizer_kwargs, 'mps', num_steps)

    return results


# Benchmark configurations
OPTIMIZERS = [
    ('SGD', optim.SGD, torch_optim.SGD, {'lr': 0.01}),
    ('SGD+momentum', optim.SGD, torch_optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
    ('Adam', optim.Adam, torch_optim.Adam, {'lr': 0.001}),
    ('AdamW', optim.AdamW, torch_optim.AdamW, {'lr': 0.001}),
    ('RMSprop', optim.RMSprop, torch_optim.RMSprop, {'lr': 0.01}),
    ('Adagrad', optim.Adagrad, torch_optim.Adagrad, {'lr': 0.01}),
    ('Adadelta', optim.Adadelta, torch_optim.Adadelta, {'lr': 1.0}),
]

PARAM_CONFIGS = [
    ('small', [(100, 100)]),  # 10K params
    ('medium', [(1000, 100), (100, 100)]),  # 110K params
    ('large', [(1000, 1000), (1000, 100)]),  # 1.1M params
]


def run_all_benchmarks() -> Dict:
    """Run all benchmarks and return results."""
    results = {}

    for opt_name, mlx_class, torch_class, opt_kwargs in OPTIMIZERS:
        results[opt_name] = {}

        for size_name, sizes in PARAM_CONFIGS:
            num_params = count_params(sizes)
            print(f"Benchmarking {opt_name} with {size_name} params ({num_params:,} total)...")

            try:
                benchmark_results = run_optimizer_benchmark(
                    mlx_class, torch_class, opt_kwargs, sizes,
                    num_steps=100
                )
                results[opt_name][size_name] = {
                    'num_params': num_params,
                    **benchmark_results
                }
            except Exception as e:
                print(f"  Error: {e}")
                results[opt_name][size_name] = {'error': str(e)}

    return results


def format_results_as_markdown(results: Dict) -> str:
    """Format benchmark results as markdown table."""
    lines = []
    lines.append("# Optimizer Benchmarks: MLX Compat vs PyTorch")
    lines.append("")
    lines.append("Time per optimization step (ms). Lower is better.")
    lines.append("")

    # Header
    header = "| Optimizer | Size | Params | MLX | PyTorch CPU | PyTorch MPS | MLX vs MPS |"
    lines.append(header)
    lines.append("|-----------|------|--------|-----|-------------|-------------|------------|")

    for opt_name, size_results in results.items():
        for size_name, data in size_results.items():
            if 'error' in data:
                continue

            num_params = data.get('num_params', 0)
            mlx_time = data.get('mlx', float('nan'))
            cpu_time = data.get('torch_cpu', float('nan'))
            mps_time = data.get('torch_mps', float('nan'))

            # Calculate speedup vs MPS
            if not np.isnan(mlx_time) and not np.isnan(mps_time) and mps_time > 0:
                speedup = mps_time / mlx_time
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            lines.append(
                f"| {opt_name} | {size_name} | {num_params:,} | "
                f"{mlx_time:.3f} | {cpu_time:.3f} | {mps_time:.3f} | {speedup_str} |"
            )

    return "\n".join(lines)


if __name__ == '__main__':
    print("=" * 60)
    print("Optimizer Benchmarks: MLX Compat vs PyTorch")
    print("=" * 60)
    print()

    if not MLX_AVAILABLE:
        print("ERROR: MLX not available")
        sys.exit(1)

    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not available, skipping comparison")

    if MPS_AVAILABLE:
        print(f"MPS Available: Yes")
    else:
        print(f"MPS Available: No (CPU-only comparison)")

    print()

    results = run_all_benchmarks()

    print()
    print(format_results_as_markdown(results))
