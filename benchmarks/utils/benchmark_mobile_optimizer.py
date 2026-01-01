"""
Benchmark for flashlight.utils.mobile_optimizer module.

Compares optimization speed against PyTorch.
"""

import time
import sys

import mlx.core as mx
import flashlight
import flashlight.nn as nn
from flashlight.utils.mobile_optimizer import optimize_for_mobile, generate_mobile_module_lints

try:
    import torch
    import torch.utils.mobile_optimizer as torch_mobile
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_mlx_model(depth: int = 10, width: int = 256):
    """Create an MLX model with specified depth."""
    layers = []
    layers.append(nn.Linear(width, width))
    layers.append(nn.BatchNorm1d(width))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.1))

    for _ in range(depth - 2):
        layers.append(nn.Linear(width, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))

    layers.append(nn.Linear(width, 10))
    return nn.Sequential(*layers)


def create_torch_model(depth: int = 10, width: int = 256):
    """Create a PyTorch model with specified depth."""
    layers = []
    layers.append(torch.nn.Linear(width, width))
    layers.append(torch.nn.BatchNorm1d(width))
    layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Dropout(0.1))

    for _ in range(depth - 2):
        layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.BatchNorm1d(width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(0.1))

    layers.append(torch.nn.Linear(width, 10))
    return torch.nn.Sequential(*layers)


def benchmark_optimization_speed():
    """Benchmark optimization speed."""
    print("\n" + "=" * 60)
    print("Optimization Speed Benchmark")
    print("=" * 60)

    depths = [5, 10, 20]

    for depth in depths:
        # MLX
        mlx_model = create_mlx_model(depth=depth)
        mlx_model.eval()

        start = time.perf_counter()
        for _ in range(10):
            optimized = optimize_for_mobile(mlx_model)
        mlx_time = (time.perf_counter() - start) / 10

        print(f"\nDepth {depth} layers:")
        print(f"  MLX optimize_for_mobile: {mlx_time * 1000:.2f} ms")

        if HAS_TORCH:
            torch_model = create_torch_model(depth=depth)
            torch_model.eval()

            try:
                start = time.perf_counter()
                for _ in range(10):
                    scripted = torch.jit.script(torch_model)
                    torch_optimized = torch_mobile.optimize_for_mobile(scripted)
                torch_time = (time.perf_counter() - start) / 10
                print(f"  PyTorch optimize_for_mobile: {torch_time * 1000:.2f} ms")
                print(f"  Speedup: {torch_time / mlx_time:.2f}x")
            except Exception as e:
                print(f"  PyTorch: Failed ({e})")


def benchmark_linting_speed():
    """Benchmark linting speed."""
    print("\n" + "=" * 60)
    print("Linting Speed Benchmark")
    print("=" * 60)

    depths = [10, 50, 100]

    for depth in depths:
        mlx_model = create_mlx_model(depth=depth)

        start = time.perf_counter()
        for _ in range(100):
            lints = generate_mobile_module_lints(mlx_model)
        elapsed = (time.perf_counter() - start) / 100

        print(f"\nDepth {depth} layers:")
        print(f"  Lint time: {elapsed * 1000:.3f} ms")
        print(f"  Lints found: {len(lints)}")


def benchmark_dropout_removal():
    """Benchmark dropout removal specifically."""
    print("\n" + "=" * 60)
    print("Dropout Removal Benchmark")
    print("=" * 60)

    # Model with many dropout layers
    layers = []
    for i in range(50):
        layers.append(nn.Linear(64, 64))
        layers.append(nn.Dropout(0.1))

    model = nn.Sequential(*layers)
    model.eval()

    # Count dropouts before
    dropout_before = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))

    start = time.perf_counter()
    optimized = optimize_for_mobile(model)
    elapsed = time.perf_counter() - start

    # Count dropouts after
    dropout_after = sum(1 for m in optimized.modules() if isinstance(m, nn.Dropout))

    print(f"Dropouts before: {dropout_before}")
    print(f"Dropouts after: {dropout_after}")
    print(f"Optimization time: {elapsed * 1000:.2f} ms")


def benchmark_inference_improvement():
    """Benchmark inference speed improvement after optimization."""
    print("\n" + "=" * 60)
    print("Inference Speed Improvement (Dropout Removal)")
    print("=" * 60)

    model = create_mlx_model(depth=10, width=256)
    model.eval()

    x = flashlight.randn(32, 256)

    # Warmup
    for _ in range(5):
        _ = model(x)
    mx.synchronize()

    # Before optimization
    start = time.perf_counter()
    for _ in range(100):
        _ = model(x)
    mx.synchronize()
    before_time = (time.perf_counter() - start) / 100

    # Optimize
    optimized = optimize_for_mobile(model)

    # After optimization
    start = time.perf_counter()
    for _ in range(100):
        _ = optimized(x)
    mx.synchronize()
    after_time = (time.perf_counter() - start) / 100

    print(f"Before optimization: {before_time * 1000:.3f} ms/forward")
    print(f"After optimization: {after_time * 1000:.3f} ms/forward")
    print(f"Speedup: {before_time / after_time:.2f}x")


def benchmark_conv_bn_fusion():
    """Benchmark Conv-BN fusion inference improvement."""
    print("\n" + "=" * 60)
    print("Conv-BN Fusion Inference Speedup")
    print("=" * 60)

    # Create a realistic conv network with multiple Conv-BN pairs
    class ConvNet(nn.Module):
        def __init__(self, num_blocks=5):
            super().__init__()
            self.blocks = nn.ModuleList()
            channels = [3, 32, 64, 128, 256, 512]

            for i in range(num_blocks):
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(),
                ))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.pool(x)

    batch_sizes = [1, 8, 32]
    image_size = 32

    for batch_size in batch_sizes:
        model = ConvNet(num_blocks=5)
        model.eval()

        x = flashlight.randn(batch_size, 3, image_size, image_size)

        # Warmup
        for _ in range(5):
            _ = model(x)
        mx.synchronize()

        # Before optimization
        start = time.perf_counter()
        for _ in range(50):
            _ = model(x)
        mx.synchronize()
        before_time = (time.perf_counter() - start) / 50

        # Optimize (fuses Conv-BN pairs)
        optimized = optimize_for_mobile(model)

        # Count BN layers before/after
        bn_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        bn_after = sum(1 for m in optimized.modules() if isinstance(m, nn.BatchNorm2d))

        # After optimization
        start = time.perf_counter()
        for _ in range(50):
            _ = optimized(x)
        mx.synchronize()
        after_time = (time.perf_counter() - start) / 50

        print(f"\nBatch size {batch_size}:")
        print(f"  BatchNorm layers: {bn_before} -> {bn_after}")
        print(f"  Before fusion: {before_time * 1000:.3f} ms/forward")
        print(f"  After fusion: {after_time * 1000:.3f} ms/forward")
        speedup = before_time / after_time if after_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")


def main():
    print("MLX Compat Mobile Optimizer Benchmark")
    print("=" * 60)

    benchmark_optimization_speed()
    benchmark_linting_speed()
    benchmark_dropout_removal()
    benchmark_inference_improvement()
    benchmark_conv_bn_fusion()

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
