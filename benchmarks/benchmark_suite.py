"""
Performance Benchmarking Suite

Benchmarks training throughput, inference latency, and memory usage.
"""

import sys
sys.path.insert(0, '..')

import time
import mlx_compat
import mlx_compat.nn as nn
import mlx_compat.optim as optim
import numpy as np


def benchmark_forward_pass(model, input_shape, num_iterations=100, warmup=10):
    """
    Benchmark forward pass throughput.

    Args:
        model: Neural network model
        input_shape: Shape of input tensor
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time per iteration (ms), throughput (samples/sec)
    """
    model.eval()

    # Warmup
    for _ in range(warmup):
        x = mlx_compat.randn(*input_shape)
        _ = model(x)

    # Benchmark
    batch_size = input_shape[0]
    start_time = time.time()

    for _ in range(num_iterations):
        x = mlx_compat.randn(*input_shape)
        _ = model(x)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput = (num_iterations * batch_size) / total_time

    return avg_time_ms, throughput


def benchmark_training_step(model, input_shape, num_classes, num_iterations=50, warmup=5):
    """
    Benchmark training step throughput.

    Args:
        model: Neural network model
        input_shape: Shape of input tensor
        num_classes: Number of output classes
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time per iteration (ms), throughput (samples/sec)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(warmup):
        x = mlx_compat.randn(*input_shape)
        y = mlx_compat.tensor(np.random.randint(0, num_classes, input_shape[0]))
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Benchmark
    batch_size = input_shape[0]
    start_time = time.time()

    for _ in range(num_iterations):
        x = mlx_compat.randn(*input_shape)
        y = mlx_compat.tensor(np.random.randint(0, num_classes, input_shape[0]))

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput = (num_iterations * batch_size) / total_time

    return avg_time_ms, throughput


def benchmark_mlp():
    """Benchmark MLP performance."""
    print("=" * 70)
    print("MLP Benchmark (MNIST-like)")
    print("=" * 70)

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Benchmark forward pass
    print("\nForward Pass (batch_size=64):")
    avg_time, throughput = benchmark_forward_pass(model, (64, 784), num_iterations=100)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")

    # Benchmark training step
    print("\nTraining Step (batch_size=64):")
    avg_time, throughput = benchmark_training_step(model, (64, 784), 10, num_iterations=50)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")


def benchmark_cnn():
    """Benchmark CNN performance."""
    print("\n" + "=" * 70)
    print("CNN Benchmark (CIFAR-10-like)")
    print("=" * 70)

    # Import CNN model
    sys.path.insert(0, '../examples')
    from models.cnn import SimpleCNN

    # Create model
    model = SimpleCNN(num_classes=10)

    # Benchmark forward pass
    print("\nForward Pass (batch_size=32):")
    avg_time, throughput = benchmark_forward_pass(model, (32, 3, 32, 32), num_iterations=50)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")

    # Benchmark training step
    print("\nTraining Step (batch_size=32):")
    avg_time, throughput = benchmark_training_step(model, (32, 3, 32, 32), 10, num_iterations=25)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")


def benchmark_resnet():
    """Benchmark ResNet performance."""
    print("\n" + "=" * 70)
    print("ResNet Benchmark (Small ResNet for CIFAR-10)")
    print("=" * 70)

    # Import ResNet model
    sys.path.insert(0, '../examples')
    from models.resnet import SmallResNet

    # Create model
    model = SmallResNet(num_classes=10)

    # Benchmark forward pass
    print("\nForward Pass (batch_size=16):")
    avg_time, throughput = benchmark_forward_pass(model, (16, 3, 32, 32), num_iterations=30)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")

    # Benchmark training step
    print("\nTraining Step (batch_size=16):")
    avg_time, throughput = benchmark_training_step(model, (16, 3, 32, 32), 10, num_iterations=15)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")


def benchmark_transformer():
    """Benchmark Transformer performance."""
    print("\n" + "=" * 70)
    print("Transformer Benchmark (Encoder Layer)")
    print("=" * 70)

    # Import Transformer model
    sys.path.insert(0, '../examples')
    from models.transformer import TransformerEncoderLayer

    # Create model
    model = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)

    # Benchmark forward pass
    print("\nForward Pass (batch_size=32, seq_len=50):")
    avg_time, throughput = benchmark_forward_pass(model, (32, 50, 256), num_iterations=50)
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")

    # Note: Training benchmark for Transformer is more complex due to masks
    print("\n  (Training benchmark skipped - requires specialized setup)")


def validate_mlp_accuracy():
    """
    Validate MLP forward pass accuracy against PyTorch.

    Creates identical MLX and PyTorch models with same weights and compares outputs.
    """
    print("\n" + "=" * 70)
    print("MLP Accuracy Validation")
    print("=" * 70)

    try:
        import torch
        import torch.nn as torch_nn
    except ImportError:
        print("\n  SKIPPED: PyTorch not available for comparison")
        return

    # Fixed input for reproducibility
    np.random.seed(42)
    input_data = np.random.randn(4, 784).astype(np.float32)

    # Create MLX model
    mlx_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    mlx_model.eval()

    # Create PyTorch model with same weights
    torch_model = torch_nn.Sequential(
        torch_nn.Linear(784, 256),
        torch_nn.ReLU(),
        torch_nn.Linear(256, 128),
        torch_nn.ReLU(),
        torch_nn.Linear(128, 10)
    )
    torch_model.eval()

    # Copy weights from MLX to PyTorch
    mlx_layers = [m for m in mlx_model.modules() if isinstance(m, nn.Linear)]
    torch_layers = [m for m in torch_model.modules() if isinstance(m, torch_nn.Linear)]

    for mlx_layer, torch_layer in zip(mlx_layers, torch_layers):
        # Get MLX weights and copy to PyTorch
        mlx_weight = mlx_layer.weight.numpy()
        mlx_bias = mlx_layer.bias.numpy()
        torch_layer.weight.data = torch.from_numpy(mlx_weight)
        torch_layer.bias.data = torch.from_numpy(mlx_bias)

    # Forward pass
    mlx_input = mlx_compat.tensor(input_data)
    torch_input = torch.from_numpy(input_data)

    with torch.no_grad():
        mlx_output = mlx_model(mlx_input).numpy()
        torch_output = torch_model(torch_input).numpy()

    # Compare outputs
    max_diff = np.max(np.abs(mlx_output - torch_output))
    mean_diff = np.mean(np.abs(mlx_output - torch_output))

    rtol, atol = 1e-4, 1e-5
    within_tol = np.allclose(mlx_output, torch_output, rtol=rtol, atol=atol)

    status = "PASS" if within_tol else "FAIL"
    print(f"\n  {status}: MLP forward pass accuracy")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Mean absolute difference: {mean_diff:.2e}")
    print(f"    Tolerance: rtol={rtol}, atol={atol}")


def print_summary():
    """Print benchmark summary and notes."""
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print("\nNotes:")
    print("  - All benchmarks run on MLX (Apple Silicon)")
    print("  - Times include data transfer and computation")
    print("  - Throughput measured in samples/second")
    print("  - Training includes forward, backward, and optimizer step")
    print("\nFor comparison with PyTorch:")
    print("  - Install PyTorch with MPS support")
    print("  - Run equivalent models on PyTorch MPS backend")
    print("  - Compare throughput and latency metrics")
    print("\nPerformance factors:")
    print("  - Layout conversions (NCHW ↔ NHWC) add overhead")
    print("  - Autograd system is tape-based (different from MLX native)")
    print("  - Some operations may not be fully optimized yet")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("mlx_compat Performance Benchmarking Suite")
    print("=" * 70)
    print("\nRunning benchmarks... This may take a few minutes.\n")

    try:
        benchmark_mlp()
        benchmark_cnn()
        benchmark_resnet()
        benchmark_transformer()
        validate_mlp_accuracy()
        print_summary()
    except KeyboardInterrupt:
        print("\n\nBenchmarking interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
