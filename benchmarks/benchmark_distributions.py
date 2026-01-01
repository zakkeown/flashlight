"""
Probability Distributions Benchmarking Suite

Benchmarks sampling throughput and log_prob computation for MLX distributions
vs PyTorch distributions.
"""

import sys
sys.path.insert(0, '..')

import time
import numpy as np

# Check for MLX
try:
    import mlx_compat
    from mlx_compat import distributions as mlx_dist
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available")

# Check for PyTorch
try:
    import torch
    import torch.distributions as torch_dist
    TORCH_AVAILABLE = True
    # Check for MPS
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    print("Warning: PyTorch not available")


def benchmark_sampling(distribution_class, dist_params, sample_shape, num_iterations=100, warmup=10):
    """
    Benchmark sampling throughput.

    Args:
        distribution_class: Distribution class to benchmark
        dist_params: Parameters to pass to distribution constructor
        sample_shape: Shape of samples to draw
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Average time per sample call (ms), samples per second
    """
    dist = distribution_class(**dist_params)

    # Warmup
    for _ in range(warmup):
        _ = dist.sample(sample_shape)

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        _ = dist.sample(sample_shape)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000

    total_samples = num_iterations * np.prod(sample_shape)
    samples_per_sec = total_samples / total_time

    return avg_time_ms, samples_per_sec


def benchmark_log_prob(distribution_class, dist_params, value_shape, num_iterations=100, warmup=10,
                       framework='mlx'):
    """
    Benchmark log_prob computation.

    Args:
        distribution_class: Distribution class to benchmark
        dist_params: Parameters to pass to distribution constructor
        value_shape: Shape of values to compute log_prob for
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations
        framework: 'mlx' or 'torch'

    Returns:
        Average time per log_prob call (ms), log_probs per second
    """
    dist = distribution_class(**dist_params)

    # Generate random values
    if framework == 'mlx':
        values = mlx_compat.rand(*value_shape)
    else:
        values = torch.rand(*value_shape)

    # Warmup
    for _ in range(warmup):
        _ = dist.log_prob(values)

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        _ = dist.log_prob(values)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000

    total_values = num_iterations * np.prod(value_shape)
    values_per_sec = total_values / total_time

    return avg_time_ms, values_per_sec


def benchmark_entropy(distribution_class, dist_params, num_iterations=1000, warmup=100):
    """
    Benchmark entropy computation.

    Returns:
        Average time per entropy call (ms)
    """
    dist = distribution_class(**dist_params)

    # Warmup
    for _ in range(warmup):
        _ = dist.entropy()

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        _ = dist.entropy()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000

    return avg_time_ms


def validate_accuracy(mlx_dist_cls, torch_dist_cls, params, mlx_params=None, torch_params=None,
                       sample_size=10000, rtol=1e-4, atol=1e-5):
    """
    Validate numerical accuracy of MLX distribution against PyTorch.

    Args:
        mlx_dist_cls: MLX distribution class
        torch_dist_cls: PyTorch distribution class
        params: Common parameters (used if mlx_params/torch_params not specified)
        mlx_params: MLX-specific parameters (optional)
        torch_params: PyTorch-specific parameters (optional)
        sample_size: Number of samples for statistical validation
        rtol: Relative tolerance for log_prob comparison
        atol: Absolute tolerance for log_prob comparison

    Returns:
        dict with accuracy metrics
    """
    if not MLX_AVAILABLE or not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "Missing framework"}

    mlx_p = mlx_params if mlx_params else params
    torch_p = torch_params if torch_params else params

    try:
        mlx_d = mlx_dist_cls(**mlx_p)
        torch_d = torch_dist_cls(**torch_p)

        # Compare log_prob on fixed values
        # Use values from the torch distribution's domain
        torch_samples = torch_d.sample((sample_size,))

        # Convert to MLX
        if hasattr(torch_samples, 'numpy'):
            sample_np = torch_samples.numpy()
        else:
            sample_np = torch_samples.detach().cpu().numpy()

        mlx_samples = mlx_compat.tensor(sample_np)

        # Compute log_prob
        mlx_logp = mlx_d.log_prob(mlx_samples)
        torch_logp = torch_d.log_prob(torch_samples)

        # Convert to numpy for comparison
        mlx_logp_np = mlx_logp.numpy() if hasattr(mlx_logp, 'numpy') else np.array(mlx_logp._mlx_array)
        torch_logp_np = torch_logp.numpy() if hasattr(torch_logp, 'numpy') else torch_logp.detach().cpu().numpy()

        # Filter out inf/nan for comparison
        valid_mask = np.isfinite(mlx_logp_np) & np.isfinite(torch_logp_np)
        if not np.any(valid_mask):
            return {"status": "error", "reason": "No valid log_prob values"}

        mlx_valid = mlx_logp_np[valid_mask]
        torch_valid = torch_logp_np[valid_mask]

        # Compute accuracy metrics
        abs_diff = np.abs(mlx_valid - torch_valid)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        rel_diff = abs_diff / (np.abs(torch_valid) + 1e-10)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # Check if within tolerance
        within_tol = np.allclose(mlx_valid, torch_valid, rtol=rtol, atol=atol)

        return {
            "status": "pass" if within_tol else "fail",
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": mean_rel_diff,
            "num_samples": int(np.sum(valid_mask)),
            "rtol": rtol,
            "atol": atol
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def print_accuracy_result(name, result):
    """Print accuracy validation result."""
    if result["status"] == "skipped":
        print(f"  {name:25s} SKIPPED: {result.get('reason', 'Unknown')}")
    elif result["status"] == "error":
        print(f"  {name:25s} ERROR: {result.get('reason', 'Unknown')}")
    elif result["status"] == "pass":
        print(f"  {name:25s} PASS (max_diff: {result['max_abs_diff']:.2e}, "
              f"mean_diff: {result['mean_abs_diff']:.2e})")
    else:
        print(f"  {name:25s} FAIL (max_diff: {result['max_abs_diff']:.2e}, "
              f"mean_diff: {result['mean_abs_diff']:.2e}, "
              f"rtol={result['rtol']}, atol={result['atol']})")


def format_speedup(mlx_val, torch_val):
    """Format speedup ratio."""
    if torch_val == 0:
        return "N/A"
    ratio = mlx_val / torch_val
    if ratio < 1:
        return f"{1/ratio:.2f}x faster"
    else:
        return f"{ratio:.2f}x slower"


def print_comparison(name, mlx_time, torch_time, unit="ms"):
    """Print formatted comparison."""
    speedup = format_speedup(mlx_time, torch_time)
    print(f"  {name:25s} MLX: {mlx_time:8.3f}{unit}  PyTorch: {torch_time:8.3f}{unit}  ({speedup})")


def benchmark_normal():
    """Benchmark Normal distribution."""
    print("\n" + "=" * 80)
    print("Normal Distribution Benchmark")
    print("=" * 80)

    loc, scale = 0.0, 1.0
    sample_shape = (10000,)
    value_shape = (10000,)

    if MLX_AVAILABLE:
        mlx_sample_time, mlx_sample_rate = benchmark_sampling(
            mlx_dist.Normal, {'loc': loc, 'scale': scale}, sample_shape
        )
        mlx_logp_time, _ = benchmark_log_prob(
            mlx_dist.Normal, {'loc': loc, 'scale': scale}, value_shape, framework='mlx'
        )
        mlx_entropy_time = benchmark_entropy(
            mlx_dist.Normal, {'loc': loc, 'scale': scale}
        )
    else:
        mlx_sample_time = mlx_logp_time = mlx_entropy_time = float('nan')
        mlx_sample_rate = 0

    if TORCH_AVAILABLE:
        torch_sample_time, torch_sample_rate = benchmark_sampling(
            torch_dist.Normal, {'loc': loc, 'scale': scale}, sample_shape
        )
        torch_logp_time, _ = benchmark_log_prob(
            torch_dist.Normal, {'loc': loc, 'scale': scale}, value_shape, framework='torch'
        )
        torch_entropy_time = benchmark_entropy(
            torch_dist.Normal, {'loc': loc, 'scale': scale}
        )
    else:
        torch_sample_time = torch_logp_time = torch_entropy_time = float('nan')
        torch_sample_rate = 0

    print(f"\nSample shape: {sample_shape}")
    print_comparison("sample()", mlx_sample_time, torch_sample_time)
    print_comparison("log_prob()", mlx_logp_time, torch_logp_time)
    print_comparison("entropy()", mlx_entropy_time, torch_entropy_time)
    print(f"\n  Throughput: MLX {mlx_sample_rate/1e6:.2f}M samples/s, PyTorch {torch_sample_rate/1e6:.2f}M samples/s")


def benchmark_gamma():
    """Benchmark Gamma distribution."""
    print("\n" + "=" * 80)
    print("Gamma Distribution Benchmark")
    print("=" * 80)

    concentration, rate = 2.0, 1.0
    sample_shape = (10000,)
    value_shape = (10000,)

    if MLX_AVAILABLE:
        mlx_sample_time, mlx_sample_rate = benchmark_sampling(
            mlx_dist.Gamma, {'concentration': concentration, 'rate': rate}, sample_shape
        )

        # For log_prob, need positive values
        dist = mlx_dist.Gamma(concentration, rate)
        values = dist.sample((10000,))

        def bench_mlx_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = dist.log_prob(values)
            return (time.perf_counter() - start) / 100 * 1000

        # Warmup
        for _ in range(10):
            _ = dist.log_prob(values)
        mlx_logp_time = bench_mlx_logp()

        mlx_entropy_time = benchmark_entropy(
            mlx_dist.Gamma, {'concentration': concentration, 'rate': rate}
        )
    else:
        mlx_sample_time = mlx_logp_time = mlx_entropy_time = float('nan')
        mlx_sample_rate = 0

    if TORCH_AVAILABLE:
        torch_sample_time, torch_sample_rate = benchmark_sampling(
            torch_dist.Gamma, {'concentration': concentration, 'rate': rate}, sample_shape
        )

        # For log_prob, need positive values
        tdist = torch_dist.Gamma(concentration, rate)
        tvalues = tdist.sample((10000,))

        def bench_torch_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = tdist.log_prob(tvalues)
            return (time.perf_counter() - start) / 100 * 1000

        # Warmup
        for _ in range(10):
            _ = tdist.log_prob(tvalues)
        torch_logp_time = bench_torch_logp()

        torch_entropy_time = benchmark_entropy(
            torch_dist.Gamma, {'concentration': concentration, 'rate': rate}
        )
    else:
        torch_sample_time = torch_logp_time = torch_entropy_time = float('nan')
        torch_sample_rate = 0

    print(f"\nSample shape: {sample_shape}")
    print_comparison("sample()", mlx_sample_time, torch_sample_time)
    print_comparison("log_prob()", mlx_logp_time, torch_logp_time)
    print_comparison("entropy()", mlx_entropy_time, torch_entropy_time)
    print(f"\n  Throughput: MLX {mlx_sample_rate/1e6:.2f}M samples/s, PyTorch {torch_sample_rate/1e6:.2f}M samples/s")


def benchmark_beta():
    """Benchmark Beta distribution."""
    print("\n" + "=" * 80)
    print("Beta Distribution Benchmark")
    print("=" * 80)

    concentration1, concentration0 = 2.0, 5.0
    sample_shape = (10000,)
    value_shape = (10000,)

    if MLX_AVAILABLE:
        mlx_sample_time, mlx_sample_rate = benchmark_sampling(
            mlx_dist.Beta, {'concentration1': concentration1, 'concentration0': concentration0},
            sample_shape
        )

        # Beta values in (0, 1)
        values = mlx_compat.rand(*value_shape) * 0.98 + 0.01
        dist = mlx_dist.Beta(concentration1, concentration0)

        def bench_mlx_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = dist.log_prob(values)
            return (time.perf_counter() - start) / 100 * 1000

        for _ in range(10):
            _ = dist.log_prob(values)
        mlx_logp_time = bench_mlx_logp()

        mlx_entropy_time = benchmark_entropy(
            mlx_dist.Beta, {'concentration1': concentration1, 'concentration0': concentration0}
        )
    else:
        mlx_sample_time = mlx_logp_time = mlx_entropy_time = float('nan')
        mlx_sample_rate = 0

    if TORCH_AVAILABLE:
        torch_sample_time, torch_sample_rate = benchmark_sampling(
            torch_dist.Beta, {'concentration1': concentration1, 'concentration0': concentration0},
            sample_shape
        )

        # Beta values in (0, 1)
        tvalues = torch.rand(*value_shape) * 0.98 + 0.01
        tdist = torch_dist.Beta(concentration1, concentration0)

        def bench_torch_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = tdist.log_prob(tvalues)
            return (time.perf_counter() - start) / 100 * 1000

        for _ in range(10):
            _ = tdist.log_prob(tvalues)
        torch_logp_time = bench_torch_logp()

        torch_entropy_time = benchmark_entropy(
            torch_dist.Beta, {'concentration1': concentration1, 'concentration0': concentration0}
        )
    else:
        torch_sample_time = torch_logp_time = torch_entropy_time = float('nan')
        torch_sample_rate = 0

    print(f"\nSample shape: {sample_shape}")
    print_comparison("sample()", mlx_sample_time, torch_sample_time)
    print_comparison("log_prob()", mlx_logp_time, torch_logp_time)
    print_comparison("entropy()", mlx_entropy_time, torch_entropy_time)
    print(f"\n  Throughput: MLX {mlx_sample_rate/1e6:.2f}M samples/s, PyTorch {torch_sample_rate/1e6:.2f}M samples/s")


def benchmark_dirichlet():
    """Benchmark Dirichlet distribution."""
    print("\n" + "=" * 80)
    print("Dirichlet Distribution Benchmark")
    print("=" * 80)

    concentration = [2.0, 3.0, 5.0]
    sample_shape = (10000,)

    if MLX_AVAILABLE:
        mlx_params = {'concentration': mlx_compat.tensor(concentration)}
        mlx_sample_time, mlx_sample_rate = benchmark_sampling(
            mlx_dist.Dirichlet, mlx_params, sample_shape
        )

        dist = mlx_dist.Dirichlet(mlx_compat.tensor(concentration))
        values = dist.sample((10000,))

        def bench_mlx_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = dist.log_prob(values)
            return (time.perf_counter() - start) / 100 * 1000

        for _ in range(10):
            _ = dist.log_prob(values)
        mlx_logp_time = bench_mlx_logp()
    else:
        mlx_sample_time = mlx_logp_time = float('nan')
        mlx_sample_rate = 0

    if TORCH_AVAILABLE:
        torch_params = {'concentration': torch.tensor(concentration)}
        torch_sample_time, torch_sample_rate = benchmark_sampling(
            torch_dist.Dirichlet, torch_params, sample_shape
        )

        tdist = torch_dist.Dirichlet(torch.tensor(concentration))
        tvalues = tdist.sample((10000,))

        def bench_torch_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = tdist.log_prob(tvalues)
            return (time.perf_counter() - start) / 100 * 1000

        for _ in range(10):
            _ = tdist.log_prob(tvalues)
        torch_logp_time = bench_torch_logp()
    else:
        torch_sample_time = torch_logp_time = float('nan')
        torch_sample_rate = 0

    print(f"\nSample shape: {sample_shape}")
    print_comparison("sample()", mlx_sample_time, torch_sample_time)
    print_comparison("log_prob()", mlx_logp_time, torch_logp_time)
    print(f"\n  Throughput: MLX {mlx_sample_rate/1e6:.2f}M samples/s, PyTorch {torch_sample_rate/1e6:.2f}M samples/s")


def benchmark_poisson():
    """Benchmark Poisson distribution."""
    print("\n" + "=" * 80)
    print("Poisson Distribution Benchmark")
    print("=" * 80)

    rate = 5.0
    sample_shape = (10000,)

    if MLX_AVAILABLE:
        mlx_sample_time, mlx_sample_rate = benchmark_sampling(
            mlx_dist.Poisson, {'rate': rate}, sample_shape
        )

        dist = mlx_dist.Poisson(rate)
        values = dist.sample((10000,))

        def bench_mlx_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = dist.log_prob(values)
            return (time.perf_counter() - start) / 100 * 1000

        for _ in range(10):
            _ = dist.log_prob(values)
        mlx_logp_time = bench_mlx_logp()
    else:
        mlx_sample_time = mlx_logp_time = float('nan')
        mlx_sample_rate = 0

    if TORCH_AVAILABLE:
        torch_sample_time, torch_sample_rate = benchmark_sampling(
            torch_dist.Poisson, {'rate': rate}, sample_shape
        )

        tdist = torch_dist.Poisson(rate)
        tvalues = tdist.sample((10000,))

        def bench_torch_logp():
            start = time.perf_counter()
            for _ in range(100):
                _ = tdist.log_prob(tvalues)
            return (time.perf_counter() - start) / 100 * 1000

        for _ in range(10):
            _ = tdist.log_prob(tvalues)
        torch_logp_time = bench_torch_logp()
    else:
        torch_sample_time = torch_logp_time = float('nan')
        torch_sample_rate = 0

    print(f"\nSample shape: {sample_shape}")
    print_comparison("sample()", mlx_sample_time, torch_sample_time)
    print_comparison("log_prob()", mlx_logp_time, torch_logp_time)
    print(f"\n  Throughput: MLX {mlx_sample_rate/1e6:.2f}M samples/s, PyTorch {torch_sample_rate/1e6:.2f}M samples/s")


def benchmark_large_batch():
    """Benchmark with large batch sizes."""
    print("\n" + "=" * 80)
    print("Large Batch Benchmark (100K samples)")
    print("=" * 80)

    sample_shape = (100000,)

    distributions = [
        ("Normal", mlx_dist.Normal if MLX_AVAILABLE else None,
         torch_dist.Normal if TORCH_AVAILABLE else None,
         {'loc': 0.0, 'scale': 1.0}),
        ("Gamma", mlx_dist.Gamma if MLX_AVAILABLE else None,
         torch_dist.Gamma if TORCH_AVAILABLE else None,
         {'concentration': 2.0, 'rate': 1.0}),
        ("Beta", mlx_dist.Beta if MLX_AVAILABLE else None,
         torch_dist.Beta if TORCH_AVAILABLE else None,
         {'concentration1': 2.0, 'concentration0': 5.0}),
    ]

    for name, mlx_cls, torch_cls, params in distributions:
        print(f"\n{name}:")

        if mlx_cls:
            mlx_time, mlx_rate = benchmark_sampling(mlx_cls, params, sample_shape, num_iterations=50)
        else:
            mlx_time, mlx_rate = float('nan'), 0

        if torch_cls:
            torch_time, torch_rate = benchmark_sampling(torch_cls, params, sample_shape, num_iterations=50)
        else:
            torch_time, torch_rate = float('nan'), 0

        print_comparison("sample(100K)", mlx_time, torch_time)
        print(f"    Throughput: MLX {mlx_rate/1e6:.2f}M/s, PyTorch {torch_rate/1e6:.2f}M/s")


def benchmark_accuracy():
    """Validate numerical accuracy of distributions against PyTorch."""
    print("\n" + "=" * 80)
    print("Accuracy Validation (log_prob comparison)")
    print("=" * 80)

    if not MLX_AVAILABLE or not TORCH_AVAILABLE:
        print("\nSkipping accuracy validation: both MLX and PyTorch required")
        return

    print("\nValidating log_prob accuracy against PyTorch (rtol=1e-4, atol=1e-5):\n")

    # Normal distribution
    result = validate_accuracy(
        mlx_dist.Normal, torch_dist.Normal,
        {'loc': 0.0, 'scale': 1.0}
    )
    print_accuracy_result("Normal", result)

    # Gamma distribution
    result = validate_accuracy(
        mlx_dist.Gamma, torch_dist.Gamma,
        {'concentration': 2.0, 'rate': 1.0}
    )
    print_accuracy_result("Gamma", result)

    # Beta distribution
    result = validate_accuracy(
        mlx_dist.Beta, torch_dist.Beta,
        {'concentration1': 2.0, 'concentration0': 5.0}
    )
    print_accuracy_result("Beta", result)

    # Poisson distribution
    result = validate_accuracy(
        mlx_dist.Poisson, torch_dist.Poisson,
        {'rate': 5.0}
    )
    print_accuracy_result("Poisson", result)

    # Dirichlet distribution (needs tensor params)
    result = validate_accuracy(
        mlx_dist.Dirichlet, torch_dist.Dirichlet,
        params=None,
        mlx_params={'concentration': mlx_compat.tensor([2.0, 3.0, 5.0])},
        torch_params={'concentration': torch.tensor([2.0, 3.0, 5.0])}
    )
    print_accuracy_result("Dirichlet", result)


def print_summary():
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print("\nNotes:")
    print("  - All benchmarks run on Apple Silicon")
    print("  - MLX uses pure MLX implementations (no scipy/numpy)")
    print("  - PyTorch uses CPU backend (MPS for distributions not fully supported)")
    print("  - Times measured with time.perf_counter()")
    print("\nInterpretation:")
    print("  - 'X.XXx faster' means MLX is faster than PyTorch")
    print("  - 'X.XXx slower' means MLX is slower than PyTorch")
    print("\nPerformance considerations:")
    print("  - MLX Gamma sampling uses Marsaglia-Tsang method (custom implementation)")
    print("  - MLX special functions (lgamma, digamma) are pure MLX (Lanczos approx)")
    print("  - PyTorch uses optimized C++ implementations")


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("mlx_compat Distributions Benchmarking Suite")
    print("=" * 80)

    print(f"\nMLX Available: {MLX_AVAILABLE}")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch MPS Available: {MPS_AVAILABLE}")

    if not MLX_AVAILABLE and not TORCH_AVAILABLE:
        print("\nError: Neither MLX nor PyTorch available. Cannot run benchmarks.")
        return

    print("\nRunning benchmarks... This may take a minute.\n")

    try:
        benchmark_normal()
        benchmark_gamma()
        benchmark_beta()
        benchmark_dirichlet()
        benchmark_poisson()
        benchmark_large_batch()
        benchmark_accuracy()
        print_summary()
    except KeyboardInterrupt:
        print("\n\nBenchmarking interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmarking: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Benchmarking complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
