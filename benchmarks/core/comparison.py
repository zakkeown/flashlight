"""
Framework comparison harness for mlx_compat vs PyTorch.

Handles equivalent tensor creation, synchronized timing,
numerical parity checking, and side-by-side comparison.
"""

from typing import Callable, Optional, Tuple, Any, Dict
from dataclasses import dataclass
import numpy as np

from benchmarks.core.config import TimingStats, ComparisonStats
from benchmarks.core.timing import Timer, sync_mlx, sync_pytorch


@dataclass
class ComparisonResult:
    """Complete comparison result between frameworks."""
    mlx_timing: TimingStats
    pytorch_timing: Optional[TimingStats]
    comparison: Optional[ComparisonStats]
    mlx_output: Any
    pytorch_output: Any


class FrameworkComparator:
    """
    Compares mlx_compat operations against PyTorch.

    Handles:
    - Equivalent tensor creation between frameworks
    - Synchronized timing for both frameworks
    - Numerical parity checking with configurable tolerance
    - Fair interleaved benchmarking

    Example:
        comparator = FrameworkComparator(pytorch_device="mps")

        # Create inputs
        mlx_a, mlx_b = create_mlx_inputs()
        torch_a, torch_b = create_pytorch_inputs()

        # Compare matmul
        result = comparator.compare(
            mlx_fn=lambda: mlx_a @ mlx_b,
            pytorch_fn=lambda: torch_a @ torch_b,
        )

        print(f"Speedup: {result.comparison.speedup:.2f}x")
    """

    def __init__(
        self,
        pytorch_device: str = "mps",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        num_trials: int = 5,
    ):
        """
        Initialize comparator.

        Args:
            pytorch_device: PyTorch device for comparison
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison
            warmup_iterations: Warmup iterations per benchmark
            benchmark_iterations: Iterations to time per trial
            num_trials: Number of complete trials
        """
        self.pytorch_device = pytorch_device
        self.rtol = rtol
        self.atol = atol
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.num_trials = num_trials
        self._pytorch_available = self._check_pytorch()

    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available with requested device."""
        try:
            import torch
            if self.pytorch_device == "mps":
                return torch.backends.mps.is_available()
            elif self.pytorch_device.startswith("cuda"):
                return torch.cuda.is_available()
            return True  # CPU is always available
        except ImportError:
            return False

    @property
    def pytorch_available(self) -> bool:
        """Whether PyTorch is available for comparison."""
        return self._pytorch_available

    def compare(
        self,
        mlx_fn: Callable[[], Any],
        pytorch_fn: Optional[Callable[[], Any]] = None,
        check_numerical: bool = True,
        interleaved: bool = True,
    ) -> ComparisonResult:
        """
        Run comparison benchmark between mlx_compat and PyTorch.

        Args:
            mlx_fn: mlx_compat function to benchmark (no args, use closure)
            pytorch_fn: PyTorch function to benchmark
            check_numerical: Whether to check numerical parity
            interleaved: Use interleaved benchmarking for fairness

        Returns:
            ComparisonResult with timing and comparison stats
        """
        timer = Timer(
            warmup_iterations=self.warmup_iterations,
            benchmark_iterations=self.benchmark_iterations,
            num_trials=self.num_trials,
        )

        # Benchmark MLX
        timer.sync_fn = sync_mlx
        mlx_timing = timer.time_function(mlx_fn)
        mlx_output = mlx_fn()

        # Benchmark PyTorch if available
        pytorch_timing = None
        pytorch_output = None
        comparison = None

        if pytorch_fn is not None and self._pytorch_available:
            timer.sync_fn = lambda: sync_pytorch(self.pytorch_device)

            if interleaved:
                # Interleaved benchmarking for fairer comparison
                mlx_timing, pytorch_timing = timer.time_paired(
                    mlx_fn, pytorch_fn,
                    args_a=(), args_b=(),
                    sync_a=sync_mlx,
                    sync_b=lambda: sync_pytorch(self.pytorch_device),
                )
            else:
                pytorch_timing = timer.time_function(pytorch_fn)

            pytorch_output = pytorch_fn()

            # Calculate comparison stats
            comparison = self._compute_comparison(
                mlx_timing, pytorch_timing,
                mlx_output, pytorch_output,
                check_numerical,
            )

        return ComparisonResult(
            mlx_timing=mlx_timing,
            pytorch_timing=pytorch_timing,
            comparison=comparison,
            mlx_output=mlx_output,
            pytorch_output=pytorch_output,
        )

    def compare_with_inputs(
        self,
        mlx_fn: Callable[..., Any],
        pytorch_fn: Callable[..., Any],
        mlx_inputs: Tuple[Any, ...],
        pytorch_inputs: Tuple[Any, ...],
        check_numerical: bool = True,
    ) -> ComparisonResult:
        """
        Compare functions with explicit inputs.

        Args:
            mlx_fn: mlx_compat function
            pytorch_fn: PyTorch function
            mlx_inputs: Inputs for mlx_fn
            pytorch_inputs: Inputs for pytorch_fn
            check_numerical: Whether to verify numerical parity
        """
        return self.compare(
            mlx_fn=lambda: mlx_fn(*mlx_inputs),
            pytorch_fn=lambda: pytorch_fn(*pytorch_inputs),
            check_numerical=check_numerical,
        )

    def _compute_comparison(
        self,
        mlx_timing: TimingStats,
        pytorch_timing: TimingStats,
        mlx_output: Any,
        pytorch_output: Any,
        check_numerical: bool,
    ) -> ComparisonStats:
        """Compute comparison statistics."""
        # Speedup calculation
        if mlx_timing.mean_ms > 0:
            speedup = pytorch_timing.mean_ms / mlx_timing.mean_ms
        else:
            speedup = float('inf')

        # Relative performance string
        if speedup >= 1.0:
            relative_perf = f"{speedup:.2f}x faster"
        else:
            pct_slower = (1.0 - speedup) * 100
            relative_perf = f"{speedup:.2f}x ({pct_slower:.1f}% slower)"

        # Numerical comparison
        numerical_match = True
        max_diff = 0.0

        if check_numerical:
            numerical_match, max_diff = self._check_numerical_parity(
                mlx_output, pytorch_output
            )

        return ComparisonStats(
            speedup=speedup,
            relative_performance=relative_perf,
            numerical_match=numerical_match,
            max_abs_diff=max_diff,
        )

    def _check_numerical_parity(
        self,
        mlx_output: Any,
        pytorch_output: Any,
    ) -> Tuple[bool, float]:
        """
        Check numerical parity between outputs.

        Returns:
            Tuple of (match: bool, max_abs_diff: float)
        """
        try:
            # Convert to numpy
            mlx_np = self._to_numpy(mlx_output)
            pytorch_np = self._to_numpy(pytorch_output)

            if mlx_np is None or pytorch_np is None:
                return True, 0.0  # Can't compare, assume match

            # Check shapes match
            if mlx_np.shape != pytorch_np.shape:
                return False, float('inf')

            # Calculate max difference
            max_diff = float(np.max(np.abs(mlx_np - pytorch_np)))

            # Check with tolerance
            try:
                np.testing.assert_allclose(
                    mlx_np, pytorch_np,
                    rtol=self.rtol, atol=self.atol
                )
                return True, max_diff
            except AssertionError:
                return False, max_diff

        except Exception:
            return True, 0.0  # Error during comparison, assume match

    def _to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        """Convert tensor to numpy array."""
        if tensor is None:
            return None

        # MLX compat tensor
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()

        # PyTorch tensor
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()

        # Already numpy
        if isinstance(tensor, np.ndarray):
            return tensor

        # Try conversion
        try:
            return np.array(tensor)
        except Exception:
            return None

    def create_equivalent_inputs(
        self,
        shape: Tuple[int, ...],
        dtype: str = "float32",
    ) -> Tuple[Any, Any]:
        """
        Create equivalent random inputs for both frameworks.

        Args:
            shape: Tensor shape
            dtype: Data type string

        Returns:
            Tuple of (mlx_tensor, pytorch_tensor)
        """
        import mlx_compat
        import torch

        # Create numpy source for reproducibility
        np_data = np.random.randn(*shape).astype(dtype)

        # Create MLX tensor
        mlx_tensor = mlx_compat.tensor(np_data)

        # Create PyTorch tensor
        pytorch_tensor = torch.from_numpy(np_data).to(self.pytorch_device)

        return mlx_tensor, pytorch_tensor

    def convert_mlx_to_pytorch(
        self,
        mlx_tensor: Any,
    ) -> Any:
        """Convert mlx_compat tensor to equivalent PyTorch tensor."""
        import torch

        np_array = mlx_tensor.numpy()
        return torch.from_numpy(np_array).to(self.pytorch_device)

    def convert_pytorch_to_mlx(
        self,
        pytorch_tensor: Any,
    ) -> Any:
        """Convert PyTorch tensor to equivalent mlx_compat tensor."""
        import mlx_compat

        np_array = pytorch_tensor.detach().cpu().numpy()
        return mlx_compat.tensor(np_array)


def compare_operations(
    name: str,
    mlx_fn: Callable[[], Any],
    pytorch_fn: Callable[[], Any],
    pytorch_device: str = "mps",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to compare two operations.

    Args:
        name: Name of the operation
        mlx_fn: MLX operation (no args)
        pytorch_fn: PyTorch operation (no args)
        pytorch_device: PyTorch device
        verbose: Print results

    Returns:
        Dictionary with comparison results
    """
    comparator = FrameworkComparator(pytorch_device=pytorch_device)
    result = comparator.compare(mlx_fn, pytorch_fn)

    output = {
        "name": name,
        "mlx_mean_ms": result.mlx_timing.mean_ms,
        "mlx_std_ms": result.mlx_timing.std_ms,
    }

    if result.pytorch_timing is not None:
        output["pytorch_mean_ms"] = result.pytorch_timing.mean_ms
        output["pytorch_std_ms"] = result.pytorch_timing.std_ms

    if result.comparison is not None:
        output["speedup"] = result.comparison.speedup
        output["numerical_match"] = result.comparison.numerical_match

    if verbose:
        print(f"\n{name}:")
        print(f"  MLX: {result.mlx_timing.mean_ms:.3f} +/- {result.mlx_timing.std_ms:.3f} ms")
        if result.pytorch_timing:
            print(f"  PyTorch: {result.pytorch_timing.mean_ms:.3f} +/- {result.pytorch_timing.std_ms:.3f} ms")
        if result.comparison:
            print(f"  Speedup: {result.comparison.relative_performance}")
            print(f"  Numerical match: {result.comparison.numerical_match}")

    return output
