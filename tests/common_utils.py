"""
Common testing utilities for flashlight tests.

Based on PyTorch's torch.testing._internal.common_utils module.
Provides utilities for numerical parity testing between MLX and PyTorch.
"""

import sys
import unittest
from typing import Union, Optional

import numpy as np

# Optional imports - may not be available in all environments
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Will be available after Phase 1
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class TestCase(unittest.TestCase):
    """
    Base test case class with MLX/PyTorch comparison utilities.

    Extends unittest.TestCase with custom assertion methods for
    comparing MLX tensors with PyTorch tensors.
    """

    def assert_tensors_close(
        self,
        actual,
        expected,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: Optional[str] = None,
    ):
        """
        Assert that two tensors are numerically close.

        Args:
            actual: MLX tensor or numpy array
            expected: PyTorch tensor or numpy array
            rtol: Relative tolerance
            atol: Absolute tolerance
            msg: Optional error message
        """
        # Convert to numpy for comparison
        if hasattr(actual, '_mlx_array'):
            # MLX Tensor wrapper (will exist after Phase 1)
            actual_np = np.array(actual._mlx_array)
        elif hasattr(actual, 'numpy'):
            # Already a numpy-convertible object
            actual_np = actual.numpy() if callable(actual.numpy) else np.array(actual)
        else:
            actual_np = np.array(actual)

        if TORCH_AVAILABLE and isinstance(expected, torch.Tensor):
            expected_np = expected.detach().cpu().numpy()
        else:
            expected_np = np.array(expected)

        # Check shapes match
        if actual_np.shape != expected_np.shape:
            raise AssertionError(
                f"Shape mismatch: actual {actual_np.shape} vs expected {expected_np.shape}"
            )

        # Check values are close
        try:
            np.testing.assert_allclose(
                actual_np,
                expected_np,
                rtol=rtol,
                atol=atol,
                err_msg=msg or "Tensors not close",
            )
        except AssertionError as e:
            # Add more detailed error information
            max_diff = np.max(np.abs(actual_np - expected_np))
            mean_diff = np.mean(np.abs(actual_np - expected_np))
            error_msg = (
                f"\n{str(e)}\n"
                f"Max absolute difference: {max_diff}\n"
                f"Mean absolute difference: {mean_diff}\n"
                f"Actual min/max: {actual_np.min()}/{actual_np.max()}\n"
                f"Expected min/max: {expected_np.min()}/{expected_np.max()}"
            )
            raise AssertionError(error_msg)

    def assert_dtype_equal(self, actual_dtype, expected_dtype, msg: Optional[str] = None):
        """
        Assert that two dtypes are equivalent.

        Handles mapping between MLX, PyTorch, and NumPy dtypes.
        """
        # Will implement dtype mapping in Phase 1
        # For now, simple equality check
        self.assertEqual(actual_dtype, expected_dtype, msg)

    def assert_shape_equal(self, actual_shape, expected_shape, msg: Optional[str] = None):
        """Assert that two shapes are equal."""
        self.assertEqual(tuple(actual_shape), tuple(expected_shape), msg)


class ToleranceTier:
    """
    Standard tolerance tiers for flashlight testing.

    These tiers provide consistent tolerance settings across the test suite.
    Use these instead of hardcoding tolerance values in individual tests.

    Tiers:
        STRICT: For simple element-wise operations (add, sub, mul, activations)
        STANDARD: For most operations (default for forward pass parity)
        RELAXED: For operations with accumulated errors (conv, matmul chains, attention)
        LOOSE: For complex approximations (special functions, distributions)
        IMPL_DIFF: For known implementation differences between MLX and PyTorch
        GRADIENT: For gradient parity testing

    Usage:
        np.testing.assert_allclose(actual, expected, **ToleranceTier.STANDARD)

        # Or use the helper function:
        assert_close(actual, expected, tier='STANDARD')
    """
    # Simple element-wise operations - highest precision expected
    STRICT = {"rtol": 1e-5, "atol": 1e-8}

    # Most operations - default for forward pass parity testing
    STANDARD = {"rtol": 1e-5, "atol": 1e-6}

    # Operations with accumulated floating-point errors
    # (convolutions, multi-head attention, RNNs, optimizer steps)
    RELAXED = {"rtol": 1e-4, "atol": 1e-5}

    # Complex approximations and special functions
    # (erfcx, gammaln, distributions with numerical edge cases)
    LOOSE = {"rtol": 1e-3, "atol": 1e-4}

    # For operations with known implementation differences between frameworks
    # (e.g., erf, erfinv where MLX and PyTorch use different algorithms)
    IMPL_DIFF = {"rtol": 1e-4, "atol": 1e-6}

    # Gradient parity testing (inherently less precise due to accumulation)
    GRADIENT = {"rtol": 1e-4, "atol": 1e-4}

    @classmethod
    def get(cls, name: str) -> dict:
        """Get tolerance tier by name."""
        tiers = {
            'STRICT': cls.STRICT,
            'STANDARD': cls.STANDARD,
            'RELAXED': cls.RELAXED,
            'LOOSE': cls.LOOSE,
            'IMPL_DIFF': cls.IMPL_DIFF,
            'GRADIENT': cls.GRADIENT,
        }
        if name not in tiers:
            raise ValueError(f"Unknown tolerance tier: {name}. Valid: {list(tiers.keys())}")
        return tiers[name]


def assert_close(
    actual,
    expected,
    tier: str = 'STANDARD',
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    msg: Optional[str] = None
):
    """
    Assert arrays are close with named tolerance tier.

    Args:
        actual: Actual values (numpy array, MLX tensor, or PyTorch tensor)
        expected: Expected values
        tier: Tolerance tier name ('STRICT', 'STANDARD', 'RELAXED', 'LOOSE', 'IMPL_DIFF', 'GRADIENT')
        rtol: Override relative tolerance (optional)
        atol: Override absolute tolerance (optional)
        msg: Optional message on failure

    Example:
        assert_close(mlx_result, torch_result, tier='RELAXED')
        assert_close(gradient, expected_grad, tier='GRADIENT')
    """
    tol = ToleranceTier.get(tier)
    final_rtol = rtol if rtol is not None else tol['rtol']
    final_atol = atol if atol is not None else tol['atol']

    # Convert to numpy arrays
    if hasattr(actual, '_mlx_array'):
        actual_np = np.array(actual._mlx_array)
    elif hasattr(actual, 'numpy'):
        actual_np = actual.numpy() if callable(actual.numpy) else np.array(actual)
    elif hasattr(actual, 'detach'):
        actual_np = actual.detach().cpu().numpy()
    else:
        actual_np = np.array(actual)

    if hasattr(expected, '_mlx_array'):
        expected_np = np.array(expected._mlx_array)
    elif hasattr(expected, 'numpy'):
        expected_np = expected.numpy() if callable(expected.numpy) else np.array(expected)
    elif hasattr(expected, 'detach'):
        expected_np = expected.detach().cpu().numpy()
    else:
        expected_np = np.array(expected)

    np.testing.assert_allclose(
        actual_np, expected_np, rtol=final_rtol, atol=final_atol, err_msg=msg
    )


def run_tests():
    """
    Run all tests in the current module.

    Compatible with PyTorch's testing convention:
        if __name__ == '__main__':
            run_tests()
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    # Get the calling module
    caller_module = sys._getframe(1).f_globals['__name__']
    suite = loader.loadTestsFromName(caller_module)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())


def make_tensor(
    *shape,
    dtype=None,
    device=None,
    requires_grad: bool = False,
    backend: str = 'torch',
):
    """
    Create a tensor for testing.

    Args:
        *shape: Tensor shape
        dtype: Data type
        device: Device ('cpu', 'cuda', 'mps')
        requires_grad: Whether to track gradients
        backend: 'torch' or 'mlx'

    Returns:
        PyTorch tensor or MLX tensor
    """
    if backend == 'torch':
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        return torch.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
    elif backend == 'mlx':
        # Will implement in Phase 1
        # import flashlight
        # return flashlight.randn(*shape, dtype=dtype, requires_grad=requires_grad)
        raise NotImplementedError("MLX backend not yet implemented (Phase 1)")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def gradcheck(
    func,
    inputs,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> bool:
    """
    Numerical gradient checking.

    Compares analytical gradients (from autograd) with numerical gradients
    (from finite differences).

    Args:
        func: Function to check (takes tensors, returns scalar)
        inputs: Input tensors (tuple or single tensor)
        eps: Finite difference epsilon
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if gradients match
    """
    # Will implement in Phase 3 (Autograd)
    raise NotImplementedError("Gradient checking not yet implemented (Phase 3)")


def set_default_dtype(dtype):
    """Set default floating point dtype for tensor creation."""
    # Will implement in Phase 1
    raise NotImplementedError("set_default_dtype not yet implemented (Phase 1)")


# Test markers and decorators
def skipIfNoMLX(func):
    """Skip test if MLX is not available."""
    if not MLX_AVAILABLE:
        return unittest.skip("MLX not available")(func)
    return func


def skipIfNoTorch(func):
    """Skip test if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        return unittest.skip("PyTorch not available")(func)
    return func


def onlyOnAppleSilicon(func):
    """Only run test on Apple Silicon (M1/M2/M3)."""
    import platform
    if platform.system() != 'Darwin' or platform.processor() != 'arm':
        return unittest.skip("Test only runs on Apple Silicon")(func)
    return func
