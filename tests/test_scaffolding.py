"""
Test Phase 0: Project Scaffolding

Verifies that the project structure is set up correctly.
"""

import unittest
import sys

# Add parent directory to path for imports
sys.path.insert(0, '..')

import mlx_compat
from tests.common_utils import TestCase


class TestScaffolding(TestCase):
    """Test project scaffolding and basic imports."""

    def test_package_import(self):
        """Test that mlx_compat package imports successfully."""
        self.assertIsNotNone(mlx_compat)

    def test_version(self):
        """Test that package version is defined."""
        self.assertEqual(mlx_compat.__version__, "0.1.0")

    def test_submodules_exist(self):
        """Test that all submodules can be imported."""
        # These should all import without errors (even if empty)
        import mlx_compat.ops
        import mlx_compat.autograd
        import mlx_compat.nn
        import mlx_compat.optim
        import mlx_compat.utils

        self.assertIsNotNone(mlx_compat.ops)
        self.assertIsNotNone(mlx_compat.autograd)
        self.assertIsNotNone(mlx_compat.nn)
        self.assertIsNotNone(mlx_compat.optim)
        self.assertIsNotNone(mlx_compat.utils)

    def test_implementation_status(self):
        """Test that implementation status can be retrieved."""
        status = mlx_compat._get_implementation_status()
        self.assertIsInstance(status, dict)
        self.assertEqual(status["phase_0_scaffolding"], "✓ Complete")
        self.assertEqual(status["phase_1_tensor_core"], "✓ Complete")


class TestCommonUtils(TestCase):
    """Test common testing utilities."""

    def test_assert_tensors_close_with_numpy(self):
        """Test assert_tensors_close with numpy arrays."""
        import numpy as np

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        # Should not raise
        self.assert_tensors_close(a, b)

    def test_assert_tensors_close_with_torch(self):
        """Test assert_tensors_close with PyTorch tensors."""
        from tests.common_utils import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        import numpy as np
        import torch

        a = np.array([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])

        # Should not raise
        self.assert_tensors_close(a, b)

    def test_assert_tensors_close_fails_on_mismatch(self):
        """Test that assert_tensors_close fails when values differ."""
        import numpy as np

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])  # Different value

        with self.assertRaises(AssertionError):
            self.assert_tensors_close(a, b, rtol=1e-10, atol=1e-10)

    def test_assert_shape_equal(self):
        """Test assert_shape_equal utility."""
        self.assert_shape_equal((3, 4, 5), (3, 4, 5))

        with self.assertRaises(AssertionError):
            self.assert_shape_equal((3, 4), (3, 5))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
