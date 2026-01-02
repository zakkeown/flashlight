"""
Test Project Scaffolding

Verifies that the project structure is set up correctly.
"""

import sys
import unittest

# Add parent directory to path for imports
sys.path.insert(0, "..")

import flashlight
from tests.common_utils import TestCase


class TestScaffolding(TestCase):
    """Test project scaffolding and basic imports."""

    def test_package_import(self):
        """Test that flashlight package imports successfully."""
        self.assertIsNotNone(flashlight)

    def test_version(self):
        """Test that package version is defined."""
        self.assertEqual(flashlight.__version__, "0.1.0")

    def test_submodules_exist(self):
        """Test that all submodules can be imported."""
        # These should all import without errors (even if empty)
        import flashlight.autograd
        import flashlight.nn
        import flashlight.ops
        import flashlight.optim
        import flashlight.utils

        self.assertIsNotNone(flashlight.ops)
        self.assertIsNotNone(flashlight.autograd)
        self.assertIsNotNone(flashlight.nn)
        self.assertIsNotNone(flashlight.optim)
        self.assertIsNotNone(flashlight.utils)


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


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
