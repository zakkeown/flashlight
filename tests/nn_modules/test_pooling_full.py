"""
Test Phase 4: Full Pooling Layer Tests

Tests all pooling layers including:
- MaxPool1d/2d/3d, AvgPool1d/2d/3d
- AdaptiveMaxPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d
- LPPool1d/2d/3d
- FractionalMaxPool2d/3d
- MaxUnpool1d/2d/3d
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Max Pooling Tests
# ============================================================================


@skipIfNoMLX
class TestMaxPool1d(TestCase):
    """Test nn.MaxPool1d."""

    def test_creation(self):
        """Test MaxPool1d creation."""
        pool = flashlight.nn.MaxPool1d(kernel_size=3)
        self.assertEqual(pool.kernel_size, 3)

    def test_forward_shape(self):
        """Test MaxPool1d forward pass output shape."""
        pool = flashlight.nn.MaxPool1d(kernel_size=3, stride=2)
        x = flashlight.randn(2, 3, 20)
        output = pool(x)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 3)


@skipIfNoMLX
class TestMaxPool3d(TestCase):
    """Test nn.MaxPool3d."""

    def test_creation(self):
        """Test MaxPool3d creation."""
        pool = flashlight.nn.MaxPool3d(kernel_size=3)
        self.assertIsNotNone(pool)

    def test_forward_shape(self):
        """Test MaxPool3d forward pass output shape."""
        pool = flashlight.nn.MaxPool3d(kernel_size=2)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape[:2], (2, 3))


# ============================================================================
# Average Pooling Tests
# ============================================================================


@skipIfNoMLX
class TestAvgPool1d(TestCase):
    """Test nn.AvgPool1d."""

    def test_forward_shape(self):
        """Test AvgPool1d forward pass output shape."""
        pool = flashlight.nn.AvgPool1d(kernel_size=3, stride=2)
        x = flashlight.randn(2, 3, 20)
        output = pool(x)
        self.assertEqual(output.shape[:2], (2, 3))


@skipIfNoMLX
class TestAvgPool3d(TestCase):
    """Test nn.AvgPool3d."""

    def test_forward_shape(self):
        """Test AvgPool3d forward pass output shape."""
        pool = flashlight.nn.AvgPool3d(kernel_size=2)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape[:2], (2, 3))


# ============================================================================
# Adaptive Pooling Tests
# ============================================================================


@skipIfNoMLX
class TestAdaptiveMaxPool1d(TestCase):
    """Test nn.AdaptiveMaxPool1d."""

    def test_forward_shape(self):
        """Test AdaptiveMaxPool1d forward pass output shape."""
        pool = flashlight.nn.AdaptiveMaxPool1d(output_size=5)
        x = flashlight.randn(2, 3, 20)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 5))


@skipIfNoMLX
class TestAdaptiveMaxPool3d(TestCase):
    """Test nn.AdaptiveMaxPool3d."""

    def test_forward_shape(self):
        """Test AdaptiveMaxPool3d forward pass output shape."""
        pool = flashlight.nn.AdaptiveMaxPool3d(output_size=(2, 2, 2))
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 2, 2, 2))


@skipIfNoMLX
class TestAdaptiveAvgPool1d(TestCase):
    """Test nn.AdaptiveAvgPool1d."""

    def test_forward_shape(self):
        """Test AdaptiveAvgPool1d forward pass output shape."""
        pool = flashlight.nn.AdaptiveAvgPool1d(output_size=5)
        x = flashlight.randn(2, 3, 20)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 5))


@skipIfNoMLX
class TestAdaptiveAvgPool3d(TestCase):
    """Test nn.AdaptiveAvgPool3d."""

    def test_forward_shape(self):
        """Test AdaptiveAvgPool3d forward pass output shape."""
        pool = flashlight.nn.AdaptiveAvgPool3d(output_size=(2, 2, 2))
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 2, 2, 2))


# ============================================================================
# LP Pooling Tests (Converted from NumPy)
# ============================================================================


@skipIfNoMLX
class TestLPPool1d(TestCase):
    """Test nn.LPPool1d."""

    def test_creation(self):
        """Test LPPool1d creation."""
        pool = flashlight.nn.LPPool1d(norm_type=2, kernel_size=3)
        self.assertEqual(pool.norm_type, 2)

    def test_forward_shape(self):
        """Test LPPool1d forward pass output shape."""
        pool = flashlight.nn.LPPool1d(norm_type=2, kernel_size=3, stride=2)
        x = flashlight.randn(2, 3, 20)
        output = pool(x)
        self.assertEqual(output.shape[:2], (2, 3))


@skipIfNoMLX
class TestLPPool2d(TestCase):
    """Test nn.LPPool2d."""

    def test_creation(self):
        """Test LPPool2d creation."""
        pool = flashlight.nn.LPPool2d(norm_type=2, kernel_size=3)
        self.assertEqual(pool.norm_type, 2)

    def test_forward_shape(self):
        """Test LPPool2d forward pass output shape."""
        pool = flashlight.nn.LPPool2d(norm_type=2, kernel_size=3, stride=2)
        x = flashlight.randn(2, 3, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape[:2], (2, 3))


@skipIfNoMLX
class TestLPPool3d(TestCase):
    """Test nn.LPPool3d (converted from NumPy to MLX)."""

    def test_creation(self):
        """Test LPPool3d creation."""
        pool = flashlight.nn.LPPool3d(norm_type=2, kernel_size=2)
        self.assertEqual(pool.norm_type, 2)

    def test_forward_shape(self):
        """Test LPPool3d forward pass output shape."""
        pool = flashlight.nn.LPPool3d(norm_type=2, kernel_size=2, stride=2)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape[:2], (2, 3))
        self.assertEqual(output.shape[2:], (4, 4, 4))

    def test_different_norm_types(self):
        """Test LPPool3d with different norm types."""
        for norm_type in [1, 2, 3]:
            pool = flashlight.nn.LPPool3d(norm_type=norm_type, kernel_size=2)
            x = flashlight.randn(1, 2, 4, 4, 4)
            output = pool(x)
            self.assertEqual(output.shape, (1, 2, 2, 2, 2))


# ============================================================================
# Fractional Max Pooling Tests (Converted from NumPy)
# ============================================================================


@skipIfNoMLX
class TestFractionalMaxPool2d(TestCase):
    """Test nn.FractionalMaxPool2d (converted from NumPy to MLX)."""

    def test_creation_with_output_size(self):
        """Test FractionalMaxPool2d creation with output_size."""
        pool = flashlight.nn.FractionalMaxPool2d(kernel_size=3, output_size=(5, 5))
        self.assertEqual(pool.output_size, (5, 5))

    def test_creation_with_output_ratio(self):
        """Test FractionalMaxPool2d creation with output_ratio."""
        pool = flashlight.nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(0.5, 0.5))
        self.assertEqual(pool.output_ratio, (0.5, 0.5))

    def test_forward_shape_with_output_size(self):
        """Test FractionalMaxPool2d forward pass with output_size."""
        pool = flashlight.nn.FractionalMaxPool2d(kernel_size=3, output_size=(5, 5))
        x = flashlight.randn(2, 3, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 5, 5))

    def test_forward_shape_with_output_ratio(self):
        """Test FractionalMaxPool2d forward pass with output_ratio."""
        pool = flashlight.nn.FractionalMaxPool2d(kernel_size=3, output_ratio=(0.5, 0.5))
        x = flashlight.randn(2, 3, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 8, 8))


@skipIfNoMLX
class TestFractionalMaxPool3d(TestCase):
    """Test nn.FractionalMaxPool3d (converted from NumPy to MLX)."""

    def test_creation(self):
        """Test FractionalMaxPool3d creation."""
        pool = flashlight.nn.FractionalMaxPool3d(kernel_size=3, output_size=(4, 4, 4))
        self.assertEqual(pool.output_size, (4, 4, 4))

    def test_forward_shape(self):
        """Test FractionalMaxPool3d forward pass."""
        pool = flashlight.nn.FractionalMaxPool3d(kernel_size=2, output_size=(4, 4, 4))
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape, (2, 3, 4, 4, 4))


# ============================================================================
# Max Unpool Tests (Converted from NumPy)
# ============================================================================


@skipIfNoMLX
class TestMaxUnpool1d(TestCase):
    """Test nn.MaxUnpool1d (converted from NumPy to MLX)."""

    def test_creation(self):
        """Test MaxUnpool1d creation."""
        unpool = flashlight.nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.assertEqual(unpool.kernel_size, 2)

    def test_forward_shape(self):
        """Test MaxUnpool1d forward pass."""
        # First do max pooling with return_indices
        pool = flashlight.nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        x = flashlight.randn(2, 3, 8)
        pooled, indices = pool(x)

        # Then unpool
        unpool = flashlight.nn.MaxUnpool1d(kernel_size=2, stride=2)
        output = unpool(pooled, indices)
        self.assertEqual(output.shape, (2, 3, 8))


@skipIfNoMLX
class TestMaxUnpool2d(TestCase):
    """Test nn.MaxUnpool2d (converted from NumPy to MLX)."""

    def test_creation(self):
        """Test MaxUnpool2d creation."""
        unpool = flashlight.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.assertEqual(unpool.kernel_size, (2, 2))

    def test_forward_shape(self):
        """Test MaxUnpool2d forward pass."""
        pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        x = flashlight.randn(2, 3, 8, 8)
        pooled, indices = pool(x)

        unpool = flashlight.nn.MaxUnpool2d(kernel_size=2, stride=2)
        output = unpool(pooled, indices)
        self.assertEqual(output.shape, (2, 3, 8, 8))


@skipIfNoMLX
class TestMaxUnpool3d(TestCase):
    """Test nn.MaxUnpool3d (converted from NumPy to MLX)."""

    def test_creation(self):
        """Test MaxUnpool3d creation."""
        unpool = flashlight.nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.assertEqual(unpool.kernel_size, (2, 2, 2))

    def test_forward_shape(self):
        """Test MaxUnpool3d forward pass."""
        pool = flashlight.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        x = flashlight.randn(2, 3, 8, 8, 8)
        pooled, indices = pool(x)

        unpool = flashlight.nn.MaxUnpool3d(kernel_size=2, stride=2)
        output = unpool(pooled, indices)
        self.assertEqual(output.shape, (2, 3, 8, 8, 8))


if __name__ == "__main__":
    unittest.main()
