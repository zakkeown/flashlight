"""
Test Phase 4: Pooling Layers

Tests the nn.layers.pooling module:
- MaxPool2d, AvgPool2d (the core 2D pooling layers)
- AdaptiveAvgPool2d with output_size=1 (global pooling)
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestMaxPool2d(TestCase):
    """Test nn.MaxPool2d."""

    def test_maxpool2d_creation(self):
        """Test MaxPool2d creation with default parameters."""
        pool = flashlight.nn.MaxPool2d(kernel_size=2)
        self.assertEqual(pool.kernel_size, (2, 2))
        self.assertEqual(pool.stride, (2, 2))
        self.assertEqual(pool.padding, (0, 0))

    def test_maxpool2d_forward_basic(self):
        """Test MaxPool2d forward pass."""
        pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2)
        x = flashlight.randn(4, 64, 32, 32)
        output = pool(x)
        self.assertEqual(output.shape, (4, 64, 16, 16))

    def test_maxpool2d_with_padding(self):
        """Test MaxPool2d with padding."""
        pool = flashlight.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = flashlight.randn(4, 64, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 64)

    def test_maxpool2d_stride_different(self):
        """Test MaxPool2d with stride different from kernel size."""
        pool = flashlight.nn.MaxPool2d(kernel_size=3, stride=1)
        x = flashlight.randn(2, 32, 10, 10)
        output = pool(x)
        self.assertEqual(output.shape, (2, 32, 8, 8))

    def test_maxpool2d_single_sample(self):
        """Test MaxPool2d with single sample."""
        pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2)
        x = flashlight.randn(1, 32, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape, (1, 32, 8, 8))


@skipIfNoMLX
class TestAvgPool2d(TestCase):
    """Test nn.AvgPool2d."""

    def test_avgpool2d_creation(self):
        """Test AvgPool2d creation."""
        pool = flashlight.nn.AvgPool2d(kernel_size=2)
        self.assertEqual(pool.kernel_size, (2, 2))
        self.assertTrue(pool.count_include_pad)

    def test_avgpool2d_forward(self):
        """Test AvgPool2d forward pass."""
        pool = flashlight.nn.AvgPool2d(kernel_size=2, stride=2)
        x = flashlight.randn(4, 64, 32, 32)
        output = pool(x)
        self.assertEqual(output.shape, (4, 64, 16, 16))

    def test_avgpool2d_with_padding(self):
        """Test AvgPool2d with padding."""
        pool = flashlight.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        x = flashlight.randn(4, 64, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 64)

    def test_avgpool2d_single_sample(self):
        """Test AvgPool2d with single sample."""
        pool = flashlight.nn.AvgPool2d(kernel_size=2, stride=2)
        x = flashlight.randn(1, 32, 16, 16)
        output = pool(x)
        self.assertEqual(output.shape, (1, 32, 8, 8))


@skipIfNoMLX
class TestAdaptiveAvgPool2d(TestCase):
    """Test nn.AdaptiveAvgPool2d."""

    def test_adaptive_avgpool2d_creation(self):
        """Test AdaptiveAvgPool2d creation."""
        pool = flashlight.nn.AdaptiveAvgPool2d(output_size=1)
        self.assertEqual(pool.output_size, (1, 1))

    def test_adaptive_avgpool2d_to_1x1(self):
        """Test AdaptiveAvgPool2d with output_size=1 (global average pool)."""
        pool = flashlight.nn.AdaptiveAvgPool2d(output_size=1)
        x = flashlight.randn(4, 64, 32, 32)
        output = pool(x)
        self.assertEqual(output.shape, (4, 64, 1, 1))

    def test_adaptive_avgpool2d_identity(self):
        """Test AdaptiveAvgPool2d with same output size as input."""
        pool = flashlight.nn.AdaptiveAvgPool2d(output_size=(8, 8))
        x = flashlight.randn(2, 32, 8, 8)
        output = pool(x)
        self.assertEqual(output.shape, (2, 32, 8, 8))


@skipIfNoMLX
class TestAdaptiveMaxPool2d(TestCase):
    """Test nn.AdaptiveMaxPool2d."""

    def test_adaptive_maxpool2d_creation(self):
        """Test AdaptiveMaxPool2d creation."""
        pool = flashlight.nn.AdaptiveMaxPool2d(output_size=1)
        self.assertEqual(pool.output_size, (1, 1))

    def test_adaptive_maxpool2d_to_1x1(self):
        """Test AdaptiveMaxPool2d with output_size=1 (global max pool)."""
        pool = flashlight.nn.AdaptiveMaxPool2d(output_size=1)
        x = flashlight.randn(4, 64, 32, 32)
        output = pool(x)
        self.assertEqual(output.shape, (4, 64, 1, 1))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
