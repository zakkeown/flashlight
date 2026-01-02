"""
Test Phase 4: Lazy Layers

Tests the lazy initialization modules:
- LazyLinear
- LazyConv1d, LazyConv2d, LazyConv3d
- LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d
- LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d
- LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d
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


@skipIfNoMLX
class TestLazyLinear(TestCase):
    """Test nn.LazyLinear."""

    def test_creation(self):
        """Test LazyLinear creation."""
        linear = flashlight.nn.LazyLinear(out_features=64)
        self.assertEqual(linear.out_features, 64)

    def test_forward_infers_in_features(self):
        """Test LazyLinear forward pass infers input features."""
        linear = flashlight.nn.LazyLinear(out_features=64)
        x = flashlight.randn(4, 32)  # 32 input features
        output = linear(x)
        self.assertEqual(output.shape, (4, 64))
        # After forward, in_features should be set
        self.assertEqual(linear.in_features, 32)

    def test_forward_different_batch_sizes(self):
        """Test LazyLinear with different batch sizes."""
        linear = flashlight.nn.LazyLinear(out_features=64)

        # First forward sets in_features
        x1 = flashlight.randn(4, 32)
        output1 = linear(x1)
        self.assertEqual(output1.shape, (4, 64))

        # Second forward with different batch size
        x2 = flashlight.randn(8, 32)
        output2 = linear(x2)
        self.assertEqual(output2.shape, (8, 64))

    def test_with_bias_false(self):
        """Test LazyLinear without bias."""
        linear = flashlight.nn.LazyLinear(out_features=64, bias=False)
        x = flashlight.randn(4, 32)
        output = linear(x)
        self.assertEqual(output.shape, (4, 64))


@skipIfNoMLX
class TestLazyConv1d(TestCase):
    """Test nn.LazyConv1d."""

    def test_creation(self):
        """Test LazyConv1d creation."""
        conv = flashlight.nn.LazyConv1d(out_channels=32, kernel_size=3)
        self.assertEqual(conv.out_channels, 32)

    def test_forward_infers_in_channels(self):
        """Test LazyConv1d forward pass infers input channels."""
        conv = flashlight.nn.LazyConv1d(out_channels=32, kernel_size=3, padding=1)
        x = flashlight.randn(4, 16, 100)  # batch, in_channels, length
        output = conv(x)
        self.assertEqual(output.shape, (4, 32, 100))
        self.assertEqual(conv.in_channels, 16)

    def test_with_stride(self):
        """Test LazyConv1d with stride."""
        conv = flashlight.nn.LazyConv1d(out_channels=32, kernel_size=3, stride=2)
        x = flashlight.randn(4, 16, 100)
        output = conv(x)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 32)


@skipIfNoMLX
class TestLazyConv2d(TestCase):
    """Test nn.LazyConv2d."""

    def test_creation(self):
        """Test LazyConv2d creation."""
        conv = flashlight.nn.LazyConv2d(out_channels=32, kernel_size=3)
        self.assertEqual(conv.out_channels, 32)

    def test_forward_infers_in_channels(self):
        """Test LazyConv2d forward pass infers input channels."""
        conv = flashlight.nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)
        x = flashlight.randn(4, 16, 8, 8)  # batch, in_channels, height, width
        output = conv(x)
        self.assertEqual(output.shape, (4, 32, 8, 8))
        self.assertEqual(conv.in_channels, 16)

    def test_with_stride_and_padding(self):
        """Test LazyConv2d with stride and padding."""
        conv = flashlight.nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1)
        x = flashlight.randn(4, 16, 32, 32)
        output = conv(x)
        self.assertEqual(output.shape, (4, 32, 16, 16))


@skipIfNoMLX
class TestLazyConv3d(TestCase):
    """Test nn.LazyConv3d."""

    def test_creation(self):
        """Test LazyConv3d creation."""
        conv = flashlight.nn.LazyConv3d(out_channels=32, kernel_size=3)
        self.assertEqual(conv.out_channels, 32)

    def test_forward_infers_in_channels(self):
        """Test LazyConv3d forward pass infers input channels."""
        conv = flashlight.nn.LazyConv3d(out_channels=32, kernel_size=3, padding=1)
        x = flashlight.randn(2, 8, 4, 4, 4)  # batch, in_channels, depth, height, width
        output = conv(x)
        self.assertEqual(output.shape, (2, 32, 4, 4, 4))
        self.assertEqual(conv.in_channels, 8)


@skipIfNoMLX
class TestLazyConvTranspose1d(TestCase):
    """Test nn.LazyConvTranspose1d."""

    def test_creation(self):
        """Test LazyConvTranspose1d creation."""
        conv = flashlight.nn.LazyConvTranspose1d(out_channels=32, kernel_size=3)
        self.assertEqual(conv.out_channels, 32)

    def test_forward_infers_in_channels(self):
        """Test LazyConvTranspose1d forward pass infers input channels."""
        conv = flashlight.nn.LazyConvTranspose1d(out_channels=32, kernel_size=3, padding=1)
        x = flashlight.randn(4, 16, 100)
        output = conv(x)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 32)
        self.assertEqual(conv.in_channels, 16)


@skipIfNoMLX
class TestLazyConvTranspose2d(TestCase):
    """Test nn.LazyConvTranspose2d."""

    def test_creation(self):
        """Test LazyConvTranspose2d creation."""
        conv = flashlight.nn.LazyConvTranspose2d(out_channels=32, kernel_size=3)
        self.assertEqual(conv.out_channels, 32)

    def test_forward_infers_in_channels(self):
        """Test LazyConvTranspose2d forward pass infers input channels."""
        conv = flashlight.nn.LazyConvTranspose2d(out_channels=32, kernel_size=3, padding=1)
        x = flashlight.randn(4, 16, 8, 8)
        output = conv(x)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 32)
        self.assertEqual(conv.in_channels, 16)


@skipIfNoMLX
class TestLazyConvTranspose3d(TestCase):
    """Test nn.LazyConvTranspose3d."""

    def test_creation(self):
        """Test LazyConvTranspose3d creation."""
        conv = flashlight.nn.LazyConvTranspose3d(out_channels=32, kernel_size=3)
        self.assertEqual(conv.out_channels, 32)

    def test_forward_infers_in_channels(self):
        """Test LazyConvTranspose3d forward pass infers input channels."""
        conv = flashlight.nn.LazyConvTranspose3d(out_channels=32, kernel_size=3, padding=1)
        x = flashlight.randn(2, 8, 4, 4, 4)
        output = conv(x)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 32)
        self.assertEqual(conv.in_channels, 8)


@skipIfNoMLX
class TestLazyBatchNorm1d(TestCase):
    """Test nn.LazyBatchNorm1d."""

    def test_creation(self):
        """Test LazyBatchNorm1d creation."""
        bn = flashlight.nn.LazyBatchNorm1d()
        self.assertIsNotNone(bn)

    def test_forward_infers_num_features(self):
        """Test LazyBatchNorm1d forward pass infers num_features."""
        bn = flashlight.nn.LazyBatchNorm1d()
        x = flashlight.randn(4, 32, 100)  # batch, channels, length
        output = bn(x)
        self.assertEqual(output.shape, (4, 32, 100))
        self.assertEqual(bn.num_features, 32)

    def test_2d_input(self):
        """Test LazyBatchNorm1d with 2D input."""
        bn = flashlight.nn.LazyBatchNorm1d()
        x = flashlight.randn(4, 32)  # batch, features
        output = bn(x)
        self.assertEqual(output.shape, (4, 32))


@skipIfNoMLX
class TestLazyBatchNorm2d(TestCase):
    """Test nn.LazyBatchNorm2d."""

    def test_creation(self):
        """Test LazyBatchNorm2d creation."""
        bn = flashlight.nn.LazyBatchNorm2d()
        self.assertIsNotNone(bn)

    def test_forward_infers_num_features(self):
        """Test LazyBatchNorm2d forward pass infers num_features."""
        bn = flashlight.nn.LazyBatchNorm2d()
        x = flashlight.randn(4, 32, 8, 8)  # batch, channels, height, width
        output = bn(x)
        self.assertEqual(output.shape, (4, 32, 8, 8))
        self.assertEqual(bn.num_features, 32)


@skipIfNoMLX
class TestLazyBatchNorm3d(TestCase):
    """Test nn.LazyBatchNorm3d."""

    def test_creation(self):
        """Test LazyBatchNorm3d creation."""
        bn = flashlight.nn.LazyBatchNorm3d()
        self.assertIsNotNone(bn)

    def test_forward_infers_num_features(self):
        """Test LazyBatchNorm3d forward pass infers num_features."""
        bn = flashlight.nn.LazyBatchNorm3d()
        x = flashlight.randn(2, 16, 4, 4, 4)  # batch, channels, depth, height, width
        output = bn(x)
        self.assertEqual(output.shape, (2, 16, 4, 4, 4))
        self.assertEqual(bn.num_features, 16)


@skipIfNoMLX
class TestLazyInstanceNorm1d(TestCase):
    """Test nn.LazyInstanceNorm1d."""

    def test_creation(self):
        """Test LazyInstanceNorm1d creation."""
        norm = flashlight.nn.LazyInstanceNorm1d()
        self.assertIsNotNone(norm)

    def test_forward_infers_num_features(self):
        """Test LazyInstanceNorm1d forward pass infers num_features."""
        norm = flashlight.nn.LazyInstanceNorm1d()
        x = flashlight.randn(4, 32, 100)
        output = norm(x)
        self.assertEqual(output.shape, (4, 32, 100))
        self.assertEqual(norm.num_features, 32)


@skipIfNoMLX
class TestLazyInstanceNorm2d(TestCase):
    """Test nn.LazyInstanceNorm2d."""

    def test_creation(self):
        """Test LazyInstanceNorm2d creation."""
        norm = flashlight.nn.LazyInstanceNorm2d()
        self.assertIsNotNone(norm)

    def test_forward_infers_num_features(self):
        """Test LazyInstanceNorm2d forward pass infers num_features."""
        norm = flashlight.nn.LazyInstanceNorm2d()
        x = flashlight.randn(4, 32, 8, 8)
        output = norm(x)
        self.assertEqual(output.shape, (4, 32, 8, 8))
        self.assertEqual(norm.num_features, 32)


@skipIfNoMLX
class TestLazyInstanceNorm3d(TestCase):
    """Test nn.LazyInstanceNorm3d."""

    def test_creation(self):
        """Test LazyInstanceNorm3d creation."""
        norm = flashlight.nn.LazyInstanceNorm3d()
        self.assertIsNotNone(norm)

    def test_forward_infers_num_features(self):
        """Test LazyInstanceNorm3d forward pass infers num_features."""
        norm = flashlight.nn.LazyInstanceNorm3d()
        x = flashlight.randn(2, 16, 4, 4, 4)
        output = norm(x)
        self.assertEqual(output.shape, (2, 16, 4, 4, 4))
        self.assertEqual(norm.num_features, 16)


@skipIfNoMLX
class TestLazyModuleReuse(TestCase):
    """Test lazy modules work correctly when reused."""

    def test_lazy_linear_reuse(self):
        """Test LazyLinear can be reused after initialization."""
        linear = flashlight.nn.LazyLinear(out_features=64)

        # First use initializes
        x1 = flashlight.randn(4, 32)
        out1 = linear(x1)
        self.assertEqual(out1.shape, (4, 64))

        # Second use with same in_features works
        x2 = flashlight.randn(8, 32)
        out2 = linear(x2)
        self.assertEqual(out2.shape, (8, 64))

    def test_lazy_conv2d_reuse(self):
        """Test LazyConv2d can be reused after initialization."""
        conv = flashlight.nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1)

        # First use initializes
        x1 = flashlight.randn(4, 16, 8, 8)
        out1 = conv(x1)
        self.assertEqual(out1.shape, (4, 32, 8, 8))

        # Second use with same in_channels works
        x2 = flashlight.randn(2, 16, 16, 16)
        out2 = conv(x2)
        self.assertEqual(out2.shape, (2, 32, 16, 16))


if __name__ == "__main__":
    unittest.main()
