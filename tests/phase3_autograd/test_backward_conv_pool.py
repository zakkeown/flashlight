"""
Test Phase 3: Convolution and Pooling Backward Operations

Tests gradient computation for convolution and pooling:
- Conv2dBackward
- MaxPool2dBackward
- AvgPool2dBackward
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
class TestConv2dBackward(TestCase):
    """Test Conv2d gradient computation."""

    def test_conv2d_simple(self):
        """Test basic Conv2d backward."""
        # Create a simple Conv2d layer
        conv = flashlight.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # Set weights to known values for predictable gradients
        with flashlight.no_grad():
            conv.weight.fill_(1.0 / 9.0)

        # Input: batch=1, channels=1, height=4, width=4
        x = flashlight.randn(1, 1, 4, 4, requires_grad=True)
        y = conv(x)
        loss = flashlight.sum(y)
        loss.backward()

        # Check that gradient exists and has correct shape
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_conv2d_multiple_channels(self):
        """Test Conv2d backward with multiple channels."""
        conv = flashlight.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)

        x = flashlight.randn(2, 3, 8, 8, requires_grad=True)
        y = conv(x)
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertIsNotNone(conv.weight.grad)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertIsNotNone(conv.bias.grad)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)

    def test_conv2d_stride(self):
        """Test Conv2d backward with stride > 1."""
        conv = flashlight.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)

        x = flashlight.randn(1, 1, 8, 8, requires_grad=True)
        y = conv(x)
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


@skipIfNoMLX
class TestMaxPool2dBackward(TestCase):
    """Test MaxPool2d gradient computation."""

    def test_maxpool2d_simple(self):
        """Test basic MaxPool2d backward."""
        pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2)

        x = flashlight.tensor([[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]], requires_grad=True)

        y = pool(x)
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # Gradient should only flow to max elements in each pool
        # For 2x2 pools: max at (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
        grad = x.grad.numpy()
        # Max elements should have gradient 1.0, others 0.0
        self.assertEqual(grad[0, 0, 1, 1], 1.0)  # max of top-left 2x2
        self.assertEqual(grad[0, 0, 1, 3], 1.0)  # max of top-right 2x2
        self.assertEqual(grad[0, 0, 3, 1], 1.0)  # max of bottom-left 2x2
        self.assertEqual(grad[0, 0, 3, 3], 1.0)  # max of bottom-right 2x2

    def test_maxpool2d_batch(self):
        """Test MaxPool2d backward with batch dimension."""
        pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2)

        x = flashlight.randn(4, 3, 8, 8, requires_grad=True)
        y = pool(x)
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


@skipIfNoMLX
class TestAvgPool2dBackward(TestCase):
    """Test AvgPool2d gradient computation."""

    def test_avgpool2d_simple(self):
        """Test basic AvgPool2d backward."""
        pool = flashlight.nn.AvgPool2d(kernel_size=2, stride=2)

        x = flashlight.tensor([[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]], requires_grad=True)

        y = pool(x)
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

        # For average pooling, gradient is distributed equally
        # Each element in a 2x2 pool gets gradient = 1/(2*2) = 0.25
        grad = x.grad.numpy()
        np.testing.assert_array_almost_equal(grad, np.full((1, 1, 4, 4), 0.25))

    def test_avgpool2d_batch(self):
        """Test AvgPool2d backward with batch dimension."""
        pool = flashlight.nn.AvgPool2d(kernel_size=2, stride=2)

        x = flashlight.randn(4, 3, 8, 8, requires_grad=True)
        y = pool(x)
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


@skipIfNoMLX
class TestConvPoolChain(TestCase):
    """Test chained convolution and pooling operations."""

    def test_conv_maxpool_chain(self):
        """Test Conv2d followed by MaxPool2d backward."""
        conv = flashlight.nn.Conv2d(1, 4, kernel_size=3, padding=1)
        pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2)

        x = flashlight.randn(1, 1, 8, 8, requires_grad=True)
        y = pool(conv(x))
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertIsNotNone(conv.weight.grad)
        self.assertIsNotNone(conv.bias.grad)

    def test_conv_avgpool_chain(self):
        """Test Conv2d followed by AvgPool2d backward."""
        conv = flashlight.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        pool = flashlight.nn.AvgPool2d(kernel_size=2, stride=2)

        x = flashlight.randn(2, 3, 16, 16, requires_grad=True)
        y = pool(conv(x))
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
