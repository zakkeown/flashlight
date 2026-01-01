"""
Test Phase 4: Full Activation Layer Tests

Tests all activation layers including:
- ReLU, LeakyReLU, ELU, ReLU6, SELU, CELU
- Sigmoid, Tanh, GELU, SiLU, Mish
- Softmax, LogSoftmax, Softmin, Softmax2d
- Hardtanh, Hardsigmoid, Hardswish, Hardshrink
- Softplus, Softshrink, Softsign, Tanhshrink
- PReLU, RReLU, GLU, Threshold
"""

import sys
sys.path.insert(0, '../..')

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
class TestReLU(TestCase):
    """Test nn.ReLU."""

    def test_forward(self):
        """Test ReLU forward pass."""
        relu = flashlight.nn.ReLU()
        x = flashlight.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(output.numpy(), expected)


@skipIfNoMLX
class TestLeakyReLU(TestCase):
    """Test nn.LeakyReLU."""

    def test_forward(self):
        """Test LeakyReLU forward pass."""
        lrelu = flashlight.nn.LeakyReLU(negative_slope=0.1)
        x = flashlight.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        output = lrelu(x)
        expected = np.array([-0.2, -0.1, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(output.numpy(), expected)


@skipIfNoMLX
class TestELU(TestCase):
    """Test nn.ELU."""

    def test_forward(self):
        """Test ELU forward pass."""
        elu = flashlight.nn.ELU(alpha=1.0)
        x = flashlight.randn(2, 3)
        output = elu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestReLU6(TestCase):
    """Test nn.ReLU6."""

    def test_forward(self):
        """Test ReLU6 forward pass (clips at 6)."""
        relu6 = flashlight.nn.ReLU6()
        x = flashlight.tensor([-1.0, 0.0, 3.0, 6.0, 10.0])
        output = relu6(x)
        expected = np.array([0.0, 0.0, 3.0, 6.0, 6.0])
        np.testing.assert_array_almost_equal(output.numpy(), expected)


@skipIfNoMLX
class TestSELU(TestCase):
    """Test nn.SELU."""

    def test_forward(self):
        """Test SELU forward pass."""
        selu = flashlight.nn.SELU()
        x = flashlight.randn(2, 3)
        output = selu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestCELU(TestCase):
    """Test nn.CELU."""

    def test_forward(self):
        """Test CELU forward pass."""
        celu = flashlight.nn.CELU(alpha=1.0)
        x = flashlight.randn(2, 3)
        output = celu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestSigmoid(TestCase):
    """Test nn.Sigmoid."""

    def test_forward(self):
        """Test Sigmoid forward pass."""
        sigmoid = flashlight.nn.Sigmoid()
        x = flashlight.tensor([0.0])
        output = sigmoid(x)
        np.testing.assert_almost_equal(output.numpy()[0], 0.5)


@skipIfNoMLX
class TestTanh(TestCase):
    """Test nn.Tanh."""

    def test_forward(self):
        """Test Tanh forward pass."""
        tanh = flashlight.nn.Tanh()
        x = flashlight.tensor([0.0])
        output = tanh(x)
        np.testing.assert_almost_equal(output.numpy()[0], 0.0)


@skipIfNoMLX
class TestGELU(TestCase):
    """Test nn.GELU."""

    def test_forward(self):
        """Test GELU forward pass."""
        gelu = flashlight.nn.GELU()
        x = flashlight.randn(2, 3)
        output = gelu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestSiLU(TestCase):
    """Test nn.SiLU (Swish)."""

    def test_forward(self):
        """Test SiLU forward pass."""
        silu = flashlight.nn.SiLU()
        x = flashlight.randn(2, 3)
        output = silu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestMish(TestCase):
    """Test nn.Mish."""

    def test_forward(self):
        """Test Mish forward pass."""
        mish = flashlight.nn.Mish()
        x = flashlight.randn(2, 3)
        output = mish(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestSoftmax(TestCase):
    """Test nn.Softmax."""

    def test_forward(self):
        """Test Softmax forward pass."""
        softmax = flashlight.nn.Softmax(dim=1)
        x = flashlight.randn(2, 5)
        output = softmax(x)
        # Softmax should sum to 1 along dim=1
        sums = output.numpy().sum(axis=1)
        np.testing.assert_array_almost_equal(sums, [1.0, 1.0])


@skipIfNoMLX
class TestLogSoftmax(TestCase):
    """Test nn.LogSoftmax."""

    def test_forward(self):
        """Test LogSoftmax forward pass."""
        logsoftmax = flashlight.nn.LogSoftmax(dim=1)
        x = flashlight.randn(2, 5)
        output = logsoftmax(x)
        # exp(LogSoftmax) should sum to 1
        sums = np.exp(output.numpy()).sum(axis=1)
        np.testing.assert_array_almost_equal(sums, [1.0, 1.0])


@skipIfNoMLX
class TestSoftmin(TestCase):
    """Test nn.Softmin."""

    def test_forward(self):
        """Test Softmin forward pass."""
        softmin = flashlight.nn.Softmin(dim=1)
        x = flashlight.randn(2, 5)
        output = softmin(x)
        sums = output.numpy().sum(axis=1)
        np.testing.assert_array_almost_equal(sums, [1.0, 1.0])


@skipIfNoMLX
class TestHardtanh(TestCase):
    """Test nn.Hardtanh."""

    def test_forward(self):
        """Test Hardtanh forward pass."""
        hardtanh = flashlight.nn.Hardtanh(min_val=-1, max_val=1)
        x = flashlight.tensor([-2.0, 0.0, 2.0])
        output = hardtanh(x)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(output.numpy(), expected)


@skipIfNoMLX
class TestHardsigmoid(TestCase):
    """Test nn.Hardsigmoid."""

    def test_forward(self):
        """Test Hardsigmoid forward pass."""
        hardsigmoid = flashlight.nn.Hardsigmoid()
        x = flashlight.randn(2, 3)
        output = hardsigmoid(x)
        self.assertEqual(output.shape, (2, 3))
        # Output should be in [0, 1]
        self.assertTrue((output.numpy() >= 0).all())
        self.assertTrue((output.numpy() <= 1).all())


@skipIfNoMLX
class TestHardswish(TestCase):
    """Test nn.Hardswish."""

    def test_forward(self):
        """Test Hardswish forward pass."""
        hardswish = flashlight.nn.Hardswish()
        x = flashlight.randn(2, 3)
        output = hardswish(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestSoftplus(TestCase):
    """Test nn.Softplus."""

    def test_forward(self):
        """Test Softplus forward pass."""
        softplus = flashlight.nn.Softplus()
        x = flashlight.randn(2, 3)
        output = softplus(x)
        # Softplus output is always positive
        self.assertTrue((output.numpy() > 0).all())


@skipIfNoMLX
class TestPReLU(TestCase):
    """Test nn.PReLU."""

    def test_creation(self):
        """Test PReLU creation."""
        prelu = flashlight.nn.PReLU()
        self.assertIsNotNone(prelu)

    def test_forward(self):
        """Test PReLU forward pass."""
        prelu = flashlight.nn.PReLU()
        x = flashlight.randn(2, 3)
        output = prelu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestGLU(TestCase):
    """Test nn.GLU (Gated Linear Unit)."""

    def test_forward(self):
        """Test GLU forward pass."""
        glu = flashlight.nn.GLU(dim=1)
        x = flashlight.randn(2, 6)  # dim=1 will split 6 -> 3
        output = glu(x)
        self.assertEqual(output.shape, (2, 3))


@skipIfNoMLX
class TestThreshold(TestCase):
    """Test nn.Threshold."""

    def test_forward(self):
        """Test Threshold forward pass."""
        threshold = flashlight.nn.Threshold(threshold=0.5, value=0.0)
        x = flashlight.tensor([0.0, 0.5, 1.0])
        output = threshold(x)
        # Values <= threshold should be set to value
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(output.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
