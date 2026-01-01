"""
Test Phase 4: Padding Layers

Tests the nn.layers.padding module:
- ZeroPad1d, ZeroPad2d, ZeroPad3d
- ConstantPad1d, ConstantPad2d, ConstantPad3d
- ReflectionPad1d, ReflectionPad2d, ReflectionPad3d
- ReplicationPad1d, ReplicationPad2d, ReplicationPad3d
- CircularPad1d, CircularPad2d, CircularPad3d
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
class TestZeroPad1d(TestCase):
    """Test nn.ZeroPad1d."""

    def test_creation(self):
        """Test ZeroPad1d creation."""
        pad = flashlight.nn.ZeroPad1d(2)
        self.assertEqual(pad.padding, (2, 2))

    def test_creation_asymmetric(self):
        """Test ZeroPad1d with asymmetric padding."""
        pad = flashlight.nn.ZeroPad1d((1, 3))
        self.assertEqual(pad.padding, (1, 3))

    def test_forward_shape(self):
        """Test ZeroPad1d forward pass output shape."""
        pad = flashlight.nn.ZeroPad1d(2)
        x = flashlight.randn(2, 3, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14))

    def test_values(self):
        """Test that padded values are zeros."""
        pad = flashlight.nn.ZeroPad1d(2)
        x = flashlight.ones(1, 1, 5)
        output = pad(x)
        # Check padded regions are zero
        np.testing.assert_array_equal(output[:, :, :2].numpy(), 0)
        np.testing.assert_array_equal(output[:, :, -2:].numpy(), 0)


@skipIfNoMLX
class TestZeroPad2d(TestCase):
    """Test nn.ZeroPad2d."""

    def test_creation(self):
        """Test ZeroPad2d creation."""
        pad = flashlight.nn.ZeroPad2d(2)
        self.assertEqual(pad.padding, (2, 2, 2, 2))

    def test_forward_shape(self):
        """Test ZeroPad2d forward pass output shape."""
        pad = flashlight.nn.ZeroPad2d((1, 2, 3, 4))
        x = flashlight.randn(2, 3, 10, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 17, 13))  # H+3+4, W+1+2


@skipIfNoMLX
class TestZeroPad3d(TestCase):
    """Test nn.ZeroPad3d."""

    def test_creation(self):
        """Test ZeroPad3d creation."""
        pad = flashlight.nn.ZeroPad3d(2)
        self.assertEqual(pad.padding, (2, 2, 2, 2, 2, 2))

    def test_forward_shape(self):
        """Test ZeroPad3d forward pass output shape."""
        pad = flashlight.nn.ZeroPad3d(1)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 10, 10, 10))


@skipIfNoMLX
class TestConstantPad1d(TestCase):
    """Test nn.ConstantPad1d."""

    def test_creation(self):
        """Test ConstantPad1d creation."""
        pad = flashlight.nn.ConstantPad1d(2, value=3.14)
        self.assertEqual(pad.value, 3.14)

    def test_forward_shape(self):
        """Test ConstantPad1d forward pass output shape."""
        pad = flashlight.nn.ConstantPad1d(2, value=0)
        x = flashlight.randn(2, 3, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14))

    def test_values(self):
        """Test that padded values are the constant."""
        pad = flashlight.nn.ConstantPad1d(2, value=5.0)
        x = flashlight.ones(1, 1, 5)
        output = pad(x)
        np.testing.assert_array_almost_equal(output[:, :, :2].numpy(), 5.0)
        np.testing.assert_array_almost_equal(output[:, :, -2:].numpy(), 5.0)


@skipIfNoMLX
class TestConstantPad2d(TestCase):
    """Test nn.ConstantPad2d."""

    def test_forward_shape(self):
        """Test ConstantPad2d forward pass output shape."""
        pad = flashlight.nn.ConstantPad2d(2, value=0)
        x = flashlight.randn(2, 3, 10, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14, 14))


@skipIfNoMLX
class TestConstantPad3d(TestCase):
    """Test nn.ConstantPad3d."""

    def test_forward_shape(self):
        """Test ConstantPad3d forward pass output shape."""
        pad = flashlight.nn.ConstantPad3d(1, value=0)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 10, 10, 10))


@skipIfNoMLX
class TestReflectionPad1d(TestCase):
    """Test nn.ReflectionPad1d."""

    def test_creation(self):
        """Test ReflectionPad1d creation."""
        pad = flashlight.nn.ReflectionPad1d(2)
        self.assertEqual(pad.padding, (2, 2))

    def test_forward_shape(self):
        """Test ReflectionPad1d forward pass output shape."""
        pad = flashlight.nn.ReflectionPad1d(2)
        x = flashlight.randn(2, 3, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        pad_torch = torch.nn.ReflectionPad1d(2)
        pad_mlx = flashlight.nn.ReflectionPad1d(2)

        x_np = np.random.randn(2, 3, 10).astype(np.float32)
        out_torch = pad_torch(torch.tensor(x_np))
        out_mlx = pad_mlx(flashlight.tensor(x_np))

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@skipIfNoMLX
class TestReflectionPad2d(TestCase):
    """Test nn.ReflectionPad2d."""

    def test_forward_shape(self):
        """Test ReflectionPad2d forward pass output shape."""
        pad = flashlight.nn.ReflectionPad2d(2)
        x = flashlight.randn(2, 3, 10, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14, 14))


@skipIfNoMLX
class TestReflectionPad3d(TestCase):
    """Test nn.ReflectionPad3d."""

    def test_forward_shape(self):
        """Test ReflectionPad3d forward pass output shape."""
        pad = flashlight.nn.ReflectionPad3d(1)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 10, 10, 10))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        pad_torch = torch.nn.ReflectionPad3d(1)
        pad_mlx = flashlight.nn.ReflectionPad3d(1)

        x_np = np.random.randn(2, 3, 6, 6, 6).astype(np.float32)
        out_torch = pad_torch(torch.tensor(x_np))
        out_mlx = pad_mlx(flashlight.tensor(x_np))

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@skipIfNoMLX
class TestReplicationPad1d(TestCase):
    """Test nn.ReplicationPad1d."""

    def test_forward_shape(self):
        """Test ReplicationPad1d forward pass output shape."""
        pad = flashlight.nn.ReplicationPad1d(2)
        x = flashlight.randn(2, 3, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14))


@skipIfNoMLX
class TestReplicationPad2d(TestCase):
    """Test nn.ReplicationPad2d."""

    def test_forward_shape(self):
        """Test ReplicationPad2d forward pass output shape."""
        pad = flashlight.nn.ReplicationPad2d(2)
        x = flashlight.randn(2, 3, 10, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14, 14))


@skipIfNoMLX
class TestReplicationPad3d(TestCase):
    """Test nn.ReplicationPad3d."""

    def test_forward_shape(self):
        """Test ReplicationPad3d forward pass output shape."""
        pad = flashlight.nn.ReplicationPad3d(1)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 10, 10, 10))


@skipIfNoMLX
class TestCircularPad1d(TestCase):
    """Test nn.CircularPad1d."""

    def test_forward_shape(self):
        """Test CircularPad1d forward pass output shape."""
        pad = flashlight.nn.CircularPad1d(2)
        x = flashlight.randn(2, 3, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14))


@skipIfNoMLX
class TestCircularPad2d(TestCase):
    """Test nn.CircularPad2d."""

    def test_forward_shape(self):
        """Test CircularPad2d forward pass output shape."""
        pad = flashlight.nn.CircularPad2d(2)
        x = flashlight.randn(2, 3, 10, 10)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 14, 14))


@skipIfNoMLX
class TestCircularPad3d(TestCase):
    """Test nn.CircularPad3d."""

    def test_forward_shape(self):
        """Test CircularPad3d forward pass output shape."""
        pad = flashlight.nn.CircularPad3d(1)
        x = flashlight.randn(2, 3, 8, 8, 8)
        output = pad(x)
        self.assertEqual(output.shape, (2, 3, 10, 10, 10))


if __name__ == "__main__":
    unittest.main()
