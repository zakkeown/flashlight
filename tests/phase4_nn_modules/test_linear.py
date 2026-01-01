"""
Test Phase 4: Linear Layers

Tests the linear modules:
- Linear
- Bilinear
- Identity
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
class TestLinear(TestCase):
    """Test nn.Linear."""

    def test_creation(self):
        """Test Linear creation with default parameters."""
        linear = flashlight.nn.Linear(in_features=64, out_features=128)
        self.assertEqual(linear.in_features, 64)
        self.assertEqual(linear.out_features, 128)

    def test_creation_with_bias(self):
        """Test Linear creation with bias."""
        linear = flashlight.nn.Linear(64, 128, bias=True)
        self.assertIsNotNone(linear.bias)
        self.assertEqual(linear.bias.shape, (128,))

    def test_creation_without_bias(self):
        """Test Linear creation without bias."""
        linear = flashlight.nn.Linear(64, 128, bias=False)
        self.assertIsNone(linear.bias)

    def test_forward_2d(self):
        """Test Linear forward pass with 2D input."""
        linear = flashlight.nn.Linear(64, 128)
        x = flashlight.randn(4, 64)  # batch, features
        output = linear(x)
        self.assertEqual(output.shape, (4, 128))

    def test_forward_3d(self):
        """Test Linear forward pass with 3D input."""
        linear = flashlight.nn.Linear(64, 128)
        x = flashlight.randn(4, 10, 64)  # batch, seq, features
        output = linear(x)
        self.assertEqual(output.shape, (4, 10, 128))

    def test_forward_4d(self):
        """Test Linear forward pass with 4D input."""
        linear = flashlight.nn.Linear(64, 128)
        x = flashlight.randn(4, 3, 8, 64)
        output = linear(x)
        self.assertEqual(output.shape, (4, 3, 8, 128))

    def test_weight_shape(self):
        """Test Linear weight has correct shape."""
        linear = flashlight.nn.Linear(64, 128)
        self.assertEqual(linear.weight.shape, (128, 64))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        in_features, out_features = 64, 128
        x_np = np.random.randn(4, in_features).astype(np.float32)
        weight_np = np.random.randn(out_features, in_features).astype(np.float32)
        bias_np = np.random.randn(out_features).astype(np.float32)

        # PyTorch
        linear_torch = torch.nn.Linear(in_features, out_features)
        linear_torch.weight.data = torch.tensor(weight_np)
        linear_torch.bias.data = torch.tensor(bias_np)

        # MLX
        linear_mlx = flashlight.nn.Linear(in_features, out_features)
        linear_mlx.weight._mlx_array = flashlight.tensor(weight_np)._mlx_array
        linear_mlx.bias._mlx_array = flashlight.tensor(bias_np)._mlx_array

        out_torch = linear_torch(torch.tensor(x_np))
        out_mlx = linear_mlx(flashlight.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6
        )


@skipIfNoMLX
class TestBilinear(TestCase):
    """Test nn.Bilinear."""

    def test_creation(self):
        """Test Bilinear creation."""
        bilinear = flashlight.nn.Bilinear(in1_features=32, in2_features=64, out_features=128)
        self.assertEqual(bilinear.in1_features, 32)
        self.assertEqual(bilinear.in2_features, 64)
        self.assertEqual(bilinear.out_features, 128)

    def test_creation_with_bias(self):
        """Test Bilinear creation with bias."""
        bilinear = flashlight.nn.Bilinear(32, 64, 128, bias=True)
        self.assertIsNotNone(bilinear.bias)

    def test_creation_without_bias(self):
        """Test Bilinear creation without bias."""
        bilinear = flashlight.nn.Bilinear(32, 64, 128, bias=False)
        self.assertIsNone(bilinear.bias)

    def test_forward_shape(self):
        """Test Bilinear forward pass."""
        bilinear = flashlight.nn.Bilinear(32, 64, 128)
        x1 = flashlight.randn(4, 32)
        x2 = flashlight.randn(4, 64)
        output = bilinear(x1, x2)
        self.assertEqual(output.shape, (4, 128))

    def test_forward_3d(self):
        """Test Bilinear forward with 3D input."""
        bilinear = flashlight.nn.Bilinear(32, 64, 128)
        x1 = flashlight.randn(4, 10, 32)
        x2 = flashlight.randn(4, 10, 64)
        output = bilinear(x1, x2)
        self.assertEqual(output.shape, (4, 10, 128))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        in1, in2, out = 32, 64, 128
        x1_np = np.random.randn(4, in1).astype(np.float32)
        x2_np = np.random.randn(4, in2).astype(np.float32)

        # PyTorch
        bilinear_torch = torch.nn.Bilinear(in1, in2, out)

        # MLX
        bilinear_mlx = flashlight.nn.Bilinear(in1, in2, out)
        # Copy weights
        bilinear_mlx.weight._mlx_array = flashlight.tensor(
            bilinear_torch.weight.detach().numpy()
        )._mlx_array
        bilinear_mlx.bias._mlx_array = flashlight.tensor(
            bilinear_torch.bias.detach().numpy()
        )._mlx_array

        out_torch = bilinear_torch(torch.tensor(x1_np), torch.tensor(x2_np))
        out_mlx = bilinear_mlx(flashlight.tensor(x1_np), flashlight.tensor(x2_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestIdentity(TestCase):
    """Test nn.Identity."""

    def test_creation(self):
        """Test Identity creation."""
        identity = flashlight.nn.Identity()
        self.assertIsNotNone(identity)

    def test_forward_2d(self):
        """Test Identity forward with 2D input."""
        identity = flashlight.nn.Identity()
        x = flashlight.randn(4, 64)
        output = identity(x)
        np.testing.assert_array_equal(x.numpy(), output.numpy())

    def test_forward_3d(self):
        """Test Identity forward with 3D input."""
        identity = flashlight.nn.Identity()
        x = flashlight.randn(4, 10, 64)
        output = identity(x)
        np.testing.assert_array_equal(x.numpy(), output.numpy())

    def test_forward_4d(self):
        """Test Identity forward with 4D input."""
        identity = flashlight.nn.Identity()
        x = flashlight.randn(4, 3, 8, 8)
        output = identity(x)
        np.testing.assert_array_equal(x.numpy(), output.numpy())


if __name__ == "__main__":
    unittest.main()
