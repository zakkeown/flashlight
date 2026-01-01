"""
Test Phase 4: Convolution Layers

Tests the nn.layers.conv module:
- Conv1d, Conv2d, Conv3d
- ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
- PyTorch parity
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
class TestConv1d(TestCase):
    """Test nn.Conv1d."""

    def test_creation(self):
        """Test Conv1d creation with default parameters."""
        conv = flashlight.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 16)

    def test_forward_shape(self):
        """Test Conv1d forward pass output shape."""
        conv = flashlight.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3)
        x = flashlight.randn(2, 3, 32)  # batch=2, channels=3, length=32
        output = conv(x)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 16)

    def test_with_padding(self):
        """Test Conv1d with padding."""
        conv = flashlight.nn.Conv1d(3, 16, kernel_size=3, padding=1)
        x = flashlight.randn(2, 3, 32)
        output = conv(x)
        self.assertEqual(output.shape[2], 32)  # Same size with padding=1

    def test_with_stride(self):
        """Test Conv1d with stride."""
        conv = flashlight.nn.Conv1d(3, 16, kernel_size=3, stride=2)
        x = flashlight.randn(2, 3, 32)
        output = conv(x)
        self.assertEqual(output.shape[2], 15)  # (32 - 3) / 2 + 1

    def test_with_groups(self):
        """Test Conv1d with groups."""
        conv = flashlight.nn.Conv1d(4, 8, kernel_size=3, groups=2)
        x = flashlight.randn(2, 4, 32)
        output = conv(x)
        self.assertEqual(output.shape[1], 8)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        in_ch, out_ch, kernel = 3, 16, 3

        conv_torch = torch.nn.Conv1d(in_ch, out_ch, kernel)
        conv_mlx = flashlight.nn.Conv1d(in_ch, out_ch, kernel)

        # Copy weights
        weight_np = conv_torch.weight.detach().numpy()
        bias_np = conv_torch.bias.detach().numpy()
        conv_mlx.weight._mlx_array = flashlight.tensor(weight_np)._mlx_array
        conv_mlx.bias._mlx_array = flashlight.tensor(bias_np)._mlx_array

        x_np = np.random.randn(2, in_ch, 32).astype(np.float32)
        out_torch = conv_torch(torch.tensor(x_np))
        out_mlx = conv_mlx(flashlight.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestConv2d(TestCase):
    """Test nn.Conv2d."""

    def test_creation(self):
        """Test Conv2d creation."""
        conv = flashlight.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 16)

    def test_forward_shape(self):
        """Test Conv2d forward pass output shape."""
        conv = flashlight.nn.Conv2d(3, 16, kernel_size=3)
        x = flashlight.randn(2, 3, 32, 32)
        output = conv(x)
        self.assertEqual(output.shape[:2], (2, 16))

    def test_with_padding(self):
        """Test Conv2d with padding."""
        conv = flashlight.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        x = flashlight.randn(2, 3, 32, 32)
        output = conv(x)
        self.assertEqual(output.shape[2:], (32, 32))

    def test_with_dilation(self):
        """Test Conv2d with dilation."""
        conv = flashlight.nn.Conv2d(3, 16, kernel_size=3, dilation=2)
        x = flashlight.randn(2, 3, 32, 32)
        output = conv(x)
        # With dilation=2, effective kernel is 5, output is 32-5+1=28
        self.assertEqual(output.shape[2], 28)


@skipIfNoMLX
class TestConv3d(TestCase):
    """Test nn.Conv3d."""

    def test_creation(self):
        """Test Conv3d creation."""
        conv = flashlight.nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3)
        self.assertEqual(conv.in_channels, 3)
        self.assertEqual(conv.out_channels, 16)

    def test_forward_shape(self):
        """Test Conv3d forward pass output shape."""
        conv = flashlight.nn.Conv3d(3, 16, kernel_size=3)
        x = flashlight.randn(2, 3, 8, 16, 16)  # batch, channels, D, H, W
        output = conv(x)
        self.assertEqual(output.shape[:2], (2, 16))


@skipIfNoMLX
class TestConvTranspose1d(TestCase):
    """Test nn.ConvTranspose1d."""

    def test_creation(self):
        """Test ConvTranspose1d creation."""
        conv = flashlight.nn.ConvTranspose1d(16, 3, kernel_size=3)
        self.assertEqual(conv.in_channels, 16)
        self.assertEqual(conv.out_channels, 3)

    def test_forward_shape(self):
        """Test ConvTranspose1d forward pass output shape."""
        conv = flashlight.nn.ConvTranspose1d(16, 3, kernel_size=3)
        x = flashlight.randn(2, 16, 32)
        output = conv(x)
        self.assertEqual(output.shape[:2], (2, 3))


@skipIfNoMLX
class TestConvTranspose2d(TestCase):
    """Test nn.ConvTranspose2d."""

    def test_creation(self):
        """Test ConvTranspose2d creation."""
        conv = flashlight.nn.ConvTranspose2d(16, 3, kernel_size=3)
        self.assertEqual(conv.in_channels, 16)
        self.assertEqual(conv.out_channels, 3)

    def test_forward_shape(self):
        """Test ConvTranspose2d forward pass output shape."""
        conv = flashlight.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2)
        x = flashlight.randn(2, 16, 16, 16)
        output = conv(x)
        self.assertEqual(output.shape[:2], (2, 3))
        # With stride=2, kernel=3: output_size = (input_size - 1) * stride + kernel
        # = (16 - 1) * 2 + 3 = 33

    def test_with_output_padding(self):
        """Test ConvTranspose2d with output_padding."""
        conv = flashlight.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, output_padding=1)
        x = flashlight.randn(2, 16, 16, 16)
        output = conv(x)
        self.assertEqual(output.shape[:2], (2, 3))


@skipIfNoMLX
class TestConvTranspose3d(TestCase):
    """Test nn.ConvTranspose3d."""

    def test_creation(self):
        """Test ConvTranspose3d creation."""
        conv = flashlight.nn.ConvTranspose3d(16, 3, kernel_size=3)
        self.assertEqual(conv.in_channels, 16)
        self.assertEqual(conv.out_channels, 3)

    def test_forward_shape(self):
        """Test ConvTranspose3d forward pass output shape."""
        conv = flashlight.nn.ConvTranspose3d(16, 3, kernel_size=3)
        x = flashlight.randn(2, 16, 8, 8, 8)
        output = conv(x)
        self.assertEqual(output.shape[:2], (2, 3))


if __name__ == '__main__':
    unittest.main()
