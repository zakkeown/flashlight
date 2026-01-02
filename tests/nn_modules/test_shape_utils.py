"""
Test Phase 4: Shape Utility Layers

Tests the shape utility modules:
- Flatten, Unflatten
- Identity
- Fold, Unfold
- Upsample, UpsamplingNearest2d, UpsamplingBilinear2d
- PixelShuffle, PixelUnshuffle
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
class TestFlatten(TestCase):
    """Test nn.Flatten."""

    def test_creation(self):
        """Test Flatten creation with default parameters."""
        flatten = flashlight.nn.Flatten()
        self.assertEqual(flatten.start_dim, 1)
        self.assertEqual(flatten.end_dim, -1)

    def test_creation_with_dims(self):
        """Test Flatten creation with custom dimensions."""
        flatten = flashlight.nn.Flatten(start_dim=2, end_dim=3)
        self.assertEqual(flatten.start_dim, 2)
        self.assertEqual(flatten.end_dim, 3)

    def test_forward_default(self):
        """Test Flatten forward with default dims."""
        flatten = flashlight.nn.Flatten()
        x = flashlight.randn(4, 3, 8, 8)  # batch, channels, height, width
        output = flatten(x)
        self.assertEqual(output.shape, (4, 192))

    def test_forward_custom_dims(self):
        """Test Flatten forward with custom dims."""
        flatten = flashlight.nn.Flatten(start_dim=2, end_dim=3)
        x = flashlight.randn(4, 3, 8, 8)
        output = flatten(x)
        self.assertEqual(output.shape, (4, 3, 64))

    def test_forward_1d_input(self):
        """Test Flatten with 1D input."""
        flatten = flashlight.nn.Flatten(start_dim=0, end_dim=-1)
        x = flashlight.randn(10)
        output = flatten(x)
        self.assertEqual(output.shape, (10,))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 3, 8, 8).astype(np.float32)

        flatten_torch = torch.nn.Flatten()
        flatten_mlx = flashlight.nn.Flatten()

        out_torch = flatten_torch(torch.tensor(x_np))
        out_mlx = flatten_mlx(flashlight.tensor(x_np))

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@skipIfNoMLX
class TestUnflatten(TestCase):
    """Test nn.Unflatten."""

    def test_creation(self):
        """Test Unflatten creation."""
        unflatten = flashlight.nn.Unflatten(dim=1, unflattened_size=(3, 8, 8))
        self.assertEqual(unflatten.dim, 1)
        self.assertEqual(unflatten.unflattened_size, (3, 8, 8))

    def test_forward_shape(self):
        """Test Unflatten forward pass."""
        unflatten = flashlight.nn.Unflatten(dim=1, unflattened_size=(3, 8, 8))
        x = flashlight.randn(4, 192)  # Flattened input
        output = unflatten(x)
        self.assertEqual(output.shape, (4, 3, 8, 8))

    def test_round_trip_with_flatten(self):
        """Test Flatten followed by Unflatten restores shape."""
        flatten = flashlight.nn.Flatten()
        unflatten = flashlight.nn.Unflatten(dim=1, unflattened_size=(3, 8, 8))

        x = flashlight.randn(4, 3, 8, 8)
        flattened = flatten(x)
        unflattened = unflatten(flattened)

        self.assertEqual(unflattened.shape, x.shape)


@skipIfNoMLX
class TestIdentity(TestCase):
    """Test nn.Identity."""

    def test_creation(self):
        """Test Identity creation."""
        identity = flashlight.nn.Identity()
        self.assertIsNotNone(identity)

    def test_forward_pass_through(self):
        """Test Identity passes input unchanged."""
        identity = flashlight.nn.Identity()
        x = flashlight.randn(4, 10)
        output = identity(x)
        np.testing.assert_array_equal(x.numpy(), output.numpy())

    def test_forward_with_args(self):
        """Test Identity ignores extra arguments."""
        identity = flashlight.nn.Identity()
        x = flashlight.randn(4, 10)
        output = identity(x)  # Should still work
        self.assertEqual(output.shape, (4, 10))


@skipIfNoMLX
class TestFold(TestCase):
    """Test nn.Fold."""

    def test_creation(self):
        """Test Fold creation."""
        fold = flashlight.nn.Fold(output_size=(4, 4), kernel_size=(2, 2))
        self.assertEqual(fold.output_size, (4, 4))

    def test_forward_shape(self):
        """Test Fold forward pass."""
        fold = flashlight.nn.Fold(output_size=(4, 4), kernel_size=(2, 2))
        # Input: (batch, C * kernel_size[0] * kernel_size[1], L)
        # For 4x4 output with 2x2 kernel: L = (4-2+1) * (4-2+1) = 9
        x = flashlight.randn(1, 12, 9)  # 3 channels, 2x2 kernel, 9 patches
        output = fold(x)
        self.assertEqual(output.shape, (1, 3, 4, 4))


@skipIfNoMLX
class TestUnfold(TestCase):
    """Test nn.Unfold."""

    def test_creation(self):
        """Test Unfold creation."""
        unfold = flashlight.nn.Unfold(kernel_size=(2, 2))
        self.assertEqual(unfold.kernel_size, (2, 2))

    def test_forward_shape(self):
        """Test Unfold forward pass."""
        unfold = flashlight.nn.Unfold(kernel_size=(2, 2))
        x = flashlight.randn(1, 3, 4, 4)  # batch, channels, height, width
        output = unfold(x)
        # Output: (batch, C * kernel_size[0] * kernel_size[1], L)
        # L = (4-2+1) * (4-2+1) = 9
        self.assertEqual(output.shape, (1, 12, 9))

    def test_with_stride(self):
        """Test Unfold with stride."""
        unfold = flashlight.nn.Unfold(kernel_size=(2, 2), stride=(2, 2))
        x = flashlight.randn(1, 3, 4, 4)
        output = unfold(x)
        # L = (4-2)/2 + 1) * ((4-2)/2 + 1) = 4
        self.assertEqual(output.shape, (1, 12, 4))

    def test_with_padding(self):
        """Test Unfold with padding."""
        unfold = flashlight.nn.Unfold(kernel_size=(2, 2), padding=(1, 1))
        x = flashlight.randn(1, 3, 4, 4)
        output = unfold(x)
        # With padding=1: (4+2-2+1) * (4+2-2+1) = 25
        self.assertEqual(output.shape, (1, 12, 25))


@skipIfNoMLX
class TestUpsample(TestCase):
    """Test nn.Upsample."""

    def test_creation_with_scale_factor(self):
        """Test Upsample creation with scale_factor."""
        upsample = flashlight.nn.Upsample(scale_factor=2)
        self.assertEqual(upsample.scale_factor, 2)

    def test_creation_with_size(self):
        """Test Upsample creation with size."""
        upsample = flashlight.nn.Upsample(size=(8, 8))
        self.assertEqual(upsample.size, (8, 8))

    def test_forward_with_scale_factor(self):
        """Test Upsample forward with scale_factor."""
        upsample = flashlight.nn.Upsample(scale_factor=2, mode="nearest")
        x = flashlight.randn(1, 3, 4, 4)
        output = upsample(x)
        self.assertEqual(output.shape, (1, 3, 8, 8))

    def test_forward_with_size(self):
        """Test Upsample forward with size."""
        upsample = flashlight.nn.Upsample(size=(8, 8), mode="nearest")
        x = flashlight.randn(1, 3, 4, 4)
        output = upsample(x)
        self.assertEqual(output.shape, (1, 3, 8, 8))

    def test_bilinear_mode(self):
        """Test Upsample with bilinear mode."""
        upsample = flashlight.nn.Upsample(scale_factor=2, mode="bilinear")
        x = flashlight.randn(1, 3, 4, 4)
        output = upsample(x)
        self.assertEqual(output.shape, (1, 3, 8, 8))


@skipIfNoMLX
class TestUpsamplingNearest2d(TestCase):
    """Test nn.UpsamplingNearest2d."""

    def test_creation(self):
        """Test UpsamplingNearest2d creation."""
        upsample = flashlight.nn.UpsamplingNearest2d(scale_factor=2)
        self.assertIsNotNone(upsample)

    def test_forward_shape(self):
        """Test UpsamplingNearest2d forward pass."""
        upsample = flashlight.nn.UpsamplingNearest2d(scale_factor=2)
        x = flashlight.randn(1, 3, 4, 4)
        output = upsample(x)
        self.assertEqual(output.shape, (1, 3, 8, 8))


@skipIfNoMLX
class TestUpsamplingBilinear2d(TestCase):
    """Test nn.UpsamplingBilinear2d."""

    def test_creation(self):
        """Test UpsamplingBilinear2d creation."""
        upsample = flashlight.nn.UpsamplingBilinear2d(scale_factor=2)
        self.assertIsNotNone(upsample)

    def test_forward_shape(self):
        """Test UpsamplingBilinear2d forward pass."""
        upsample = flashlight.nn.UpsamplingBilinear2d(scale_factor=2)
        x = flashlight.randn(1, 3, 4, 4)
        output = upsample(x)
        self.assertEqual(output.shape, (1, 3, 8, 8))


@skipIfNoMLX
class TestPixelShuffle(TestCase):
    """Test nn.PixelShuffle."""

    def test_creation(self):
        """Test PixelShuffle creation."""
        shuffle = flashlight.nn.PixelShuffle(upscale_factor=2)
        self.assertEqual(shuffle.upscale_factor, 2)

    def test_forward_shape(self):
        """Test PixelShuffle forward pass."""
        shuffle = flashlight.nn.PixelShuffle(upscale_factor=2)
        # Input: (N, C * r^2, H, W) -> Output: (N, C, H*r, W*r)
        x = flashlight.randn(1, 12, 4, 4)  # 12 = 3 * 2^2
        output = shuffle(x)
        self.assertEqual(output.shape, (1, 3, 8, 8))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(1, 12, 4, 4).astype(np.float32)

        shuffle_torch = torch.nn.PixelShuffle(2)
        shuffle_mlx = flashlight.nn.PixelShuffle(2)

        out_torch = shuffle_torch(torch.tensor(x_np))
        out_mlx = shuffle_mlx(flashlight.tensor(x_np))

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@skipIfNoMLX
class TestPixelUnshuffle(TestCase):
    """Test nn.PixelUnshuffle."""

    def test_creation(self):
        """Test PixelUnshuffle creation."""
        unshuffle = flashlight.nn.PixelUnshuffle(downscale_factor=2)
        self.assertEqual(unshuffle.downscale_factor, 2)

    def test_forward_shape(self):
        """Test PixelUnshuffle forward pass."""
        unshuffle = flashlight.nn.PixelUnshuffle(downscale_factor=2)
        # Input: (N, C, H, W) -> Output: (N, C * r^2, H/r, W/r)
        x = flashlight.randn(1, 3, 8, 8)
        output = unshuffle(x)
        self.assertEqual(output.shape, (1, 12, 4, 4))

    def test_round_trip_with_pixel_shuffle(self):
        """Test PixelShuffle followed by PixelUnshuffle."""
        shuffle = flashlight.nn.PixelShuffle(2)
        unshuffle = flashlight.nn.PixelUnshuffle(2)

        x = flashlight.randn(1, 12, 4, 4)
        shuffled = shuffle(x)
        unshuffled = unshuffle(shuffled)

        np.testing.assert_allclose(x.numpy(), unshuffled.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
