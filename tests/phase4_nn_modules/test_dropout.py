"""
Test Phase 4: Dropout Layers

Tests the nn.layers.dropout module:
- Dropout
- Dropout1d, Dropout2d, Dropout3d
- AlphaDropout, FeatureAlphaDropout
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestDropout(TestCase):
    """Test nn.Dropout."""

    def test_dropout_creation(self):
        """Test Dropout creation with default parameters."""
        dropout = mlx_compat.nn.Dropout()
        self.assertEqual(dropout.p, 0.5)

    def test_dropout_creation_with_p(self):
        """Test Dropout creation with custom probability."""
        dropout = mlx_compat.nn.Dropout(p=0.3)
        self.assertEqual(dropout.p, 0.3)

    def test_dropout_train_mode(self):
        """Test Dropout in training mode applies dropout."""
        dropout = mlx_compat.nn.Dropout(p=0.5)
        dropout.train()

        # Use larger tensor for statistical significance
        x = mlx_compat.ones(1000, 1000)
        output = dropout(x)

        # Check that some values are zero (dropout applied)
        output_np = output.numpy()
        zeros = (output_np == 0).sum()
        self.assertGreater(zeros, 0)

    def test_dropout_eval_mode(self):
        """Test Dropout in eval mode passes input unchanged."""
        dropout = mlx_compat.nn.Dropout(p=0.5)
        dropout.eval()

        x = mlx_compat.randn(10, 10)
        output = dropout(x)

        # Output should equal input in eval mode
        np.testing.assert_array_equal(output.numpy(), x.numpy())

    def test_dropout_p_zero(self):
        """Test Dropout with p=0 passes input unchanged."""
        dropout = mlx_compat.nn.Dropout(p=0.0)
        dropout.train()

        x = mlx_compat.randn(10, 10)
        output = dropout(x)

        np.testing.assert_array_equal(output.numpy(), x.numpy())

    def test_dropout_invalid_p(self):
        """Test Dropout with invalid p raises error."""
        with self.assertRaises(ValueError):
            mlx_compat.nn.Dropout(p=-0.1)
        with self.assertRaises(ValueError):
            mlx_compat.nn.Dropout(p=1.5)

    def test_dropout_shape_preserved(self):
        """Test Dropout preserves input shape."""
        dropout = mlx_compat.nn.Dropout(p=0.5)
        dropout.train()

        x = mlx_compat.randn(4, 8, 16)
        output = dropout(x)

        self.assertEqual(output.shape, x.shape)


@skipIfNoMLX
class TestDropout1d(TestCase):
    """Test nn.Dropout1d."""

    def test_dropout1d_creation(self):
        """Test Dropout1d creation."""
        dropout = mlx_compat.nn.Dropout1d(p=0.5)
        self.assertEqual(dropout.p, 0.5)

    def test_dropout1d_forward_3d(self):
        """Test Dropout1d forward pass with 3D input."""
        dropout = mlx_compat.nn.Dropout1d(p=0.5)
        dropout.train()

        x = mlx_compat.randn(4, 64, 32)  # (N, C, L)
        output = dropout(x)

        self.assertEqual(output.shape, (4, 64, 32))

    def test_dropout1d_forward_2d(self):
        """Test Dropout1d forward pass with 2D input."""
        dropout = mlx_compat.nn.Dropout1d(p=0.5)
        dropout.train()

        x = mlx_compat.randn(64, 32)  # (C, L)
        output = dropout(x)

        self.assertEqual(output.shape, (64, 32))

    def test_dropout1d_eval_mode(self):
        """Test Dropout1d in eval mode."""
        dropout = mlx_compat.nn.Dropout1d(p=0.5)
        dropout.eval()

        x = mlx_compat.randn(4, 64, 32)
        output = dropout(x)

        np.testing.assert_array_equal(output.numpy(), x.numpy())


@skipIfNoMLX
class TestDropout2d(TestCase):
    """Test nn.Dropout2d."""

    def test_dropout2d_creation(self):
        """Test Dropout2d creation."""
        dropout = mlx_compat.nn.Dropout2d(p=0.5)
        self.assertEqual(dropout.p, 0.5)

    def test_dropout2d_forward_4d(self):
        """Test Dropout2d forward pass with 4D input."""
        dropout = mlx_compat.nn.Dropout2d(p=0.5)
        dropout.train()

        x = mlx_compat.randn(4, 64, 8, 8)  # (N, C, H, W)
        output = dropout(x)

        self.assertEqual(output.shape, (4, 64, 8, 8))

    def test_dropout2d_forward_3d(self):
        """Test Dropout2d forward pass with 3D input."""
        dropout = mlx_compat.nn.Dropout2d(p=0.5)
        dropout.train()

        x = mlx_compat.randn(64, 8, 8)  # (C, H, W)
        output = dropout(x)

        self.assertEqual(output.shape, (64, 8, 8))

    def test_dropout2d_eval_mode(self):
        """Test Dropout2d in eval mode."""
        dropout = mlx_compat.nn.Dropout2d(p=0.5)
        dropout.eval()

        x = mlx_compat.randn(4, 64, 8, 8)
        output = dropout(x)

        np.testing.assert_array_equal(output.numpy(), x.numpy())


@skipIfNoMLX
class TestDropout3d(TestCase):
    """Test nn.Dropout3d."""

    def test_dropout3d_creation(self):
        """Test Dropout3d creation."""
        dropout = mlx_compat.nn.Dropout3d(p=0.5)
        self.assertEqual(dropout.p, 0.5)

    def test_dropout3d_forward_5d(self):
        """Test Dropout3d forward pass with 5D input."""
        dropout = mlx_compat.nn.Dropout3d(p=0.5)
        dropout.train()

        x = mlx_compat.randn(2, 16, 4, 4, 4)  # (N, C, D, H, W)
        output = dropout(x)

        self.assertEqual(output.shape, (2, 16, 4, 4, 4))

    def test_dropout3d_eval_mode(self):
        """Test Dropout3d in eval mode."""
        dropout = mlx_compat.nn.Dropout3d(p=0.5)
        dropout.eval()

        x = mlx_compat.randn(2, 16, 4, 4, 4)
        output = dropout(x)

        np.testing.assert_array_equal(output.numpy(), x.numpy())


@skipIfNoMLX
class TestAlphaDropout(TestCase):
    """Test nn.AlphaDropout."""

    def test_alpha_dropout_creation(self):
        """Test AlphaDropout creation."""
        dropout = mlx_compat.nn.AlphaDropout(p=0.5)
        self.assertEqual(dropout.p, 0.5)

    def test_alpha_dropout_forward(self):
        """Test AlphaDropout forward pass."""
        dropout = mlx_compat.nn.AlphaDropout(p=0.5)
        dropout.train()

        x = mlx_compat.randn(10, 10)
        output = dropout(x)

        self.assertEqual(output.shape, (10, 10))

    def test_alpha_dropout_eval_mode(self):
        """Test AlphaDropout in eval mode."""
        dropout = mlx_compat.nn.AlphaDropout(p=0.5)
        dropout.eval()

        x = mlx_compat.randn(10, 10)
        output = dropout(x)

        np.testing.assert_array_equal(output.numpy(), x.numpy())


@skipIfNoMLX
class TestFeatureAlphaDropout(TestCase):
    """Test nn.FeatureAlphaDropout."""

    def test_feature_alpha_dropout_creation(self):
        """Test FeatureAlphaDropout creation."""
        dropout = mlx_compat.nn.FeatureAlphaDropout(p=0.5)
        self.assertEqual(dropout.p, 0.5)

    def test_feature_alpha_dropout_forward(self):
        """Test FeatureAlphaDropout forward pass."""
        dropout = mlx_compat.nn.FeatureAlphaDropout(p=0.5)
        dropout.train()

        x = mlx_compat.randn(4, 64, 8, 8)
        output = dropout(x)

        self.assertEqual(output.shape, (4, 64, 8, 8))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
