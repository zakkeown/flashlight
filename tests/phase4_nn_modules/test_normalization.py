"""
Test Phase 4: Normalization Layers

Tests the nn.layers.normalization module:
- BatchNorm1d, BatchNorm2d, BatchNorm3d
- LayerNorm
- GroupNorm
- InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
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
class TestBatchNorm1d(TestCase):
    """Test nn.BatchNorm1d."""

    def test_batchnorm1d_creation(self):
        """Test BatchNorm1d creation with default parameters."""
        bn = flashlight.nn.BatchNorm1d(64)
        self.assertEqual(bn.num_features, 64)
        self.assertEqual(bn.eps, 1e-5)
        self.assertEqual(bn.momentum, 0.1)
        self.assertTrue(bn.affine)
        self.assertTrue(bn.track_running_stats)

    def test_batchnorm1d_forward_2d(self):
        """Test BatchNorm1d forward pass with 2D input [N, C]."""
        bn = flashlight.nn.BatchNorm1d(32)
        bn.train()
        x = flashlight.randn(8, 32)
        output = bn(x)
        self.assertEqual(output.shape, (8, 32))

    def test_batchnorm1d_forward_3d(self):
        """Test BatchNorm1d forward pass with 3D input [N, C, L]."""
        bn = flashlight.nn.BatchNorm1d(32)
        bn.train()
        x = flashlight.randn(8, 32, 16)
        output = bn(x)
        self.assertEqual(output.shape, (8, 32, 16))

    def test_batchnorm1d_eval_mode(self):
        """Test BatchNorm1d in eval mode uses running stats."""
        bn = flashlight.nn.BatchNorm1d(16)
        # Do a forward pass in training mode first
        bn.train()
        x = flashlight.randn(4, 16)
        _ = bn(x)
        # Now eval mode
        bn.eval()
        y = flashlight.randn(4, 16)
        output = bn(y)
        self.assertEqual(output.shape, (4, 16))

    def test_batchnorm1d_no_affine(self):
        """Test BatchNorm1d without affine parameters."""
        bn = flashlight.nn.BatchNorm1d(32, affine=False)
        self.assertIsNone(bn.weight)
        self.assertIsNone(bn.bias)
        x = flashlight.randn(8, 32)
        output = bn(x)
        self.assertEqual(output.shape, (8, 32))


@skipIfNoMLX
class TestBatchNorm2d(TestCase):
    """Test nn.BatchNorm2d."""

    def test_batchnorm2d_creation(self):
        """Test BatchNorm2d creation with default parameters."""
        bn = flashlight.nn.BatchNorm2d(64)
        self.assertEqual(bn.num_features, 64)
        self.assertTrue(bn.affine)
        self.assertTrue(bn.track_running_stats)

    def test_batchnorm2d_forward_train(self):
        """Test BatchNorm2d forward pass in training mode."""
        bn = flashlight.nn.BatchNorm2d(64)
        bn.train()
        x = flashlight.randn(4, 64, 8, 8)
        output = bn(x)
        self.assertEqual(output.shape, (4, 64, 8, 8))

    def test_batchnorm2d_forward_eval(self):
        """Test BatchNorm2d forward pass in eval mode."""
        bn = flashlight.nn.BatchNorm2d(64)
        # Train first to populate running stats
        bn.train()
        x = flashlight.randn(4, 64, 8, 8)
        _ = bn(x)
        # Eval mode
        bn.eval()
        y = flashlight.randn(4, 64, 8, 8)
        output = bn(y)
        self.assertEqual(output.shape, (4, 64, 8, 8))

    def test_batchnorm2d_no_affine(self):
        """Test BatchNorm2d without affine parameters."""
        bn = flashlight.nn.BatchNorm2d(64, affine=False)
        self.assertIsNone(bn.weight)
        self.assertIsNone(bn.bias)
        x = flashlight.randn(4, 64, 8, 8)
        output = bn(x)
        self.assertEqual(output.shape, (4, 64, 8, 8))

    def test_batchnorm2d_running_stats_update(self):
        """Test that running stats are updated during training."""
        bn = flashlight.nn.BatchNorm2d(16)
        bn.train()
        # Initial running mean should be zeros
        initial_mean = bn.running_mean.numpy().copy()
        x = flashlight.randn(4, 16, 4, 4) + 2.0  # Add offset to create non-zero mean
        _ = bn(x)
        updated_mean = bn.running_mean.numpy()
        # Running mean should have changed
        self.assertFalse(np.allclose(initial_mean, updated_mean))


@skipIfNoMLX
class TestBatchNorm3d(TestCase):
    """Test nn.BatchNorm3d."""

    def test_batchnorm3d_creation(self):
        """Test BatchNorm3d creation with default parameters."""
        bn = flashlight.nn.BatchNorm3d(32)
        self.assertEqual(bn.num_features, 32)

    def test_batchnorm3d_forward(self):
        """Test BatchNorm3d forward pass."""
        bn = flashlight.nn.BatchNorm3d(16)
        bn.train()
        x = flashlight.randn(2, 16, 4, 4, 4)
        output = bn(x)
        self.assertEqual(output.shape, (2, 16, 4, 4, 4))

    def test_batchnorm3d_eval(self):
        """Test BatchNorm3d in eval mode."""
        bn = flashlight.nn.BatchNorm3d(16)
        bn.train()
        x = flashlight.randn(2, 16, 4, 4, 4)
        _ = bn(x)
        bn.eval()
        y = flashlight.randn(2, 16, 4, 4, 4)
        output = bn(y)
        self.assertEqual(output.shape, (2, 16, 4, 4, 4))


@skipIfNoMLX
class TestLayerNorm(TestCase):
    """Test nn.LayerNorm."""

    def test_layernorm_creation(self):
        """Test LayerNorm creation with default parameters."""
        ln = flashlight.nn.LayerNorm(128)
        self.assertEqual(ln.normalized_shape, (128,))
        self.assertEqual(ln.eps, 1e-5)
        self.assertTrue(ln.elementwise_affine)

    def test_layernorm_forward_1d(self):
        """Test LayerNorm forward pass with 1D normalized shape."""
        ln = flashlight.nn.LayerNorm(64)
        x = flashlight.randn(4, 10, 64)
        output = ln(x)
        self.assertEqual(output.shape, (4, 10, 64))

    def test_layernorm_forward_2d(self):
        """Test LayerNorm forward pass with 2D input."""
        ln = flashlight.nn.LayerNorm(64)
        x = flashlight.randn(8, 64)
        output = ln(x)
        self.assertEqual(output.shape, (8, 64))

    def test_layernorm_no_affine(self):
        """Test LayerNorm without affine parameters."""
        ln = flashlight.nn.LayerNorm(64, elementwise_affine=False)
        self.assertIsNone(ln.weight)
        self.assertIsNone(ln.bias)
        x = flashlight.randn(4, 64)
        output = ln(x)
        self.assertEqual(output.shape, (4, 64))

    def test_layernorm_parameters(self):
        """Test LayerNorm has correct parameters."""
        ln = flashlight.nn.LayerNorm(64)
        self.assertIsNotNone(ln.weight)
        self.assertIsNotNone(ln.bias)
        self.assertEqual(ln.weight.shape, (64,))
        self.assertEqual(ln.bias.shape, (64,))

    def test_layernorm_normalization(self):
        """Test that LayerNorm actually normalizes the output."""
        ln = flashlight.nn.LayerNorm(64, elementwise_affine=False)
        x = flashlight.randn(4, 64) * 5 + 10  # Non-zero mean and high variance
        output = ln(x)
        # Mean should be close to 0, var close to 1 per sample
        output_np = output.numpy()
        means = output_np.mean(axis=-1)
        vars_ = output_np.var(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(vars_, 1.0, atol=1e-4)


@skipIfNoMLX
class TestGroupNorm(TestCase):
    """Test nn.GroupNorm."""

    def test_groupnorm_creation(self):
        """Test GroupNorm creation with default parameters."""
        gn = flashlight.nn.GroupNorm(4, 64)
        self.assertEqual(gn.num_groups, 4)
        self.assertEqual(gn.num_channels, 64)
        self.assertTrue(gn.affine)

    def test_groupnorm_forward(self):
        """Test GroupNorm forward pass."""
        gn = flashlight.nn.GroupNorm(4, 64)
        x = flashlight.randn(2, 64, 8, 8)
        output = gn(x)
        self.assertEqual(output.shape, (2, 64, 8, 8))

    def test_groupnorm_no_affine(self):
        """Test GroupNorm without affine parameters."""
        gn = flashlight.nn.GroupNorm(4, 64, affine=False)
        self.assertIsNone(gn.weight)
        self.assertIsNone(gn.bias)

    def test_groupnorm_invalid_groups(self):
        """Test GroupNorm with invalid num_groups."""
        with self.assertRaises(ValueError):
            flashlight.nn.GroupNorm(3, 64)  # 64 not divisible by 3


@skipIfNoMLX
class TestInstanceNorm1d(TestCase):
    """Test nn.InstanceNorm1d."""

    def test_instancenorm1d_creation(self):
        """Test InstanceNorm1d creation."""
        inn = flashlight.nn.InstanceNorm1d(64)
        self.assertEqual(inn.num_features, 64)

    def test_instancenorm1d_forward(self):
        """Test InstanceNorm1d forward pass."""
        inn = flashlight.nn.InstanceNorm1d(32)
        x = flashlight.randn(4, 32, 16)
        output = inn(x)
        self.assertEqual(output.shape, (4, 32, 16))


@skipIfNoMLX
class TestInstanceNorm2d(TestCase):
    """Test nn.InstanceNorm2d."""

    def test_instancenorm2d_creation(self):
        """Test InstanceNorm2d creation."""
        inn = flashlight.nn.InstanceNorm2d(64)
        self.assertEqual(inn.num_features, 64)

    def test_instancenorm2d_forward(self):
        """Test InstanceNorm2d forward pass."""
        inn = flashlight.nn.InstanceNorm2d(64)
        x = flashlight.randn(4, 64, 8, 8)
        output = inn(x)
        self.assertEqual(output.shape, (4, 64, 8, 8))


@skipIfNoMLX
class TestInstanceNorm3d(TestCase):
    """Test nn.InstanceNorm3d."""

    def test_instancenorm3d_creation(self):
        """Test InstanceNorm3d creation."""
        inn = flashlight.nn.InstanceNorm3d(32)
        self.assertEqual(inn.num_features, 32)

    def test_instancenorm3d_forward(self):
        """Test InstanceNorm3d forward pass."""
        inn = flashlight.nn.InstanceNorm3d(16)
        x = flashlight.randn(2, 16, 4, 4, 4)
        output = inn(x)
        self.assertEqual(output.shape, (2, 16, 4, 4, 4))


@skipIfNoMLX
class TestLocalResponseNorm(TestCase):
    """Test nn.LocalResponseNorm."""

    def test_creation(self):
        """Test LocalResponseNorm creation."""
        lrn = flashlight.nn.LocalResponseNorm(size=5)
        self.assertEqual(lrn.size, 5)

    def test_creation_with_params(self):
        """Test LocalResponseNorm creation with custom parameters."""
        lrn = flashlight.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        self.assertEqual(lrn.alpha, 1e-4)
        self.assertEqual(lrn.beta, 0.75)

    def test_forward_shape(self):
        """Test LocalResponseNorm forward pass."""
        lrn = flashlight.nn.LocalResponseNorm(size=5)
        x = flashlight.randn(4, 32, 8, 8)
        output = lrn(x)
        self.assertEqual(output.shape, (4, 32, 8, 8))


@skipIfNoMLX
class TestCrossMapLRN2d(TestCase):
    """Test nn.CrossMapLRN2d."""

    def test_creation(self):
        """Test CrossMapLRN2d creation."""
        lrn = flashlight.nn.CrossMapLRN2d(size=5, alpha=1e-4, beta=0.75)
        self.assertEqual(lrn.size, 5)

    def test_forward_shape(self):
        """Test CrossMapLRN2d forward pass."""
        lrn = flashlight.nn.CrossMapLRN2d(size=5)
        x = flashlight.randn(4, 32, 8, 8)
        output = lrn(x)
        self.assertEqual(output.shape, (4, 32, 8, 8))

    def test_forward_small_channels(self):
        """Test CrossMapLRN2d with few channels."""
        lrn = flashlight.nn.CrossMapLRN2d(size=3)
        x = flashlight.randn(2, 4, 4, 4)
        output = lrn(x)
        self.assertEqual(output.shape, (2, 4, 4, 4))


@skipIfNoMLX
class TestRMSNorm(TestCase):
    """Test nn.RMSNorm."""

    def test_creation(self):
        """Test RMSNorm creation."""
        rms = flashlight.nn.RMSNorm(normalized_shape=64)
        self.assertIsNotNone(rms)

    def test_forward_shape_3d(self):
        """Test RMSNorm forward pass with 3D input."""
        rms = flashlight.nn.RMSNorm(normalized_shape=64)
        x = flashlight.randn(4, 10, 64)
        output = rms(x)
        self.assertEqual(output.shape, (4, 10, 64))

    def test_forward_shape_2d(self):
        """Test RMSNorm with 2D input."""
        rms = flashlight.nn.RMSNorm(normalized_shape=64)
        x = flashlight.randn(4, 64)
        output = rms(x)
        self.assertEqual(output.shape, (4, 64))


@skipIfNoMLX
class TestSyncBatchNorm(TestCase):
    """Test nn.SyncBatchNorm."""

    def test_creation(self):
        """Test SyncBatchNorm creation."""
        bn = flashlight.nn.SyncBatchNorm(num_features=32)
        self.assertEqual(bn.num_features, 32)

    def test_forward_shape(self):
        """Test SyncBatchNorm forward pass (same as BatchNorm in single-device)."""
        bn = flashlight.nn.SyncBatchNorm(num_features=32)
        x = flashlight.randn(4, 32, 8, 8)
        output = bn(x)
        self.assertEqual(output.shape, (4, 32, 8, 8))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
