"""
Test Phase 1: View Semantics

Tests view operations (reshape, transpose, permute, squeeze, etc.)
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
class TestReshape(TestCase):
    """Test reshape and view operations."""

    def test_reshape_function(self):
        """Test reshape function."""
        x = mlx_compat.arange(12)
        y = mlx_compat.reshape(x, (3, 4))
        self.assert_shape_equal(y.shape, (3, 4))
        self.assertEqual(y.numel, 12)

    def test_reshape_method(self):
        """Test reshape as tensor method."""
        x = mlx_compat.arange(12)
        y = x.reshape(3, 4)
        self.assert_shape_equal(y.shape, (3, 4))

    def test_reshape_tuple_arg(self):
        """Test reshape with tuple argument."""
        x = mlx_compat.arange(12)
        y = x.reshape((3, 4))
        self.assert_shape_equal(y.shape, (3, 4))

    def test_reshape_inferred_dim(self):
        """Test reshape with -1 (inferred dimension)."""
        x = mlx_compat.arange(12)
        y = x.reshape(-1, 4)
        self.assert_shape_equal(y.shape, (3, 4))

    def test_view_method(self):
        """Test view method (alias for reshape)."""
        x = mlx_compat.arange(12)
        y = x.view(3, 4)
        self.assert_shape_equal(y.shape, (3, 4))

    def test_view_is_view(self):
        """Test that reshape creates a view."""
        x = mlx_compat.arange(12)
        y = x.reshape(3, 4)
        self.assertTrue(y.is_view)
        self.assertIsNotNone(y._base)


@skipIfNoMLX
class TestTranspose(TestCase):
    """Test transpose operations."""

    def test_transpose_function(self):
        """Test transpose function."""
        x = mlx_compat.arange(12).reshape(3, 4)
        y = mlx_compat.transpose(x, 0, 1)
        self.assert_shape_equal(y.shape, (4, 3))

    def test_transpose_method(self):
        """Test transpose as method."""
        x = mlx_compat.arange(12).reshape(3, 4)
        y = x.transpose(0, 1)
        self.assert_shape_equal(y.shape, (4, 3))

    def test_t_method_2d(self):
        """Test .t() method for 2D tensors."""
        x = mlx_compat.arange(12).reshape(3, 4)
        y = x.t()
        self.assert_shape_equal(y.shape, (4, 3))

    def test_t_method_non_2d_raises(self):
        """Test that .t() raises for non-2D tensors."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        with self.assertRaises(RuntimeError):
            x.t()

    def test_transpose_3d(self):
        """Test transpose on 3D tensor."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = x.transpose(0, 2)
        self.assert_shape_equal(y.shape, (4, 3, 2))


@skipIfNoMLX
class TestPermute(TestCase):
    """Test permute operations."""

    def test_permute_function(self):
        """Test permute function."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = mlx_compat.permute(x, 2, 0, 1)
        self.assert_shape_equal(y.shape, (4, 2, 3))

    def test_permute_method(self):
        """Test permute as method."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = x.permute(2, 0, 1)
        self.assert_shape_equal(y.shape, (4, 2, 3))

    def test_permute_tuple_arg(self):
        """Test permute with tuple argument."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = x.permute((2, 0, 1))
        self.assert_shape_equal(y.shape, (4, 2, 3))

    def test_permute_all_dims(self):
        """Test permuting all dimensions."""
        x = mlx_compat.arange(120).reshape(2, 3, 4, 5)
        y = x.permute(3, 1, 0, 2)
        self.assert_shape_equal(y.shape, (5, 3, 2, 4))


@skipIfNoMLX
class TestSqueeze(TestCase):
    """Test squeeze/unsqueeze operations."""

    def test_squeeze_all(self):
        """Test squeeze all size-1 dimensions."""
        x = mlx_compat.zeros(1, 3, 1, 4)
        y = mlx_compat.squeeze(x)
        self.assert_shape_equal(y.shape, (3, 4))

    def test_squeeze_specific_dim(self):
        """Test squeeze specific dimension."""
        x = mlx_compat.zeros(1, 3, 1, 4)
        y = mlx_compat.squeeze(x, 0)
        self.assert_shape_equal(y.shape, (3, 1, 4))

    def test_squeeze_method(self):
        """Test squeeze as method."""
        x = mlx_compat.zeros(1, 3, 1, 4)
        y = x.squeeze()
        self.assert_shape_equal(y.shape, (3, 4))

    def test_squeeze_no_op(self):
        """Test squeeze on dimension that's not size 1."""
        x = mlx_compat.zeros(1, 3, 4)
        y = x.squeeze(1)  # Dimension 1 is size 3, not 1
        self.assert_shape_equal(y.shape, (1, 3, 4))  # No change

    def test_unsqueeze_function(self):
        """Test unsqueeze function."""
        x = mlx_compat.zeros(3, 4)
        y = mlx_compat.unsqueeze(x, 0)
        self.assert_shape_equal(y.shape, (1, 3, 4))

    def test_unsqueeze_method(self):
        """Test unsqueeze as method."""
        x = mlx_compat.zeros(3, 4)
        y = x.unsqueeze(1)
        self.assert_shape_equal(y.shape, (3, 1, 4))

    def test_unsqueeze_end(self):
        """Test unsqueeze at end."""
        x = mlx_compat.zeros(3, 4)
        y = x.unsqueeze(2)
        self.assert_shape_equal(y.shape, (3, 4, 1))


@skipIfNoMLX
class TestFlatten(TestCase):
    """Test flatten operation."""

    def test_flatten_all(self):
        """Test flatten all dimensions."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = mlx_compat.flatten(x)
        self.assert_shape_equal(y.shape, (24,))

    def test_flatten_method(self):
        """Test flatten as method."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = x.flatten()
        self.assert_shape_equal(y.shape, (24,))

    def test_flatten_range(self):
        """Test flatten specific range of dimensions."""
        x = mlx_compat.arange(120).reshape(2, 3, 4, 5)
        y = x.flatten(1, 2)
        self.assert_shape_equal(y.shape, (2, 12, 5))

    def test_flatten_start_only(self):
        """Test flatten from start_dim to end."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = x.flatten(1)
        self.assert_shape_equal(y.shape, (2, 12))

    def test_flatten_negative_indices(self):
        """Test flatten with negative indices."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = x.flatten(0, -1)
        self.assert_shape_equal(y.shape, (24,))


@skipIfNoMLX
class TestContiguous(TestCase):
    """Test contiguous operation."""

    def test_contiguous(self):
        """Test contiguous (no-op in MLX)."""
        x = mlx_compat.arange(12).reshape(3, 4)
        y = mlx_compat.contiguous(x)
        self.assert_shape_equal(y.shape, x.shape)

    def test_contiguous_method(self):
        """Test contiguous as method."""
        x = mlx_compat.arange(12).reshape(3, 4)
        y = x.contiguous()
        self.assert_shape_equal(y.shape, x.shape)


@skipIfNoMLX
class TestViewTracking(TestCase):
    """Test view tracking for autograd."""

    def test_view_tracks_base(self):
        """Test that views track their base tensor."""
        x = mlx_compat.arange(12)
        y = x.reshape(3, 4)
        self.assertTrue(y.is_view)
        self.assertIs(y._base, x)

    def test_view_of_view_tracks_original_base(self):
        """Test that view of view tracks original base."""
        x = mlx_compat.arange(12)
        y = x.reshape(3, 4)
        z = y.transpose(0, 1)
        self.assertTrue(z.is_view)
        self.assertIs(z._base, x)  # Should track original base

    def test_view_preserves_requires_grad(self):
        """Test that views preserve requires_grad."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = x.reshape(2, 2)
        self.assertTrue(y.requires_grad)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
