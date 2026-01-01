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


@skipIfNoMLX
class TestBroadcastTo(TestCase):
    """Test broadcast_to operation (available via mlx_compat.broadcast_to)."""

    def test_broadcast_to_basic(self):
        """Test basic broadcast_to operation."""
        x = mlx_compat.tensor([[1], [2], [3]])
        y = mlx_compat.broadcast_to(x, (3, 4))
        self.assert_shape_equal(y.shape, (3, 4))
        # Each row should be broadcast
        self.assertEqual(y[0, 0].item(), 1)
        self.assertEqual(y[0, 3].item(), 1)
        self.assertEqual(y[2, 0].item(), 3)

    def test_broadcast_to_add_dims(self):
        """Test broadcast_to adding new dimensions."""
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.broadcast_to(x, (2, 3))
        self.assert_shape_equal(y.shape, (2, 3))


@skipIfNoMLX
class TestTile(TestCase):
    """Test tile operations (numpy-style repeat)."""

    def test_tile_basic(self):
        """Test basic tile operation."""
        x = mlx_compat.tensor([1, 2])
        y = mlx_compat.tile(x, (3,))
        self.assert_shape_equal(y.shape, (6,))
        expected = np.array([1, 2, 1, 2, 1, 2])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_tile_2d(self):
        """Test tile on 2D tensor."""
        x = mlx_compat.tensor([[1, 2], [3, 4]])
        y = mlx_compat.tile(x, (2, 3))
        self.assert_shape_equal(y.shape, (4, 6))


@skipIfNoMLX
class TestStack(TestCase):
    """Test stack operations."""

    def test_stack_default_dim(self):
        """Test stack along default dimension."""
        a = mlx_compat.tensor([1, 2, 3])
        b = mlx_compat.tensor([4, 5, 6])
        result = mlx_compat.stack([a, b])
        self.assert_shape_equal(result.shape, (2, 3))

    def test_stack_dim1(self):
        """Test stack along dimension 1."""
        a = mlx_compat.tensor([1, 2, 3])
        b = mlx_compat.tensor([4, 5, 6])
        result = mlx_compat.stack([a, b], dim=1)
        self.assert_shape_equal(result.shape, (3, 2))

    def test_stack_3d(self):
        """Test stack 2D tensors into 3D."""
        a = mlx_compat.randn(3, 4)
        b = mlx_compat.randn(3, 4)
        c = mlx_compat.randn(3, 4)
        result = mlx_compat.stack([a, b, c], dim=0)
        self.assert_shape_equal(result.shape, (3, 3, 4))


@skipIfNoMLX
class TestCat(TestCase):
    """Test concatenate operations."""

    def test_cat_1d(self):
        """Test cat on 1D tensors."""
        a = mlx_compat.tensor([1, 2])
        b = mlx_compat.tensor([3, 4, 5])
        result = mlx_compat.cat([a, b])
        self.assert_shape_equal(result.shape, (5,))
        np.testing.assert_array_equal(result.numpy(), [1, 2, 3, 4, 5])

    def test_cat_2d_dim0(self):
        """Test cat on 2D tensors along dim 0."""
        a = mlx_compat.randn(2, 3)
        b = mlx_compat.randn(4, 3)
        result = mlx_compat.cat([a, b], dim=0)
        self.assert_shape_equal(result.shape, (6, 3))

    def test_cat_2d_dim1(self):
        """Test cat on 2D tensors along dim 1."""
        a = mlx_compat.randn(3, 2)
        b = mlx_compat.randn(3, 4)
        result = mlx_compat.cat([a, b], dim=1)
        self.assert_shape_equal(result.shape, (3, 6))


@skipIfNoMLX
class TestSplit(TestCase):
    """Test split operations."""

    def test_split_equal(self):
        """Test split into equal parts."""
        x = mlx_compat.arange(12).reshape(4, 3)
        parts = mlx_compat.split(x, 2, dim=0)
        self.assertEqual(len(parts), 2)
        self.assert_shape_equal(parts[0].shape, (2, 3))
        self.assert_shape_equal(parts[1].shape, (2, 3))

    def test_split_sizes(self):
        """Test split with specific sizes."""
        x = mlx_compat.arange(10)
        parts = mlx_compat.split(x, [2, 3, 5])
        self.assertEqual(len(parts), 3)
        self.assert_shape_equal(parts[0].shape, (2,))
        self.assert_shape_equal(parts[1].shape, (3,))
        self.assert_shape_equal(parts[2].shape, (5,))


@skipIfNoMLX
class TestChunk(TestCase):
    """Test chunk operations."""

    def test_chunk_equal(self):
        """Test chunk into equal parts."""
        x = mlx_compat.arange(12)
        chunks = mlx_compat.chunk(x, 4)
        self.assertEqual(len(chunks), 4)
        for chunk in chunks:
            self.assert_shape_equal(chunk.shape, (3,))

    def test_chunk_unequal(self):
        """Test chunk when size doesn't divide evenly."""
        x = mlx_compat.arange(10)
        chunks = mlx_compat.chunk(x, 3)
        # Should be [4, 4, 2] or similar
        self.assertEqual(len(chunks), 3)


@skipIfNoMLX
class TestNarrow(TestCase):
    """Test narrow operation."""

    def test_narrow_basic(self):
        """Test basic narrow operation."""
        x = mlx_compat.arange(10)
        y = mlx_compat.narrow(x, 0, 2, 5)
        self.assert_shape_equal(y.shape, (5,))
        np.testing.assert_array_equal(y.numpy(), [2, 3, 4, 5, 6])

    def test_narrow_2d(self):
        """Test narrow on 2D tensor."""
        x = mlx_compat.arange(20).reshape(4, 5)
        y = mlx_compat.narrow(x, 0, 1, 2)
        self.assert_shape_equal(y.shape, (2, 5))


@skipIfNoMLX
class TestMoveaxis(TestCase):
    """Test moveaxis/swapaxes operations."""

    def test_moveaxis_single(self):
        """Test moveaxis single dimension."""
        x = mlx_compat.arange(24).reshape(2, 3, 4)
        y = mlx_compat.moveaxis(x, 0, -1)
        self.assert_shape_equal(y.shape, (3, 4, 2))

    def test_swapaxes(self):
        """Test swapaxes operation."""
        x = mlx_compat.arange(12).reshape(3, 4)
        y = mlx_compat.swapaxes(x, 0, 1)
        self.assert_shape_equal(y.shape, (4, 3))


@skipIfNoMLX
class TestFlip(TestCase):
    """Test flip operations."""

    def test_flip_1d(self):
        """Test flip on 1D tensor."""
        x = mlx_compat.tensor([1, 2, 3, 4, 5])
        y = mlx_compat.flip(x, [0])
        np.testing.assert_array_equal(y.numpy(), [5, 4, 3, 2, 1])

    def test_flip_2d_both(self):
        """Test flip on both dimensions."""
        x = mlx_compat.arange(6).reshape(2, 3)
        y = mlx_compat.flip(x, [0, 1])
        expected = np.array([[5, 4, 3], [2, 1, 0]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_fliplr(self):
        """Test fliplr (horizontal flip)."""
        x = mlx_compat.tensor([[1, 2, 3], [4, 5, 6]])
        y = mlx_compat.fliplr(x)
        expected = np.array([[3, 2, 1], [6, 5, 4]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_flipud(self):
        """Test flipud (vertical flip)."""
        x = mlx_compat.tensor([[1, 2], [3, 4], [5, 6]])
        y = mlx_compat.flipud(x)
        expected = np.array([[5, 6], [3, 4], [1, 2]])
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestRollRot90(TestCase):
    """Test roll and rotation operations."""

    def test_roll_1d(self):
        """Test roll on 1D tensor."""
        x = mlx_compat.tensor([1, 2, 3, 4, 5])
        y = mlx_compat.roll(x, 2)
        np.testing.assert_array_equal(y.numpy(), [4, 5, 1, 2, 3])

    def test_roll_negative(self):
        """Test roll with negative shift."""
        x = mlx_compat.tensor([1, 2, 3, 4, 5])
        y = mlx_compat.roll(x, -2)
        np.testing.assert_array_equal(y.numpy(), [3, 4, 5, 1, 2])

    def test_rot90_basic(self):
        """Test 90 degree rotation."""
        x = mlx_compat.tensor([[1, 2], [3, 4]])
        y = mlx_compat.rot90(x, 1)
        expected = np.array([[2, 4], [1, 3]])
        np.testing.assert_array_equal(y.numpy(), expected)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
