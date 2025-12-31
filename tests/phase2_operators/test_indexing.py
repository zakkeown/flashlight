"""
Test Phase 2: Indexing Operations

Tests indexing operations (where, masked_fill, index_select, nonzero, etc.)
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
class TestIndexSelect(TestCase):
    """Test index_select operation."""

    def test_index_select_1d(self):
        """Test index_select on 1D tensor."""
        x = mlx_compat.tensor([10.0, 20.0, 30.0, 40.0])
        idx = mlx_compat.tensor([0, 2, 3], dtype=mlx_compat.int32)
        result = mlx_compat.index_select(x, 0, idx)
        np.testing.assert_array_equal(result.numpy(), np.array([10.0, 30.0, 40.0]))

    def test_index_select_2d(self):
        """Test index_select on 2D tensor."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        idx = mlx_compat.tensor([0, 2], dtype=mlx_compat.int32)
        result = mlx_compat.index_select(x, 0, idx)
        expected = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_index_select_dim1(self):
        """Test index_select along dim 1."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = mlx_compat.tensor([0, 2], dtype=mlx_compat.int32)
        result = mlx_compat.index_select(x, 1, idx)
        expected = np.array([[1.0, 3.0], [4.0, 6.0]])
        np.testing.assert_array_equal(result.numpy(), expected)


@skipIfNoMLX
class TestWhere(TestCase):
    """Test where operation."""

    def test_where_tensors(self):
        """Test where with tensor inputs."""
        condition = mlx_compat.tensor([True, False, True, False])
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y = mlx_compat.tensor([10.0, 20.0, 30.0, 40.0])
        result = mlx_compat.where(condition, x, y)
        expected = np.array([1.0, 20.0, 3.0, 40.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_where_scalars(self):
        """Test where with scalar inputs."""
        condition = mlx_compat.tensor([True, False, True])
        result = mlx_compat.where(condition, 1.0, 0.0)
        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_where_mixed(self):
        """Test where with tensor and scalar."""
        condition = mlx_compat.tensor([True, False, True])
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        result = mlx_compat.where(condition, x, 0.0)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_where_broadcasting(self):
        """Test where with broadcasting."""
        condition = mlx_compat.tensor([[True, False], [False, True]])
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[10.0, 20.0], [30.0, 40.0]])
        result = mlx_compat.where(condition, x, y)
        expected = np.array([[1.0, 20.0], [30.0, 4.0]])
        np.testing.assert_array_equal(result.numpy(), expected)


@skipIfNoMLX
class TestMaskedFill(TestCase):
    """Test masked_fill operation."""

    def test_masked_fill_1d(self):
        """Test masked_fill on 1D tensor."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        mask = mlx_compat.tensor([True, False, True, False])
        result = mlx_compat.masked_fill(x, mask, -1.0)
        expected = np.array([-1.0, 2.0, -1.0, 4.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_masked_fill_2d(self):
        """Test masked_fill on 2D tensor."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = mlx_compat.tensor([[True, False], [False, True]])
        result = mlx_compat.masked_fill(x, mask, 0.0)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_masked_fill_requires_grad(self):
        """Test that masked_fill propagates requires_grad."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        mask = mlx_compat.tensor([True, False, True])
        result = mlx_compat.masked_fill(x, mask, 0.0)
        self.assertTrue(result.requires_grad)


@skipIfNoMLX
class TestMaskedSelect(TestCase):
    """Test masked_select operation."""

    def test_masked_select_1d(self):
        """Test masked_select on 1D tensor."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        mask = mlx_compat.tensor([True, False, True, True])
        result = mlx_compat.masked_select(x, mask)
        expected = np.array([1.0, 3.0, 4.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_masked_select_2d(self):
        """Test masked_select on 2D tensor (returns 1D)."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = mlx_compat.tensor([[True, False], [False, True]])
        result = mlx_compat.masked_select(x, mask)
        # Should return flattened selected elements
        expected = np.array([1.0, 4.0])
        np.testing.assert_array_equal(result.numpy(), expected)


@skipIfNoMLX
class TestNonzero(TestCase):
    """Test nonzero operation."""

    def test_nonzero_1d(self):
        """Test nonzero on 1D tensor."""
        x = mlx_compat.tensor([0.0, 1.0, 0.0, 2.0])
        result = mlx_compat.nonzero(x)
        # Should return 2D array of indices: [[1], [3]]
        self.assert_shape_equal(result.shape, (2, 1))
        np.testing.assert_array_equal(result.numpy(), np.array([[1], [3]]))

    def test_nonzero_2d(self):
        """Test nonzero on 2D tensor."""
        x = mlx_compat.tensor([[0.0, 1.0], [2.0, 0.0]])
        result = mlx_compat.nonzero(x)
        # Should return indices where elements are non-zero
        self.assert_shape_equal(result.shape, (2, 2))

    def test_nonzero_as_tuple(self):
        """Test nonzero with as_tuple=True."""
        x = mlx_compat.tensor([[0.0, 1.0], [2.0, 0.0]])
        result = mlx_compat.nonzero(x, as_tuple=True)
        self.assertEqual(len(result), 2)  # One tensor per dimension
        # First tensor: row indices [0, 1]
        # Second tensor: col indices [1, 0]


@skipIfNoMLX
class TestTake(TestCase):
    """Test take operation."""

    def test_take_1d(self):
        """Test take on 1D tensor."""
        x = mlx_compat.tensor([10.0, 20.0, 30.0, 40.0])
        idx = mlx_compat.tensor([0, 2, 3], dtype=mlx_compat.int32)
        result = mlx_compat.take(x, idx)
        expected = np.array([10.0, 30.0, 40.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_take_2d_flat(self):
        """Test take on 2D tensor with flat indices."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        idx = mlx_compat.tensor([0, 2, 3], dtype=mlx_compat.int32)
        result = mlx_compat.take(x, idx)
        # Flatten: [1, 2, 3, 4], take indices 0, 2, 3
        expected = np.array([1.0, 3.0, 4.0])
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
