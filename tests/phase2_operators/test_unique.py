"""
Test unique and unique_consecutive operations.
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@skipIfNoMLX
class TestUnique(TestCase):
    """Test unique operation."""

    def test_unique_basic(self):
        """Test basic unique operation."""
        x = mlx_compat.tensor([1., 2., 1., 3., 2., 1.])
        result = mlx_compat.unique(x)
        expected = np.array([1., 2., 3.])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unique_already_unique(self):
        """Test unique on already unique tensor."""
        x = mlx_compat.tensor([1., 2., 3., 4., 5.])
        result = mlx_compat.unique(x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())

    def test_unique_2d(self):
        """Test unique on 2D tensor (flattened)."""
        x = mlx_compat.tensor([[1., 2.], [2., 3.], [1., 3.]])
        result = mlx_compat.unique(x)
        expected = np.array([1., 2., 3.])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unique_return_inverse(self):
        """Test unique with return_inverse."""
        x = mlx_compat.tensor([1., 2., 1., 3., 2., 1.])
        result, inverse = mlx_compat.unique(x, return_inverse=True)
        expected_unique = np.array([1., 2., 3.])
        np.testing.assert_array_equal(result.numpy(), expected_unique)
        # Check that we can reconstruct original from inverse
        reconstructed = result.numpy()[inverse.numpy().astype(int)]
        np.testing.assert_array_equal(reconstructed, x.numpy())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_unique_parity(self):
        """Test parity with PyTorch unique."""
        data = np.array([3., 1., 4., 1., 5., 9., 2., 6., 5., 3.], dtype=np.float32)

        mlx_x = mlx_compat.tensor(data)
        torch_x = torch.tensor(data)

        mlx_result = mlx_compat.unique(mlx_x)
        torch_result = torch.unique(torch_x)

        np.testing.assert_allclose(
            np.sort(mlx_result.numpy()),
            np.sort(torch_result.numpy()),
            rtol=1e-5
        )


@skipIfNoMLX
class TestUniqueConsecutive(TestCase):
    """Test unique_consecutive operation."""

    def test_unique_consecutive_basic(self):
        """Test basic unique_consecutive."""
        x = mlx_compat.tensor([1., 1., 2., 2., 2., 3., 1., 1.])
        result = mlx_compat.unique_consecutive(x)
        expected = np.array([1., 2., 3., 1.])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unique_consecutive_no_dups(self):
        """Test unique_consecutive with no consecutive duplicates."""
        x = mlx_compat.tensor([1., 2., 3., 4., 5.])
        result = mlx_compat.unique_consecutive(x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())

    def test_unique_consecutive_return_inverse(self):
        """Test unique_consecutive with return_inverse."""
        x = mlx_compat.tensor([1., 1., 2., 2., 3., 3., 3.])
        result, inverse = mlx_compat.unique_consecutive(x, return_inverse=True)
        expected_unique = np.array([1., 2., 3.])
        expected_inverse = np.array([0, 0, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(result.numpy(), expected_unique)
        np.testing.assert_array_equal(inverse.numpy(), expected_inverse)

    def test_unique_consecutive_return_counts(self):
        """Test unique_consecutive with return_counts."""
        x = mlx_compat.tensor([1., 1., 2., 2., 2., 3.])
        result, counts = mlx_compat.unique_consecutive(x, return_counts=True)
        expected_unique = np.array([1., 2., 3.])
        expected_counts = np.array([2, 3, 1])
        np.testing.assert_array_equal(result.numpy(), expected_unique)
        np.testing.assert_array_equal(counts.numpy(), expected_counts)


if __name__ == '__main__':
    unittest.main()
