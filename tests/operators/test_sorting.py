"""
Test Phase 2: Sorting and Selection Operators

Tests sorting operations (sort, argsort, topk, kthvalue, msort)
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np

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
class TestSort(TestCase):
    """Test sort operations."""

    def test_sort_1d(self):
        """Test sorting a 1D tensor."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        values, indices = flashlight.sort(x)
        expected_values = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
        np.testing.assert_array_equal(values.numpy(), expected_values)
        # Check indices map back correctly
        for i, idx in enumerate(indices.numpy()):
            self.assertAlmostEqual(x.numpy()[idx], expected_values[i])

    def test_sort_descending(self):
        """Test sorting in descending order."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        values, indices = flashlight.sort(x, descending=True)
        expected_values = np.array([5.0, 4.0, 3.0, 1.0, 1.0])
        np.testing.assert_array_equal(values.numpy(), expected_values)

    def test_sort_2d_dim0(self):
        """Test sorting 2D tensor along dim 0."""
        x = flashlight.tensor([[3.0, 1.0], [2.0, 4.0], [1.0, 3.0]])
        values, indices = flashlight.sort(x, dim=0)
        expected_values = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 4.0]])
        np.testing.assert_array_equal(values.numpy(), expected_values)

    def test_sort_2d_dim1(self):
        """Test sorting 2D tensor along dim 1 (last dim, default)."""
        x = flashlight.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        values, indices = flashlight.sort(x, dim=-1)
        expected_values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(values.numpy(), expected_values)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_sort_parity(self):
        """Test parity with PyTorch sort."""
        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        mlx_x = flashlight.tensor(data)
        torch_x = torch.tensor(data)

        mlx_vals, mlx_idx = flashlight.sort(mlx_x, dim=-1)
        torch_vals, torch_idx = torch.sort(torch_x, dim=-1)

        np.testing.assert_allclose(mlx_vals.numpy(), torch_vals.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
class TestArgsort(TestCase):
    """Test argsort operations."""

    def test_argsort_1d(self):
        """Test argsort on 1D tensor."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        indices = flashlight.argsort(x)
        # Verify indices sort the array
        sorted_vals = x.numpy()[indices.numpy()]
        expected = np.array([1.0, 1.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(sorted_vals, expected)

    def test_argsort_descending(self):
        """Test argsort in descending order."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        indices = flashlight.argsort(x, descending=True)
        sorted_vals = x.numpy()[indices.numpy()]
        expected = np.array([5.0, 4.0, 3.0, 1.0, 1.0])
        np.testing.assert_array_equal(sorted_vals, expected)

    def test_argsort_2d(self):
        """Test argsort on 2D tensor."""
        x = flashlight.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        indices = flashlight.argsort(x, dim=-1)
        # First row: [3,1,2] -> indices [1,2,0]
        # Second row: [6,4,5] -> indices [1,2,0]
        expected = np.array([[1, 2, 0], [1, 2, 0]])
        np.testing.assert_array_equal(indices.numpy(), expected)


@skipIfNoMLX
class TestTopk(TestCase):
    """Test topk operations."""

    def test_topk_basic(self):
        """Test basic topk functionality."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        values, indices = flashlight.topk(x, k=3)
        expected_values = np.array([9.0, 6.0, 5.0])
        np.testing.assert_array_equal(values.numpy(), expected_values)
        # Check indices point to correct values
        for val, idx in zip(values.numpy(), indices.numpy()):
            self.assertAlmostEqual(x.numpy()[idx], val)

    def test_topk_smallest(self):
        """Test topk with largest=False (smallest k)."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        values, indices = flashlight.topk(x, k=3, largest=False)
        expected_values = np.array([1.0, 1.0, 2.0])
        np.testing.assert_array_equal(values.numpy(), expected_values)

    def test_topk_2d(self):
        """Test topk on 2D tensor."""
        x = flashlight.tensor([[3.0, 1.0, 4.0, 2.0], [8.0, 5.0, 6.0, 7.0]])
        values, indices = flashlight.topk(x, k=2, dim=-1)
        # First row: top 2 are 4, 3
        # Second row: top 2 are 8, 7
        expected_values = np.array([[4.0, 3.0], [8.0, 7.0]])
        np.testing.assert_array_equal(values.numpy(), expected_values)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_topk_parity(self):
        """Test parity with PyTorch topk."""
        np.random.seed(42)
        data = np.random.randn(3, 10).astype(np.float32)

        mlx_x = flashlight.tensor(data)
        torch_x = torch.tensor(data)

        mlx_vals, mlx_idx = flashlight.topk(mlx_x, k=3, dim=-1)
        torch_vals, torch_idx = torch.topk(torch_x, k=3, dim=-1)

        np.testing.assert_allclose(mlx_vals.numpy(), torch_vals.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
class TestKthvalue(TestCase):
    """Test kthvalue operations."""

    def test_kthvalue_basic(self):
        """Test basic kthvalue functionality."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        value, index = flashlight.kthvalue(x, k=2)  # 2nd smallest
        # Sorted: [1, 1, 3, 4, 5], 2nd is 1
        self.assertAlmostEqual(value.numpy().item(), 1.0)

    def test_kthvalue_k3(self):
        """Test kthvalue for k=3."""
        x = flashlight.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        value, index = flashlight.kthvalue(x, k=3)  # 3rd smallest
        # Sorted: [1, 1, 3, 4, 5], 3rd is 3
        self.assertAlmostEqual(value.numpy().item(), 3.0)


@skipIfNoMLX
class TestMsort(TestCase):
    """Test msort operations."""

    def test_msort_2d(self):
        """Test msort (sort along first dimension)."""
        x = flashlight.tensor([[3.0, 1.0], [2.0, 4.0], [1.0, 3.0]])
        result = flashlight.msort(x)
        expected = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
