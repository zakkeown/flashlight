"""
Test Phase 3: Reduction Backward Operations

Tests gradient computation for reduction operations:
- SumBackward
- MeanBackward
- MaxBackward
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


@skipIfNoMLX
class TestSumBackward(TestCase):
    """Test SumBackward gradient computation."""

    def test_sum_global(self):
        """Test global sum backward."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = flashlight.sum(x)
        y.backward()

        # d(sum(x))/dx = 1 for all elements
        expected = np.ones((2, 2))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_sum_dim(self):
        """Test sum along dimension backward."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = flashlight.sum(x, dim=1)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.ones((2, 2))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_sum_dim_keepdim(self):
        """Test sum with keepdim=True."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = flashlight.sum(x, dim=0, keepdim=True)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.ones((2, 2))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_sum_3d(self):
        """Test sum on 3D tensor."""
        x = flashlight.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        y = flashlight.sum(x, dim=1)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.ones((2, 2, 2))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestMeanBackward(TestCase):
    """Test MeanBackward gradient computation."""

    def test_mean_global(self):
        """Test global mean backward."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = flashlight.mean(x)
        y.backward()

        # d(mean(x))/dx = 1/n for all elements
        expected = np.ones((2, 2)) / 4.0
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_mean_dim(self):
        """Test mean along dimension backward."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = flashlight.mean(x, dim=1)
        loss = flashlight.sum(y)
        loss.backward()

        # Each element contributes 1/2 to its row mean, and we sum both row means
        expected = np.ones((2, 2)) / 2.0
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_mean_dim_keepdim(self):
        """Test mean with keepdim=True."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = flashlight.mean(x, dim=0, keepdim=True)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.ones((2, 2)) / 2.0
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestMaxBackward(TestCase):
    """Test MaxBackward gradient computation."""

    def test_max_global_simple(self):
        """Test global max backward - simple case."""
        x = flashlight.tensor([1.0, 3.0, 2.0], requires_grad=True)
        y = x.max()
        y.backward()

        # Gradient flows only to max element
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_global_ties_distributed(self):
        """Test global max with ties - gradient distributed equally (PyTorch behavior)."""
        x = flashlight.tensor([3.0, 1.0, 3.0], requires_grad=True)
        y = x.max()
        y.backward()

        # Gradient distributed equally among tied max values
        expected = np.array([0.5, 0.0, 0.5])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_global_2d(self):
        """Test global max on 2D tensor."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        y = x.max()
        y.backward()

        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_dim_simple(self):
        """Test max along dimension backward."""
        x = flashlight.tensor([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
        values, indices = x.max(dim=1)
        loss = values.sum()
        loss.backward()

        # Max of row 0 is at [0,1]=4, row 1 is at [1,0]=3
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_dim_ties(self):
        """Test max along dimension with ties - gradient distributed equally."""
        x = flashlight.tensor([[4.0, 4.0], [3.0, 3.0]], requires_grad=True)
        values, indices = x.max(dim=1)
        loss = values.sum()
        loss.backward()

        # Gradient distributed equally among tied max values along dim
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_dim_keepdim(self):
        """Test max with keepdim=True."""
        x = flashlight.tensor([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
        values, indices = x.max(dim=1, keepdim=True)
        loss = values.sum()
        loss.backward()

        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_dim0(self):
        """Test max along dimension 0."""
        x = flashlight.tensor([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
        values, indices = x.max(dim=0)
        loss = values.sum()
        loss.backward()

        # Max of col 0 is at [1,0]=3, col 1 is at [0,1]=4
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_3d(self):
        """Test max on 3D tensor."""
        x = flashlight.tensor(
            [[[1.0, 5.0], [3.0, 2.0]], [[4.0, 1.0], [2.0, 6.0]]], requires_grad=True
        )
        values, indices = x.max(dim=2)
        loss = values.sum()
        loss.backward()

        # Shape is (2, 2, 2), reduce along dim 2
        expected = np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_negative_values(self):
        """Test max with negative values."""
        x = flashlight.tensor([-5.0, -2.0, -8.0], requires_grad=True)
        y = x.max()
        y.backward()

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_gradient_accumulation(self):
        """Test gradient accumulation through max."""
        x = flashlight.tensor([1.0, 3.0, 2.0], requires_grad=True)

        # Two forward passes
        y1 = x.max()
        y2 = x.max()
        loss = y1 + y2
        loss.backward()

        # Gradient should be accumulated (2x)
        expected = np.array([0.0, 2.0, 0.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestMaxBackwardEdgeCases(TestCase):
    """Test MaxBackward edge cases."""

    def test_max_single_element(self):
        """Test max on single element tensor."""
        x = flashlight.tensor([5.0], requires_grad=True)
        y = x.max()
        y.backward()

        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_all_same(self):
        """Test max when all elements are the same (gradient distributed equally)."""
        x = flashlight.tensor([2.0, 2.0, 2.0], requires_grad=True)
        y = x.max()
        y.backward()

        # Gradient distributed equally among all tied max values
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_max_large_tensor(self):
        """Test max on larger tensor."""
        data = np.random.randn(10, 20).astype(np.float32)
        x = flashlight.tensor(data, requires_grad=True)
        y = x.max()
        y.backward()

        # Find where the max is
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        expected = np.zeros_like(data)
        expected[max_idx] = 1.0

        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
