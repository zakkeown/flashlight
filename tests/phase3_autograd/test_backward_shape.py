"""
Test Phase 3: Shape Operation Backward Operations

Tests gradient computation for shape operations:
- CatBackward
- ViewBackward
- TransposeBackward
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
class TestCatBackward(TestCase):
    """Test CatBackward gradient computation."""

    def test_cat_dim0(self):
        """Test cat along dim 0 backward."""
        x = flashlight.tensor([[1.0, 2.0]], requires_grad=True)
        y = flashlight.tensor([[3.0, 4.0]], requires_grad=True)
        z = flashlight.cat([x, y], dim=0)
        loss = flashlight.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((1, 2)))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.ones((1, 2)))

    def test_cat_dim1(self):
        """Test cat along dim 1 backward."""
        x = flashlight.tensor([[1.0], [2.0]], requires_grad=True)
        y = flashlight.tensor([[3.0], [4.0]], requires_grad=True)
        z = flashlight.cat([x, y], dim=1)
        loss = flashlight.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 1)))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.ones((2, 1)))

    def test_cat_multiple(self):
        """Test cat with multiple tensors."""
        x = flashlight.tensor([1.0], requires_grad=True)
        y = flashlight.tensor([2.0], requires_grad=True)
        w = flashlight.tensor([3.0], requires_grad=True)
        z = flashlight.cat([x, y, w], dim=0)
        loss = flashlight.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(1))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.ones(1))
        np.testing.assert_array_almost_equal(w.grad.numpy(), np.ones(1))


@skipIfNoMLX
class TestViewBackward(TestCase):
    """Test ViewBackward gradient computation."""

    def test_view_flatten(self):
        """Test view flatten backward."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.view(4)
        loss = flashlight.sum(y)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 2)))

    def test_view_reshape(self):
        """Test view reshape backward."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        y = x.view(2, 3)
        loss = flashlight.sum(y)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(6))

    def test_view_chain(self):
        """Test chained view operations."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.view(6).view(3, 2).view(2, 3)
        loss = flashlight.sum(y)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 3)))


@skipIfNoMLX
class TestTransposeBackward(TestCase):
    """Test TransposeBackward gradient computation."""

    def test_transpose_2d(self):
        """Test 2D transpose backward."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.transpose(0, 1)
        loss = flashlight.sum(y)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 3)))

    def test_transpose_3d(self):
        """Test 3D transpose backward."""
        x = flashlight.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
        )
        y = x.transpose(0, 2)
        loss = flashlight.sum(y)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 2, 2)))

    def test_transpose_t(self):
        """Test .t() method backward."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        y = x.t()
        loss = flashlight.sum(y)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((3, 2)))


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
