"""
Test Phase 3: Trigonometric Backward Operations

Tests gradient computation for trigonometric functions:
- SinBackward
- CosBackward
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
class TestSinBackward(TestCase):
    """Test SinBackward gradient computation."""

    def test_sin_simple(self):
        """Test basic sin backward."""
        x = mlx_compat.tensor([0.0, np.pi/4, np.pi/2], requires_grad=True)
        y = mlx_compat.sin(x)
        loss = mlx_compat.sum(y)
        loss.backward()

        # d(sin(x))/dx = cos(x)
        x_np = np.array([0.0, np.pi/4, np.pi/2])
        expected = np.cos(x_np)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)

    def test_sin_2d(self):
        """Test sin backward on 2D tensor."""
        x = mlx_compat.tensor([[0.0, np.pi], [np.pi/2, np.pi*3/2]], requires_grad=True)
        y = mlx_compat.sin(x)
        loss = mlx_compat.sum(y)
        loss.backward()

        x_np = np.array([[0.0, np.pi], [np.pi/2, np.pi*3/2]])
        expected = np.cos(x_np)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestCosBackward(TestCase):
    """Test CosBackward gradient computation."""

    def test_cos_simple(self):
        """Test basic cos backward."""
        x = mlx_compat.tensor([0.0, np.pi/4, np.pi/2], requires_grad=True)
        y = mlx_compat.cos(x)
        loss = mlx_compat.sum(y)
        loss.backward()

        # d(cos(x))/dx = -sin(x)
        x_np = np.array([0.0, np.pi/4, np.pi/2])
        expected = -np.sin(x_np)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)

    def test_cos_2d(self):
        """Test cos backward on 2D tensor."""
        x = mlx_compat.tensor([[0.0, np.pi], [np.pi/2, np.pi*3/2]], requires_grad=True)
        y = mlx_compat.cos(x)
        loss = mlx_compat.sum(y)
        loss.backward()

        x_np = np.array([[0.0, np.pi], [np.pi/2, np.pi*3/2]])
        expected = -np.sin(x_np)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestTrigChain(TestCase):
    """Test chained trigonometric operations."""

    def test_sin_cos_chain(self):
        """Test sin(cos(x)) backward."""
        x = mlx_compat.tensor([0.0, np.pi/4, np.pi/2], requires_grad=True)
        y = mlx_compat.sin(mlx_compat.cos(x))
        loss = mlx_compat.sum(y)
        loss.backward()

        # d(sin(cos(x)))/dx = cos(cos(x)) * (-sin(x))
        x_np = np.array([0.0, np.pi/4, np.pi/2])
        expected = np.cos(np.cos(x_np)) * (-np.sin(x_np))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
