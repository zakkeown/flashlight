"""
Test Phase 3: Basic Backward Pass

Tests basic autograd functionality:
- Simple gradient computation
- Gradient accumulation
- Leaf tensor gradients
- Scalar backward
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
class TestBasicBackward(TestCase):
    """Test basic backward pass functionality."""

    def test_simple_add_backward(self):
        """Test backward pass through addition."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = flashlight.tensor([4.0, 5.0, 6.0], requires_grad=True)
        z = flashlight.add(x, y)

        # Backward from sum (scalar)
        loss = flashlight.sum(z)
        loss.backward()

        # d(sum(x+y))/dx = 1, d(sum(x+y))/dy = 1
        expected_grad = np.ones(3)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_grad)
        np.testing.assert_array_almost_equal(y.grad.numpy(), expected_grad)

    def test_scalar_mul_backward(self):
        """Test backward through scalar multiplication."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2.0

        # Backward
        loss = flashlight.sum(y)
        loss.backward()

        # d(sum(2*x))/dx = 2
        expected_grad = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_grad)

    def test_chain_backward(self):
        """Test backward through a chain of operations."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # y = x + 1
        y = flashlight.add(x, 1.0)

        # z = y * 2
        z = y * 2.0

        # loss = sum(z) = sum(2*(x+1)) = 2*sum(x) + 6
        loss = flashlight.sum(z)
        loss.backward()

        # d(loss)/dx = 2
        expected_grad = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_grad)

    def test_leaf_tensor_check(self):
        """Test that only leaf tensors get gradients."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)
        y = x * 2

        # x is a leaf, y is not
        self.assertTrue(x.is_leaf)
        self.assertFalse(y.is_leaf)

        # After backward, only x should have grad
        loss = flashlight.sum(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        # y.grad should be None (non-leaf tensors don't accumulate gradients)
        # (This is PyTorch behavior)

    def test_zero_grad(self):
        """Test zeroing gradients."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2
        loss = flashlight.sum(y)
        loss.backward()

        # Check gradient exists
        self.assertIsNotNone(x.grad)

        # Zero gradient
        x.zero_grad()
        self.assertIsNone(x.grad)

    def test_requires_grad_false(self):
        """Test that tensors without requires_grad don't get gradients."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=False)
        y = x * 2
        loss = flashlight.sum(y)

        # Should raise error when trying to backward
        with self.assertRaises(RuntimeError):
            loss.backward()

    def test_non_scalar_backward_requires_gradient(self):
        """Test that backward on non-scalar requires gradient argument."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2

        # Should raise error - y is not a scalar
        with self.assertRaises(RuntimeError):
            y.backward()

    def test_non_scalar_backward_with_gradient(self):
        """Test backward on non-scalar with gradient argument."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2

        # Provide gradient
        grad_output = flashlight.tensor([1.0, 1.0, 1.0])
        y.backward(gradient=grad_output)

        # d(y)/dx = 2
        expected_grad = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_grad)


@skipIfNoMLX
class TestGradientAccumulation(TestCase):
    """Test gradient accumulation for tensors used multiple times."""

    def test_accumulation_simple(self):
        """Test gradient accumulation when tensor is used twice."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        # Use x twice
        y = x + x  # y = 2x

        loss = flashlight.sum(y)
        loss.backward()

        # Gradient should be accumulated: d(2x)/dx = 2
        expected_grad = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_grad)

    def test_accumulation_complex(self):
        """Test gradient accumulation in complex graph."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)

        # Use x in multiple paths
        y1 = x * 2
        y2 = x * 3
        z = flashlight.add(y1, y2)  # z = 2x + 3x = 5x

        loss = flashlight.sum(z)
        loss.backward()

        # d(5x)/dx = 5
        expected_grad = np.array([5.0, 5.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_grad)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
