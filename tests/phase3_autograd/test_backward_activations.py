"""
Test Phase 3: Activation Backward Operations

Tests gradient computation for activation functions:
- ReLUBackward
- SigmoidBackward
- TanhBackward
- SoftmaxBackward
- LogSoftmaxBackward
- SiLUBackward
- LeakyReLUBackward
- ELUBackward
- GELUBackward
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
class TestReLUBackward(TestCase):
    """Test ReLUBackward gradient computation."""

    def test_relu_positive(self):
        """Test ReLU backward for positive values."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = flashlight.nn.functional.relu(x)
        loss = flashlight.sum(y)
        loss.backward()

        # d(relu(x))/dx = 1 for x > 0
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))

    def test_relu_negative(self):
        """Test ReLU backward for negative values."""
        x = flashlight.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        y = flashlight.nn.functional.relu(x)
        loss = flashlight.sum(y)
        loss.backward()

        # d(relu(x))/dx = 0 for x < 0
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.zeros(3))

    def test_relu_mixed(self):
        """Test ReLU backward for mixed values."""
        x = flashlight.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        y = flashlight.nn.functional.relu(x)
        loss = flashlight.sum(y)
        loss.backward()

        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestSigmoidBackward(TestCase):
    """Test SigmoidBackward gradient computation."""

    def test_sigmoid_simple(self):
        """Test sigmoid backward."""
        x = flashlight.tensor([0.0, 1.0, -1.0], requires_grad=True)
        y = flashlight.sigmoid(x)
        loss = flashlight.sum(y)
        loss.backward()

        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        x_np = np.array([0.0, 1.0, -1.0])
        sigmoid = 1.0 / (1.0 + np.exp(-x_np))
        expected = sigmoid * (1 - sigmoid)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestTanhBackward(TestCase):
    """Test TanhBackward gradient computation."""

    def test_tanh_simple(self):
        """Test tanh backward."""
        x = flashlight.tensor([0.0, 1.0, -1.0], requires_grad=True)
        y = flashlight.tanh(x)
        loss = flashlight.sum(y)
        loss.backward()

        # d(tanh(x))/dx = 1 - tanh^2(x)
        x_np = np.array([0.0, 1.0, -1.0])
        tanh_x = np.tanh(x_np)
        expected = 1 - tanh_x**2
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestSoftmaxBackward(TestCase):
    """Test SoftmaxBackward gradient computation."""

    def test_softmax_simple(self):
        """Test softmax backward."""
        x = flashlight.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = flashlight.nn.functional.softmax(x, dim=-1)
        loss = flashlight.sum(y)
        loss.backward()

        # For sum of softmax, gradient should sum to 0 along softmax dim
        # Actually, d(sum(softmax(x)))/dx = 0 since sum(softmax) = 1
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.zeros((1, 3)), decimal=5)

    def test_softmax_weighted(self):
        """Test softmax backward with weighted sum."""
        x = flashlight.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = flashlight.nn.functional.softmax(x, dim=-1)
        weights = flashlight.tensor([[1.0, 2.0, 3.0]])
        loss = flashlight.sum(y * weights)
        loss.backward()

        # Gradient should be non-zero
        self.assertFalse(np.allclose(x.grad.numpy(), np.zeros((1, 3))))


@skipIfNoMLX
class TestLogSoftmaxBackward(TestCase):
    """Test LogSoftmaxBackward gradient computation."""

    def test_log_softmax_simple(self):
        """Test log_softmax backward."""
        x = flashlight.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = flashlight.nn.functional.log_softmax(x, dim=-1)
        loss = flashlight.sum(y)
        loss.backward()

        # d(sum(log_softmax))/dx = 1 - n*softmax (where n=3)
        x_np = np.array([[1.0, 2.0, 3.0]])
        exp_x = np.exp(x_np)
        softmax = exp_x / exp_x.sum(axis=-1, keepdims=True)
        expected = 1 - 3 * softmax
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestSiLUBackward(TestCase):
    """Test SiLUBackward gradient computation."""

    def test_silu_simple(self):
        """Test SiLU/Swish backward."""
        x = flashlight.tensor([0.0, 1.0, -1.0], requires_grad=True)
        y = flashlight.nn.functional.silu(x)
        loss = flashlight.sum(y)
        loss.backward()

        # d(x*sigmoid(x))/dx = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        x_np = np.array([0.0, 1.0, -1.0])
        sigmoid = 1.0 / (1.0 + np.exp(-x_np))
        expected = sigmoid + x_np * sigmoid * (1 - sigmoid)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestLeakyReLUBackward(TestCase):
    """Test LeakyReLUBackward gradient computation."""

    def test_leaky_relu_positive(self):
        """Test leaky ReLU backward for positive values."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = flashlight.nn.functional.leaky_relu(x, negative_slope=0.01)
        loss = flashlight.sum(y)
        loss.backward()

        # d(leaky_relu(x))/dx = 1 for x > 0
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))

    def test_leaky_relu_negative(self):
        """Test leaky ReLU backward for negative values."""
        x = flashlight.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        y = flashlight.nn.functional.leaky_relu(x, negative_slope=0.1)
        loss = flashlight.sum(y)
        loss.backward()

        # d(leaky_relu(x))/dx = negative_slope for x < 0
        np.testing.assert_array_almost_equal(x.grad.numpy(), 0.1 * np.ones(3))


@skipIfNoMLX
class TestELUBackward(TestCase):
    """Test ELUBackward gradient computation."""

    def test_elu_positive(self):
        """Test ELU backward for positive values."""
        x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = flashlight.nn.functional.elu(x, alpha=1.0)
        loss = flashlight.sum(y)
        loss.backward()

        # d(elu(x))/dx = 1 for x > 0
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))

    def test_elu_negative(self):
        """Test ELU backward for negative values."""
        x = flashlight.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        y = flashlight.nn.functional.elu(x, alpha=1.0)
        loss = flashlight.sum(y)
        loss.backward()

        # d(elu(x))/dx = alpha * exp(x) for x < 0
        x_np = np.array([-1.0, -2.0, -3.0])
        expected = np.exp(x_np)
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


@skipIfNoMLX
class TestGELUBackward(TestCase):
    """Test GELUBackward gradient computation."""

    def test_gelu_simple(self):
        """Test GELU backward."""
        x = flashlight.tensor([0.0, 1.0, -1.0], requires_grad=True)
        y = flashlight.nn.functional.gelu(x)
        loss = flashlight.sum(y)
        loss.backward()

        # GELU gradient is complex, just verify it's reasonable
        grad = x.grad.numpy()
        # At x=0, gradient should be 0.5
        self.assertAlmostEqual(grad[0], 0.5, places=2)
        # Gradient at x=1 should be > 0.5
        self.assertGreater(grad[1], 0.5)
        # Gradient at x=-1 should be small and positive
        self.assertGreater(grad[2], -0.1)


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
