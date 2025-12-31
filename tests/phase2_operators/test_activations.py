"""
Test Phase 2: Activation Functions

Tests activation functions (relu, gelu, sigmoid, tanh, softmax, etc.)
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
class TestReLU(TestCase):
    """Test ReLU activation."""

    def test_relu_positive(self):
        """Test ReLU on positive values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.relu(x)
        np.testing.assert_array_equal(y.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_relu_negative(self):
        """Test ReLU on negative values."""
        x = mlx_compat.tensor([-1.0, -2.0, -3.0])
        y = mlx_compat.relu(x)
        np.testing.assert_array_equal(y.numpy(), np.array([0.0, 0.0, 0.0]))

    def test_relu_mixed(self):
        """Test ReLU on mixed values."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = mlx_compat.relu(x)
        np.testing.assert_array_equal(y.numpy(), np.array([0.0, 0.0, 0.0, 1.0, 2.0]))

    def test_relu_requires_grad(self):
        """Test that ReLU propagates requires_grad."""
        x = mlx_compat.tensor([1.0, -1.0], requires_grad=True)
        y = mlx_compat.relu(x)
        self.assertTrue(y.requires_grad)


@skipIfNoMLX
class TestGELU(TestCase):
    """Test GELU activation."""

    def test_gelu_shape(self):
        """Test GELU preserves shape."""
        x = mlx_compat.randn(10, 20)
        y = mlx_compat.gelu(x)
        self.assert_shape_equal(y.shape, x.shape)

    def test_gelu_zero(self):
        """Test GELU at zero."""
        x = mlx_compat.tensor([0.0])
        y = mlx_compat.gelu(x)
        # GELU(0) = 0
        self.assertAlmostEqual(y.item(), 0.0, places=5)

    def test_gelu_positive(self):
        """Test GELU on positive values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.gelu(x)
        # GELU(x) ≈ x for large positive x
        # Check that output is positive and less than or equal to input for positive values
        y_np = y.numpy()
        x_np = x.numpy()
        self.assertTrue((y_np > 0).all())
        # For x > 1, GELU(x) should be close to x
        self.assertTrue((y_np < x_np + 0.1).all())


@skipIfNoMLX
class TestSigmoid(TestCase):
    """Test sigmoid activation."""

    def test_sigmoid_zero(self):
        """Test sigmoid at zero."""
        x = mlx_compat.tensor([0.0])
        y = mlx_compat.sigmoid(x)
        self.assertAlmostEqual(y.item(), 0.5, places=5)

    def test_sigmoid_range(self):
        """Test that sigmoid outputs are in (0, 1)."""
        x = mlx_compat.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = mlx_compat.sigmoid(x)
        y_np = y.numpy()
        self.assertTrue((y_np > 0).all())
        self.assertTrue((y_np < 1).all())

    def test_sigmoid_symmetry(self):
        """Test sigmoid symmetry: sigmoid(x) + sigmoid(-x) = 1."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y_pos = mlx_compat.sigmoid(x)
        y_neg = mlx_compat.sigmoid(-x)
        sum_result = y_pos + y_neg
        np.testing.assert_array_almost_equal(sum_result.numpy(), np.ones(3), decimal=5)


@skipIfNoMLX
class TestTanh(TestCase):
    """Test tanh activation."""

    def test_tanh_zero(self):
        """Test tanh at zero."""
        x = mlx_compat.tensor([0.0])
        y = mlx_compat.tanh(x)
        self.assertAlmostEqual(y.item(), 0.0, places=5)

    def test_tanh_range(self):
        """Test that tanh outputs are in (-1, 1) or very close."""
        x = mlx_compat.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = mlx_compat.tanh(x)
        y_np = y.numpy()
        # tanh asymptotically approaches -1 and 1, so use >= and <=
        self.assertTrue((y_np >= -1).all())
        self.assertTrue((y_np <= 1).all())
        # For less extreme values, should be strictly within bounds
        x_small = mlx_compat.tensor([-1.0, 0.0, 1.0])
        y_small = mlx_compat.tanh(x_small)
        y_small_np = y_small.numpy()
        self.assertTrue((y_small_np > -1).all())
        self.assertTrue((y_small_np < 1).all())

    def test_tanh_antisymmetry(self):
        """Test tanh antisymmetry: tanh(-x) = -tanh(x)."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y_pos = mlx_compat.tanh(x)
        y_neg = mlx_compat.tanh(-x)
        np.testing.assert_array_almost_equal(y_neg.numpy(), -y_pos.numpy(), decimal=5)


@skipIfNoMLX
class TestSoftmax(TestCase):
    """Test softmax activation."""

    def test_softmax_1d(self):
        """Test softmax on 1D tensor."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.softmax(x)
        # Softmax should sum to 1
        self.assertAlmostEqual(y.numpy().sum(), 1.0, places=5)

    def test_softmax_2d(self):
        """Test softmax on 2D tensor."""
        x = mlx_compat.randn(3, 5)
        y = mlx_compat.softmax(x, dim=1)
        # Each row should sum to 1
        row_sums = y.numpy().sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=5)

    def test_softmax_properties(self):
        """Test softmax properties: all positive, sum to 1."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y = mlx_compat.softmax(x)
        y_np = y.numpy()
        # All values should be positive
        self.assertTrue((y_np > 0).all())
        # Should sum to 1
        self.assertAlmostEqual(y_np.sum(), 1.0, places=5)

    def test_softmax_max_value(self):
        """Test that max input gets highest softmax value."""
        x = mlx_compat.tensor([1.0, 5.0, 3.0])
        y = mlx_compat.softmax(x)
        # Index 1 should have the highest value
        self.assertEqual(y.numpy().argmax(), 1)


@skipIfNoMLX
class TestLogSoftmax(TestCase):
    """Test log_softmax activation."""

    def test_log_softmax_shape(self):
        """Test log_softmax preserves shape."""
        x = mlx_compat.randn(3, 5)
        y = mlx_compat.log_softmax(x, dim=1)
        self.assert_shape_equal(y.shape, x.shape)

    def test_log_softmax_values(self):
        """Test log_softmax produces negative values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.log_softmax(x)
        # log_softmax should produce negative values (since softmax < 1)
        self.assertTrue((y.numpy() <= 0).all())

    def test_log_softmax_consistency(self):
        """Test log_softmax is consistent with log(softmax)."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0])
        y_log_softmax = mlx_compat.log_softmax(x)
        y_softmax = mlx_compat.softmax(x)
        y_log = mlx_compat.log(y_softmax)
        np.testing.assert_array_almost_equal(y_log_softmax.numpy(), y_log.numpy(), decimal=5)


@skipIfNoMLX
class TestSiLU(TestCase):
    """Test SiLU (Swish) activation."""

    def test_silu_zero(self):
        """Test SiLU at zero."""
        x = mlx_compat.tensor([0.0])
        y = mlx_compat.silu(x)
        # SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        self.assertAlmostEqual(y.item(), 0.0, places=5)

    def test_silu_positive(self):
        """Test SiLU on positive values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.silu(x)
        # SiLU(x) = x * sigmoid(x), should be positive for positive x
        self.assertTrue((y.numpy() > 0).all())

    def test_silu_swish_alias(self):
        """Test that swish is an alias for silu."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y_silu = mlx_compat.silu(x)
        y_swish = mlx_compat.swish(x)
        np.testing.assert_array_equal(y_silu.numpy(), y_swish.numpy())


@skipIfNoMLX
class TestLeakyReLU(TestCase):
    """Test Leaky ReLU activation."""

    def test_leaky_relu_positive(self):
        """Test Leaky ReLU on positive values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.leaky_relu(x)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_leaky_relu_negative(self):
        """Test Leaky ReLU on negative values."""
        x = mlx_compat.tensor([-1.0, -2.0, -3.0])
        y = mlx_compat.leaky_relu(x, negative_slope=0.01)
        expected = x.numpy() * 0.01
        np.testing.assert_array_almost_equal(y.numpy(), expected, decimal=5)

    def test_leaky_relu_custom_slope(self):
        """Test Leaky ReLU with custom slope."""
        x = mlx_compat.tensor([-1.0])
        y = mlx_compat.leaky_relu(x, negative_slope=0.2)
        self.assertAlmostEqual(y.item(), -0.2, places=5)


@skipIfNoMLX
class TestELU(TestCase):
    """Test ELU activation."""

    def test_elu_positive(self):
        """Test ELU on positive values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0])
        y = mlx_compat.elu(x)
        # ELU(x) = x for x > 0
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_elu_negative(self):
        """Test ELU on negative values."""
        x = mlx_compat.tensor([0.0])
        y = mlx_compat.elu(x)
        # ELU(0) = 0
        self.assertAlmostEqual(y.item(), 0.0, places=5)

    def test_elu_range(self):
        """Test ELU output range for negative inputs."""
        x = mlx_compat.tensor([-1.0, -2.0, -5.0])
        y = mlx_compat.elu(x, alpha=1.0)
        y_np = y.numpy()
        # For negative x, ELU(x) = alpha * (exp(x) - 1) which is in (-alpha, 0)
        self.assertTrue((y_np > -1.0).all())
        self.assertTrue((y_np < 0.0).all())


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
