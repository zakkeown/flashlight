"""
Test Phase 3: Arithmetic Backward Operations

Tests gradient computation for arithmetic operations:
- AddBackward
- SubBackward
- MulBackward
- DivBackward
- MatmulBackward
- PowBackward
- SqrtBackward
- ExpBackward
- LogBackward
- AbsBackward
- NegBackward
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
class TestAddBackward(TestCase):
    """Test AddBackward gradient computation."""

    def test_add_simple(self):
        """Test basic addition backward."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = mlx_compat.tensor([4.0, 5.0, 6.0], requires_grad=True)
        z = x + y
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.ones(3))

    def test_add_broadcast(self):
        """Test addition with broadcasting."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = mlx_compat.tensor([1.0, 1.0], requires_grad=True)
        z = x + y
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 2)))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.array([2.0, 2.0]))

    def test_add_scalar(self):
        """Test addition with scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = x + 5.0
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))


@skipIfNoMLX
class TestSubBackward(TestCase):
    """Test SubBackward gradient computation."""

    def test_sub_simple(self):
        """Test basic subtraction backward."""
        x = mlx_compat.tensor([5.0, 6.0, 7.0], requires_grad=True)
        y = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = x - y
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))
        np.testing.assert_array_almost_equal(y.grad.numpy(), -np.ones(3))

    def test_sub_broadcast(self):
        """Test subtraction with broadcasting - gradient for broadcast dim is summed."""
        x = mlx_compat.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)
        y = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        z = x - y
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones((2, 2)))
        # SubBackward now properly handles unbroadcast - gradient for y is summed over broadcast dim
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.array([-2.0, -2.0]))


@skipIfNoMLX
class TestMulBackward(TestCase):
    """Test MulBackward gradient computation."""

    def test_mul_simple(self):
        """Test basic multiplication backward."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = mlx_compat.tensor([4.0, 5.0, 6.0], requires_grad=True)
        z = x * y
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(x*y)/dx = y, d(x*y)/dy = x
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_mul_scalar(self):
        """Test multiplication by scalar."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = x * 3.0
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([3.0, 3.0, 3.0]))

    def test_mul_self(self):
        """Test x * x (gradient accumulation)."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = x * x
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(x^2)/dx = 2x
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([2.0, 4.0, 6.0]))


@skipIfNoMLX
class TestDivBackward(TestCase):
    """Test DivBackward gradient computation."""

    def test_div_simple(self):
        """Test basic division backward."""
        x = mlx_compat.tensor([4.0, 6.0, 8.0], requires_grad=True)
        y = mlx_compat.tensor([2.0, 3.0, 4.0], requires_grad=True)
        z = x / y
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([0.5, 1/3, 0.25]))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.array([-1.0, -2/3, -0.5]))

    def test_div_scalar(self):
        """Test division by scalar."""
        x = mlx_compat.tensor([2.0, 4.0, 6.0], requires_grad=True)
        z = x / 2.0
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([0.5, 0.5, 0.5]))


@skipIfNoMLX
class TestMatmulBackward(TestCase):
    """Test MatmulBackward gradient computation."""

    def test_matmul_simple(self):
        """Test basic matrix multiplication backward."""
        A = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = mlx_compat.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        C = mlx_compat.matmul(A, B)
        loss = mlx_compat.sum(C)
        loss.backward()

        # d(AB)/dA = ones @ B.T = sum(B, axis=0) repeated
        # d(AB)/dB = A.T @ ones = sum(A, axis=1) repeated
        expected_grad_A = np.array([[11.0, 15.0], [11.0, 15.0]])  # B.sum(axis=1)
        expected_grad_B = np.array([[4.0, 4.0], [6.0, 6.0]])  # A.sum(axis=0).T

        np.testing.assert_array_almost_equal(A.grad.numpy(), expected_grad_A)
        np.testing.assert_array_almost_equal(B.grad.numpy(), expected_grad_B)

    def test_matmul_mv(self):
        """Test matrix-vector multiplication backward."""
        A = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        v = mlx_compat.tensor([[1.0], [1.0]], requires_grad=True)  # Column vector
        C = mlx_compat.matmul(A, v)
        loss = mlx_compat.sum(C)
        loss.backward()

        self.assertEqual(A.grad.shape, (2, 2))
        self.assertEqual(v.grad.shape, (2, 1))


@skipIfNoMLX
class TestPowBackward(TestCase):
    """Test PowBackward gradient computation."""

    def test_pow_simple(self):
        """Test basic power backward."""
        x = mlx_compat.tensor([2.0, 3.0, 4.0], requires_grad=True)
        z = x ** 2
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(x^2)/dx = 2x
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([4.0, 6.0, 8.0]))

    def test_pow_3(self):
        """Test cube backward."""
        x = mlx_compat.tensor([2.0, 3.0], requires_grad=True)
        z = x ** 3
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(x^3)/dx = 3x^2
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([12.0, 27.0]))


@skipIfNoMLX
class TestSqrtBackward(TestCase):
    """Test SqrtBackward gradient computation."""

    def test_sqrt_simple(self):
        """Test basic sqrt backward."""
        x = mlx_compat.tensor([4.0, 9.0, 16.0], requires_grad=True)
        z = mlx_compat.sqrt(x)
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(sqrt(x))/dx = 1/(2*sqrt(x))
        expected = 1.0 / (2.0 * np.sqrt(np.array([4.0, 9.0, 16.0])))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestExpBackward(TestCase):
    """Test ExpBackward gradient computation."""

    def test_exp_simple(self):
        """Test basic exp backward."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0], requires_grad=True)
        z = mlx_compat.exp(x)
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(exp(x))/dx = exp(x)
        expected = np.exp(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestLogBackward(TestCase):
    """Test LogBackward gradient computation."""

    def test_log_simple(self):
        """Test basic log backward."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = mlx_compat.log(x)
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(log(x))/dx = 1/x
        expected = 1.0 / np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


@skipIfNoMLX
class TestAbsBackward(TestCase):
    """Test AbsBackward gradient computation."""

    def test_abs_positive(self):
        """Test abs backward for positive values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
        z = mlx_compat.abs(x)
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(|x|)/dx = sign(x) = 1 for positive
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.ones(3))

    def test_abs_negative(self):
        """Test abs backward for negative values."""
        x = mlx_compat.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        z = mlx_compat.abs(x)
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(|x|)/dx = sign(x) = -1 for negative
        np.testing.assert_array_almost_equal(x.grad.numpy(), -np.ones(3))

    def test_abs_mixed(self):
        """Test abs backward for mixed values."""
        x = mlx_compat.tensor([-2.0, 3.0, -1.0], requires_grad=True)
        z = mlx_compat.abs(x)
        loss = mlx_compat.sum(z)
        loss.backward()

        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([-1.0, 1.0, -1.0]))


@skipIfNoMLX
class TestNegBackward(TestCase):
    """Test NegBackward gradient computation."""

    def test_neg_simple(self):
        """Test basic negation backward."""
        x = mlx_compat.tensor([1.0, -2.0, 3.0], requires_grad=True)
        z = -x
        loss = mlx_compat.sum(z)
        loss.backward()

        # d(-x)/dx = -1
        np.testing.assert_array_almost_equal(x.grad.numpy(), -np.ones(3))


@skipIfNoMLX
class TestChainedArithmetic(TestCase):
    """Test chained arithmetic operations."""

    def test_chain_add_mul(self):
        """Test chain of add and mul."""
        x = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        y = mlx_compat.tensor([3.0, 4.0], requires_grad=True)

        z = (x + y) * x  # = x^2 + xy
        loss = mlx_compat.sum(z)
        loss.backward()

        # d/dx = 2x + y = [5, 8]
        # d/dy = x = [1, 2]
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([5.0, 8.0]))
        np.testing.assert_array_almost_equal(y.grad.numpy(), np.array([1.0, 2.0]))

    def test_chain_complex(self):
        """Test complex chain of operations."""
        x = mlx_compat.tensor([2.0], requires_grad=True)

        # f(x) = (x^2 + x) * x = x^3 + x^2
        z = (x * x + x) * x
        z.backward()

        # df/dx = 3x^2 + 2x = 3*4 + 4 = 16
        np.testing.assert_array_almost_equal(x.grad.numpy(), np.array([16.0]))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
