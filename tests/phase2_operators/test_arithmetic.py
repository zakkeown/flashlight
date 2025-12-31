"""
Test Phase 2: Arithmetic Operators

Tests arithmetic operations (add, sub, mul, div, matmul, etc.)
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
class TestAdd(TestCase):
    """Test addition operations."""

    def test_add_tensors(self):
        """Test adding two tensors."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        b = mlx_compat.tensor([4.0, 5.0, 6.0])
        c = mlx_compat.add(a, b)
        np.testing.assert_array_equal(c.numpy(), np.array([5.0, 7.0, 9.0]))

    def test_add_scalar(self):
        """Test adding a scalar."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        c = mlx_compat.add(a, 10.0)
        np.testing.assert_array_equal(c.numpy(), np.array([11.0, 12.0, 13.0]))

    def test_add_with_alpha(self):
        """Test add with alpha parameter."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        b = mlx_compat.tensor([4.0, 5.0, 6.0])
        c = mlx_compat.add(a, b, alpha=2.0)
        # out = a + 2 * b = [1,2,3] + 2*[4,5,6] = [9,12,15]
        np.testing.assert_array_equal(c.numpy(), np.array([9.0, 12.0, 15.0]))

    def test_add_broadcasting(self):
        """Test add with broadcasting."""
        a = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = mlx_compat.tensor([10.0, 20.0])
        c = mlx_compat.add(a, b)
        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        np.testing.assert_array_equal(c.numpy(), expected)

    def test_add_requires_grad(self):
        """Test that add propagates requires_grad."""
        a = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        b = mlx_compat.tensor([3.0, 4.0])
        c = mlx_compat.add(a, b)
        self.assertTrue(c.requires_grad)


@skipIfNoMLX
class TestSub(TestCase):
    """Test subtraction operations."""

    def test_sub_tensors(self):
        """Test subtracting two tensors."""
        a = mlx_compat.tensor([5.0, 6.0, 7.0])
        b = mlx_compat.tensor([1.0, 2.0, 3.0])
        c = mlx_compat.sub(a, b)
        np.testing.assert_array_equal(c.numpy(), np.array([4.0, 4.0, 4.0]))

    def test_sub_scalar(self):
        """Test subtracting a scalar."""
        a = mlx_compat.tensor([10.0, 20.0, 30.0])
        c = mlx_compat.sub(a, 5.0)
        np.testing.assert_array_equal(c.numpy(), np.array([5.0, 15.0, 25.0]))

    def test_sub_with_alpha(self):
        """Test sub with alpha parameter."""
        a = mlx_compat.tensor([10.0, 20.0, 30.0])
        b = mlx_compat.tensor([1.0, 2.0, 3.0])
        c = mlx_compat.sub(a, b, alpha=2.0)
        # out = a - 2 * b = [10,20,30] - 2*[1,2,3] = [8,16,24]
        np.testing.assert_array_equal(c.numpy(), np.array([8.0, 16.0, 24.0]))


@skipIfNoMLX
class TestMul(TestCase):
    """Test multiplication operations."""

    def test_mul_tensors(self):
        """Test multiplying two tensors."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        b = mlx_compat.tensor([2.0, 3.0, 4.0])
        c = mlx_compat.mul(a, b)
        np.testing.assert_array_equal(c.numpy(), np.array([2.0, 6.0, 12.0]))

    def test_mul_scalar(self):
        """Test multiplying by a scalar."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        c = mlx_compat.mul(a, 10.0)
        np.testing.assert_array_equal(c.numpy(), np.array([10.0, 20.0, 30.0]))

    def test_mul_broadcasting(self):
        """Test mul with broadcasting."""
        a = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = mlx_compat.tensor([10.0, 20.0])
        c = mlx_compat.mul(a, b)
        expected = np.array([[10.0, 40.0], [30.0, 80.0]])
        np.testing.assert_array_equal(c.numpy(), expected)


@skipIfNoMLX
class TestDiv(TestCase):
    """Test division operations."""

    def test_div_tensors(self):
        """Test dividing two tensors."""
        a = mlx_compat.tensor([6.0, 8.0, 10.0])
        b = mlx_compat.tensor([2.0, 4.0, 5.0])
        c = mlx_compat.div(a, b)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([3.0, 2.0, 2.0]))

    def test_div_scalar(self):
        """Test dividing by a scalar."""
        a = mlx_compat.tensor([10.0, 20.0, 30.0])
        c = mlx_compat.div(a, 10.0)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_div_rounding_trunc(self):
        """Test division with truncation rounding."""
        a = mlx_compat.tensor([7.0, -7.0, 8.0])
        b = mlx_compat.tensor([3.0, 3.0, 3.0])
        c = mlx_compat.div(a, b, rounding_mode='trunc')
        np.testing.assert_array_equal(c.numpy(), np.array([2.0, -2.0, 2.0]))

    def test_div_rounding_floor(self):
        """Test division with floor rounding."""
        a = mlx_compat.tensor([7.0, -7.0, 8.0])
        b = mlx_compat.tensor([3.0, 3.0, 3.0])
        c = mlx_compat.div(a, b, rounding_mode='floor')
        np.testing.assert_array_equal(c.numpy(), np.array([2.0, -3.0, 2.0]))


@skipIfNoMLX
class TestMatmul(TestCase):
    """Test matrix multiplication operations."""

    def test_matmul_2d(self):
        """Test 2D matrix multiplication."""
        a = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = mlx_compat.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = mlx_compat.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.numpy(), expected)

    def test_matmul_1d_dot(self):
        """Test 1D matmul (dot product)."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        b = mlx_compat.tensor([4.0, 5.0, 6.0])
        c = mlx_compat.matmul(a, b)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertAlmostEqual(c.item(), 32.0, places=5)

    def test_matmul_broadcasting(self):
        """Test matmul with batch dimensions."""
        a = mlx_compat.randn(2, 3, 4)  # Batch of 2, 3x4 matrices
        b = mlx_compat.randn(2, 4, 5)  # Batch of 2, 4x5 matrices
        c = mlx_compat.matmul(a, b)
        self.assert_shape_equal(c.shape, (2, 3, 5))

    def test_mm_2d_only(self):
        """Test mm (2D only)."""
        a = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = mlx_compat.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = mlx_compat.mm(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.numpy(), expected)

    def test_mm_non_2d_raises(self):
        """Test that mm raises for non-2D tensors."""
        a = mlx_compat.randn(2, 3, 4)
        b = mlx_compat.randn(2, 4, 5)
        with self.assertRaises(RuntimeError):
            mlx_compat.mm(a, b)

    def test_bmm_3d(self):
        """Test bmm (batch matrix multiplication)."""
        a = mlx_compat.randn(10, 3, 4)
        b = mlx_compat.randn(10, 4, 5)
        c = mlx_compat.bmm(a, b)
        self.assert_shape_equal(c.shape, (10, 3, 5))

    def test_bmm_non_3d_raises(self):
        """Test that bmm raises for non-3D tensors."""
        a = mlx_compat.randn(3, 4)
        b = mlx_compat.randn(4, 5)
        with self.assertRaises(RuntimeError):
            mlx_compat.bmm(a, b)


@skipIfNoMLX
class TestPow(TestCase):
    """Test power operations."""

    def test_pow_scalar_exponent(self):
        """Test power with scalar exponent."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        c = mlx_compat.pow(a, 2.0)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([1.0, 4.0, 9.0]))

    def test_pow_tensor_exponent(self):
        """Test power with tensor exponent."""
        a = mlx_compat.tensor([2.0, 3.0, 4.0])
        b = mlx_compat.tensor([2.0, 3.0, 2.0])
        c = mlx_compat.pow(a, b)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([4.0, 27.0, 16.0]))

    def test_pow_negative_exponent(self):
        """Test power with negative exponent."""
        a = mlx_compat.tensor([2.0, 4.0, 8.0])
        c = mlx_compat.pow(a, -1.0)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([0.5, 0.25, 0.125]))


@skipIfNoMLX
class TestElementwiseUnary(TestCase):
    """Test unary element-wise operations."""

    def test_sqrt(self):
        """Test square root."""
        a = mlx_compat.tensor([1.0, 4.0, 9.0, 16.0])
        c = mlx_compat.sqrt(a)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([1.0, 2.0, 3.0, 4.0]))

    def test_exp(self):
        """Test exponential."""
        a = mlx_compat.tensor([0.0, 1.0, 2.0])
        c = mlx_compat.exp(a)
        expected = np.exp(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_almost_equal(c.numpy(), expected, decimal=5)

    def test_log(self):
        """Test natural logarithm."""
        a = mlx_compat.tensor([1.0, np.e, np.e**2])
        c = mlx_compat.log(a)
        np.testing.assert_array_almost_equal(c.numpy(), np.array([0.0, 1.0, 2.0]), decimal=5)

    def test_abs_positive(self):
        """Test absolute value of positive numbers."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        c = mlx_compat.abs(a)
        np.testing.assert_array_equal(c.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_abs_negative(self):
        """Test absolute value of negative numbers."""
        a = mlx_compat.tensor([-1.0, -2.0, -3.0])
        c = mlx_compat.abs(a)
        np.testing.assert_array_equal(c.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_abs_mixed(self):
        """Test absolute value of mixed signs."""
        a = mlx_compat.tensor([1.0, -2.0, 3.0, -4.0])
        c = mlx_compat.abs(a)
        np.testing.assert_array_equal(c.numpy(), np.array([1.0, 2.0, 3.0, 4.0]))

    def test_neg(self):
        """Test negation."""
        a = mlx_compat.tensor([1.0, -2.0, 3.0])
        c = mlx_compat.neg(a)
        np.testing.assert_array_equal(c.numpy(), np.array([-1.0, 2.0, -3.0]))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
