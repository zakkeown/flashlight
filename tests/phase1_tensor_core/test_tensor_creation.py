"""
Test Phase 1: Tensor Creation

Tests tensor creation operations and factory functions.
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
class TestTensorCreation(TestCase):
    """Test basic tensor creation."""

    def test_tensor_from_list(self):
        """Test creating tensor from Python list."""
        # Integer list infers int32 in MLX
        t = mlx_compat.tensor([1, 2, 3])
        self.assert_shape_equal(t.shape, (3,))
        # MLX infers int32 from integer list, unlike PyTorch which defaults to float32
        self.assertEqual(t.dtype, mlx_compat.int32)

        # Float list infers float32
        t_float = mlx_compat.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t_float.dtype, mlx_compat.float32)

    def test_tensor_from_nested_list(self):
        """Test creating tensor from nested list."""
        t = mlx_compat.tensor([[1, 2], [3, 4]])
        self.assert_shape_equal(t.shape, (2, 2))

    def test_tensor_from_numpy(self):
        """Test creating tensor from NumPy array."""
        arr = np.array([1.0, 2.0, 3.0])
        t = mlx_compat.from_numpy(arr)
        self.assert_shape_equal(t.shape, (3,))
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_tensor_with_dtype(self):
        """Test tensor creation with explicit dtype."""
        t = mlx_compat.tensor([1, 2, 3], dtype=mlx_compat.int32)
        self.assertEqual(t.dtype, mlx_compat.int32)

    def test_tensor_with_requires_grad(self):
        """Test tensor creation with requires_grad."""
        t = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        self.assertTrue(t.requires_grad)
        self.assertIsNone(t.grad)

    def test_tensor_properties(self):
        """Test tensor properties."""
        t = mlx_compat.tensor([[1, 2, 3], [4, 5, 6]])
        self.assert_shape_equal(t.shape, (2, 3))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.numel, 6)
        self.assertTrue(t.is_leaf)


@skipIfNoMLX
class TestZerosOnes(TestCase):
    """Test zeros() and ones() factory functions."""

    def test_zeros(self):
        """Test zeros creation."""
        t = mlx_compat.zeros(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_zeros_with_dtype(self):
        """Test zeros with specific dtype."""
        t = mlx_compat.zeros(5, dtype=mlx_compat.int32)
        self.assertEqual(t.dtype, mlx_compat.int32)

    def test_zeros_tuple_shape(self):
        """Test zeros with tuple shape."""
        t = mlx_compat.zeros((2, 3, 4))
        self.assert_shape_equal(t.shape, (2, 3, 4))

    def test_ones(self):
        """Test ones creation."""
        t = mlx_compat.ones(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4)))

    def test_ones_with_dtype(self):
        """Test ones with specific dtype."""
        t = mlx_compat.ones(5, dtype=mlx_compat.float16)
        self.assertEqual(t.dtype, mlx_compat.float16)

    def test_full(self):
        """Test full creation."""
        t = mlx_compat.full((3, 4), 7)
        self.assert_shape_equal(t.shape, (3, 4))
        np.testing.assert_array_equal(t.numpy(), np.full((3, 4), 7))

    def test_empty_warns(self):
        """Test that empty() issues a warning (MLX doesn't support uninitialized)."""
        with self.assertWarns(UserWarning):
            t = mlx_compat.empty(3, 4)
        # Should return zeros in MLX
        self.assert_shape_equal(t.shape, (3, 4))


@skipIfNoMLX
class TestLikeVariants(TestCase):
    """Test *_like factory functions."""

    def test_zeros_like(self):
        """Test zeros_like."""
        x = mlx_compat.randn(3, 4)
        t = mlx_compat.zeros_like(x)
        self.assert_shape_equal(t.shape, x.shape)
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_ones_like(self):
        """Test ones_like."""
        x = mlx_compat.randn(2, 3)
        t = mlx_compat.ones_like(x)
        self.assert_shape_equal(t.shape, x.shape)

    def test_full_like(self):
        """Test full_like."""
        x = mlx_compat.zeros(3, 4)
        t = mlx_compat.full_like(x, 5)
        np.testing.assert_array_equal(t.numpy(), np.full((3, 4), 5))

    def test_like_preserves_dtype(self):
        """Test that *_like preserves dtype by default."""
        x = mlx_compat.zeros(3, dtype=mlx_compat.int32)
        t = mlx_compat.ones_like(x)
        self.assertEqual(t.dtype, mlx_compat.int32)

    def test_like_with_dtype_override(self):
        """Test *_like with dtype override."""
        x = mlx_compat.zeros(3, dtype=mlx_compat.int32)
        t = mlx_compat.ones_like(x, dtype=mlx_compat.float32)
        self.assertEqual(t.dtype, mlx_compat.float32)


@skipIfNoMLX
class TestSequences(TestCase):
    """Test sequence generation functions."""

    def test_arange_single_arg(self):
        """Test arange with single argument."""
        t = mlx_compat.arange(10)
        self.assert_shape_equal(t.shape, (10,))
        np.testing.assert_array_equal(t.numpy(), np.arange(10))

    def test_arange_start_end(self):
        """Test arange with start and end."""
        t = mlx_compat.arange(2, 10)
        np.testing.assert_array_equal(t.numpy(), np.arange(2, 10))

    def test_arange_with_step(self):
        """Test arange with step."""
        t = mlx_compat.arange(0, 10, 2)
        np.testing.assert_array_equal(t.numpy(), np.arange(0, 10, 2))

    def test_linspace(self):
        """Test linspace."""
        t = mlx_compat.linspace(0, 1, 11)
        expected = np.linspace(0, 1, 11)
        np.testing.assert_allclose(t.numpy(), expected, rtol=1e-5)

    def test_linspace_steps(self):
        """Test linspace with different number of steps."""
        t = mlx_compat.linspace(0, 10, 5)
        self.assert_shape_equal(t.shape, (5,))

    def test_logspace(self):
        """Test logspace."""
        t = mlx_compat.logspace(0, 3, 4)
        expected = np.array([1, 10, 100, 1000], dtype=np.float32)
        np.testing.assert_allclose(t.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestIdentity(TestCase):
    """Test identity matrix functions."""

    def test_eye_square(self):
        """Test eye for square matrix."""
        t = mlx_compat.eye(3)
        self.assert_shape_equal(t.shape, (3, 3))
        np.testing.assert_array_equal(t.numpy(), np.eye(3))

    def test_eye_rectangular(self):
        """Test eye for rectangular matrix."""
        t = mlx_compat.eye(3, 5)
        self.assert_shape_equal(t.shape, (3, 5))
        expected = np.eye(3, 5)
        np.testing.assert_array_equal(t.numpy(), expected)

    def test_eye_dtype(self):
        """Test eye with specific dtype."""
        t = mlx_compat.eye(3, dtype=mlx_compat.int32)
        self.assertEqual(t.dtype, mlx_compat.int32)


@skipIfNoMLX
class TestRandom(TestCase):
    """Test random tensor generation."""

    def test_randn_shape(self):
        """Test randn shape."""
        t = mlx_compat.randn(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))

    def test_randn_distribution(self):
        """Test that randn produces reasonable values (rough check)."""
        t = mlx_compat.randn(1000)
        mean = t.numpy().mean()
        std = t.numpy().std()
        # Should be roughly N(0, 1)
        self.assertLess(abs(mean), 0.2)  # Mean close to 0
        self.assertLess(abs(std - 1.0), 0.2)  # Std close to 1

    def test_rand_shape(self):
        """Test rand shape."""
        t = mlx_compat.rand(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))

    def test_rand_range(self):
        """Test that rand produces values in [0, 1)."""
        t = mlx_compat.rand(100)
        arr = t.numpy()
        self.assertTrue((arr >= 0).all())
        self.assertTrue((arr < 1).all())

    def test_randint_shape(self):
        """Test randint shape."""
        t = mlx_compat.randint(0, 10, (3, 4))
        self.assert_shape_equal(t.shape, (3, 4))

    def test_randint_range(self):
        """Test that randint produces values in correct range."""
        t = mlx_compat.randint(0, 10, (100,))
        arr = t.numpy()
        self.assertTrue((arr >= 0).all())
        self.assertTrue((arr < 10).all())

    def test_randint_single_arg(self):
        """Test randint with single argument."""
        t = mlx_compat.randint(10, size=(5,))
        self.assert_shape_equal(t.shape, (5,))


@skipIfNoMLX
class TestClone(TestCase):
    """Test tensor cloning."""

    def test_clone(self):
        """Test clone creates a copy."""
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.clone(x)
        self.assert_shape_equal(y.shape, x.shape)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_clone_preserves_requires_grad(self):
        """Test that clone preserves requires_grad."""
        x = mlx_compat.tensor([1.0, 2.0], requires_grad=True)
        y = mlx_compat.clone(x)
        self.assertTrue(y.requires_grad)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
