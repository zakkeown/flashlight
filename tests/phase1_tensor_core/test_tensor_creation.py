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


@skipIfNoMLX
class TestTriangular(TestCase):
    """Test triangular matrix functions."""

    def test_tril_basic(self):
        """Test lower triangular matrix."""
        x = mlx_compat.ones(3, 3)
        y = mlx_compat.tril(x)
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_tril_diagonal(self):
        """Test tril with diagonal offset."""
        x = mlx_compat.ones(3, 3)
        y = mlx_compat.tril(x, diagonal=1)
        expected = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_triu_basic(self):
        """Test upper triangular matrix."""
        x = mlx_compat.ones(3, 3)
        y = mlx_compat.triu(x)
        expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_triu_diagonal(self):
        """Test triu with diagonal offset."""
        x = mlx_compat.ones(3, 3)
        y = mlx_compat.triu(x, diagonal=-1)
        expected = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestDiagonal(TestCase):
    """Test diagonal functions."""

    def test_diag_1d_to_2d(self):
        """Test diag creating 2D diagonal matrix from 1D."""
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.diag(x)
        self.assert_shape_equal(y.shape, (3, 3))
        expected = np.diag([1, 2, 3])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diag_2d_to_1d(self):
        """Test diag extracting 1D diagonal from 2D."""
        x = mlx_compat.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = mlx_compat.diag(x)
        self.assert_shape_equal(y.shape, (3,))
        expected = np.array([1, 5, 9])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diag_offset(self):
        """Test diag with diagonal offset."""
        x = mlx_compat.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = mlx_compat.diag(x, diagonal=1)
        expected = np.array([2, 6])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diagflat(self):
        """Test diagflat function."""
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.diagflat(x)
        self.assert_shape_equal(y.shape, (3, 3))


@skipIfNoMLX
class TestMeshgrid(TestCase):
    """Test meshgrid function."""

    def test_meshgrid_2d(self):
        """Test 2D meshgrid with ij indexing."""
        x = mlx_compat.arange(3)
        y = mlx_compat.arange(4)
        # Default is 'ij' indexing in mlx_compat
        X, Y = mlx_compat.meshgrid(x, y, indexing='ij')
        self.assert_shape_equal(X.shape, (3, 4))
        self.assert_shape_equal(Y.shape, (3, 4))

    def test_meshgrid_indexing_xy(self):
        """Test meshgrid with xy indexing."""
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.tensor([4, 5])
        X, Y = mlx_compat.meshgrid(x, y, indexing='xy')
        self.assert_shape_equal(X.shape, (2, 3))

    def test_meshgrid_indexing_ij(self):
        """Test meshgrid with ij indexing."""
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.tensor([4, 5])
        X, Y = mlx_compat.meshgrid(x, y, indexing='ij')
        self.assert_shape_equal(X.shape, (3, 2))


@skipIfNoMLX
class TestWhere(TestCase):
    """Test where function."""

    def test_where_basic(self):
        """Test basic where operation."""
        condition = mlx_compat.tensor([True, False, True])
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.tensor([4, 5, 6])
        result = mlx_compat.where(condition, x, y)
        np.testing.assert_array_equal(result.numpy(), [1, 5, 3])

    def test_where_broadcast(self):
        """Test where with broadcasting."""
        condition = mlx_compat.tensor([[True], [False]])
        x = mlx_compat.tensor([1, 2, 3])
        y = mlx_compat.tensor([4, 5, 6])
        result = mlx_compat.where(condition, x, y)
        self.assert_shape_equal(result.shape, (2, 3))


@skipIfNoMLX
class TestRandomDistributions(TestCase):
    """Test random distribution functions."""

    def test_normal_basic(self):
        """Test normal distribution."""
        result = mlx_compat.normal(0, 1, (100,))
        self.assert_shape_equal(result.shape, (100,))
        # Should be roughly N(0, 1)
        self.assertLess(abs(result.numpy().mean()), 0.3)

    def test_normal_with_mean_std(self):
        """Test normal with custom mean and std."""
        result = mlx_compat.normal(5, 2, (1000,))
        mean = result.numpy().mean()
        self.assertLess(abs(mean - 5), 0.5)

    def test_bernoulli_basic(self):
        """Test Bernoulli distribution."""
        probs = mlx_compat.tensor([0.3, 0.5, 0.7])
        result = mlx_compat.bernoulli(probs)
        self.assert_shape_equal(result.shape, (3,))
        # Values should be 0 or 1
        self.assertTrue(((result.numpy() == 0) | (result.numpy() == 1)).all())

    def test_multinomial_basic(self):
        """Test multinomial sampling."""
        probs = mlx_compat.tensor([0.5, 0.3, 0.2])
        samples = mlx_compat.multinomial(probs, 100, replacement=True)
        self.assert_shape_equal(samples.shape, (100,))
        # All samples should be in [0, 1, 2]
        self.assertTrue((samples.numpy() >= 0).all())
        self.assertTrue((samples.numpy() < 3).all())

    def test_randperm(self):
        """Test random permutation."""
        result = mlx_compat.randperm(10)
        self.assert_shape_equal(result.shape, (10,))
        # Should be a permutation of 0-9
        sorted_result = np.sort(result.numpy())
        np.testing.assert_array_equal(sorted_result, np.arange(10))


@skipIfNoMLX
class TestRandomLike(TestCase):
    """Test random *_like functions."""

    def test_rand_like(self):
        """Test rand_like function."""
        x = mlx_compat.zeros(3, 4)
        y = mlx_compat.rand_like(x)
        self.assert_shape_equal(y.shape, x.shape)
        self.assertTrue((y.numpy() >= 0).all())
        self.assertTrue((y.numpy() < 1).all())

    def test_randn_like(self):
        """Test randn_like function."""
        x = mlx_compat.zeros(3, 4)
        y = mlx_compat.randn_like(x)
        self.assert_shape_equal(y.shape, x.shape)

    def test_randint_like(self):
        """Test randint_like function."""
        x = mlx_compat.zeros(3, 4, dtype=mlx_compat.int32)
        y = mlx_compat.randint_like(x, 0, 10)
        self.assert_shape_equal(y.shape, x.shape)
        self.assertTrue((y.numpy() >= 0).all())
        self.assertTrue((y.numpy() < 10).all())


@skipIfNoMLX
class TestAsTensor(TestCase):
    """Test as_tensor and scalar_tensor functions."""

    def test_as_tensor_from_list(self):
        """Test as_tensor from list."""
        result = mlx_compat.as_tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])

    def test_as_tensor_from_numpy(self):
        """Test as_tensor from numpy array."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = mlx_compat.as_tensor(arr)
        np.testing.assert_array_equal(result.numpy(), arr)

    def test_scalar_tensor(self):
        """Test scalar_tensor function."""
        result = mlx_compat.scalar_tensor(3.14)
        self.assertEqual(result.ndim, 0)
        self.assertAlmostEqual(result.item(), 3.14, places=5)


@skipIfNoMLX
class TestCartesianProd(TestCase):
    """Test cartesian_prod function."""

    def test_cartesian_prod_2d(self):
        """Test cartesian product of two tensors."""
        a = mlx_compat.tensor([1, 2])
        b = mlx_compat.tensor([3, 4, 5])
        result = mlx_compat.cartesian_prod(a, b)
        # Result should have 2*3 = 6 rows, 2 columns
        self.assert_shape_equal(result.shape, (6, 2))


@skipIfNoMLX
class TestPoisson(TestCase):
    """Test Poisson distribution."""

    def test_poisson_basic(self):
        """Test Poisson distribution."""
        rate = mlx_compat.tensor([5.0, 10.0])
        result = mlx_compat.poisson(rate)
        self.assert_shape_equal(result.shape, (2,))
        # All values should be non-negative integers
        self.assertTrue((result.numpy() >= 0).all())

    def test_poisson_scalar(self):
        """Test Poisson with scalar rate."""
        result = mlx_compat.poisson(mlx_compat.tensor([3.0] * 100))
        mean = result.numpy().mean()
        # Mean should be close to rate (3.0)
        self.assertLess(abs(mean - 3.0), 1.0)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
