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
    import flashlight
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestTensorCreation(TestCase):
    """Test basic tensor creation."""

    def test_tensor_from_list(self):
        """Test creating tensor from Python list."""
        # Integer list infers int32 in MLX
        t = flashlight.tensor([1, 2, 3])
        self.assert_shape_equal(t.shape, (3,))
        # MLX infers int32 from integer list, unlike PyTorch which defaults to float32
        self.assertEqual(t.dtype, flashlight.int32)

        # Float list infers float32
        t_float = flashlight.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t_float.dtype, flashlight.float32)

    def test_tensor_from_nested_list(self):
        """Test creating tensor from nested list."""
        t = flashlight.tensor([[1, 2], [3, 4]])
        self.assert_shape_equal(t.shape, (2, 2))

    def test_tensor_from_numpy(self):
        """Test creating tensor from NumPy array."""
        arr = np.array([1.0, 2.0, 3.0])
        t = flashlight.from_numpy(arr)
        self.assert_shape_equal(t.shape, (3,))
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_tensor_with_dtype(self):
        """Test tensor creation with explicit dtype."""
        t = flashlight.tensor([1, 2, 3], dtype=flashlight.int32)
        self.assertEqual(t.dtype, flashlight.int32)

    def test_tensor_with_requires_grad(self):
        """Test tensor creation with requires_grad."""
        t = flashlight.tensor([1.0, 2.0], requires_grad=True)
        self.assertTrue(t.requires_grad)
        self.assertIsNone(t.grad)

    def test_tensor_properties(self):
        """Test tensor properties."""
        t = flashlight.tensor([[1, 2, 3], [4, 5, 6]])
        self.assert_shape_equal(t.shape, (2, 3))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.numel, 6)
        self.assertTrue(t.is_leaf)


@skipIfNoMLX
class TestZerosOnes(TestCase):
    """Test zeros() and ones() factory functions."""

    def test_zeros(self):
        """Test zeros creation."""
        t = flashlight.zeros(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_zeros_with_dtype(self):
        """Test zeros with specific dtype."""
        t = flashlight.zeros(5, dtype=flashlight.int32)
        self.assertEqual(t.dtype, flashlight.int32)

    def test_zeros_tuple_shape(self):
        """Test zeros with tuple shape."""
        t = flashlight.zeros((2, 3, 4))
        self.assert_shape_equal(t.shape, (2, 3, 4))

    def test_ones(self):
        """Test ones creation."""
        t = flashlight.ones(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4)))

    def test_ones_with_dtype(self):
        """Test ones with specific dtype."""
        t = flashlight.ones(5, dtype=flashlight.float16)
        self.assertEqual(t.dtype, flashlight.float16)

    def test_full(self):
        """Test full creation."""
        t = flashlight.full((3, 4), 7)
        self.assert_shape_equal(t.shape, (3, 4))
        np.testing.assert_array_equal(t.numpy(), np.full((3, 4), 7))

    def test_empty_warns(self):
        """Test that empty() issues a warning (MLX doesn't support uninitialized)."""
        with self.assertWarns(UserWarning):
            t = flashlight.empty(3, 4)
        # Should return zeros in MLX
        self.assert_shape_equal(t.shape, (3, 4))


@skipIfNoMLX
class TestLikeVariants(TestCase):
    """Test *_like factory functions."""

    def test_zeros_like(self):
        """Test zeros_like."""
        x = flashlight.randn(3, 4)
        t = flashlight.zeros_like(x)
        self.assert_shape_equal(t.shape, x.shape)
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_ones_like(self):
        """Test ones_like."""
        x = flashlight.randn(2, 3)
        t = flashlight.ones_like(x)
        self.assert_shape_equal(t.shape, x.shape)

    def test_full_like(self):
        """Test full_like."""
        x = flashlight.zeros(3, 4)
        t = flashlight.full_like(x, 5)
        np.testing.assert_array_equal(t.numpy(), np.full((3, 4), 5))

    def test_like_preserves_dtype(self):
        """Test that *_like preserves dtype by default."""
        x = flashlight.zeros(3, dtype=flashlight.int32)
        t = flashlight.ones_like(x)
        self.assertEqual(t.dtype, flashlight.int32)

    def test_like_with_dtype_override(self):
        """Test *_like with dtype override."""
        x = flashlight.zeros(3, dtype=flashlight.int32)
        t = flashlight.ones_like(x, dtype=flashlight.float32)
        self.assertEqual(t.dtype, flashlight.float32)


@skipIfNoMLX
class TestSequences(TestCase):
    """Test sequence generation functions."""

    def test_arange_single_arg(self):
        """Test arange with single argument."""
        t = flashlight.arange(10)
        self.assert_shape_equal(t.shape, (10,))
        np.testing.assert_array_equal(t.numpy(), np.arange(10))

    def test_arange_start_end(self):
        """Test arange with start and end."""
        t = flashlight.arange(2, 10)
        np.testing.assert_array_equal(t.numpy(), np.arange(2, 10))

    def test_arange_with_step(self):
        """Test arange with step."""
        t = flashlight.arange(0, 10, 2)
        np.testing.assert_array_equal(t.numpy(), np.arange(0, 10, 2))

    def test_linspace(self):
        """Test linspace."""
        t = flashlight.linspace(0, 1, 11)
        expected = np.linspace(0, 1, 11)
        np.testing.assert_allclose(t.numpy(), expected, rtol=1e-5)

    def test_linspace_steps(self):
        """Test linspace with different number of steps."""
        t = flashlight.linspace(0, 10, 5)
        self.assert_shape_equal(t.shape, (5,))

    def test_logspace(self):
        """Test logspace."""
        t = flashlight.logspace(0, 3, 4)
        expected = np.array([1, 10, 100, 1000], dtype=np.float32)
        np.testing.assert_allclose(t.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestIdentity(TestCase):
    """Test identity matrix functions."""

    def test_eye_square(self):
        """Test eye for square matrix."""
        t = flashlight.eye(3)
        self.assert_shape_equal(t.shape, (3, 3))
        np.testing.assert_array_equal(t.numpy(), np.eye(3))

    def test_eye_rectangular(self):
        """Test eye for rectangular matrix."""
        t = flashlight.eye(3, 5)
        self.assert_shape_equal(t.shape, (3, 5))
        expected = np.eye(3, 5)
        np.testing.assert_array_equal(t.numpy(), expected)

    def test_eye_dtype(self):
        """Test eye with specific dtype."""
        t = flashlight.eye(3, dtype=flashlight.int32)
        self.assertEqual(t.dtype, flashlight.int32)


@skipIfNoMLX
class TestRandom(TestCase):
    """Test random tensor generation."""

    def test_randn_shape(self):
        """Test randn shape."""
        t = flashlight.randn(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))

    def test_randn_distribution(self):
        """Test that randn produces reasonable values (rough check)."""
        t = flashlight.randn(1000)
        mean = t.numpy().mean()
        std = t.numpy().std()
        # Should be roughly N(0, 1)
        self.assertLess(abs(mean), 0.2)  # Mean close to 0
        self.assertLess(abs(std - 1.0), 0.2)  # Std close to 1

    def test_rand_shape(self):
        """Test rand shape."""
        t = flashlight.rand(3, 4)
        self.assert_shape_equal(t.shape, (3, 4))

    def test_rand_range(self):
        """Test that rand produces values in [0, 1)."""
        t = flashlight.rand(100)
        arr = t.numpy()
        self.assertTrue((arr >= 0).all())
        self.assertTrue((arr < 1).all())

    def test_randint_shape(self):
        """Test randint shape."""
        t = flashlight.randint(0, 10, (3, 4))
        self.assert_shape_equal(t.shape, (3, 4))

    def test_randint_range(self):
        """Test that randint produces values in correct range."""
        t = flashlight.randint(0, 10, (100,))
        arr = t.numpy()
        self.assertTrue((arr >= 0).all())
        self.assertTrue((arr < 10).all())

    def test_randint_single_arg(self):
        """Test randint with single argument."""
        t = flashlight.randint(10, size=(5,))
        self.assert_shape_equal(t.shape, (5,))


@skipIfNoMLX
class TestClone(TestCase):
    """Test tensor cloning."""

    def test_clone(self):
        """Test clone creates a copy."""
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.clone(x)
        self.assert_shape_equal(y.shape, x.shape)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_clone_preserves_requires_grad(self):
        """Test that clone preserves requires_grad."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)
        y = flashlight.clone(x)
        self.assertTrue(y.requires_grad)


@skipIfNoMLX
class TestTriangular(TestCase):
    """Test triangular matrix functions."""

    def test_tril_basic(self):
        """Test lower triangular matrix."""
        x = flashlight.ones(3, 3)
        y = flashlight.tril(x)
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_tril_diagonal(self):
        """Test tril with diagonal offset."""
        x = flashlight.ones(3, 3)
        y = flashlight.tril(x, diagonal=1)
        expected = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_triu_basic(self):
        """Test upper triangular matrix."""
        x = flashlight.ones(3, 3)
        y = flashlight.triu(x)
        expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_triu_diagonal(self):
        """Test triu with diagonal offset."""
        x = flashlight.ones(3, 3)
        y = flashlight.triu(x, diagonal=-1)
        expected = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.numpy(), expected)


@skipIfNoMLX
class TestDiagonal(TestCase):
    """Test diagonal functions."""

    def test_diag_1d_to_2d(self):
        """Test diag creating 2D diagonal matrix from 1D."""
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.diag(x)
        self.assert_shape_equal(y.shape, (3, 3))
        expected = np.diag([1, 2, 3])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diag_2d_to_1d(self):
        """Test diag extracting 1D diagonal from 2D."""
        x = flashlight.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = flashlight.diag(x)
        self.assert_shape_equal(y.shape, (3,))
        expected = np.array([1, 5, 9])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diag_offset(self):
        """Test diag with diagonal offset."""
        x = flashlight.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = flashlight.diag(x, diagonal=1)
        expected = np.array([2, 6])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diagflat(self):
        """Test diagflat function."""
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.diagflat(x)
        self.assert_shape_equal(y.shape, (3, 3))


@skipIfNoMLX
class TestMeshgrid(TestCase):
    """Test meshgrid function."""

    def test_meshgrid_2d(self):
        """Test 2D meshgrid with ij indexing."""
        x = flashlight.arange(3)
        y = flashlight.arange(4)
        # Default is 'ij' indexing in flashlight
        X, Y = flashlight.meshgrid(x, y, indexing='ij')
        self.assert_shape_equal(X.shape, (3, 4))
        self.assert_shape_equal(Y.shape, (3, 4))

    def test_meshgrid_indexing_xy(self):
        """Test meshgrid with xy indexing."""
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.tensor([4, 5])
        X, Y = flashlight.meshgrid(x, y, indexing='xy')
        self.assert_shape_equal(X.shape, (2, 3))

    def test_meshgrid_indexing_ij(self):
        """Test meshgrid with ij indexing."""
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.tensor([4, 5])
        X, Y = flashlight.meshgrid(x, y, indexing='ij')
        self.assert_shape_equal(X.shape, (3, 2))


@skipIfNoMLX
class TestWhere(TestCase):
    """Test where function."""

    def test_where_basic(self):
        """Test basic where operation."""
        condition = flashlight.tensor([True, False, True])
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.tensor([4, 5, 6])
        result = flashlight.where(condition, x, y)
        np.testing.assert_array_equal(result.numpy(), [1, 5, 3])

    def test_where_broadcast(self):
        """Test where with broadcasting."""
        condition = flashlight.tensor([[True], [False]])
        x = flashlight.tensor([1, 2, 3])
        y = flashlight.tensor([4, 5, 6])
        result = flashlight.where(condition, x, y)
        self.assert_shape_equal(result.shape, (2, 3))


@skipIfNoMLX
class TestRandomDistributions(TestCase):
    """Test random distribution functions."""

    def test_normal_basic(self):
        """Test normal distribution."""
        result = flashlight.normal(0, 1, (100,))
        self.assert_shape_equal(result.shape, (100,))
        # Should be roughly N(0, 1)
        self.assertLess(abs(result.numpy().mean()), 0.3)

    def test_normal_with_mean_std(self):
        """Test normal with custom mean and std."""
        result = flashlight.normal(5, 2, (1000,))
        mean = result.numpy().mean()
        self.assertLess(abs(mean - 5), 0.5)

    def test_bernoulli_basic(self):
        """Test Bernoulli distribution."""
        probs = flashlight.tensor([0.3, 0.5, 0.7])
        result = flashlight.bernoulli(probs)
        self.assert_shape_equal(result.shape, (3,))
        # Values should be 0 or 1
        self.assertTrue(((result.numpy() == 0) | (result.numpy() == 1)).all())

    def test_multinomial_basic(self):
        """Test multinomial sampling."""
        probs = flashlight.tensor([0.5, 0.3, 0.2])
        samples = flashlight.multinomial(probs, 100, replacement=True)
        self.assert_shape_equal(samples.shape, (100,))
        # All samples should be in [0, 1, 2]
        self.assertTrue((samples.numpy() >= 0).all())
        self.assertTrue((samples.numpy() < 3).all())

    def test_randperm(self):
        """Test random permutation."""
        result = flashlight.randperm(10)
        self.assert_shape_equal(result.shape, (10,))
        # Should be a permutation of 0-9
        sorted_result = np.sort(result.numpy())
        np.testing.assert_array_equal(sorted_result, np.arange(10))


@skipIfNoMLX
class TestRandomLike(TestCase):
    """Test random *_like functions."""

    def test_rand_like(self):
        """Test rand_like function."""
        x = flashlight.zeros(3, 4)
        y = flashlight.rand_like(x)
        self.assert_shape_equal(y.shape, x.shape)
        self.assertTrue((y.numpy() >= 0).all())
        self.assertTrue((y.numpy() < 1).all())

    def test_randn_like(self):
        """Test randn_like function."""
        x = flashlight.zeros(3, 4)
        y = flashlight.randn_like(x)
        self.assert_shape_equal(y.shape, x.shape)

    def test_randint_like(self):
        """Test randint_like function."""
        x = flashlight.zeros(3, 4, dtype=flashlight.int32)
        y = flashlight.randint_like(x, 0, 10)
        self.assert_shape_equal(y.shape, x.shape)
        self.assertTrue((y.numpy() >= 0).all())
        self.assertTrue((y.numpy() < 10).all())


@skipIfNoMLX
class TestAsTensor(TestCase):
    """Test as_tensor and scalar_tensor functions."""

    def test_as_tensor_from_list(self):
        """Test as_tensor from list."""
        result = flashlight.as_tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])

    def test_as_tensor_from_numpy(self):
        """Test as_tensor from numpy array."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = flashlight.as_tensor(arr)
        np.testing.assert_array_equal(result.numpy(), arr)

    def test_scalar_tensor(self):
        """Test scalar_tensor function."""
        result = flashlight.scalar_tensor(3.14)
        self.assertEqual(result.ndim, 0)
        self.assertAlmostEqual(result.item(), 3.14, places=5)


@skipIfNoMLX
class TestCartesianProd(TestCase):
    """Test cartesian_prod function."""

    def test_cartesian_prod_2d(self):
        """Test cartesian product of two tensors."""
        a = flashlight.tensor([1, 2])
        b = flashlight.tensor([3, 4, 5])
        result = flashlight.cartesian_prod(a, b)
        # Result should have 2*3 = 6 rows, 2 columns
        self.assert_shape_equal(result.shape, (6, 2))


@skipIfNoMLX
class TestPoisson(TestCase):
    """Test Poisson distribution."""

    def test_poisson_basic(self):
        """Test Poisson distribution."""
        rate = flashlight.tensor([5.0, 10.0])
        result = flashlight.poisson(rate)
        self.assert_shape_equal(result.shape, (2,))
        # All values should be non-negative integers
        self.assertTrue((result.numpy() >= 0).all())

    def test_poisson_scalar(self):
        """Test Poisson with scalar rate."""
        result = flashlight.poisson(flashlight.tensor([3.0] * 100))
        mean = result.numpy().mean()
        # Mean should be close to rate (3.0)
        self.assertLess(abs(mean - 3.0), 1.0)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
