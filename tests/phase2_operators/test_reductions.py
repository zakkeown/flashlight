"""
Test Phase 2: Reduction Operations

Tests reduction operations (sum, mean, max, min, argmax, var, std, etc.)
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
class TestSum(TestCase):
    """Test sum reduction."""

    def test_sum_all(self):
        """Test sum of all elements."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.sum(x)
        self.assertAlmostEqual(result.item(), 10.0, places=5)

    def test_sum_along_dim(self):
        """Test sum along a dimension."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.sum(x, dim=0)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([4.0, 6.0]))

    def test_sum_keepdim(self):
        """Test sum with keepdim."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.sum(x, dim=1, keepdim=True)
        self.assert_shape_equal(result.shape, (2, 1))
        np.testing.assert_array_almost_equal(result.numpy(), np.array([[3.0], [7.0]]))

    def test_sum_requires_grad(self):
        """Test that sum propagates requires_grad."""
        x = flashlight.tensor([1.0, 2.0], requires_grad=True)
        result = flashlight.sum(x)
        self.assertTrue(result.requires_grad)


@skipIfNoMLX
class TestMean(TestCase):
    """Test mean reduction."""

    def test_mean_all(self):
        """Test mean of all elements."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.mean(x)
        self.assertAlmostEqual(result.item(), 2.5, places=5)

    def test_mean_along_dim(self):
        """Test mean along a dimension."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.mean(x, dim=0)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([2.0, 3.0]))

    def test_mean_keepdim(self):
        """Test mean with keepdim."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = flashlight.mean(x, dim=1, keepdim=True)
        self.assert_shape_equal(result.shape, (2, 1))
        np.testing.assert_array_almost_equal(result.numpy(), np.array([[2.0], [5.0]]))


@skipIfNoMLX
class TestMax(TestCase):
    """Test max reduction."""

    def test_max_all(self):
        """Test max of all elements."""
        x = flashlight.tensor([1.0, 5.0, 3.0, 2.0])
        result = flashlight.max(x)
        self.assertAlmostEqual(result.item(), 5.0, places=5)

    def test_max_along_dim(self):
        """Test max along a dimension."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        values, indices = flashlight.max(x, dim=1)
        np.testing.assert_array_almost_equal(values.numpy(), np.array([5.0, 3.0]))
        np.testing.assert_array_equal(indices.numpy(), np.array([1, 0]))

    def test_max_keepdim(self):
        """Test max with keepdim."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        values, indices = flashlight.max(x, dim=1, keepdim=True)
        self.assert_shape_equal(values.shape, (2, 1))
        self.assert_shape_equal(indices.shape, (2, 1))


@skipIfNoMLX
class TestMin(TestCase):
    """Test min reduction."""

    def test_min_all(self):
        """Test min of all elements."""
        x = flashlight.tensor([1.0, 5.0, 3.0, 2.0])
        result = flashlight.min(x)
        self.assertAlmostEqual(result.item(), 1.0, places=5)

    def test_min_along_dim(self):
        """Test min along a dimension."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        values, indices = flashlight.min(x, dim=1)
        np.testing.assert_array_almost_equal(values.numpy(), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(indices.numpy(), np.array([0, 1]))

    def test_min_keepdim(self):
        """Test min with keepdim."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        values, indices = flashlight.min(x, dim=1, keepdim=True)
        self.assert_shape_equal(values.shape, (2, 1))


@skipIfNoMLX
class TestArgmax(TestCase):
    """Test argmax reduction."""

    def test_argmax_all(self):
        """Test argmax of flattened tensor."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = flashlight.argmax(x)
        # Should be index 1 in flattened array
        self.assertEqual(result.item(), 1)

    def test_argmax_along_dim(self):
        """Test argmax along a dimension."""
        x = flashlight.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        result = flashlight.argmax(x, dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 2]))

    def test_argmax_keepdim(self):
        """Test argmax with keepdim."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = flashlight.argmax(x, dim=1, keepdim=True)
        self.assert_shape_equal(result.shape, (2, 1))


@skipIfNoMLX
class TestArgmin(TestCase):
    """Test argmin reduction."""

    def test_argmin_all(self):
        """Test argmin of flattened tensor."""
        x = flashlight.tensor([[5.0, 1.0], [3.0, 2.0]])
        result = flashlight.argmin(x)
        # Should be index 1 in flattened array
        self.assertEqual(result.item(), 1)

    def test_argmin_along_dim(self):
        """Test argmin along a dimension."""
        x = flashlight.tensor([[5.0, 1.0, 3.0], [4.0, 6.0, 2.0]])
        result = flashlight.argmin(x, dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 2]))

    def test_argmin_keepdim(self):
        """Test argmin with keepdim."""
        x = flashlight.tensor([[5.0, 1.0], [3.0, 2.0]])
        result = flashlight.argmin(x, dim=1, keepdim=True)
        self.assert_shape_equal(result.shape, (2, 1))


@skipIfNoMLX
class TestVar(TestCase):
    """Test variance reduction."""

    def test_var_all(self):
        """Test variance of all elements."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.var(x)
        # Variance with Bessel's correction: 1.666...
        expected = np.var([1.0, 2.0, 3.0, 4.0], ddof=1)
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_var_biased(self):
        """Test biased variance."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.var(x, unbiased=False)
        expected = np.var([1.0, 2.0, 3.0, 4.0], ddof=0)
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_var_along_dim(self):
        """Test variance along a dimension."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = flashlight.var(x, dim=1)
        expected = np.var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], axis=1, ddof=1)
        np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=4)


@skipIfNoMLX
class TestStd(TestCase):
    """Test standard deviation reduction."""

    def test_std_all(self):
        """Test std of all elements."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.std(x)
        expected = np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
        self.assertAlmostEqual(result.item(), expected, places=4)

    def test_std_biased(self):
        """Test biased std."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = flashlight.std(x, unbiased=False)
        expected = np.std([1.0, 2.0, 3.0, 4.0], ddof=0)
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_std_along_dim(self):
        """Test std along a dimension."""
        x = flashlight.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = flashlight.std(x, dim=1)
        expected = np.std([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], axis=1, ddof=1)
        np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=4)


@skipIfNoMLX
class TestProd(TestCase):
    """Test product reduction."""

    def test_prod_all(self):
        """Test product of all elements."""
        x = flashlight.tensor([2.0, 3.0, 4.0])
        result = flashlight.prod(x)
        self.assertAlmostEqual(result.item(), 24.0, places=5)

    def test_prod_along_dim(self):
        """Test product along a dimension."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = flashlight.prod(x, dim=1)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([2.0, 12.0]))


@skipIfNoMLX
class TestAll(TestCase):
    """Test all reduction."""

    def test_all_true(self):
        """Test all with all True values."""
        x = flashlight.tensor([True, True, True])
        result = flashlight.all(x)
        self.assertTrue(result.item())

    def test_all_false(self):
        """Test all with some False values."""
        x = flashlight.tensor([True, False, True])
        result = flashlight.all(x)
        self.assertFalse(result.item())

    def test_all_along_dim(self):
        """Test all along a dimension."""
        x = flashlight.tensor([[True, True], [True, False]])
        result = flashlight.all(x, dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False]))


@skipIfNoMLX
class TestAny(TestCase):
    """Test any reduction."""

    def test_any_true(self):
        """Test any with some True values."""
        x = flashlight.tensor([False, True, False])
        result = flashlight.any(x)
        self.assertTrue(result.item())

    def test_any_false(self):
        """Test any with all False values."""
        x = flashlight.tensor([False, False, False])
        result = flashlight.any(x)
        self.assertFalse(result.item())

    def test_any_along_dim(self):
        """Test any along a dimension."""
        x = flashlight.tensor([[False, False], [True, False]])
        result = flashlight.any(x, dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([False, True]))


@skipIfNoMLX
class TestTensorMethods(TestCase):
    """Test tensor instance methods for reductions (tensor.sum() vs flashlight.sum(tensor))."""

    def test_tensor_sum(self):
        """Test tensor.sum() method."""
        x = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        result = x.sum()
        self.assertAlmostEqual(result.item(), 10.0, places=5)

    def test_tensor_sum_dim(self):
        """Test tensor.sum(dim) method."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = x.sum(dim=0)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([4.0, 6.0]))

    def test_tensor_sum_keepdim(self):
        """Test tensor.sum(keepdim=True) method."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = x.sum(dim=1, keepdim=True)
        self.assert_shape_equal(result.shape, (2, 1))

    def test_tensor_prod(self):
        """Test tensor.prod() method."""
        x = flashlight.tensor([2.0, 3.0, 4.0])
        result = x.prod()
        self.assertAlmostEqual(result.item(), 24.0, places=5)

    def test_tensor_prod_dim(self):
        """Test tensor.prod(dim) method."""
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = x.prod(dim=1)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([2.0, 12.0]))

    def test_tensor_max_no_dim(self):
        """Test tensor.max() returns single tensor."""
        x = flashlight.tensor([1.0, 5.0, 3.0])
        result = x.max()
        self.assertAlmostEqual(result.item(), 5.0, places=5)

    def test_tensor_max_with_dim(self):
        """Test tensor.max(dim) returns tuple."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        values, indices = x.max(dim=1)
        np.testing.assert_array_almost_equal(values.numpy(), np.array([5.0, 3.0]))
        np.testing.assert_array_equal(indices.numpy(), np.array([1, 0]))

    def test_tensor_min_no_dim(self):
        """Test tensor.min() returns single tensor."""
        x = flashlight.tensor([1.0, 5.0, 3.0])
        result = x.min()
        self.assertAlmostEqual(result.item(), 1.0, places=5)

    def test_tensor_min_with_dim(self):
        """Test tensor.min(dim) returns tuple."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        values, indices = x.min(dim=1)
        np.testing.assert_array_almost_equal(values.numpy(), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(indices.numpy(), np.array([0, 1]))

    def test_tensor_argmax(self):
        """Test tensor.argmax() method."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = x.argmax()
        self.assertEqual(result.item(), 1)

    def test_tensor_argmax_dim(self):
        """Test tensor.argmax(dim) method."""
        x = flashlight.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = x.argmax(dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 0]))

    def test_tensor_argmin(self):
        """Test tensor.argmin() method."""
        x = flashlight.tensor([[5.0, 1.0], [3.0, 2.0]])
        result = x.argmin()
        self.assertEqual(result.item(), 1)

    def test_tensor_argmin_dim(self):
        """Test tensor.argmin(dim) method."""
        x = flashlight.tensor([[5.0, 1.0], [3.0, 2.0]])
        result = x.argmin(dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([1, 1]))

    def test_tensor_all(self):
        """Test tensor.all() method."""
        x = flashlight.tensor([True, True, True])
        self.assertTrue(x.all().item())
        y = flashlight.tensor([True, False, True])
        self.assertFalse(y.all().item())

    def test_tensor_all_dim(self):
        """Test tensor.all(dim) method."""
        x = flashlight.tensor([[True, True], [True, False]])
        result = x.all(dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False]))

    def test_tensor_any(self):
        """Test tensor.any() method."""
        x = flashlight.tensor([False, True, False])
        self.assertTrue(x.any().item())
        y = flashlight.tensor([False, False, False])
        self.assertFalse(y.any().item())

    def test_tensor_any_dim(self):
        """Test tensor.any(dim) method."""
        x = flashlight.tensor([[False, False], [True, False]])
        result = x.any(dim=1)
        np.testing.assert_array_equal(result.numpy(), np.array([False, True]))

    def test_tensor_abs(self):
        """Test tensor.abs() method."""
        x = flashlight.tensor([-1.0, 2.0, -3.0])
        result = x.abs()
        np.testing.assert_array_almost_equal(result.numpy(), np.array([1.0, 2.0, 3.0]))

    def test_tensor_clamp(self):
        """Test tensor.clamp() method."""
        x = flashlight.tensor([-1.0, 0.5, 2.0])
        result = x.clamp(min=0.0, max=1.0)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([0.0, 0.5, 1.0]))

    def test_tensor_clamp_min_only(self):
        """Test tensor.clamp(min=...) method."""
        x = flashlight.tensor([-1.0, 0.5, 2.0])
        result = x.clamp(min=0.0)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([0.0, 0.5, 2.0]))

    def test_tensor_clamp_max_only(self):
        """Test tensor.clamp(max=...) method."""
        x = flashlight.tensor([-1.0, 0.5, 2.0])
        result = x.clamp(max=1.0)
        np.testing.assert_array_almost_equal(result.numpy(), np.array([-1.0, 0.5, 1.0]))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
