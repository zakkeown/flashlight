"""
Test Phase 1: Dtype and Device

Tests dtype system and device management.
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
class TestDtype(TestCase):
    """Test dtype system."""

    def test_dtype_attributes(self):
        """Test that dtypes exist."""
        self.assertIsNotNone(mlx_compat.float32)
        self.assertIsNotNone(mlx_compat.float16)
        self.assertIsNotNone(mlx_compat.int32)
        self.assertIsNotNone(mlx_compat.int64)
        self.assertIsNotNone(mlx_compat.bool)

    def test_dtype_aliases(self):
        """Test dtype aliases."""
        self.assertEqual(mlx_compat.float, mlx_compat.float32)
        self.assertEqual(mlx_compat.half, mlx_compat.float16)
        self.assertEqual(mlx_compat.int, mlx_compat.int32)
        self.assertEqual(mlx_compat.long, mlx_compat.int64)

    def test_tensor_dtype(self):
        """Test tensor dtype property."""
        t = mlx_compat.tensor([1, 2, 3], dtype=mlx_compat.int32)
        self.assertEqual(t.dtype, mlx_compat.int32)

    def test_dtype_conversion_float(self):
        """Test dtype conversion to float."""
        t = mlx_compat.tensor([1, 2, 3], dtype=mlx_compat.int32)
        t_float = t.float()
        self.assertEqual(t_float.dtype, mlx_compat.float32)

    def test_dtype_conversion_int(self):
        """Test dtype conversion to int."""
        t = mlx_compat.tensor([1.5, 2.7, 3.2])
        t_int = t.int()
        self.assertEqual(t_int.dtype, mlx_compat.int32)

    def test_dtype_conversion_long(self):
        """Test dtype conversion to long."""
        t = mlx_compat.tensor([1, 2, 3])
        t_long = t.long()
        self.assertEqual(t_long.dtype, mlx_compat.int64)

    def test_dtype_conversion_half(self):
        """Test dtype conversion to half."""
        t = mlx_compat.tensor([1.0, 2.0, 3.0])
        t_half = t.half()
        self.assertEqual(t_half.dtype, mlx_compat.float16)

    def test_dtype_conversion_bool(self):
        """Test dtype conversion to bool."""
        t = mlx_compat.tensor([0, 1, 2])
        t_bool = t.bool()
        self.assertEqual(t_bool.dtype, mlx_compat.bool)

    def test_to_dtype(self):
        """Test .to() with dtype."""
        t = mlx_compat.tensor([1, 2, 3])
        t_int = t.to(dtype=mlx_compat.int32)
        self.assertEqual(t_int.dtype, mlx_compat.int32)

    def test_type_method(self):
        """Test .type() method."""
        t = mlx_compat.tensor([1.0, 2.0, 3.0])
        t_int = t.type(mlx_compat.int32)
        self.assertEqual(t_int.dtype, mlx_compat.int32)

    def test_float64_warning(self):
        """Test that float64 issues a warning."""
        with self.assertWarns(UserWarning):
            t = mlx_compat.tensor([1, 2, 3], dtype='float64')
        # Should fall back to float32
        self.assertEqual(t.dtype, mlx_compat.float32)


@skipIfNoMLX
class TestDevice(TestCase):
    """Test device management."""

    def test_device_creation(self):
        """Test Device creation."""
        dev = mlx_compat.Device('cpu')
        self.assertEqual(dev.type, 'cpu')

    def test_device_cuda(self):
        """Test CUDA device (warns about unified memory)."""
        dev = mlx_compat.Device('cuda')
        self.assertEqual(dev.type, 'cuda')

    def test_device_with_index(self):
        """Test device with index."""
        dev = mlx_compat.Device('cuda:0')
        self.assertEqual(dev.type, 'cuda')
        self.assertEqual(dev.index, 0)

    def test_tensor_device(self):
        """Test tensor device property."""
        t = mlx_compat.tensor([1, 2, 3])
        self.assertIsNotNone(t.device)

    def test_to_device(self):
        """Test .to() with device."""
        t = mlx_compat.tensor([1, 2, 3])
        t_cuda = t.to(device='cuda')
        self.assertEqual(t_cuda.device.type, 'cuda')

    def test_cuda_method(self):
        """Test .cuda() method."""
        t = mlx_compat.tensor([1, 2, 3])
        t_cuda = t.cuda()
        self.assertEqual(t_cuda.device.type, 'cuda')

    def test_cpu_method(self):
        """Test .cpu() method."""
        t = mlx_compat.tensor([1, 2, 3])
        t_cpu = t.cpu()
        self.assertEqual(t_cpu.device.type, 'cpu')

    def test_device_count(self):
        """Test device_count() function."""
        count = mlx_compat.device_count()
        # Should return 1 if MLX is available
        self.assertGreaterEqual(count, 0)

    def test_is_available(self):
        """Test is_available() function."""
        available = mlx_compat.is_available()
        # Should be True if this test is running
        self.assertTrue(available)


@skipIfNoMLX
class TestTensorConversions(TestCase):
    """Test tensor type conversions."""

    def test_numpy_conversion(self):
        """Test .numpy() conversion."""
        t = mlx_compat.tensor([1, 2, 3])
        arr = t.numpy()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, np.array([1, 2, 3]))

    def test_tolist_conversion(self):
        """Test .tolist() conversion."""
        t = mlx_compat.tensor([[1, 2], [3, 4]])
        lst = t.tolist()
        self.assertEqual(lst, [[1, 2], [3, 4]])

    def test_item_single_element(self):
        """Test .item() for single element tensor."""
        t = mlx_compat.tensor([42])
        val = t.item()
        self.assertEqual(val, 42)

    def test_item_scalar(self):
        """Test .item() for scalar tensor."""
        t = mlx_compat.tensor(3.14)
        val = t.item()
        self.assertAlmostEqual(val, 3.14, places=5)

    def test_item_multiple_elements_raises(self):
        """Test that .item() raises for multi-element tensors."""
        t = mlx_compat.tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            t.item()


@skipIfNoMLX
class TestOperatorOverloading(TestCase):
    """Test operator overloading."""

    def test_add(self):
        """Test addition operator."""
        a = mlx_compat.tensor([1, 2, 3])
        b = mlx_compat.tensor([4, 5, 6])
        c = a + b
        np.testing.assert_array_equal(c.numpy(), np.array([5, 7, 9]))

    def test_add_scalar(self):
        """Test addition with scalar."""
        a = mlx_compat.tensor([1, 2, 3])
        c = a + 10
        np.testing.assert_array_equal(c.numpy(), np.array([11, 12, 13]))

    def test_sub(self):
        """Test subtraction operator."""
        a = mlx_compat.tensor([5, 6, 7])
        b = mlx_compat.tensor([1, 2, 3])
        c = a - b
        np.testing.assert_array_equal(c.numpy(), np.array([4, 4, 4]))

    def test_mul(self):
        """Test multiplication operator."""
        a = mlx_compat.tensor([1, 2, 3])
        b = mlx_compat.tensor([2, 3, 4])
        c = a * b
        np.testing.assert_array_equal(c.numpy(), np.array([2, 6, 12]))

    def test_div(self):
        """Test division operator."""
        a = mlx_compat.tensor([6.0, 8.0, 10.0])
        b = mlx_compat.tensor([2.0, 4.0, 5.0])
        c = a / b
        np.testing.assert_array_almost_equal(c.numpy(), np.array([3.0, 2.0, 2.0]))

    def test_matmul(self):
        """Test matrix multiplication operator."""
        a = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = mlx_compat.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.numpy(), expected)

    def test_neg(self):
        """Test negation operator."""
        a = mlx_compat.tensor([1, -2, 3])
        b = -a
        np.testing.assert_array_equal(b.numpy(), np.array([-1, 2, -3]))

    def test_comparison_eq(self):
        """Test equality comparison."""
        a = mlx_compat.tensor([1, 2, 3])
        b = mlx_compat.tensor([1, 0, 3])
        c = a == b
        np.testing.assert_array_equal(c.numpy(), np.array([True, False, True]))

    def test_comparison_lt(self):
        """Test less than comparison."""
        a = mlx_compat.tensor([1, 2, 3])
        b = mlx_compat.tensor([2, 2, 1])
        c = a < b
        np.testing.assert_array_equal(c.numpy(), np.array([True, False, False]))


@skipIfNoMLX
class TestIndexing(TestCase):
    """Test tensor indexing."""

    def test_basic_indexing(self):
        """Test basic indexing."""
        t = mlx_compat.tensor([[1, 2, 3], [4, 5, 6]])
        elem = t[0, 1]
        self.assertEqual(elem.item(), 2)

    def test_slice_indexing(self):
        """Test slice indexing."""
        t = mlx_compat.arange(10)
        sliced = t[2:5]
        np.testing.assert_array_equal(sliced.numpy(), np.array([2, 3, 4]))

    def test_indexing_creates_view(self):
        """Test that indexing creates a view."""
        t = mlx_compat.arange(10)
        view = t[2:5]
        self.assertTrue(view.is_view)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
