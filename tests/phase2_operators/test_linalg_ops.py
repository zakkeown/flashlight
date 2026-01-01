"""
Test Phase 2: Linear Algebra Operators (torch.* level)

Tests einsum, tensordot, diag, diagonal, triu, tril, trace, outer, inner, etc.
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@skipIfNoMLX
class TestEinsum(TestCase):
    """Test einsum operations."""

    def test_einsum_matmul(self):
        """Test einsum for matrix multiplication."""
        a = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = flashlight.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = flashlight.einsum('ij,jk->ik', a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)

    def test_einsum_batch_matmul(self):
        """Test einsum for batch matrix multiplication."""
        a = flashlight.randn(2, 3, 4)
        b = flashlight.randn(2, 4, 5)
        c = flashlight.einsum('bij,bjk->bik', a, b)
        self.assertEqual(c.shape, (2, 3, 5))

    def test_einsum_trace(self):
        """Test einsum for trace."""
        a = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        trace = flashlight.einsum('ii->', a)
        self.assertAlmostEqual(trace.numpy().item(), 5.0)

    def test_einsum_outer(self):
        """Test einsum for outer product."""
        a = flashlight.tensor([1.0, 2.0, 3.0])
        b = flashlight.tensor([4.0, 5.0])
        c = flashlight.einsum('i,j->ij', a, b)
        expected = np.array([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]])
        np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_einsum_parity(self):
        """Test parity with PyTorch einsum."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)
        b_data = np.random.randn(4, 5).astype(np.float32)

        mlx_a = flashlight.tensor(a_data)
        mlx_b = flashlight.tensor(b_data)
        torch_a = torch.tensor(a_data)
        torch_b = torch.tensor(b_data)

        mlx_result = flashlight.einsum('ij,jk->ik', mlx_a, mlx_b)
        torch_result = torch.einsum('ij,jk->ik', torch_a, torch_b)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )


@skipIfNoMLX
class TestTensordot(TestCase):
    """Test tensordot operations."""

    def test_tensordot_1d(self):
        """Test tensordot on 1D tensors (dot product)."""
        a = flashlight.tensor([1.0, 2.0, 3.0])
        b = flashlight.tensor([4.0, 5.0, 6.0])
        c = flashlight.tensordot(a, b, dims=1)
        # 1*4 + 2*5 + 3*6 = 32
        self.assertAlmostEqual(c.numpy().item(), 32.0)

    def test_tensordot_matmul(self):
        """Test tensordot for matrix multiplication."""
        a = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = flashlight.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = flashlight.tensordot(a, b, dims=1)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestDiag(TestCase):
    """Test diag operations."""

    def test_diag_vector_to_matrix(self):
        """Test creating diagonal matrix from vector."""
        v = flashlight.tensor([1.0, 2.0, 3.0])
        m = flashlight.diag(v)
        expected = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        np.testing.assert_array_equal(m.numpy(), expected)

    def test_diag_matrix_to_vector(self):
        """Test extracting diagonal from matrix."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        v = flashlight.diag(m)
        expected = np.array([1., 5., 9.])
        np.testing.assert_array_equal(v.numpy(), expected)

    def test_diag_offset_positive(self):
        """Test diag with positive offset."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        v = flashlight.diag(m, diagonal=1)
        expected = np.array([2., 6.])
        np.testing.assert_array_equal(v.numpy(), expected)

    def test_diag_offset_negative(self):
        """Test diag with negative offset."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        v = flashlight.diag(m, diagonal=-1)
        expected = np.array([4., 8.])
        np.testing.assert_array_equal(v.numpy(), expected)


@skipIfNoMLX
class TestDiagonal(TestCase):
    """Test diagonal operations."""

    def test_diagonal_basic(self):
        """Test basic diagonal extraction."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        d = flashlight.diagonal(m)
        expected = np.array([1., 5., 9.])
        np.testing.assert_array_equal(d.numpy(), expected)

    def test_diagonal_offset(self):
        """Test diagonal with offset."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        d = flashlight.diagonal(m, offset=1)
        expected = np.array([2., 6.])
        np.testing.assert_array_equal(d.numpy(), expected)


@skipIfNoMLX
class TestTriuTril(TestCase):
    """Test triu and tril operations."""

    def test_triu_basic(self):
        """Test basic upper triangular."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        u = flashlight.triu(m)
        expected = np.array([[1., 2., 3.], [0., 5., 6.], [0., 0., 9.]])
        np.testing.assert_array_equal(u.numpy(), expected)

    def test_triu_diagonal_1(self):
        """Test upper triangular with diagonal offset 1."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        u = flashlight.triu(m, diagonal=1)
        expected = np.array([[0., 2., 3.], [0., 0., 6.], [0., 0., 0.]])
        np.testing.assert_array_equal(u.numpy(), expected)

    def test_tril_basic(self):
        """Test basic lower triangular."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        l = flashlight.tril(m)
        expected = np.array([[1., 0., 0.], [4., 5., 0.], [7., 8., 9.]])
        np.testing.assert_array_equal(l.numpy(), expected)

    def test_tril_diagonal_neg1(self):
        """Test lower triangular with diagonal offset -1."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        l = flashlight.tril(m, diagonal=-1)
        expected = np.array([[0., 0., 0.], [4., 0., 0.], [7., 8., 0.]])
        np.testing.assert_array_equal(l.numpy(), expected)


@skipIfNoMLX
class TestTrace(TestCase):
    """Test trace operations."""

    def test_trace_basic(self):
        """Test basic trace (sum of diagonal)."""
        m = flashlight.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        t = flashlight.trace(m)
        # 1 + 5 + 9 = 15
        self.assertAlmostEqual(t.numpy().item(), 15.0)

    def test_trace_identity(self):
        """Test trace of identity matrix."""
        m = flashlight.eye(5)
        t = flashlight.trace(m)
        self.assertAlmostEqual(t.numpy().item(), 5.0)


@skipIfNoMLX
class TestOuter(TestCase):
    """Test outer product operations."""

    def test_outer_basic(self):
        """Test basic outer product."""
        a = flashlight.tensor([1., 2., 3.])
        b = flashlight.tensor([4., 5.])
        c = flashlight.outer(a, b)
        expected = np.array([[4., 5.], [8., 10.], [12., 15.]])
        np.testing.assert_array_equal(c.numpy(), expected)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_outer_parity(self):
        """Test parity with PyTorch outer."""
        a_data = np.array([1., 2., 3., 4.], dtype=np.float32)
        b_data = np.array([5., 6., 7.], dtype=np.float32)

        mlx_a = flashlight.tensor(a_data)
        mlx_b = flashlight.tensor(b_data)
        torch_a = torch.tensor(a_data)
        torch_b = torch.tensor(b_data)

        mlx_result = flashlight.outer(mlx_a, mlx_b)
        torch_result = torch.outer(torch_a, torch_b)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )


@skipIfNoMLX
class TestInner(TestCase):
    """Test inner product operations."""

    def test_inner_1d(self):
        """Test inner product of 1D vectors (dot product)."""
        a = flashlight.tensor([1., 2., 3.])
        b = flashlight.tensor([4., 5., 6.])
        c = flashlight.inner(a, b)
        # 1*4 + 2*5 + 3*6 = 32
        self.assertAlmostEqual(c.numpy().item(), 32.0)


@skipIfNoMLX
class TestDot(TestCase):
    """Test dot product operations."""

    def test_dot_1d(self):
        """Test dot product of 1D vectors."""
        a = flashlight.tensor([1., 2., 3.])
        b = flashlight.tensor([4., 5., 6.])
        c = flashlight.dot(a, b)
        # 1*4 + 2*5 + 3*6 = 32
        self.assertAlmostEqual(c.numpy().item(), 32.0)


@skipIfNoMLX
class TestKron(TestCase):
    """Test Kronecker product operations."""

    def test_kron_basic(self):
        """Test basic Kronecker product."""
        a = flashlight.tensor([[1., 2.], [3., 4.]])
        b = flashlight.tensor([[0., 5.], [6., 7.]])
        c = flashlight.kron(a, b)
        # Result is 4x4
        self.assertEqual(c.shape, (4, 4))
        # Check some values
        # kron(A,B)[0,0] = A[0,0] * B[0,0] = 1 * 0 = 0
        # kron(A,B)[0,1] = A[0,0] * B[0,1] = 1 * 5 = 5
        self.assertAlmostEqual(c.numpy()[0, 0], 0.0)
        self.assertAlmostEqual(c.numpy()[0, 1], 5.0)


if __name__ == '__main__':
    unittest.main()
