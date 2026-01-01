"""
Test Phase 2: torch.linalg Namespace

Tests torch.linalg functions (norm, svd, qr, cholesky, inv, solve, eig, etc.)
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@skipIfNoMLX
class TestLinalgNorm(TestCase):
    """Test linalg.norm operations."""

    def test_norm_frobenius(self):
        """Test Frobenius norm (default for matrices)."""
        m = mlx_compat.tensor([[1., 2.], [3., 4.]])
        n = mlx_compat.linalg.norm(m)
        # sqrt(1 + 4 + 9 + 16) = sqrt(30) = 5.477...
        expected = np.sqrt(30)
        self.assertAlmostEqual(n.numpy().item(), expected, places=4)

    def test_norm_vector_2(self):
        """Test 2-norm for vectors."""
        v = mlx_compat.tensor([3., 4.])
        n = mlx_compat.linalg.norm(v, ord=2)
        # sqrt(9 + 16) = 5
        self.assertAlmostEqual(n.numpy().item(), 5.0)

    def test_norm_vector_1(self):
        """Test 1-norm for vectors."""
        v = mlx_compat.tensor([1., -2., 3.])
        n = mlx_compat.linalg.norm(v, ord=1)
        # |1| + |-2| + |3| = 6
        self.assertAlmostEqual(n.numpy().item(), 6.0)

    def test_norm_vector_inf(self):
        """Test infinity norm for vectors."""
        v = mlx_compat.tensor([1., -5., 3.])
        n = mlx_compat.linalg.norm(v, ord=float('inf'))
        # max(|1|, |-5|, |3|) = 5
        self.assertAlmostEqual(n.numpy().item(), 5.0)

    def test_norm_keepdim(self):
        """Test norm with keepdim."""
        m = mlx_compat.tensor([[1., 2.], [3., 4.]])
        n = mlx_compat.linalg.norm(m, dim=1, keepdim=True)
        self.assertEqual(n.shape, (2, 1))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_norm_parity(self):
        """Test parity with PyTorch linalg.norm."""
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)

        mlx_x = mlx_compat.tensor(data)
        torch_x = torch.tensor(data)

        mlx_result = mlx_compat.linalg.norm(mlx_x)
        torch_result = torch.linalg.norm(torch_x)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestLinalgSVD(TestCase):
    """Test linalg.svd operations."""

    def test_svd_basic(self):
        """Test basic SVD."""
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        m = mlx_compat.tensor(data)
        U, S, Vh = mlx_compat.linalg.svd(m)

        # U should be (3, 3), S should be (3,), Vh should be (4, 4) for full
        # But MLX returns reduced SVD
        self.assertEqual(len(S.shape), 1)
        self.assertTrue(S.shape[0] <= min(3, 4))

    def test_svd_reconstruction(self):
        """Test that SVD can reconstruct the matrix."""
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        m = mlx_compat.tensor(data)
        U, S, Vh = mlx_compat.linalg.svd(m)

        # Reconstruct: U @ diag(S) @ Vh
        # Note: need to handle dimensions properly
        k = S.shape[0]
        U_k = U[:, :k] if len(U.shape) > 1 else U
        Vh_k = Vh[:k, :] if len(Vh.shape) > 1 else Vh

        reconstructed = mlx_compat.matmul(
            mlx_compat.matmul(U_k, mlx_compat.diag(S)),
            Vh_k
        )
        np.testing.assert_allclose(
            reconstructed.numpy(), data, rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestLinalgSvdvals(TestCase):
    """Test linalg.svdvals operations."""

    def test_svdvals_basic(self):
        """Test svdvals returns singular values only."""
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        m = mlx_compat.tensor(data)
        S = mlx_compat.linalg.svdvals(m)
        self.assertEqual(len(S.shape), 1)


@skipIfNoMLX
class TestLinalgQR(TestCase):
    """Test linalg.qr operations."""

    def test_qr_basic(self):
        """Test basic QR decomposition."""
        np.random.seed(42)
        data = np.random.randn(4, 3).astype(np.float32)
        m = mlx_compat.tensor(data)
        Q, R = mlx_compat.linalg.qr(m)

        # Q should be orthogonal, R should be upper triangular
        # Check Q @ R = M
        reconstructed = mlx_compat.matmul(Q, R)
        np.testing.assert_allclose(
            reconstructed.numpy(), data, rtol=1e-4, atol=1e-4
        )

    def test_qr_orthogonal(self):
        """Test that Q is orthogonal (Q^T @ Q = I)."""
        np.random.seed(42)
        data = np.random.randn(4, 3).astype(np.float32)
        m = mlx_compat.tensor(data)
        Q, R = mlx_compat.linalg.qr(m)

        QtQ = mlx_compat.matmul(Q.transpose(-1, -2), Q)
        k = min(4, 3)
        expected_I = np.eye(k, dtype=np.float32)
        np.testing.assert_allclose(
            QtQ.numpy()[:k, :k], expected_I, rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestLinalgCholesky(TestCase):
    """Test linalg.cholesky operations."""

    def test_cholesky_basic(self):
        """Test basic Cholesky decomposition."""
        # Create a positive definite matrix
        m = mlx_compat.tensor([[4., 2.], [2., 5.]])
        L = mlx_compat.linalg.cholesky(m)

        # L should be lower triangular
        # Check L @ L^T = M
        reconstructed = mlx_compat.matmul(L, L.transpose(-1, -2))
        np.testing.assert_allclose(
            reconstructed.numpy(), m.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_cholesky_random(self):
        """Test Cholesky on random positive definite matrix."""
        np.random.seed(42)
        A = np.random.randn(4, 4).astype(np.float32)
        # Make positive definite: A^T @ A + small diagonal
        data = A.T @ A + 0.1 * np.eye(4, dtype=np.float32)

        m = mlx_compat.tensor(data)
        L = mlx_compat.linalg.cholesky(m)

        reconstructed = mlx_compat.matmul(L, L.transpose(-1, -2))
        np.testing.assert_allclose(
            reconstructed.numpy(), data, rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestLinalgInv(TestCase):
    """Test linalg.inv operations."""

    def test_inv_basic(self):
        """Test basic matrix inverse."""
        m = mlx_compat.tensor([[1., 2.], [3., 4.]])
        m_inv = mlx_compat.linalg.inv(m)

        # M @ M_inv should be identity
        product = mlx_compat.matmul(m, m_inv)
        expected = np.eye(2, dtype=np.float32)
        np.testing.assert_allclose(
            product.numpy(), expected, rtol=1e-4, atol=1e-4
        )

    def test_inv_random(self):
        """Test inverse of random matrix."""
        np.random.seed(42)
        data = np.random.randn(4, 4).astype(np.float32)
        # Make invertible by adding to diagonal
        data += np.eye(4, dtype=np.float32) * 2

        m = mlx_compat.tensor(data)
        m_inv = mlx_compat.linalg.inv(m)

        product = mlx_compat.matmul(m, m_inv)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_allclose(
            product.numpy(), expected, rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestLinalgSolve(TestCase):
    """Test linalg.solve operations."""

    def test_solve_basic(self):
        """Test basic linear system solve."""
        # Solve Ax = b
        A = mlx_compat.tensor([[3., 1.], [1., 2.]])
        b = mlx_compat.tensor([[9.], [8.]])
        x = mlx_compat.linalg.solve(A, b)

        # Check A @ x = b
        result = mlx_compat.matmul(A, x)
        np.testing.assert_allclose(
            result.numpy(), b.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_solve_multiple_rhs(self):
        """Test solve with multiple right-hand sides."""
        np.random.seed(42)
        A_data = np.random.randn(3, 3).astype(np.float32)
        A_data += np.eye(3) * 2  # Make well-conditioned
        b_data = np.random.randn(3, 2).astype(np.float32)

        A = mlx_compat.tensor(A_data)
        b = mlx_compat.tensor(b_data)
        x = mlx_compat.linalg.solve(A, b)

        result = mlx_compat.matmul(A, x)
        np.testing.assert_allclose(
            result.numpy(), b_data, rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestLinalgEig(TestCase):
    """Test linalg.eig operations."""

    def test_eig_basic(self):
        """Test basic eigenvalue decomposition."""
        # Simple matrix with known eigenvalues
        m = mlx_compat.tensor([[2., 0.], [0., 3.]])
        eigenvalues, eigenvectors = mlx_compat.linalg.eig(m)

        # Eigenvalues should be 2 and 3
        eig_vals = np.sort(np.abs(eigenvalues.numpy()))
        np.testing.assert_allclose(eig_vals, [2.0, 3.0], rtol=1e-4)


@skipIfNoMLX
class TestLinalgEigh(TestCase):
    """Test linalg.eigh operations."""

    def test_eigh_basic(self):
        """Test eigenvalue decomposition of symmetric matrix."""
        # Symmetric matrix
        m = mlx_compat.tensor([[4., 2.], [2., 3.]])
        eigenvalues, eigenvectors = mlx_compat.linalg.eigh(m)

        # Check V @ diag(eigenvalues) @ V^T = M
        # For symmetric matrices, eigenvectors are orthonormal
        reconstructed = mlx_compat.matmul(
            mlx_compat.matmul(eigenvectors, mlx_compat.diag(eigenvalues)),
            eigenvectors.transpose(-1, -2)
        )
        np.testing.assert_allclose(
            reconstructed.numpy(), m.numpy(), rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestLinalgDet(TestCase):
    """Test linalg.det operations."""

    def test_det_basic(self):
        """Test basic determinant."""
        m = mlx_compat.tensor([[1., 2.], [3., 4.]])
        d = mlx_compat.linalg.det(m)
        # det = 1*4 - 2*3 = -2
        self.assertAlmostEqual(d.numpy().item(), -2.0, places=4)

    def test_det_identity(self):
        """Test determinant of identity matrix."""
        m = mlx_compat.eye(4)
        d = mlx_compat.linalg.det(m)
        self.assertAlmostEqual(d.numpy().item(), 1.0, places=4)

    def test_det_singular(self):
        """Test determinant of singular matrix."""
        m = mlx_compat.tensor([[1., 2.], [2., 4.]])  # Rows are multiples
        d = mlx_compat.linalg.det(m)
        self.assertAlmostEqual(d.numpy().item(), 0.0, places=4)


@skipIfNoMLX
class TestLinalgSlogdet(TestCase):
    """Test linalg.slogdet operations."""

    def test_slogdet_basic(self):
        """Test sign and log determinant."""
        m = mlx_compat.tensor([[1., 2.], [3., 4.]])
        sign, logabsdet = mlx_compat.linalg.slogdet(m)
        # det = -2, so sign = -1, log|det| = log(2)
        self.assertAlmostEqual(sign.numpy().item(), -1.0, places=4)
        self.assertAlmostEqual(logabsdet.numpy().item(), np.log(2), places=4)

    def test_slogdet_positive(self):
        """Test slogdet with positive determinant."""
        # Create positive definite matrix
        m = mlx_compat.tensor([[4., 2.], [2., 5.]])
        sign, logabsdet = mlx_compat.linalg.slogdet(m)
        # det = 4*5 - 2*2 = 16, so sign = 1, log|det| = log(16)
        self.assertAlmostEqual(sign.numpy().item(), 1.0, places=4)
        self.assertAlmostEqual(logabsdet.numpy().item(), np.log(16), places=3)


@skipIfNoMLX
class TestLinalgPinv(TestCase):
    """Test linalg.pinv operations."""

    def test_pinv_square(self):
        """Test pseudoinverse of square matrix."""
        m = mlx_compat.tensor([[1., 2.], [3., 4.]])
        m_pinv = mlx_compat.linalg.pinv(m)

        # For invertible matrix, pinv = inv
        # M @ M_pinv should be close to identity
        product = mlx_compat.matmul(m, m_pinv)
        expected = np.eye(2, dtype=np.float32)
        np.testing.assert_allclose(
            product.numpy(), expected, rtol=1e-4, atol=1e-5
        )

    def test_pinv_tall(self):
        """Test pseudoinverse of tall matrix (more rows than cols)."""
        np.random.seed(42)
        data = np.random.randn(6, 3).astype(np.float32)
        m = mlx_compat.tensor(data)
        m_pinv = mlx_compat.linalg.pinv(m)

        # For tall matrix: pinv(M) @ M should be close to identity (3x3)
        product = mlx_compat.matmul(m_pinv, m)
        expected = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(
            product.numpy(), expected, rtol=1e-4, atol=1e-5
        )

    def test_pinv_wide(self):
        """Test pseudoinverse of wide matrix (more cols than rows)."""
        np.random.seed(42)
        data = np.random.randn(3, 6).astype(np.float32)
        m = mlx_compat.tensor(data)
        m_pinv = mlx_compat.linalg.pinv(m)

        # For wide matrix: M @ pinv(M) should be close to identity (3x3)
        product = mlx_compat.matmul(m, m_pinv)
        expected = np.eye(3, dtype=np.float32)
        np.testing.assert_allclose(
            product.numpy(), expected, rtol=1e-4, atol=1e-5
        )

    def test_pinv_reconstruction(self):
        """Test that M @ pinv(M) @ M = M."""
        np.random.seed(42)
        data = np.random.randn(4, 3).astype(np.float32)
        m = mlx_compat.tensor(data)
        m_pinv = mlx_compat.linalg.pinv(m)

        reconstructed = mlx_compat.matmul(mlx_compat.matmul(m, m_pinv), m)
        np.testing.assert_allclose(
            reconstructed.numpy(), data, rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestLinalgCross(TestCase):
    """Test linalg.cross operations."""

    def test_cross_basic(self):
        """Test basic cross product."""
        a = mlx_compat.tensor([1., 0., 0.])
        b = mlx_compat.tensor([0., 1., 0.])
        c = mlx_compat.linalg.cross(a, b)
        # i x j = k
        expected = np.array([0., 0., 1.])
        np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)

    def test_cross_another(self):
        """Test another cross product."""
        a = mlx_compat.tensor([1., 2., 3.])
        b = mlx_compat.tensor([4., 5., 6.])
        c = mlx_compat.linalg.cross(a, b)
        # [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4] = [-3, 6, -3]
        expected = np.array([-3., 6., -3.])
        np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
