"""
Linear Algebra Parity Tests

Tests numerical parity between flashlight.linalg and PyTorch torch.linalg.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import flashlight


class TestMatrixNormParity:
    """Test matrix norm parity with PyTorch."""

    @pytest.mark.parity
    def test_norm_frobenius_parity(self):
        """Test Frobenius norm matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(10, 10).astype(np.float32)

        torch_out = torch.linalg.norm(torch.tensor(x_np), ord="fro")
        mlx_out = flashlight.linalg.norm(flashlight.tensor(x_np), ord="fro")

        max_diff = abs(torch_out.item() - float(mlx_out._mlx_array))
        assert max_diff < 1e-4, f"Frobenius norm mismatch: {max_diff}"

    @pytest.mark.parity
    def test_norm_2_parity(self):
        """Test 2-norm matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(10).astype(np.float32)

        torch_out = torch.linalg.norm(torch.tensor(x_np), ord=2)
        mlx_out = flashlight.linalg.norm(flashlight.tensor(x_np), ord=2)

        max_diff = abs(torch_out.item() - float(mlx_out._mlx_array))
        assert max_diff < 1e-5, f"2-norm mismatch: {max_diff}"

    @pytest.mark.parity
    def test_norm_inf_parity(self):
        """Test infinity norm matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(10).astype(np.float32)

        torch_out = torch.linalg.norm(torch.tensor(x_np), ord=float("inf"))
        mlx_out = flashlight.linalg.norm(flashlight.tensor(x_np), ord=float("inf"))

        max_diff = abs(torch_out.item() - float(mlx_out._mlx_array))
        assert max_diff < 1e-6, f"inf-norm mismatch: {max_diff}"


class TestDeterminantParity:
    """Test determinant parity with PyTorch."""

    @pytest.mark.parity
    def test_det_parity(self):
        """Test determinant matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.linalg.det(torch.tensor(x_np))
        mlx_out = flashlight.linalg.det(flashlight.tensor(x_np))

        # Determinant can have larger numerical errors
        max_diff = abs(torch_out.item() - float(mlx_out._mlx_array))
        assert max_diff < 1e-2, f"det mismatch: {max_diff}"

    @pytest.mark.parity
    def test_slogdet_parity(self):
        """Test sign and log determinant matches PyTorch."""
        np.random.seed(42)
        # Use positive definite matrix for stable log determinant
        A = np.random.randn(5, 5).astype(np.float32)
        x_np = A @ A.T + 0.1 * np.eye(5, dtype=np.float32)

        torch_sign, torch_logdet = torch.linalg.slogdet(torch.tensor(x_np))
        mlx_sign, mlx_logdet = flashlight.linalg.slogdet(flashlight.tensor(x_np))

        assert abs(torch_sign.item() - float(mlx_sign._mlx_array)) < 1e-5
        assert abs(torch_logdet.item() - float(mlx_logdet._mlx_array)) < 1e-3


class TestInverseParity:
    """Test matrix inverse parity with PyTorch."""

    @pytest.mark.parity
    def test_inv_parity(self):
        """Test matrix inverse matches PyTorch."""
        np.random.seed(42)
        # Create well-conditioned matrix
        A = np.random.randn(5, 5).astype(np.float32)
        x_np = A @ A.T + np.eye(5, dtype=np.float32)

        torch_out = torch.linalg.inv(torch.tensor(x_np))
        mlx_out = flashlight.linalg.inv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"inv mismatch: {max_diff}"


class TestPseudoinverseParity:
    """Test pseudoinverse parity with PyTorch."""

    @pytest.mark.parity
    def test_pinv_square_parity(self):
        """Test pseudoinverse of square matrix matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.linalg.pinv(torch.tensor(x_np))
        mlx_out = flashlight.linalg.pinv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-3, f"pinv square mismatch: {max_diff}"

    @pytest.mark.parity
    def test_pinv_tall_parity(self):
        """Test pseudoinverse of tall matrix matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(10, 5).astype(np.float32)

        torch_out = torch.linalg.pinv(torch.tensor(x_np))
        mlx_out = flashlight.linalg.pinv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-3, f"pinv tall mismatch: {max_diff}"

    @pytest.mark.parity
    def test_pinv_wide_parity(self):
        """Test pseudoinverse of wide matrix matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 10).astype(np.float32)

        torch_out = torch.linalg.pinv(torch.tensor(x_np))
        mlx_out = flashlight.linalg.pinv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-3, f"pinv wide mismatch: {max_diff}"

    @pytest.mark.parity
    def test_pinv_reconstruction_parity(self):
        """Test pinv satisfies A @ pinv(A) @ A ≈ A."""
        np.random.seed(42)
        x_np = np.random.randn(6, 4).astype(np.float32)

        mlx_x = flashlight.tensor(x_np)
        mlx_pinv = flashlight.linalg.pinv(mlx_x)

        # A @ pinv(A) @ A should ≈ A
        reconstruction = mlx_x @ mlx_pinv @ mlx_x

        max_diff = np.max(np.abs(np.array(reconstruction._mlx_array) - x_np))
        assert max_diff < 1e-4, f"pinv reconstruction mismatch: {max_diff}"


class TestSVDParity:
    """Test SVD parity with PyTorch."""

    @pytest.mark.parity
    def test_svd_parity(self):
        """Test SVD matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(6, 4).astype(np.float32)

        # Use full_matrices=False for reduced SVD (compatible shapes)
        torch_U, torch_S, torch_Vh = torch.linalg.svd(torch.tensor(x_np), full_matrices=False)
        mlx_U, mlx_S, mlx_Vh = flashlight.linalg.svd(flashlight.tensor(x_np))

        # Singular values should match
        max_diff_S = np.max(np.abs(torch_S.numpy() - np.array(mlx_S._mlx_array)))
        assert max_diff_S < 1e-4, f"SVD singular values mismatch: {max_diff_S}"

        # Reconstruction should match: U @ diag(S) @ Vh ≈ X
        torch_recon = torch_U @ torch.diag(torch_S) @ torch_Vh
        # For MLX, use reduced matrices for reconstruction
        k = min(6, 4)
        mlx_U_k = mlx_U[:, :k] if mlx_U.shape[1] > k else mlx_U
        mlx_Vh_k = mlx_Vh[:k, :] if mlx_Vh.shape[0] > k else mlx_Vh
        mlx_recon = mlx_U_k @ flashlight.diag(mlx_S) @ mlx_Vh_k

        max_diff_recon = np.max(np.abs(torch_recon.numpy() - np.array(mlx_recon._mlx_array)))
        assert max_diff_recon < 1e-4, f"SVD reconstruction mismatch: {max_diff_recon}"


class TestQRParity:
    """Test QR decomposition parity with PyTorch."""

    @pytest.mark.parity
    def test_qr_parity(self):
        """Test QR decomposition matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(6, 4).astype(np.float32)

        torch_Q, torch_R = torch.linalg.qr(torch.tensor(x_np))
        mlx_Q, mlx_R = flashlight.linalg.qr(flashlight.tensor(x_np))

        # Reconstruction should match: Q @ R ≈ X
        torch_recon = torch_Q @ torch_R
        mlx_recon = mlx_Q @ mlx_R

        max_diff = np.max(np.abs(torch_recon.numpy() - np.array(mlx_recon._mlx_array)))
        assert max_diff < 1e-4, f"QR reconstruction mismatch: {max_diff}"

        # Q should be orthogonal: Q^T @ Q ≈ I
        mlx_QtQ = mlx_Q.t() @ mlx_Q
        I = np.eye(4, dtype=np.float32)
        max_diff_orth = np.max(np.abs(np.array(mlx_QtQ._mlx_array) - I))
        assert max_diff_orth < 1e-4, f"Q orthogonality mismatch: {max_diff_orth}"


class TestCholeskyParity:
    """Test Cholesky decomposition parity with PyTorch."""

    @pytest.mark.parity
    def test_cholesky_parity(self):
        """Test Cholesky decomposition matches PyTorch."""
        np.random.seed(42)
        # Create positive definite matrix
        A = np.random.randn(5, 5).astype(np.float32)
        x_np = A @ A.T + np.eye(5, dtype=np.float32)

        torch_L = torch.linalg.cholesky(torch.tensor(x_np))
        mlx_L = flashlight.linalg.cholesky(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_L.numpy() - np.array(mlx_L._mlx_array)))
        assert max_diff < 1e-4, f"Cholesky mismatch: {max_diff}"

        # L @ L^T should reconstruct original
        mlx_recon = mlx_L @ mlx_L.t()
        max_diff_recon = np.max(np.abs(np.array(mlx_recon._mlx_array) - x_np))
        assert max_diff_recon < 1e-4, f"Cholesky reconstruction mismatch: {max_diff_recon}"


class TestCrossParity:
    """Test cross product parity with PyTorch."""

    @pytest.mark.parity
    def test_cross_3d_parity(self):
        """Test 3D cross product matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(3).astype(np.float32)
        b_np = np.random.randn(3).astype(np.float32)

        torch_out = torch.linalg.cross(torch.tensor(a_np), torch.tensor(b_np))
        mlx_out = flashlight.linalg.cross(flashlight.tensor(a_np), flashlight.tensor(b_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"cross 3D mismatch: {max_diff}"

    @pytest.mark.parity
    def test_cross_batched_parity(self):
        """Test batched cross product matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(10, 3).astype(np.float32)
        b_np = np.random.randn(10, 3).astype(np.float32)

        torch_out = torch.linalg.cross(torch.tensor(a_np), torch.tensor(b_np))
        mlx_out = flashlight.linalg.cross(flashlight.tensor(a_np), flashlight.tensor(b_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"cross batched mismatch: {max_diff}"


class TestEigenParity:
    """Test eigenvalue decomposition parity with PyTorch."""

    @pytest.mark.parity
    def test_eigh_symmetric_parity(self):
        """Test symmetric eigenvalue decomposition matches PyTorch."""
        np.random.seed(42)
        A = np.random.randn(5, 5).astype(np.float32)
        x_np = (A + A.T) / 2  # Make symmetric

        torch_vals, torch_vecs = torch.linalg.eigh(torch.tensor(x_np))
        mlx_vals, mlx_vecs = flashlight.linalg.eigh(flashlight.tensor(x_np))

        # Eigenvalues should match (sorted)
        max_diff_vals = np.max(
            np.abs(np.sort(torch_vals.numpy()) - np.sort(np.array(mlx_vals._mlx_array)))
        )
        assert max_diff_vals < 1e-4, f"eigh eigenvalues mismatch: {max_diff_vals}"

        # A @ V should equal V @ diag(eigenvalues) for each eigenvector
        # Due to sign ambiguity, check A @ v = lambda * v
        mlx_x = flashlight.tensor(x_np)
        Av = mlx_x @ mlx_vecs
        lambda_v = mlx_vecs * mlx_vals.unsqueeze(0)
        max_diff_recon = np.max(np.abs(np.array(Av._mlx_array) - np.array(lambda_v._mlx_array)))
        assert max_diff_recon < 1e-3, f"eigh reconstruction mismatch: {max_diff_recon}"


class TestMatmulParity:
    """Test matrix multiplication parity with PyTorch."""

    @pytest.mark.parity
    def test_matmul_2d_parity(self):
        """Test 2D matrix multiplication matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(10, 20).astype(np.float32)
        b_np = np.random.randn(20, 15).astype(np.float32)

        torch_out = torch.matmul(torch.tensor(a_np), torch.tensor(b_np))
        mlx_out = flashlight.matmul(flashlight.tensor(a_np), flashlight.tensor(b_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"matmul 2D mismatch: {max_diff}"

    @pytest.mark.parity
    def test_matmul_batched_parity(self):
        """Test batched matrix multiplication matches PyTorch."""
        np.random.seed(42)
        a_np = np.random.randn(4, 10, 20).astype(np.float32)
        b_np = np.random.randn(4, 20, 15).astype(np.float32)

        torch_out = torch.matmul(torch.tensor(a_np), torch.tensor(b_np))
        mlx_out = flashlight.matmul(flashlight.tensor(a_np), flashlight.tensor(b_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"matmul batched mismatch: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
