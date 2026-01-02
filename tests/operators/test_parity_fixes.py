"""
Test numerical parity fixes for angle, mode, and lobpcg.

These tests verify that the fixes for the following issues work correctly:
1. torch.angle - Complex number support
2. torch.mode - Correct index behavior based on array length
3. torch.lobpcg - Eigenvector sign normalization
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlx.core as mx

    import flashlight

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


def skipIfNoTorch(func):
    """Skip test if PyTorch is not available."""
    return unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")(func)


@skipIfNoMLX
class TestAngleParity(TestCase):
    """Test torch.angle numerical parity with PyTorch."""

    @skipIfNoTorch
    def test_angle_real_positive(self):
        """Test angle with positive real values."""
        values = [1.0, 2.0, 3.0, 100.0]
        pt_tensor = torch.tensor(values)
        mlx_tensor = flashlight.tensor(values)

        pt_result = torch.angle(pt_tensor).numpy()
        mlx_result = np.array(flashlight.angle(mlx_tensor)._mlx_array)

        np.testing.assert_allclose(pt_result, mlx_result, rtol=1e-5, atol=1e-6)

    @skipIfNoTorch
    def test_angle_real_negative(self):
        """Test angle with negative real values (should return pi)."""
        values = [-1.0, -2.0, -3.0, -100.0]
        pt_tensor = torch.tensor(values)
        mlx_tensor = flashlight.tensor(values)

        pt_result = torch.angle(pt_tensor).numpy()
        mlx_result = np.array(flashlight.angle(mlx_tensor)._mlx_array)

        np.testing.assert_allclose(pt_result, mlx_result, rtol=1e-5, atol=1e-6)

    @skipIfNoTorch
    def test_angle_real_mixed(self):
        """Test angle with mixed real values."""
        values = [1.0, -1.0, 0.0, 2.5, -3.5]
        pt_tensor = torch.tensor(values)
        mlx_tensor = flashlight.tensor(values)

        pt_result = torch.angle(pt_tensor).numpy()
        mlx_result = np.array(flashlight.angle(mlx_tensor)._mlx_array)

        np.testing.assert_allclose(pt_result, mlx_result, rtol=1e-5, atol=1e-6)

    @skipIfNoTorch
    def test_angle_real_inf(self):
        """Test angle with infinity values."""
        values = [float("inf"), float("-inf")]
        pt_tensor = torch.tensor(values)
        mlx_tensor = flashlight.tensor(values)

        pt_result = torch.angle(pt_tensor).numpy()
        mlx_result = np.array(flashlight.angle(mlx_tensor)._mlx_array)

        np.testing.assert_allclose(pt_result, mlx_result, rtol=1e-5, atol=1e-6)

    @skipIfNoTorch
    def test_angle_real_nan(self):
        """Test angle with NaN values (should return NaN)."""
        values = [float("nan")]
        pt_tensor = torch.tensor(values)
        mlx_tensor = flashlight.tensor(values)

        pt_result = torch.angle(pt_tensor).numpy()
        mlx_result = np.array(flashlight.angle(mlx_tensor)._mlx_array)

        # Both should be NaN
        self.assertTrue(np.isnan(pt_result[0]))
        self.assertTrue(np.isnan(mlx_result[0]))

    @skipIfNoTorch
    def test_angle_complex(self):
        """Test angle with complex values."""
        real = [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.5, -2.0]
        imag = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.5, 3.0]

        # PyTorch complex tensor
        pt_complex = torch.complex(torch.tensor(real), torch.tensor(imag))
        pt_result = torch.angle(pt_complex).numpy()

        # MLX complex tensor
        complex_np = np.array(real) + 1j * np.array(imag)
        complex_mlx = mx.array(complex_np)
        mlx_tensor = flashlight.Tensor._from_mlx_array(complex_mlx)
        mlx_result = np.array(flashlight.angle(mlx_tensor)._mlx_array)

        np.testing.assert_allclose(pt_result, mlx_result, rtol=1e-5, atol=1e-6)


@skipIfNoMLX
class TestModeParity(TestCase):
    """Test torch.mode numerical parity with PyTorch."""

    @skipIfNoTorch
    def test_mode_small_array_last_index(self):
        """Test that mode returns LAST index for small arrays (length <= 23)."""
        # Array with mode 0 at first and last position
        for n in [5, 10, 15, 22]:  # All lengths <= 23
            data = [0] + list(range(1, n)) + [0]
            pt_tensor = torch.tensor([data])
            mlx_tensor = flashlight.tensor([data])

            pt_vals, pt_idx = torch.mode(pt_tensor, dim=1)
            mlx_vals, mlx_idx = flashlight.mode(mlx_tensor, dim=1)

            # Should return last index
            expected_idx = len(data) - 1
            self.assertEqual(
                pt_idx.item(), expected_idx, f"PyTorch should return last index for len={len(data)}"
            )
            self.assertEqual(
                int(mlx_idx._mlx_array.item()),
                expected_idx,
                f"MLX should return last index for len={len(data)}",
            )

    @skipIfNoTorch
    def test_mode_large_array_first_index(self):
        """Test that mode returns FIRST index for large arrays (length >= 24)."""
        # Array with mode 0 at first and last position
        for n in [24, 30, 50]:  # All lengths >= 24
            data = [0] + list(range(1, n)) + [0]
            pt_tensor = torch.tensor([data])
            mlx_tensor = flashlight.tensor([data])

            pt_vals, pt_idx = torch.mode(pt_tensor, dim=1)
            mlx_vals, mlx_idx = flashlight.mode(mlx_tensor, dim=1)

            # Should return first index
            expected_idx = 0
            self.assertEqual(
                pt_idx.item(),
                expected_idx,
                f"PyTorch should return first index for len={len(data)}",
            )
            self.assertEqual(
                int(mlx_idx._mlx_array.item()),
                expected_idx,
                f"MLX should return first index for len={len(data)}",
            )

    @skipIfNoTorch
    def test_mode_threshold_boundary(self):
        """Test the exact boundary between small and large array behavior."""
        # Length 23: should return LAST index
        data_23 = [0] + list(range(1, 22)) + [0]  # length 23
        pt_tensor = torch.tensor([data_23])
        mlx_tensor = flashlight.tensor([data_23])

        _, pt_idx = torch.mode(pt_tensor, dim=1)
        _, mlx_idx = flashlight.mode(mlx_tensor, dim=1)

        self.assertEqual(pt_idx.item(), 22)  # Last index
        self.assertEqual(int(mlx_idx._mlx_array.item()), 22)

        # Length 24: should return FIRST index
        data_24 = [0] + list(range(1, 23)) + [0]  # length 24
        pt_tensor = torch.tensor([data_24])
        mlx_tensor = flashlight.tensor([data_24])

        _, pt_idx = torch.mode(pt_tensor, dim=1)
        _, mlx_idx = flashlight.mode(mlx_tensor, dim=1)

        self.assertEqual(pt_idx.item(), 0)  # First index
        self.assertEqual(int(mlx_idx._mlx_array.item()), 0)

    @skipIfNoTorch
    def test_mode_values_match(self):
        """Test that mode values always match."""
        test_cases = [
            [1, 2, 2, 1],
            [0, 1, 2, 3, 0],
            [3, 2, 1, 3, 2, 1],
            [5, 1, 5, 1, 5],
            [1, 1, 2, 2, 3],
        ]

        for data in test_cases:
            pt_tensor = torch.tensor([data])
            mlx_tensor = flashlight.tensor([data])

            pt_vals, _ = torch.mode(pt_tensor, dim=1)
            mlx_vals, _ = flashlight.mode(mlx_tensor, dim=1)

            self.assertEqual(
                pt_vals.item(),
                int(mlx_vals._mlx_array.item()),
                f"Mode values should match for {data}",
            )

    @skipIfNoTorch
    def test_mode_multidimensional(self):
        """Test mode with multi-dimensional arrays."""
        data = np.array(
            [[[1, 2, 2, 3], [4, 4, 5, 5], [6, 6, 6, 7]], [[8, 8, 8, 9], [1, 1, 2, 2], [3, 4, 4, 4]]]
        )
        pt_tensor = torch.tensor(data)
        mlx_tensor = flashlight.tensor(data.tolist())

        for dim in [0, 1, 2]:
            pt_vals, pt_idx = torch.mode(pt_tensor, dim=dim)
            mlx_vals, mlx_idx = flashlight.mode(mlx_tensor, dim=dim)

            pt_vals_np = pt_vals.numpy()
            pt_idx_np = pt_idx.numpy()
            mlx_vals_np = np.array(mlx_vals._mlx_array)
            mlx_idx_np = np.array(mlx_idx._mlx_array)

            np.testing.assert_array_equal(
                pt_vals_np, mlx_vals_np, f"Values should match for dim={dim}"
            )
            np.testing.assert_array_equal(
                pt_idx_np, mlx_idx_np, f"Indices should match for dim={dim}"
            )


@skipIfNoMLX
class TestLobpcgParity(TestCase):
    """Test torch.lobpcg numerical parity with PyTorch."""

    @skipIfNoTorch
    def test_lobpcg_eigenvalues(self):
        """Test that lobpcg eigenvalues match PyTorch."""
        np.random.seed(42)
        n = 10
        k = 2

        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = A_np @ A_np.T + np.eye(n, dtype=np.float32) * 0.1
        X_np = np.random.randn(n, k).astype(np.float32)

        A_pt = torch.tensor(A_np)
        X_pt = torch.tensor(X_np)
        pt_vals, _ = torch.lobpcg(A_pt, k=k, X=X_pt)

        A_mlx = flashlight.tensor(A_np.tolist())
        X_mlx = flashlight.tensor(X_np.tolist())
        mlx_vals, _ = flashlight.lobpcg(A_mlx, k=k, X=X_mlx)

        pt_vals_np = pt_vals.numpy()
        mlx_vals_np = np.array(mlx_vals._mlx_array)

        np.testing.assert_allclose(pt_vals_np, mlx_vals_np, rtol=1e-4, atol=1e-5)

    @skipIfNoTorch
    def test_lobpcg_eigenvectors_up_to_sign(self):
        """Test that lobpcg eigenvectors match PyTorch up to sign."""
        np.random.seed(42)
        n = 10
        k = 2

        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = A_np @ A_np.T + np.eye(n, dtype=np.float32) * 0.1
        X_np = np.random.randn(n, k).astype(np.float32)

        A_pt = torch.tensor(A_np)
        X_pt = torch.tensor(X_np)
        _, pt_vecs = torch.lobpcg(A_pt, k=k, X=X_pt)

        A_mlx = flashlight.tensor(A_np.tolist())
        X_mlx = flashlight.tensor(X_np.tolist())
        _, mlx_vecs = flashlight.lobpcg(A_mlx, k=k, X=X_mlx)

        pt_vecs_np = pt_vecs.numpy()
        mlx_vecs_np = np.array(mlx_vecs._mlx_array)

        # Compare each eigenvector up to sign
        for i in range(k):
            direct_diff = np.abs(pt_vecs_np[:, i] - mlx_vecs_np[:, i]).max()
            flipped_diff = np.abs(pt_vecs_np[:, i] + mlx_vecs_np[:, i]).max()
            best_diff = min(direct_diff, flipped_diff)

            self.assertLess(
                best_diff, 1e-3, f"Eigenvector {i} should match up to sign (diff={best_diff})"
            )

    @skipIfNoTorch
    def test_lobpcg_eigenvectors_are_unit(self):
        """Test that lobpcg returns unit eigenvectors."""
        np.random.seed(42)
        n = 10
        k = 2

        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = A_np @ A_np.T + np.eye(n, dtype=np.float32) * 0.1
        X_np = np.random.randn(n, k).astype(np.float32)

        A_mlx = flashlight.tensor(A_np.tolist())
        X_mlx = flashlight.tensor(X_np.tolist())
        _, mlx_vecs = flashlight.lobpcg(A_mlx, k=k, X=X_mlx)

        mlx_vecs_np = np.array(mlx_vecs._mlx_array)

        for i in range(k):
            norm = np.linalg.norm(mlx_vecs_np[:, i])
            self.assertAlmostEqual(
                norm, 1.0, places=5, msg=f"Eigenvector {i} should have unit norm"
            )

    @skipIfNoTorch
    def test_lobpcg_eigenvectors_satisfy_eigenequation(self):
        """Test that A @ v = lambda * v for each eigenpair."""
        np.random.seed(42)
        n = 10
        k = 2

        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = A_np @ A_np.T + np.eye(n, dtype=np.float32) * 0.1
        X_np = np.random.randn(n, k).astype(np.float32)

        A_mlx = flashlight.tensor(A_np.tolist())
        X_mlx = flashlight.tensor(X_np.tolist())
        mlx_vals, mlx_vecs = flashlight.lobpcg(A_mlx, k=k, X=X_mlx)

        mlx_vals_np = np.array(mlx_vals._mlx_array)
        mlx_vecs_np = np.array(mlx_vecs._mlx_array)

        for i in range(k):
            v = mlx_vecs_np[:, i]
            lam = mlx_vals_np[i]

            # A @ v should equal lambda * v
            Av = A_np @ v
            lam_v = lam * v

            residual = np.abs(Av - lam_v).max()
            self.assertLess(
                residual, 1e-4, f"Eigenpair {i} should satisfy eigenequation (residual={residual})"
            )

    @skipIfNoTorch
    def test_lobpcg_largest_eigenvalues(self):
        """Test that lobpcg returns the largest eigenvalues by default."""
        np.random.seed(42)
        n = 10
        k = 2

        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = A_np @ A_np.T + np.eye(n, dtype=np.float32) * 0.1
        X_np = np.random.randn(n, k).astype(np.float32)

        A_mlx = flashlight.tensor(A_np.tolist())
        X_mlx = flashlight.tensor(X_np.tolist())
        mlx_vals, _ = flashlight.lobpcg(A_mlx, k=k, X=X_mlx)

        mlx_vals_np = np.array(mlx_vals._mlx_array)

        # Compute all eigenvalues with numpy
        all_eigenvalues = np.linalg.eigvalsh(A_np)
        largest_k = np.sort(all_eigenvalues)[-k:][::-1]

        np.testing.assert_allclose(mlx_vals_np, largest_k, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
