"""
Tensor Factory Functions Parity Tests

Tests numerical parity between mlx_compat factory functions and PyTorch.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import mlx_compat


class TestZerosOnesParity:
    """Test zeros and ones factory functions parity."""

    @pytest.mark.parity
    def test_zeros_parity(self):
        """Test zeros matches PyTorch."""
        torch_out = torch.zeros(5, 10)
        mlx_out = mlx_compat.zeros(5, 10)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0, f"zeros mismatch: {max_diff}"

    @pytest.mark.parity
    def test_ones_parity(self):
        """Test ones matches PyTorch."""
        torch_out = torch.ones(5, 10)
        mlx_out = mlx_compat.ones(5, 10)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0, f"ones mismatch: {max_diff}"

    @pytest.mark.parity
    def test_zeros_like_parity(self):
        """Test zeros_like matches PyTorch."""
        x = np.random.randn(3, 4, 5).astype(np.float32)

        torch_out = torch.zeros_like(torch.tensor(x))
        mlx_out = mlx_compat.zeros_like(mlx_compat.tensor(x))

        assert torch_out.shape == mlx_out.shape
        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0

    @pytest.mark.parity
    def test_ones_like_parity(self):
        """Test ones_like matches PyTorch."""
        x = np.random.randn(3, 4, 5).astype(np.float32)

        torch_out = torch.ones_like(torch.tensor(x))
        mlx_out = mlx_compat.ones_like(mlx_compat.tensor(x))

        assert torch_out.shape == mlx_out.shape
        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0


class TestFullParity:
    """Test full factory function parity."""

    @pytest.mark.parity
    def test_full_parity(self):
        """Test full matches PyTorch."""
        torch_out = torch.full((5, 10), 3.14)
        mlx_out = mlx_compat.full((5, 10), 3.14)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6, f"full mismatch: {max_diff}"

    @pytest.mark.parity
    def test_full_like_parity(self):
        """Test full_like matches PyTorch."""
        x = np.random.randn(3, 4).astype(np.float32)

        torch_out = torch.full_like(torch.tensor(x), 2.5)
        mlx_out = mlx_compat.full_like(mlx_compat.tensor(x), 2.5)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6


class TestArangeParity:
    """Test arange factory function parity."""

    @pytest.mark.parity
    def test_arange_end_parity(self):
        """Test arange with end only matches PyTorch."""
        torch_out = torch.arange(10)
        mlx_out = mlx_compat.arange(10)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0, f"arange end mismatch: {max_diff}"

    @pytest.mark.parity
    def test_arange_start_end_parity(self):
        """Test arange with start and end matches PyTorch."""
        torch_out = torch.arange(5, 15)
        mlx_out = mlx_compat.arange(5, 15)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0

    @pytest.mark.parity
    def test_arange_step_parity(self):
        """Test arange with step matches PyTorch."""
        torch_out = torch.arange(0, 10, 2)
        mlx_out = mlx_compat.arange(0, 10, 2)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0

    @pytest.mark.parity
    def test_arange_float_parity(self):
        """Test arange with float step matches PyTorch."""
        torch_out = torch.arange(0.0, 1.0, 0.1)
        mlx_out = mlx_compat.arange(0.0, 1.0, 0.1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6


class TestLinspaceParity:
    """Test linspace factory function parity."""

    @pytest.mark.parity
    def test_linspace_parity(self):
        """Test linspace matches PyTorch."""
        torch_out = torch.linspace(0, 10, 50)
        mlx_out = mlx_compat.linspace(0, 10, 50)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"linspace mismatch: {max_diff}"

    @pytest.mark.parity
    def test_linspace_negative_parity(self):
        """Test linspace with negative range matches PyTorch."""
        torch_out = torch.linspace(-5, 5, 21)
        mlx_out = mlx_compat.linspace(-5, 5, 21)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5


class TestLogspaceParity:
    """Test logspace factory function parity."""

    @pytest.mark.parity
    def test_logspace_parity(self):
        """Test logspace matches PyTorch."""
        torch_out = torch.logspace(0, 2, 10)
        mlx_out = mlx_compat.logspace(0, 2, 10)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"logspace mismatch: {max_diff}"

    @pytest.mark.parity
    def test_logspace_base_parity(self):
        """Test logspace with custom base matches PyTorch."""
        torch_out = torch.logspace(0, 2, 10, base=2)
        mlx_out = mlx_compat.logspace(0, 2, 10, base=2)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4


class TestEyeParity:
    """Test eye factory function parity."""

    @pytest.mark.parity
    def test_eye_square_parity(self):
        """Test square eye matches PyTorch."""
        torch_out = torch.eye(5)
        mlx_out = mlx_compat.eye(5)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0

    @pytest.mark.parity
    def test_eye_rectangular_parity(self):
        """Test rectangular eye matches PyTorch."""
        torch_out = torch.eye(3, 5)
        mlx_out = mlx_compat.eye(3, 5)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0


class TestEmptyParity:
    """Test empty factory function parity (shape only)."""

    @pytest.mark.parity
    def test_empty_shape_parity(self):
        """Test empty has correct shape like PyTorch."""
        torch_out = torch.empty(5, 10)
        mlx_out = mlx_compat.empty(5, 10)

        assert torch_out.shape == mlx_out.shape

    @pytest.mark.parity
    def test_empty_like_shape_parity(self):
        """Test empty_like has correct shape like PyTorch."""
        x = np.random.randn(3, 4, 5).astype(np.float32)

        torch_out = torch.empty_like(torch.tensor(x))
        mlx_out = mlx_compat.empty_like(mlx_compat.tensor(x))

        assert torch_out.shape == mlx_out.shape


class TestRandomParity:
    """Test random factory functions parity (statistical)."""

    @pytest.mark.parity
    def test_randn_statistics_parity(self):
        """Test randn has similar statistics to PyTorch."""
        torch.manual_seed(42)
        torch_out = torch.randn(10000)

        mlx_compat.random.manual_seed(42)
        mlx_out = mlx_compat.randn(10000)

        # Both should have mean ≈ 0, std ≈ 1
        torch_mean = torch_out.mean().item()
        torch_std = torch_out.std().item()

        mlx_data = np.array(mlx_out._mlx_array)
        mlx_mean = mlx_data.mean()
        mlx_std = mlx_data.std()

        assert abs(torch_mean) < 0.05 and abs(mlx_mean) < 0.05
        assert abs(torch_std - 1) < 0.05 and abs(mlx_std - 1) < 0.05

    @pytest.mark.parity
    def test_rand_statistics_parity(self):
        """Test rand has similar statistics to PyTorch."""
        torch.manual_seed(42)
        torch_out = torch.rand(10000)

        mlx_compat.random.manual_seed(42)
        mlx_out = mlx_compat.rand(10000)

        # Both should have mean ≈ 0.5
        torch_mean = torch_out.mean().item()
        mlx_mean = np.array(mlx_out._mlx_array).mean()

        assert abs(torch_mean - 0.5) < 0.02 and abs(mlx_mean - 0.5) < 0.02


class TestTriangularParity:
    """Test triangular matrix factory functions parity."""

    @pytest.mark.parity
    def test_tril_parity(self):
        """Test tril matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.tril(torch.tensor(x_np))
        mlx_out = mlx_compat.tril(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6

    @pytest.mark.parity
    def test_triu_parity(self):
        """Test triu matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.triu(torch.tensor(x_np))
        mlx_out = mlx_compat.triu(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6

    @pytest.mark.parity
    def test_tril_diagonal_parity(self):
        """Test tril with diagonal offset matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.tril(torch.tensor(x_np), diagonal=1)
        mlx_out = mlx_compat.tril(mlx_compat.tensor(x_np), diagonal=1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6


class TestDiagonalParity:
    """Test diagonal factory functions parity."""

    @pytest.mark.parity
    def test_diag_vector_parity(self):
        """Test diag from vector matches PyTorch."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        torch_out = torch.diag(torch.tensor(x_np))
        mlx_out = mlx_compat.diag(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff == 0

    @pytest.mark.parity
    def test_diag_extract_parity(self):
        """Test diag extraction from matrix matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.diag(torch.tensor(x_np))
        mlx_out = mlx_compat.diag(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6

    @pytest.mark.parity
    def test_diagonal_parity(self):
        """Test diagonal function matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.diagonal(torch.tensor(x_np))
        mlx_out = mlx_compat.diagonal(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6


class TestMeshgridParity:
    """Test meshgrid factory function parity."""

    @pytest.mark.parity
    def test_meshgrid_parity(self):
        """Test meshgrid matches PyTorch."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array([4.0, 5.0], dtype=np.float32)

        torch_X, torch_Y = torch.meshgrid(
            torch.tensor(x_np), torch.tensor(y_np), indexing="xy"
        )
        mlx_X, mlx_Y = mlx_compat.meshgrid(
            mlx_compat.tensor(x_np), mlx_compat.tensor(y_np), indexing="xy"
        )

        max_diff_X = np.max(np.abs(torch_X.numpy() - np.array(mlx_X._mlx_array)))
        max_diff_Y = np.max(np.abs(torch_Y.numpy() - np.array(mlx_Y._mlx_array)))

        assert max_diff_X < 1e-6
        assert max_diff_Y < 1e-6


class TestStackCatParity:
    """Test stack and cat factory functions parity."""

    @pytest.mark.parity
    def test_stack_parity(self):
        """Test stack matches PyTorch."""
        np.random.seed(42)
        x1 = np.random.randn(3, 4).astype(np.float32)
        x2 = np.random.randn(3, 4).astype(np.float32)

        torch_out = torch.stack([torch.tensor(x1), torch.tensor(x2)])
        mlx_out = mlx_compat.stack([mlx_compat.tensor(x1), mlx_compat.tensor(x2)])

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6

    @pytest.mark.parity
    def test_cat_parity(self):
        """Test cat matches PyTorch."""
        np.random.seed(42)
        x1 = np.random.randn(3, 4).astype(np.float32)
        x2 = np.random.randn(2, 4).astype(np.float32)

        torch_out = torch.cat([torch.tensor(x1), torch.tensor(x2)], dim=0)
        mlx_out = mlx_compat.cat([mlx_compat.tensor(x1), mlx_compat.tensor(x2)], dim=0)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6


class TestWhereParity:
    """Test where factory function parity."""

    @pytest.mark.parity
    def test_where_parity(self):
        """Test where matches PyTorch."""
        np.random.seed(42)
        cond = np.random.rand(5, 5) > 0.5
        x = np.random.randn(5, 5).astype(np.float32)
        y = np.random.randn(5, 5).astype(np.float32)

        torch_out = torch.where(
            torch.tensor(cond), torch.tensor(x), torch.tensor(y)
        )
        mlx_out = mlx_compat.where(
            mlx_compat.tensor(cond), mlx_compat.tensor(x), mlx_compat.tensor(y)
        )

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
