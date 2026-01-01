"""
Special Functions Parity Tests

Tests numerical parity between flashlight.special and PyTorch torch.special.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import flashlight


class TestGammaFunctionsParity:
    """Test gamma function parity with PyTorch."""

    @pytest.mark.parity
    def test_gammaln_parity(self):
        """Test gammaln matches PyTorch."""
        np.random.seed(42)
        x_np = np.abs(np.random.randn(100).astype(np.float32)) + 0.1

        torch_out = torch.special.gammaln(torch.tensor(x_np))
        mlx_out = flashlight.special.gammaln(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"gammaln mismatch: {max_diff}"

    @pytest.mark.parity
    def test_digamma_parity(self):
        """Test digamma matches PyTorch."""
        np.random.seed(42)
        x_np = np.abs(np.random.randn(100).astype(np.float32)) + 1.0

        torch_out = torch.special.digamma(torch.tensor(x_np))
        mlx_out = flashlight.special.digamma(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"digamma mismatch: {max_diff}"

    @pytest.mark.parity
    def test_polygamma_parity(self):
        """Test polygamma (order 1) matches PyTorch."""
        np.random.seed(42)
        x_np = np.abs(np.random.randn(50).astype(np.float32)) + 1.0

        torch_out = torch.special.polygamma(1, torch.tensor(x_np))
        mlx_out = flashlight.special.polygamma(1, flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"polygamma mismatch: {max_diff}"

    @pytest.mark.parity
    def test_multigammaln_parity(self):
        """Test multigammaln matches PyTorch."""
        x_np = np.array([3.0, 5.0, 10.0, 20.0], dtype=np.float32)

        for p in [2, 3, 4]:
            torch_out = torch.special.multigammaln(torch.tensor(x_np), p)
            mlx_out = flashlight.special.multigammaln(flashlight.tensor(x_np), p)

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-4, f"multigammaln (p={p}) mismatch: {max_diff}"


class TestIncompleteGammaParity:
    """Test incomplete gamma function parity with PyTorch."""

    @pytest.mark.parity
    def test_gammainc_parity(self):
        """Test gammainc matches PyTorch."""
        a_np = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        x_np = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)

        torch_out = torch.special.gammainc(torch.tensor(a_np), torch.tensor(x_np))
        mlx_out = flashlight.special.gammainc(flashlight.tensor(a_np), flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"gammainc mismatch: {max_diff}"

    @pytest.mark.parity
    def test_gammaincc_parity(self):
        """Test gammaincc matches PyTorch."""
        a_np = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        x_np = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)

        torch_out = torch.special.gammaincc(torch.tensor(a_np), torch.tensor(x_np))
        mlx_out = flashlight.special.gammaincc(flashlight.tensor(a_np), flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"gammaincc mismatch: {max_diff}"


class TestBesselFunctionsParity:
    """Test Bessel function parity with PyTorch."""

    @pytest.mark.parity
    def test_bessel_j0_parity(self):
        """Test bessel_j0 matches PyTorch."""
        x_np = np.linspace(0, 10, 50).astype(np.float32)

        torch_out = torch.special.bessel_j0(torch.tensor(x_np))
        mlx_out = flashlight.special.bessel_j0(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"bessel_j0 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_bessel_j1_parity(self):
        """Test bessel_j1 matches PyTorch."""
        x_np = np.linspace(0, 10, 50).astype(np.float32)

        torch_out = torch.special.bessel_j1(torch.tensor(x_np))
        mlx_out = flashlight.special.bessel_j1(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"bessel_j1 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_bessel_y0_parity(self):
        """Test bessel_y0 matches PyTorch."""
        x_np = np.linspace(0.5, 10, 50).astype(np.float32)

        torch_out = torch.special.bessel_y0(torch.tensor(x_np))
        mlx_out = flashlight.special.bessel_y0(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"bessel_y0 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_bessel_y1_parity(self):
        """Test bessel_y1 matches PyTorch."""
        x_np = np.linspace(0.5, 10, 50).astype(np.float32)

        torch_out = torch.special.bessel_y1(torch.tensor(x_np))
        mlx_out = flashlight.special.bessel_y1(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-2, f"bessel_y1 mismatch: {max_diff}"


class TestModifiedBesselParity:
    """Test modified Bessel function parity with PyTorch."""

    @pytest.mark.parity
    def test_i0_parity(self):
        """Test modified_bessel_i0 matches PyTorch."""
        x_np = np.linspace(0, 5, 50).astype(np.float32)

        torch_out = torch.special.i0(torch.tensor(x_np))
        mlx_out = flashlight.special.modified_bessel_i0(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"i0 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_i1_parity(self):
        """Test modified_bessel_i1 matches PyTorch."""
        x_np = np.linspace(0, 5, 50).astype(np.float32)

        torch_out = torch.special.i1(torch.tensor(x_np))
        mlx_out = flashlight.special.modified_bessel_i1(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"i1 mismatch: {max_diff}"

    @pytest.mark.parity
    def test_i0e_parity(self):
        """Test i0e matches PyTorch."""
        x_np = np.linspace(0, 10, 50).astype(np.float32)

        torch_out = torch.special.i0e(torch.tensor(x_np))
        mlx_out = flashlight.special.i0e(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"i0e mismatch: {max_diff}"

    @pytest.mark.parity
    def test_i1e_parity(self):
        """Test i1e matches PyTorch."""
        x_np = np.linspace(0, 10, 50).astype(np.float32)

        torch_out = torch.special.i1e(torch.tensor(x_np))
        mlx_out = flashlight.special.i1e(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"i1e mismatch: {max_diff}"


class TestNormalDistParity:
    """Test normal distribution function parity with PyTorch."""

    @pytest.mark.parity
    def test_ndtr_parity(self):
        """Test ndtr matches PyTorch."""
        x_np = np.linspace(-3, 3, 50).astype(np.float32)

        torch_out = torch.special.ndtr(torch.tensor(x_np))
        mlx_out = flashlight.special.ndtr(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"ndtr mismatch: {max_diff}"

    @pytest.mark.parity
    def test_ndtri_parity(self):
        """Test ndtri matches PyTorch."""
        x_np = np.linspace(0.01, 0.99, 50).astype(np.float32)

        torch_out = torch.special.ndtri(torch.tensor(x_np))
        mlx_out = flashlight.special.ndtri(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"ndtri mismatch: {max_diff}"

    @pytest.mark.parity
    def test_log_ndtr_parity(self):
        """Test log_ndtr matches PyTorch."""
        x_np = np.linspace(-3, 3, 50).astype(np.float32)

        torch_out = torch.special.log_ndtr(torch.tensor(x_np))
        mlx_out = flashlight.special.log_ndtr(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-4, f"log_ndtr mismatch: {max_diff}"


class TestPolynomialsParity:
    """Test polynomial function parity with PyTorch."""

    @pytest.mark.parity
    def test_chebyshev_t_parity(self):
        """Test chebyshev_polynomial_t matches PyTorch."""
        x_np = np.linspace(-1, 1, 20).astype(np.float32)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n_np = np.full(20, n_val, dtype=np.int64)
            torch_out = torch.special.chebyshev_polynomial_t(torch.tensor(x_np), torch.tensor(n_np))
            mlx_out = flashlight.special.chebyshev_polynomial_t(
                flashlight.tensor(x_np), flashlight.tensor(n_np)
            )

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-5, f"chebyshev_t (n={n_val}) mismatch: {max_diff}"

    @pytest.mark.parity
    def test_chebyshev_u_parity(self):
        """Test chebyshev_polynomial_u matches PyTorch."""
        x_np = np.linspace(-1, 1, 20).astype(np.float32)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n_np = np.full(20, n_val, dtype=np.int64)
            torch_out = torch.special.chebyshev_polynomial_u(torch.tensor(x_np), torch.tensor(n_np))
            mlx_out = flashlight.special.chebyshev_polynomial_u(
                flashlight.tensor(x_np), flashlight.tensor(n_np)
            )

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-5, f"chebyshev_u (n={n_val}) mismatch: {max_diff}"

    @pytest.mark.parity
    def test_hermite_h_parity(self):
        """Test hermite_polynomial_h matches PyTorch."""
        x_np = np.linspace(-2, 2, 20).astype(np.float32)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n_np = np.full(20, n_val, dtype=np.int64)
            torch_out = torch.special.hermite_polynomial_h(torch.tensor(x_np), torch.tensor(n_np))
            mlx_out = flashlight.special.hermite_polynomial_h(
                flashlight.tensor(x_np), flashlight.tensor(n_np)
            )

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-4, f"hermite_h (n={n_val}) mismatch: {max_diff}"

    @pytest.mark.parity
    def test_hermite_he_parity(self):
        """Test hermite_polynomial_he matches PyTorch."""
        x_np = np.linspace(-2, 2, 20).astype(np.float32)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n_np = np.full(20, n_val, dtype=np.int64)
            torch_out = torch.special.hermite_polynomial_he(torch.tensor(x_np), torch.tensor(n_np))
            mlx_out = flashlight.special.hermite_polynomial_he(
                flashlight.tensor(x_np), flashlight.tensor(n_np)
            )

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-4, f"hermite_he (n={n_val}) mismatch: {max_diff}"

    @pytest.mark.parity
    def test_laguerre_parity(self):
        """Test laguerre_polynomial_l matches PyTorch."""
        x_np = np.linspace(0, 5, 20).astype(np.float32)

        for n_val in [0, 1, 2, 3, 4]:
            n_np = np.full(20, n_val, dtype=np.int64)
            torch_out = torch.special.laguerre_polynomial_l(torch.tensor(x_np), torch.tensor(n_np))
            mlx_out = flashlight.special.laguerre_polynomial_l(
                flashlight.tensor(x_np), flashlight.tensor(n_np)
            )

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-4, f"laguerre (n={n_val}) mismatch: {max_diff}"

    @pytest.mark.parity
    def test_legendre_parity(self):
        """Test legendre_polynomial_p matches PyTorch."""
        x_np = np.linspace(-1, 1, 20).astype(np.float32)

        for n_val in [0, 1, 2, 3, 4, 5]:
            n_np = np.full(20, n_val, dtype=np.int64)
            torch_out = torch.special.legendre_polynomial_p(torch.tensor(x_np), torch.tensor(n_np))
            mlx_out = flashlight.special.legendre_polynomial_p(
                flashlight.tensor(x_np), flashlight.tensor(n_np)
            )

            max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
            assert max_diff < 1e-5, f"legendre (n={n_val}) mismatch: {max_diff}"


class TestErrorFunctionsParity:
    """Test error function parity with PyTorch."""

    @pytest.mark.parity
    def test_erf_parity(self):
        """Test erf matches PyTorch."""
        x_np = np.linspace(-3, 3, 100).astype(np.float32)

        torch_out = torch.erf(torch.tensor(x_np))
        mlx_out = flashlight.special.erf(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"erf mismatch: {max_diff}"

    @pytest.mark.parity
    def test_erfc_parity(self):
        """Test erfc matches PyTorch."""
        x_np = np.linspace(-3, 3, 100).astype(np.float32)

        torch_out = torch.erfc(torch.tensor(x_np))
        mlx_out = flashlight.special.erfc(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"erfc mismatch: {max_diff}"

    @pytest.mark.parity
    def test_erfinv_parity(self):
        """Test erfinv matches PyTorch."""
        x_np = np.linspace(-0.9, 0.9, 50).astype(np.float32)

        torch_out = torch.erfinv(torch.tensor(x_np))
        mlx_out = flashlight.special.erfinv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"erfinv mismatch: {max_diff}"

    @pytest.mark.parity
    def test_erfcx_parity(self):
        """Test erfcx matches PyTorch."""
        x_np = np.linspace(0, 3, 50).astype(np.float32)

        torch_out = torch.special.erfcx(torch.tensor(x_np))
        mlx_out = flashlight.special.erfcx(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-3, f"erfcx mismatch: {max_diff}"


class TestSimpleFunctionsParity:
    """Test simple function parity with PyTorch."""

    @pytest.mark.parity
    def test_expit_parity(self):
        """Test expit (sigmoid) matches PyTorch."""
        x_np = np.linspace(-5, 5, 50).astype(np.float32)

        torch_out = torch.sigmoid(torch.tensor(x_np))
        mlx_out = flashlight.special.expit(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"expit mismatch: {max_diff}"

    @pytest.mark.parity
    def test_logit_parity(self):
        """Test logit matches PyTorch."""
        x_np = np.linspace(0.1, 0.9, 50).astype(np.float32)

        torch_out = torch.logit(torch.tensor(x_np))
        mlx_out = flashlight.special.logit(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"logit mismatch: {max_diff}"

    @pytest.mark.parity
    def test_softmax_parity(self):
        """Test softmax matches PyTorch."""
        x_np = np.random.randn(10, 5).astype(np.float32)

        torch_out = torch.softmax(torch.tensor(x_np), dim=-1)
        mlx_out = flashlight.special.softmax(flashlight.tensor(x_np), dim=-1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"softmax mismatch: {max_diff}"

    @pytest.mark.parity
    def test_log_softmax_parity(self):
        """Test log_softmax matches PyTorch."""
        x_np = np.random.randn(10, 5).astype(np.float32)

        torch_out = torch.log_softmax(torch.tensor(x_np), dim=-1)
        mlx_out = flashlight.special.log_softmax(flashlight.tensor(x_np), dim=-1)

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"log_softmax mismatch: {max_diff}"

    @pytest.mark.parity
    def test_xlogy_parity(self):
        """Test xlogy matches PyTorch."""
        x_np = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        torch_out = torch.special.xlogy(torch.tensor(x_np), torch.tensor(y_np))
        mlx_out = flashlight.special.xlogy(flashlight.tensor(x_np), flashlight.tensor(y_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"xlogy mismatch: {max_diff}"

    @pytest.mark.parity
    def test_xlog1py_parity(self):
        """Test xlog1py matches PyTorch."""
        x_np = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)

        torch_out = torch.special.xlog1py(torch.tensor(x_np), torch.tensor(y_np))
        mlx_out = flashlight.special.xlog1py(flashlight.tensor(x_np), flashlight.tensor(y_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"xlog1py mismatch: {max_diff}"

    @pytest.mark.parity
    def test_entr_parity(self):
        """Test entr matches PyTorch."""
        x_np = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)

        torch_out = torch.special.entr(torch.tensor(x_np))
        mlx_out = flashlight.special.entr(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out._mlx_array)))
        assert max_diff < 1e-5, f"entr mismatch: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
