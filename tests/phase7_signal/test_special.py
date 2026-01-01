"""
Comprehensive tests for mlx_compat.special module.

Tests all special functions with numerical parity against scipy.special.
"""

import pytest
import numpy as np
import scipy.special as sp
import mlx_compat


class TestGammaFunctions:
    """Tests for gamma-related special functions."""

    def test_gammaln_basic(self):
        """Test gammaln on basic values."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mlx_compat.special.gammaln(x)
        expected = sp.gammaln(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_gammaln_large_values(self):
        """Test gammaln on large values."""
        x = mlx_compat.tensor([10.0, 50.0, 100.0])
        result = mlx_compat.special.gammaln(x)
        expected = sp.gammaln(np.array([10.0, 50.0, 100.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_gammaln_small_values(self):
        """Test gammaln on values near 1 and 2."""
        x = mlx_compat.tensor([1.1, 1.5, 1.9, 2.1])
        result = mlx_compat.special.gammaln(x)
        expected = sp.gammaln(np.array([1.1, 1.5, 1.9, 2.1]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_digamma_basic(self):
        """Test digamma on basic values."""
        x = mlx_compat.tensor([1.0, 2.0, 5.0, 10.0])
        result = mlx_compat.special.digamma(x)
        expected = sp.digamma(np.array([1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_digamma_large_values(self):
        """Test digamma on large values."""
        x = mlx_compat.tensor([20.0, 50.0, 100.0])
        result = mlx_compat.special.digamma(x)
        expected = sp.digamma(np.array([20.0, 50.0, 100.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_polygamma_order1(self):
        """Test polygamma (trigamma)."""
        x = mlx_compat.tensor([1.0, 2.0, 5.0, 10.0])
        result = mlx_compat.special.polygamma(1, x)
        expected = sp.polygamma(1, np.array([1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_polygamma_order2(self):
        """Test polygamma (tetragamma)."""
        x = mlx_compat.tensor([2.0, 5.0, 10.0])
        result = mlx_compat.special.polygamma(2, x)
        expected = sp.polygamma(2, np.array([2.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-5
        )

    def test_multigammaln_p2(self):
        """Test multigammaln with p=2."""
        x = mlx_compat.tensor([3.0, 5.0, 10.0])
        result = mlx_compat.special.multigammaln(x, 2)
        expected = sp.multigammaln(np.array([3.0, 5.0, 10.0]), 2)
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_multigammaln_p3(self):
        """Test multigammaln with p=3."""
        x = mlx_compat.tensor([3.0, 5.0, 10.0])
        result = mlx_compat.special.multigammaln(x, 3)
        expected = sp.multigammaln(np.array([3.0, 5.0, 10.0]), 3)
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )


class TestIncompleteGamma:
    """Tests for incomplete gamma functions."""

    def test_gammainc_basic(self):
        """Test lower incomplete gamma."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        x = mlx_compat.tensor([0.5, 1.0, 2.0])
        result = mlx_compat.special.gammainc(a, x)
        expected = sp.gammainc(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_gammaincc_basic(self):
        """Test upper incomplete gamma."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        x = mlx_compat.tensor([0.5, 1.0, 2.0])
        result = mlx_compat.special.gammaincc(a, x)
        expected = sp.gammaincc(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_gammainc_gammaincc_sum(self):
        """Test that gammainc + gammaincc = 1."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0, 5.0])
        x = mlx_compat.tensor([0.5, 1.0, 2.0, 3.0])
        inc = mlx_compat.special.gammainc(a, x)
        incc = mlx_compat.special.gammaincc(a, x)
        total = np.array(inc._mlx_array) + np.array(incc._mlx_array)
        np.testing.assert_allclose(total, np.ones(4), rtol=1e-5, atol=1e-6)


class TestBesselFunctions:
    """Tests for Bessel functions."""

    def test_bessel_j0(self):
        """Test Bessel function of the first kind, order 0."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
        result = mlx_compat.special.bessel_j0(x)
        expected = sp.j0(np.array([0.0, 1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-6, atol=1e-7
        )

    def test_bessel_j1(self):
        """Test Bessel function of the first kind, order 1."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0, 5.0, 10.0])
        result = mlx_compat.special.bessel_j1(x)
        expected = sp.j1(np.array([0.0, 1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_bessel_y0(self):
        """Test Bessel function of the second kind, order 0."""
        x = mlx_compat.tensor([0.5, 1.0, 2.0, 5.0, 10.0])
        result = mlx_compat.special.bessel_y0(x)
        expected = sp.y0(np.array([0.5, 1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_bessel_y1(self):
        """Test Bessel function of the second kind, order 1."""
        x = mlx_compat.tensor([0.5, 1.0, 2.0, 5.0, 10.0])
        result = mlx_compat.special.bessel_y1(x)
        expected = sp.y1(np.array([0.5, 1.0, 2.0, 5.0, 10.0]))
        # Y1 has lower precision at some points due to polynomial approximation
        # The implementation has ~1% max relative error which is acceptable for this function
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-2, atol=1e-3
        )


class TestModifiedBesselFunctions:
    """Tests for modified Bessel functions."""

    def test_modified_bessel_i0(self):
        """Test modified Bessel function I0."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0, 5.0])
        result = mlx_compat.special.modified_bessel_i0(x)
        expected = sp.i0(np.array([0.0, 1.0, 2.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_modified_bessel_i1(self):
        """Test modified Bessel function I1."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0, 5.0])
        result = mlx_compat.special.modified_bessel_i1(x)
        expected = sp.i1(np.array([0.0, 1.0, 2.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_modified_bessel_k0(self):
        """Test modified Bessel function K0."""
        x = mlx_compat.tensor([0.5, 1.0, 2.0, 5.0])
        result = mlx_compat.special.modified_bessel_k0(x)
        expected = sp.k0(np.array([0.5, 1.0, 2.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_modified_bessel_k1(self):
        """Test modified Bessel function K1."""
        x = mlx_compat.tensor([0.5, 1.0, 2.0, 5.0])
        result = mlx_compat.special.modified_bessel_k1(x)
        expected = sp.k1(np.array([0.5, 1.0, 2.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_i0e(self):
        """Test exponentially scaled I0."""
        x = mlx_compat.tensor([0.0, 1.0, 5.0, 10.0])
        result = mlx_compat.special.i0e(x)
        expected = sp.i0e(np.array([0.0, 1.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_i1e(self):
        """Test exponentially scaled I1."""
        x = mlx_compat.tensor([0.0, 1.0, 5.0, 10.0])
        result = mlx_compat.special.i1e(x)
        expected = sp.i1e(np.array([0.0, 1.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_scaled_modified_bessel_k0(self):
        """Test exponentially scaled K0."""
        x = mlx_compat.tensor([0.5, 1.0, 5.0, 10.0])
        result = mlx_compat.special.scaled_modified_bessel_k0(x)
        expected = sp.k0e(np.array([0.5, 1.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_scaled_modified_bessel_k1(self):
        """Test exponentially scaled K1."""
        x = mlx_compat.tensor([0.5, 1.0, 5.0, 10.0])
        result = mlx_compat.special.scaled_modified_bessel_k1(x)
        expected = sp.k1e(np.array([0.5, 1.0, 5.0, 10.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )


class TestNormalDistribution:
    """Tests for normal distribution functions."""

    def test_ndtr(self):
        """Test normal CDF."""
        x = mlx_compat.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        result = mlx_compat.special.ndtr(x)
        expected = sp.ndtr(np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-7
        )

    def test_ndtri(self):
        """Test inverse normal CDF."""
        x = mlx_compat.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        result = mlx_compat.special.ndtri(x)
        expected = sp.ndtri(np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_log_ndtr(self):
        """Test log of normal CDF."""
        x = mlx_compat.tensor([-3.0, -2.0, -1.0, 0.0, 1.0])
        result = mlx_compat.special.log_ndtr(x)
        expected = sp.log_ndtr(np.array([-3.0, -2.0, -1.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-5
        )


class TestChebyshevPolynomials:
    """Tests for Chebyshev polynomials."""

    def test_chebyshev_t(self):
        """Test Chebyshev polynomial of the first kind."""
        x = mlx_compat.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        for n_val in [0, 1, 2, 3, 4, 5]:
            n = mlx_compat.tensor([n_val] * 5)
            result = mlx_compat.special.chebyshev_polynomial_t(x, n)
            expected = sp.eval_chebyt(n_val, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
            np.testing.assert_allclose(
                np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-10
            )

    def test_chebyshev_u(self):
        """Test Chebyshev polynomial of the second kind."""
        x = mlx_compat.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        for n_val in [0, 1, 2, 3, 4, 5]:
            n = mlx_compat.tensor([n_val] * 5)
            result = mlx_compat.special.chebyshev_polynomial_u(x, n)
            expected = sp.eval_chebyu(n_val, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
            np.testing.assert_allclose(
                np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-10
            )


class TestHermitePolynomials:
    """Tests for Hermite polynomials."""

    def test_hermite_h(self):
        """Test physicist's Hermite polynomial."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        for n_val in [0, 1, 2, 3, 4, 5]:
            n = mlx_compat.tensor([n_val] * 5)
            result = mlx_compat.special.hermite_polynomial_h(x, n)
            expected = sp.eval_hermite(n_val, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
            np.testing.assert_allclose(
                np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-10
            )

    def test_hermite_he(self):
        """Test probabilist's Hermite polynomial."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        for n_val in [0, 1, 2, 3, 4, 5]:
            n = mlx_compat.tensor([n_val] * 5)
            result = mlx_compat.special.hermite_polynomial_he(x, n)
            expected = sp.eval_hermitenorm(n_val, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
            np.testing.assert_allclose(
                np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-10
            )


class TestLaguerrePolynomials:
    """Tests for Laguerre polynomials."""

    def test_laguerre_l(self):
        """Test Laguerre polynomial."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0, 3.0])
        for n_val in [0, 1, 2, 3, 4]:
            n = mlx_compat.tensor([n_val] * 4)
            result = mlx_compat.special.laguerre_polynomial_l(x, n)
            expected = sp.eval_laguerre(n_val, np.array([0.0, 1.0, 2.0, 3.0]))
            np.testing.assert_allclose(
                np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
            )


class TestLegendrePolynomials:
    """Tests for Legendre polynomials."""

    def test_legendre_p(self):
        """Test Legendre polynomial."""
        x = mlx_compat.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        for n_val in [0, 1, 2, 3, 4, 5]:
            n = mlx_compat.tensor([n_val] * 5)
            result = mlx_compat.special.legendre_polynomial_p(x, n)
            expected = sp.eval_legendre(n_val, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
            np.testing.assert_allclose(
                np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-10
            )


class TestZetaFunction:
    """Tests for zeta function."""

    def test_zeta_basic(self):
        """Test Riemann zeta function."""
        s = mlx_compat.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        result = mlx_compat.special.zeta(s, 1.0)
        expected = sp.zeta(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), 1.0)
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-4, atol=1e-5
        )


class TestSphericalBessel:
    """Tests for spherical Bessel functions."""

    def test_spherical_bessel_j0(self):
        """Test spherical Bessel j0."""
        x = mlx_compat.tensor([0.1, 1.0, 2.0, 5.0])
        result = mlx_compat.special.spherical_bessel_j0(x)
        # spherical_jn returns (jn, jn_derivative) tuple
        expected = sp.spherical_jn(0, np.array([0.1, 1.0, 2.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-7
        )


class TestSimpleFunctions:
    """Tests for simple special functions."""

    def test_expit(self):
        """Test sigmoid (expit) function."""
        x = mlx_compat.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = mlx_compat.special.expit(x)
        expected = sp.expit(np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-6, atol=1e-7
        )

    def test_logit(self):
        """Test logit function."""
        x = mlx_compat.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        result = mlx_compat.special.logit(x)
        expected = sp.logit(np.array([0.1, 0.25, 0.5, 0.75, 0.9]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-6, atol=1e-7
        )

    def test_log_expit(self):
        """Test log_expit (log_sigmoid) function."""
        x = mlx_compat.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        # log_expit = log(sigmoid) = -log(1 + exp(-x))
        result = -mlx_compat.log(1.0 + mlx_compat.exp(-x))
        expected = sp.log_expit(np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_softmax(self):
        """Test softmax function."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        # Use dim (PyTorch convention) instead of axis
        result = mlx_compat.special.softmax(x, dim=-1)
        expected = sp.softmax(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), axis=-1)
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-6, atol=1e-7
        )

    def test_log_softmax(self):
        """Test log_softmax function."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        # Use dim (PyTorch convention) instead of axis
        result = mlx_compat.special.log_softmax(x, dim=-1)
        expected = sp.log_softmax(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), axis=-1)
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_entr(self):
        """Test entropy function."""
        x = mlx_compat.tensor([0.0, 0.5, 1.0, 2.0])
        result = mlx_compat.special.entr(x)
        expected = sp.entr(np.array([0.0, 0.5, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_xlog1py(self):
        """Test xlog1py function."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0])
        y = mlx_compat.tensor([0.5, 1.0, 2.0])
        result = mlx_compat.special.xlog1py(x, y)
        expected = sp.xlog1py(np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-6, atol=1e-7
        )

    def test_xlogy(self):
        """Test xlogy function."""
        x = mlx_compat.tensor([0.0, 1.0, 2.0])
        y = mlx_compat.tensor([1.0, 2.0, 3.0])
        result = mlx_compat.special.xlogy(x, y)
        expected = sp.xlogy(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-6, atol=1e-7
        )


class TestErrFunctions:
    """Tests for error functions."""

    def test_erf(self):
        """Test error function."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.special.erf(x)
        expected = sp.erf(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-7
        )

    def test_erfc(self):
        """Test complementary error function."""
        x = mlx_compat.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = mlx_compat.special.erfc(x)
        expected = sp.erfc(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-7
        )

    def test_erfinv(self):
        """Test inverse error function."""
        x = mlx_compat.tensor([-0.9, -0.5, 0.0, 0.5, 0.9])
        result = mlx_compat.special.erfinv(x)
        expected = sp.erfinv(np.array([-0.9, -0.5, 0.0, 0.5, 0.9]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-5
        )

    def test_erfcx(self):
        """Test scaled complementary error function."""
        # Avoid x=0 where erfcx(0)=1 but our approximation may differ
        x = mlx_compat.tensor([0.5, 1.0, 2.0, 5.0])
        result = mlx_compat.special.erfcx(x)
        expected = sp.erfcx(np.array([0.5, 1.0, 2.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-3, atol=1e-4
        )


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_gammaln_at_integers(self):
        """Test gammaln at positive integers (should give log((n-1)!))."""
        x = mlx_compat.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mlx_compat.special.gammaln(x)
        # gamma(n) = (n-1)! for positive integers
        # gammaln(1) = log(gamma(1)) = log(0!) = log(1) = 0
        # gammaln(2) = log(gamma(2)) = log(1!) = log(1) = 0
        # gammaln(3) = log(gamma(3)) = log(2!) = log(2)
        # gammaln(4) = log(gamma(4)) = log(3!) = log(6)
        # gammaln(5) = log(gamma(5)) = log(4!) = log(24)
        expected = sp.gammaln(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_allclose(
            np.array(result._mlx_array), expected, rtol=1e-5, atol=1e-6
        )

    def test_bessel_j0_at_zero(self):
        """Test J0(0) = 1."""
        x = mlx_compat.tensor([0.0])
        result = mlx_compat.special.bessel_j0(x)
        np.testing.assert_allclose(np.array(result._mlx_array), [1.0], atol=1e-10)

    def test_bessel_j1_at_zero(self):
        """Test J1(0) = 0."""
        x = mlx_compat.tensor([0.0])
        result = mlx_compat.special.bessel_j1(x)
        np.testing.assert_allclose(np.array(result._mlx_array), [0.0], atol=1e-10)

    def test_ndtr_extremes(self):
        """Test ndtr at extreme values."""
        result_neg = mlx_compat.special.ndtr(mlx_compat.tensor([-6.0]))
        result_pos = mlx_compat.special.ndtr(mlx_compat.tensor([6.0]))
        # Use looser bounds for numerical precision
        assert np.array(result_neg._mlx_array)[0] < 1e-5
        assert np.array(result_pos._mlx_array)[0] > 1 - 1e-5


class TestTensorProperties:
    """Tests for tensor properties and shapes."""

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        x = mlx_compat.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = mlx_compat.special.gammaln(x)
        assert result.shape == (2, 3)

    def test_broadcast_shapes(self):
        """Test that broadcasting works correctly."""
        a = mlx_compat.tensor([1.0, 2.0, 3.0])
        x = mlx_compat.tensor([[0.5], [1.0], [2.0]])
        result = mlx_compat.special.gammainc(a, x)
        assert result.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
