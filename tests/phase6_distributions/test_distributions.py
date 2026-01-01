"""
Test Phase 6: Probability Distributions

Tests for all probability distributions in flashlight.distributions.
Includes numerical parity tests against PyTorch.
"""

import sys

sys.path.insert(0, "../..")

import math
import unittest

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import torch
    import torch.distributions as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlx.core as mx

    import flashlight
    from flashlight import distributions as dist

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


def requires_torch(func):
    """Skip test if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        return unittest.skip("PyTorch not available")(func)
    return func


@skipIfNoMLX
class TestNormal(TestCase):
    """Test Normal distribution."""

    def test_basic_properties(self):
        """Test mean, variance, stddev."""
        loc, scale = 2.0, 3.0
        n = dist.Normal(loc, scale)

        self.assertAlmostEqual(float(n.mean.numpy()), loc, places=5)
        self.assertAlmostEqual(float(n.variance.numpy()), scale**2, places=5)
        self.assertAlmostEqual(float(n.stddev.numpy()), scale, places=5)

    def test_sample_shape(self):
        """Test sample shape."""
        n = dist.Normal(0.0, 1.0)
        samples = n.sample((100,))
        self.assertEqual(samples.shape, (100,))

    @requires_torch
    def test_log_prob_parity(self):
        """Test log_prob matches PyTorch."""
        loc, scale = 1.5, 2.0
        mlx_n = dist.Normal(loc, scale)
        torch_n = td.Normal(loc, scale)

        test_vals = [0.0, 1.0, 2.0, -1.0, 5.0]
        for v in test_vals:
            mlx_lp = float(mlx_n.log_prob(flashlight.tensor(v)).numpy())
            torch_lp = float(torch_n.log_prob(torch.tensor(v)))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=5)

    @requires_torch
    def test_entropy_parity(self):
        """Test entropy matches PyTorch."""
        loc, scale = 1.0, 2.0
        mlx_n = dist.Normal(loc, scale)
        torch_n = td.Normal(loc, scale)

        mlx_ent = float(mlx_n.entropy().numpy())
        torch_ent = float(torch_n.entropy())
        self.assertAlmostEqual(mlx_ent, torch_ent, places=5)

    def test_cdf_icdf(self):
        """Test CDF and inverse CDF."""
        n = dist.Normal(0.0, 1.0)

        # CDF at 0 should be 0.5
        cdf_0 = float(n.cdf(flashlight.tensor(0.0)).numpy())
        self.assertAlmostEqual(cdf_0, 0.5, places=5)

        # ICDF at 0.5 should be 0
        icdf_05 = float(n.icdf(flashlight.tensor(0.5)).numpy())
        self.assertAlmostEqual(icdf_05, 0.0, places=4)


@skipIfNoMLX
class TestGamma(TestCase):
    """Test Gamma distribution."""

    def test_basic_properties(self):
        """Test mean, variance."""
        concentration, rate = 3.0, 2.0
        g = dist.Gamma(concentration, rate)

        expected_mean = concentration / rate
        expected_var = concentration / rate**2

        self.assertAlmostEqual(float(g.mean.numpy()), expected_mean, places=5)
        self.assertAlmostEqual(float(g.variance.numpy()), expected_var, places=5)

    def test_sample_shape(self):
        """Test sample shape."""
        g = dist.Gamma(2.0, 1.0)
        samples = g.sample((100,))
        self.assertEqual(samples.shape, (100,))
        # All samples should be positive
        self.assertTrue(all(s > 0 for s in samples.numpy()))

    def test_sample_statistics(self):
        """Test that sample mean/variance converge to theoretical."""
        concentration, rate = 5.0, 2.0
        g = dist.Gamma(concentration, rate)

        samples = g.sample((10000,))
        sample_mean = float(samples.mean().numpy())
        sample_var = float(samples.var().numpy())

        expected_mean = concentration / rate
        expected_var = concentration / rate**2

        # Allow 10% tolerance for sample statistics
        self.assertAlmostEqual(sample_mean, expected_mean, delta=0.1 * expected_mean)
        self.assertAlmostEqual(sample_var, expected_var, delta=0.2 * expected_var)

    @requires_torch
    def test_log_prob_parity(self):
        """Test log_prob matches PyTorch."""
        concentration, rate = 2.0, 1.5
        mlx_g = dist.Gamma(concentration, rate)
        torch_g = td.Gamma(concentration, rate)

        test_vals = [0.5, 1.0, 2.0, 5.0]
        for v in test_vals:
            mlx_lp = float(mlx_g.log_prob(flashlight.tensor(v)).numpy())
            torch_lp = float(torch_g.log_prob(torch.tensor(v)))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=4)

    @requires_torch
    def test_entropy_parity(self):
        """Test entropy matches PyTorch."""
        concentration, rate = 3.0, 2.0
        mlx_g = dist.Gamma(concentration, rate)
        torch_g = td.Gamma(concentration, rate)

        mlx_ent = float(mlx_g.entropy().numpy())
        torch_ent = float(torch_g.entropy())
        self.assertAlmostEqual(mlx_ent, torch_ent, places=4)


@skipIfNoMLX
class TestBeta(TestCase):
    """Test Beta distribution."""

    def test_basic_properties(self):
        """Test mean, variance."""
        a, b = 2.0, 5.0
        beta = dist.Beta(a, b)

        expected_mean = a / (a + b)
        expected_var = a * b / ((a + b) ** 2 * (a + b + 1))

        self.assertAlmostEqual(float(beta.mean.numpy()), expected_mean, places=5)
        self.assertAlmostEqual(float(beta.variance.numpy()), expected_var, places=5)

    def test_sample_range(self):
        """Test samples are in [0, 1]."""
        beta = dist.Beta(2.0, 3.0)
        samples = beta.sample((1000,))

        self.assertTrue(all(0 <= s <= 1 for s in samples.numpy()))

    @requires_torch
    def test_log_prob_parity(self):
        """Test log_prob matches PyTorch."""
        a, b = 2.0, 5.0
        mlx_beta = dist.Beta(a, b)
        torch_beta = td.Beta(a, b)

        test_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        for v in test_vals:
            mlx_lp = float(mlx_beta.log_prob(flashlight.tensor(v)).numpy())
            torch_lp = float(torch_beta.log_prob(torch.tensor(v)))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=4)

    @requires_torch
    def test_entropy_parity(self):
        """Test entropy matches PyTorch."""
        a, b = 3.0, 4.0
        mlx_beta = dist.Beta(a, b)
        torch_beta = td.Beta(a, b)

        mlx_ent = float(mlx_beta.entropy().numpy())
        torch_ent = float(torch_beta.entropy())
        self.assertAlmostEqual(mlx_ent, torch_ent, places=4)


@skipIfNoMLX
class TestPoisson(TestCase):
    """Test Poisson distribution."""

    def test_basic_properties(self):
        """Test mean, variance."""
        rate = 5.0
        p = dist.Poisson(rate)

        self.assertAlmostEqual(float(p.mean.numpy()), rate, places=5)
        self.assertAlmostEqual(float(p.variance.numpy()), rate, places=5)

    def test_sample_nonnegative(self):
        """Test samples are non-negative integers."""
        p = dist.Poisson(3.0)
        samples = p.sample((100,))

        for s in samples.numpy():
            self.assertGreaterEqual(s, 0)
            self.assertEqual(s, int(s))

    @requires_torch
    def test_log_prob_parity(self):
        """Test log_prob matches PyTorch."""
        rate = 4.0
        mlx_p = dist.Poisson(rate)
        torch_p = td.Poisson(rate)

        test_vals = [0, 1, 2, 3, 5, 10]
        for v in test_vals:
            mlx_lp = float(mlx_p.log_prob(flashlight.tensor(float(v))).numpy())
            torch_lp = float(torch_p.log_prob(torch.tensor(float(v))))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=4)


@skipIfNoMLX
class TestBinomial(TestCase):
    """Test Binomial distribution."""

    def test_basic_properties(self):
        """Test mean, variance."""
        n, p = 10, 0.3
        b = dist.Binomial(n, probs=p)

        expected_mean = n * p
        expected_var = n * p * (1 - p)

        self.assertAlmostEqual(float(b.mean.numpy()), expected_mean, places=5)
        self.assertAlmostEqual(float(b.variance.numpy()), expected_var, places=5)

    def test_sample_range(self):
        """Test samples are in [0, n]."""
        n, p = 20, 0.5
        b = dist.Binomial(n, probs=p)
        samples = b.sample((100,))

        for s in samples.numpy():
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, n)

    @requires_torch
    def test_log_prob_parity(self):
        """Test log_prob matches PyTorch."""
        n, p = 10, 0.4
        mlx_b = dist.Binomial(n, probs=p)
        torch_b = td.Binomial(n, probs=p)

        test_vals = [0, 2, 4, 6, 10]
        for v in test_vals:
            mlx_lp = float(mlx_b.log_prob(flashlight.tensor(float(v))).numpy())
            torch_lp = float(torch_b.log_prob(torch.tensor(float(v))))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=4)


@skipIfNoMLX
class TestDirichlet(TestCase):
    """Test Dirichlet distribution."""

    def test_sample_simplex(self):
        """Test samples sum to 1."""
        alpha = flashlight.tensor([1.0, 2.0, 3.0])
        d = dist.Dirichlet(alpha)
        samples = d.sample((100,))

        for i in range(100):
            sample_sum = float(samples[i].sum().numpy())
            self.assertAlmostEqual(sample_sum, 1.0, places=4)

    def test_mean(self):
        """Test mean is normalized concentration."""
        alpha = flashlight.tensor([2.0, 4.0, 4.0])
        d = dist.Dirichlet(alpha)

        expected_mean = [0.2, 0.4, 0.4]
        actual_mean = d.mean.numpy()

        for i in range(3):
            self.assertAlmostEqual(actual_mean[i], expected_mean[i], places=5)

    @requires_torch
    def test_log_prob_parity(self):
        """Test log_prob matches PyTorch."""
        alpha = [2.0, 3.0, 5.0]
        mlx_d = dist.Dirichlet(flashlight.tensor(alpha))
        torch_d = td.Dirichlet(torch.tensor(alpha))

        test_val = [0.2, 0.3, 0.5]
        mlx_lp = float(mlx_d.log_prob(flashlight.tensor(test_val)).numpy())
        torch_lp = float(torch_d.log_prob(torch.tensor(test_val)))
        self.assertAlmostEqual(mlx_lp, torch_lp, places=4)


@skipIfNoMLX
class TestSpecialFunctions(TestCase):
    """Test the special functions used by distributions."""

    def test_lgamma(self):
        """Test lgamma function."""
        from scipy import special as sp

        from flashlight.ops.special import lgamma

        test_vals = [0.5, 1.0, 2.0, 5.0, 10.0]
        for v in test_vals:
            mlx_result = float(lgamma(mx.array(v)))
            scipy_result = float(sp.gammaln(v))
            self.assertAlmostEqual(mlx_result, scipy_result, places=5)

    def test_digamma(self):
        """Test digamma function."""
        from scipy import special as sp

        from flashlight.ops.special import digamma

        test_vals = [0.5, 1.0, 2.0, 5.0, 10.0]
        for v in test_vals:
            mlx_result = float(digamma(mx.array(v)))
            scipy_result = float(sp.digamma(v))
            self.assertAlmostEqual(mlx_result, scipy_result, places=4)

    def test_betaln(self):
        """Test betaln function."""
        from scipy import special as sp

        from flashlight.ops.special import betaln

        test_pairs = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (5.0, 2.0)]
        for a, b in test_pairs:
            mlx_result = float(betaln(mx.array(a), mx.array(b)))
            scipy_result = float(sp.betaln(a, b))
            self.assertAlmostEqual(mlx_result, scipy_result, places=5)

    def test_gammainc(self):
        """Test gammainc function."""
        from scipy import special as sp

        from flashlight.ops.special import gammainc

        test_pairs = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (5.0, 2.0)]
        for a, x in test_pairs:
            mlx_result = float(gammainc(mx.array(a), mx.array(x)))
            scipy_result = float(sp.gammainc(a, x))
            self.assertAlmostEqual(mlx_result, scipy_result, places=4)

    def test_i0(self):
        """Test Bessel I0 function."""
        from scipy import special as sp

        from flashlight.ops.special import i0

        test_vals = [0.0, 0.5, 1.0, 2.0, 5.0]
        for v in test_vals:
            mlx_result = float(i0(mx.array(v)))
            scipy_result = float(sp.i0(v))
            rel_diff = abs(mlx_result - scipy_result) / (abs(scipy_result) + 1e-10)
            self.assertLess(rel_diff, 1e-4)

    def test_i1(self):
        """Test Bessel I1 function."""
        from scipy import special as sp

        from flashlight.ops.special import i1

        test_vals = [0.5, 1.0, 2.0, 5.0]
        for v in test_vals:
            mlx_result = float(i1(mx.array(v)))
            scipy_result = float(sp.i1(v))
            rel_diff = abs(mlx_result - scipy_result) / (abs(scipy_result) + 1e-10)
            self.assertLess(rel_diff, 1e-4)


@skipIfNoMLX
class TestNoScipyDependency(TestCase):
    """Test that distributions don't import scipy at runtime."""

    def test_no_scipy_import(self):
        """Verify scipy is not imported by distributions."""
        import sys

        # Remove scipy from modules if present
        scipy_modules = [k for k in sys.modules.keys() if k.startswith("scipy")]
        for m in scipy_modules:
            del sys.modules[m]

        # Reload distributions
        import importlib

        importlib.reload(dist)

        # Create and use distributions
        n = dist.Normal(0.0, 1.0)
        _ = n.sample((10,))
        _ = n.log_prob(flashlight.tensor(0.0))
        _ = n.entropy()

        g = dist.Gamma(2.0, 1.0)
        _ = g.sample((10,))
        _ = g.log_prob(flashlight.tensor(1.0))

        b = dist.Beta(2.0, 3.0)
        _ = b.sample((10,))
        _ = b.log_prob(flashlight.tensor(0.5))

        # Check scipy was not imported
        scipy_loaded = any(k.startswith("scipy") for k in sys.modules.keys())
        self.assertFalse(scipy_loaded, "scipy was imported by distributions")


if __name__ == "__main__":
    unittest.main()
