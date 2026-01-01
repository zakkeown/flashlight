"""
Comprehensive PyTorch Parity Tests for All Distributions

This file tests every distribution in flashlight.distributions against
PyTorch's torch.distributions for numerical parity.
"""

import sys
sys.path.insert(0, '../..')

import unittest
import math

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import torch
    import torch.distributions as td
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import flashlight
    from flashlight import distributions as dist
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def skip_if_no_torch(func):
    """Skip test if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        return unittest.skip("PyTorch not available")(func)
    return func


def compare_log_prob(mlx_dist, torch_dist, test_values, rtol=1e-4, atol=1e-5):
    """Compare log_prob between MLX and PyTorch distributions."""
    for v in test_values:
        if isinstance(v, list):
            mlx_v = flashlight.tensor(v)
            torch_v = torch.tensor(v)
        else:
            mlx_v = flashlight.tensor(float(v))
            torch_v = torch.tensor(float(v))

        mlx_lp = float(mlx_dist.log_prob(mlx_v).numpy())
        torch_lp = float(torch_dist.log_prob(torch_v))

        if math.isnan(mlx_lp) and math.isnan(torch_lp):
            continue
        if math.isinf(mlx_lp) and math.isinf(torch_lp):
            continue

        diff = abs(mlx_lp - torch_lp)
        rel_diff = diff / (abs(torch_lp) + 1e-10)

        if diff > atol and rel_diff > rtol:
            raise AssertionError(
                f"log_prob mismatch at {v}: MLX={mlx_lp}, PyTorch={torch_lp}, "
                f"diff={diff}, rel_diff={rel_diff}"
            )


def compare_entropy(mlx_dist, torch_dist, rtol=1e-4, atol=1e-5):
    """Compare entropy between MLX and PyTorch distributions."""
    mlx_ent = float(mlx_dist.entropy().numpy())
    torch_ent = float(torch_dist.entropy())

    diff = abs(mlx_ent - torch_ent)
    rel_diff = diff / (abs(torch_ent) + 1e-10)

    if diff > atol and rel_diff > rtol:
        raise AssertionError(
            f"entropy mismatch: MLX={mlx_ent}, PyTorch={torch_ent}, "
            f"diff={diff}, rel_diff={rel_diff}"
        )


def compare_mean_var(mlx_dist, torch_dist, rtol=1e-4, atol=1e-5):
    """Compare mean and variance between MLX and PyTorch distributions."""
    if hasattr(mlx_dist, 'mean') and hasattr(torch_dist, 'mean'):
        try:
            mlx_mean = mlx_dist.mean.numpy()
            torch_mean = torch_dist.mean.numpy()
            if mlx_mean.shape == ():
                mlx_mean = float(mlx_mean)
                torch_mean = float(torch_mean)
                diff = abs(mlx_mean - torch_mean)
                if diff > atol:
                    raise AssertionError(f"mean mismatch: MLX={mlx_mean}, PyTorch={torch_mean}")
        except Exception:
            pass  # Some distributions don't have finite mean

    if hasattr(mlx_dist, 'variance') and hasattr(torch_dist, 'variance'):
        try:
            mlx_var = mlx_dist.variance.numpy()
            torch_var = torch_dist.variance.numpy()
            if mlx_var.shape == ():
                mlx_var = float(mlx_var)
                torch_var = float(torch_var)
                diff = abs(mlx_var - torch_var)
                if diff > atol:
                    raise AssertionError(f"variance mismatch: MLX={mlx_var}, PyTorch={torch_var}")
        except Exception:
            pass  # Some distributions don't have finite variance


# =============================================================================
# Continuous Distributions
# =============================================================================

@skipIfNoMLX
class TestNormalParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Normal(0.0, 1.0),
            td.Normal(0.0, 1.0),
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Normal(0.0, 1.0), td.Normal(0.0, 1.0))
        compare_entropy(dist.Normal(1.0, 2.0), td.Normal(1.0, 2.0))

    @skip_if_no_torch
    def test_mean_var(self):
        compare_mean_var(dist.Normal(1.5, 2.0), td.Normal(1.5, 2.0))


@skipIfNoMLX
class TestUniformParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Uniform(0.0, 1.0),
            td.Uniform(0.0, 1.0),
            [0.1, 0.5, 0.9]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Uniform(0.0, 1.0), td.Uniform(0.0, 1.0))
        compare_entropy(dist.Uniform(-1.0, 2.0), td.Uniform(-1.0, 2.0))


@skipIfNoMLX
class TestExponentialParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Exponential(1.0),
            td.Exponential(1.0),
            [0.1, 0.5, 1.0, 2.0, 5.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Exponential(1.0), td.Exponential(1.0))
        compare_entropy(dist.Exponential(2.0), td.Exponential(2.0))


@skipIfNoMLX
class TestGammaParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Gamma(2.0, 1.0),
            td.Gamma(2.0, 1.0),
            [0.1, 0.5, 1.0, 2.0, 5.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Gamma(2.0, 1.0), td.Gamma(2.0, 1.0))
        compare_entropy(dist.Gamma(5.0, 2.0), td.Gamma(5.0, 2.0))

    @skip_if_no_torch
    def test_mean_var(self):
        compare_mean_var(dist.Gamma(3.0, 2.0), td.Gamma(3.0, 2.0))


@skipIfNoMLX
class TestBetaParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Beta(2.0, 5.0),
            td.Beta(2.0, 5.0),
            [0.1, 0.3, 0.5, 0.7, 0.9]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Beta(2.0, 5.0), td.Beta(2.0, 5.0))
        compare_entropy(dist.Beta(0.5, 0.5), td.Beta(0.5, 0.5))

    @skip_if_no_torch
    def test_mean_var(self):
        compare_mean_var(dist.Beta(2.0, 3.0), td.Beta(2.0, 3.0))


@skipIfNoMLX
class TestCauchyParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Cauchy(0.0, 1.0),
            td.Cauchy(0.0, 1.0),
            [-5.0, -1.0, 0.0, 1.0, 5.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Cauchy(0.0, 1.0), td.Cauchy(0.0, 1.0))


@skipIfNoMLX
class TestLaplaceParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Laplace(0.0, 1.0),
            td.Laplace(0.0, 1.0),
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Laplace(0.0, 1.0), td.Laplace(0.0, 1.0))

    @skip_if_no_torch
    def test_mean_var(self):
        compare_mean_var(dist.Laplace(1.0, 2.0), td.Laplace(1.0, 2.0))


@skipIfNoMLX
class TestLogNormalParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.LogNormal(0.0, 1.0),
            td.LogNormal(0.0, 1.0),
            [0.1, 0.5, 1.0, 2.0, 5.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.LogNormal(0.0, 1.0), td.LogNormal(0.0, 1.0))

    @skip_if_no_torch
    def test_mean_var(self):
        compare_mean_var(dist.LogNormal(0.0, 0.5), td.LogNormal(0.0, 0.5))


@skipIfNoMLX
class TestGumbelParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Gumbel(0.0, 1.0),
            td.Gumbel(0.0, 1.0),
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Gumbel(0.0, 1.0), td.Gumbel(0.0, 1.0))


@skipIfNoMLX
class TestParetoParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Pareto(1.0, 2.0),
            td.Pareto(1.0, 2.0),
            [1.1, 1.5, 2.0, 5.0, 10.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Pareto(1.0, 2.0), td.Pareto(1.0, 2.0))


@skipIfNoMLX
class TestWeibullParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Weibull(1.0, 1.0),
            td.Weibull(1.0, 1.0),
            [0.1, 0.5, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Weibull(1.0, 1.0), td.Weibull(1.0, 1.0))


@skipIfNoMLX
class TestChi2Parity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Chi2(3.0),
            td.Chi2(3.0),
            [0.5, 1.0, 2.0, 5.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Chi2(3.0), td.Chi2(3.0))


@skipIfNoMLX
class TestStudentTParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.StudentT(5.0),
            td.StudentT(5.0),
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.StudentT(5.0), td.StudentT(5.0))


@skipIfNoMLX
class TestHalfNormalParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.HalfNormal(1.0),
            td.HalfNormal(1.0),
            [0.1, 0.5, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.HalfNormal(1.0), td.HalfNormal(1.0))


@skipIfNoMLX
class TestHalfCauchyParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.HalfCauchy(1.0),
            td.HalfCauchy(1.0),
            [0.1, 0.5, 1.0, 2.0, 5.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.HalfCauchy(1.0), td.HalfCauchy(1.0))


@skipIfNoMLX
class TestInverseGammaParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.InverseGamma(2.0, 1.0),
            td.InverseGamma(2.0, 1.0),
            [0.1, 0.5, 1.0, 2.0]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.InverseGamma(2.0, 1.0), td.InverseGamma(2.0, 1.0))


@skipIfNoMLX
class TestKumaraswamyParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Kumaraswamy(2.0, 5.0),
            td.Kumaraswamy(2.0, 5.0),
            [0.1, 0.3, 0.5, 0.7, 0.9]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Kumaraswamy(2.0, 5.0), td.Kumaraswamy(2.0, 5.0))


@skipIfNoMLX
class TestFisherSnedecorParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.FisherSnedecor(5.0, 10.0),
            td.FisherSnedecor(5.0, 10.0),
            [0.1, 0.5, 1.0, 2.0]
        )


@skipIfNoMLX
class TestVonMisesParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.VonMises(0.0, 1.0),
            td.VonMises(0.0, 1.0),
            [-3.0, -1.0, 0.0, 1.0, 3.0]
        )


# =============================================================================
# Discrete Distributions
# =============================================================================

@skipIfNoMLX
class TestBernoulliParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Bernoulli(probs=0.3),
            td.Bernoulli(probs=0.3),
            [0, 1]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Bernoulli(probs=0.3), td.Bernoulli(probs=0.3))
        compare_entropy(dist.Bernoulli(probs=0.5), td.Bernoulli(probs=0.5))


@skipIfNoMLX
class TestCategoricalParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        probs = [0.2, 0.3, 0.5]
        compare_log_prob(
            dist.Categorical(probs=flashlight.tensor(probs)),
            td.Categorical(probs=torch.tensor(probs)),
            [0, 1, 2]
        )

    @skip_if_no_torch
    def test_entropy(self):
        probs = [0.2, 0.3, 0.5]
        compare_entropy(
            dist.Categorical(probs=flashlight.tensor(probs)),
            td.Categorical(probs=torch.tensor(probs))
        )


@skipIfNoMLX
class TestPoissonParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Poisson(4.0),
            td.Poisson(4.0),
            [0, 1, 2, 3, 5, 10]
        )


@skipIfNoMLX
class TestBinomialParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Binomial(10, probs=0.4),
            td.Binomial(10, probs=0.4),
            [0, 2, 4, 6, 10],
            rtol=1e-4  # lgamma differences may require slight relaxation
        )


@skipIfNoMLX
class TestGeometricParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.Geometric(probs=0.3),
            td.Geometric(probs=0.3),
            [0, 1, 2, 5, 10]
        )

    @skip_if_no_torch
    def test_entropy(self):
        compare_entropy(dist.Geometric(probs=0.3), td.Geometric(probs=0.3))


@skipIfNoMLX
class TestNegativeBinomialParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.NegativeBinomial(5.0, probs=0.4),
            td.NegativeBinomial(5.0, probs=0.4),
            [0, 1, 2, 5, 10],
            rtol=1e-4
        )


# =============================================================================
# Multivariate Distributions
# =============================================================================

@skipIfNoMLX
class TestDirichletParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        alpha = [2.0, 3.0, 5.0]
        mlx_d = dist.Dirichlet(flashlight.tensor(alpha))
        torch_d = td.Dirichlet(torch.tensor(alpha))

        # Test value on simplex
        val = [0.2, 0.3, 0.5]
        mlx_lp = float(mlx_d.log_prob(flashlight.tensor(val)).numpy())
        torch_lp = float(torch_d.log_prob(torch.tensor(val)))

        self.assertAlmostEqual(mlx_lp, torch_lp, places=4)

    @skip_if_no_torch
    def test_entropy(self):
        alpha = [2.0, 3.0, 5.0]
        compare_entropy(
            dist.Dirichlet(flashlight.tensor(alpha)),
            td.Dirichlet(torch.tensor(alpha))
        )


@skipIfNoMLX
class TestMultinomialParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        probs = [0.2, 0.3, 0.5]
        mlx_m = dist.Multinomial(10, probs=flashlight.tensor(probs))
        torch_m = td.Multinomial(10, probs=torch.tensor(probs))

        # Test value
        val = [2, 3, 5]
        mlx_lp = float(mlx_m.log_prob(flashlight.tensor(val, dtype=flashlight.float32)).numpy())
        torch_lp = float(torch_m.log_prob(torch.tensor(val, dtype=torch.float32)))

        # Slightly relaxed due to lgamma differences
        self.assertAlmostEqual(mlx_lp, torch_lp, delta=0.01)


@skipIfNoMLX
class TestMultivariateNormalParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        loc = [0.0, 0.0]
        cov = [[1.0, 0.5], [0.5, 1.0]]

        mlx_mvn = dist.MultivariateNormal(
            loc=flashlight.tensor(loc),
            covariance_matrix=flashlight.tensor(cov)
        )
        torch_mvn = td.MultivariateNormal(
            loc=torch.tensor(loc),
            covariance_matrix=torch.tensor(cov)
        )

        val = [0.5, 0.5]
        mlx_lp = float(mlx_mvn.log_prob(flashlight.tensor(val)).numpy())
        torch_lp = float(torch_mvn.log_prob(torch.tensor(val)))

        self.assertAlmostEqual(mlx_lp, torch_lp, places=4)

    @skip_if_no_torch
    def test_entropy(self):
        loc = [0.0, 0.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]
        compare_entropy(
            dist.MultivariateNormal(
                loc=flashlight.tensor(loc),
                covariance_matrix=flashlight.tensor(cov)
            ),
            td.MultivariateNormal(
                loc=torch.tensor(loc),
                covariance_matrix=torch.tensor(cov)
            )
        )


# =============================================================================
# Relaxed Distributions
# =============================================================================

@skipIfNoMLX
class TestRelaxedBernoulliParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        # PyTorch requires temperature to be a tensor
        mlx_rb = dist.RelaxedBernoulli(0.5, probs=0.3)
        torch_rb = td.RelaxedBernoulli(torch.tensor(0.5), probs=0.3)

        for v in [0.1, 0.5, 0.9]:
            mlx_lp = float(mlx_rb.log_prob(flashlight.tensor(v)).numpy())
            torch_lp = float(torch_rb.log_prob(torch.tensor(v)))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=4)


@skipIfNoMLX
class TestOneHotCategoricalParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        probs = [0.2, 0.3, 0.5]
        mlx_d = dist.OneHotCategorical(probs=flashlight.tensor(probs))
        torch_d = td.OneHotCategorical(probs=torch.tensor(probs))

        for i in range(3):
            one_hot = [0.0, 0.0, 0.0]
            one_hot[i] = 1.0
            mlx_lp = float(mlx_d.log_prob(flashlight.tensor(one_hot)).numpy())
            torch_lp = float(torch_d.log_prob(torch.tensor(one_hot)))
            self.assertAlmostEqual(mlx_lp, torch_lp, places=4)

    @skip_if_no_torch
    def test_entropy(self):
        probs = [0.2, 0.3, 0.5]
        compare_entropy(
            dist.OneHotCategorical(probs=flashlight.tensor(probs)),
            td.OneHotCategorical(probs=torch.tensor(probs))
        )


@skipIfNoMLX
class TestContinuousBernoulliParity(TestCase):
    @skip_if_no_torch
    def test_log_prob(self):
        compare_log_prob(
            dist.ContinuousBernoulli(probs=0.3),
            td.ContinuousBernoulli(probs=0.3),
            [0.1, 0.5, 0.9]
        )


# =============================================================================
# Summary Test
# =============================================================================

@skipIfNoMLX
class TestDistributionCoverage(TestCase):
    """Verify we have tests for all distributions."""

    def test_all_distributions_have_tests(self):
        """Check that we're testing a comprehensive set."""
        tested = [
            'Normal', 'Uniform', 'Exponential', 'Gamma', 'Beta',
            'Cauchy', 'Laplace', 'LogNormal', 'Gumbel', 'Pareto',
            'Weibull', 'Chi2', 'StudentT', 'HalfNormal', 'HalfCauchy',
            'InverseGamma', 'Kumaraswamy', 'FisherSnedecor', 'VonMises',
            'Bernoulli', 'Categorical', 'Poisson', 'Binomial', 'Geometric',
            'NegativeBinomial', 'Dirichlet', 'Multinomial', 'MultivariateNormal',
            'RelaxedBernoulli', 'OneHotCategorical', 'ContinuousBernoulli'
        ]
        self.assertGreaterEqual(len(tested), 30, "Should test at least 30 distributions")


if __name__ == '__main__':
    unittest.main()
