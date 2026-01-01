"""
Distributions Module

PyTorch-compatible torch.distributions module for MLX.
Provides probability distributions for sampling and log probability computation.
"""

from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .chi2 import Chi2
from .constraint_registry import biject_to, transform_to
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .fisher_snedecor import FisherSnedecor
from .gamma import Gamma
from .generalized_pareto import GeneralizedPareto
from .geometric import Geometric
from .gumbel import Gumbel
from .half_cauchy import HalfCauchy
from .half_normal import HalfNormal
from .independent import Independent
from .inverse_gamma import InverseGamma
from .kl import kl_divergence, register_kl
from .kumaraswamy import Kumaraswamy
from .laplace import Laplace
from .lkj_cholesky import LKJCholesky
from .log_normal import LogNormal
from .logistic_normal import LogisticNormal
from .mixture_same_family import MixtureSameFamily
from .multinomial import Multinomial
from .multivariate_normal import LowRankMultivariateNormal, MultivariateNormal
from .negative_binomial import NegativeBinomial
from .normal import Normal
from .one_hot_categorical import OneHotCategorical, OneHotCategoricalStraightThrough
from .pareto import Pareto
from .poisson import Poisson
from .relaxed_bernoulli import RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical
from .studentT import StudentT
from .transformed_distribution import TransformedDistribution
from .transforms import (
    AbsTransform,
    AffineTransform,
    CatTransform,
    ComposeTransform,
    CorrCholeskyTransform,
    CumulativeDistributionTransform,
    ExpTransform,
    IndependentTransform,
    LowerCholeskyTransform,
    PositiveDefiniteTransform,
    PowerTransform,
    ReshapeTransform,
    SigmoidTransform,
    SoftmaxTransform,
    SoftplusTransform,
    StackTransform,
    StickBreakingTransform,
    TanhTransform,
    Transform,
)
from .uniform import Uniform
from .von_mises import VonMises
from .weibull import Weibull
from .wishart import Wishart

# Identity transform (empty ComposeTransform like PyTorch)
identity_transform = ComposeTransform([])

from .constraints import (
    Constraint,
    boolean,
    cat,
    corr_cholesky,
    dependent,
    dependent_property,
    greater_than,
    greater_than_eq,
    half_open_interval,
    independent,
    integer_interval,
    interval,
    is_dependent,
    less_than,
    lower_cholesky,
    lower_triangular,
    multinomial,
    nonnegative,
    nonnegative_integer,
    one_hot,
    positive,
    positive_definite,
    positive_integer,
    positive_semidefinite,
    real,
    real_vector,
    simplex,
    square,
    stack,
    symmetric,
    unit_interval,
)

__all__ = [
    # Base classes
    "Distribution",
    "ExponentialFamily",
    # Continuous distributions
    "Normal",
    "Uniform",
    "Beta",
    "Gamma",
    "Exponential",
    "Laplace",
    "Cauchy",
    "Chi2",
    "StudentT",
    "LogNormal",
    "Gumbel",
    "Pareto",
    "Weibull",
    "HalfNormal",
    "HalfCauchy",
    "VonMises",
    "ContinuousBernoulli",
    "LogisticNormal",
    "InverseGamma",
    "Kumaraswamy",
    "GeneralizedPareto",
    "FisherSnedecor",
    # Discrete distributions
    "Bernoulli",
    "Categorical",
    "Poisson",
    "Binomial",
    "Multinomial",
    "Geometric",
    "NegativeBinomial",
    # Multivariate distributions
    "Dirichlet",
    "MultivariateNormal",
    "LowRankMultivariateNormal",
    "Wishart",
    "LKJCholesky",
    # One-hot distributions
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    # Relaxed distributions
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    # Wrapper distributions
    "Independent",
    "MixtureSameFamily",
    "TransformedDistribution",
    # Transforms
    "Transform",
    "ComposeTransform",
    "ExpTransform",
    "PowerTransform",
    "SigmoidTransform",
    "TanhTransform",
    "AbsTransform",
    "AffineTransform",
    "SoftmaxTransform",
    "SoftplusTransform",
    "LowerCholeskyTransform",
    "PositiveDefiniteTransform",
    "CorrCholeskyTransform",
    "StickBreakingTransform",
    "CatTransform",
    "StackTransform",
    "IndependentTransform",
    "ReshapeTransform",
    "CumulativeDistributionTransform",
    # KL divergence
    "kl_divergence",
    "register_kl",
    # Identity transform
    "identity_transform",
    # Constraint registry
    "biject_to",
    "transform_to",
    # Constraints
    "Constraint",
    "boolean",
    "cat",
    "corr_cholesky",
    "dependent",
    "dependent_property",
    "greater_than",
    "greater_than_eq",
    "half_open_interval",
    "independent",
    "integer_interval",
    "interval",
    "is_dependent",
    "less_than",
    "lower_cholesky",
    "lower_triangular",
    "multinomial",
    "nonnegative",
    "nonnegative_integer",
    "one_hot",
    "positive",
    "positive_definite",
    "positive_integer",
    "positive_semidefinite",
    "real",
    "real_vector",
    "simplex",
    "square",
    "stack",
    "symmetric",
    "unit_interval",
]
