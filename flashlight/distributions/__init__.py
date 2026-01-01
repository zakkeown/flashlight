"""
Distributions Module

PyTorch-compatible torch.distributions module for MLX.
Provides probability distributions for sampling and log probability computation.
"""

from .distribution import Distribution
from .exp_family import ExponentialFamily
from .normal import Normal
from .bernoulli import Bernoulli
from .categorical import Categorical
from .uniform import Uniform
from .beta import Beta
from .gamma import Gamma
from .exponential import Exponential
from .poisson import Poisson
from .binomial import Binomial
from .multinomial import Multinomial
from .dirichlet import Dirichlet
from .laplace import Laplace
from .cauchy import Cauchy
from .chi2 import Chi2
from .studentT import StudentT
from .log_normal import LogNormal
from .gumbel import Gumbel
from .pareto import Pareto
from .weibull import Weibull
from .geometric import Geometric
from .negative_binomial import NegativeBinomial
from .half_normal import HalfNormal
from .half_cauchy import HalfCauchy
from .multivariate_normal import MultivariateNormal, LowRankMultivariateNormal
from .one_hot_categorical import OneHotCategorical, OneHotCategoricalStraightThrough
from .von_mises import VonMises
from .wishart import Wishart
from .relaxed_bernoulli import RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical
from .continuous_bernoulli import ContinuousBernoulli
from .logistic_normal import LogisticNormal
from .inverse_gamma import InverseGamma
from .kumaraswamy import Kumaraswamy
from .lkj_cholesky import LKJCholesky
from .generalized_pareto import GeneralizedPareto
from .fisher_snedecor import FisherSnedecor
from .independent import Independent
from .mixture_same_family import MixtureSameFamily
from .transformed_distribution import TransformedDistribution

from .transforms import (
    Transform,
    ComposeTransform,
    ExpTransform,
    PowerTransform,
    SigmoidTransform,
    TanhTransform,
    AbsTransform,
    AffineTransform,
    SoftmaxTransform,
    SoftplusTransform,
    LowerCholeskyTransform,
    PositiveDefiniteTransform,
    CorrCholeskyTransform,
    StickBreakingTransform,
    CatTransform,
    StackTransform,
    IndependentTransform,
    ReshapeTransform,
    CumulativeDistributionTransform,
)

from .kl import kl_divergence, register_kl

from .constraint_registry import biject_to, transform_to

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
    'Distribution',
    'ExponentialFamily',
    # Continuous distributions
    'Normal',
    'Uniform',
    'Beta',
    'Gamma',
    'Exponential',
    'Laplace',
    'Cauchy',
    'Chi2',
    'StudentT',
    'LogNormal',
    'Gumbel',
    'Pareto',
    'Weibull',
    'HalfNormal',
    'HalfCauchy',
    'VonMises',
    'ContinuousBernoulli',
    'LogisticNormal',
    'InverseGamma',
    'Kumaraswamy',
    'GeneralizedPareto',
    'FisherSnedecor',
    # Discrete distributions
    'Bernoulli',
    'Categorical',
    'Poisson',
    'Binomial',
    'Multinomial',
    'Geometric',
    'NegativeBinomial',
    # Multivariate distributions
    'Dirichlet',
    'MultivariateNormal',
    'LowRankMultivariateNormal',
    'Wishart',
    'LKJCholesky',
    # One-hot distributions
    'OneHotCategorical',
    'OneHotCategoricalStraightThrough',
    # Relaxed distributions
    'RelaxedBernoulli',
    'RelaxedOneHotCategorical',
    # Wrapper distributions
    'Independent',
    'MixtureSameFamily',
    'TransformedDistribution',
    # Transforms
    'Transform',
    'ComposeTransform',
    'ExpTransform',
    'PowerTransform',
    'SigmoidTransform',
    'TanhTransform',
    'AbsTransform',
    'AffineTransform',
    'SoftmaxTransform',
    'SoftplusTransform',
    'LowerCholeskyTransform',
    'PositiveDefiniteTransform',
    'CorrCholeskyTransform',
    'StickBreakingTransform',
    'CatTransform',
    'StackTransform',
    'IndependentTransform',
    'ReshapeTransform',
    'CumulativeDistributionTransform',
    # KL divergence
    'kl_divergence',
    'register_kl',
    # Identity transform
    'identity_transform',
    # Constraint registry
    'biject_to',
    'transform_to',
    # Constraints
    'Constraint',
    'boolean',
    'cat',
    'corr_cholesky',
    'dependent',
    'dependent_property',
    'greater_than',
    'greater_than_eq',
    'half_open_interval',
    'independent',
    'integer_interval',
    'interval',
    'is_dependent',
    'less_than',
    'lower_cholesky',
    'lower_triangular',
    'multinomial',
    'nonnegative',
    'nonnegative_integer',
    'one_hot',
    'positive',
    'positive_definite',
    'positive_integer',
    'positive_semidefinite',
    'real',
    'real_vector',
    'simplex',
    'square',
    'stack',
    'symmetric',
    'unit_interval',
]
