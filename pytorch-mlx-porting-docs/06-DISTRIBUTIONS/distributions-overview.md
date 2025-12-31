# Probability Distributions

## Overview

PyTorch provides a comprehensive library of probability distributions in `torch.distributions`. These are essential for probabilistic models, variational inference, reinforcement learning, and generative models. The library supports sampling, log probability computation, entropy, and KL divergence.

**Reference Files:**
- `torch/distributions/distribution.py` - Base Distribution class
- `torch/distributions/normal.py` - Example implementation
- `torch/distributions/kl.py` - KL divergence registry
- `torch/distributions/transforms.py` - Bijective transforms
- `torch/distributions/constraints.py` - Parameter constraints

## Distribution Hierarchy

```
Distribution (Abstract Base)
├── ExponentialFamily (Natural exponential family)
│   ├── Normal, Gamma, Chi2, Beta, Dirichlet
│   ├── Exponential, Bernoulli, Binomial
│   └── Poisson, Categorical, Multinomial
├── Continuous Distributions
│   ├── Normal, Uniform, Laplace, Cauchy
│   ├── Beta, Gamma, Exponential, Weibull
│   ├── LogNormal, Gumbel, StudentT
│   └── MultivariateNormal, Wishart
├── Discrete Distributions
│   ├── Bernoulli, Binomial, Categorical
│   ├── Poisson, Geometric, NegativeBinomial
│   └── Multinomial, OneHotCategorical
├── Relaxed Distributions (for gradient estimation)
│   ├── RelaxedBernoulli, RelaxedOneHotCategorical
│   └── (Gumbel-Softmax trick)
└── Transformed Distributions
    └── TransformedDistribution + Transforms
```

---

## Base Distribution Class

All distributions inherit from `Distribution` and implement a common interface.

### Core Properties

```python
class Distribution:
    # Shape properties
    batch_shape: torch.Size  # Shape over which parameters are batched
    event_shape: torch.Size  # Shape of a single sample

    # Constraints
    arg_constraints: dict[str, Constraint]  # Parameter constraints
    support: Constraint  # Valid values for samples

    # Statistics
    mean: Tensor      # Expected value
    variance: Tensor  # Variance
    stddev: Tensor    # Standard deviation (sqrt(variance))
    mode: Tensor      # Most likely value
```

### Core Methods

```python
class Distribution:
    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """Non-differentiable sampling (uses no_grad internally)."""

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        """Reparameterized sampling (differentiable through sample)."""

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability density/mass at value."""

    def entropy(self) -> Tensor:
        """Entropy of the distribution."""

    def cdf(self, value: Tensor) -> Tensor:
        """Cumulative distribution function."""

    def icdf(self, value: Tensor) -> Tensor:
        """Inverse CDF (quantile function)."""

    def expand(self, batch_shape) -> Distribution:
        """Expand batch dimensions."""

    def enumerate_support(self, expand=True) -> Tensor:
        """Enumerate all values in support (discrete only)."""
```

### Shape Semantics

```
sample_shape + batch_shape + event_shape = output_shape

sample_shape: Number of independent samples
batch_shape: Parameter broadcasting (independent distributions)
event_shape: Dimensionality of each sample
```

Example:

```python
# Scalar normal: batch_shape=(), event_shape=()
d = Normal(loc=0., scale=1.)
d.sample((100,))  # Shape: (100,)

# Batched normal: batch_shape=(3,), event_shape=()
d = Normal(loc=torch.zeros(3), scale=torch.ones(3))
d.sample((100,))  # Shape: (100, 3)

# Multivariate normal: batch_shape=(), event_shape=(5,)
d = MultivariateNormal(loc=torch.zeros(5), covariance_matrix=torch.eye(5))
d.sample((100,))  # Shape: (100, 5)
```

---

## Common Distributions

### Normal (Gaussian)

```python
from torch.distributions import Normal

# Create distribution
d = Normal(loc=0., scale=1.)  # μ=0, σ=1

# Batched creation
d = Normal(
    loc=torch.tensor([0., 1., 2.]),
    scale=torch.tensor([1., 0.5, 2.])
)

# Operations
samples = d.sample((1000,))       # Non-differentiable
samples = d.rsample((1000,))      # Differentiable (for VAEs, etc.)
log_p = d.log_prob(samples)       # Log probability
entropy = d.entropy()             # H = 0.5 * log(2πeσ²)
```

### Categorical

```python
from torch.distributions import Categorical

# From probabilities (must sum to 1)
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
d = Categorical(probs=probs)

# From logits (unnormalized log probabilities)
logits = torch.tensor([1., 2., 3., 4.])
d = Categorical(logits=logits)

# Sample returns indices
samples = d.sample((100,))  # Values in {0, 1, 2, 3}
log_p = d.log_prob(samples)
```

### Bernoulli

```python
from torch.distributions import Bernoulli

# Probability of 1
d = Bernoulli(probs=0.7)
# Or from logits
d = Bernoulli(logits=0.8)  # logit = log(p / (1-p))

samples = d.sample((100,))  # 0 or 1
log_p = d.log_prob(torch.tensor([0., 1., 1., 0.]))
```

### MultivariateNormal

```python
from torch.distributions import MultivariateNormal

# Full covariance matrix
mean = torch.zeros(3)
cov = torch.eye(3)
d = MultivariateNormal(loc=mean, covariance_matrix=cov)

# Precision matrix (inverse covariance)
d = MultivariateNormal(loc=mean, precision_matrix=torch.eye(3))

# Scale-tril (Cholesky factor)
d = MultivariateNormal(loc=mean, scale_tril=torch.eye(3))

samples = d.sample((100,))  # Shape: (100, 3)
log_p = d.log_prob(samples)  # Shape: (100,) - one value per event
```

### Beta

```python
from torch.distributions import Beta

# Parameters: concentration1 (α), concentration0 (β)
d = Beta(concentration1=2., concentration0=5.)  # α=2, β=5

# Support: (0, 1)
samples = d.sample((100,))  # Values in (0, 1)
log_p = d.log_prob(samples)

# Mean = α / (α + β)
print(d.mean)  # ≈ 0.286
```

### Gamma

```python
from torch.distributions import Gamma

# concentration (α, shape), rate (β, inverse scale)
d = Gamma(concentration=2., rate=0.5)

# Support: (0, ∞)
samples = d.sample((100,))
log_p = d.log_prob(samples)

# Mean = α / β
print(d.mean)  # 4.0
```

---

## Reparameterization (rsample)

### Why Reparameterization Matters

Standard sampling creates a stochastic node that blocks gradients:

```python
# Gradient cannot flow through sample()
z = dist.sample()
loss = f(z)
loss.backward()  # No gradient to distribution parameters
```

Reparameterized sampling allows gradients to flow:

```python
# Normal rsample: z = μ + σ * ε, where ε ~ N(0,1)
z = dist.rsample()  # Gradients flow to μ and σ
loss = f(z)
loss.backward()  # Works!
```

### Check Reparameterization Support

```python
dist.has_rsample  # True if rsample is differentiable
```

Distributions with `has_rsample=True`:
- Normal, LogNormal, Exponential
- Gamma, Beta, Dirichlet
- Uniform, Laplace, Cauchy
- MultivariateNormal, StudentT
- RelaxedBernoulli, RelaxedOneHotCategorical

Distributions with `has_rsample=False` (discrete):
- Bernoulli, Categorical, Multinomial
- Poisson, Binomial, Geometric

---

## Relaxed Distributions

For discrete distributions, use relaxed versions for gradient estimation.

### RelaxedBernoulli (Concrete Distribution)

```python
from torch.distributions import RelaxedBernoulli

# Temperature controls relaxation (lower = more discrete-like)
d = RelaxedBernoulli(temperature=0.5, probs=0.7)

# Samples in (0, 1) instead of {0, 1}
samples = d.rsample((100,))  # Differentiable!
```

### RelaxedOneHotCategorical (Gumbel-Softmax)

```python
from torch.distributions import RelaxedOneHotCategorical

logits = torch.tensor([1., 2., 3., 4.])
d = RelaxedOneHotCategorical(temperature=0.5, logits=logits)

# Samples are soft one-hot vectors
samples = d.rsample((100,))  # Shape: (100, 4), values sum to 1
```

### Temperature Annealing

```python
# Start with high temperature (smooth), anneal to low (discrete-like)
for epoch in range(epochs):
    temperature = max(0.5, 2.0 * (0.9 ** epoch))
    d = RelaxedOneHotCategorical(temperature=temperature, logits=logits)
    samples = d.rsample()
```

---

## KL Divergence

### kl_divergence()

```python
from torch.distributions import kl_divergence, Normal

p = Normal(loc=0., scale=1.)
q = Normal(loc=1., scale=2.)

kl = kl_divergence(p, q)  # KL(p || q)
```

### Registered KL Pairs

PyTorch maintains a registry of closed-form KL divergences:

```python
# Automatic dispatch based on distribution types
kl_divergence(Normal(...), Normal(...))  # Uses closed form
kl_divergence(Bernoulli(...), Bernoulli(...))  # Uses closed form
```

### Custom KL Registration

```python
from torch.distributions import register_kl

@register_kl(MyDistribution, Normal)
def kl_my_normal(p, q):
    # Return KL(p || q)
    return ...
```

### Monte Carlo KL

For distributions without closed-form KL:

```python
def mc_kl_divergence(p, q, num_samples=1000):
    samples = p.rsample((num_samples,))
    return (p.log_prob(samples) - q.log_prob(samples)).mean()
```

---

## Transforms and TransformedDistribution

### Bijective Transforms

```python
from torch.distributions import transforms

# Available transforms
transforms.ExpTransform()           # exp(x)
transforms.SigmoidTransform()       # 1 / (1 + exp(-x))
transforms.AffineTransform(loc, scale)  # loc + scale * x
transforms.SoftmaxTransform()       # softmax
transforms.StickBreakingTransform() # for simplex
transforms.TanhTransform()          # tanh(x)

# Compose transforms
t = transforms.ComposeTransform([
    transforms.AffineTransform(0, 2),
    transforms.ExpTransform()
])
```

### TransformedDistribution

Create new distributions by transforming existing ones:

```python
from torch.distributions import TransformedDistribution, Normal
from torch.distributions.transforms import ExpTransform

# LogNormal = exp(Normal)
base_dist = Normal(loc=0., scale=1.)
log_normal = TransformedDistribution(base_dist, ExpTransform())

# Equivalent to:
from torch.distributions import LogNormal
log_normal = LogNormal(loc=0., scale=1.)
```

### Normalizing Flows (Multiple Transforms)

```python
base_dist = Normal(torch.zeros(10), torch.ones(10))
transforms_list = [
    transforms.AffineTransform(loc=torch.randn(10), scale=torch.rand(10)),
    transforms.TanhTransform(),
    transforms.AffineTransform(loc=torch.randn(10), scale=torch.rand(10)),
]
flow = TransformedDistribution(base_dist, transforms_list)

# Sample and compute log prob
samples = flow.rsample((100,))
log_prob = flow.log_prob(samples)  # Accounts for Jacobian determinants
```

---

## Constraints

### Parameter Constraints

```python
from torch.distributions import constraints

# Common constraints
constraints.real                  # (-∞, +∞)
constraints.positive              # (0, +∞)
constraints.unit_interval         # [0, 1]
constraints.simplex              # Sums to 1, all positive
constraints.lower_cholesky       # Lower triangular, positive diagonal
constraints.positive_definite    # Symmetric positive definite
```

### Constraint Validation

```python
d = Normal(loc=0., scale=-1., validate_args=True)
# Raises ValueError: scale must be positive

# Disable validation for speed
Distribution.set_default_validate_args(False)
```

### Transform Between Constraint Spaces

```python
from torch.distributions.transforms import biject_to

# Get transform from unconstrained to constrained space
transform = biject_to(constraints.positive)

# Transform: R -> R+
unconstrained = torch.randn(5)
positive = transform(unconstrained)  # All positive
```

---

## Independent Distribution

Reinterpret batch dimensions as event dimensions:

```python
from torch.distributions import Normal, Independent

# Base: batch_shape=(10,), event_shape=()
base = Normal(torch.zeros(10), torch.ones(10))
base.log_prob(torch.zeros(10)).shape  # (10,) - per-dimension log probs

# Independent: batch_shape=(), event_shape=(10,)
ind = Independent(base, reinterpreted_batch_ndims=1)
ind.log_prob(torch.zeros(10)).shape  # () - single scalar (sum of log probs)
```

---

## Mixture Distributions

### MixtureSameFamily

```python
from torch.distributions import MixtureSameFamily, Categorical, Normal

# Mixture of 3 Gaussians
mix = Categorical(torch.ones(3) / 3)  # Uniform mixture weights
comp = Normal(
    loc=torch.tensor([-1., 0., 1.]),
    scale=torch.tensor([0.5, 0.5, 0.5])
)
gmm = MixtureSameFamily(mix, comp)

samples = gmm.sample((1000,))
log_prob = gmm.log_prob(samples)
```

---

## Practical Examples

### Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder()  # Returns (mu, log_var)
        self.decoder = Decoder()
        self.latent_dim = latent_dim

    def forward(self, x):
        # Encode
        mu, log_var = self.encoder(x)
        std = (0.5 * log_var).exp()

        # Reparameterized sampling
        q_z = Normal(mu, std)
        z = q_z.rsample()  # Differentiable!

        # Decode
        x_recon = self.decoder(z)

        # Losses
        recon_loss = -Normal(x_recon, 1.).log_prob(x).sum(dim=-1)

        # KL divergence to prior
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl_loss = kl_divergence(q_z, p_z).sum(dim=-1)

        return recon_loss.mean() + kl_loss.mean()
```

### Policy Gradient (Reinforce)

```python
def reinforce_loss(policy_net, states, actions, returns):
    # Get action distribution from policy
    logits = policy_net(states)
    action_dist = Categorical(logits=logits)

    # Log probability of taken actions
    log_probs = action_dist.log_prob(actions)

    # Policy gradient loss
    loss = -(log_probs * returns).mean()

    # Add entropy bonus for exploration
    entropy = action_dist.entropy().mean()
    loss = loss - 0.01 * entropy

    return loss
```

### Bayesian Neural Network

```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Weight distribution parameters
        self.w_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.w_log_var = nn.Parameter(torch.zeros(out_features, in_features) - 3)

        # Bias distribution parameters
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_log_var = nn.Parameter(torch.zeros(out_features) - 3)

    def forward(self, x):
        # Sample weights
        w_std = (0.5 * self.w_log_var).exp()
        w_dist = Normal(self.w_mu, w_std)
        w = w_dist.rsample()

        # Sample bias
        b_std = (0.5 * self.b_log_var).exp()
        b_dist = Normal(self.b_mu, b_std)
        b = b_dist.rsample()

        return F.linear(x, w, b)

    def kl_divergence(self):
        # KL to prior N(0, 1)
        prior = Normal(0., 1.)

        w_std = (0.5 * self.w_log_var).exp()
        w_post = Normal(self.w_mu, w_std)
        kl_w = kl_divergence(w_post, prior).sum()

        b_std = (0.5 * self.b_log_var).exp()
        b_post = Normal(self.b_mu, b_std)
        kl_b = kl_divergence(b_post, prior).sum()

        return kl_w + kl_b
```

---

## MLX Mapping

### Core Differences

MLX has fewer built-in distributions. Common approach:

```python
import mlx.core as mx
import mlx.nn as nn

# Normal distribution using primitives
def normal_sample(mean, std, shape, key):
    eps = mx.random.normal(shape, key=key)
    return mean + std * eps

def normal_log_prob(x, mean, std):
    var = std ** 2
    return -0.5 * ((x - mean) ** 2 / var + mx.log(2 * mx.pi * var))

# Categorical using Gumbel trick
def categorical_sample(logits, key):
    gumbel = -mx.log(-mx.log(mx.random.uniform(logits.shape, key=key)))
    return mx.argmax(logits + gumbel, axis=-1)
```

### Implementing Distribution Classes

```python
class Normal:
    def __init__(self, loc, scale):
        self.loc = mx.array(loc)
        self.scale = mx.array(scale)

    def sample(self, shape=(), key=None):
        eps = mx.random.normal(shape + self.loc.shape, key=key)
        return self.loc + self.scale * eps

    def log_prob(self, value):
        var = self.scale ** 2
        return -0.5 * ((value - self.loc) ** 2 / var + mx.log(2 * mx.pi * var))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2
```

---

## Complete Distribution List

### Continuous

| Distribution | Parameters | Support |
|--------------|------------|---------|
| Normal | loc, scale | R |
| LogNormal | loc, scale | R+ |
| Uniform | low, high | [low, high] |
| Exponential | rate | R+ |
| Gamma | concentration, rate | R+ |
| Beta | concentration1, concentration0 | (0, 1) |
| Cauchy | loc, scale | R |
| Laplace | loc, scale | R |
| StudentT | df, loc, scale | R |
| Gumbel | loc, scale | R |
| Weibull | scale, concentration | R+ |

### Discrete

| Distribution | Parameters | Support |
|--------------|------------|---------|
| Bernoulli | probs/logits | {0, 1} |
| Binomial | total_count, probs/logits | {0, ..., n} |
| Categorical | probs/logits | {0, ..., K-1} |
| Multinomial | total_count, probs/logits | Counts summing to n |
| Poisson | rate | N ∪ {0} |
| Geometric | probs/logits | N |
| NegativeBinomial | total_count, probs/logits | N ∪ {0} |

### Multivariate

| Distribution | Parameters | Support |
|--------------|------------|---------|
| MultivariateNormal | loc, covariance/precision/scale_tril | R^d |
| Dirichlet | concentration | Simplex |
| Wishart | df, covariance/precision/scale_tril | Positive definite |

---

## Detailed Distribution Reference

### Continuous Distributions

#### Normal (Gaussian)

**PDF:** $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

```python
Normal(loc=μ, scale=σ)
# loc: mean (μ)
# scale: standard deviation (σ > 0)
# Support: ℝ
# has_rsample: True
```

**Statistics:**
- Mean: μ
- Variance: σ²
- Entropy: $\frac{1}{2}\log(2\pi e\sigma^2)$

**Reparameterization:** `z = μ + σ * ε`, where `ε ~ N(0,1)`

---

#### LogNormal

**PDF:** $p(x) = \frac{1}{x\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\log x - \mu)^2}{2\sigma^2}\right)$

```python
LogNormal(loc=μ, scale=σ)
# Parameters are for the underlying normal
# Support: ℝ⁺
# has_rsample: True
```

**Statistics:**
- Mean: $\exp(\mu + \sigma^2/2)$
- Variance: $(\exp(\sigma^2) - 1) \exp(2\mu + \sigma^2)$

---

#### Uniform

**PDF:** $p(x) = \frac{1}{b - a}$ for $x \in [a, b]$

```python
Uniform(low=a, high=b)
# Support: [low, high]
# has_rsample: True
```

**Statistics:**
- Mean: $(a + b) / 2$
- Variance: $(b - a)^2 / 12$

**Reparameterization:** `z = low + (high - low) * u`, where `u ~ U(0,1)`

---

#### Exponential

**PDF:** $p(x) = \lambda \exp(-\lambda x)$

```python
Exponential(rate=λ)
# rate: inverse scale (λ > 0)
# Support: ℝ⁺
# has_rsample: True
```

**Statistics:**
- Mean: $1/\lambda$
- Variance: $1/\lambda^2$

**Reparameterization:** `z = -log(u) / λ`, where `u ~ U(0,1)`

---

#### Gamma

**PDF:** $p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} \exp(-\beta x)$

```python
Gamma(concentration=α, rate=β)
# concentration: shape (α > 0)
# rate: inverse scale (β > 0)
# Support: ℝ⁺
# has_rsample: True
```

**Statistics:**
- Mean: $\alpha / \beta$
- Variance: $\alpha / \beta^2$

**Implementation:** Uses rejection sampling with reparameterization gradient.

---

#### Beta

**PDF:** $p(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$

```python
Beta(concentration1=α, concentration0=β)
# concentration1: α > 0
# concentration0: β > 0
# Support: (0, 1)
# has_rsample: True
```

**Statistics:**
- Mean: $\alpha / (\alpha + \beta)$
- Variance: $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

**Implementation:** Derived from two Gamma distributions: $X \sim \text{Gamma}(\alpha, 1)$, $Y \sim \text{Gamma}(\beta, 1)$, then $X/(X+Y) \sim \text{Beta}(\alpha, \beta)$.

---

#### Chi2 (Chi-Squared)

**PDF:** $p(x) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} \exp(-x/2)$

```python
Chi2(df=k)
# df: degrees of freedom (k > 0)
# Support: ℝ⁺
# has_rsample: True (inherits from Gamma)
```

**Statistics:**
- Mean: k
- Variance: 2k

**Implementation:** `Chi2(df) = Gamma(concentration=df/2, rate=0.5)`

---

#### StudentT (Student's t)

**PDF:** $p(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})} \left(1 + \frac{(x-\mu)^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$

```python
StudentT(df=ν, loc=μ, scale=σ)
# df: degrees of freedom (ν > 0)
# loc: location (μ)
# scale: scale (σ > 0)
# Support: ℝ
# has_rsample: True
```

**Statistics:**
- Mean: μ (if ν > 1)
- Variance: $\sigma^2 \nu / (\nu - 2)$ (if ν > 2)

---

#### Cauchy

**PDF:** $p(x) = \frac{1}{\pi\gamma\left(1 + \left(\frac{x-x_0}{\gamma}\right)^2\right)}$

```python
Cauchy(loc=x₀, scale=γ)
# loc: location/median (x₀)
# scale: half-width at half-maximum (γ > 0)
# Support: ℝ
# has_rsample: True
```

**Note:** Mean and variance are undefined.

---

#### Laplace

**PDF:** $p(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)$

```python
Laplace(loc=μ, scale=b)
# loc: location/mean (μ)
# scale: diversity (b > 0)
# Support: ℝ
# has_rsample: True
```

**Statistics:**
- Mean: μ
- Variance: 2b²

---

#### Gumbel

**PDF:** $p(x) = \frac{1}{\beta} \exp\left(-z - \exp(-z)\right)$, where $z = (x - \mu)/\beta$

```python
Gumbel(loc=μ, scale=β)
# loc: location (μ)
# scale: scale (β > 0)
# Support: ℝ
# has_rsample: True
```

**Statistics:**
- Mean: $\mu + \beta \gamma$ (γ = Euler-Mascheroni constant ≈ 0.5772)
- Variance: $\frac{\pi^2 \beta^2}{6}$

---

#### Weibull

**PDF:** $p(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} \exp\left(-\left(\frac{x}{\lambda}\right)^k\right)$

```python
Weibull(scale=λ, concentration=k)
# scale: λ > 0
# concentration: shape (k > 0)
# Support: ℝ⁺
# has_rsample: True
```

**Statistics:**
- Mean: $\lambda \Gamma(1 + 1/k)$

---

#### Pareto

**PDF:** $p(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}$ for $x \geq x_m$

```python
Pareto(scale=x_m, alpha=α)
# scale: minimum value (x_m > 0)
# alpha: shape (α > 0)
# Support: [scale, ∞)
# has_rsample: True
```

**Statistics:**
- Mean: $\frac{\alpha x_m}{\alpha - 1}$ (if α > 1)

---

#### VonMises (Circular Distribution)

**PDF:** $p(\theta) = \frac{\exp(\kappa \cos(\theta - \mu))}{2\pi I_0(\kappa)}$

```python
VonMises(loc=μ, concentration=κ)
# loc: mean direction (μ in radians)
# concentration: κ ≥ 0 (higher = more concentrated)
# Support: (-π, π]
# has_rsample: False (uses rejection sampling)
```

**Statistics:**
- Circular mean: μ
- Circular variance: $1 - I_1(\kappa)/I_0(\kappa)$

**Implementation:** Uses Best-Fisher rejection sampling algorithm.

---

#### Kumaraswamy

**PDF:** $p(x) = ab x^{a-1}(1-x^a)^{b-1}$

```python
Kumaraswamy(concentration1=a, concentration0=b)
# concentration1: a > 0
# concentration0: b > 0
# Support: (0, 1)
# has_rsample: True
```

Similar to Beta but with closed-form CDF: $F(x) = 1 - (1-x^a)^b$

**Implementation:** Uses inverse CDF via transforms:
```python
# Sample: u ~ U(0,1), then x = (1 - (1-u)^(1/b))^(1/a)
```

---

#### HalfNormal

**PDF:** $p(x) = \sqrt{\frac{2}{\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)$

```python
HalfNormal(scale=σ)
# scale: σ > 0
# Support: ℝ⁺
# has_rsample: True
```

**Implementation:** `|X|` where `X ~ Normal(0, σ)`

---

#### HalfCauchy

```python
HalfCauchy(scale=γ)
# scale: γ > 0
# Support: ℝ⁺
# has_rsample: True
```

**Implementation:** `|X|` where `X ~ Cauchy(0, γ)`

---

#### InverseGamma

**PDF:** $p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha-1} \exp(-\beta/x)$

```python
InverseGamma(concentration=α, rate=β)
# concentration: shape (α > 0)
# rate: scale (β > 0)
# Support: ℝ⁺
# has_rsample: True
```

**Implementation:** `1/X` where `X ~ Gamma(α, β)`

---

#### FisherSnedecor (F-distribution)

```python
FisherSnedecor(df1=d₁, df2=d₂)
# df1: numerator degrees of freedom
# df2: denominator degrees of freedom
# Support: ℝ⁺
# has_rsample: True
```

---

#### GeneralizedPareto

```python
GeneralizedPareto(loc=μ, scale=σ, concentration=ξ)
# Support: Depends on ξ
# has_rsample: True
```

Used in extreme value theory.

---

### Discrete Distributions

#### Bernoulli

**PMF:** $p(x) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$

```python
Bernoulli(probs=p)
Bernoulli(logits=l)  # p = sigmoid(l)
# Support: {0, 1}
# has_rsample: False
```

**Statistics:**
- Mean: p
- Variance: p(1-p)

---

#### Binomial

**PMF:** $p(k) = \binom{n}{k} p^k (1-p)^{n-k}$

```python
Binomial(total_count=n, probs=p)
Binomial(total_count=n, logits=l)
# Support: {0, 1, ..., n}
# has_rsample: False
```

**Statistics:**
- Mean: np
- Variance: np(1-p)

---

#### Categorical

**PMF:** $p(x=i) = p_i$

```python
Categorical(probs=p)  # p sums to 1
Categorical(logits=l)  # p = softmax(l)
# Support: {0, 1, ..., K-1}
# has_rsample: False
```

**Implementation:** Uses Gumbel-max trick for sampling.

---

#### Multinomial

```python
Multinomial(total_count=n, probs=p)
Multinomial(total_count=n, logits=l)
# Support: Counts summing to n
# has_rsample: False
```

---

#### Poisson

**PMF:** $p(k) = \frac{\lambda^k e^{-\lambda}}{k!}$

```python
Poisson(rate=λ)
# rate: λ > 0
# Support: ℕ ∪ {0}
# has_rsample: False
```

**Statistics:**
- Mean: λ
- Variance: λ

---

#### Geometric

**PMF:** $p(k) = p(1-p)^{k-1}$ for $k \geq 1$

```python
Geometric(probs=p)
Geometric(logits=l)
# Support: {1, 2, 3, ...}
# has_rsample: False
```

**Statistics:**
- Mean: 1/p
- Variance: (1-p)/p²

---

#### NegativeBinomial

**PMF:** $p(k) = \binom{k+r-1}{k} p^r (1-p)^k$

```python
NegativeBinomial(total_count=r, probs=p)
NegativeBinomial(total_count=r, logits=l)
# Support: ℕ ∪ {0}
# has_rsample: False
```

---

### Multivariate Distributions

#### MultivariateNormal

**PDF:** $p(x) = \frac{1}{\sqrt{(2\pi)^k|\Sigma|}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$

```python
MultivariateNormal(loc=μ, covariance_matrix=Σ)
MultivariateNormal(loc=μ, precision_matrix=Λ)  # Λ = Σ⁻¹
MultivariateNormal(loc=μ, scale_tril=L)  # Σ = LL^T
# Support: ℝᵈ
# has_rsample: True
```

**Reparameterization:** `z = μ + L @ ε`, where `ε ~ N(0, I)`

---

#### LowRankMultivariateNormal

Memory-efficient parameterization: $\Sigma = W W^T + D$ where W is low-rank and D is diagonal.

```python
LowRankMultivariateNormal(loc=μ, cov_factor=W, cov_diag=D)
# Support: ℝᵈ
# has_rsample: True
```

---

#### Dirichlet

**PDF:** $p(x) = \frac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)} \prod_i x_i^{\alpha_i - 1}$

```python
Dirichlet(concentration=α)
# concentration: α > 0 (vector)
# Support: Simplex (x_i > 0, sum to 1)
# has_rsample: True
```

**Statistics:**
- Mean: $\alpha_i / \sum_j \alpha_j$

**Implementation:** Normalized Gamma samples: $X_i \sim \text{Gamma}(\alpha_i, 1)$, then $x_i = X_i / \sum_j X_j$.

---

#### Wishart

Distribution over positive-definite matrices.

```python
Wishart(df=ν, covariance_matrix=V)
Wishart(df=ν, precision_matrix=Λ)
Wishart(df=ν, scale_tril=L)
# df: degrees of freedom (ν ≥ p)
# Support: p×p positive definite matrices
# has_rsample: True
```

---

#### LKJCholesky

Distribution over Cholesky factors of correlation matrices.

```python
LKJCholesky(dim=d, concentration=η)
# dim: matrix dimension
# concentration: η > 0 (η=1 gives uniform)
# Support: d×d lower-triangular with unit diagonal
# has_rsample: False
```

Used in Bayesian correlation matrix inference.

---

### Relaxed (Continuous) Distributions

#### RelaxedBernoulli (Concrete/BinConcrete)

Continuous relaxation of Bernoulli using logistic distribution.

```python
RelaxedBernoulli(temperature=τ, probs=p)
RelaxedBernoulli(temperature=τ, logits=l)
# temperature: τ > 0 (lower = more discrete)
# Support: (0, 1)
# has_rsample: True
```

**Reparameterization:** $x = \sigma((l + \text{Logistic}(0,1))/\tau)$

---

#### RelaxedOneHotCategorical (Gumbel-Softmax)

Continuous relaxation of Categorical.

```python
RelaxedOneHotCategorical(temperature=τ, probs=p)
RelaxedOneHotCategorical(temperature=τ, logits=l)
# temperature: τ > 0
# Support: Interior of simplex
# has_rsample: True
```

**Reparameterization:** $x_i = \frac{\exp((l_i + g_i)/\tau)}{\sum_j \exp((l_j + g_j)/\tau)}$, where $g_i \sim \text{Gumbel}(0,1)$

---

#### ContinuousBernoulli

A proper exponential family distribution on (0,1).

```python
ContinuousBernoulli(probs=p)
ContinuousBernoulli(logits=l)
# Support: (0, 1)
# has_rsample: True
```

Unlike RelaxedBernoulli, this is a true distribution (not a relaxation).

---

### Utility Distributions

#### OneHotCategorical

```python
OneHotCategorical(probs=p)
OneHotCategorical(logits=l)
# Support: One-hot vectors
# has_rsample: False
```

Returns one-hot encoded samples instead of indices.

---

#### OneHotCategoricalStraightThrough

Same as OneHotCategorical but with straight-through gradient estimator for backward pass.

```python
OneHotCategoricalStraightThrough(probs=p)
# Useful for discrete VAEs
```

---

#### LogisticNormal

Logit-normal distribution on the simplex.

```python
LogisticNormal(loc=μ, scale=σ)
# Support: Simplex
# has_rsample: True
```

**Implementation:** `softmax(Normal(μ, σ).rsample())`

---

## Implementation Patterns

### Custom Distribution Template

```python
from torch.distributions import Distribution, constraints

class MyDistribution(Distribution):
    arg_constraints = {
        'param1': constraints.positive,
        'param2': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, param1, param2, validate_args=None):
        self.param1, self.param2 = broadcast_all(param1, param2)
        batch_shape = self.param1.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # Reparameterized sampling logic
        eps = torch.randn(shape, device=self.param1.device)
        return self.param1 + self.param2 * eps

    def log_prob(self, value):
        # Log probability computation
        return -0.5 * ((value - self.param1) / self.param2).pow(2) - self.param2.log()

    @property
    def mean(self):
        return self.param1

    @property
    def variance(self):
        return self.param2.pow(2)
```

### Registering Custom KL Divergence

```python
from torch.distributions import register_kl

@register_kl(MyDistribution, Normal)
def kl_my_normal(p, q):
    # KL(p || q) where p is MyDistribution, q is Normal
    var_p = p.variance
    var_q = q.variance
    return 0.5 * (
        var_p / var_q +
        (q.mean - p.mean).pow(2) / var_q -
        1 +
        var_q.log() - var_p.log()
    )
```

---

## Best Practices

1. **Use rsample for training** - Enables gradient flow through sampling
2. **Validate args during development** - Catch invalid parameters early
3. **Disable validation in production** - Improves performance
4. **Use logits not probs** - More numerically stable
5. **Prefer closed-form KL** - More accurate than Monte Carlo
6. **Temperature anneal relaxed distributions** - Start smooth, end discrete
7. **Use scale_tril for MultivariateNormal** - Most efficient parameterization
8. **Check has_rsample before differentiating through samples** - Not all distributions support it

---

## MLX Implementation Priorities

For MLX porting, prioritize distributions by usage frequency:

**High Priority:**
- Normal, Categorical, Bernoulli (core ML)
- MultivariateNormal, Dirichlet (probabilistic models)

**Medium Priority:**
- Beta, Gamma, Exponential (priors)
- Uniform, LogNormal (transforms)
- RelaxedBernoulli, RelaxedOneHotCategorical (differentiable discrete)

**Lower Priority:**
- VonMises (directional statistics)
- Wishart, LKJCholesky (correlation modeling)
- Pareto, Weibull (survival analysis)
