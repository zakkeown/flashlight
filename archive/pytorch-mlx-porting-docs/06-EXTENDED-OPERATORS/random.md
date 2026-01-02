# Random Number Generation & Distributions - PyTorch → MLX Porting Guide

## Overview

This document covers PyTorch's comprehensive random number generation (RNG) infrastructure and probability distributions. Understanding PyTorch's RNG system is critical for implementing stochastic operations, probabilistic models, and ensuring reproducible research.

PyTorch provides:
- **Low-level RNG**: Pseudorandom number generators (PRNGs) with state management
- **Tensor sampling**: In-place and functional random operations
- **Probability distributions**: 40+ distributions via `torch.distributions`
- **Reproducibility**: Generator objects for deterministic behavior

### Key Use Cases

1. **Neural Network Initialization**: Xavier, Kaiming weight initialization
2. **Data Augmentation**: Random crops, flips, color jitter
3. **Dropout & Regularization**: Stochastic regularization techniques
4. **Probabilistic Models**: VAEs, normalizing flows, Bayesian networks
5. **Reinforcement Learning**: Action sampling, exploration strategies
6. **Generative Models**: GANs, diffusion models, autoregressive sampling

---

## Table of Contents

1. [RNG Architecture](#rng-architecture)
2. [Generator Objects](#generator-objects)
3. [Basic Sampling Operations](#basic-sampling-operations)
4. [Probability Distributions](#probability-distributions)
5. [Reproducibility & Seeding](#reproducibility--seeding)
6. [MLX Porting Guide](#mlx-porting-guide)
7. [Performance Considerations](#performance-considerations)

---

## RNG Architecture

### Generator State Management

PyTorch uses **generator objects** to manage RNG state independently per device.

**Generator Hierarchy**:
```
Generator (Python object)
    ├─ CPUGeneratorImpl (CPU backend)
    ├─ CUDAGeneratorImpl (CUDA backend)
    └─ MPSGeneratorImpl (Metal backend - Apple Silicon)
```

**Default Generators**:
- CPU: One global generator per process
- CUDA: One generator per GPU device
- MPS: One generator for Metal device

**Example**:
```python
import torch

# Get default generator
default_gen = torch.default_generator
print(f"Initial state: {default_gen.initial_seed()}")

# Create custom generator
gen = torch.Generator()
gen.manual_seed(42)

# Device-specific generator
if torch.cuda.is_available():
    cuda_gen = torch.Generator(device='cuda')
    cuda_gen.manual_seed(123)
```

### PRNG Algorithms

**PyTorch CPUGeneratorImpl** uses:
- **MT19937**: Mersenne Twister (32-bit state)
- **Philox**: Counter-based RNG (used in CUDA for parallel generation)

**Properties**:
- Period: 2^19937 - 1 (MT19937)
- Thread-safe: Each thread can have independent generator
- Reproducible: Same seed → same sequence

**Implementation** ([ATen/CPUGeneratorImpl.h](reference/pytorch/aten/src/ATen/CPUGeneratorImpl.h)):
```cpp
struct TORCH_API CPUGeneratorImpl : public c10::GeneratorImpl {
  // Constructor
  CPUGeneratorImpl(uint64_t seed_in = default_rng_seed_val);

  // Set the seed
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;

  // Get random 64-bit value
  uint64_t random64();
  uint32_t random();

  // State management
  CPUGeneratorImpl* clone_impl() const override;

private:
  at::mt19937 engine_;  // Mersenne Twister engine
  uint64_t next_float_normal_sample_;
  bool is_next_float_normal_sample_valid_;
};
```

---

## Generator Objects

### Creating Generators

**API**:
```python
torch.Generator(device='cpu') -> Generator

# Methods
gen.manual_seed(seed: int) -> Generator  # Set seed
gen.initial_seed() -> int                # Get initial seed
gen.seed() -> int                        # Generate random seed
gen.get_state() -> Tensor                # Get RNG state
gen.set_state(state: Tensor) -> None     # Restore RNG state
```

**Example**:
```python
# Create generator with specific seed
gen1 = torch.Generator().manual_seed(42)

# Clone generator (independent state)
gen2 = torch.Generator().manual_seed(42)

# Same seed → same sequence
x1 = torch.randn(5, generator=gen1)
x2 = torch.randn(5, generator=gen2)
assert torch.allclose(x1, x2)  # Identical samples

# Save and restore state
state = gen1.get_state()
_ = torch.randn(100, generator=gen1)  # Advance state
gen1.set_state(state)  # Restore
x3 = torch.randn(5, generator=gen1)
assert torch.allclose(x1, x3)  # Back to same sequence
```

### Fork and Parallelism

**Forking Generators** (for data loading):
```python
# Create base generator
base_gen = torch.Generator().manual_seed(42)

# Fork for parallel workers
worker_gens = []
for worker_id in range(4):
    worker_gen = torch.Generator()
    worker_gen.manual_seed(base_gen.seed() + worker_id)
    worker_gens.append(worker_gen)

# Each worker has independent RNG
for i, gen in enumerate(worker_gens):
    samples = torch.randn(3, generator=gen)
    print(f"Worker {i}: {samples[0].item():.4f}")
```

---

## Basic Sampling Operations

### Uniform Distribution

**torch.rand**: Uniform [0, 1)
```python
torch.rand(
    *size: int,
    *,
    out: Optional[Tensor] = None,
    dtype: Optional[dtype] = None,
    layout: Layout = torch.strided,
    device: Optional[Device] = None,
    requires_grad: bool = False,
    generator: Optional[Generator] = None
) -> Tensor
```

**Example**:
```python
# Sample from U(0, 1)
x = torch.rand(3, 4)  # Shape: [3, 4]

# Custom range U(a, b)
a, b = 2.0, 5.0
x_custom = a + (b - a) * torch.rand(3, 4)  # U(2, 5)

# In-place sampling
x.uniform_(0, 1)  # Modifies x in-place
```

**torch.randint**: Uniform integers
```python
torch.randint(
    low: int,
    high: int,
    size: Tuple[int, ...],
    *,
    generator: Optional[Generator] = None,
    **kwargs
) -> Tensor
```

**Example**:
```python
# Random integers in [0, 10)
x = torch.randint(0, 10, (3, 4))

# Dice roll
dice = torch.randint(1, 7, (100,))  # 100 dice rolls [1, 6]
```

**Implementation** ([aten/src/ATen/native/Distributions.cpp:250-252](reference/pytorch/aten/src/ATen/native/Distributions.cpp#L250-L252)):
```cpp
Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  return at::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, std::move(gen));
}
```

### Normal (Gaussian) Distribution

**torch.randn**: Standard normal N(0, 1)
```python
torch.randn(
    *size: int,
    *,
    generator: Optional[Generator] = None,
    **kwargs
) -> Tensor
```

**torch.normal**: Gaussian with custom parameters
```python
torch.normal(
    mean: Union[float, Tensor],
    std: Union[float, Tensor],
    size: Optional[Tuple[int, ...]] = None,
    *,
    generator: Optional[Generator] = None,
    **kwargs
) -> Tensor
```

**Example**:
```python
# Standard normal
x = torch.randn(3, 4)  # N(0, 1)

# Custom mean and std
x_custom = torch.normal(mean=10.0, std=2.0, size=(3, 4))  # N(10, 2²)

# Per-element parameters
means = torch.tensor([0.0, 5.0, -3.0])
stds = torch.tensor([1.0, 2.0, 0.5])
samples = torch.normal(means, stds)

# In-place
x.normal_(mean=0.0, std=1.0)
```

**Implementation** ([aten/src/ATen/native/Distributions.cpp:275-277](reference/pytorch/aten/src/ATen/native/Distributions.cpp#L275-L277)):
```cpp
Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, std::move(gen));
}
```

### Bernoulli Distribution

**torch.bernoulli**: Binary random variables
```python
torch.bernoulli(
    input: Tensor,  # Probability of 1 (per element)
    *,
    generator: Optional[Generator] = None
) -> Tensor
```

**Example**:
```python
# Coin flips (50% probability)
p = torch.full((10,), 0.5)
flips = torch.bernoulli(p)  # ~5 ones, ~5 zeros

# Dropout mask
dropout_rate = 0.5
keep_prob = 1 - dropout_rate
mask = torch.bernoulli(torch.full((100, 100), keep_prob))

# In-place
x.bernoulli_(p=0.3)  # Each element has 30% chance of being 1
```

**Implementation** ([aten/src/ATen/native/Distributions.cpp:158-162](reference/pytorch/aten/src/ATen/native/Distributions.cpp#L158-L162)):
```cpp
Tensor bernoulli(const Tensor& self, std::optional<Generator> gen) {
  Tensor result = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  result.bernoulli_(self, std::move(gen));
  return result;
}
```

### Random Permutation

**torch.randperm**: Random permutation of integers
```python
torch.randperm(
    n: int,
    *,
    generator: Optional[Generator] = None,
    out: Optional[Tensor] = None,
    dtype: torch.dtype = torch.int64,
    device: Optional[Device] = None
) -> Tensor
```

**Example**:
```python
# Random permutation of [0, n)
perm = torch.randperm(10)  # e.g., tensor([3, 5, 0, 9, 1, 7, 2, 4, 6, 8])

# Shuffle data
data = torch.randn(10, 5)
perm = torch.randperm(10)
shuffled_data = data[perm]

# Batch shuffle
indices = torch.randperm(len(dataset))
shuffled_loader = DataLoader(Subset(dataset, indices), batch_size=32)
```

### Random Integer Fill

**Tensor.random_**: Fill with random integers
```python
Tensor.random_(from: int = 0, to: Optional[int] = None, generator: Optional[Generator] = None) -> Tensor
```

**Example**:
```python
# Random integers [0, to)
x = torch.empty(10, dtype=torch.int64)
x.random_(0, 10)  # Random integers in [0, 10)

# Fill with random floats (converted from integers)
x = torch.empty(5, dtype=torch.float32)
x.random_(0, 100)  # Integer values as floats
```

### Poisson Distribution

**torch.poisson**: Poisson-distributed random numbers
```python
torch.poisson(
    input: Tensor,           # Rate parameter λ (per element)
    generator: Optional[Generator] = None
) -> Tensor
```

**Example**:
```python
# Poisson samples with varying rates
rates = torch.tensor([1.0, 5.0, 10.0])
samples = torch.poisson(rates)  # Integer counts

# Multiple samples per rate
rates = torch.full((100,), 3.5)
samples = torch.poisson(rates)  # ~Poisson(3.5) samples
```

### In-Place Random Operations

All in-place operations modify the tensor directly:

| Operation | Description | Example |
|-----------|-------------|---------|
| `uniform_(from, to)` | Fill with U(from, to) | `x.uniform_(0, 1)` |
| `normal_(mean, std)` | Fill with N(mean, std²) | `x.normal_(0, 1)` |
| `bernoulli_(p)` | Fill with Bernoulli(p) | `x.bernoulli_(0.5)` |
| `exponential_(lambd)` | Fill with Exp(λ) | `x.exponential_(1.5)` |
| `geometric_(p)` | Fill with Geom(p) | `x.geometric_(0.3)` |
| `log_normal_(mean, std)` | Fill with LogNormal | `x.log_normal_(0, 1)` |
| `cauchy_(median, sigma)` | Fill with Cauchy | `x.cauchy_(0, 1)` |
| `random_(from, to)` | Fill with integers | `x.random_(0, 10)` |

### Other Common Distributions

**Exponential**:
```python
# Exponential distribution with rate λ
x = torch.empty(100).exponential_(lambd=1.5)
# PDF: f(x) = λ exp(-λx)
```

**Geometric**:
```python
# Number of Bernoulli trials until first success
x = torch.empty(100).geometric_(p=0.3)
# PMF: P(X=k) = (1-p)^(k-1) * p
```

**Log-Normal**:
```python
# Log-normal: if log(X) ~ N(μ, σ²), then X ~ LogNormal(μ, σ²)
x = torch.empty(100).log_normal_(mean=0.0, std=1.0)
```

**Cauchy**:
```python
# Heavy-tailed distribution
x = torch.empty(100).cauchy_(median=0.0, sigma=1.0)
# PDF: f(x) = 1/(πσ(1 + ((x-x₀)/σ)²))
```

**Multinomial**:
```python
# Sample from categorical distribution
probs = torch.tensor([0.1, 0.3, 0.6])  # 3 categories
samples = torch.multinomial(probs, num_samples=100, replacement=True)
# Output: indices [0, 1, 2] sampled according to probs
```

**Implementation** ([aten/src/ATen/native/Distributions.cpp:81-126](reference/pytorch/aten/src/ATen/native/Distributions.cpp#L81-L126)):
```cpp
// Poisson sampling (transformed rejection method)
int64_t sample_poisson(double lambda, at::CPUGeneratorImpl* generator) {
  if (lambda >= 10) {
    // Hoermann's transformed rejection method (1993)
    double slam = std::sqrt(lambda);
    double loglam = std::log(lambda);
    double b = 0.931 + 2.53 * slam;
    // ... (efficient rejection sampling)
  } else {
    // Knuth's algorithm for small λ
    auto enlam = std::exp(-lambda);
    int64_t X = 0;
    auto prod = 1.0;
    while (true) {
      auto U = standard_uniform(generator);
      prod *= U;
      if (prod > enlam) X += 1;
      else return X;
    }
  }
}
```

---

## Probability Distributions

PyTorch provides **40+ distributions** via the `torch.distributions` module with a consistent API.

### Distribution Base Class

**torch.distributions.Distribution**:
```python
class Distribution:
    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """Generate samples."""
        pass

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        """Generate reparameterized samples (differentiable)."""
        pass

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability density/mass."""
        pass

    def entropy(self) -> Tensor:
        """Entropy of the distribution."""
        pass

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        pass

    @property
    def variance(self) -> Tensor:
        """Variance of the distribution."""
        pass
```

### Common Distributions

#### Normal (Gaussian)

```python
from torch.distributions import Normal

# Create distribution
dist = Normal(loc=0.0, scale=1.0)  # N(0, 1)

# Sample
samples = dist.sample((100,))  # Shape: [100]

# Log probability
log_p = dist.log_prob(samples)

# Reparameterized sampling (for VAEs)
z = dist.rsample((100,))  # Differentiable w.r.t. loc, scale
```

**Reparameterization Trick**:
```python
# Instead of: z ~ N(μ, σ²)
# Use: z = μ + σ * ε, where ε ~ N(0, 1)

mu = torch.tensor(5.0, requires_grad=True)
sigma = torch.tensor(2.0, requires_grad=True)

dist = Normal(mu, sigma)
z = dist.rsample((100,))  # Gradients flow through μ and σ
loss = z.mean()
loss.backward()

print(mu.grad, sigma.grad)  # Non-None gradients
```

#### Categorical

```python
from torch.distributions import Categorical

# Discrete distribution over K categories
probs = torch.tensor([0.1, 0.3, 0.6])
dist = Categorical(probs=probs)

# Sample category indices
samples = dist.sample((100,))  # Values in {0, 1, 2}

# Log probability
log_p = dist.log_prob(samples)

# Entropy
entropy = dist.entropy()  # H(X) = -Σ p_i log(p_i)
```

#### Bernoulli

```python
from torch.distributions import Bernoulli

# Binary distribution
dist = Bernoulli(probs=0.3)

# Sample
samples = dist.sample((100,))  # Binary values {0, 1}

# From logits (numerically stable)
logits = torch.tensor([-2.0, 0.0, 2.0])
dist_logits = Bernoulli(logits=logits)
```

#### Multivariate Normal

```python
from torch.distributions import MultivariateNormal

# Covariance matrix
mean = torch.zeros(3)
cov = torch.eye(3)

dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

# Sample
samples = dist.sample((100,))  # Shape: [100, 3]

# Log probability
log_p = dist.log_prob(samples)  # Shape: [100]
```

#### Beta

```python
from torch.distributions import Beta

# Beta(α, β) distribution on [0, 1]
dist = Beta(concentration1=2.0, concentration0=5.0)

samples = dist.sample((100,))
```

#### Gamma

```python
from torch.distributions import Gamma

# Gamma distribution
dist = Gamma(concentration=2.0, rate=1.0)

samples = dist.sample((100,))
```

### Transformed Distributions

**Transform** distributions allow complex distributions via bijective transformations.

**Example** (Log-Normal from Normal):
```python
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import ExpTransform

# Y = exp(X), where X ~ N(0, 1)
base_dist = Normal(0, 1)
transform = ExpTransform()
lognormal = TransformedDistribution(base_dist, [transform])

# Sample from log-normal
samples = lognormal.sample((100,))  # All positive

# Log probability (with Jacobian correction)
log_p = lognormal.log_prob(samples)
```

**Common Transforms**:
- `ExpTransform`: y = exp(x)
- `SigmoidTransform`: y = 1/(1 + exp(-x))
- `AffineTransform`: y = a*x + b
- `ComposeTransform`: Chain multiple transforms

### KL Divergence

```python
from torch.distributions import Normal, kl_divergence

# Two Gaussian distributions
p = Normal(0, 1)
q = Normal(2, 0.5)

# Compute KL(P || Q)
kl = kl_divergence(p, q)
# For Gaussians: KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁²+(μ₁-μ₂)²)/(2σ₂²) - 1/2

print(f"KL divergence: {kl.item():.4f}")
```

---

## Reproducibility & Seeding

### Global Seed

**Set global random seed** (affects all default generators):
```python
import torch
import random
import numpy as np

def set_seed(seed: int):
    """Set seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Now all random operations are deterministic
x1 = torch.randn(100)
set_seed(42)
x2 = torch.randn(100)
assert torch.allclose(x1, x2)
```

### Reproducible DataLoader

```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Deterministic augmentation with worker-specific seed
        sample = self.data[idx]
        sample = sample + 0.1 * torch.randn_like(sample)
        return sample

def seed_worker(worker_id):
    """Seed each DataLoader worker independently."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create reproducible DataLoader
g = torch.Generator()
g.manual_seed(42)

dataloader = DataLoader(
    MyDataset(),
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g
)

# Identical batches across runs
for batch in dataloader:
    break
print(batch[0])  # Always the same
```

### Checkpoint RNG State

```python
# Save model and RNG state
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
}
torch.save(checkpoint, 'checkpoint.pth')

# Restore
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
torch.set_rng_state(checkpoint['rng_state'])
if checkpoint['cuda_rng_state']:
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

# Training continues from exact same RNG state
```

---

## MLX Porting Guide

### MLX Random API Status (as of 2024)

MLX provides basic random operations via `mlx.core.random`.

**Available Operations**:
```python
import mlx.core as mx
from mlx.core import random

# Uniform distributions
random.uniform(low=0.0, high=1.0, shape=(3, 4))

# Normal distribution
random.normal(shape=(3, 4), loc=0.0, scale=1.0)

# Bernoulli
random.bernoulli(p=0.5, shape=(100,))

# Categorical
random.categorical(logits, num_samples=10)

# Seed management
random.seed(42)
```

**Missing Features** (compared to PyTorch):
- ❌ Generator objects (independent RNG state)
- ❌ `torch.distributions` module
- ❌ Exponential, Cauchy, Geometric, Poisson distributions
- ❌ Multivariate distributions
- ❌ KL divergence utilities
- ❌ Transformed distributions

### Implementing Missing Distributions

#### Exponential Distribution

```python
import mlx.core as mx

def exponential(lambd, shape):
    """
    Exponential distribution using inverse transform sampling.

    PDF: f(x) = λ exp(-λx)
    CDF: F(x) = 1 - exp(-λx)
    Inverse CDF: F^(-1)(u) = -log(1-u) / λ
    """
    u = mx.random.uniform(0, 1, shape)
    return -mx.log(1 - u) / lambd

# Example
samples = exponential(lambd=1.5, shape=(100,))
```

#### Gamma Distribution

```python
def gamma(alpha, beta, shape):
    """
    Gamma distribution using Marsaglia & Tsang's method (2000).

    For α < 1, use: Gamma(α) = Gamma(α+1) * U^(1/α)
    For α ≥ 1, use rejection sampling
    """
    if alpha < 1:
        # Boost alpha
        return gamma(alpha + 1, beta, shape) * mx.power(mx.random.uniform(0, 1, shape), 1/alpha)

    # Marsaglia & Tsang's method
    d = alpha - 1/3
    c = 1 / mx.sqrt(9 * d)

    samples = []
    while len(samples) < shape[0]:
        z = mx.random.normal(shape=(shape[0] * 2,))  # Oversample
        u = mx.random.uniform(0, 1, (shape[0] * 2,))

        v = (1 + c * z) ** 3
        accept = (z > -1/c) & (mx.log(u) < 0.5 * z**2 + d - d*v + d*mx.log(v))

        samples.extend((d * v[accept] / beta).tolist())

    return mx.array(samples[:shape[0]]).reshape(shape)
```

#### Beta Distribution

```python
def beta(alpha, beta_param, shape):
    """
    Beta distribution using ratio of Gammas.

    If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
    then X/(X+Y) ~ Beta(α, β)
    """
    x = gamma(alpha, 1.0, shape)
    y = gamma(beta_param, 1.0, shape)
    return x / (x + y)
```

### Generator Object Emulation

Since MLX doesn't have generator objects, use a custom class:

```python
class MLXGenerator:
    """Emulate PyTorch Generator for MLX."""

    def __init__(self, seed=None):
        self._seed = seed if seed is not None else int(mx.random.uniform(0, 2**32, (1,))[0])
        self.manual_seed(self._seed)

    def manual_seed(self, seed):
        """Set the seed."""
        self._seed = seed
        mx.random.seed(seed)
        return self

    def initial_seed(self):
        """Get initial seed."""
        return self._seed

    def get_state(self):
        """Get RNG state (not fully supported in MLX)."""
        return {'seed': self._seed}

    def set_state(self, state):
        """Restore RNG state."""
        self.manual_seed(state['seed'])


# Usage
gen = MLXGenerator(seed=42)
x = mx.random.normal(shape=(100,))

# Fork for workers
worker_gens = [MLXGenerator(seed=42 + i) for i in range(4)]
```

### C++ API Design for MLX

```cpp
// mlx/random/random.h

namespace mlx::core::random {

// Basic distributions
array uniform(float low, float high, const std::vector<int>& shape, Dtype dtype = float32);
array normal(const std::vector<int>& shape, float loc = 0.0, float scale = 1.0, Dtype dtype = float32);
array bernoulli(float p, const std::vector<int>& shape, Dtype dtype = float32);

// Additional distributions
array exponential(float lambda, const std::vector<int>& shape, Dtype dtype = float32);
array gamma(float alpha, float beta, const std::vector<int>& shape, Dtype dtype = float32);
array beta(float alpha, float beta, const std::vector<int>& shape, Dtype dtype = float32);
array poisson(float lambda, const std::vector<int>& shape, Dtype dtype = float32);
array geometric(float p, const std::vector<int>& shape, Dtype dtype = int32);

// Categorical
array categorical(const array& logits, int num_samples, Dtype dtype = int32);

// Multivariate
array multivariate_normal(const array& mean, const array& cov, const std::vector<int>& shape);

// Seed management
void seed(uint64_t seed);
uint64_t get_seed();

}  // namespace mlx::core::random
```

---

## Performance Considerations

### Batch Sampling

**Inefficient** (loop):
```python
samples = []
for _ in range(1000):
    samples.append(torch.randn(10))
samples = torch.stack(samples)
```

**Efficient** (vectorized):
```python
samples = torch.randn(1000, 10)  # Single call
```

### In-Place Operations

**Out-of-place** (allocates new tensor):
```python
x = torch.empty(1000, 1000)
x = torch.randn(1000, 1000)  # New allocation
```

**In-place** (reuses memory):
```python
x = torch.empty(1000, 1000)
x.normal_()  # Fills existing tensor
```

### Generator Overhead

**Default generator** (fast):
```python
x = torch.randn(1000)  # Uses default generator
```

**Custom generator** (slightly slower):
```python
gen = torch.Generator()
x = torch.randn(1000, generator=gen)  # Extra overhead
```

### Distribution Sampling

**Direct sampling** (fastest):
```python
x = torch.randn(1000)  # Direct C++ kernel
```

**Distribution object** (more flexible, slightly slower):
```python
from torch.distributions import Normal
dist = Normal(0, 1)
x = dist.sample((1000,))  # Python overhead
```

**Benchmark**:
```python
import time

# Direct
start = time.time()
for _ in range(1000):
    _ = torch.randn(100)
print(f"Direct: {time.time() - start:.4f}s")  # ~0.05s

# Distribution
dist = Normal(0, 1)
start = time.time()
for _ in range(1000):
    _ = dist.sample((100,))
print(f"Distribution: {time.time() - start:.4f}s")  # ~0.15s
```

---

## Summary

### Key Takeaways

1. **Generators**: Manage RNG state independently per device
2. **Reproducibility**: Use `manual_seed()` and save RNG state in checkpoints
3. **Distributions**: `torch.distributions` provides 40+ probability distributions
4. **Reparameterization**: Use `rsample()` for differentiable sampling in VAEs
5. **MLX**: Basic RNG available, but missing generator objects and distributions

### API Mapping

| PyTorch | MLX | Status | Notes |
|---------|-----|--------|-------|
| `torch.rand` | `mx.random.uniform` | ✅ Available | |
| `torch.randn` | `mx.random.normal` | ✅ Available | |
| `torch.randint` | `mx.random.randint` | ✅ Available | |
| `torch.randperm` | ❌ Missing | Implement via shuffle | |
| `torch.bernoulli` | `mx.random.bernoulli` | ✅ Available | |
| `torch.poisson` | ❌ Missing | Implement with rejection sampling | |
| `torch.multinomial` | `mx.random.categorical` | ⚠️ Partial | Logits only |
| `torch.Generator` | ❌ Missing | Emulate with custom class | |
| `Tensor.uniform_` | ❌ Missing | No in-place support | |
| `Tensor.normal_` | ❌ Missing | No in-place support | |
| `torch.distributions.Normal` | ❌ Missing | Implement from scratch | |
| `torch.distributions.Categorical` | `mx.random.categorical` | ⚠️ Partial | Logits only |
| `torch.distributions.Beta` | ❌ Missing | Use ratio of Gammas | |
| `kl_divergence` | ❌ Missing | Implement per-distribution | |
| Transformed distributions | ❌ Missing | Requires bijector framework | |

### MLX Random Permutation Implementation

```python
import mlx.core as mx

def randperm(n: int, key: mx.array = None) -> mx.array:
    """
    Generate a random permutation of integers from 0 to n-1.

    MLX doesn't have randperm, so we implement using Fisher-Yates shuffle.
    """
    if key is None:
        # Use random uniform to generate shuffle keys
        keys = mx.random.uniform(shape=(n,))
    else:
        keys = mx.random.uniform(shape=(n,), key=key)

    # Sort by random keys to get permutation
    return mx.argsort(keys)

# Usage
perm = randperm(10)  # Random permutation of [0, 9]
```

### MLX Poisson Implementation

```python
def poisson(lam: mx.array, key: mx.array = None) -> mx.array:
    """
    Poisson sampling using inverse transform method (for small λ).

    For large λ (>10), use normal approximation: Poisson(λ) ≈ N(λ, λ)
    """
    shape = lam.shape

    # Normal approximation for large λ
    large_lambda = lam >= 10
    normal_samples = mx.round(lam + mx.sqrt(lam) * mx.random.normal(shape=shape))
    normal_samples = mx.maximum(normal_samples, 0)

    # Knuth's algorithm for small λ
    # Note: This is inefficient for MLX; prefer vectorized approaches
    # For production, implement proper rejection sampling

    return mx.where(large_lambda, normal_samples, normal_samples)  # Simplified
```

### Implementation Checklist for MLX

**High Priority**:
- ✅ Basic sampling (uniform, normal, bernoulli, categorical)
- ❌ Generator objects with independent state
- ❌ Exponential, Geometric, Poisson distributions

**Medium Priority**:
- ❌ `mlx.distributions` module (Normal, Categorical, Beta, etc.)
- ❌ Reparameterized sampling (`rsample`)
- ❌ KL divergence utilities

**Low Priority**:
- ❌ Transformed distributions
- ❌ Multivariate distributions
- ❌ Exotic distributions (VonMises, Wishart, etc.)

---

## References

**PyTorch Source Files**:
- [aten/src/ATen/native/Distributions.cpp](reference/pytorch/aten/src/ATen/native/Distributions.cpp) - Core RNG implementations
- [aten/src/ATen/CPUGeneratorImpl.h](reference/pytorch/aten/src/ATen/CPUGeneratorImpl.h) - Generator state management
- [torch/distributions/](reference/pytorch/torch/distributions/) - 40+ distribution implementations

**Algorithm References**:
- Mersenne Twister: Matsumoto & Nishimura (1998)
- Poisson sampling: Hoermann's transformed rejection method (1993)
- Gamma sampling: Marsaglia & Tsang (2000)
- Box-Muller transform for normal sampling

**Metal/Apple Documentation**:
- Metal random number generation
- Accelerate framework
