# Special Mathematical Functions (torch.special)

## Purpose

The `torch.special` module provides special mathematical functions commonly used in scientific computing, statistics, and machine learning. These include:
- Error functions (used in probability and statistics)
- Gamma functions (fundamental to many distributions)
- Bessel functions (physics and signal processing)
- Orthogonal polynomials (numerical methods)
- Other transcendental functions

**Reference Files:**
- `torch/special/__init__.py` - Python API and documentation (~1,500 lines)

---

## Overview

### Function Categories

```
torch.special Module
├── Error Functions
│   ├── erf, erfc, erfcx, erfinv
│   └── ndtr, ndtri, log_ndtr
│
├── Gamma Functions
│   ├── gammaln, digamma (psi), polygamma
│   ├── gammainc, gammaincc
│   └── multigammaln
│
├── Bessel Functions
│   ├── First kind: bessel_j0, bessel_j1
│   ├── Second kind: bessel_y0, bessel_y1
│   ├── Modified: modified_bessel_i0/i1, modified_bessel_k0/k1
│   ├── Scaled: i0, i0e, i1, i1e, scaled_modified_bessel_k0/k1
│   └── Spherical: spherical_bessel_j0
│
├── Orthogonal Polynomials
│   ├── Chebyshev: chebyshev_polynomial_t/u/v/w
│   ├── Shifted Chebyshev: shifted_chebyshev_polynomial_t/u/v/w
│   ├── Hermite: hermite_polynomial_h, hermite_polynomial_he
│   ├── Laguerre: laguerre_polynomial_l
│   └── Legendre: legendre_polynomial_p
│
├── Exponential/Logarithmic
│   ├── exp2, expm1, expit
│   ├── log1p, logit
│   ├── xlogy, xlog1py
│   └── logsumexp
│
├── Activation-Related
│   ├── softmax, log_softmax
│   └── sinc
│
└── Other
    ├── entr (entropy)
    ├── zeta (Riemann zeta)
    ├── round (banker's rounding)
    └── airy_ai (Airy function)
```

---

## Error Functions

### erf (Error Function)

**Purpose**: Compute the error function, fundamental to probability theory.

**Formula**:
$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt$$

**Signature**:
```python
torch.special.erf(input, *, out=None) -> Tensor
```

**Properties**:
- Range: (-1, 1)
- erf(0) = 0
- erf(∞) = 1
- erf(-x) = -erf(x) (odd function)

**Usage**:
```python
import torch

x = torch.tensor([0., -1., 10.])
torch.special.erf(x)
# tensor([0.0000, -0.8427, 1.0000])

# Relationship to normal CDF
# CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
```

---

### erfc (Complementary Error Function)

**Formula**:
$$\text{erfc}(x) = 1 - \text{erf}(x)$$

**Signature**:
```python
torch.special.erfc(input, *, out=None) -> Tensor
```

**Usage**:
```python
torch.special.erfc(torch.tensor([0., -1., 10.]))
# tensor([1.0000, 1.8427, 0.0000])
```

---

### erfcx (Scaled Complementary Error Function)

**Formula**:
$$\text{erfcx}(x) = e^{x^2} \cdot \text{erfc}(x)$$

**Purpose**: Numerically stable for large positive x (where erfc underflows).

**Signature**:
```python
torch.special.erfcx(input, *, out=None) -> Tensor
```

---

### erfinv (Inverse Error Function)

**Formula**: The inverse such that erfinv(erf(x)) = x

**Domain**: (-1, 1)

**Signature**:
```python
torch.special.erfinv(input, *, out=None) -> Tensor
```

**Usage**:
```python
x = torch.tensor([0., 0.5, -0.9])
torch.special.erfinv(x)
# tensor([0.0000, 0.4769, -1.1631])
```

---

### ndtr (Normal CDF)

**Purpose**: Standard normal cumulative distribution function.

**Formula**:
$$\Phi(x) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

**Signature**:
```python
torch.special.ndtr(input, *, out=None) -> Tensor
```

**Usage**:
```python
x = torch.tensor([0., 1., 2.])
torch.special.ndtr(x)
# tensor([0.5000, 0.8413, 0.9772])
```

---

### ndtri (Inverse Normal CDF)

**Purpose**: Inverse of ndtr (quantile function).

**Signature**:
```python
torch.special.ndtri(input, *, out=None) -> Tensor
```

**Usage**:
```python
p = torch.tensor([0.5, 0.8413, 0.9772])
torch.special.ndtri(p)
# tensor([0.0000, 1.0000, 2.0000])
```

---

### log_ndtr (Log Normal CDF)

**Purpose**: Log of normal CDF, numerically stable for small probabilities.

**Signature**:
```python
torch.special.log_ndtr(input, *, out=None) -> Tensor
```

---

## Gamma Functions

### gammaln (Log Gamma)

**Purpose**: Natural log of the absolute value of the gamma function.

**Formula**:
$$\text{gammaln}(x) = \ln|\Gamma(x)|$$

**Signature**:
```python
torch.special.gammaln(input, *, out=None) -> Tensor
```

**Usage**:
```python
x = torch.arange(0.5, 2, 0.5)
torch.special.gammaln(x)
# tensor([0.5724, 0.0000, -0.1208])

# Factorial: n! = exp(gammaln(n+1))
n = torch.tensor([5.])
torch.exp(torch.special.gammaln(n + 1))  # 120.0
```

---

### digamma / psi (Logarithmic Derivative of Gamma)

**Formula**:
$$\psi(x) = \frac{d}{dx}\ln\Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}$$

**Signature**:
```python
torch.special.digamma(input, *, out=None) -> Tensor
torch.special.psi(input, *, out=None) -> Tensor  # Alias
```

**Usage**:
```python
x = torch.tensor([1., 0.5])
torch.special.digamma(x)
# tensor([-0.5772, -1.9635])
```

---

### polygamma (Higher Derivatives)

**Formula**:
$$\psi^{(n)}(x) = \frac{d^n}{dx^n}\psi(x)$$

**Signature**:
```python
torch.special.polygamma(n, input, *, out=None) -> Tensor
```

**Usage**:
```python
x = torch.tensor([1., 0.5])
torch.special.polygamma(1, x)  # First derivative (trigamma)
# tensor([1.6449, 4.9348])
```

---

### gammainc / gammaincc (Incomplete Gamma)

**Purpose**: Regularized incomplete gamma functions.

**Formulas**:
$$\gamma(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} dt$$
$$\Gamma_c(a, x) = 1 - \gamma(a, x)$$

**Signatures**:
```python
torch.special.gammainc(input, other, *, out=None) -> Tensor
torch.special.gammaincc(input, other, *, out=None) -> Tensor
```

---

### multigammaln (Multivariate Log Gamma)

**Purpose**: Log of the multivariate gamma function.

**Signature**:
```python
torch.special.multigammaln(input, p, *, out=None) -> Tensor
```

---

## Bessel Functions

### First Kind (J)

**Purpose**: Solutions to Bessel's differential equation.

**Signatures**:
```python
torch.special.bessel_j0(input, *, out=None) -> Tensor  # Order 0
torch.special.bessel_j1(input, *, out=None) -> Tensor  # Order 1
```

**Usage**:
```python
x = torch.linspace(0, 10, 100)
j0 = torch.special.bessel_j0(x)
j1 = torch.special.bessel_j1(x)
```

---

### Second Kind (Y)

**Purpose**: Bessel functions of the second kind (Neumann functions).

**Signatures**:
```python
torch.special.bessel_y0(input, *, out=None) -> Tensor
torch.special.bessel_y1(input, *, out=None) -> Tensor
```

---

### Modified Bessel Functions (I, K)

**Purpose**: Modified Bessel functions for exponentially growing/decaying solutions.

**I (First Kind - Growing)**:
```python
torch.special.modified_bessel_i0(input, *, out=None) -> Tensor
torch.special.modified_bessel_i1(input, *, out=None) -> Tensor
```

**K (Second Kind - Decaying)**:
```python
torch.special.modified_bessel_k0(input, *, out=None) -> Tensor
torch.special.modified_bessel_k1(input, *, out=None) -> Tensor
```

---

### Scaled Bessel Functions

**Purpose**: Numerically stable versions for large arguments.

**Signatures**:
```python
# Scaled I functions: exp(-|x|) * I_n(x)
torch.special.i0(input, *, out=None) -> Tensor
torch.special.i0e(input, *, out=None) -> Tensor  # Exponentially scaled
torch.special.i1(input, *, out=None) -> Tensor
torch.special.i1e(input, *, out=None) -> Tensor

# Scaled K functions
torch.special.scaled_modified_bessel_k0(input, *, out=None) -> Tensor
torch.special.scaled_modified_bessel_k1(input, *, out=None) -> Tensor
```

---

### Spherical Bessel

**Signature**:
```python
torch.special.spherical_bessel_j0(input, *, out=None) -> Tensor
```

**Formula**:
$$j_0(x) = \frac{\sin(x)}{x}$$

---

## Orthogonal Polynomials

### Chebyshev Polynomials

**Four types (T, U, V, W)**:

**Signatures**:
```python
torch.special.chebyshev_polynomial_t(input, n, *, out=None) -> Tensor
torch.special.chebyshev_polynomial_u(input, n, *, out=None) -> Tensor
torch.special.chebyshev_polynomial_v(input, n, *, out=None) -> Tensor
torch.special.chebyshev_polynomial_w(input, n, *, out=None) -> Tensor
```

**Recurrence (Type T)**:
- T_0(x) = 1
- T_1(x) = x
- T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)

**Shifted Chebyshev** (domain [0, 1] instead of [-1, 1]):
```python
torch.special.shifted_chebyshev_polynomial_t(input, n, *, out=None) -> Tensor
torch.special.shifted_chebyshev_polynomial_u(input, n, *, out=None) -> Tensor
torch.special.shifted_chebyshev_polynomial_v(input, n, *, out=None) -> Tensor
torch.special.shifted_chebyshev_polynomial_w(input, n, *, out=None) -> Tensor
```

---

### Hermite Polynomials

**Two conventions**:

**Physicist's Hermite (H_n)**:
```python
torch.special.hermite_polynomial_h(input, n, *, out=None) -> Tensor
```

**Probabilist's Hermite (He_n)**:
```python
torch.special.hermite_polynomial_he(input, n, *, out=None) -> Tensor
```

---

### Laguerre Polynomials

**Signature**:
```python
torch.special.laguerre_polynomial_l(input, n, *, out=None) -> Tensor
```

---

### Legendre Polynomials

**Signature**:
```python
torch.special.legendre_polynomial_p(input, n, *, out=None) -> Tensor
```

---

## Exponential and Logarithmic

### exp2 (Base-2 Exponential)

**Formula**: 2^x

**Signature**:
```python
torch.special.exp2(input, *, out=None) -> Tensor
```

---

### expm1 (exp(x) - 1)

**Purpose**: Numerically stable for small x.

**Signature**:
```python
torch.special.expm1(input, *, out=None) -> Tensor
```

---

### expit (Sigmoid)

**Formula**: 1 / (1 + exp(-x))

**Signature**:
```python
torch.special.expit(input, *, out=None) -> Tensor
```

**Note**: Equivalent to `torch.sigmoid`.

---

### log1p (log(1 + x))

**Purpose**: Numerically stable for small x.

**Signature**:
```python
torch.special.log1p(input, *, out=None) -> Tensor
```

---

### logit (Inverse Sigmoid)

**Formula**: log(x / (1 - x))

**Signature**:
```python
torch.special.logit(input, eps=None, *, out=None) -> Tensor
```

---

### xlogy / xlog1py

**Formulas**:
- xlogy(x, y) = x * log(y), with 0 * log(0) = 0
- xlog1py(x, y) = x * log1p(y), with 0 * log1p(y) = 0

**Signatures**:
```python
torch.special.xlogy(input, other, *, out=None) -> Tensor
torch.special.xlog1py(input, other, *, out=None) -> Tensor
```

---

### logsumexp

**Formula**: log(sum(exp(x)))

**Signature**:
```python
torch.special.logsumexp(input, dim, keepdim=False, *, out=None) -> Tensor
```

---

## Activation and Other Functions

### softmax / log_softmax

**Signatures**:
```python
torch.special.softmax(input, dim, *, dtype=None) -> Tensor
torch.special.log_softmax(input, dim, *, dtype=None) -> Tensor
```

---

### sinc (Normalized Sinc)

**Formula**: sin(πx) / (πx), with sinc(0) = 1

**Signature**:
```python
torch.special.sinc(input, *, out=None) -> Tensor
```

---

### entr (Entropy)

**Formula**:
- -x * ln(x) for x > 0
- 0 for x = 0
- -∞ for x < 0

**Signature**:
```python
torch.special.entr(input, *, out=None) -> Tensor
```

---

### zeta (Riemann Zeta)

**Formula**:
$$\zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}$$

**Signature**:
```python
torch.special.zeta(input, other, *, out=None) -> Tensor
```

---

### round (Banker's Rounding)

**Purpose**: Round to nearest even (ties go to even).

**Signature**:
```python
torch.special.round(input, *, out=None) -> Tensor
```

---

### airy_ai (Airy Function)

**Purpose**: Solution to Airy's differential equation y'' - xy = 0.

**Signature**:
```python
torch.special.airy_ai(input, *, out=None) -> Tensor
```

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `torch.special.erf` | `mx.erf` |
| `torch.special.erfc` | `mx.erfc` (may need custom impl) |
| `torch.special.expit` | `mx.sigmoid` |
| `torch.special.log1p` | `mx.log1p` |
| `torch.special.exp2` | `2 ** x` or custom |
| `torch.special.expm1` | `mx.expm1` |
| `torch.special.softmax` | `mx.softmax` |
| `torch.special.logsumexp` | `mx.logsumexp` |
| `torch.special.sinc` | Custom implementation |

### Custom Implementations for MLX

```python
import mlx.core as mx
import math

def erfc_mlx(x):
    """Complementary error function"""
    return 1.0 - mx.erf(x)

def erfcx_mlx(x):
    """Scaled complementary error function"""
    return mx.exp(x * x) * erfc_mlx(x)

def ndtr_mlx(x):
    """Standard normal CDF"""
    return 0.5 * (1 + mx.erf(x / math.sqrt(2)))

def sinc_mlx(x):
    """Normalized sinc function"""
    # Handle x = 0 case
    pi_x = math.pi * x
    return mx.where(x == 0, mx.ones_like(x), mx.sin(pi_x) / pi_x)

def logit_mlx(x, eps=1e-6):
    """Logit function (inverse sigmoid)"""
    x = mx.clip(x, eps, 1 - eps)
    return mx.log(x / (1 - x))

def gammaln_mlx(x):
    """Log gamma function (Stirling approximation for large x)"""
    # Use Lanczos approximation or scipy-like implementation
    # This is a simplified version
    return (x - 0.5) * mx.log(x) - x + 0.5 * math.log(2 * math.pi)

def digamma_mlx(x):
    """Digamma function (approximate)"""
    # Asymptotic expansion for large x
    return mx.log(x) - 0.5 / x - 1.0 / (12 * x * x)

def xlogy_mlx(x, y):
    """x * log(y) with 0 * log(0) = 0"""
    result = x * mx.log(y)
    return mx.where(x == 0, mx.zeros_like(result), result)
```

### Bessel Functions for MLX

Bessel functions require more complex implementations, typically using series expansions or asymptotic approximations:

```python
def bessel_j0_mlx(x):
    """Bessel J0 (simplified polynomial approximation)"""
    # For |x| < 3, use polynomial approximation
    # For |x| >= 3, use asymptotic expansion
    # This is a placeholder - real implementation needs careful numerics
    ax = mx.abs(x)

    # Small x approximation
    y = x * x
    small = 1 - y/4 + y*y/64 - y*y*y/2304

    # Large x asymptotic
    z = 3.0 / ax
    large = mx.sqrt(0.636619772 / ax) * mx.cos(ax - 0.785398164)

    return mx.where(ax < 3.0, small, large)
```

---

## Use Cases

### Statistical Computing

```python
# Normal distribution operations
x = torch.randn(1000)
cdf = torch.special.ndtr(x)
quantile = torch.special.ndtri(torch.tensor([0.025, 0.975]))

# Gamma distribution
from torch.distributions import Gamma
# Uses gammaln internally for log_prob
```

### Signal Processing

```python
# Bessel functions for filter design
x = torch.linspace(0, 10, 1000)
j0 = torch.special.bessel_j0(x)
j1 = torch.special.bessel_j1(x)
```

### Machine Learning

```python
# Binary cross-entropy with logits
logits = model(x)
probs = torch.special.expit(logits)

# Numerical stability
log_probs = torch.special.log_softmax(logits, dim=-1)
```

---

## Summary Table

| Category | Functions |
|----------|-----------|
| Error | erf, erfc, erfcx, erfinv, ndtr, ndtri, log_ndtr |
| Gamma | gammaln, digamma, psi, polygamma, gammainc, gammaincc, multigammaln |
| Bessel (J) | bessel_j0, bessel_j1 |
| Bessel (Y) | bessel_y0, bessel_y1 |
| Modified Bessel | modified_bessel_i0/i1, modified_bessel_k0/k1 |
| Scaled Bessel | i0, i0e, i1, i1e, scaled_modified_bessel_k0/k1 |
| Spherical | spherical_bessel_j0 |
| Chebyshev | chebyshev_polynomial_t/u/v/w, shifted variants |
| Hermite | hermite_polynomial_h, hermite_polynomial_he |
| Other Polynomials | laguerre_polynomial_l, legendre_polynomial_p |
| Exp/Log | exp2, expm1, expit, log1p, logit, xlogy, xlog1py, logsumexp |
| Activation | softmax, log_softmax, sinc |
| Other | entr, zeta, round, airy_ai |
