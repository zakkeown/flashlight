# Activation Layers

## Overview

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. PyTorch provides 26+ activation functions as `nn.Module` classes in `torch.nn.modules.activation`.

**Source**: `torch/nn/modules/activation.py`

---

## Activation Categories

| Category | Functions | Characteristics |
|----------|-----------|-----------------|
| **Standard** | ReLU, Sigmoid, Tanh | Classic, widely used |
| **ReLU Variants** | LeakyReLU, PReLU, ReLU6, ELU, SELU | Address dying ReLU |
| **Modern** | GELU, SiLU/Swish, Mish | Transformer-era |
| **Hard Approximations** | Hardtanh, Hardsigmoid, Hardswish | Efficient on hardware |
| **Shrinkage** | Softshrink, Hardshrink, Tanhshrink | Sparse activations |
| **Smooth** | Softplus, Softsign | Smooth ReLU alternatives |
| **Gating** | GLU | Gated linear unit |
| **Normalization** | Softmax, LogSoftmax, Softmin | Probability outputs |

---

## Standard Activations

### ReLU

```python
class ReLU(Module):
    def __init__(self, inplace: bool = False)
```

**Formula**: `ReLU(x) = max(0, x)`

**Properties**:
- Simple, computationally efficient
- Sparse activations (exactly zero for negative inputs)
- May suffer from "dying ReLU" (neurons that never activate)

```python
m = nn.ReLU()
m_inplace = nn.ReLU(inplace=True)  # Memory efficient
```

### Sigmoid

```python
class Sigmoid(Module):
    def __init__(self)
```

**Formula**: `Sigmoid(x) = 1 / (1 + exp(-x))`

**Properties**:
- Output range: (0, 1)
- Suitable for binary classification output
- Gradient vanishing for large |x|

### Tanh

```python
class Tanh(Module):
    def __init__(self)
```

**Formula**: `Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**Properties**:
- Output range: (-1, 1)
- Zero-centered (unlike Sigmoid)
- Common in RNNs/LSTMs

---

## ReLU Variants

### LeakyReLU

```python
class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False)
```

**Formula**:
```
LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
             = x if x >= 0 else negative_slope * x
```

**Properties**:
- Allows small gradient for negative inputs
- Addresses dying ReLU problem

### PReLU (Parametric ReLU)

```python
class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25)
```

**Formula**: `PReLU(x) = max(0, x) + a * min(0, x)`

**Properties**:
- Learnable negative slope `a`
- `num_parameters=1`: Single slope for all channels
- `num_parameters=C`: Per-channel slopes

```python
# Single learnable parameter
m = nn.PReLU()

# Per-channel parameters
m = nn.PReLU(num_parameters=64)  # 64 channels
```

### ReLU6

```python
class ReLU6(Hardtanh):
    def __init__(self, inplace: bool = False)
```

**Formula**: `ReLU6(x) = min(max(0, x), 6)`

**Properties**:
- Capped at 6 for mobile deployment
- Used in MobileNet architectures

### ELU (Exponential Linear Unit)

```python
class ELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False)
```

**Formula**:
```
ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
```

**Properties**:
- Smooth for negative inputs
- Non-zero mean activations
- Self-normalizing properties

### CELU (Continuously Differentiable ELU)

```python
class CELU(Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False)
```

**Formula**: `CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`

**Properties**:
- Continuously differentiable everywhere
- Smoother than ELU

### SELU (Scaled ELU)

```python
class SELU(Module):
    def __init__(self, inplace: bool = False)
```

**Formula**:
```
SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
```
Where:
- `alpha = 1.6732632423543772848170429916717`
- `scale = 1.0507009873554804934193349852946`

**Properties**:
- Self-normalizing (activations converge to zero mean, unit variance)
- Requires specific initialization (`lecun_normal`)
- Best with `AlphaDropout`

### RReLU (Randomized ReLU)

```python
class RReLU(Module):
    def __init__(self, lower: float = 1/8, upper: float = 1/3, inplace: bool = False)
```

**Formula**:
```
RReLU(x) = x if x >= 0 else a * x
where a ~ Uniform(lower, upper) during training
      a = (lower + upper) / 2 during evaluation
```

**Properties**:
- Regularization via randomized slope
- Different behavior in train/eval modes

---

## Modern Activations

### GELU (Gaussian Error Linear Unit)

```python
class GELU(Module):
    def __init__(self, approximate: str = 'none')
```

**Formula**:
```
GELU(x) = x * Φ(x)
```
Where Φ(x) is the CDF of standard normal distribution.

**Approximation** (`approximate='tanh'`):
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**Properties**:
- Used in BERT, GPT, ViT
- Smooth, non-monotonic
- Better gradient flow than ReLU

```python
m = nn.GELU()
m_approx = nn.GELU(approximate='tanh')  # Faster
```

### SiLU / Swish

```python
class SiLU(Module):
    def __init__(self, inplace: bool = False)
```

**Formula**: `SiLU(x) = x * sigmoid(x)`

**Properties**:
- Self-gated activation
- Smooth, non-monotonic
- Used in EfficientNet, Mamba

### Mish

```python
class Mish(Module):
    def __init__(self, inplace: bool = False)
```

**Formula**: `Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`

**Properties**:
- Self-regularizing
- Smooth, non-monotonic
- Used in YOLOv4, EfficientDet

---

## Hard Approximations

Efficient approximations for mobile/edge deployment.

### Hardtanh

```python
class Hardtanh(Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False)
```

**Formula**:
```
Hardtanh(x) = max_val if x > max_val else (min_val if x < min_val else x)
```

### Hardsigmoid

```python
class Hardsigmoid(Module):
    def __init__(self, inplace: bool = False)
```

**Formula**:
```
Hardsigmoid(x) = 0 if x <= -3 else (1 if x >= 3 else x/6 + 0.5)
```

### Hardswish

```python
class Hardswish(Module):
    def __init__(self, inplace: bool = False)
```

**Formula**:
```
Hardswish(x) = 0 if x <= -3 else (x if x >= 3 else x * (x + 3) / 6)
```

**Properties**:
- Used in MobileNetV3
- Efficient approximation of Swish

---

## Shrinkage Activations

### Softshrink

```python
class Softshrink(Module):
    def __init__(self, lambd: float = 0.5)
```

**Formula**:
```
Softshrink(x) = x - lambd if x > lambd else (x + lambd if x < -lambd else 0)
```

### Hardshrink

```python
class Hardshrink(Module):
    def __init__(self, lambd: float = 0.5)
```

**Formula**:
```
Hardshrink(x) = x if |x| > lambd else 0
```

### Tanhshrink

```python
class Tanhshrink(Module):
    def __init__(self)
```

**Formula**: `Tanhshrink(x) = x - tanh(x)`

---

## Smooth Activations

### Softplus

```python
class Softplus(Module):
    def __init__(self, beta: float = 1, threshold: float = 20)
```

**Formula**: `Softplus(x) = (1/beta) * log(1 + exp(beta * x))`

For `x > threshold`, returns `x` to avoid overflow.

**Properties**:
- Smooth approximation of ReLU
- Always positive
- Used as log-variance in VAEs

### Softsign

```python
class Softsign(Module):
    def __init__(self)
```

**Formula**: `Softsign(x) = x / (1 + |x|)`

**Properties**:
- Output range: (-1, 1)
- Smoother than tanh, slower saturation

---

## Gating Activations

### GLU (Gated Linear Unit)

```python
class GLU(Module):
    def __init__(self, dim: int = -1)
```

**Formula**: `GLU(a, b) = a ⊗ sigmoid(b)`

Where input is split into `a` and `b` along `dim`.

**Properties**:
- Halves the dimension
- Used in language models, ConvS2S

```python
m = nn.GLU(dim=1)
input = torch.randn(4, 10)  # Must be even on split dim
output = m(input)  # Shape: (4, 5)
```

---

## Normalization Activations

### Softmax

```python
class Softmax(Module):
    def __init__(self, dim: int = None)
```

**Formula**: `Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)`

```python
m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)  # Sums to 1 along dim=1
```

### LogSoftmax

```python
class LogSoftmax(Module):
    def __init__(self, dim: int = None)
```

**Formula**: `LogSoftmax(x_i) = log(exp(x_i) / Σ_j exp(x_j))`

**Properties**:
- Numerically more stable than log(softmax(x))
- Used with NLLLoss

### Softmin

```python
class Softmin(Module):
    def __init__(self, dim: int = None)
```

**Formula**: `Softmin(x_i) = exp(-x_i) / Σ_j exp(-x_j)`

---

## Threshold

```python
class Threshold(Module):
    def __init__(self, threshold: float, value: float, inplace: bool = False)
```

**Formula**:
```
Threshold(x) = x if x > threshold else value
```

---

## Functional API

All activations are available as functions:

```python
import torch.nn.functional as F

# Standard
F.relu(x, inplace=False)
F.sigmoid(x)
F.tanh(x)

# ReLU variants
F.leaky_relu(x, negative_slope=0.01)
F.prelu(x, weight)
F.relu6(x)
F.elu(x, alpha=1.0)
F.celu(x, alpha=1.0)
F.selu(x)
F.rrelu(x, lower=1/8, upper=1/3, training=False)

# Modern
F.gelu(x, approximate='none')
F.silu(x)
F.mish(x)

# Hard approximations
F.hardtanh(x, min_val=-1.0, max_val=1.0)
F.hardsigmoid(x)
F.hardswish(x)

# Shrinkage
F.softshrink(x, lambd=0.5)
F.hardshrink(x, lambd=0.5)
F.tanhshrink(x)

# Smooth
F.softplus(x, beta=1, threshold=20)
F.softsign(x)

# Gating
F.glu(x, dim=-1)

# Normalization
F.softmax(x, dim=None)
F.log_softmax(x, dim=None)
F.softmin(x, dim=None)

# Threshold
F.threshold(x, threshold, value)
```

---

## Comparison Table

| Activation | Formula | Range | Gradient at 0 | Use Case |
|------------|---------|-------|---------------|----------|
| ReLU | max(0, x) | [0, ∞) | 0.5 | Default for CNNs |
| LeakyReLU | max(αx, x) | (-∞, ∞) | ~1 | Avoid dying ReLU |
| ELU | x or α(eˣ-1) | (-α, ∞) | 1 | Faster convergence |
| SELU | scaled ELU | fixed | ~1 | Self-normalizing |
| GELU | x·Φ(x) | (-∞, ∞) | 0.5 | Transformers |
| SiLU/Swish | x·σ(x) | (-∞, ∞) | 0.5 | EfficientNet |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | 0.25 | Binary output |
| Tanh | tanh(x) | (-1, 1) | 1 | RNNs |
| Softmax | exp/Σexp | (0, 1) | - | Multi-class |

---

## MLX Mapping

### Direct Mappings

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `nn.ReLU` | `mx.nn.ReLU` / `mx.maximum(x, 0)` | Direct |
| `nn.LeakyReLU` | `mx.nn.LeakyReLU` | Direct |
| `nn.GELU` | `mx.nn.GELU` | Direct |
| `nn.SiLU` | `mx.nn.SiLU` | Direct |
| `nn.Sigmoid` | `mx.sigmoid` | Direct |
| `nn.Tanh` | `mx.tanh` | Direct |
| `nn.Softmax` | `mx.softmax` | Direct |

### Implementation Required

```python
import mlx.core as mx
import mlx.nn as nn

class ELU(nn.Module):
    """ELU activation for MLX."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x):
        return mx.where(x > 0, x, self.alpha * (mx.exp(x) - 1))

class SELU(nn.Module):
    """SELU activation for MLX."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * mx.where(
            x > 0, x, self.alpha * (mx.exp(x) - 1)
        )

class Mish(nn.Module):
    """Mish activation for MLX."""
    def __call__(self, x):
        return x * mx.tanh(mx.log(1 + mx.exp(x)))

class Hardswish(nn.Module):
    """Hardswish activation for MLX."""
    def __call__(self, x):
        return x * mx.clip(x + 3, 0, 6) / 6

class GLU(nn.Module):
    """GLU activation for MLX."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def __call__(self, x):
        a, b = mx.split(x, 2, axis=self.dim)
        return a * mx.sigmoid(b)

class PReLU(nn.Module):
    """PReLU activation for MLX."""
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = mx.full((num_parameters,), init)

    def __call__(self, x):
        return mx.maximum(0, x) + self.weight * mx.minimum(0, x)
```

---

## Usage Recommendations

| Architecture | Recommended Activation |
|--------------|----------------------|
| CNNs (general) | ReLU, LeakyReLU |
| Transformers | GELU |
| MobileNets | ReLU6, Hardswish |
| Self-normalizing | SELU |
| LLMs | SiLU/Swish, GELU |
| Object Detection | Mish, LeakyReLU |
| Binary Classification | Sigmoid (output) |
| Multi-class | Softmax (output) |

---

## Implementation Files

- `torch/nn/modules/activation.py` - Module classes
- `torch/nn/functional.py` - Functional implementations
- `aten/src/ATen/native/Activation.cpp` - CPU kernels
- `aten/src/ATen/native/cuda/Activation.cu` - CUDA kernels

**Key Line Ranges**:
- ReLU: 104-150
- Sigmoid: 337-361
- GELU: 779-820
- SiLU: 435-482
- Softmax: 1017-1085
