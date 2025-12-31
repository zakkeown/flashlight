# PyTorch Weight Initialization (torch.nn.init)

## Purpose

This document provides a comprehensive analysis of PyTorch's weight initialization module (`torch.nn.init`). Proper weight initialization is critical for:

1. Avoiding vanishing/exploding gradients during training
2. Enabling faster convergence
3. Achieving stable training dynamics
4. Ensuring reproducible model behavior

**Source**: [torch/nn/init.py](../../reference/pytorch/torch/nn/init.py)

## Architecture Overview

### Initialization Philosophy

Weight initialization in neural networks aims to:
- Maintain activation variance across layers (forward pass)
- Maintain gradient variance across layers (backward pass)
- Prevent saturation of activation functions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Initialization Categories                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Basic Fills     │  │ Variance-Aware   │  │  Advanced     │  │
│  │                  │  │                  │  │               │  │
│  │  • uniform_      │  │  • xavier_*      │  │  • orthogonal_│  │
│  │  • normal_       │  │  • kaiming_*     │  │  • sparse_    │  │
│  │  • constant_     │  │                  │  │  • dirac_     │  │
│  │  • ones_/zeros_  │  │                  │  │               │  │
│  │  • eye_          │  │                  │  │               │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Helper Functions                                        │   │
│  │  • calculate_gain(nonlinearity)                          │   │
│  │  • _calculate_fan_in_and_fan_out(tensor)                 │   │
│  │  • _calculate_correct_fan(tensor, mode)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Functions

### 1. Basic Fill Operations

These functions provide simple value assignment to tensors.

#### uniform_(tensor, a=0.0, b=1.0)

Fills tensor with values from uniform distribution U(a, b).

```python
def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Fill the input Tensor with values drawn from U(a, b)."""
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=generator)
```

**Usage**:
```python
w = torch.empty(3, 5)
nn.init.uniform_(w, a=-0.5, b=0.5)
```

#### normal_(tensor, mean=0.0, std=1.0)

Fills tensor with values from normal distribution N(mean, std²).

```python
def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Fill the input Tensor with values drawn from N(mean, std²)."""
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)
```

**Usage**:
```python
w = torch.empty(3, 5)
nn.init.normal_(w, mean=0.0, std=0.02)
```

#### trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

Fills tensor with values from a truncated normal distribution. Values outside [a, b] are redrawn.

```python
def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> Tensor:
```

**Algorithm**: Uses inverse CDF transform:
1. Compute CDF bounds: l = Φ((a - mean)/std), u = Φ((b - mean)/std)
2. Sample uniformly in [2l-1, 2u-1]
3. Apply inverse error function (erfinv)
4. Scale and shift to target distribution
5. Clamp to [a, b]

**Usage** (common in Vision Transformers):
```python
# ViT-style initialization
w = torch.empty(768, 768)
nn.init.trunc_normal_(w, std=0.02)
```

#### constant_(tensor, val)

Fills tensor with a constant value.

```python
def constant_(tensor: Tensor, val: float) -> Tensor:
    with torch.no_grad():
        return tensor.fill_(val)
```

#### ones_(tensor) / zeros_(tensor)

Convenience functions for filling with 1s or 0s.

```python
def ones_(tensor: Tensor) -> Tensor:
    return _no_grad_fill_(tensor, 1.0)

def zeros_(tensor: Tensor) -> Tensor:
    return tensor.zero_()
```

#### eye_(tensor)

Fills a 2D tensor with the identity matrix.

```python
def eye_(tensor: Tensor) -> Tensor:
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    with torch.no_grad():
        torch.eye(*tensor.shape, out=tensor, requires_grad=tensor.requires_grad)
    return tensor
```

**Usage** (identity initialization for residual connections):
```python
w = torch.empty(512, 512)
nn.init.eye_(w)  # Start as identity transform
```

---

### 2. Variance-Aware Initialization

These methods calculate initialization bounds based on layer dimensions to maintain signal variance.

#### Fan Calculation

**Key helper function**:
```python
def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """
    For a tensor of shape [out_features, in_features, *kernel_size]:
    - fan_in = in_features * prod(kernel_size)
    - fan_out = out_features * prod(kernel_size)
    """
    num_input_fmaps = tensor.size(1)   # in_features
    num_output_fmaps = tensor.size(0)  # out_features
    receptive_field_size = 1
    for s in tensor.shape[2:]:
        receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out
```

**Example calculations**:
```
Linear(512, 256):     shape [256, 512]    → fan_in=512, fan_out=256
Conv2d(64, 128, 3):   shape [128, 64, 3, 3] → fan_in=576, fan_out=1152
```

#### calculate_gain(nonlinearity, param=None)

Returns the recommended gain value for different activation functions.

| Nonlinearity | Gain |
|--------------|------|
| Linear/Conv/Sigmoid | 1 |
| Tanh | 5/3 ≈ 1.667 |
| ReLU | √2 ≈ 1.414 |
| LeakyReLU(α) | √(2 / (1 + α²)) |
| SELU | 3/4 = 0.75 |

```python
def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    if nonlinearity in ["linear", "conv1d", "conv2d", ...] or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        negative_slope = param if param else 0.01
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
```

---

### 3. Xavier (Glorot) Initialization

**Paper**: "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)

**Goal**: Maintain variance of activations and gradients across layers for sigmoid/tanh activations.

**Derivation**: For variance preservation:
- Var(output) = Var(input) requires: Var(W) = 2 / (fan_in + fan_out)

#### xavier_uniform_(tensor, gain=1.0)

Samples from U(-a, a) where:
```
a = gain × √(6 / (fan_in + fan_out))
```

```python
def xavier_uniform_(tensor: Tensor, gain: float = 1.0, generator=None) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Uniform bounds from std
    return _no_grad_uniform_(tensor, -a, a, generator)
```

#### xavier_normal_(tensor, gain=1.0)

Samples from N(0, std²) where:
```
std = gain × √(2 / (fan_in + fan_out))
```

```python
def xavier_normal_(tensor: Tensor, gain: float = 1.0, generator=None) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0.0, std, generator)
```

**Usage**:
```python
# For layers followed by tanh
w = torch.empty(512, 512)
nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('tanh'))
```

---

### 4. Kaiming (He) Initialization

**Paper**: "Delving deep into rectifiers" (He et al., 2015)

**Goal**: Maintain variance for ReLU/LeakyReLU networks where half of activations are zeroed.

**Derivation**: ReLU zeros negative values, halving the variance. Compensation:
- fan_in mode: Var(W) = 2 / fan_in
- fan_out mode: Var(W) = 2 / fan_out

#### kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

Samples from U(-bound, bound) where:
```
bound = gain × √(3 / fan_mode)
gain = √(2 / (1 + a²))  # for leaky_relu with negative_slope=a
```

```python
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,                      # LeakyReLU negative slope
    mode: str = "fan_in",              # 'fan_in' or 'fan_out'
    nonlinearity: str = "leaky_relu",  # Activation function
    generator=None,
) -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)
```

#### kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

Samples from N(0, std²) where:
```
std = gain / √fan_mode
```

```python
def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator=None,
) -> Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)
```

**Mode Selection**:
- `fan_in`: Preserves variance in forward pass (default, recommended for inference)
- `fan_out`: Preserves variance in backward pass (useful for training stability)

**Usage**:
```python
# For Conv2d followed by ReLU
conv = nn.Conv2d(64, 128, 3)
nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(conv.bias)

# For LeakyReLU with negative_slope=0.2
nn.init.kaiming_normal_(conv.weight, a=0.2, nonlinearity='leaky_relu')
```

**Important Note**: Fan calculations assume weight is used as `x @ w.T`. If using `x @ w`, pass transposed weight.

---

### 5. Advanced Initialization Methods

#### orthogonal_(tensor, gain=1.0)

**Paper**: "Exact solutions to the nonlinear dynamics of learning" (Saxe et al., 2013)

Creates (semi-)orthogonal matrices via QR decomposition.

**Algorithm**:
1. Generate random Gaussian matrix
2. Compute QR factorization
3. Adjust signs to ensure uniform distribution
4. Scale by gain

```python
def orthogonal_(tensor: Tensor, gain: float = 1, generator=None) -> Tensor:
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new_empty((rows, cols)).normal_(0, 1, generator=generator)

    if rows < cols:
        flattened.t_()

    # QR decomposition
    q, r = torch.linalg.qr(flattened)
    # Make Q uniform (Haar measure)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor
```

**Properties**:
- Preserves norm of input vectors exactly
- Prevents gradient explosion/vanishing in very deep networks
- Useful for RNNs and very deep architectures

**Usage**:
```python
# For RNN hidden-to-hidden weights
rnn_hh = torch.empty(256, 256)
nn.init.orthogonal_(rnn_hh)
```

#### sparse_(tensor, sparsity, std=0.01)

**Paper**: "Deep learning via Hessian-free optimization" (Martens, 2010)

Creates sparse weight matrices with a fraction of zeros per column.

```python
def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01, generator=None) -> Tensor:
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = math.ceil(sparsity * rows)

    with torch.no_grad():
        tensor.normal_(0, std, generator=generator)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor
```

**Usage**:
```python
# 90% of each column is zero
w = torch.empty(1000, 500)
nn.init.sparse_(w, sparsity=0.9, std=0.01)
```

#### dirac_(tensor, groups=1)

Initializes {3,4,5}D convolutional tensors as Dirac delta functions, preserving input channels.

```python
def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """
    For Conv2d: tensor[d, d, k//2, k//2] = 1 (identity at center)
    All other weights = 0
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    with torch.no_grad():
        tensor.zero_()
        for g in range(groups):
            for d in range(min_dim):
                # Place 1 at center of kernel
                if dimensions == 4:  # Conv2d
                    tensor[g * out_chans_per_grp + d, d,
                           tensor.size(2) // 2, tensor.size(3) // 2] = 1
```

**Usage** (identity convolution for residual learning):
```python
# Identity initialization for skip connections
conv = nn.Conv2d(64, 64, 3, padding=1)
nn.init.dirac_(conv.weight)
nn.init.zeros_(conv.bias)
```

---

## Common Initialization Patterns

### Pattern 1: CNN with ReLU

```python
def init_cnn(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
```

### Pattern 2: Transformer / Vision Transformer

```python
def init_transformer(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
```

### Pattern 3: RNN / LSTM

```python
def init_rnn(model):
    for name, param in model.named_parameters():
        if 'weight_ih' in name:  # Input-hidden weights
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:  # Hidden-hidden weights
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # Set forget gate bias to 1 for LSTM
            if 'bias_hh' in name:
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
```

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `nn.init.uniform_(t, a, b)` | `mx.random.uniform(low=a, high=b, shape=t.shape)` |
| `nn.init.normal_(t, mean, std)` | `mx.random.normal(shape=t.shape) * std + mean` |
| `nn.init.zeros_(t)` | `mx.zeros(t.shape)` |
| `nn.init.ones_(t)` | `mx.ones(t.shape)` |
| `nn.init.constant_(t, val)` | `mx.full(t.shape, val)` |
| `nn.init.eye_(t)` | `mx.eye(t.shape[0], t.shape[1])` |

### Variance-Aware Initialization in MLX

MLX doesn't have built-in Xavier/Kaiming, but they're simple to implement:

```python
import mlx.core as mx
import math

def calculate_fan(shape):
    """Calculate fan_in and fan_out for a weight tensor."""
    if len(shape) < 2:
        raise ValueError("Fan calculation requires at least 2D tensor")
    fan_in = shape[1]
    fan_out = shape[0]
    if len(shape) > 2:
        receptive_field = math.prod(shape[2:])
        fan_in *= receptive_field
        fan_out *= receptive_field
    return fan_in, fan_out

def xavier_uniform(shape, gain=1.0):
    fan_in, fan_out = calculate_fan(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return mx.random.uniform(low=-a, high=a, shape=shape)

def xavier_normal(shape, gain=1.0):
    fan_in, fan_out = calculate_fan(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return mx.random.normal(shape=shape) * std

def kaiming_uniform(shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan_in, fan_out = calculate_fan(shape)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = math.sqrt(2.0 / (1 + a**2)) if nonlinearity == 'leaky_relu' else math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return mx.random.uniform(low=-bound, high=bound, shape=shape)

def kaiming_normal(shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan_in, fan_out = calculate_fan(shape)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = math.sqrt(2.0 / (1 + a**2)) if nonlinearity == 'leaky_relu' else math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    return mx.random.normal(shape=shape) * std

def orthogonal(shape, gain=1.0):
    """Orthogonal initialization via QR decomposition."""
    rows, cols = shape[0], math.prod(shape[1:])
    if rows < cols:
        flat = mx.random.normal(shape=(cols, rows))
    else:
        flat = mx.random.normal(shape=(rows, cols))

    q, r = mx.linalg.qr(flat)
    d = mx.diag(r)
    ph = mx.sign(d)
    q = q * ph

    if rows < cols:
        q = q.T

    return (q * gain).reshape(shape)

def trunc_normal(shape, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialization."""
    # Simplified rejection sampling approach
    samples = mx.random.normal(shape=shape) * std + mean
    # Clamp to bounds (MLX approach)
    return mx.clip(samples, a, b)
```

### MLX Module Initialization Pattern

```python
import mlx.nn as nn

class MLXModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # MLX Linear uses Kaiming-like initialization by default
        self.linear = nn.Linear(in_features, out_features)

        # Custom initialization
        self.linear.weight = kaiming_normal(
            (out_features, in_features),
            nonlinearity='relu'
        )
        self.linear.bias = mx.zeros((out_features,))
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **In-place ops** | `init.uniform_(tensor)` modifies in-place | MLX returns new array |
| **Generator** | Explicit `torch.Generator` | `mx.random.seed()` global |
| **Default init** | Varies by layer type | Simpler defaults |
| **QR decomposition** | `torch.linalg.qr` | `mx.linalg.qr` |

---

## Summary

### When to Use Each Method

| Method | Use Case |
|--------|----------|
| **xavier_uniform/normal** | Sigmoid, Tanh, or linear activations |
| **kaiming_uniform/normal** | ReLU, LeakyReLU, PReLU |
| **orthogonal** | RNN hidden weights, very deep networks |
| **trunc_normal** | Transformers, Vision Transformers |
| **sparse** | Sparse networks, regularization |
| **dirac** | Skip connections in CNNs |

### Quick Reference Table

| Function | Distribution | Formula |
|----------|--------------|---------|
| `uniform_(a,b)` | U(a,b) | — |
| `normal_(μ,σ)` | N(μ,σ²) | — |
| `xavier_uniform` | U(-a,a) | a = gain × √(6/(fan_in+fan_out)) |
| `xavier_normal` | N(0,σ²) | σ = gain × √(2/(fan_in+fan_out)) |
| `kaiming_uniform` | U(-b,b) | b = gain × √(3/fan) |
| `kaiming_normal` | N(0,σ²) | σ = gain / √fan |
