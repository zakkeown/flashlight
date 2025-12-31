# PyTorch Linear Layers

## Purpose

This document provides comprehensive documentation of PyTorch's linear (fully-connected) modules. Linear layers are fundamental for:

1. Dense neural network layers
2. Classification heads
3. Projection layers in transformers
4. Feature transformation
5. Output layers

**Source**: [torch/nn/modules/linear.py](../../reference/pytorch/torch/nn/modules/linear.py)

## Architecture Overview

### Linear Module Hierarchy

```
                    Module
          ____________|____________
         |           |            |
      Linear      Bilinear     Identity
         |
    LazyLinear
```

### Module Summary

| Module | Formula | Parameters | Use Case |
|--------|---------|------------|----------|
| **Linear** | `y = xW^T + b` | W, b | Standard dense layer |
| **Bilinear** | `y = x1^T A x2 + b` | A, b | Interaction modeling |
| **Identity** | `y = x` | None | Placeholder, skip connections |
| **LazyLinear** | Same as Linear | Deferred | Dynamic input size |

---

## 1. Linear

Applies an affine linear transformation: `y = xW^T + b`

### Formula

```
output = input @ weight.T + bias
```

Where:
- `input`: `(*, in_features)`
- `weight`: `(out_features, in_features)`
- `bias`: `(out_features,)`
- `output`: `(*, out_features)`

### Constructor

```python
nn.Linear(
    in_features: int,      # Size of input features
    out_features: int,     # Size of output features
    bias: bool = True,     # Include learnable bias
    device=None,
    dtype=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | int | required | Input dimension (H_in) |
| `out_features` | int | required | Output dimension (H_out) |
| `bias` | bool | True | If True, adds learnable bias |

### Attributes

```python
layer.weight  # Tensor of shape (out_features, in_features)
layer.bias    # Tensor of shape (out_features,) or None
layer.in_features   # int
layer.out_features  # int
```

### Shape

- **Input**: `(*, H_in)` where `*` means any number of dimensions
- **Output**: `(*, H_out)` where all but last dimension are unchanged

### Weight Initialization

```python
def reset_parameters(self) -> None:
    # Kaiming uniform with a=sqrt(5) is equivalent to:
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
```

### Examples

```python
# Basic linear layer
linear = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = linear(input)  # (128, 30)

# Without bias
linear_no_bias = nn.Linear(20, 30, bias=False)

# Multi-dimensional input
input_3d = torch.randn(10, 20, 30)  # (batch, seq, features)
linear = nn.Linear(30, 40)
output = linear(input_3d)  # (10, 20, 40)

# Classification head
classifier = nn.Linear(512, 10)  # 512 features -> 10 classes
logits = classifier(features)
```

### Common Patterns

**MLP (Multi-Layer Perceptron)**:
```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**Transformer FFN**:
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
```

---

## 2. Bilinear

Applies a bilinear transformation: `y = x1^T A x2 + b`

Takes two inputs and computes their bilinear interaction.

### Formula

```
output[k] = sum_i sum_j (input1[i] * weight[k, i, j] * input2[j]) + bias[k]
```

Or in matrix form:
```
output = x1^T @ weight @ x2 + bias
```

### Constructor

```python
nn.Bilinear(
    in1_features: int,     # Size of first input
    in2_features: int,     # Size of second input
    out_features: int,     # Size of output
    bias: bool = True,
    device=None,
    dtype=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in1_features` | int | Dimension of first input (must be > 0) |
| `in2_features` | int | Dimension of second input (must be > 0) |
| `out_features` | int | Dimension of output (must be > 0) |
| `bias` | bool | Include learnable bias |

### Attributes

```python
layer.weight  # Tensor of shape (out_features, in1_features, in2_features)
layer.bias    # Tensor of shape (out_features,) or None
```

### Shape

- **Input1**: `(*, H_in1)` where `H_in1 = in1_features`
- **Input2**: `(*, H_in2)` where `H_in2 = in2_features`
- **Output**: `(*, H_out)` where `H_out = out_features`

All dimensions except the last must match between inputs.

### Weight Initialization

```python
def reset_parameters(self) -> None:
    bound = 1 / math.sqrt(self.weight.size(1))  # 1/sqrt(in1_features)
    init.uniform_(self.weight, -bound, bound)
    if self.bias is not None:
        init.uniform_(self.bias, -bound, bound)
```

### Examples

```python
# Basic bilinear layer
bilinear = nn.Bilinear(20, 30, 40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = bilinear(input1, input2)  # (128, 40)

# Same size inputs
bilinear_same = nn.Bilinear(50, 50, 10)
x1 = torch.randn(32, 50)
x2 = torch.randn(32, 50)
output = bilinear_same(x1, x2)  # (32, 10)
```

### Use Cases

**Feature Interaction**:
```python
# Model interaction between user and item embeddings
class BilinearInteraction(nn.Module):
    def __init__(self, user_dim, item_dim, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(user_dim, item_dim, out_dim)

    def forward(self, user_emb, item_emb):
        return self.bilinear(user_emb, item_emb)
```

**Attention Score**:
```python
# Bilinear attention between query and key
class BilinearAttention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(query_dim, key_dim, 1, bias=False)

    def forward(self, query, keys):
        # query: (batch, query_dim)
        # keys: (batch, seq_len, key_dim)
        batch, seq_len, _ = keys.shape
        query = query.unsqueeze(1).expand(-1, seq_len, -1)
        scores = self.bilinear(query, keys).squeeze(-1)  # (batch, seq_len)
        return F.softmax(scores, dim=-1)
```

---

## 3. Identity

A placeholder module that returns its input unchanged.

### Constructor

```python
nn.Identity(*args, **kwargs)  # All arguments are ignored
```

### Shape

- **Input**: `(*)` - Any shape
- **Output**: `(*)` - Same as input

### Examples

```python
# Basic usage
identity = nn.Identity()
x = torch.randn(128, 20)
output = identity(x)  # Same as x

# Arguments are ignored (for compatibility)
identity = nn.Identity(54, unused_arg=0.1)
output = identity(x)  # Still works
```

### Use Cases

**Conditional Skip Connection**:
```python
class ConditionalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_projection=False):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        # Use Identity if no projection needed
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if use_projection else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))
```

**Placeholder in Sequential**:
```python
# Disable a layer without changing architecture
class FlexibleModel(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(50, 10)
        )
```

**Optional Normalization**:
```python
def build_norm(norm_type, dim):
    if norm_type == 'layer':
        return nn.LayerNorm(dim)
    elif norm_type == 'batch':
        return nn.BatchNorm1d(dim)
    else:
        return nn.Identity()  # No normalization
```

---

## 4. LazyLinear

A Linear module where `in_features` is inferred from the first input.

### Key Properties

- `in_features` determined at first forward pass
- Weights are `UninitializedParameter` until first forward
- Converts to regular `Linear` after initialization
- Useful for dynamic architectures

### Constructor

```python
nn.LazyLinear(
    out_features: int,     # Output dimension (required)
    bias: bool = True,
    device=None,
    dtype=None,
)
# Note: No in_features parameter!
```

### Examples

```python
# in_features will be inferred
lazy_linear = nn.LazyLinear(30)
print(lazy_linear.weight)  # UninitializedParameter

# First forward pass initializes weights
input = torch.randn(128, 20)
output = lazy_linear(input)  # (128, 30)

print(lazy_linear.in_features)  # 20 (inferred)
print(lazy_linear.weight.shape)  # (30, 20)
```

### Limitations

1. **Cannot serialize before initialization**: Save/load requires initialization
2. **TorchScript compatibility**: May have issues with scripting
3. **Multiple calls**: Must have consistent input sizes after first call

### Use Cases

**Dynamic Feature Extraction**:
```python
class FlexibleClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = create_backbone()  # Unknown output size
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        features = self.backbone(x)  # Dynamic size
        return self.classifier(features)
```

**Prototyping**:
```python
# Quick prototyping without calculating sizes
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LazyLinear(256),  # Input size computed automatically
    nn.ReLU(),
    nn.LazyLinear(10)
)

# Initialize by running dummy input
dummy = torch.randn(1, 3, 224, 224)
model(dummy)  # Now all layers are initialized
```

---

## 5. Parameter Counts

### Linear Layer

```
Parameters = out_features * in_features + out_features  (with bias)
           = out_features * in_features                 (without bias)

Example: Linear(512, 256)
         = 256 * 512 + 256 = 131,328 parameters
```

### Bilinear Layer

```
Parameters = out_features * in1_features * in2_features + out_features  (with bias)
           = out_features * in1_features * in2_features                 (without bias)

Example: Bilinear(20, 30, 40)
         = 40 * 20 * 30 + 40 = 24,040 parameters
```

---

## MLX Mapping

### Direct Mapping

```python
# PyTorch                    # MLX
nn.Linear(in, out)           mlx.nn.Linear(in, out)
nn.Identity()                (return x directly)
nn.Bilinear(...)             # Manual implementation needed
nn.LazyLinear(...)           # Manual implementation needed
```

### MLX Linear

```python
import mlx.core as mx
import mlx.nn as nn

# Built-in MLX Linear
linear = nn.Linear(20, 30)
x = mx.random.normal(shape=(128, 20))
output = linear(x)  # (128, 30)
```

### MLX Bilinear (Manual Implementation)

```python
import mlx.core as mx
import mlx.nn as nn
import math

class Bilinear(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        # Initialize weight: (out, in1, in2)
        bound = 1 / math.sqrt(in1_features)
        self.weight = mx.random.uniform(
            low=-bound, high=bound,
            shape=(out_features, in1_features, in2_features)
        )

        if bias:
            self.bias = mx.random.uniform(
                low=-bound, high=bound,
                shape=(out_features,)
            )
        else:
            self.bias = None

    def __call__(self, x1, x2):
        # x1: (*, in1), x2: (*, in2)
        # einsum: batch dimensions preserved
        output = mx.einsum('...i,oij,...j->...o', x1, self.weight, x2)
        if self.bias is not None:
            output = output + self.bias
        return output
```

### MLX Identity

```python
class Identity(nn.Module):
    def __call__(self, x):
        return x
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Weight shape | `(out, in)` | `(out, in)` (same) |
| LazyLinear | Built-in | Manual |
| Bilinear | Built-in | Manual |
| Initialization | Kaiming uniform | May differ |

---

## Common Patterns

### Tied Weights (Weight Sharing)

```python
# Share weights between encoder and decoder
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_dim, bias=True)
        # Tie weights: decoder.weight = encoder.weight.T
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded
```

### Low-Rank Factorization

```python
# Reduce parameters with low-rank approximation
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # Instead of (out, in), use (out, rank) @ (rank, in)
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.up(self.down(x))

# Parameters: in*rank + rank*out vs in*out
# For in=1024, out=1024, rank=64:
# Low-rank: 1024*64 + 64*1024 = 131,072
# Full: 1024*1024 = 1,048,576 (8x more)
```

### Mixture of Experts

```python
class MoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1)
        return (expert_outputs * gate_scores.unsqueeze(-2)).sum(-1)
```

---

## Summary

### Quick Reference

| Module | Input | Output | Parameters | Use Case |
|--------|-------|--------|------------|----------|
| Linear | `(*, in)` | `(*, out)` | `out*in + out` | Dense layers |
| Bilinear | `(*, in1)`, `(*, in2)` | `(*, out)` | `out*in1*in2 + out` | Feature interaction |
| Identity | `(*)` | `(*)` | 0 | Placeholder |
| LazyLinear | `(*, in)` | `(*, out)` | Deferred | Dynamic input |

### Memory/Compute Trade-offs

```
Linear(1024, 1024):     ~4 MB parameters, ~2M FLOPs per sample
Linear(1024, 4096):     ~16 MB parameters, ~8M FLOPs per sample
Bilinear(64, 64, 64):   ~1 MB parameters, ~0.5M FLOPs per sample
```

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/linear.py`

**Functional API**:
- `F.linear(input, weight, bias=None)` - Linear transformation
- `F.bilinear(input1, input2, weight, bias=None)` - Bilinear transformation

**Internal**:
- `NonDynamicallyQuantizableLinear` - Used internally for quantization edge cases
