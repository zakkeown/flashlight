# PyTorch Dropout Layers

## Purpose

This document provides comprehensive documentation of PyTorch's dropout modules. Dropout is a regularization technique essential for:

1. Preventing overfitting during training
2. Breaking co-adaptation of neurons
3. Enabling model ensembling (at test time)
4. Self-normalizing networks (AlphaDropout)

**Source**: [torch/nn/modules/dropout.py](../../reference/pytorch/torch/nn/modules/dropout.py)

**Reference Paper**: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (2014)

## Architecture Overview

### Dropout Module Hierarchy

```
                          Module
                             |
                       _DropoutNd (base)
          ___________________|___________________
         |          |          |          |          |          |
      Dropout  Dropout1d  Dropout2d  Dropout3d  AlphaDropout  FeatureAlphaDropout
    (element)  (channel)  (channel)  (channel)    (SELU)       (SELU + channel)
```

### Dropout Types Summary

| Type | Drops | Use With | Input Shape |
|------|-------|----------|-------------|
| `Dropout` | Elements | Any layer | `(*)` |
| `Dropout1d` | Channels | Conv1d | `(N, C, L)` |
| `Dropout2d` | Channels | Conv2d | `(N, C, H, W)` |
| `Dropout3d` | Channels | Conv3d | `(N, C, D, H, W)` |
| `AlphaDropout` | Elements | SELU | `(*)` |
| `FeatureAlphaDropout` | Channels | SELU + Conv | `(N, C, ...)` |

---

## Key Concepts

### Inverted Dropout

PyTorch uses **inverted dropout**, which scales outputs during training (not inference):

```
Training:   output = input * mask / (1 - p)
Inference:  output = input  (identity function)
```

This ensures that expected values match between training and inference without requiring scaling at test time.

### Element vs Channel Dropout

```
Element-wise (Dropout):              Channel-wise (Dropout2d):
┌─────────────────────┐              ┌─────────────────────┐
│ [x, 0, x, 0, x]     │              │ [x, x, x, x, x]     │ <- Channel 0 (kept)
│ [0, x, x, x, 0]     │              │ [0, 0, 0, 0, 0]     │ <- Channel 1 (dropped)
│ [x, x, 0, x, x]     │              │ [x, x, x, x, x]     │ <- Channel 2 (kept)
└─────────────────────┘              └─────────────────────┘
   Random per element                   Entire channel dropped
```

---

## 1. Base Class: _DropoutNd

```python
class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
```

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | float | 0.5 | Probability of an element being zeroed |
| `inplace` | bool | False | Modify input tensor in-place |

---

## 2. Dropout

Standard element-wise dropout. Zeros random elements with probability `p`.

### Formula

```
Training:   y[i] = x[i] / (1 - p)  if mask[i] = 1
            y[i] = 0               if mask[i] = 0
            where mask ~ Bernoulli(1 - p)

Inference:  y = x  (identity)
```

### Constructor

```python
nn.Dropout(
    p: float = 0.5,      # Drop probability
    inplace: bool = False
)
```

### Shape

- **Input**: `(*)` - Any shape
- **Output**: `(*)` - Same shape as input

### Example

```python
# Standard dropout
dropout = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = dropout(input)  # 20% of elements zeroed, rest scaled by 1/(1-0.2)=1.25

# In-place dropout (saves memory)
dropout_inplace = nn.Dropout(p=0.5, inplace=True)
output = dropout_inplace(input)  # Modifies input directly

# Training vs Evaluation
model = nn.Sequential(nn.Linear(10, 10), nn.Dropout(0.5))
model.train()   # Dropout active
model.eval()    # Dropout disabled (identity function)
```

### Common Positions in Networks

```python
# After activation (most common)
nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

# In Transformer layers
class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        self.attn = nn.MultiheadAttention(d_model, 8, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x, x, x)[0])
        x = x + self.dropout(self.ffn(x))
        return x
```

---

## 3. Dropout1d

Drops entire channels (1D feature maps). Better for spatially correlated features.

### Why Channel Dropout?

When adjacent values in a feature map are correlated (common in conv layers), element-wise dropout doesn't regularize effectively. Dropping entire channels forces the network to learn more robust features.

### Constructor

```python
nn.Dropout1d(
    p: float = 0.5,
    inplace: bool = False
)
```

### Shape

- **Input**: `(N, C, L)` or `(C, L)` (unbatched)
- **Output**: Same as input

### Example

```python
# After Conv1d
model = nn.Sequential(
    nn.Conv1d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Dropout1d(p=0.2)  # Drops 20% of channels
)

input = torch.randn(8, 16, 100)  # (batch, channels, length)
output = model(input)

# What happens internally:
# 1. Generate mask of shape (8, 32, 1) with 80% ones
# 2. Broadcast mask to (8, 32, 100)
# 3. Multiply input by mask and scale
```

---

## 4. Dropout2d

Drops entire channels (2D feature maps). Standard for CNNs.

### Constructor

```python
nn.Dropout2d(
    p: float = 0.5,
    inplace: bool = False
)
```

### Shape

- **Input**: `(N, C, H, W)` or `(N, C, L)` (see warning below)
- **Output**: Same as input

### Warning

```python
# Historical behavior: 3D inputs treated as (N, C, L), not (C, H, W)
# For unbatched 3D inputs, use Dropout1d explicitly
```

### Example

```python
# CNN with spatial dropout
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout = nn.Dropout2d(0.25)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)  # Drop entire feature maps
        x = F.relu(self.conv2(x))
        return x

# Spatial dropout in practice:
dropout2d = nn.Dropout2d(p=0.3)
features = torch.randn(4, 64, 28, 28)  # (batch, channels, H, W)
output = dropout2d(features)
# ~19 channels (30%) will be entirely zeroed per sample
```

### Visual Example

```
Input Feature Maps (4 channels):     After Dropout2d (p=0.5):
┌────┐ ┌────┐ ┌────┐ ┌────┐         ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ A  │ │ B  │ │ C  │ │ D  │    →    │ A' │ │ 0  │ │ C' │ │ 0  │
└────┘ └────┘ └────┘ └────┘         └────┘ └────┘ └────┘ └────┘
                                    (scaled)  (dropped)  (scaled)  (dropped)
```

---

## 5. Dropout3d

Drops entire channels (3D feature maps). For video/volumetric data.

### Constructor

```python
nn.Dropout3d(
    p: float = 0.5,
    inplace: bool = False
)
```

### Shape

- **Input**: `(N, C, D, H, W)` or `(C, D, H, W)` (unbatched)
- **Output**: Same as input

### Example

```python
# 3D CNN for video processing
class Video3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):  # x: (batch, channels, frames, H, W)
        x = F.relu(self.conv(x))
        x = self.dropout(x)  # Drop entire 3D feature volumes
        return x

video = torch.randn(2, 3, 16, 112, 112)  # (batch, C, T, H, W)
model = Video3DCNN()
output = model(video)
```

---

## 6. AlphaDropout

Special dropout for Self-Normalizing Neural Networks (SNNs) with SELU activation.

### Key Properties

- Maintains zero mean and unit variance of inputs
- Sets dropped values to the SELU negative saturation value (not zero)
- Must be used with SELU activation

### Formula

```
α ≈ 1.6733     # SELU scale parameter
λ ≈ 1.0507     # SELU scaling factor

# Saturation value for dropped elements
a' = -λα = -1.7581

# Alpha dropout transformation:
Training:   y = (x * mask + a' * (1 - mask) - a) / b
            where a, b are computed to maintain zero mean, unit variance

Inference:  y = x
```

### Constructor

```python
nn.AlphaDropout(
    p: float = 0.5,
    inplace: bool = False
)
```

### Shape

- **Input**: `(*)` - Any shape
- **Output**: `(*)` - Same as input

### Example

```python
# Self-Normalizing Neural Network
class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.AlphaDropout(p=0.1)

        # LeCun normal initialization for SELU
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = self.dropout(x)
        x = F.selu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Important: SELU + AlphaDropout maintain self-normalization
# Without AlphaDropout, SELU's self-normalizing property is broken
```

### When to Use

- Always pair with SELU activation
- Deep feedforward networks where batch normalization isn't suitable
- When you want normalization without explicit BatchNorm layers

---

## 7. FeatureAlphaDropout

Channel-wise AlphaDropout for convolutional SNNs.

### Constructor

```python
nn.FeatureAlphaDropout(
    p: float = 0.5,
    inplace: bool = False
)
```

### Shape

- **Input**: `(N, C, D, H, W)` or `(C, D, H, W)` (unbatched)
- **Output**: Same as input

### Example

```python
# Self-normalizing CNN
class SN_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout = nn.FeatureAlphaDropout(p=0.2)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.dropout(x)  # Channel dropout maintaining self-normalization
        x = F.selu(self.conv2(x))
        return x
```

---

## Training vs Evaluation Mode

### Critical: Mode Affects Behavior

```python
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.Dropout(0.5)
)

# Training mode: Dropout active
model.train()
out_train = model(x)  # ~50% of values zeroed, rest scaled by 2

# Evaluation mode: Dropout disabled
model.eval()
out_eval = model(x)   # Identity function, no dropout

# Verify mode
print(model.training)  # True or False
```

### Common Pitfalls

```python
# WRONG: Forgetting to set eval mode for inference
model.train()
predictions = model(test_data)  # Dropout still active!

# CORRECT: Always set eval mode for inference
model.eval()
with torch.no_grad():
    predictions = model(test_data)

# Using context managers
with torch.inference_mode():
    model.eval()
    predictions = model(test_data)
```

---

## MLX Mapping

### Direct Mapping

```python
# PyTorch                      # MLX
nn.Dropout(p)                  mlx.nn.Dropout(p)
```

### MLX Dropout Implementation

```python
import mlx.core as mx
import mlx.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")
        self.p = p

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x

        # Generate Bernoulli mask
        mask = mx.random.bernoulli(1 - self.p, shape=x.shape)
        # Scale by 1/(1-p) for inverted dropout
        return x * mask / (1 - self.p)


class Dropout2d(nn.Module):
    """Channel-wise dropout for 2D feature maps."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x

        # x shape: (N, C, H, W) or (N, H, W, C) depending on convention
        # For channels-last (NHWC), generate mask of shape (N, 1, 1, C)
        if x.ndim == 4:
            # Assuming NHWC (MLX default)
            mask_shape = (x.shape[0], 1, 1, x.shape[3])
        elif x.ndim == 3:
            mask_shape = (x.shape[0], 1, x.shape[2])
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D")

        mask = mx.random.bernoulli(1 - self.p, shape=mask_shape)
        return x * mask / (1 - self.p)
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| `inplace` | Supported | Not needed (functional) |
| `AlphaDropout` | Built-in | Manual implementation |
| `FeatureAlphaDropout` | Built-in | Manual implementation |
| Training mode | `model.train()` / `model.eval()` | Module `.training` attribute |

---

## Common Patterns

### Dropout in Transformers

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Residual dropout
        self.dropout1 = nn.Dropout(dropout)  # FFN dropout
        self.dropout2 = nn.Dropout(dropout)  # Output dropout

    def forward(self, src):
        # Self-attention with residual
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # FFN with residual
        src2 = self.linear2(self.dropout1(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
```

### Variational Dropout (Locked Dropout)

Same mask used across time steps in RNNs:

```python
class LockedDropout(nn.Module):
    """Dropout with same mask across sequence dimension."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (seq_len, batch, features) or (batch, seq_len, features)
        if not self.training or self.p == 0:
            return x

        # Generate mask for single time step, broadcast across sequence
        if x.dim() == 3:
            # Assume (batch, seq_len, features) - mask shape (batch, 1, features)
            mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
            mask = mask / (1 - self.p)
            return x * mask.expand_as(x)
        return x
```

### DropPath (Stochastic Depth)

Used in Vision Transformers and EfficientNet:

```python
class DropPath(nn.Module):
    """Drop entire residual path with probability p."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x

        keep_prob = 1 - self.p
        # Shape: (batch_size, 1, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        return x.div(keep_prob) * random_tensor
```

---

## Summary

### Quick Reference

| Module | Drops | Use Case | Typical p |
|--------|-------|----------|-----------|
| `Dropout` | Elements | FC layers, general | 0.1-0.5 |
| `Dropout1d` | Channels | After Conv1d | 0.1-0.3 |
| `Dropout2d` | Channels | After Conv2d | 0.1-0.3 |
| `Dropout3d` | Channels | After Conv3d | 0.1-0.3 |
| `AlphaDropout` | Elements | With SELU | 0.05-0.1 |
| `FeatureAlphaDropout` | Channels | Conv + SELU | 0.05-0.1 |

### Typical Dropout Rates

| Application | Layer Type | Recommended p |
|-------------|------------|---------------|
| Vision (CNNs) | Conv layers | 0.1-0.25 |
| Vision (CNNs) | FC layers | 0.5 |
| NLP (Transformers) | Attention | 0.1 |
| NLP (Transformers) | FFN | 0.1 |
| Speech (RNNs) | Recurrent | 0.2-0.3 |
| Speech (RNNs) | FC | 0.5 |

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/dropout.py`

**Functional API**:
- `F.dropout(input, p, training, inplace)`
- `F.dropout1d(input, p, training, inplace)`
- `F.dropout2d(input, p, training, inplace)`
- `F.dropout3d(input, p, training, inplace)`
- `F.alpha_dropout(input, p, training, inplace)`
- `F.feature_alpha_dropout(input, p, training, inplace)`
