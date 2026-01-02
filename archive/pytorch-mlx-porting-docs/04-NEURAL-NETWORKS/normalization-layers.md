# PyTorch Normalization Layers

## Purpose

This document provides comprehensive documentation of PyTorch's normalization modules. Normalization is critical for:

1. Accelerating training convergence
2. Enabling higher learning rates
3. Reducing internal covariate shift
4. Regularization effects
5. Stabilizing deep network training

**Source Files**:
- [torch/nn/modules/normalization.py](../../reference/pytorch/torch/nn/modules/normalization.py) - LayerNorm, GroupNorm, RMSNorm
- [torch/nn/modules/batchnorm.py](../../reference/pytorch/torch/nn/modules/batchnorm.py) - BatchNorm1d/2d/3d, SyncBatchNorm
- [torch/nn/modules/instancenorm.py](../../reference/pytorch/torch/nn/modules/instancenorm.py) - InstanceNorm1d/2d/3d

## Architecture Overview

### Normalization Module Hierarchy

```
                                    Module
                                       |
                    ┌──────────────────┼──────────────────┐
                    |                  |                  |
                _NormBase         LayerNorm           RMSNorm
                    |                  |                  |
        ┌───────────┼───────────┐   GroupNorm        (standalone)
        |                       |
   _BatchNorm            _InstanceNorm
        |                       |
   ┌────┼────┐             ┌────┼────┐
   |    |    |             |    |    |
  BN1d BN2d BN3d         IN1d IN2d IN3d
                SyncBatchNorm
```

### Normalization Types Comparison

| Type | Normalizes Over | Use Case | Batch Dependency |
|------|----------------|----------|------------------|
| **BatchNorm** | (N, H, W) per channel | CNNs | Yes |
| **LayerNorm** | (C, H, W) per sample | Transformers, RNNs | No |
| **InstanceNorm** | (H, W) per sample & channel | Style transfer | No |
| **GroupNorm** | (H, W) per group of channels | Small batch CNNs | No |
| **RMSNorm** | Same as LayerNorm, no mean | LLMs (faster) | No |

### Visual Comparison

```
Input: (N, C, H, W) - Batch of feature maps

BatchNorm:              LayerNorm:             InstanceNorm:          GroupNorm:
Normalize across        Normalize across       Normalize across       Normalize across
N, H, W for each C      C, H, W for each N     H, W for each N, C     groups of C

┌───────────────┐       ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ N ┌─┬─┬─┐     │       │ N ┌─┬─┬─┐     │      │ N ┌─┬─┬─┐     │      │ N ┌─┬─┬─┐     │
│   │■│■│■│ C=0 │       │   │■│■│■│     │      │   │■│ │ │     │      │   │■│■│ │G=0  │
│   ├─┼─┼─┤     │       │   │■│■│■│     │      │   ├─┼─┼─┤     │      │   │■│■│ │     │
│   │■│■│■│     │       │   │■│■│■│     │      │   │ │■│ │     │      │   ├─┼─┼─┤     │
│   ├─┼─┼─┤     │       │   └─┴─┴─┘     │      │   ├─┼─┼─┤     │      │   │ │ │■│G=1  │
│   │■│■│■│     │       │   One norm    │      │   │ │ │■│     │      │   │ │ │■│     │
│   └─┴─┴─┘     │       │   per sample  │      │   └─┴─┴─┘     │      │   └─┴─┴─┘     │
│   All N same  │       │               │      │   Each (n,c)  │      │   Per group   │
└───────────────┘       └───────────────┘      └───────────────┘      └───────────────┘
```

---

## Common Formula

All normalization layers (except RMSNorm) follow:

```
y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
```

Where:
- `E[x]` = mean computed over normalization dimensions
- `Var[x]` = variance computed over normalization dimensions
- `eps` = small constant for numerical stability
- `gamma` = learnable scale (weight)
- `beta` = learnable shift (bias)

---

## 1. BatchNorm (BatchNorm1d, BatchNorm2d, BatchNorm3d)

Normalizes across the batch dimension and spatial dimensions, independently per channel.

### Key Properties

- **Training**: Uses mini-batch statistics, updates running statistics
- **Evaluation**: Uses running statistics (accumulated during training)
- **Batch dependency**: Requires reasonable batch size (typically >= 16)

### Constructor

```python
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_features` | int | required | C from input shape |
| `eps` | float | 1e-5 | Numerical stability |
| `momentum` | float | 0.1 | Running stats update factor |
| `affine` | bool | True | Learnable gamma, beta |
| `track_running_stats` | bool | True | Track running mean/var |

### Shape

| Module | Input Shape | Output Shape |
|--------|-------------|--------------|
| BatchNorm1d | `(N, C)` or `(N, C, L)` | Same as input |
| BatchNorm2d | `(N, C, H, W)` | Same as input |
| BatchNorm3d | `(N, C, D, H, W)` | Same as input |

### Running Statistics Update

```python
# Exponential moving average
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var = (1 - momentum) * running_var + momentum * batch_var

# Or cumulative moving average if momentum=None
running_mean = running_mean + (batch_mean - running_mean) / num_batches_tracked
```

### Example

```python
# CNN with BatchNorm
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)  # No bias needed
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Training vs Evaluation
model = ConvBlock(3, 64)
model.train()  # Uses batch statistics
model.eval()   # Uses running statistics

# Accessing running statistics
print(model.bn.running_mean)  # Shape: (64,)
print(model.bn.running_var)   # Shape: (64,)
```

### Weight Initialization

```python
def reset_parameters(self):
    self.running_mean.zero_()
    self.running_var.fill_(1)
    self.num_batches_tracked.zero_()
    if self.affine:
        init.ones_(self.weight)   # gamma = 1
        init.zeros_(self.bias)    # beta = 0
```

---

## 2. LayerNorm

Normalizes across the last D dimensions (typically channel and spatial), independently per sample.

### Key Properties

- **No batch dependency**: Works with batch_size=1
- **Same statistics for train/eval**: Uses input statistics always
- **Common in**: Transformers, RNNs, NLP models

### Formula

```
y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
```

Mean and variance computed over `normalized_shape` dimensions.

### Constructor

```python
nn.LayerNorm(
    normalized_shape,           # int or tuple: dimensions to normalize
    eps=1e-5,                   # Numerical stability
    elementwise_affine=True,    # Learnable per-element scale/bias
    bias=True,                  # Include bias (if elementwise_affine=True)
    device=None,
    dtype=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `normalized_shape` | int or tuple | Shape of dimensions to normalize over |
| `eps` | float | Added to denominator for stability |
| `elementwise_affine` | bool | If True, has learnable weight and bias |
| `bias` | bool | If False, no additive bias (when elementwise_affine=True) |

### Shape

- **Input**: `(N, *)` where `*` ends with `normalized_shape`
- **Output**: Same as input
- **Weight/Bias**: Shape = `normalized_shape`

### Examples

```python
# NLP: Normalize over embedding dimension
batch, seq_len, embed_dim = 32, 128, 512
x = torch.randn(batch, seq_len, embed_dim)
layer_norm = nn.LayerNorm(embed_dim)  # Normalize over last dim
output = layer_norm(x)  # (32, 128, 512)

# Vision: Normalize over channel and spatial
N, C, H, W = 32, 64, 56, 56
x = torch.randn(N, C, H, W)
layer_norm = nn.LayerNorm([C, H, W])  # Normalize over C, H, W
output = layer_norm(x)  # (32, 64, 56, 56)

# Transformer usage
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 8)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 3. GroupNorm

Divides channels into groups and normalizes within each group. Hybrid between LayerNorm and InstanceNorm.

### Key Properties

- **No batch dependency**: Works with any batch size
- **Flexible grouping**: Can emulate LayerNorm (1 group) or InstanceNorm (C groups)
- **Good for**: Small batch training, object detection, segmentation

### Formula

```
y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
```

Statistics computed over (H, W) for each group of channels.

### Constructor

```python
nn.GroupNorm(
    num_groups,      # Number of groups to divide channels into
    num_channels,    # Total number of channels (must be divisible by num_groups)
    eps=1e-5,
    affine=True,     # Learnable per-channel scale/bias
    device=None,
    dtype=None,
)
```

### Constraints

```python
# num_channels must be divisible by num_groups
assert num_channels % num_groups == 0
```

### Special Cases

| num_groups | Equivalent To |
|------------|---------------|
| 1 | LayerNorm (all channels in one group) |
| num_channels | InstanceNorm (each channel its own group) |
| 32 | Typical choice (from GroupNorm paper) |

### Shape

- **Input**: `(N, C, *)` where `C = num_channels`
- **Output**: Same as input
- **Weight/Bias**: Shape = `(num_channels,)`

### Example

```python
# 6 channels, 3 groups (2 channels per group)
x = torch.randn(20, 6, 10, 10)
gn = nn.GroupNorm(num_groups=3, num_channels=6)
output = gn(x)

# Equivalent to InstanceNorm
in_equiv = nn.GroupNorm(num_groups=6, num_channels=6)

# Equivalent to LayerNorm over channels
ln_equiv = nn.GroupNorm(num_groups=1, num_channels=6)

# ResNet with GroupNorm (replacing BatchNorm)
class ResBlockGN(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + x)
```

---

## 4. InstanceNorm (InstanceNorm1d, InstanceNorm2d, InstanceNorm3d)

Normalizes each sample and each channel independently. Common in style transfer.

### Key Properties

- **No batch dependency**
- **No cross-sample information**
- **Removes style information**: Useful for domain adaptation

### Constructor

```python
nn.InstanceNorm1d(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)
nn.InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)
nn.InstanceNorm3d(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)
```

**Note**: Default `affine=False` and `track_running_stats=False` (unlike BatchNorm).

### Shape

| Module | Input Shape | Normalized Over |
|--------|-------------|-----------------|
| InstanceNorm1d | `(N, C, L)` | L |
| InstanceNorm2d | `(N, C, H, W)` | H, W |
| InstanceNorm3d | `(N, C, D, H, W)` | D, H, W |

### Example

```python
# Style transfer network
class StyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        # ... more layers

    def forward(self, x):
        x = F.relu(self.in1(self.conv1(x)))
        return x

# InstanceNorm removes per-instance style while preserving content
input_image = torch.randn(1, 3, 256, 256)
in2d = nn.InstanceNorm2d(3)
normalized = in2d(input_image)
# Each channel of the single image is normalized to zero mean, unit variance
```

---

## 5. RMSNorm

Root Mean Square Layer Normalization - faster alternative to LayerNorm (no mean subtraction).

### Key Properties

- **Faster than LayerNorm**: Skips mean computation
- **Used in modern LLMs**: LLaMA, Mistral, etc.
- **Simpler gradient**: No mean centering

### Formula

```
y_i = x_i / RMS(x) * gamma_i

where RMS(x) = sqrt(eps + (1/n) * sum(x_i^2))
```

### Constructor

```python
nn.RMSNorm(
    normalized_shape,          # Dimensions to normalize
    eps=None,                  # Default: finfo(dtype).eps
    elementwise_affine=True,   # Learnable scale
    device=None,
    dtype=None,
)
```

**Note**: No bias parameter (only scale).

### Example

```python
# Modern LLM layer
class LLaMALayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 8)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

# Basic usage
rms_norm = nn.RMSNorm(512)
x = torch.randn(32, 128, 512)
output = rms_norm(x)
```

---

## 6. SyncBatchNorm

Synchronized Batch Normalization across multiple GPUs/processes.

### Key Properties

- **Computes statistics across all GPUs**: Larger effective batch size
- **Essential for distributed training**: When per-GPU batch is small
- **Requires**: Distributed training setup

### Constructor

```python
nn.SyncBatchNorm(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    process_group=None,  # Which processes to sync across
    device=None,
    dtype=None,
)
```

### Converting Model to SyncBatchNorm

```python
# Convert all BatchNorm layers to SyncBatchNorm
model = MyModel()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Use with DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model)
```

---

## 7. LocalResponseNorm

AlexNet-style local response normalization (largely obsolete).

### Formula

```
b_c = a_c * (k + (alpha/n) * sum_{c'=max(0,c-n/2)}^{min(N-1,c+n/2)} a_{c'}^2)^{-beta}
```

### Constructor

```python
nn.LocalResponseNorm(
    size,           # Number of neighboring channels
    alpha=1e-4,     # Multiplicative factor
    beta=0.75,      # Exponent
    k=1.0,          # Additive factor
)
```

### Example

```python
# Historical AlexNet style
lrn = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1)
x = torch.randn(32, 96, 55, 55)
output = lrn(x)
```

**Note**: Rarely used in modern architectures. BatchNorm is preferred.

---

## MLX Mapping

### Direct Mappings

```python
# PyTorch                    # MLX
nn.LayerNorm(dim)            mlx.nn.LayerNorm(dim)
nn.RMSNorm(dim)              mlx.nn.RMSNorm(dim)
nn.BatchNorm*                # Manual implementation needed
nn.GroupNorm                 # Manual implementation needed
nn.InstanceNorm*             # Manual implementation needed
```

### MLX LayerNorm

```python
import mlx.core as mx
import mlx.nn as nn

# Built-in MLX LayerNorm
layer_norm = nn.LayerNorm(512)
x = mx.random.normal(shape=(32, 128, 512))
output = layer_norm(x)
```

### MLX RMSNorm

```python
# Built-in MLX RMSNorm
rms_norm = nn.RMSNorm(512)
output = rms_norm(x)
```

### MLX BatchNorm (Manual Implementation)

```python
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x):
        # x shape: (N, H, W, C) in MLX (channels-last)
        if self.training:
            # Compute batch statistics
            mean = mx.mean(x, axis=(0, 1, 2))
            var = mx.var(x, axis=(0, 1, 2))

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        if self.weight is not None:
            x = x * self.weight + self.bias

        return x
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Channel order | NCHW | NHWC |
| BatchNorm | Built-in | Manual |
| GroupNorm | Built-in | Manual |
| InstanceNorm | Built-in | Manual |
| SyncBatchNorm | Built-in | N/A (single device) |

---

## When to Use Each

| Scenario | Recommended | Why |
|----------|-------------|-----|
| CNNs, large batch | BatchNorm | Best convergence |
| CNNs, small batch | GroupNorm | No batch dependency |
| Transformers | LayerNorm | Standard for attention |
| LLMs | RMSNorm | Faster, works well |
| Style transfer | InstanceNorm | Removes style info |
| Distributed training | SyncBatchNorm | Full batch stats |
| Batch size = 1 | LayerNorm or GroupNorm | No batch dependency |

---

## Summary

### Quick Reference

| Module | Normalizes Over | Parameters | Use Case |
|--------|-----------------|------------|----------|
| BatchNorm1d | (N, L) per C | gamma, beta per C | 1D CNNs |
| BatchNorm2d | (N, H, W) per C | gamma, beta per C | 2D CNNs |
| BatchNorm3d | (N, D, H, W) per C | gamma, beta per C | 3D CNNs |
| LayerNorm | Last D dims | gamma, beta element-wise | Transformers |
| GroupNorm | (H, W) per group | gamma, beta per C | Small batch |
| InstanceNorm | (H, W) per (N, C) | Optional gamma, beta | Style transfer |
| RMSNorm | Last D dims | gamma only | LLMs |
| SyncBatchNorm | Global (N, H, W) | Same as BatchNorm | Distributed |

### Parameter Counts

```
BatchNorm2d(C):    2 * C  (gamma, beta)
LayerNorm(H):      2 * H  (gamma, beta)
GroupNorm(G, C):   2 * C  (gamma, beta)
RMSNorm(H):        H      (gamma only)
```

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/normalization.py`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/batchnorm.py`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/instancenorm.py`

**Functional API**:
- `F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)`
- `F.layer_norm(input, normalized_shape, weight, bias, eps)`
- `F.group_norm(input, num_groups, weight, bias, eps)`
- `F.instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)`
- `F.rms_norm(input, normalized_shape, weight, eps)`
- `F.local_response_norm(input, size, alpha, beta, k)`
