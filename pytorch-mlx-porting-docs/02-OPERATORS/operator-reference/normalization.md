# Normalization Operators Reference

## Overview

Normalization operators standardize activations to improve training stability, accelerate convergence, and enable higher learning rates. They are essential components of modern deep learning architectures.

**Key Normalization Types**:
- **Batch Normalization**: Normalizes across batch dimension
- **Layer Normalization**: Normalizes across feature dimension (batch-independent)
- **Instance Normalization**: Normalizes per sample per channel
- **Group Normalization**: Normalizes groups of channels
- **RMS Normalization**: Root mean square normalization (no mean centering)

**Common Properties**:
- **Training vs Eval Mode**: Different behavior during training and inference
- **Running Statistics**: Exponential moving average of mean/variance (batch norm)
- **Affine Transforms**: Learnable scale (gamma) and shift (beta) parameters
- **Numerical Stability**: Small epsilon added to variance

**Applications**:
- **Stabilize Training**: Reduce internal covariate shift
- **Enable Higher Learning Rates**: Less sensitive to initialization
- **Regularization Effect**: Noise from batch statistics acts as regularizer
- **Gradient Flow**: Improved gradient propagation in deep networks

---

## Batch Normalization Operators

### batch_norm

**Purpose**: Normalize activations across batch dimension with running statistics

**Signature**: `batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor`

**YAML Definition** (native_functions.yaml:1112):
```yaml
- func: batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
```

**Algorithm**:
```python
# Training mode:
batch_mean = mean(input, dim=[0, 2, 3])  # For 4D: [N,C,H,W] -> mean over N,H,W
batch_var = var(input, dim=[0, 2, 3], unbiased=False)

# Normalize
normalized = (input - batch_mean) / sqrt(batch_var + eps)

# Affine transform
output = weight * normalized + bias

# Update running statistics
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var = (1 - momentum) * running_var + momentum * batch_var

# Eval mode:
normalized = (input - running_mean) / sqrt(running_var + eps)
output = weight * normalized + bias
```

**Common Use**: Standard batch normalization in CNNs

**MLX Equivalent**: `mx.normalize()` with manual running statistics

---

### native_batch_norm

**Purpose**: Low-level batch norm returning normalized output and statistics

**Signature**: `native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)`

**YAML Definition** (native_functions.yaml:4449):
```yaml
- func: native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
```

**Returns**: `(output, save_mean, save_invstd)`
- `output`: Normalized and affine-transformed tensor
- `save_mean`: Batch mean (for backward pass)
- `save_invstd`: Inverse standard deviation 1/sqrt(var + eps) (for backward pass)

**Common Use**: Backend implementation for batch norm

---

### _native_batch_norm_legit

**Purpose**: Native batch norm with in-place running statistics update

**Signature**: `_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)`

**YAML Definition** (native_functions.yaml:4463):
```yaml
- func: _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
```

**Key Difference**: In-place updates to running_mean and running_var (marked with `!`)

**Common Use**: Efficient batch norm implementation

---

## Layer Normalization Operators

### layer_norm

**Purpose**: Normalize across feature dimensions (batch-independent)

**Signature**: `layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, bool cudnn_enable) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
```

**Algorithm**:
```python
# Compute mean and variance over last D dimensions (normalized_shape)
dims = tuple(range(-len(normalized_shape), 0))
mean = input.mean(dim=dims, keepdim=True)
var = input.var(dim=dims, unbiased=False, keepdim=True)

# Normalize
normalized = (input - mean) / torch.sqrt(var + eps)

# Affine transform (per-element)
if weight is not None:
    normalized = normalized * weight
if bias is not None:
    normalized = normalized + bias

return normalized
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor |
| `normalized_shape` | int[] | Shape of dimensions to normalize (last D dims) |
| `weight` | Tensor? | Learnable scale (gamma), shape = normalized_shape |
| `bias` | Tensor? | Learnable shift (beta), shape = normalized_shape |
| `eps` | float | Small constant for numerical stability |

**Shape**:
- Input: `(*, normalized_shape)` where `*` is any leading dimensions
- Output: Same as input

**Usage Example**:
```python
# NLP: normalize over embedding dimension
x = torch.randn(32, 128, 512)  # (batch, seq, embed)
out = F.layer_norm(x, [512], weight, bias)

# Vision: normalize over C, H, W
x = torch.randn(32, 64, 56, 56)  # (batch, C, H, W)
out = F.layer_norm(x, [64, 56, 56], weight, bias)
```

**MLX Equivalent**: `mlx.nn.LayerNorm` (built-in)

---

### native_layer_norm

**Purpose**: Low-level layer norm returning normalized output and statistics

**Signature**: `native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)`

**Returns**: `(output, mean, rstd)`
- `output`: Normalized tensor
- `mean`: Mean computed over normalized dimensions
- `rstd`: Reciprocal standard deviation 1/sqrt(var + eps)

**Common Use**: Backend implementation, custom backward passes

---

## Group Normalization Operators

### group_norm

**Purpose**: Normalize within groups of channels (batch-independent)

**Signature**: `group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps, bool cudnn_enabled) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
```

**Algorithm**:
```python
N, C, *spatial = input.shape
assert C % num_groups == 0

# Reshape to (N, num_groups, C//num_groups, *spatial)
x = input.view(N, num_groups, C // num_groups, *spatial)

# Compute mean and var per group
mean = x.mean(dim=tuple(range(2, x.dim())), keepdim=True)
var = x.var(dim=tuple(range(2, x.dim())), unbiased=False, keepdim=True)

# Normalize
x = (x - mean) / torch.sqrt(var + eps)

# Reshape back to (N, C, *spatial)
x = x.view(N, C, *spatial)

# Affine transform (per-channel)
if weight is not None:
    x = x * weight.view(1, C, *([1] * len(spatial)))
if bias is not None:
    x = x + bias.view(1, C, *([1] * len(spatial)))

return x
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor of shape (N, C, *) |
| `num_groups` | int | Number of groups (C must be divisible) |
| `weight` | Tensor? | Per-channel scale, shape (C,) |
| `bias` | Tensor? | Per-channel shift, shape (C,) |
| `eps` | float | Numerical stability constant |

**Special Cases**:
- `num_groups=1`: Equivalent to LayerNorm over channels
- `num_groups=C`: Equivalent to InstanceNorm

**Usage Example**:
```python
x = torch.randn(20, 6, 10, 10)
# 6 channels, 3 groups (2 channels per group)
out = F.group_norm(x, num_groups=3, weight=weight, bias=bias)
```

---

### native_group_norm

**Purpose**: Low-level group norm returning normalized output and statistics

**Signature**: `native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)`

**Returns**: `(output, mean, rstd)`

---

## Instance Normalization Operators

### instance_norm

**Purpose**: Normalize per sample per channel (batch-independent)

**Signature**: `instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
```

**Algorithm**:
```python
N, C, *spatial = input.shape

if use_input_stats:
    # Compute stats per (N, C) pair
    mean = input.mean(dim=tuple(range(2, input.dim())), keepdim=True)
    var = input.var(dim=tuple(range(2, input.dim())), unbiased=False, keepdim=True)
else:
    # Use running statistics
    mean = running_mean.view(1, C, *([1] * len(spatial)))
    var = running_var.view(1, C, *([1] * len(spatial)))

# Normalize
normalized = (input - mean) / torch.sqrt(var + eps)

# Affine transform (optional)
if weight is not None:
    normalized = normalized * weight.view(1, C, *([1] * len(spatial)))
if bias is not None:
    normalized = normalized + bias.view(1, C, *([1] * len(spatial)))

return normalized
```

**Key Difference from BatchNorm**: Computes statistics per (sample, channel) pair, not across batch

**Usage Example**:
```python
x = torch.randn(4, 64, 256, 256)
out = F.instance_norm(x, weight=weight, bias=bias, use_input_stats=True)
```

**Common Use**: Style transfer, domain adaptation

---

## RMS Normalization Operators

### rms_norm

**Purpose**: Normalize by root mean square (no mean subtraction)

**Signature**: `rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, float? eps) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
```

**Algorithm**:
```python
# RMS over last D dimensions
dims = tuple(range(-len(normalized_shape), 0))
rms = torch.sqrt(torch.mean(input ** 2, dim=dims, keepdim=True) + eps)

# Normalize
normalized = input / rms

# Scale (no bias in RMSNorm)
if weight is not None:
    normalized = normalized * weight

return normalized
```

**Key Difference from LayerNorm**:
- No mean subtraction
- No bias parameter
- Faster computation
- Used in modern LLMs (LLaMA, Mistral)

**Usage Example**:
```python
x = torch.randn(32, 128, 4096)  # (batch, seq, hidden)
out = F.rms_norm(x, [4096], weight=weight)
```

**MLX Equivalent**: `mlx.nn.RMSNorm` (built-in)

---

## Local Response Normalization

### local_response_norm

**Purpose**: Normalize across neighboring channels (AlexNet-style)

**Signature**: `local_response_norm(Tensor self, int size, float alpha, float beta, float k) -> Tensor`

**Formula**:
```
b_c = a_c * (k + (alpha/size) * sum_{c'} a_{c'}^2)^{-beta}
```
Where sum is over neighboring channels within size/2 of c.

**Usage Example**:
```python
x = torch.randn(32, 96, 55, 55)
out = F.local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=1)
```

**Note**: Largely obsolete, replaced by BatchNorm in modern architectures.

---

## Summary

**Total Operators**: 13 normalization operators

| Operator | Status | Normalizes Over | Use Case |
|----------|--------|-----------------|----------|
| `batch_norm` | Documented | (N, H, W) per C | CNNs, large batch |
| `native_batch_norm` | Documented | Same | Backend |
| `_native_batch_norm_legit` | Documented | Same | Efficient |
| `layer_norm` | Documented | Last D dims | Transformers |
| `native_layer_norm` | Documented | Same | Backend |
| `group_norm` | Documented | Groups of C | Small batch CNNs |
| `native_group_norm` | Documented | Same | Backend |
| `instance_norm` | Documented | (H, W) per (N, C) | Style transfer |
| `rms_norm` | Documented | Last D dims | LLMs |
| `local_response_norm` | Documented | Neighboring C | (Legacy) |

**Progress**: 10 / 13 normalization operators (77%)

### Key Concepts

- **Batch Norm**: Requires batch statistics, different train/eval behavior
- **Layer Norm**: Batch-independent, common in transformers
- **Group Norm**: Compromise between Layer and Instance
- **Instance Norm**: Per-sample, removes style information
- **RMS Norm**: Faster alternative to LayerNorm

### PyTorch -> MLX Mapping

| PyTorch | MLX |
|---------|-----|
| `F.layer_norm` | `mlx.nn.LayerNorm` (built-in) |
| `F.rms_norm` | `mlx.nn.RMSNorm` (built-in) |
| `F.batch_norm` | Manual implementation |
| `F.group_norm` | Manual implementation |
| `F.instance_norm` | Manual implementation |
