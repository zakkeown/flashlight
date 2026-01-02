# Dropout Operators Reference

## Overview

Dropout operators randomly zero elements during training to prevent overfitting and co-adaptation of neurons. They are essential regularization techniques in deep learning.

**Key Dropout Types**:
- **Element Dropout**: Randomly zero individual elements
- **Channel Dropout**: Randomly zero entire channels (for conv layers)
- **Alpha Dropout**: Maintains self-normalizing properties (for SELU)

**Common Properties**:
- **Training vs Eval Mode**: Active only during training
- **Inverted Dropout**: Scale outputs by 1/(1-p) during training
- **In-place Option**: Modify tensor in-place for memory efficiency
- **Deterministic Inference**: Identity function during evaluation

**Applications**:
- **Regularization**: Prevent overfitting
- **Ensemble Effect**: Implicit ensemble of subnetworks
- **Bayesian Approximation**: MC Dropout for uncertainty estimation
- **Feature Independence**: Break co-adaptation of neurons

---

## Core Dropout Operators

### dropout

**Purpose**: Randomly zero elements with probability p, scale remaining by 1/(1-p)

**Signature**: `dropout(Tensor input, float p, bool train, bool inplace) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: dropout(Tensor input, float p, bool train) -> Tensor
  variants: function
```

**Algorithm**:
```python
if not train or p == 0:
    return input

# Generate Bernoulli mask
mask = torch.bernoulli(torch.full_like(input, 1 - p))

# Apply mask and scale (inverted dropout)
if inplace:
    input.mul_(mask).div_(1 - p)
    return input
else:
    return input * mask / (1 - p)
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor of any shape |
| `p` | float | Probability of element being zeroed (0 to 1) |
| `train` | bool | If True, apply dropout; if False, identity |
| `inplace` | bool | If True, modify input in-place |

**Shape**: Input: `(*)`, Output: `(*)` (same shape)

**Gradient Formula**:
```python
# Backward pass uses same mask:
grad_input = grad_output * mask / (1 - p)
```

**Usage Example**:
```python
x = torch.randn(20, 16)
# Functional API
out = F.dropout(x, p=0.5, training=True)

# Module API
dropout = nn.Dropout(p=0.5)
out = dropout(x)
```

**MLX Equivalent**:
```python
def dropout(x, p, training):
    if not training or p == 0:
        return x
    mask = mx.random.bernoulli(1 - p, shape=x.shape)
    return x * mask / (1 - p)
```

---

### dropout_

**Purpose**: In-place version of dropout

**Signature**: `dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)`

**YAML Definition**:
```yaml
- func: dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  variants: function
```

**Key Difference**: Modifies input tensor directly, saves memory

**Usage Example**:
```python
x = torch.randn(20, 16)
F.dropout_(x, p=0.5, training=True)  # x is modified in-place
```

---

### feature_dropout

**Purpose**: Randomly zero entire channels (feature maps)

**Signature**: `feature_dropout(Tensor input, float p, bool train) -> Tensor`

**Algorithm**:
```python
if not train or p == 0:
    return input

# Generate mask for channels only (not spatial dims)
# For 4D input (N,C,H,W): mask shape is (N,C,1,1)
mask_shape = list(input.shape)
mask_shape[2:] = [1] * (input.dim() - 2)  # Broadcast over spatial dims
mask = torch.bernoulli(torch.full(mask_shape, 1 - p))

return input * mask / (1 - p)
```

**Common Use**: Spatial dropout for CNNs (Dropout2d, Dropout3d)

---

## Channel-wise Dropout Operators

### dropout1d

**Purpose**: Randomly zero entire 1D channels

**Signature**: `dropout1d(Tensor input, float p, bool train) -> Tensor`

**Shape**:
- Input: `(N, C, L)` or `(C, L)`
- Output: Same as input

**Algorithm**:
```python
# Mask shape: (N, C, 1) - broadcast over length dimension
mask = torch.bernoulli(torch.full((N, C, 1), 1 - p))
return input * mask / (1 - p)
```

**Usage Example**:
```python
x = torch.randn(20, 16, 32)  # (batch, channels, length)
out = F.dropout1d(x, p=0.2, training=True)
```

---

### dropout2d

**Purpose**: Randomly zero entire 2D channels (feature maps)

**Signature**: `dropout2d(Tensor input, float p, bool train, bool inplace) -> Tensor`

**YAML Definition**:
```yaml
- func: dropout2d(Tensor input, float p, bool train) -> Tensor
```

**Shape**:
- Input: `(N, C, H, W)` or `(N, C, L)`
- Output: Same as input

**Algorithm**:
```python
# Mask shape: (N, C, 1, 1) - broadcast over spatial dimensions
mask = torch.bernoulli(torch.full((N, C, 1, 1), 1 - p))
return input * mask / (1 - p)
```

**Usage Example**:
```python
# After Conv2d
x = torch.randn(20, 16, 32, 32)  # (batch, channels, H, W)
out = F.dropout2d(x, p=0.2, training=True)
# ~20% of channels (entire 32x32 feature maps) are zeroed
```

**MLX Equivalent**:
```python
def dropout2d(x, p, training):
    if not training or p == 0:
        return x
    # x: (N, H, W, C) in MLX
    N, H, W, C = x.shape
    mask = mx.random.bernoulli(1 - p, shape=(N, 1, 1, C))
    return x * mask / (1 - p)
```

---

### dropout3d

**Purpose**: Randomly zero entire 3D channels (volumetric feature maps)

**Signature**: `dropout3d(Tensor input, float p, bool train, bool inplace) -> Tensor`

**Shape**:
- Input: `(N, C, D, H, W)` or `(C, D, H, W)`
- Output: Same as input

**Algorithm**:
```python
# Mask shape: (N, C, 1, 1, 1) - broadcast over volumetric dimensions
mask = torch.bernoulli(torch.full((N, C, 1, 1, 1), 1 - p))
return input * mask / (1 - p)
```

**Usage Example**:
```python
# 3D CNN for video/medical imaging
x = torch.randn(4, 16, 8, 32, 32)  # (batch, channels, depth, H, W)
out = F.dropout3d(x, p=0.2, training=True)
```

---

## Alpha Dropout Operators (Self-Normalizing)

### alpha_dropout

**Purpose**: Dropout that maintains self-normalizing property (for SELU activation)

**Signature**: `alpha_dropout(Tensor input, float p, bool train, bool inplace) -> Tensor`

**YAML Definition**:
```yaml
- func: alpha_dropout(Tensor input, float p, bool train) -> Tensor
```

**Algorithm**:
```python
if not train or p == 0:
    return input

# SELU constants
alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946

# Saturation value for dropped elements
a_prime = -scale * alpha  # ≈ -1.7581

# Compute affine parameters to maintain mean=0, var=1
keep_prob = 1 - p
a = ((1 - keep_prob) * (1 + keep_prob * a_prime**2))**(-0.5)
b = -a * (1 - keep_prob) * a_prime

# Apply dropout
mask = torch.bernoulli(torch.full_like(input, keep_prob))
output = mask * input + (1 - mask) * a_prime

# Affine transform to preserve statistics
return a * output + b
```

**Key Difference from Regular Dropout**:
- Dropped values set to negative saturation (not zero)
- Output scaled to preserve zero mean, unit variance
- Must be used with SELU activation

**Usage Example**:
```python
# Self-normalizing network
x = torch.randn(20, 16)
x = F.selu(linear(x))  # SELU activation
x = F.alpha_dropout(x, p=0.1, training=True)
```

---

### feature_alpha_dropout

**Purpose**: Channel-wise alpha dropout for convolutional SELU networks

**Signature**: `feature_alpha_dropout(Tensor input, float p, bool train, bool inplace) -> Tensor`

**YAML Definition**:
```yaml
- func: feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
```

**Shape**:
- Input: `(N, C, D, H, W)` or `(C, D, H, W)`
- Output: Same as input

**Algorithm**: Same as alpha_dropout but mask broadcasts over spatial dimensions

**Usage Example**:
```python
# Self-normalizing CNN
x = torch.randn(4, 16, 32, 32)
x = F.selu(conv(x))
x = F.feature_alpha_dropout(x, p=0.2, training=True)
```

---

## Summary

### Operator Coverage

| Operator | Status | Shape | Use Case |
|----------|--------|-------|----------|
| `dropout` | Core | `(*)` | General dropout |
| `dropout_` | Core | `(*)` | In-place variant |
| `dropout1d` | Core | `(N,C,L)` | After Conv1d |
| `dropout2d` | Core | `(N,C,H,W)` | After Conv2d |
| `dropout3d` | Core | `(N,C,D,H,W)` | After Conv3d |
| `alpha_dropout` | Core | `(*)` | With SELU |
| `feature_alpha_dropout` | Core | `(N,C,...)` | Conv + SELU |

**Total Operators**: 7 dropout operators

### Key Concepts

- **Inverted Dropout**: Scale by 1/(1-p) during training (not inference)
- **Element vs Channel**: Channel dropout for spatially correlated features
- **Alpha Dropout**: Preserves self-normalizing properties
- **Training Mode**: Only active when `training=True`

### PyTorch → MLX Mapping

| PyTorch | MLX Implementation |
|---------|-------------------|
| `F.dropout(x, p)` | `x * mx.random.bernoulli(1-p, x.shape) / (1-p)` |
| `F.dropout2d(x, p)` | Mask shape `(N, 1, 1, C)` for NHWC |
| `F.alpha_dropout(x, p)` | Manual implementation with SELU constants |

### Implementation Notes

1. **Mask Reuse**: For variational dropout, same mask used across time steps
2. **Gradient Flow**: Gradient only flows through non-dropped elements
3. **Numerical Stability**: Division by (1-p) can be unstable for p close to 1
4. **Memory**: In-place variants save memory but prevent gradient computation on input

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/dropout.py` (Module classes)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/functional.py` (Functional API)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/Dropout.cpp` (C++ implementation)
