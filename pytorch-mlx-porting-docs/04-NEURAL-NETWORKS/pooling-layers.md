# Pooling Layers

## Overview

Pooling layers reduce spatial dimensions of feature maps while retaining important information. PyTorch provides a comprehensive set of pooling operations for 1D, 2D, and 3D inputs, essential for CNN architectures.

**Reference File:** `torch/nn/modules/pooling.py`

## Pooling Layer Hierarchy

```
Pooling Operations
├── Max Pooling
│   ├── MaxPool1d, MaxPool2d, MaxPool3d
│   └── MaxUnpool1d, MaxUnpool2d, MaxUnpool3d
├── Average Pooling
│   └── AvgPool1d, AvgPool2d, AvgPool3d
├── Adaptive Pooling
│   ├── AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d
│   └── AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d
├── Fractional Max Pooling
│   └── FractionalMaxPool2d, FractionalMaxPool3d
└── LP Pooling
    └── LPPool1d, LPPool2d, LPPool3d
```

---

## Max Pooling

Takes the maximum value within each pooling window. Most common for CNN feature extraction.

### MaxPool2d

```python
class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple,     # Window size
        stride: int | tuple = None,    # Default: kernel_size
        padding: int | tuple = 0,      # Implicit -inf padding
        dilation: int | tuple = 1,     # Spacing between kernel elements
        return_indices: bool = False,  # Return argmax indices
        ceil_mode: bool = False        # Use ceil for output size
    )
```

### Mathematical Formulation

For 2D input with shape (N, C, H, W):

```
out(N_i, C_j, h, w) = max_{m=0..kH-1, n=0..kW-1}
    input(N_i, C_j, stride[0]*h + m, stride[1]*w + n)
```

### Output Size Calculation

**floor mode (default):**
```
H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
W_out = floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
```

**ceil mode:**
```
H_out = ceil((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `kernel_size` | Size of pooling window. Single int or tuple (kH, kW) |
| `stride` | Step size. Default equals kernel_size (non-overlapping) |
| `padding` | Implicit negative infinity padding on both sides |
| `dilation` | Spacing between kernel elements (atrous/dilated pooling) |
| `return_indices` | If True, also return argmax indices (for MaxUnpool) |
| `ceil_mode` | Use ceil instead of floor for output shape calculation |

### Shape Conventions

| Variant | Input Shape | Output Shape |
|---------|-------------|--------------|
| MaxPool1d | (N, C, L) or (C, L) | (N, C, L_out) |
| MaxPool2d | (N, C, H, W) or (C, H, W) | (N, C, H_out, W_out) |
| MaxPool3d | (N, C, D, H, W) or (C, D, H, W) | (N, C, D_out, H_out, W_out) |

### Usage Example

```python
# Basic 2x2 max pooling with stride 2
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(1, 64, 32, 32)
out = pool(x)  # Shape: (1, 64, 16, 16)

# With return_indices for unpooling
pool_with_indices = nn.MaxPool2d(2, stride=2, return_indices=True)
out, indices = pool_with_indices(x)

# Dilated max pooling
pool_dilated = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=2)
```

---

## Max Unpooling

Partial inverse of max pooling using stored indices. Non-maximal values are set to zero.

### MaxUnpool2d

```python
class MaxUnpool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple,
        stride: int | tuple = None,
        padding: int | tuple = 0
    )

    def forward(
        self,
        input: Tensor,
        indices: Tensor,              # From MaxPool with return_indices=True
        output_size: list[int] = None # Resolve ambiguous output shape
    ) -> Tensor
```

### Output Size Calculation

```
H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
```

### Usage Example

```python
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)

x = torch.tensor([[[[1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.]]]])

pooled, indices = pool(x)
# pooled: [[[[6., 8.], [14., 16.]]]]

unpooled = unpool(pooled, indices)
# [[[[0., 0., 0., 0.],
#    [0., 6., 0., 8.],
#    [0., 0., 0., 0.],
#    [0., 14., 0., 16.]]]]
```

---

## Average Pooling

Computes the mean value within each pooling window.

### AvgPool2d

```python
class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple,
        stride: int | tuple = None,
        padding: int | tuple = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,  # Include padding in average
        divisor_override: int = None     # Custom divisor
    )
```

### Mathematical Formulation

```
out(N_i, C_j, h, w) = (1 / (kH * kW)) *
    sum_{m=0..kH-1, n=0..kW-1} input(N_i, C_j, stride[0]*h + m, stride[1]*w + n)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `count_include_pad` | If True, include zero-padded elements in average denominator |
| `divisor_override` | Override the default divisor (kernel area) with a custom value |

### Usage Example

```python
# Standard average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Exclude padding from average calculation
avg_pool_no_pad = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

# Custom divisor (normalize by 9 regardless of actual kernel coverage)
avg_pool_custom = nn.AvgPool2d(3, stride=1, padding=1, divisor_override=9)
```

---

## Adaptive Pooling

Automatically computes kernel size and stride to achieve target output size. Input-size-agnostic pooling.

### AdaptiveAvgPool2d

```python
class AdaptiveAvgPool2d(Module):
    def __init__(
        self,
        output_size: int | tuple | None  # Target (H_out, W_out)
    )
```

### Key Feature

- **No kernel_size/stride required**: Automatically computed from input and output sizes
- **None values**: Preserve input dimension (e.g., `(None, 7)` keeps H, sets W=7)

### Common Use Cases

```python
# Global average pooling (for classification heads)
gap = nn.AdaptiveAvgPool2d(1)  # Output: (N, C, 1, 1)

# Fixed output size regardless of input
adaptive = nn.AdaptiveAvgPool2d((7, 7))  # Always output 7x7

# Partial adaptation
partial = nn.AdaptiveAvgPool2d((None, 7))  # Keep H, set W=7
```

### AdaptiveMaxPool2d

Same interface as AdaptiveAvgPool2d, but with optional `return_indices`:

```python
class AdaptiveMaxPool2d(Module):
    def __init__(
        self,
        output_size: int | tuple,
        return_indices: bool = False
    )
```

---

## Fractional Max Pooling

Stochastic pooling with pseudo-random stride selection. Provides regularization effect.

### FractionalMaxPool2d

```python
class FractionalMaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int | tuple,
        output_size: int | tuple = None,   # Target output (exclusive)
        output_ratio: float | tuple = None, # Ratio of input (exclusive)
        return_indices: bool = False
    )
```

**Note:** Exactly one of `output_size` or `output_ratio` must be specified.

### Usage Example

```python
# Target specific output size
frac_pool = nn.FractionalMaxPool2d(3, output_size=(13, 12))

# Target output as ratio of input (0.5 = half size)
frac_pool_ratio = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))

x = torch.randn(1, 64, 50, 32)
out = frac_pool(x)  # Shape: (1, 64, 13, 12)
```

### Reference
- Paper: "Fractional Max-Pooling" by Ben Graham (https://arxiv.org/abs/1412.6071)

---

## LP Pooling (Power-Average Pooling)

Generalized pooling using p-norm: `f(X) = (sum(x^p))^(1/p)`

### LPPool2d

```python
class LPPool2d(Module):
    def __init__(
        self,
        norm_type: float,             # The power p
        kernel_size: int | tuple,
        stride: int | tuple = None,
        ceil_mode: bool = False
    )
```

### Special Cases

| `norm_type` | Behavior |
|-------------|----------|
| p = 1 | Sum pooling (proportional to average) |
| p = 2 | L2 pooling (Euclidean norm) |
| p → ∞ | Approaches max pooling |

### Usage Example

```python
# L2 pooling
l2_pool = nn.LPPool2d(norm_type=2, kernel_size=3, stride=2)

# L1 pooling (sum)
l1_pool = nn.LPPool2d(norm_type=1, kernel_size=3, stride=2)
```

---

## Functional API

All pooling operations are also available as functions:

```python
import torch.nn.functional as F

# Max pooling
out = F.max_pool2d(input, kernel_size=2, stride=2)
out, indices = F.max_pool2d(input, 2, return_indices=True)

# Average pooling
out = F.avg_pool2d(input, kernel_size=2, stride=2)

# Adaptive pooling
out = F.adaptive_avg_pool2d(input, output_size=(1, 1))
out = F.adaptive_max_pool2d(input, output_size=(7, 7))

# LP pooling
out = F.lp_pool2d(input, norm_type=2, kernel_size=3, stride=2)

# Fractional max pooling
out = F.fractional_max_pool2d(input, kernel_size=3, output_size=(13, 12))
```

---

## Gradient Formulas

### Max Pooling Backward

Gradient flows only to the maximum element:

```
∂L/∂x_i = ∂L/∂y * 1_{x_i = max(window)}
```

All other positions receive zero gradient.

### Average Pooling Backward

Gradient is distributed equally:

```
∂L/∂x_i = ∂L/∂y / (kH * kW)
```

If `count_include_pad=False`, divide only by non-padded elements.

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `MaxPool2d` | `mlx.nn.MaxPool2d` |
| `AvgPool2d` | `mlx.nn.AvgPool2d` |
| `AdaptiveAvgPool2d(1)` | Global pooling pattern |

### Implementation Notes

1. **Adaptive Pooling**: May need custom implementation computing dynamic kernel sizes
2. **MaxUnpool**: Not commonly available; implement with scatter operations
3. **Fractional Pooling**: Typically not available; may need custom implementation
4. **LP Pooling**: Compose from power, sum, and root operations

### Porting Example

```python
# PyTorch
pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

# MLX equivalent
import mlx.nn as nn_mlx
pool_mlx = nn_mlx.MaxPool2d(kernel_size=2, stride=2, padding=0)
```

---

## Common Patterns

### Global Average Pooling (GAP)

Standard for classification heads in modern CNNs:

```python
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ...  # Feature extractor
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # (N, C, H, W)
        x = self.gap(x)        # (N, C, 1, 1)
        x = x.flatten(1)       # (N, C)
        return self.fc(x)
```

### Downsampling in CNNs

```python
# Strided convolution (learnable, often preferred)
nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

# Max pooling (non-learnable, preserves strong features)
nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling (non-learnable, preserves smooth features)
nn.AvgPool2d(kernel_size=2, stride=2)
```

### Feature Pyramid with Adaptive Pooling

```python
# Multi-scale feature extraction
scales = [nn.AdaptiveAvgPool2d(s) for s in [(1,1), (2,2), (4,4), (8,8)]]
features = [pool(x) for pool in scales]
```

---

## Summary Table

| Layer | Parameters | Key Features |
|-------|------------|--------------|
| MaxPool | kernel, stride, padding, dilation | Returns max value, optional indices |
| AvgPool | kernel, stride, padding | Mean value, count_include_pad option |
| AdaptiveMaxPool | output_size | Input-size agnostic max |
| AdaptiveAvgPool | output_size | Input-size agnostic mean |
| FractionalMaxPool | kernel, output_size/ratio | Stochastic stride |
| LPPool | norm_type, kernel, stride | Generalized p-norm |
| MaxUnpool | kernel, stride, padding | Inverse of MaxPool with indices |
