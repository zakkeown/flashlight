# Padding Layers

## Overview

Padding layers add values to tensor boundaries, essential for controlling spatial dimensions in CNNs. PyTorch provides multiple padding modes with distinct behaviors at boundaries.

**Reference File:** `torch/nn/modules/padding.py`

## Padding Mode Hierarchy

```
Padding Layers
├── Constant Padding
│   ├── ConstantPad1d, ConstantPad2d, ConstantPad3d
│   └── ZeroPad1d, ZeroPad2d, ZeroPad3d (value=0)
├── Reflection Padding
│   └── ReflectionPad1d, ReflectionPad2d, ReflectionPad3d
├── Replication Padding
│   └── ReplicationPad1d, ReplicationPad2d, ReplicationPad3d
└── Circular Padding
    └── CircularPad1d, CircularPad2d, CircularPad3d
```

---

## Padding Specification

### Tuple Format

Padding is specified from **innermost to outermost dimensions**:

| Variant | Tuple Format |
|---------|-------------|
| 1D | (left, right) |
| 2D | (left, right, top, bottom) |
| 3D | (left, right, top, bottom, front, back) |

**Single integer**: Same padding on all sides

### Output Size Calculation

```
H_out = H_in + padding_top + padding_bottom
W_out = W_in + padding_left + padding_right
D_out = D_in + padding_front + padding_back
```

**Negative padding**: Removes elements from boundaries (cropping)

---

## Constant Padding

Pads with a specified constant value.

### ConstantPad2d

```python
class ConstantPad2d(Module):
    def __init__(
        self,
        padding: int | tuple,  # (left, right, top, bottom)
        value: float           # Padding value
    )
```

### Example

```python
# Pad all sides with value 3.5
m = nn.ConstantPad2d(2, 3.5)
input = torch.randn(1, 1, 3, 3)
output = m(input)  # Shape: (1, 1, 7, 7)

# Asymmetric padding: (left=3, right=0, top=2, bottom=1)
m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
```

### Visual Example

Input (3x3):
```
[[0., 1., 2.],
 [3., 4., 5.],
 [6., 7., 8.]]
```

ConstantPad2d(1, 0) output (5x5):
```
[[0., 0., 0., 0., 0.],
 [0., 0., 1., 2., 0.],
 [0., 3., 4., 5., 0.],
 [0., 6., 7., 8., 0.],
 [0., 0., 0., 0., 0.]]
```

---

## Zero Padding

Special case of constant padding with value=0. Most common in CNNs.

### ZeroPad2d

```python
class ZeroPad2d(ConstantPad2d):
    def __init__(
        self,
        padding: int | tuple  # (left, right, top, bottom)
    )
```

Internally calls `ConstantPad2d.__init__(padding, 0.0)`.

### Example

```python
# Symmetric zero padding
m = nn.ZeroPad2d(2)
input = torch.randn(1, 64, 32, 32)
output = m(input)  # Shape: (1, 64, 36, 36)

# Asymmetric: more padding on left and top
m = nn.ZeroPad2d((3, 1, 2, 0))  # left=3, right=1, top=2, bottom=0
```

---

## Reflection Padding

Pads using reflection of input boundary. Values near edges are mirrored.

### ReflectionPad2d

```python
class ReflectionPad2d(Module):
    def __init__(
        self,
        padding: int | tuple  # (left, right, top, bottom)
    )
```

**Constraint**: Padding size must be **less than** the corresponding input dimension.

### Example

```python
m = nn.ReflectionPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
# [[0., 1., 2.],
#  [3., 4., 5.],
#  [6., 7., 8.]]

output = m(input)
# [[8., 7., 6., 7., 8., 7., 6.],
#  [5., 4., 3., 4., 5., 4., 3.],
#  [2., 1., 0., 1., 2., 1., 0.],  <- Original row 0 mirrored
#  [5., 4., 3., 4., 5., 4., 3.],  <- Original row 1
#  [8., 7., 6., 7., 8., 7., 6.],  <- Original row 2
#  [5., 4., 3., 4., 5., 4., 3.],
#  [2., 1., 0., 1., 2., 1., 0.]]
```

### Reflection Pattern

For 1D: `[... d, c, b | a, b, c, d | c, b, a ...]`

The boundary element `a` is **not** duplicated (unlike replication).

---

## Replication Padding

Pads by replicating edge values. Also called "edge" padding.

### ReplicationPad2d

```python
class ReplicationPad2d(Module):
    def __init__(
        self,
        padding: int | tuple  # (left, right, top, bottom)
    )
```

### Example

```python
m = nn.ReplicationPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
# [[0., 1., 2.],
#  [3., 4., 5.],
#  [6., 7., 8.]]

output = m(input)
# [[0., 0., 0., 1., 2., 2., 2.],
#  [0., 0., 0., 1., 2., 2., 2.],  <- Top edge replicated
#  [0., 0., 0., 1., 2., 2., 2.],
#  [3., 3., 3., 4., 5., 5., 5.],  <- Original row 1 with edge replication
#  [6., 6., 6., 7., 8., 8., 8.],
#  [6., 6., 6., 7., 8., 8., 8.],  <- Bottom edge replicated
#  [6., 6., 6., 7., 8., 8., 8.]]
```

### Replication Pattern

For 1D: `[a, a, a | a, b, c, d | d, d, d]`

Edge values are **duplicated** to fill padding region.

---

## Circular Padding

Pads using circular/periodic wrapping. Values from opposite side are used.

### CircularPad2d

```python
class CircularPad2d(Module):
    def __init__(
        self,
        padding: int | tuple  # (left, right, top, bottom)
    )
```

**Constraint**: Padding size should be **less than or equal to** the corresponding input dimension.

### Example

```python
m = nn.CircularPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
# [[0., 1., 2.],
#  [3., 4., 5.],
#  [6., 7., 8.]]

output = m(input)
# [[4., 5., 3., 4., 5., 3., 4.],  <- Wrapped from middle/bottom
#  [7., 8., 6., 7., 8., 6., 7.],
#  [1., 2., 0., 1., 2., 0., 1.],  <- Original row 0 with wrap
#  [4., 5., 3., 4., 5., 3., 4.],  <- Original row 1 with wrap
#  [7., 8., 6., 7., 8., 6., 7.],  <- Original row 2 with wrap
#  [1., 2., 0., 1., 2., 0., 1.],
#  [4., 5., 3., 4., 5., 3., 4.]]
```

### Circular Pattern

For 1D: `[... c, d | a, b, c, d | a, b ...]`

Values wrap around periodically (like a torus in 2D).

---

## Functional API

All padding modes available via `F.pad`:

```python
import torch.nn.functional as F

# General padding function
output = F.pad(
    input,                    # Input tensor
    pad,                      # Tuple of padding sizes
    mode='constant',          # 'constant', 'reflect', 'replicate', 'circular'
    value=0                   # For constant mode only
)

# Examples
x = torch.randn(1, 1, 3, 3)

# Zero padding
F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

# Reflection padding
F.pad(x, (1, 1, 1, 1), mode='reflect')

# Replication padding
F.pad(x, (1, 1, 1, 1), mode='replicate')

# Circular padding
F.pad(x, (1, 1, 1, 1), mode='circular')
```

---

## Shape Conventions

### All Padding Layers

| Variant | Input Shape | Output Shape |
|---------|-------------|--------------|
| Pad1d | (N, C, W) or (C, W) | (N, C, W + pad_left + pad_right) |
| Pad2d | (N, C, H, W) or (C, H, W) | (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right) |
| Pad3d | (N, C, D, H, W) or (C, D, H, W) | (N, C, D + pad_front + pad_back, H + pad_top + pad_bottom, W + pad_left + pad_right) |

---

## Mode Comparison

### Visual Comparison (1D)

Input: `[a, b, c, d]`, padding=2 on each side

| Mode | Result |
|------|--------|
| Constant (0) | `[0, 0, a, b, c, d, 0, 0]` |
| Reflection | `[c, b, a, b, c, d, c, b]` |
| Replication | `[a, a, a, b, c, d, d, d]` |
| Circular | `[c, d, a, b, c, d, a, b]` |

### When to Use Each Mode

| Mode | Use Case |
|------|----------|
| **Zero/Constant** | Default for CNNs, introduces bias at edges |
| **Reflection** | Seamless textures, reduces edge artifacts |
| **Replication** | Preserves edge values, less edge artifacts than zero |
| **Circular** | Periodic signals, wrap-around semantics |

---

## Gradient Behavior

### Zero/Constant Padding

- Gradient is **zero** in padded regions
- Original values receive full gradient

### Reflection Padding

- Gradient is **accumulated** at reflected positions
- Boundary elements receive gradient from multiple reflected copies

### Replication Padding

- Gradient is **accumulated** at edge elements
- Edge elements receive gradient from all their replicated copies

### Circular Padding

- Gradient **wraps around** like the forward pass
- Each position receives gradient from its wrapped copies

---

## Common Patterns

### Same Padding for Convolutions

Maintain spatial dimensions after convolution:

```python
class SameConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Calculate padding for "same" output size
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        return self.conv(self.pad(x))
```

### Asymmetric Padding for Specific Kernels

```python
# For kernel_size=3, stride=2: need asymmetric padding
# Input 7x7 -> Output 4x4 requires padding=(1,0,1,0)
pad = nn.ZeroPad2d((1, 0, 1, 0))  # left=1, right=0, top=1, bottom=0
```

### Pre-padding Before Pooling

```python
class PaddedMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, pad_mode='reflect'):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.pool = nn.MaxPool2d(kernel_size, stride, padding=0)

    def forward(self, x):
        return self.pool(self.pad(x))
```

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `F.pad` | `mlx.core.pad` |
| ZeroPad2d | Compose with `mlx.core.pad` |

### Implementation Notes

1. **Padding order**: Verify MLX uses same tuple ordering (left, right, top, bottom)
2. **Mode names**: Check MLX uses same mode strings ('reflect', 'replicate', etc.)
3. **Module wrappers**: May need to create custom `nn.Module` equivalents

### Porting Example

```python
# PyTorch
pad = nn.ReflectionPad2d(2)
out = pad(x)

# MLX equivalent
import mlx.core as mx

def reflection_pad_2d(x, padding):
    """PyTorch-style ReflectionPad2d for MLX."""
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    # left, right, top, bottom -> MLX format
    return mx.pad(x, [(0, 0), (0, 0), (padding[2], padding[3]), (padding[0], padding[1])],
                  mode='reflect')
```

---

## Implementation Details

### All Padding Layers Use F.pad

```python
class ReflectionPad2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "reflect")

class ReplicationPad2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "replicate")

class ConstantPad2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "constant", self.value)
```

### Input Dimension Validation

Each padding layer validates input dimensions:
- Pad1d: expects 2D or 3D input
- Pad2d: expects 3D or 4D input
- Pad3d: expects 4D or 5D input

---

## Summary Table

| Layer | Mode | Value | Constraints |
|-------|------|-------|-------------|
| ZeroPad{1,2,3}d | constant | 0 | None |
| ConstantPad{1,2,3}d | constant | any float | None |
| ReflectionPad{1,2,3}d | reflect | edge mirror | pad < input_size |
| ReplicationPad{1,2,3}d | replicate | edge copy | None |
| CircularPad{1,2,3}d | circular | wrap around | pad <= input_size |
