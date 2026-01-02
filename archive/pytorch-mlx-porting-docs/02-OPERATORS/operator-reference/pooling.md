# Pooling Operators

## Purpose

Pooling operators perform spatial dimensionality reduction by aggregating local regions of input tensors. They are essential for:
- **Downsampling**: Reducing spatial dimensions while preserving important features
- **Translation invariance**: Making networks robust to small spatial shifts
- **Receptive field expansion**: Enabling deeper layers to see larger input regions
- **Memory efficiency**: Reducing computation in subsequent layers

## Pooling Classes Overview

| Category | Classes | Description |
|----------|---------|-------------|
| **Max Pooling** | MaxPool1d/2d/3d | Take maximum in local window |
| **Average Pooling** | AvgPool1d/2d/3d | Take mean in local window |
| **Adaptive Pooling** | AdaptiveMaxPool, AdaptiveAvgPool | Pool to specified output size |
| **Fractional Pooling** | FractionalMaxPool2d/3d | Stochastic step size pooling |
| **LP Pooling** | LPPool1d/2d/3d | Lp-norm based pooling |
| **Unpooling** | MaxUnpool1d/2d/3d | Partial inverse of max pooling |

---

## Mathematical Foundations

### Max Pooling

For each spatial window of size `(kH, kW)`:

```
output[n, c, h, w] = max_{m=0..kH-1, n=0..kW-1} input[n, c, stride[0]*h + m, stride[1]*w + n]
```

The maximum value in the local region is selected. Padding uses `-inf` (negative infinity) for boundary extension.

### Average Pooling

For each spatial window:

```
output[n, c, h, w] = (1/k) * sum_{m=0..kH-1, n=0..kW-1} input[n, c, stride[0]*h + m, stride[1]*w + n]
```

Where `k` is the count of elements (affected by `count_include_pad`).

### LP Pooling

Generalized norm-based pooling:

```
output[n, c, h, w] = (sum_{x in window} |x|^p)^(1/p)
```

Where:
- `p = 1`: Sum pooling (proportional to average)
- `p = 2`: L2 pooling (root mean square-like)
- `p = inf`: Max pooling

### Output Size Formula

For standard pooling:
```
L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

With `ceil_mode=True`:
```
L_out = ceil((L_in + 2*padding - dilation*(kernel_size-1) - 1 + (stride-1)) / stride + 1)
```

---

## Max Pooling Layers

### MaxPool1d

**Purpose**: 1D max pooling for temporal/sequential data.

**Signature**:
```python
class torch.nn.MaxPool1d(
    kernel_size: int,
    stride: int = None,           # Defaults to kernel_size
    padding: int = 0,
    dilation: int = 1,
    return_indices: bool = False,
    ceil_mode: bool = False
)
```

**Input/Output Shapes**:
- Input: `(N, C, L_in)` or `(C, L_in)`
- Output: `(N, C, L_out)` or `(C, L_out)`

**Source**: `torch/nn/modules/pooling.py:79-153`

**Forward Implementation**:
```python
def forward(self, input: Tensor):
    return F.max_pool1d(
        input,
        self.kernel_size,
        self.stride,
        self.padding,
        self.dilation,
        ceil_mode=self.ceil_mode,
        return_indices=self.return_indices,
    )
```

**Usage Example**:
```python
import torch
import torch.nn as nn

# Pool of size=3, stride=2
m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)  # (batch, channels, length)
output = m(input)  # (20, 16, 24)
```

### MaxPool2d

**Purpose**: 2D max pooling for image data.

**Signature**:
```python
class torch.nn.MaxPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False
)
```

**Input/Output Shapes**:
- Input: `(N, C, H_in, W_in)` or `(C, H_in, W_in)`
- Output: `(N, C, H_out, W_out)` or `(C, H_out, W_out)`

**Source**: `torch/nn/modules/pooling.py:155-233`

**Usage Examples**:
```python
# Square kernel, stride=2
m = nn.MaxPool2d(2, stride=2)
input = torch.randn(1, 64, 32, 32)
output = m(input)  # (1, 64, 16, 16) - half resolution

# Non-square kernel
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)

# With indices for unpooling
m = nn.MaxPool2d(2, stride=2, return_indices=True)
output, indices = m(input)
```

### MaxPool3d

**Purpose**: 3D max pooling for volumetric/video data.

**Signature**:
```python
class torch.nn.MaxPool3d(
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    return_indices: bool = False,
    ceil_mode: bool = False
)
```

**Input/Output Shapes**:
- Input: `(N, C, D_in, H_in, W_in)` or `(C, D_in, H_in, W_in)`
- Output: `(N, C, D_out, H_out, W_out)` or `(C, D_out, H_out, W_out)`

**Source**: `torch/nn/modules/pooling.py:235-317`

**Usage Example**:
```python
# Square kernel
m = nn.MaxPool3d(3, stride=2)

# Non-square kernel for video (temporal=3, spatial=2x2)
m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
input = torch.randn(20, 16, 50, 44, 31)
output = m(input)
```

---

## Average Pooling Layers

### AvgPool1d

**Purpose**: 1D average pooling for temporal data.

**Signature**:
```python
class torch.nn.AvgPool1d(
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True
)
```

**Source**: `torch/nn/modules/pooling.py:595-678`

**Parameters**:
- `count_include_pad`: If `True`, include zero-padding in averaging calculation

**Usage Example**:
```python
m = nn.AvgPool1d(3, stride=2)
input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]])
output = m(input)
# tensor([[[2., 4., 6.]]])  # Average of each 3-element window
```

### AvgPool2d

**Purpose**: 2D average pooling for image data.

**Signature**:
```python
class torch.nn.AvgPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None
)
```

**Source**: `torch/nn/modules/pooling.py:680-787`

**Parameters**:
- `divisor_override`: If specified, overrides the divisor used for averaging

**Usage Example**:
```python
# Standard 2x2 average pooling
m = nn.AvgPool2d(2, stride=2)
input = torch.randn(20, 16, 50, 32)
output = m(input)  # (20, 16, 25, 16)

# Non-square kernel
m = nn.AvgPool2d((3, 2), stride=(2, 1))
output = m(input)
```

### AvgPool3d

**Purpose**: 3D average pooling for volumetric/video data.

**Signature**:
```python
class torch.nn.AvgPool3d(
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None
)
```

**Source**: `torch/nn/modules/pooling.py:790-911`

---

## Adaptive Pooling Layers

Adaptive pooling layers automatically compute kernel size and stride to produce a specified output size, regardless of input size.

### AdaptiveMaxPool1d

**Signature**:
```python
class torch.nn.AdaptiveMaxPool1d(
    output_size: int,
    return_indices: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1313-1342`

### AdaptiveMaxPool2d

**Purpose**: Produce fixed output spatial dimensions.

**Signature**:
```python
class torch.nn.AdaptiveMaxPool2d(
    output_size: Union[int, Tuple[int, int]],
    return_indices: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1344-1385`

**Usage Example**:
```python
# Fixed 5x7 output, any input size
m = nn.AdaptiveMaxPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
output = m(input)  # (1, 64, 5, 7)

# Square output
m = nn.AdaptiveMaxPool2d(7)
output = m(input)  # (1, 64, 7, 7)

# Keep one dimension, adapt the other
m = nn.AdaptiveMaxPool2d((None, 7))
input = torch.randn(1, 64, 10, 9)
output = m(input)  # (1, 64, 10, 7)
```

### AdaptiveMaxPool3d

**Signature**:
```python
class torch.nn.AdaptiveMaxPool3d(
    output_size: Union[int, Tuple[int, int, int]],
    return_indices: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1387-1429`

### AdaptiveAvgPool1d

**Signature**:
```python
class torch.nn.AdaptiveAvgPool1d(output_size: int)
```

**Source**: `torch/nn/modules/pooling.py:1442-1471`

### AdaptiveAvgPool2d

**Purpose**: Global Average Pooling (GAP) when `output_size=(1, 1)`.

**Signature**:
```python
class torch.nn.AdaptiveAvgPool2d(
    output_size: Union[int, Tuple[int, int]]
)
```

**Source**: `torch/nn/modules/pooling.py:1473-1511`

**Usage Example**:
```python
# Global average pooling (common in modern CNNs)
gap = nn.AdaptiveAvgPool2d((1, 1))
input = torch.randn(32, 512, 7, 7)
output = gap(input)  # (32, 512, 1, 1)
output = output.flatten(1)  # (32, 512) - ready for classifier

# Specific output size
m = nn.AdaptiveAvgPool2d((5, 7))
output = m(input)  # (32, 512, 5, 7)
```

### AdaptiveAvgPool3d

**Signature**:
```python
class torch.nn.AdaptiveAvgPool3d(
    output_size: Union[int, Tuple[int, int, int]]
)
```

**Source**: `torch/nn/modules/pooling.py:1513-1551`

---

## Fractional Max Pooling

Fractional max pooling uses stochastic step sizes to achieve non-integer downsampling factors, as described in "Fractional Max-Pooling" by Ben Graham (2014).

### FractionalMaxPool2d

**Signature**:
```python
class torch.nn.FractionalMaxPool2d(
    kernel_size: Union[int, Tuple[int, int]],
    output_size: Union[int, Tuple[int, int]] = None,
    output_ratio: Union[float, Tuple[float, float]] = None,
    return_indices: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:913-1000`

**Parameters**:
- `output_size`: Target output size (mutually exclusive with `output_ratio`)
- `output_ratio`: Output size as ratio of input (0 < ratio < 1)

**Note**: Exactly one of `output_size` or `output_ratio` must be specified.

**Usage Example**:
```python
# Fixed output size
m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
input = torch.randn(20, 16, 50, 32)
output = m(input)  # (20, 16, 13, 12)

# Ratio-based (50% downsampling)
m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
output = m(input)  # (20, 16, 25, 16)
```

### FractionalMaxPool3d

**Signature**:
```python
class torch.nn.FractionalMaxPool3d(
    kernel_size: Union[int, Tuple[int, int, int]],
    output_size: Union[int, Tuple[int, int, int]] = None,
    output_ratio: Union[float, Tuple[float, float, float]] = None,
    return_indices: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1002-1095`

---

## LP Pooling Layers

LP pooling computes the Lp-norm over local regions, generalizing max and average pooling.

### LPPool1d

**Signature**:
```python
class torch.nn.LPPool1d(
    norm_type: float,
    kernel_size: int,
    stride: int = None,
    ceil_mode: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1123-1168`

**Mathematical Definition**:
```
f(X) = (sum_{x in X} |x|^p)^(1/p)
```

**Usage Example**:
```python
# L2 pooling
m = nn.LPPool1d(2, 3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
```

### LPPool2d

**Signature**:
```python
class torch.nn.LPPool2d(
    norm_type: float,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    ceil_mode: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1170-1228`

**Usage Example**:
```python
# L2 pooling
m = nn.LPPool2d(2, 3, stride=2)
input = torch.randn(20, 16, 50, 32)
output = m(input)

# Non-square kernel with fractional norm
m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
```

### LPPool3d

**Signature**:
```python
class torch.nn.LPPool3d(
    norm_type: float,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    ceil_mode: bool = False
)
```

**Source**: `torch/nn/modules/pooling.py:1230-1292`

---

## Max Unpooling Layers

Unpooling layers compute a partial inverse of max pooling, using stored indices to place values back at their original positions.

### MaxUnpool1d

**Signature**:
```python
class torch.nn.MaxUnpool1d(
    kernel_size: int,
    stride: int = None,
    padding: int = 0
)
```

**Forward Signature**:
```python
def forward(
    self,
    input: Tensor,
    indices: Tensor,
    output_size: List[int] = None
) -> Tensor
```

**Source**: `torch/nn/modules/pooling.py:324-405`

**Usage Example**:
```python
pool = nn.MaxPool1d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool1d(2, stride=2)

input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
output, indices = pool(input)
# output: tensor([[[2., 4., 6., 8.]]])

reconstructed = unpool(output, indices)
# tensor([[[0., 2., 0., 4., 0., 6., 0., 8.]]])
# Non-maximal positions are set to zero
```

### MaxUnpool2d

**Signature**:
```python
class torch.nn.MaxUnpool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0
)
```

**Source**: `torch/nn/modules/pooling.py:407-501`

**Usage Example**:
```python
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)

input = torch.tensor([[[[1.,  2.,  3.,  4.],
                         [5.,  6.,  7.,  8.],
                         [9., 10., 11., 12.],
                         [13., 14., 15., 16.]]]])

output, indices = pool(input)
# output: tensor([[[[6., 8.], [14., 16.]]]])

reconstructed = unpool(output, indices)
# tensor([[[[0.,  0.,  0.,  0.],
#           [0.,  6.,  0.,  8.],
#           [0.,  0.,  0.,  0.],
#           [0., 14.,  0., 16.]]]])
```

### MaxUnpool3d

**Signature**:
```python
class torch.nn.MaxUnpool3d(
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0
)
```

**Source**: `torch/nn/modules/pooling.py:503-580`

---

## Functional API

All pooling operations are available as functions in `torch.nn.functional`:

```python
import torch.nn.functional as F

# Max pooling
F.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
F.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
F.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

# Average pooling
F.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
F.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
F.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

# Adaptive pooling
F.adaptive_max_pool1d(input, output_size, return_indices=False)
F.adaptive_max_pool2d(input, output_size, return_indices=False)
F.adaptive_max_pool3d(input, output_size, return_indices=False)
F.adaptive_avg_pool1d(input, output_size)
F.adaptive_avg_pool2d(input, output_size)
F.adaptive_avg_pool3d(input, output_size)

# Fractional max pooling
F.fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)
F.fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

# LP pooling
F.lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
F.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
F.lp_pool3d(input, norm_type, kernel_size, stride=None, ceil_mode=False)

# Unpooling
F.max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
F.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
F.max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None)
```

---

## Key Parameters Explained

### kernel_size
Size of the pooling window. Can be:
- `int`: Square/cubic window
- `Tuple[int, ...]`: Different size per dimension

### stride
Step size for the sliding window. Defaults to `kernel_size`.
- `stride=kernel_size`: Non-overlapping windows
- `stride < kernel_size`: Overlapping windows

### padding
Zero-padding added to input boundaries.
- Max pooling: Padded with `-inf`
- Average pooling: Padded with `0`

### dilation
Spacing between kernel elements (atrous pooling).
- `dilation=1`: Standard pooling
- `dilation=2`: Dilated pooling (skip every other element)

### ceil_mode
Output size calculation mode:
- `False`: Use floor division (default)
- `True`: Use ceiling (ensures all input elements are covered)

### count_include_pad (AvgPool only)
Whether to include padding in the averaging denominator:
- `True`: Count padding as zeros in average
- `False`: Only count actual input values

### return_indices (MaxPool only)
Whether to return indices of maximum elements:
- Needed for `MaxUnpool` operations
- Useful for visualization

---

## Gradient Behavior

### Max Pooling Gradients
- Gradient flows only to the maximum element in each window
- Sparse gradients (most elements receive zero gradient)
- Indices are used to route gradients during backpropagation

```python
# Visualization of max pooling gradient
input = torch.tensor([[[[1., 2.],
                         [3., 4.]]]], requires_grad=True)
output = F.max_pool2d(input, 2)
output.backward()
print(input.grad)
# tensor([[[[0., 0.],
#           [0., 1.]]]])  # Only max element (4) gets gradient
```

### Average Pooling Gradients
- Gradient is distributed evenly to all elements in the window
- Dense gradients (all elements receive gradient)

```python
input = torch.tensor([[[[1., 2.],
                         [3., 4.]]]], requires_grad=True)
output = F.avg_pool2d(input, 2)
output.backward()
print(input.grad)
# tensor([[[[0.25, 0.25],
#           [0.25, 0.25]]]])  # Gradient split equally
```

---

## Pooling Comparison

| Property | Max Pool | Avg Pool | LP Pool |
|----------|----------|----------|---------|
| Operation | max | mean | Lp-norm |
| Gradient | Sparse | Dense | Dense |
| Translation invariance | High | Medium | Medium |
| Information preservation | Edges/features | Smooth regions | Balanced |
| Typical use | Early CNN layers | Later layers, GAP | Specialized |
| Parameter count | 0 | 0 | 0 |

---

## Common Patterns

### Global Average Pooling (GAP)

Replace final fully-connected layers with adaptive average pooling:

```python
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(...)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)         # (N, 512, H, W)
        x = self.gap(x)              # (N, 512, 1, 1)
        x = x.flatten(1)             # (N, 512)
        return self.classifier(x)
```

### Encoder-Decoder with Unpooling

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, x):
        pooled, indices = self.pool(x)
        return pooled, indices

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x, indices):
        return self.unpool(x, indices)
```

### Strided Pooling vs Strided Convolution

```python
# Option 1: Max pooling for downsampling
model1 = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2)  # 2x downsample
)

# Option 2: Strided convolution (learnable downsampling)
model2 = nn.Sequential(
    nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 2x downsample
    nn.ReLU()
)
```

---

## MLX Mapping

### Direct Mappings

| PyTorch | MLX Equivalent |
|---------|----------------|
| `F.max_pool2d` | `mx.nn.MaxPool2d` or `mx.max_pool2d` |
| `F.avg_pool2d` | `mx.nn.AvgPool2d` or `mx.avg_pool2d` |
| `nn.AdaptiveAvgPool2d` | Custom implementation |

### Layout Considerations

PyTorch uses NCHW (channels-first), MLX uses NHWC (channels-last):

```python
import mlx.core as mx
import mlx.nn as nn

def max_pool2d_pytorch_compat(input_nchw, kernel_size, stride=None, padding=0):
    """Max pooling with PyTorch-compatible interface."""
    if stride is None:
        stride = kernel_size

    # NCHW -> NHWC
    input_nhwc = mx.transpose(input_nchw, [0, 2, 3, 1])

    # Perform pooling
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_nhwc = pool(input_nhwc)

    # NHWC -> NCHW
    return mx.transpose(output_nhwc, [0, 3, 1, 2])

def avg_pool2d_pytorch_compat(input_nchw, kernel_size, stride=None, padding=0):
    """Average pooling with PyTorch-compatible interface."""
    if stride is None:
        stride = kernel_size

    # NCHW -> NHWC
    input_nhwc = mx.transpose(input_nchw, [0, 2, 3, 1])

    # Perform pooling
    pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    output_nhwc = pool(input_nhwc)

    # NHWC -> NCHW
    return mx.transpose(output_nhwc, [0, 3, 1, 2])
```

### Adaptive Pooling Implementation

MLX doesn't have built-in adaptive pooling. Implement as:

```python
def adaptive_avg_pool2d(input_nhwc, output_size):
    """Adaptive average pooling for MLX."""
    N, H_in, W_in, C = input_nhwc.shape
    H_out, W_out = output_size

    # Compute kernel size and stride to achieve output size
    stride_h = H_in // H_out
    stride_w = W_in // W_out
    kernel_h = H_in - (H_out - 1) * stride_h
    kernel_w = W_in - (W_out - 1) * stride_w

    pool = nn.AvgPool2d(
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w)
    )
    return pool(input_nhwc)

def global_avg_pool2d(input_nhwc):
    """Global average pooling - pool entire spatial dimension."""
    # Input: (N, H, W, C)
    return mx.mean(input_nhwc, axis=(1, 2), keepdims=True)  # (N, 1, 1, C)
```

### Fractional Max Pooling

Not available in MLX; implement manually or use standard pooling with appropriate sizes.

### Max Unpooling

Not available in MLX. For encoder-decoder architectures, consider:
- Bilinear upsampling + convolution
- Transposed convolution
- Nearest neighbor upsampling

```python
def upsample_nearest(input_nhwc, scale_factor=2):
    """Nearest neighbor upsampling as alternative to unpooling."""
    N, H, W, C = input_nhwc.shape
    # Repeat along height and width
    x = mx.repeat(input_nhwc, scale_factor, axis=1)  # (N, H*2, W, C)
    x = mx.repeat(x, scale_factor, axis=2)           # (N, H*2, W*2, C)
    return x
```

---

## Implementation Notes

### Internal Structure

The pooling modules follow a common pattern:

```python
class _MaxPoolNd(Module):
    """Base class for all max pooling layers."""
    __constants__ = [
        "kernel_size", "stride", "padding",
        "dilation", "return_indices", "ceil_mode"
    ]

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
```

### Backend Dispatch

Pooling operations dispatch to optimized backends:
- **CPU**: Vectorized C++ kernels
- **CUDA**: cuDNN pooling operations
- **MPS**: Metal Performance Shaders

### Performance Considerations

1. **Overlapping vs Non-overlapping**: Non-overlapping (`stride=kernel_size`) is more efficient
2. **Dilation**: Rarely used for pooling; increases computation
3. **Return indices**: Adds memory overhead for storing indices

---

## Implementation Files

**Module Definitions**:
- `torch/nn/modules/pooling.py` - All pooling layer classes

**Functional API**:
- `torch/nn/functional.py` - Pooling functions

**Native Functions**:
- `aten/src/ATen/native/native_functions.yaml` - Function declarations

**CPU Kernels**:
- `aten/src/ATen/native/cpu/MaxPoolKernel.cpp`
- `aten/src/ATen/native/cpu/AvgPoolKernel.cpp`
- `aten/src/ATen/native/AdaptiveAveragePooling.cpp`
- `aten/src/ATen/native/AdaptiveMaxPooling.cpp`

**MPS Kernels**:
- `aten/src/ATen/native/mps/operations/Pooling.mm`

**CUDA Kernels**:
- `aten/src/ATen/native/cuda/AveragePool2d.cu`
- `aten/src/ATen/native/cuda/MaxPool2d.cu`
- `aten/src/ATen/native/cudnn/Pooling.cpp`

**Gradients**:
- `tools/autograd/derivatives.yaml` - Backward definitions
