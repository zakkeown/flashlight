# PyTorch Convolutional Layers

## Purpose

This document provides comprehensive documentation of PyTorch's convolutional neural network modules. Convolutions are fundamental building blocks for:

1. Computer vision (image classification, object detection, segmentation)
2. Audio processing (speech recognition, music generation)
3. Time series analysis
4. Natural language processing (1D convolutions for text)

**Source**: [torch/nn/modules/conv.py](../../reference/pytorch/torch/nn/modules/conv.py)

## Architecture Overview

### Convolution Module Hierarchy

```
                              Module
                                 |
                            _ConvNd (base)
                 ________________|________________
                |                                 |
    ┌───────────┼───────────┐         ┌──────────┴──────────┐
   Conv1d    Conv2d    Conv3d    _ConvTransposeNd
                                        |
                              ┌─────────┼─────────┐
                         ConvTranspose1d  ConvTranspose2d  ConvTranspose3d

    Lazy Variants (in_channels inferred at runtime):
    LazyConv1d, LazyConv2d, LazyConv3d
    LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d
```

### Convolution vs Transposed Convolution

| Type | Purpose | Spatial Effect | Use Case |
|------|---------|----------------|----------|
| **Convolution** | Feature extraction | Typically reduces spatial size | Encoders, classification |
| **Transposed Convolution** | Learned upsampling | Increases spatial size | Decoders, segmentation, GANs |

---

## 1. Base Class: _ConvNd

The abstract base class for all convolutional modules.

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_channels` | int | Number of channels in the input |
| `out_channels` | int | Number of channels produced by the convolution |
| `kernel_size` | int or tuple | Size of the convolving kernel |
| `stride` | int or tuple | Stride of the convolution (default: 1) |
| `padding` | int, tuple, or str | Padding added to input (default: 0) |
| `dilation` | int or tuple | Spacing between kernel elements (default: 1) |
| `groups` | int | Number of blocked connections (default: 1) |
| `bias` | bool | If True, adds learnable bias (default: True) |
| `padding_mode` | str | `'zeros'`, `'reflect'`, `'replicate'`, `'circular'` |

### Weight Initialization

All convolutional layers use Kaiming uniform initialization:

```python
def reset_parameters(self) -> None:
    # Kaiming uniform with a=sqrt(5) is equivalent to
    # uniform(-1/sqrt(k), 1/sqrt(k)) where k = weight.size(1) * prod(kernel_size)
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
```

### Groups Explained

- **groups=1**: Standard convolution, all inputs connected to all outputs
- **groups=2**: Two parallel convolutions, each seeing half the channels
- **groups=in_channels**: Depthwise convolution (each channel convolved separately)

```
groups=1 (standard):           groups=2:                  groups=in_channels (depthwise):
  in_channels                    in_channels                  in_channels
      |                          /        \                    | | | | |
      v                         v          v                   v v v v v
  [  Conv  ] ──> out_channels  [Conv1][Conv2] ──> out         [C][C][C][C][C]
                                                               | | | | |
                                                               v v v v v
                                                              out_channels
```

---

## 2. Conv1d

Applies a 1D convolution over an input signal composed of several input planes.

### Formula

```
out(N_i, C_out_j) = bias(C_out_j) + Σ_{k=0}^{C_in-1} weight(C_out_j, k) ⋆ input(N_i, k)
```

Where ⋆ is the valid cross-correlation operator.

### Constructor

```python
nn.Conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] | str = 0,
    dilation: int | tuple[int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',  # 'zeros', 'reflect', 'replicate', 'circular'
    device=None,
    dtype=None,
)
```

### Shape

- **Input**: `(N, C_in, L_in)` or `(C_in, L_in)` (unbatched)
- **Output**: `(N, C_out, L_out)` or `(C_out, L_out)`

### Output Size Formula

```
L_out = floor((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
```

### Weight Shape

```
weight: (out_channels, in_channels // groups, kernel_size)
bias: (out_channels,)
```

### Example

```python
# Basic 1D convolution
conv = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2)
input = torch.randn(20, 16, 50)  # (batch, channels, length)
output = conv(input)  # (20, 33, 24)

# With padding='same' (output size = input size when stride=1)
conv_same = nn.Conv1d(16, 33, kernel_size=3, padding='same')
output = conv_same(input)  # (20, 33, 50)

# Depthwise convolution
conv_dw = nn.Conv1d(16, 16, kernel_size=3, groups=16)  # Each channel separate
```

---

## 3. Conv2d

Applies a 2D convolution over an input image composed of several input planes.

### Formula

```
out(N_i, C_out_j) = bias(C_out_j) + Σ_{k=0}^{C_in-1} weight(C_out_j, k) ⋆ input(N_i, k)
```

### Constructor

```python
nn.Conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | str = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
)
```

### Shape

- **Input**: `(N, C_in, H_in, W_in)` or `(C_in, H_in, W_in)`
- **Output**: `(N, C_out, H_out, W_out)` or `(C_out, H_out, W_out)`

### Output Size Formula

```
H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
W_out = floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
```

### Weight Shape

```
weight: (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
bias: (out_channels,)
```

### Example

```python
# Square kernels and equal stride
conv = nn.Conv2d(16, 33, kernel_size=3, stride=2)
input = torch.randn(20, 16, 50, 100)  # (batch, channels, height, width)
output = conv(input)  # (20, 33, 24, 49)

# Non-square kernels with padding and dilation
conv = nn.Conv2d(16, 33, kernel_size=(3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

# Depthwise separable convolution (MobileNet style)
depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
pointwise = nn.Conv2d(32, 64, kernel_size=1)
# Forward: output = pointwise(depthwise(input))
```

### Common Patterns

**Same Padding** (output same size as input when stride=1):
```python
conv = nn.Conv2d(3, 64, kernel_size=3, padding='same')
# Or manually for kernel_size=3: padding=1
# For kernel_size=5: padding=2
# Formula: padding = (kernel_size - 1) // 2
```

**Strided Convolution** (downsampling):
```python
conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# Halves spatial dimensions (common in ResNet)
```

---

## 3. Conv3d

Applies a 3D convolution over an input signal composed of several input planes.

### Constructor

```python
nn.Conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] | str = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
)
```

### Shape

- **Input**: `(N, C_in, D_in, H_in, W_in)` or `(C_in, D_in, H_in, W_in)`
- **Output**: `(N, C_out, D_out, H_out, W_out)` or `(C_out, D_out, H_out, W_out)`

### Output Size Formula

```
D_out = floor((D_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
H_out = floor((H_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
W_out = floor((W_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1)
```

### Weight Shape

```
weight: (out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2])
bias: (out_channels,)
```

### Example

```python
# Video processing (depth = temporal dimension)
conv = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
video = torch.randn(1, 3, 16, 224, 224)  # (batch, channels, frames, height, width)
output = conv(video)  # (1, 64, 16, 112, 112)

# Medical imaging (3D volumes)
conv = nn.Conv3d(1, 32, kernel_size=3, padding=1)
ct_scan = torch.randn(1, 1, 64, 256, 256)  # (batch, channels, depth, height, width)
```

---

## 4. ConvTranspose1d

Applies a 1D transposed convolution (also known as fractionally-strided convolution or deconvolution).

### Purpose

- Learned upsampling
- Gradient of Conv1d with respect to its input
- Decoder in autoencoders

### Constructor

```python
nn.ConvTranspose1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
    output_padding: int | tuple[int] = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int | tuple[int] = 1,
    padding_mode: str = 'zeros',  # Only 'zeros' supported
    device=None,
    dtype=None,
)
```

### Shape

- **Input**: `(N, C_in, L_in)` or `(C_in, L_in)`
- **Output**: `(N, C_out, L_out)` or `(C_out, L_out)`

### Output Size Formula

```
L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
```

### Weight Shape (Note: Transposed!)

```
weight: (in_channels, out_channels // groups, kernel_size)
bias: (out_channels,)
```

### Example

```python
# Upsampling
upsample = nn.ConvTranspose1d(16, 16, kernel_size=3, stride=2, padding=1)
input = torch.randn(1, 16, 12)
output = upsample(input)  # Approximately doubles length

# Exact output size specification (useful when stride > 1)
downsample = nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=1)
upsample = nn.ConvTranspose1d(16, 16, kernel_size=3, stride=2, padding=1)

input = torch.randn(1, 16, 12)
h = downsample(input)  # (1, 16, 6)
output = upsample(h, output_size=input.size())  # (1, 16, 12) - exact match
```

---

## 5. ConvTranspose2d

Applies a 2D transposed convolution operator over an input image.

### Constructor

```python
nn.ConvTranspose2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int | tuple[int, int] = 1,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
)
```

### Shape

- **Input**: `(N, C_in, H_in, W_in)` or `(C_in, H_in, W_in)`
- **Output**: `(N, C_out, H_out, W_out)` or `(C_out, H_out, W_out)`

### Output Size Formula

```
H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
```

### Example

```python
# Basic upsampling (2x spatial size)
upsample = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
input = torch.randn(1, 64, 16, 16)
output = upsample(input)  # (1, 32, 32, 32)

# U-Net style decoder
class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # After concat

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # Skip connection
        return self.conv(x)

# GAN generator upsampling
gen_block = nn.Sequential(
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True)
)
```

### Output Padding Explained

When `stride > 1`, multiple input sizes can map to the same output size in Conv2d. `output_padding` resolves this ambiguity:

```python
# Conv2d with stride=2 maps 12x12 and 13x13 both to 6x6
# ConvTranspose2d needs output_padding to specify which size to produce
conv = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
deconv = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1)

input_12 = torch.randn(1, 16, 12, 12)
input_13 = torch.randn(1, 16, 13, 13)

h_12 = conv(input_12)  # (1, 16, 6, 6)
h_13 = conv(input_13)  # (1, 16, 6, 6) - same output size!

# Use output_size to recover original
out_12 = deconv(h_12, output_size=input_12.size())  # (1, 16, 12, 12)
out_13 = deconv(h_13, output_size=input_13.size())  # (1, 16, 13, 13)
```

---

## 6. ConvTranspose3d

Applies a 3D transposed convolution operator over an input image composed of several input planes.

### Constructor

```python
nn.ConvTranspose3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int | tuple[int, int, int] = 1,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
)
```

### Shape

- **Input**: `(N, C_in, D_in, H_in, W_in)` or `(C_in, D_in, H_in, W_in)`
- **Output**: `(N, C_out, D_out, H_out, W_out)` or `(C_out, D_out, H_out, W_out)`

### Example

```python
# 3D U-Net decoder
upsample = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
input = torch.randn(1, 128, 8, 16, 16)
output = upsample(input)  # (1, 64, 16, 32, 32)
```

---

## 7. Lazy Convolutions

Lazy modules defer weight initialization until the first forward pass, automatically inferring `in_channels`.

### Available Lazy Variants

- `LazyConv1d`, `LazyConv2d`, `LazyConv3d`
- `LazyConvTranspose1d`, `LazyConvTranspose2d`, `LazyConvTranspose3d`

### Constructor (LazyConv2d Example)

```python
nn.LazyConv2d(
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',
    device=None,
    dtype=None,
)
# Note: No in_channels parameter!
```

### Example

```python
# in_channels inferred from first input
lazy_conv = nn.LazyConv2d(64, kernel_size=3, padding=1)
print(lazy_conv.weight)  # UninitializedParameter

input = torch.randn(1, 3, 224, 224)  # 3 channels
output = lazy_conv(input)

print(lazy_conv.weight.shape)  # (64, 3, 3, 3) - now initialized!
print(type(lazy_conv))  # Still LazyConv2d, but cls_to_become = Conv2d
```

### Use Cases

- Dynamic architectures where input channels vary
- Prototyping without knowing input shapes
- Avoiding manual channel tracking

---

## 8. Padding Modes

### Available Modes

| Mode | Description | Edge Behavior |
|------|-------------|---------------|
| `'zeros'` | Zero padding (default) | Pad with 0s |
| `'reflect'` | Reflection padding | Mirror at edges |
| `'replicate'` | Replication padding | Repeat edge values |
| `'circular'` | Circular padding | Wrap around |

### Visual Example (1D, padding=2)

```
Input:        [1, 2, 3, 4, 5]

zeros:        [0, 0, 1, 2, 3, 4, 5, 0, 0]
reflect:      [3, 2, 1, 2, 3, 4, 5, 4, 3]
replicate:    [1, 1, 1, 2, 3, 4, 5, 5, 5]
circular:     [4, 5, 1, 2, 3, 4, 5, 1, 2]
```

### Usage

```python
# Non-zero padding mode requires F.pad + convolution
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, padding_mode='reflect')
```

**Note**: `ConvTranspose*d` only supports `'zeros'` padding mode.

---

## 9. Dilation (Atrous Convolution)

Dilation inserts gaps between kernel elements, increasing the receptive field without increasing parameters.

### Visual Example (2D kernel with dilation)

```
dilation=1 (standard):        dilation=2:               dilation=3:
[x x x]                       [x . x . x]               [x . . x . . x]
[x x x]                       [. . . . .]               [. . . . . . .]
[x x x]                       [x . x . x]               [. . . . . . .]
                              [. . . . .]               [x . . x . . x]
                              [x . x . x]               [. . . . . . .]
                                                        [. . . . . . .]
                                                        [x . . x . . x]

Effective kernel size: 3x3    Effective: 5x5            Effective: 7x7
Parameters: 9                 Parameters: 9              Parameters: 9
```

### Use Cases

- Semantic segmentation (DeepLab)
- Increasing receptive field without pooling
- Multi-scale feature extraction

```python
# Dilated convolution for large receptive field
conv = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
# Effective receptive field of 5x5 with only 3x3 kernel
```

---

## MLX Mapping

### Direct Mappings

```python
# PyTorch                           # MLX
nn.Conv1d(in, out, k)               mlx.nn.Conv1d(in, out, k)
nn.Conv2d(in, out, k)               mlx.nn.Conv2d(in, out, k)
nn.Conv1d(in, out, k, groups=in)    mlx.nn.Conv1d(in, out, k, groups=in)  # Depthwise
```

### MLX Implementation Notes

MLX convolutions differ in a few ways:
- Use `(N, H, W, C)` channel-last format by default (vs PyTorch's `(N, C, H, W)`)
- May require explicit `mx.transpose` for compatibility

### Example MLX Conv2d

```python
import mlx.core as mx
import mlx.nn as nn
import math

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Weight shape: (out_channels, kH, kW, in_channels // groups)
        # MLX uses channels-last by default
        fan_in = (in_channels // groups) * kernel_size[0] * kernel_size[1]
        bound = 1 / math.sqrt(fan_in)

        self.weight = mx.random.uniform(
            low=-bound, high=bound,
            shape=(out_channels, kernel_size[0], kernel_size[1], in_channels // groups)
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x):
        # x shape: (N, H, W, C) in MLX
        y = mx.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        return y
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Channel order** | NCHW | NHWC (default) |
| **Weight shape** | (out, in/g, kH, kW) | (out, kH, kW, in/g) |
| **padding_mode** | 4 modes | Limited |
| **Lazy variants** | Built-in | Not built-in |
| **Complex tensors** | Supported | Not supported |

### Conversion Pattern

```python
def convert_pytorch_conv2d(pt_conv):
    """Convert PyTorch Conv2d weights to MLX format"""
    # PyTorch: (out, in, kH, kW) -> MLX: (out, kH, kW, in)
    weight = pt_conv.weight.detach().numpy()
    weight_mlx = mx.array(weight.transpose(0, 2, 3, 1))

    bias_mlx = None
    if pt_conv.bias is not None:
        bias_mlx = mx.array(pt_conv.bias.detach().numpy())

    return weight_mlx, bias_mlx
```

---

## Common Patterns

### ResNet Block

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
```

### Depthwise Separable Convolution

```python
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.bn(x))
```

---

## Summary

### Quick Reference

| Module | Input Shape | Output Shape | Use Case |
|--------|-------------|--------------|----------|
| Conv1d | (N, C, L) | (N, C', L') | Audio, time series |
| Conv2d | (N, C, H, W) | (N, C', H', W') | Images |
| Conv3d | (N, C, D, H, W) | (N, C', D', H', W') | Video, 3D volumes |
| ConvTranspose1d | (N, C, L) | (N, C', L') | 1D upsampling |
| ConvTranspose2d | (N, C, H, W) | (N, C', H', W') | Image upsampling, decoder |
| ConvTranspose3d | (N, C, D, H, W) | (N, C', D', H', W') | 3D upsampling |

### Parameter Count

```
Standard Conv2d:  out_channels * (in_channels/groups * kH * kW) + out_channels
                = out * in/g * k^2 + out  (for square kernel)

Example: Conv2d(64, 128, 3) = 128 * 64 * 9 + 128 = 73,856 parameters
```

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/conv.py`

**Functional API**:
- `F.conv1d`, `F.conv2d`, `F.conv3d`
- `F.conv_transpose1d`, `F.conv_transpose2d`, `F.conv_transpose3d`

**YAML Definitions** (`aten/src/ATen/native/native_functions.yaml`):
- `conv1d`, `conv2d`, `conv3d`
- `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`
