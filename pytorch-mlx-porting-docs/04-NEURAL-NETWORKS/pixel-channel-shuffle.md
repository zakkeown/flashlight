# PixelShuffle, PixelUnshuffle, and ChannelShuffle

## Overview

These modules perform spatial-channel rearrangement operations commonly used in:
- **Super-resolution networks** (PixelShuffle/PixelUnshuffle)
- **Efficient network architectures** like ShuffleNet (ChannelShuffle)

**Reference Files:**
- `torch/nn/modules/pixelshuffle.py` - PixelShuffle, PixelUnshuffle modules
- `torch/nn/modules/channelshuffle.py` - ChannelShuffle module
- `torch/nn/functional.py` - Functional implementations
- `torch/_refs/nn/functional/__init__.py` - Reference implementations

---

## PixelShuffle

### Purpose

Rearranges elements from channel dimension to spatial dimensions, effectively upscaling the spatial resolution. This is the key operation in sub-pixel convolution for super-resolution.

**Paper:** [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158) (Shi et al., 2016)

### Shape Transformation

```
Input:  (*, C × r², H, W)
Output: (*, C, H × r, W × r)
```

Where `r` is the upscale factor.

### Algorithm

The operation can be understood as:
1. Reshape: `(N, C*r², H, W)` → `(N, C, r, r, H, W)`
2. Permute: `(N, C, r, r, H, W)` → `(N, C, H, r, W, r)`
3. Reshape: `(N, C, H, r, W, r)` → `(N, C, H*r, W*r)`

**Pseudocode:**
```python
def pixel_shuffle(input, upscale_factor):
    r = upscale_factor
    batch, channels, height, width = input.shape

    # Channels must be divisible by r²
    assert channels % (r * r) == 0
    out_channels = channels // (r * r)

    # Reshape to expose the r×r blocks
    x = input.reshape(batch, out_channels, r, r, height, width)

    # Permute to interleave spatial dimensions
    x = x.permute(0, 1, 4, 2, 5, 3)  # (N, C, H, r, W, r)

    # Flatten to final shape
    return x.reshape(batch, out_channels, height * r, width * r)
```

### PyTorch API

```python
import torch.nn as nn

# Module form
pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

# Input: (1, 8, 4, 4) - 8 channels, 4×4 spatial
input = torch.randn(1, 8, 4, 4)

# Output: (1, 2, 8, 8) - 2 channels, 8×8 spatial
# 8 channels → 8 / (2²) = 2 channels
# 4×4 spatial → 4×2 × 4×2 = 8×8 spatial
output = pixel_shuffle(input)

# Functional form
output = torch.nn.functional.pixel_shuffle(input, upscale_factor=2)
```

### Example: Super-Resolution Upsampling

```python
class SubPixelConv(nn.Module):
    """Efficient upsampling via sub-pixel convolution."""

    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        # Conv outputs r² times more channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * upscale_factor ** 2,
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)  # (N, C*r², H, W)
        x = self.pixel_shuffle(x)  # (N, C, H*r, W*r)
        return x

# Example: 2× upsampling
upsample = SubPixelConv(64, 64, upscale_factor=2)
x = torch.randn(1, 64, 32, 32)
y = upsample(x)  # (1, 64, 64, 64)
```

---

## PixelUnshuffle

### Purpose

The inverse of PixelShuffle. Rearranges elements from spatial dimensions to channel dimension, effectively downscaling the spatial resolution while increasing channels.

### Shape Transformation

```
Input:  (*, C, H × r, W × r)
Output: (*, C × r², H, W)
```

Where `r` is the downscale factor.

### Algorithm

```python
def pixel_unshuffle(input, downscale_factor):
    r = downscale_factor
    batch, channels, height, width = input.shape

    # Spatial dims must be divisible by r
    assert height % r == 0 and width % r == 0

    out_height = height // r
    out_width = width // r
    out_channels = channels * r * r

    # Reshape to expose the r×r blocks
    x = input.reshape(batch, channels, out_height, r, out_width, r)

    # Permute to group r×r blocks into channels
    x = x.permute(0, 1, 3, 5, 2, 4)  # (N, C, r, r, H, W)

    # Flatten to final shape
    return x.reshape(batch, out_channels, out_height, out_width)
```

### PyTorch API

```python
# Module form
pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=3)

# Input: (1, 1, 12, 12)
input = torch.randn(1, 1, 12, 12)

# Output: (1, 9, 4, 4)
# 1 channel → 1 × 3² = 9 channels
# 12×12 spatial → 12/3 × 12/3 = 4×4 spatial
output = pixel_unshuffle(input)
```

### Round-Trip Property

```python
r = 2
x = torch.randn(1, 4, 8, 8)

# Unshuffle then shuffle recovers original
y = F.pixel_unshuffle(x, r)  # (1, 16, 4, 4)
z = F.pixel_shuffle(y, r)     # (1, 4, 8, 8)
assert torch.allclose(x, z)

# Shuffle then unshuffle also recovers (with correct input shape)
x = torch.randn(1, 16, 4, 4)
y = F.pixel_shuffle(x, r)     # (1, 4, 8, 8)
z = F.pixel_unshuffle(y, r)   # (1, 16, 4, 4)
assert torch.allclose(x, z)
```

---

## ChannelShuffle

### Purpose

Shuffles channels to enable information exchange between channel groups. Essential for efficient architectures like ShuffleNet where grouped convolutions would otherwise isolate channel groups.

**Paper:** [ShuffleNet: An Extremely Computation-Efficient CNN for Mobile Devices](https://arxiv.org/abs/1707.01083) (Zhang et al., 2017)

### Algorithm

Given `g` groups, the operation:
1. Reshape channels into groups: `(N, C, H, W)` → `(N, g, C/g, H, W)`
2. Transpose groups and channels: `(N, g, C/g, H, W)` → `(N, C/g, g, H, W)`
3. Flatten back: `(N, C/g, g, H, W)` → `(N, C, H, W)`

This effectively interleaves channels from different groups.

**Pseudocode:**
```python
def channel_shuffle(input, groups):
    batch, channels, height, width = input.shape

    # Channels must be divisible by groups
    assert channels % groups == 0
    channels_per_group = channels // groups

    # Reshape: (N, C, H, W) -> (N, g, C/g, H, W)
    x = input.reshape(batch, groups, channels_per_group, height, width)

    # Transpose groups and channels_per_group: (N, g, C/g, H, W) -> (N, C/g, g, H, W)
    x = x.permute(0, 2, 1, 3, 4)

    # Flatten: (N, C/g, g, H, W) -> (N, C, H, W)
    return x.reshape(batch, channels, height, width)
```

### Visualization

```
Before shuffle (4 channels, 2 groups):
Group 0: [C0, C1]
Group 1: [C2, C3]

After shuffle:
Group 0: [C0, C2]  (first from each original group)
Group 1: [C1, C3]  (second from each original group)
```

### PyTorch API

```python
import torch.nn as nn

# Create module
channel_shuffle = nn.ChannelShuffle(groups=2)

# Input with 4 channels
input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
# Channels: [[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]], [[13,14],[15,16]]]

output = channel_shuffle(input)
# After shuffle with groups=2:
# Channels: [[[1,2],[3,4]], [[9,10],[11,12]], [[5,6],[7,8]], [[13,14],[15,16]]]
# Note: channels 0,2 swapped with 1,3 positions
```

### Example: ShuffleNet Block

```python
class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        mid_channels = out_channels // 4

        # 1×1 group conv (reduces channels)
        self.gconv1 = nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Channel shuffle
        self.shuffle = nn.ChannelShuffle(groups)

        # 3×3 depthwise conv
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                                 groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1×1 group conv (expands channels)
        self.gconv2 = nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.gconv1(x)))
        out = self.shuffle(out)  # Mix information between groups
        out = self.bn2(self.dwconv(out))
        out = self.bn3(self.gconv2(out))
        return F.relu(out + x)  # Residual connection
```

---

## Gradient Formulas

### PixelShuffle Backward

The backward pass is simply `PixelUnshuffle`:

```python
# Forward: pixel_shuffle(x, r) -> y
# Backward: pixel_unshuffle(grad_y, r) -> grad_x

def pixel_shuffle_backward(grad_output, upscale_factor):
    return pixel_unshuffle(grad_output, upscale_factor)
```

### PixelUnshuffle Backward

The backward pass is simply `PixelShuffle`:

```python
def pixel_unshuffle_backward(grad_output, downscale_factor):
    return pixel_shuffle(grad_output, downscale_factor)
```

### ChannelShuffle Backward

Channel shuffle is its own inverse when applied twice with proper group calculation:

```python
def channel_shuffle_backward(grad_output, groups):
    # Inverse shuffle is the same operation with channels_per_group as the group count
    batch, channels, height, width = grad_output.shape
    channels_per_group = channels // groups

    # The inverse permutation
    x = grad_output.reshape(batch, channels_per_group, groups, height, width)
    x = x.permute(0, 2, 1, 3, 4)
    return x.reshape(batch, channels, height, width)
```

---

## MLX Implementation

### PixelShuffle for MLX

```python
import mlx.core as mx

def pixel_shuffle_mlx(x, upscale_factor):
    """
    Rearrange (*, C*r², H, W) -> (*, C, H*r, W*r)

    Note: MLX uses NHWC layout by default, so we work with that.
    For NCHW input, transpose first.
    """
    r = upscale_factor

    # Assuming NCHW input (PyTorch convention)
    batch, channels, height, width = x.shape

    assert channels % (r * r) == 0, f"Channels {channels} not divisible by {r*r}"
    out_channels = channels // (r * r)

    # Reshape to (N, C, r, r, H, W)
    x = mx.reshape(x, (batch, out_channels, r, r, height, width))

    # Permute to (N, C, H, r, W, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))

    # Reshape to (N, C, H*r, W*r)
    return mx.reshape(x, (batch, out_channels, height * r, width * r))


def pixel_shuffle_mlx_nhwc(x, upscale_factor):
    """
    NHWC version: (N, H, W, C*r²) -> (N, H*r, W*r, C)

    This is more natural for MLX's default layout.
    """
    r = upscale_factor
    batch, height, width, channels = x.shape

    assert channels % (r * r) == 0
    out_channels = channels // (r * r)

    # Reshape to (N, H, W, C, r, r)
    x = mx.reshape(x, (batch, height, width, out_channels, r, r))

    # Permute to (N, H, r, W, r, C)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))

    # Reshape to (N, H*r, W*r, C)
    return mx.reshape(x, (batch, height * r, width * r, out_channels))
```

### PixelUnshuffle for MLX

```python
def pixel_unshuffle_mlx(x, downscale_factor):
    """
    Rearrange (*, C, H*r, W*r) -> (*, C*r², H, W)
    """
    r = downscale_factor
    batch, channels, height, width = x.shape

    assert height % r == 0 and width % r == 0
    out_height = height // r
    out_width = width // r
    out_channels = channels * r * r

    # Reshape to (N, C, H, r, W, r)
    x = mx.reshape(x, (batch, channels, out_height, r, out_width, r))

    # Permute to (N, C, r, r, H, W)
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))

    # Reshape to (N, C*r², H, W)
    return mx.reshape(x, (batch, out_channels, out_height, out_width))


def pixel_unshuffle_mlx_nhwc(x, downscale_factor):
    """
    NHWC version: (N, H*r, W*r, C) -> (N, H, W, C*r²)
    """
    r = downscale_factor
    batch, height, width, channels = x.shape

    assert height % r == 0 and width % r == 0
    out_height = height // r
    out_width = width // r
    out_channels = channels * r * r

    # Reshape to (N, H, r, W, r, C)
    x = mx.reshape(x, (batch, out_height, r, out_width, r, channels))

    # Permute to (N, H, W, C, r, r)
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))

    # Reshape to (N, H, W, C*r²)
    return mx.reshape(x, (batch, out_height, out_width, out_channels))
```

### ChannelShuffle for MLX

```python
def channel_shuffle_mlx(x, groups):
    """
    Shuffle channels between groups (NCHW layout).
    """
    batch, channels, height, width = x.shape

    assert channels % groups == 0, f"Channels {channels} not divisible by groups {groups}"
    channels_per_group = channels // groups

    # Reshape: (N, C, H, W) -> (N, g, C/g, H, W)
    x = mx.reshape(x, (batch, groups, channels_per_group, height, width))

    # Transpose: (N, g, C/g, H, W) -> (N, C/g, g, H, W)
    x = mx.transpose(x, (0, 2, 1, 3, 4))

    # Flatten: (N, C/g, g, H, W) -> (N, C, H, W)
    return mx.reshape(x, (batch, channels, height, width))


def channel_shuffle_mlx_nhwc(x, groups):
    """
    Shuffle channels between groups (NHWC layout).
    """
    batch, height, width, channels = x.shape

    assert channels % groups == 0
    channels_per_group = channels // groups

    # Reshape: (N, H, W, C) -> (N, H, W, g, C/g)
    x = mx.reshape(x, (batch, height, width, groups, channels_per_group))

    # Transpose: (N, H, W, g, C/g) -> (N, H, W, C/g, g)
    x = mx.transpose(x, (0, 1, 2, 4, 3))

    # Flatten: (N, H, W, C/g, g) -> (N, H, W, C)
    return mx.reshape(x, (batch, height, width, channels))
```

### MLX Module Classes

```python
import mlx.nn as nn

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def __call__(self, x):
        return pixel_shuffle_mlx_nhwc(x, self.upscale_factor)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = downscale_factor

    def __call__(self, x):
        return pixel_unshuffle_mlx_nhwc(x, self.downscale_factor)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def __call__(self, x):
        return channel_shuffle_mlx_nhwc(x, self.groups)
```

---

## Key Differences: PyTorch vs MLX

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Default layout | NCHW | NHWC |
| In-place ops | Supported | Not supported |
| Implementation | C++/CUDA kernels | Reshape + transpose |
| Gradient | Automatic | Automatic via transforms |

---

## Common Use Cases

### 1. Super-Resolution Networks (ESPCN, EDSR)

```python
# Efficient 4× upsampling
upscale = nn.Sequential(
    nn.Conv2d(64, 64 * 16, 3, padding=1),  # 64 -> 1024 channels
    nn.PixelShuffle(4),                      # 1024 -> 64 channels, 4× spatial
    nn.ReLU()
)
```

### 2. ShuffleNet Blocks

```python
# Grouped conv + shuffle pattern
def shufflenet_unit(x, groups=4):
    x = group_conv(x, groups=groups)
    x = channel_shuffle(x, groups=groups)  # Mix information
    x = depthwise_conv(x)
    return x
```

### 3. Feature Pyramid Downsampling

```python
# Downsample while preserving information
downsample = nn.Sequential(
    nn.PixelUnshuffle(2),  # 2× downsample, 4× channels
    nn.Conv2d(in_ch * 4, out_ch, 1)  # Reduce channels
)
```

---

## Performance Considerations

1. **Memory efficiency**: PixelShuffle is more memory-efficient than transposed convolution for upsampling
2. **No learned parameters**: All three operations are parameter-free
3. **Reshape/transpose only**: Operations are cheap - just memory rearrangement
4. **NHWC layout**: MLX's native layout may require transposition when porting NCHW PyTorch models

---

## Summary Table

| Operation | Input Shape | Output Shape | Use Case |
|-----------|-------------|--------------|----------|
| PixelShuffle(r) | (N, C×r², H, W) | (N, C, H×r, W×r) | Upsampling |
| PixelUnshuffle(r) | (N, C, H×r, W×r) | (N, C×r², H, W) | Downsampling |
| ChannelShuffle(g) | (N, C, H, W) | (N, C, H, W) | Group mixing |
