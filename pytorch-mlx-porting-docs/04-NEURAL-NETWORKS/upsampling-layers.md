# PyTorch Upsampling Layers

## Purpose

This document provides comprehensive documentation of PyTorch's upsampling modules. Upsampling is essential for:

1. Decoder networks (autoencoders, VAEs)
2. Semantic segmentation (U-Net, DeepLab)
3. Super-resolution
4. Generative models (GANs)
5. Feature pyramid networks

**Source**: [torch/nn/modules/upsampling.py](../../reference/pytorch/torch/nn/modules/upsampling.py)

## Architecture Overview

### Upsampling Module Hierarchy

```
                    Module
                       |
                   Upsample
                 _____|_____
                |           |
    UpsamplingNearest2d  UpsamplingBilinear2d
         (deprecated)       (deprecated)
```

### Upsampling Methods Comparison

| Method | Algorithm | Quality | Speed | Learnable |
|--------|-----------|---------|-------|-----------|
| **Nearest** | Copy nearest pixel | Blocky | Fastest | No |
| **Linear** (1D) | Linear interpolation | Smooth | Fast | No |
| **Bilinear** (2D) | 2D linear interpolation | Smooth | Fast | No |
| **Bicubic** (2D) | Cubic interpolation | Smoothest | Slower | No |
| **Trilinear** (3D) | 3D linear interpolation | Smooth | Fast | No |
| **ConvTranspose** | Learned upsampling | Best | Variable | Yes |

---

## 1. Upsample

The main upsampling module supporting 1D, 2D, and 3D inputs with various interpolation modes.

### Constructor

```python
nn.Upsample(
    size: int | tuple = None,           # Target output size
    scale_factor: float | tuple = None,  # Multiplicative factor
    mode: str = 'nearest',               # Interpolation algorithm
    align_corners: bool = None,          # Corner alignment (see below)
    recompute_scale_factor: bool = None, # Recompute scale from size
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | int or tuple | Target output spatial size |
| `scale_factor` | float or tuple | Multiplier for spatial dimensions |
| `mode` | str | `'nearest'`, `'linear'`, `'bilinear'`, `'bicubic'`, `'trilinear'` |
| `align_corners` | bool | Align corner pixels (for linear modes) |
| `recompute_scale_factor` | bool | Recompute scale from computed size |

**Note**: Specify either `size` OR `scale_factor`, not both.

### Mode Compatibility

| Mode | Input Dims | Valid For |
|------|------------|-----------|
| `nearest` | 3D, 4D, 5D | All dimensions |
| `linear` | 3D only | 1D temporal |
| `bilinear` | 4D only | 2D spatial |
| `bicubic` | 4D only | 2D spatial |
| `trilinear` | 5D only | 3D volumetric |

### Shape

- **Input**: `(N, C, W)`, `(N, C, H, W)`, or `(N, C, D, H, W)`
- **Output**: `(N, C, W')`, `(N, C, H', W')`, or `(N, C, D', H', W')`

```
Output size = floor(Input size * scale_factor)
```

### Examples

```python
# 2x upsampling with nearest neighbor
upsample = nn.Upsample(scale_factor=2, mode='nearest')
input = torch.randn(1, 3, 32, 32)
output = upsample(input)  # (1, 3, 64, 64)

# Bilinear upsampling to specific size
upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
output = upsample(input)  # (1, 3, 128, 128)

# 3D upsampling for video/volumetric data
upsample_3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
volume = torch.randn(1, 1, 16, 32, 32)
output = upsample_3d(volume)  # (1, 1, 32, 64, 64)
```

---

## 2. Interpolation Modes

### Nearest Neighbor

Copies the value of the nearest input pixel. Fast but produces blocky results.

```
Input:              Upsample 2x:
[1, 2]              [1, 1, 2, 2]
[3, 4]     →        [1, 1, 2, 2]
                    [3, 3, 4, 4]
                    [3, 3, 4, 4]
```

```python
upsample = nn.Upsample(scale_factor=2, mode='nearest')
```

### Bilinear Interpolation

Linearly interpolates in both spatial dimensions. Smoother results.

```
Input:              Upsample 2x (align_corners=False):
[1, 2]              [1.00, 1.25, 1.75, 2.00]
[3, 4]     →        [1.50, 1.75, 2.25, 2.50]
                    [2.50, 2.75, 3.25, 3.50]
                    [3.00, 3.25, 3.75, 4.00]
```

```python
upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
```

### Bicubic Interpolation

Uses cubic interpolation for smoother results, especially on natural images.

```python
upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
# Better for super-resolution tasks
```

---

## 3. align_corners Explained

The `align_corners` parameter affects how input and output pixels are mapped.

### align_corners=False (Default)

Pixels are assumed to be centered in their grid cells. Output corners can differ from input corners.

```
Input 2x2:                    Output 4x4:
┌─────┬─────┐                 ┌──┬──┬──┬──┐
│  1  │  2  │                 │1.0│1.25│1.75│2.0│
├─────┼─────┤   ───────→      ├──┼──┼──┼──┤
│  3  │  4  │                 │1.5│1.75│2.25│2.5│
└─────┴─────┘                 ├──┼──┼──┼──┤
                              │2.5│2.75│3.25│3.5│
Centers matter                ├──┼──┼──┼──┤
                              │3.0│3.25│3.75│4.0│
                              └──┴──┴──┴──┘
```

### align_corners=True

Corner pixels of input and output are aligned exactly.

```
Input 2x2:                    Output 4x4:
┌─────┬─────┐                 ┌──┬──┬──┬──┐
│  1  │  2  │                 │1.0│1.33│1.67│2.0│
├─────┼─────┤   ───────→      ├──┼──┼──┼──┤
│  3  │  4  │                 │1.67│2.0│2.33│2.67│
└─────┴─────┘                 ├──┼──┼──┼──┤
                              │2.33│2.67│3.0│3.33│
Corners aligned               ├──┼──┼──┼──┤
                              │3.0│3.33│3.67│4.0│
                              └──┴──┴──┴──┘
```

### Which to Use?

| Use Case | Recommendation |
|----------|----------------|
| General upsampling | `align_corners=False` |
| Matching input/output corners exactly | `align_corners=True` |
| Semantic segmentation | `align_corners=False` |
| Repeating exact same upsampling | Either (be consistent) |

**Best Practice**: Stick with `align_corners=False` (default) for most cases.

---

## 4. Deprecated Convenience Classes

### UpsamplingNearest2d

Deprecated in favor of `nn.Upsample(..., mode='nearest')`.

```python
# Deprecated:
upsample = nn.UpsamplingNearest2d(scale_factor=2)

# Preferred:
upsample = nn.Upsample(scale_factor=2, mode='nearest')
```

### UpsamplingBilinear2d

Deprecated in favor of `nn.Upsample(..., mode='bilinear', align_corners=True)`.

```python
# Deprecated (uses align_corners=True):
upsample = nn.UpsamplingBilinear2d(scale_factor=2)

# Preferred:
upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

# Or with modern default:
upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
```

---

## 5. F.interpolate (Functional API)

The underlying functional API offers more flexibility.

### Signature

```python
F.interpolate(
    input,                          # Input tensor
    size=None,                      # Output size
    scale_factor=None,              # Scale multiplier
    mode='nearest',                 # Interpolation mode
    align_corners=None,             # Corner alignment
    recompute_scale_factor=None,    # Recompute scale
    antialias=False,                # Anti-aliasing for downsampling
)
```

### antialias Parameter

For downsampling, `antialias=True` applies a low-pass filter to reduce aliasing.

```python
# Downsampling with anti-aliasing
output = F.interpolate(
    input,
    scale_factor=0.5,
    mode='bilinear',
    align_corners=False,
    antialias=True  # Prevents aliasing artifacts
)
```

### Examples

```python
# Dynamic upsampling (size determined at runtime)
def forward(self, x, target_size):
    return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

# Asymmetric scaling
output = F.interpolate(x, scale_factor=(2, 4), mode='bilinear', align_corners=False)
# Height doubled, width quadrupled
```

---

## 6. Common Patterns

### U-Net Decoder

```python
class UNetDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),  # After concat
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Skip connection
        return self.conv(x)
```

### PixelShuffle Alternative

For super-resolution, `PixelShuffle` can be more efficient:

```python
# Traditional upsampling
upsample = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(64, 64, 3, padding=1)
)

# PixelShuffle (sub-pixel convolution)
upsample = nn.Sequential(
    nn.Conv2d(64, 64 * 4, 3, padding=1),  # 4 = 2^2 for 2x upscale
    nn.PixelShuffle(2)
)
# More efficient for learned upsampling
```

### Upsampling + Conv vs ConvTranspose

```python
# Approach 1: Upsample + Conv (checkerboard-free)
decoder = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(256, 128, 3, padding=1),
    nn.ReLU()
)

# Approach 2: ConvTranspose (can cause checkerboard artifacts)
decoder = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
    nn.ReLU()
)

# Approach 1 is often preferred to avoid checkerboard artifacts
```

### Feature Pyramid Network (FPN)

```python
class FPNTopDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lateral = nn.Conv2d(in_ch, out_ch, 1)  # 1x1 for channel reduction
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, x_prev):
        # x: current level, x_prev: previous (lower-res) level
        lateral = self.lateral(x)
        # Upsample previous level to match current
        upsampled = F.interpolate(
            x_prev, size=lateral.shape[2:], mode='nearest'
        )
        return self.smooth(lateral + upsampled)
```

---

## 7. MLX Mapping

### MLX Upsample Approaches

MLX doesn't have a direct `nn.Upsample` module, but provides similar functionality:

```python
import mlx.core as mx
import mlx.nn as nn

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x):
        # x shape: (N, H, W, C) in MLX (channels-last)
        if self.mode == 'nearest':
            return self._nearest_upsample(x)
        elif self.mode == 'bilinear':
            return self._bilinear_upsample(x)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _nearest_upsample(self, x):
        # Nearest neighbor via repeat
        N, H, W, C = x.shape
        s = self.scale_factor

        # Repeat along height and width
        x = mx.repeat(x, s, axis=1)  # (N, H*s, W, C)
        x = mx.repeat(x, s, axis=2)  # (N, H*s, W*s, C)
        return x

    def _bilinear_upsample(self, x):
        # Bilinear requires grid sampling
        # This is a simplified version
        N, H, W, C = x.shape
        new_H = int(H * self.scale_factor)
        new_W = int(W * self.scale_factor)

        # Use mx.image.resize if available, or implement grid sample
        # MLX provides resize functionality
        return mx.image.resize(x, (new_H, new_W), method='bilinear')
```

### Using mx.image.resize

```python
# MLX native image resize
output = mx.image.resize(
    image,          # (H, W, C) or (N, H, W, C)
    (new_H, new_W), # Target size
    method='bilinear'  # or 'nearest', 'lanczos3', etc.
)
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Channel order | NCHW | NHWC |
| Module | `nn.Upsample` | Manual / `mx.image.resize` |
| Modes | nearest, bilinear, bicubic, trilinear | nearest, bilinear, lanczos |
| align_corners | Supported | May differ |
| 3D support | Built-in | Manual implementation |

---

## 8. Performance Considerations

### Mode Speed Comparison

```
Fastest → Slowest:
nearest > bilinear > bicubic

Memory usage:
All modes: O(output_size)
```

### When to Use Each

| Mode | Use Case |
|------|----------|
| `nearest` | Segmentation masks, labels, fast preview |
| `bilinear` | General images, feature maps |
| `bicubic` | High-quality image upscaling |
| `trilinear` | 3D medical imaging, video |

### Tips

1. **Prefer Upsample + Conv over ConvTranspose** to avoid checkerboard artifacts
2. **Use nearest for segmentation masks** to preserve sharp boundaries
3. **Use bilinear for feature maps** in encoder-decoder networks
4. **Consider PixelShuffle** for learned super-resolution
5. **Use antialias=True for downsampling** to reduce aliasing

---

## Summary

### Quick Reference

| Module | Modes | Input Dims | Use Case |
|--------|-------|------------|----------|
| `Upsample` | all | 3D, 4D, 5D | General upsampling |
| `UpsamplingNearest2d` | nearest | 4D | (Deprecated) |
| `UpsamplingBilinear2d` | bilinear | 4D | (Deprecated) |

### Common Configurations

```python
# 2x nearest upsampling
nn.Upsample(scale_factor=2, mode='nearest')

# 2x bilinear upsampling
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

# Upsample to specific size
nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

# 4x bicubic (high quality)
nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
```

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/upsampling.py`

**Functional API**:
- `F.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)`
- `F.upsample(...)` (deprecated alias)
- `F.upsample_nearest(...)` (deprecated)
- `F.upsample_bilinear(...)` (deprecated)

**Related Modules**:
- `nn.PixelShuffle` - Sub-pixel convolution upsampling
- `nn.PixelUnshuffle` - Inverse of PixelShuffle
- `nn.ConvTranspose2d` - Learned upsampling via transposed convolution
