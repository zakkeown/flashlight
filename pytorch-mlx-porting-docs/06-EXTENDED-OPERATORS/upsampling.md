# Upsampling & Interpolation - PyTorch → MLX Porting Guide

## Overview

This document covers PyTorch's upsampling and interpolation operations, which are essential for resizing tensors in computer vision, generative models, and signal processing applications. These operations enable spatial transformations of feature maps in neural networks.

PyTorch provides a comprehensive `torch.nn.functional.interpolate` API that supports multiple interpolation modes across 1D, 2D, and 3D tensors.

### Key Use Cases

1. **Image Upscaling**: Super-resolution, GAN generators (StyleGAN, DCGAN)
2. **Semantic Segmentation**: Decoder paths in U-Net, FCN, DeepLab
3. **Object Detection**: Feature pyramid networks (FPN), multi-scale processing
4. **Video Processing**: Temporal interpolation, frame rate conversion
5. **Audio Processing**: Sample rate conversion, time-stretching

---

## Table of Contents

1. [Interpolation Modes](#interpolation-modes)
2. [torch.nn.functional.interpolate API](#torchnnfunctionalinterpolate-api)
3. [Nearest Neighbor Interpolation](#nearest-neighbor-interpolation)
4. [Bilinear Interpolation](#bilinear-interpolation)
5. [Bicubic Interpolation](#bicubic-interpolation)
6. [Trilinear Interpolation](#trilinear-interpolation)
7. [Area Interpolation](#area-interpolation)
8. [Grid Sampling](#grid-sampling)
9. [MLX Porting Guide](#mlx-porting-guide)
10. [Performance Considerations](#performance-considerations)

---

## Interpolation Modes

PyTorch supports 7 interpolation modes:

| Mode | Dimensions | Description | Use Case |
|------|------------|-------------|----------|
| **`nearest`** | 1D/2D/3D | Nearest neighbor (replication) | Fast upsampling, classification |
| **`nearest-exact`** | 1D/2D/3D | Exact nearest (Scikit-Image compatible) | Pixel-accurate resampling |
| **`linear`** | 1D | Linear interpolation | Audio, 1D signals |
| **`bilinear`** | 2D | Bilinear interpolation | Image upsampling, segmentation |
| **`bicubic`** | 2D | Bicubic interpolation | High-quality image resize |
| **`trilinear`** | 3D | Trilinear interpolation | Video, 3D volumes |
| **`area`** | 1D/2D/3D | Adaptive average pooling | Downsampling, anti-aliasing |

---

## torch.nn.functional.interpolate API

**Primary Upsampling Function**:
```python
torch.nn.functional.interpolate(
    input: Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,  # Output size
    scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,  # Scale multiplier
    mode: str = 'nearest',  # Interpolation mode
    align_corners: Optional[bool] = None,  # Coordinate alignment
    recompute_scale_factor: Optional[bool] = None,  # Recompute scales
    antialias: bool = False  # Anti-aliasing filter
) -> Tensor
```

**Implementation** ([torch/nn/functional.py:4614-4939](reference/pytorch/torch/nn/functional.py#L4614-L4939)):

Key dispatch logic ([lines 4811-4920](reference/pytorch/torch/nn/functional.py#L4811-L4920)):
```python
# Nearest neighbor
if input.dim() == 3 and mode == "nearest":
    return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)
if input.dim() == 4 and mode == "nearest":
    return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)
if input.dim() == 5 and mode == "nearest":
    return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)

# Bilinear/trilinear
if input.dim() == 4 and mode == "bilinear":
    if antialias:
        return torch._C._nn._upsample_bilinear2d_aa(input, output_size, align_corners, scale_factors)
    return torch._C._nn.upsample_bilinear2d(input, output_size, align_corners, scale_factors)

if input.dim() == 5 and mode == "trilinear":
    return torch._C._nn.upsample_trilinear3d(input, output_size, align_corners, scale_factors)

# Bicubic
if input.dim() == 4 and mode == "bicubic":
    if antialias:
        return torch._C._nn._upsample_bicubic2d_aa(input, output_size, align_corners, scale_factors)
    return torch._C._nn.upsample_bicubic2d(input, output_size, align_corners, scale_factors)

# Area (uses adaptive pooling)
if input.dim() == 4 and mode == "area":
    return adaptive_avg_pool2d(input, output_size)
```

**Output Size Computation** ([aten/src/ATen/native/UpSample.cpp:10-31](reference/pytorch/aten/src/ATen/native/UpSample.cpp#L10-L31)):
```cpp
c10::SmallVector<int64_t, 3> compute_output_size(
    c10::IntArrayRef input_size,
    at::OptionalIntArrayRef output_size,
    std::optional<c10::ArrayRef<double>> scale_factors) {

  const auto spatial_dimensions = static_cast<int64_t>(input_size.size()) - 2;

  if (output_size) {
    TORCH_CHECK(!scale_factors, "Must specify exactly one of output_size and scale_factors");
    return {output_size->data(), output_size->data() + output_size->size()};
  }

  if (scale_factors) {
    TORCH_CHECK(!output_size, "Must specify exactly one of output_size and scale_factors");
    c10::SmallVector<int64_t, 3> ret;
    for (const auto i : c10::irange(spatial_dimensions)) {
      const double odim = static_cast<double>(input_size[i+2]) * scale_factors.value()[i];
      ret.push_back(c10::checked_convert<int64_t>(odim, "int64_t"));
    }
    return ret;
  }

  TORCH_CHECK(false, "Must specify exactly one of output_size and scale_factors");
}
```

**Example Usage**:
```python
import torch
import torch.nn.functional as F

# Input image: [batch, channels, height, width]
image = torch.randn(1, 3, 64, 64)

# Upsampling by size
upsampled = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)
# Output shape: [1, 3, 128, 128]

# Upsampling by scale factor
upsampled_2x = F.interpolate(image, scale_factor=2.0, mode='bilinear', align_corners=False)
# Output shape: [1, 3, 128, 128]

# Different scales per dimension
upsampled_asymmetric = F.interpolate(image, scale_factor=(2.0, 1.5), mode='bilinear', align_corners=False)
# Output shape: [1, 3, 128, 96]

# Downsampling with anti-aliasing
downsampled = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False, antialias=True)
# Output shape: [1, 3, 32, 32]
```

---

## Nearest Neighbor Interpolation

**Algorithm**: Replicates the nearest input pixel value.

**Formula**:
```
output[i, j] = input[round(i * scale_h), round(j * scale_w)]

where:
- scale_h = input_height / output_height
- scale_w = input_width / output_width
```

**Coordinate Mapping**:
```python
# PyTorch "nearest" mode (OpenCV-style, buggy)
input_x = floor((output_x + 0.5) * scale_w)

# PyTorch "nearest-exact" mode (Scikit-Image/PIL-style, correct)
input_x = round(output_x * scale_w)
```

**Example**:
```python
# 2x2 input
input = torch.tensor([[[[1, 2],
                        [3, 4]]]], dtype=torch.float32)

# Upsample to 4x4 using nearest
output_nearest = F.interpolate(input, size=(4, 4), mode='nearest')
# Output:
# [[[[1, 1, 2, 2],
#    [1, 1, 2, 2],
#    [3, 3, 4, 4],
#    [3, 3, 4, 4]]]]

# Nearest-exact gives slightly different results for edge cases
output_exact = F.interpolate(input, size=(4, 4), mode='nearest-exact')
```

**Properties**:
- ✅ Fastest interpolation method
- ✅ Preserves input values exactly
- ✅ No overshoot or undershoot
- ❌ Blocky artifacts
- ❌ Not smooth or differentiable

**Use Cases**:
- Classification networks (spatial structure less important)
- Integer upscaling for pixel art
- Semantic segmentation masks (preserve class labels)

---

## Bilinear Interpolation

**Algorithm**: Interpolates using the 4 nearest neighbors with linear weighting.

**Formula** (2D):
```
output[i, j] = (1-α)(1-β)·I₀₀ + α(1-β)·I₁₀ + (1-α)β·I₀₁ + αβ·I₁₁

where:
- I₀₀, I₁₀, I₀₁, I₁₁: Four nearest neighbor pixels
- α, β: Fractional distances in x, y directions
```

**Coordinate Mapping** (align_corners=False):
```python
# Map output coordinates to input coordinates
input_x = (output_x + 0.5) * scale_w - 0.5
input_y = (output_y + 0.5) * scale_h - 0.5

# Get integer and fractional parts
x0 = floor(input_x)
x1 = x0 + 1
alpha = input_x - x0  # Fractional part

# Bilinear weights
w00 = (1 - alpha) * (1 - beta)
w10 = alpha * (1 - beta)
w01 = (1 - alpha) * beta
w11 = alpha * beta

# Interpolated value
output = w00*I[y0, x0] + w10*I[y0, x1] + w01*I[y1, x0] + w11*I[y1, x1]
```

**Coordinate Mapping** (align_corners=True):
```python
# Align corner pixels exactly
input_x = output_x * (input_width - 1) / (output_width - 1)
input_y = output_y * (input_height - 1) / (output_height - 1)

# Example: 4x4 -> 8x8
# align_corners=False: scale = 4/8 = 0.5, offset = 0.5*(0.5 - 1) = -0.25
# align_corners=True:  scale = 3/7 = 0.4286
```

**Implementation** ([aten/src/ATen/native/UpSampleBilinear2d.cpp:160-169](reference/pytorch/aten/src/ATen/native/UpSampleBilinear2d.cpp#L160-L169)):
```cpp
Tensor upsample_bilinear2d(
    const Tensor& input,
    at::OptionalIntArrayRef output_size,
    bool align_corners,
    std::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  return at::upsample_bilinear2d(input, osize, align_corners, scale_h, scale_w);
}
```

**Example**:
```python
# Original 2x2 image
image = torch.tensor([[[[0.0, 1.0],
                        [2.0, 3.0]]]], dtype=torch.float32)

# Bilinear upsample to 4x4 (align_corners=False)
upsampled = F.interpolate(image, size=(4, 4), mode='bilinear', align_corners=False)
# Output (approximate):
# [[[[0.00, 0.33, 0.67, 1.00],
#    [0.67, 1.00, 1.33, 1.67],
#    [1.33, 1.67, 2.00, 2.33],
#    [2.00, 2.33, 2.67, 3.00]]]]

# With align_corners=True (corners exactly preserved)
upsampled_aligned = F.interpolate(image, size=(4, 4), mode='bilinear', align_corners=True)
# Output:
# [[[[0.0, 0.333, 0.667, 1.0],
#    [0.667, 1.0, 1.333, 1.667],
#    [1.333, 1.667, 2.0, 2.333],
#    [2.0, 2.333, 2.667, 3.0]]]]
```

**Anti-Aliasing** (for downsampling):
```python
# Without anti-aliasing (aliasing artifacts)
downsampled = F.interpolate(large_image, size=(64, 64), mode='bilinear', antialias=False)

# With anti-aliasing (smoother, less aliasing)
downsampled_aa = F.interpolate(large_image, size=(64, 64), mode='bilinear', antialias=True)
```

**Properties**:
- ✅ Smooth results
- ✅ Differentiable (good for training)
- ✅ Fast computation
- ❌ May blur sharp edges
- ❌ Slight overshoot possible near edges

**Use Cases**:
- Image upsampling in GANs (DCGAN, StyleGAN)
- Semantic segmentation decoders (U-Net, DeepLab)
- Feature pyramid networks (FPN)
- General-purpose image resizing

---

## Bicubic Interpolation

**Algorithm**: Interpolates using a 4x4 neighborhood with cubic weighting.

**Cubic Kernel** (Catmull-Rom spline):
```python
def cubic_weight(x):
    """Bicubic interpolation kernel."""
    x = abs(x)
    if x <= 1:
        return 1.5*x**3 - 2.5*x**2 + 1
    elif x < 2:
        return -0.5*x**3 + 2.5*x**2 - 4*x + 2
    else:
        return 0
```

**Formula** (2D):
```
output[i, j] = Σ(m=0 to 3) Σ(n=0 to 3) w(α - m) · w(β - n) · I[y + m, x + n]

where:
- I: Input image
- w: Cubic weight function
- α, β: Fractional coordinates
- 4x4 neighborhood centered at (x, y)
```

**Example**:
```python
# High-quality image upscaling
low_res = torch.randn(1, 3, 64, 64)

# Bicubic upsampling (smoother than bilinear)
high_res = F.interpolate(low_res, size=(256, 256), mode='bicubic', align_corners=False)

# With anti-aliasing for downsampling
downscaled = F.interpolate(high_res, size=(64, 64), mode='bicubic', antialias=True)
```

**Overshoot Handling**:
```python
# Bicubic can produce values outside [0, 1] range
image_uint8 = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
image_float = image_uint8.float() / 255.0

upsampled = F.interpolate(image_float, size=(128, 128), mode='bicubic', align_corners=False)

# Clamp to valid range
upsampled_clamped = upsampled.clamp(0, 1)

# For uint8 images, PyTorch automatically saturates
upsampled_uint8 = F.interpolate(image_uint8.float(), size=(128, 128), mode='bicubic')
# Values automatically clamped to [0, 255]
```

**Properties**:
- ✅ High-quality results
- ✅ Sharper than bilinear
- ✅ Differentiable
- ❌ Slower than bilinear (4x4 vs 2x2 neighborhood)
- ❌ Can overshoot (values < 0 or > 255 for images)
- ❌ Ringing artifacts near sharp edges

**Use Cases**:
- High-quality photo upscaling
- Super-resolution networks (SRCNN, ESRGAN)
- Professional image editing
- Print-quality resizing

---

## Trilinear Interpolation

**Algorithm**: 3D extension of bilinear interpolation for volumetric data.

**Formula** (3D):
```
output[i, j, k] = Σ(dz=0 to 1) Σ(dy=0 to 1) Σ(dx=0 to 1)
                  w_x(dx) · w_y(dy) · w_z(dz) · I[z+dz, y+dy, x+dx]

where:
- I: Input volume
- w_x, w_y, w_z: Linear weights in x, y, z directions
- 2x2x2 neighborhood
```

**Example** (Video/3D Medical Imaging):
```python
# 3D volume: [batch, channels, depth, height, width]
volume = torch.randn(1, 1, 64, 64, 64)  # 64³ CT scan

# Trilinear upsampling to 128³
upsampled_volume = F.interpolate(volume, size=(128, 128, 128), mode='trilinear', align_corners=False)
# Output shape: [1, 1, 128, 128, 128]

# Video upsampling: [batch, channels, frames, height, width]
video = torch.randn(1, 3, 16, 64, 64)  # 16 frames

# Upsample frames and spatial dimensions
upsampled_video = F.interpolate(video, size=(32, 128, 128), mode='trilinear', align_corners=False)
# Output: [1, 3, 32, 128, 128] - 32 frames at 128x128
```

**Properties**:
- ✅ Smooth 3D interpolation
- ✅ Preserves temporal continuity (for video)
- ❌ Memory intensive (3D tensors)
- ❌ Slower than 2D interpolation

**Use Cases**:
- 3D medical image analysis (CT, MRI)
- Video super-resolution
- Volumetric rendering
- 3D segmentation (V-Net, 3D U-Net)

---

## Area Interpolation

**Algorithm**: Adaptive average pooling (downsampling only).

**Implementation**:
```python
# For upsampling, "area" mode uses adaptive average pooling
# which is equivalent to downsampling with averaging

# 2D area interpolation (downsampling)
large_image = torch.randn(1, 3, 256, 256)
downsampled = F.interpolate(large_image, size=(64, 64), mode='area')

# Equivalent to adaptive average pooling
downsampled_equiv = F.adaptive_avg_pool2d(large_image, (64, 64))
assert torch.allclose(downsampled, downsampled_equiv)
```

**Properties**:
- ✅ Good for downsampling (anti-aliasing effect)
- ✅ Preserves average intensity
- ❌ Only useful for downsampling
- ❌ Not suitable for upsampling

**Use Cases**:
- Thumbnail generation
- Multi-scale processing
- Downsampling before encoding

---

## Grid Sampling

For more advanced warping and spatial transformations, PyTorch provides `torch.nn.functional.grid_sample`.

**API**:
```python
torch.nn.functional.grid_sample(
    input: Tensor,  # [N, C, H_in, W_in]
    grid: Tensor,   # [N, H_out, W_out, 2] - sampling locations
    mode: str = 'bilinear',  # 'bilinear', 'nearest', 'bicubic'
    padding_mode: str = 'zeros',  # 'zeros', 'border', 'reflection'
    align_corners: bool = False
) -> Tensor  # [N, C, H_out, W_out]
```

**Coordinate Convention**:
```python
# Grid values in range [-1, 1]
# (-1, -1): top-left corner
# (+1, +1): bottom-right corner
# (0, 0): center

# Example: Identity grid (no transformation)
def make_identity_grid(N, H, W):
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    return grid.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, H, W, 2]
```

**Example** (Affine Transformation):
```python
import torch
import torch.nn.functional as F

# Input image
image = torch.randn(1, 3, 64, 64)

# Create affine transformation matrix (rotate 45 degrees)
theta = torch.tensor([
    [0.7071, -0.7071, 0],
    [0.7071,  0.7071, 0]
], dtype=torch.float32).unsqueeze(0)  # [1, 2, 3]

# Generate sampling grid
grid = F.affine_grid(theta, image.size(), align_corners=False)
# Shape: [1, 64, 64, 2]

# Sample from image using grid
transformed = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
# Output: [1, 3, 64, 64] - rotated image
```

**Use Cases**:
- Spatial transformer networks (STN)
- Deformable convolutions
- Optical flow warping
- Perspective correction

---

## MLX Porting Guide

### MLX Resize API Status (as of 2024)

MLX does not currently have a built-in `interpolate` function. Implementations must be created from scratch.

**Missing Operations**:
- ❌ `nn.functional.interpolate`
- ❌ Bilinear/bicubic upsampling
- ❌ `grid_sample` for arbitrary transformations

### Implementing Interpolation in MLX

#### Nearest Neighbor Upsampling

```python
import mlx.core as mx

def upsample_nearest_2d(x, scale_factor):
    """
    Nearest neighbor upsampling for 2D images.

    Args:
        x: Input tensor [N, C, H, W]
        scale_factor: Upsampling factor (int or tuple)

    Returns:
        Upsampled tensor [N, C, H*scale, W*scale]
    """
    if isinstance(scale_factor, (int, float)):
        scale_h = scale_w = int(scale_factor)
    else:
        scale_h, scale_w = scale_factor

    N, C, H, W = x.shape

    # Repeat along height and width
    x = mx.repeat(x, scale_h, axis=2)  # [N, C, H*scale, W]
    x = mx.repeat(x, scale_w, axis=3)  # [N, C, H*scale, W*scale]

    return x


# Example usage
image = mx.random.normal((1, 3, 64, 64))
upsampled = upsample_nearest_2d(image, scale_factor=2)
print(upsampled.shape)  # [1, 3, 128, 128]
```

#### Bilinear Upsampling

```python
def bilinear_interpolate_2d(x, size, align_corners=False):
    """
    Bilinear interpolation for 2D images.

    Args:
        x: Input tensor [N, C, H_in, W_in]
        size: Output size (H_out, W_out)
        align_corners: Align corner pixels

    Returns:
        Upsampled tensor [N, C, H_out, W_out]
    """
    N, C, H_in, W_in = x.shape
    H_out, W_out = size

    # Compute output coordinates
    if align_corners:
        # Align corners exactly
        y_coords = mx.linspace(0, H_in - 1, H_out)
        x_coords = mx.linspace(0, W_in - 1, W_out)
    else:
        # Half-pixel offset
        scale_h = H_in / H_out
        scale_w = W_in / W_out
        y_coords = (mx.arange(H_out, dtype=mx.float32) + 0.5) * scale_h - 0.5
        x_coords = (mx.arange(W_out, dtype=mx.float32) + 0.5) * scale_w - 0.5

    # Clip to valid range
    y_coords = mx.clip(y_coords, 0, H_in - 1)
    x_coords = mx.clip(x_coords, 0, W_in - 1)

    # Get integer and fractional parts
    y0 = mx.floor(y_coords).astype(mx.int32)
    x0 = mx.floor(x_coords).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H_in - 1)
    x1 = mx.minimum(x0 + 1, W_in - 1)

    # Compute weights
    wy1 = y_coords - y0.astype(mx.float32)
    wx1 = x_coords - x0.astype(mx.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # Create output tensor
    output = mx.zeros((N, C, H_out, W_out), dtype=x.dtype)

    # Bilinear interpolation (vectorized over batch and channels)
    for i in range(H_out):
        for j in range(W_out):
            # Get 4 neighbors
            val00 = x[:, :, y0[i], x0[j]]
            val01 = x[:, :, y0[i], x1[j]]
            val10 = x[:, :, y1[i], x0[j]]
            val11 = x[:, :, y1[i], x1[j]]

            # Weighted sum
            w00 = wy0[i] * wx0[j]
            w01 = wy0[i] * wx1[j]
            w10 = wy1[i] * wx0[j]
            w11 = wy1[i] * wx1[j]

            output[:, :, i, j] = w00 * val00 + w01 * val01 + w10 * val10 + w11 * val11

    return output


# Example usage
image = mx.random.normal((1, 3, 64, 64))
upsampled = bilinear_interpolate_2d(image, size=(128, 128), align_corners=False)
print(upsampled.shape)  # [1, 3, 128, 128]
```

#### Optimized Metal Shader Implementation

For better performance, implement bilinear interpolation as a Metal shader:

```cpp
// mlx/backend/metal/kernels/upsample.metal

#include <metal_stdlib>
using namespace metal;

kernel void bilinear_upsample_2d(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& C [[buffer(3)]],
    constant int& H_in [[buffer(4)]],
    constant int& W_in [[buffer(5)]],
    constant int& H_out [[buffer(6)]],
    constant int& W_out [[buffer(7)]],
    constant bool& align_corners [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int n = gid.z;
    const int c = gid.y;
    const int h_out = gid.x / W_out;
    const int w_out = gid.x % W_out;

    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;

    // Compute input coordinates
    float scale_h = float(H_in) / float(H_out);
    float scale_w = float(W_in) / float(W_out);

    float y_in, x_in;
    if (align_corners) {
        y_in = h_out * (H_in - 1) / float(H_out - 1);
        x_in = w_out * (W_in - 1) / float(W_out - 1);
    } else {
        y_in = (h_out + 0.5f) * scale_h - 0.5f;
        x_in = (w_out + 0.5f) * scale_w - 0.5f;
    }

    // Clip to valid range
    y_in = clamp(y_in, 0.0f, float(H_in - 1));
    x_in = clamp(x_in, 0.0f, float(W_in - 1));

    // Get integer parts
    int y0 = int(floor(y_in));
    int x0 = int(floor(x_in));
    int y1 = min(y0 + 1, H_in - 1);
    int x1 = min(x0 + 1, W_in - 1);

    // Fractional parts
    float wy1 = y_in - y0;
    float wx1 = x_in - x0;
    float wy0 = 1.0f - wy1;
    float wx0 = 1.0f - wx1;

    // Input indices
    int idx00 = ((n * C + c) * H_in + y0) * W_in + x0;
    int idx01 = ((n * C + c) * H_in + y0) * W_in + x1;
    int idx10 = ((n * C + c) * H_in + y1) * W_in + x0;
    int idx11 = ((n * C + c) * H_in + y1) * W_in + x1;

    // Bilinear interpolation
    float val = wy0 * wx0 * input[idx00] +
                wy0 * wx1 * input[idx01] +
                wy1 * wx0 * input[idx10] +
                wy1 * wx1 * input[idx11];

    // Output index
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = val;
}
```

### C++ API Design for MLX

```cpp
// mlx/nn/upsample.h

namespace mlx::core::nn {

// Nearest neighbor upsampling
array upsample_nearest_2d(
    const array& input,
    std::optional<std::pair<int, int>> size = std::nullopt,
    std::optional<std::pair<float, float>> scale_factor = std::nullopt
);

// Bilinear upsampling
array upsample_bilinear_2d(
    const array& input,
    std::optional<std::pair<int, int>> size = std::nullopt,
    std::optional<std::pair<float, float>> scale_factor = std::nullopt,
    bool align_corners = false,
    bool antialias = false
);

// General interpolate function (PyTorch-compatible)
array interpolate(
    const array& input,
    std::optional<std::vector<int>> size = std::nullopt,
    std::optional<std::vector<float>> scale_factor = std::nullopt,
    const std::string& mode = "nearest",
    bool align_corners = false,
    bool antialias = false
);

// Grid sampling for arbitrary transformations
array grid_sample(
    const array& input,
    const array& grid,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool align_corners = false
);

}  // namespace mlx::core::nn
```

---

## Performance Considerations

### Mode Performance Comparison

**Speed Ranking** (fastest to slowest):
1. **Nearest** - Simple indexing, no interpolation
2. **Area** - Efficient average pooling
3. **Bilinear** - 4 samples per output pixel
4. **Trilinear** - 8 samples per output pixel
5. **Bicubic** - 16 samples per output pixel (4x4 neighborhood)

**Benchmark** (1024x1024 image, CUDA):
```python
import torch
import time

image = torch.randn(1, 3, 1024, 1024, device='cuda')

# Nearest: ~0.5ms
start = time.time()
_ = F.interpolate(image, size=(2048, 2048), mode='nearest')
torch.cuda.synchronize()
print(f"Nearest: {(time.time() - start) * 1000:.2f}ms")

# Bilinear: ~1.2ms
start = time.time()
_ = F.interpolate(image, size=(2048, 2048), mode='bilinear', align_corners=False)
torch.cuda.synchronize()
print(f"Bilinear: {(time.time() - start) * 1000:.2f}ms")

# Bicubic: ~3.5ms
start = time.time()
_ = F.interpolate(image, size=(2048, 2048), mode='bicubic', align_corners=False)
torch.cuda.synchronize()
print(f"Bicubic: {(time.time() - start) * 1000:.2f}ms")
```

### Memory Layout

**Channels-Last Format** (NHWC) is faster on some hardware:
```python
# Channels-first (NCHW) - default
image_nchw = torch.randn(1, 3, 512, 512)

# Channels-last (NHWC) - faster on some GPUs
image_nhwc = image_nchw.to(memory_format=torch.channels_last)

# Interpolate (automatically uses optimized kernels for channels-last)
upsampled = F.interpolate(image_nhwc, scale_factor=2, mode='bilinear')
```

### Align Corners Trade-offs

**`align_corners=False` (Recommended)**:
- ✅ Scale-equivariant (resize(resize(x)) preserves relative positions)
- ✅ Consistent behavior across different output sizes
- ✅ Matches OpenCV, Pillow, scikit-image

**`align_corners=True`**:
- ✅ Preserves corner pixel values exactly
- ❌ Not scale-equivariant
- ❌ Less common in modern frameworks

### Anti-Aliasing for Downsampling

**When to use `antialias=True`**:
- Downsampling by >2x
- High-frequency content (text, fine details)
- Matching Pillow/PIL quality

**Performance Impact**:
- ~20-30% slower than without anti-aliasing
- Significantly better quality for downsampling

---

## Common Pitfalls and Best Practices

### 1. Align Corners Confusion

**Problem**: Different `align_corners` settings produce different results.

**Solution**: Use `align_corners=False` for consistency with modern libraries:
```python
# Recommended (matches Pillow, OpenCV)
upsampled = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)

# Avoid unless specifically needed
upsampled_aligned = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=True)
```

### 2. Bicubic Overshoot

**Problem**: Bicubic can produce negative values or values > 255.

**Solution**: Clamp output to valid range:
```python
upsampled = F.interpolate(image, size=(512, 512), mode='bicubic')
upsampled_clamped = upsampled.clamp(0, 1)  # For [0, 1] range
```

### 3. Mode Mismatch

**Problem**: Using wrong mode for tensor dimensions.

**Solution**: Match mode to input dimensions:
```python
# 3D input (1D signal): use 'linear'
signal_3d = torch.randn(1, 1, 100)
upsampled_1d = F.interpolate(signal_3d, size=200, mode='linear', align_corners=False)

# 4D input (2D image): use 'bilinear' or 'bicubic'
image_4d = torch.randn(1, 3, 64, 64)
upsampled_2d = F.interpolate(image_4d, size=(128, 128), mode='bilinear', align_corners=False)

# 5D input (3D volume): use 'trilinear'
volume_5d = torch.randn(1, 1, 32, 64, 64)
upsampled_3d = F.interpolate(volume_5d, size=(64, 128, 128), mode='trilinear', align_corners=False)
```

### 4. Anti-Aliasing Restrictions

**Problem**: `antialias=True` only works for specific modes.

**Solution**: Only use with bilinear/bicubic on 4D tensors:
```python
# Correct
downsampled = F.interpolate(image_4d, size=(32, 32), mode='bilinear', antialias=True)

# Error: antialias not supported for nearest
# downsampled = F.interpolate(image_4d, size=(32, 32), mode='nearest', antialias=True)
```

---

## Summary

### Key Takeaways

1. **Nearest**: Fastest, blocky artifacts, good for classification
2. **Bilinear**: General-purpose, smooth, good for segmentation
3. **Bicubic**: High-quality, slower, best for photo upscaling
4. **Trilinear**: 3D volumes and video
5. **Area**: Downsampling only (anti-aliasing effect)
6. **Grid Sample**: Arbitrary transformations (STN, warping)

### API Mapping

| PyTorch | MLX | Status | Notes |
|---------|-----|--------|-------|
| `F.interpolate(..., mode='nearest')` | ❌ Missing | Implement using `mx.repeat` | |
| `F.interpolate(..., mode='bilinear')` | ❌ Missing | Implement with Metal shader | |
| `F.interpolate(..., mode='bicubic')` | ❌ Missing | Implement with 4x4 kernel | |
| `F.interpolate(..., mode='trilinear')` | ❌ Missing | 3D extension of bilinear | |
| `F.grid_sample` | ❌ Missing | Requires custom kernel | |
| `align_corners` parameter | ❌ Missing | Coordinate mapping mode | |
| `antialias` parameter | ❌ Missing | Low-pass filter before resample | |

### Implementation Checklist for MLX

**High Priority**:
- ❌ Nearest neighbor (1D/2D/3D)
- ❌ Bilinear interpolation (2D)
- ❌ `align_corners` support

**Medium Priority**:
- ❌ Bicubic interpolation (2D)
- ❌ Trilinear interpolation (3D)
- ❌ Anti-aliasing for downsampling

**Low Priority**:
- ❌ Grid sampling
- ❌ Area mode (can use adaptive pooling)

---

## References

**PyTorch Source Files**:
- [aten/src/ATen/native/UpSample.cpp](reference/pytorch/aten/src/ATen/native/UpSample.cpp) - Output size computation
- [aten/src/ATen/native/UpSampleBilinear2d.cpp](reference/pytorch/aten/src/ATen/native/UpSampleBilinear2d.cpp) - Bilinear implementation
- [aten/src/ATen/native/UpSampleBicubic2d.cpp](reference/pytorch/aten/src/ATen/native/UpSampleBicubic2d.cpp) - Bicubic implementation
- [torch/nn/functional.py:4614-4939](reference/pytorch/torch/nn/functional.py#L4614-L4939) - Interpolate API

**Algorithm References**:
- Bilinear interpolation (linear algebra textbooks)
- Bicubic interpolation (Catmull-Rom spline, 1974)
- Keys, R. (1981). "Cubic convolution interpolation for digital image processing"
- Spatial Transformer Networks (Jaderberg et al., 2015)

**Metal/Apple Documentation**:
- Metal Shading Language Specification
- Metal Performance Shaders - Image filters
- Accelerate framework vImage
