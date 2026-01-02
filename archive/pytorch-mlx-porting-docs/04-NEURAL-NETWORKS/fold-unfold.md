# Fold and Unfold Operations

## Overview

Fold and Unfold (also known as col2im and im2col) are operations for extracting and reconstructing sliding local blocks from tensors. They are fundamental for implementing convolutions as matrix multiplications and for various image processing tasks.

**Reference File:** `torch/nn/modules/fold.py`

## Concepts

```
Unfold (im2col):  Image → Columns
  Extracts sliding blocks into columns

Fold (col2im):    Columns → Image
  Combines columns back into image (summing overlaps)
```

### Visual Representation

```
Input Image (4x4):          Unfold (2x2 kernel, stride=1):
┌───┬───┬───┬───┐          Output (4 values × 9 positions):
│ a │ b │ c │ d │
├───┼───┼───┼───┤          ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ e │ f │ g │ h │   →      │a,b│b,c│c,d│e,f│f,g│g,h│i,j│j,k│k,l│
├───┼───┼───┼───┤          │e,f│f,g│g,h│i,j│j,k│k,l│m,n│n,o│o,p│
│ i │ j │ k │ l │          └───┴───┴───┴───┴───┴───┴───┴───┴───┘
├───┼───┼───┼───┤            ↑
│ m │ n │ o │ p │          Each column is one flattened 2x2 patch
└───┴───┴───┴───┘
```

---

## nn.Unfold

Extracts sliding local blocks from a batched input tensor.

### Class Definition

```python
class Unfold(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],  # Size of sliding blocks
        dilation: int | tuple[int, int] = 1, # Spacing between kernel points
        padding: int | tuple[int, int] = 0,  # Zero padding on input
        stride: int | tuple[int, int] = 1    # Stride of sliding blocks
    )
```

### Shape Transformation

```
Input:  (N, C, H, W)
Output: (N, C × kernel_h × kernel_w, L)

where L = number of blocks = ((H + 2×pad - dilation×(kernel-1) - 1) / stride + 1)
                           × ((W + 2×pad - dilation×(kernel-1) - 1) / stride + 1)
```

### Basic Usage

```python
import torch
import torch.nn as nn

# Create unfold layer
unfold = nn.Unfold(kernel_size=(3, 3))

# Input: batch=1, channels=3, height=4, width=4
x = torch.randn(1, 3, 4, 4)

# Output: batch=1, 3×3×3=27 values per patch, 4 patches (2×2 grid)
out = unfold(x)
print(out.shape)  # torch.Size([1, 27, 4])
```

### With Stride and Padding

```python
# Extract 3x3 patches with stride 2, padding 1
unfold = nn.Unfold(kernel_size=3, stride=2, padding=1)

x = torch.randn(1, 1, 8, 8)
out = unfold(x)

# Number of patches: ((8 + 2×1 - 1×(3-1) - 1) / 2 + 1)² = 4² = 16
print(out.shape)  # torch.Size([1, 9, 16])
```

### With Dilation

```python
# Dilated unfold (à trous algorithm)
unfold = nn.Unfold(kernel_size=3, dilation=2)

x = torch.randn(1, 1, 7, 7)
out = unfold(x)

# Effective kernel size: 1 + dilation × (kernel_size - 1) = 1 + 2×2 = 5
# Patches: ((7 - 2×(3-1) - 1) / 1 + 1)² = 3² = 9
print(out.shape)  # torch.Size([1, 9, 9])
```

---

## nn.Fold

Combines sliding local blocks into a large containing tensor.

### Class Definition

```python
class Fold(nn.Module):
    def __init__(
        self,
        output_size: int | tuple[int, int],  # Target spatial size
        kernel_size: int | tuple[int, int],  # Size of sliding blocks
        dilation: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        stride: int | tuple[int, int] = 1
    )
```

### Shape Transformation

```
Input:  (N, C × kernel_h × kernel_w, L)
Output: (N, C, output_size[0], output_size[1])
```

### Basic Usage

```python
# Create fold layer
fold = nn.Fold(output_size=(4, 4), kernel_size=(2, 2))

# Input: batch=1, 1×2×2=4 values per patch, 9 patches
x = torch.randn(1, 4, 9)

# Output: batch=1, channels=1, height=4, width=4
out = fold(x)
print(out.shape)  # torch.Size([1, 1, 4, 4])
```

### Important: Overlapping Blocks Sum

When blocks overlap, Fold **sums** the overlapping values:

```python
# Create matching unfold/fold pair
unfold = nn.Unfold(kernel_size=2, stride=1)
fold = nn.Fold(output_size=(3, 3), kernel_size=2, stride=1)

# Original image
x = torch.ones(1, 1, 3, 3)

# Unfold then fold
unfolded = unfold(x)  # Shape: (1, 4, 4)
folded = fold(unfolded)  # Shape: (1, 1, 3, 3)

print(folded)
# tensor([[[[1., 2., 1.],
#           [2., 4., 2.],
#           [1., 2., 1.]]]])
#
# Center values are larger because they appear in more patches!
```

### Inverse Relationship

To properly invert unfold, divide by the overlap count:

```python
# Create divisor by folding ones
input_ones = torch.ones_like(x)
divisor = fold(unfold(input_ones))

# Proper inverse
reconstructed = fold(unfold(x)) / divisor
print(torch.allclose(x, reconstructed))  # True
```

---

## Functional Interface

### F.unfold()

```python
import torch.nn.functional as F

x = torch.randn(1, 3, 4, 4)
out = F.unfold(x, kernel_size=2, dilation=1, padding=0, stride=1)
```

### F.fold()

```python
x = torch.randn(1, 12, 9)  # 3 channels × 2×2 kernel, 9 patches
out = F.fold(x, output_size=(4, 4), kernel_size=2)
```

---

## Convolution as Matrix Multiplication

Unfold enables implementing convolution as matrix multiplication:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input and weights
inp = torch.randn(1, 3, 10, 12)   # (N, C_in, H, W)
w = torch.randn(2, 3, 4, 5)       # (C_out, C_in, kH, kW)

# Standard convolution
conv_out = F.conv2d(inp, w)
print(conv_out.shape)  # torch.Size([1, 2, 7, 8])

# Equivalent using unfold + matmul + reshape
inp_unf = F.unfold(inp, kernel_size=(4, 5))  # (1, 3×4×5=60, 56)
w_flat = w.view(w.size(0), -1)               # (2, 60)

# Matrix multiply: (1, 56, 60) @ (60, 2) -> (1, 56, 2) -> (1, 2, 56)
out_unf = inp_unf.transpose(1, 2).matmul(w_flat.t()).transpose(1, 2)

# Reshape to output size
out = out_unf.view(1, 2, 7, 8)

# Verify equivalence
print((conv_out - out).abs().max())  # ~1e-6 (numerical precision)
```

### Why This Matters

1. **Understanding**: Shows convolution is just pattern matching via dot products
2. **Optimization**: Can use optimized GEMM libraries
3. **Flexibility**: Enables custom convolution variants
4. **Attention**: Similar pattern used in vision transformers (patch extraction)

---

## Common Use Cases

### 1. Image Patch Extraction

```python
# Extract non-overlapping 8x8 patches
patch_size = 8
unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

# Image: (1, 3, 32, 32)
image = torch.randn(1, 3, 32, 32)

# Patches: (1, 3×8×8=192, 16)  # 16 patches
patches = unfold(image)

# Reshape to (1, 16, 3, 8, 8)
patches = patches.view(1, 3, 8, 8, 16).permute(0, 4, 1, 2, 3)
print(patches.shape)  # torch.Size([1, 16, 3, 8, 8])
```

### 2. Sliding Window Operations

```python
def sliding_window_max(x, kernel_size):
    """Compute max over sliding windows."""
    unfold = nn.Unfold(kernel_size=kernel_size, stride=1)

    # x: (N, C, H, W)
    patches = unfold(x)  # (N, C×k×k, L)

    # Reshape and compute max
    N, CKK, L = patches.shape
    C = x.size(1)
    K = kernel_size
    patches = patches.view(N, C, K*K, L)

    return patches.max(dim=2).values  # (N, C, L)
```

### 3. Vision Transformer Patch Embedding

```python
class PatchEmbed(nn.Module):
    """Extract patches and embed them."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # Linear projection
        self.proj = nn.Linear(in_chans * patch_size ** 2, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        patches = self.unfold(x)  # (B, C×P×P, N)
        patches = patches.transpose(1, 2)  # (B, N, C×P×P)
        return self.proj(patches)  # (B, N, embed_dim)
```

### 4. Local Attention Patterns

```python
def local_attention_weights(q, k, window_size):
    """Compute attention within local windows."""
    B, C, H, W = q.shape

    # Extract windows
    unfold = nn.Unfold(kernel_size=window_size, stride=1, padding=window_size//2)

    q_windows = unfold(q)  # (B, C×W×W, H×W)
    k_windows = unfold(k)

    # Reshape for attention
    q_windows = q_windows.view(B, C, window_size**2, H*W)
    k_windows = k_windows.view(B, C, window_size**2, H*W)

    # Compute local attention
    attn = torch.einsum('bcwn,bcvn->bwvn', q_windows, k_windows)
    return attn.softmax(dim=2)
```

---

## Tensor.unfold()

There's also a tensor method for 1D unfolding along a dimension:

```python
# Different from nn.Unfold!
x = torch.arange(10).float()

# Unfold dimension 0 with size 3, step 2
unfolded = x.unfold(dimension=0, size=3, step=2)
print(unfolded)
# tensor([[0., 1., 2.],
#         [2., 3., 4.],
#         [4., 5., 6.],
#         [6., 7., 8.]])
```

---

## MLX Mapping

### MLX Approach

MLX doesn't have direct `fold`/`unfold` operations but they can be implemented using reshape and stride tricks, or by using `as_strided`:

```python
import mlx.core as mx

def unfold_mlx(x, kernel_size, stride=1, padding=0):
    """Unfold operation for MLX (2D images)."""
    B, C, H, W = x.shape
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sH, sW = stride if isinstance(stride, tuple) else (stride, stride)

    # Add padding
    if padding > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
        H, W = H + 2*padding, W + 2*padding

    # Output dimensions
    out_h = (H - kH) // sH + 1
    out_w = (W - kW) // sW + 1

    # Extract patches using loops (inefficient but clear)
    patches = []
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW]
            patches.append(patch.reshape(B, -1, 1))

    return mx.concatenate(patches, axis=2)


def fold_mlx(patches, output_size, kernel_size, stride=1, padding=0):
    """Fold operation for MLX."""
    B, CKK, L = patches.shape
    H, W = output_size
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sH, sW = stride if isinstance(stride, tuple) else (stride, stride)

    C = CKK // (kH * kW)

    # Padded output
    pH, pW = H + 2*padding, W + 2*padding
    out_h = (pH - kH) // sH + 1
    out_w = (pW - kW) // sW + 1

    output = mx.zeros((B, C, pH, pW))

    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = patches[:, :, idx].reshape(B, C, kH, kW)
            # This would need scatter_add for proper overlap handling
            # Simplified version (no overlap):
            if sH == kH and sW == kW:
                output[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW] = patch
            idx += 1

    # Remove padding
    if padding > 0:
        output = output[:, :, padding:-padding, padding:-padding]

    return output
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Built-in | `nn.Fold`, `nn.Unfold` | No direct equivalent |
| Performance | Optimized CUDA kernels | Manual implementation |
| Dilation | Supported | Manual handling |
| Batch support | Native | Manual handling |

### Alternative: Convolution Approach

For patch extraction in MLX, convolution with identity weights can work:

```python
import mlx.core as mx
import mlx.nn as nn

def extract_patches(x, patch_size):
    """Extract patches using depthwise conv trick."""
    B, C, H, W = x.shape
    P = patch_size

    # Create identity kernel for each channel
    kernel = mx.eye(C * P * P).reshape(C * P * P, C, P, P)

    # Depthwise-style convolution
    # (This is a conceptual approach; actual implementation may vary)
    patches = mx.conv2d(x, kernel, stride=P)

    return patches.reshape(B, C * P * P, -1)
```

---

## Best Practices

1. **Match parameters** - Fold and Unfold must have matching kernel/stride/padding

2. **Handle overlaps** - Divide by overlap count for proper reconstruction

3. **Memory efficiency** - Large kernel sizes create large intermediate tensors

4. **Use functional for simple cases** - `F.unfold()` and `F.fold()` for one-off use

5. **Consider alternatives** - For standard convolutions, use `nn.Conv2d`

---

## Summary

| Operation | Purpose | Alias |
|-----------|---------|-------|
| `nn.Unfold` | Extract sliding blocks | im2col |
| `nn.Fold` | Reconstruct from blocks | col2im |
| `F.unfold()` | Functional unfold | - |
| `F.fold()` | Functional fold | - |
| `Tensor.unfold()` | 1D unfold along dimension | - |

### Shape Formulas

**Unfold output size (L):**
```
L = ⌊(H + 2×pad - dilation×(kernel-1) - 1) / stride + 1⌋
  × ⌊(W + 2×pad - dilation×(kernel-1) - 1) / stride + 1⌋
```

**Output dimensions:**
```
Unfold: (N, C, H, W) → (N, C×kH×kW, L)
Fold:   (N, C×kH×kW, L) → (N, C, H_out, W_out)
```
