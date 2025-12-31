# Convolution and Pooling Operators

## Purpose

Convolution and pooling operators form the backbone of convolutional neural networks (CNNs), enabling spatial feature extraction and dimensionality reduction. This document covers Tier 1 convolution and pooling operators essential for computer vision tasks.

**Tier 1 Convolution/Pooling Operators** (5 total):
- `conv1d` - 1D convolution (temporal/sequential data)
- `conv2d` - 2D convolution (images)
- `conv3d` - 3D convolution (video/volumetric data)
- `max_pool2d` - 2D max pooling
- `avg_pool2d` - 2D average pooling

## Common Properties

**Compute Intensity**: Most computationally expensive operators in CNNs

**Optimization**: Leverage optimized libraries (cuDNN, MKL-DNN, MPS)

**Memory**: Often implemented via im2col or Winograd algorithms

**Gradients**: Complex backward passes with gradient computation for weights and inputs

**Backend Support**: Highly optimized on all backends (CPU, CUDA, MPS)

## Convolution Theory

### Mathematical Definition

**Discrete Convolution**:
```
(f * g)[n] = Σ_m f[m] * g[n - m]
```

**2D Convolution** (for images):
```
output[b,c_out,h,w] = Σ_{c_in} Σ_{kh} Σ_{kw}
                      input[b, c_in, h*stride + kh, w*stride + kw]
                      * weight[c_out, c_in, kh, kw]
                      + bias[c_out]
```

Where:
- `b`: batch index
- `c_in`, `c_out`: input/output channels
- `h`, `w`: spatial dimensions
- `kh`, `kw`: kernel size

### Key Parameters

**stride**: Step size for sliding window
- `stride=1`: Dense output (same resolution with padding)
- `stride=2`: Downsample by 2x
- Larger stride = smaller output

**padding**: Add zeros around input
- `padding=0`: "valid" (no padding)
- `padding=(kernel_size-1)//2`: "same" (preserve size with stride=1)

**dilation**: Spacing between kernel elements (atrous/dilated convolution)
- `dilation=1`: Standard convolution
- `dilation>1`: Increases receptive field without adding parameters

**groups**: Split channels into groups
- `groups=1`: Standard convolution
- `groups=in_channels`: Depthwise convolution
- `1 < groups < in_channels`: Grouped convolution

### Output Size Formula

```
out_size = floor((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

## Operator Details

### conv2d (2D Convolution)

**Purpose**: Extract spatial features from 2D data (images)

**Signature**:
```python
conv2d(Tensor input, Tensor weight, Tensor? bias=None,
       int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
```

**Input Shapes**:
- `input`: `(N, C_in, H_in, W_in)`
- `weight`: `(C_out, C_in/groups, kH, kW)`
- `bias`: `(C_out)` (optional)
- `output`: `(N, C_out, H_out, W_out)`

**YAML Definition** (`native_functions.yaml:1757-1759`):
```yaml
- func: conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv2d_symint
```

**High-Level Implementation** (`aten/src/ATen/native/Convolution.cpp`):
```cpp
Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

  // Delegates to optimized backend
  return at::convolution(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      /*transposed=*/false,
      /*output_padding=*/{0, 0},
      groups);
}
```

**Backend Dispatch**:
- **CPU**: Uses MKL-DNN (oneDNN) for optimized GEMM-based convolution
- **CUDA**: Uses cuDNN library (highly optimized)
- **MPS**: Uses MPSGraphConvolution2DOp

**MPS Implementation** (`mps/operations/Convolution.mm`):
```objective-c
Tensor conv2d_mps(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSGraph* graph = make_mps_graph();

    // Create placeholders
    auto inputTensor = mpsGraphRankedPlaceHolder(graph, input);
    auto weightTensor = mpsGraphRankedPlaceHolder(graph, weight);

    // Create convolution descriptor
    MPSGraphConvolution2DOpDescriptor* desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride[1]
                                                        strideInY:stride[0]
                                                  dilationRateInX:dilation[1]
                                                  dilationRateInY:dilation[0]
                                                           groups:groups
                                                     paddingLeft:padding[1]
                                                     paddingRight:padding[1]
                                                       paddingTop:padding[0]
                                                    paddingBottom:padding[0]
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    // Perform convolution
    MPSGraphTensor* outputTensor = [graph convolution2DWithSourceTensor:inputTensor
                                                          weightsTensor:weightTensor
                                                             descriptor:desc
                                                                   name:nil];

    // Add bias if present
    if (bias.has_value()) {
      auto biasTensor = mpsGraphRankedPlaceHolder(graph, *bias);
      outputTensor = [graph additionWithPrimaryTensor:outputTensor
                                      secondaryTensor:biasTensor
                                                 name:nil];
    }

    return runMPSGraph(stream, graph, {inputTensor, weightTensor}, outputTensor);
  }
}
```

**MLX Equivalent**:
```python
import mlx.core as mx
import mlx.nn as nn

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """2D convolution"""
    # MLX uses (N, H, W, C) format by default (channels-last)
    # PyTorch uses (N, C, H, W) format (channels-first)

    # Convert PyTorch NCHW to MLX NHWC
    input_nhwc = mx.transpose(input, [0, 2, 3, 1])

    # Weight: PyTorch (C_out, C_in, kH, kW) → MLX (kH, kW, C_in, C_out)
    weight_mlx = mx.transpose(weight, [2, 3, 1, 0])

    # Perform convolution
    output = mx.conv2d(
        input_nhwc,
        weight_mlx,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    # Add bias
    if bias is not None:
        output = output + bias.reshape(1, 1, 1, -1)

    # Convert back to NCHW
    output_nchw = mx.transpose(output, [0, 3, 1, 2])

    return output_nchw
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: conv2d(Tensor input, Tensor weight, Tensor? bias, ...)
  input: conv2d_backward_input(grad, input.sizes(), weight, ...)
  weight: conv2d_backward_weight(weight.sizes(), grad, input, ...)
  bias: sum(grad, [0, 2, 3])  # Sum over N, H, W dimensions
```

**Usage Examples**:
```python
import torch
import torch.nn.functional as F

# Example 1: Basic conv2d
input = torch.randn(1, 3, 32, 32)  # (N, C, H, W)
weight = torch.randn(16, 3, 3, 3)  # (out_channels, in_channels, kH, kW)
bias = torch.randn(16)

output = F.conv2d(input, weight, bias, stride=1, padding=1)
# Output shape: (1, 16, 32, 32)  # Same size due to padding=1

# Example 2: Strided convolution (downsampling)
output = F.conv2d(input, weight, bias, stride=2, padding=1)
# Output shape: (1, 16, 16, 16)  # Half size

# Example 3: Dilated convolution (increased receptive field)
output = F.conv2d(input, weight, bias, stride=1, padding=2, dilation=2)
# Larger receptive field without increasing parameters

# Example 4: Depthwise convolution
input_dw = torch.randn(1, 32, 64, 64)
weight_dw = torch.randn(32, 1, 3, 3)  # Each input channel has its own filter
output_dw = F.conv2d(input_dw, weight_dw, groups=32)  # Depthwise
# Output shape: (1, 32, 62, 62)

# Example 5: As nn.Module
conv_layer = torch.nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True
)
output = conv_layer(input)
```

**Output Size Calculation**:
```python
# For H dimension (W is analogous)
H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)

# Examples:
# stride=1, padding=1, kernel=3, dilation=1: H_out = H_in (preserve size)
# stride=2, padding=1, kernel=3, dilation=1: H_out = H_in//2 (downsample 2x)
# stride=1, padding=2, kernel=3, dilation=2: H_out = H_in (dilated, same size)
```

---

### conv1d (1D Convolution)

**Purpose**: Extract features from sequential/temporal data

**Signature**:
```python
conv1d(Tensor input, Tensor weight, Tensor? bias=None,
       int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
```

**Input Shapes**:
- `input`: `(N, C_in, L_in)` where L is sequence length
- `weight`: `(C_out, C_in/groups, kL)`
- `output`: `(N, C_out, L_out)`

**YAML Definition** (`native_functions.yaml:1753-1755`):
```yaml
- func: conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv1d_symint
```

**MLX Equivalent**:
```python
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """1D convolution"""
    # Convert NCL → NLC
    input_nlc = mx.transpose(input, [0, 2, 1])

    # Weight: (C_out, C_in, kL) → (kL, C_in, C_out)
    weight_mlx = mx.transpose(weight, [2, 1, 0])

    output = mx.conv1d(
        input_nlc,
        weight_mlx,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    if bias is not None:
        output = output + bias.reshape(1, 1, -1)

    # Convert back to NCL
    return mx.transpose(output, [0, 2, 1])
```

**Usage Examples**:
```python
# Text/sequence processing
input = torch.randn(32, 256, 100)  # (batch, channels, seq_len)
weight = torch.randn(512, 256, 3)  # (out_channels, in_channels, kernel_size)

output = F.conv1d(input, weight, stride=1, padding=1)
# Output: (32, 512, 100)

# Temporal convolution for audio
audio = torch.randn(8, 1, 16000)  # (batch, channels, samples)
conv = torch.nn.Conv1d(1, 32, kernel_size=25, stride=10, padding=12)
features = conv(audio)  # Extract audio features
```

---

### conv3d (3D Convolution)

**Purpose**: Extract features from volumetric/video data

**Signature**:
```python
conv3d(Tensor input, Tensor weight, Tensor? bias=None,
       int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor
```

**Input Shapes**:
- `input`: `(N, C_in, D_in, H_in, W_in)` (depth, height, width)
- `weight`: `(C_out, C_in/groups, kD, kH, kW)`
- `output`: `(N, C_out, D_out, H_out, W_out)`

**YAML Definition** (`native_functions.yaml:1761-1763`):
```yaml
- func: conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] dilation=1, SymInt groups=1) -> Tensor
  dispatch:
    CompositeImplicitAutograd: conv3d_symint
```

**Usage Examples**:
```python
# Video processing
video = torch.randn(4, 3, 16, 112, 112)  # (batch, RGB, frames, H, W)
weight = torch.randn(64, 3, 3, 7, 7)  # (out_ch, in_ch, kD, kH, kW)

output = F.conv3d(video, weight, stride=(1, 2, 2), padding=(1, 3, 3))
# Temporal stride=1, spatial stride=2

# Medical imaging (CT/MRI scans)
scan = torch.randn(2, 1, 64, 128, 128)  # (batch, channels, D, H, W)
conv3d_layer = torch.nn.Conv3d(1, 32, kernel_size=3, padding=1)
features = conv3d_layer(scan)
```

**Performance Note**: Conv3d is very memory-intensive (5D tensors)

---

### max_pool2d (Max Pooling 2D)

**Purpose**: Downsample by taking maximum in local regions

**Signature**:
```python
max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
           int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
```

**Behavior**:
```python
# For each spatial window of size kernel_size
output[n, c, h, w] = max(input[n, c, h*stride:h*stride+kernel, w*stride:w*stride+kernel])
```

**Parameters**:
- `kernel_size`: Size of pooling window
- `stride`: Step size (defaults to kernel_size if not specified)
- `padding`: Zero-padding around input
- `dilation`: Spacing between elements (rarely used for pooling)
- `ceil_mode`: Use ceiling instead of floor for output size calculation

**YAML Definition** (`native_functions.yaml:3957-3961`):
```yaml
- func: max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  dispatch:
    CompositeImplicitAutograd: max_pool2d
    MPS: mps_max_pool2d
```

**CPU Implementation** (`native/cpu/MaxPoolKernel.cpp`):
```cpp
void max_pool2d_kernel(
    const Tensor& input,
    int kH, int kW,
    int dH, int dW,
    int padH, int padW,
    int dilationH, int dilationW,
    Tensor& output,
    Tensor& indices) {

  // Parallel over batch and channels
  at::parallel_for(0, nBatch * nInputPlane, 0, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b) {
      int64_t n = b / nInputPlane;
      int64_t c = b % nInputPlane;

      for (int64_t oh = 0; oh < outputHeight; ++oh) {
        for (int64_t ow = 0; ow < outputWidth; ++ow) {
          // Find max in kernel window
          scalar_t max_val = std::numeric_limits<scalar_t>::lowest();
          int64_t max_idx = -1;

          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t ih = oh * dH - padH + kh * dilationH;
              int64_t iw = ow * dW - padW + kw * dilationW;

              if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                scalar_t val = input[n][c][ih][iw];
                if (val > max_val) {
                  max_val = val;
                  max_idx = ih * inputWidth + iw;
                }
              }
            }
          }

          output[n][c][oh][ow] = max_val;
          indices[n][c][oh][ow] = max_idx;  // For backward pass
        }
      }
    }
  });
}
```

**MLX Equivalent**:
```python
def max_pool2d(input, kernel_size, stride=None, padding=0):
    """Max pooling 2D"""
    if stride is None:
        stride = kernel_size

    # Convert NCHW → NHWC
    input_nhwc = mx.transpose(input, [0, 2, 3, 1])

    output = mx.max_pool2d(
        input_nhwc,
        pool_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Convert back to NCHW
    return mx.transpose(output, [0, 3, 1, 2])
```

**Gradient**:
Gradient flows only to maximum element in each window (using saved indices).

**Usage Examples**:
```python
# Example 1: Standard max pooling (2x2, stride=2)
input = torch.randn(1, 64, 32, 32)
output = F.max_pool2d(input, kernel_size=2, stride=2)
# Output: (1, 64, 16, 16)  # Half resolution

# Example 2: Non-overlapping 2x2 pooling
input = torch.tensor([[[[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]]]], dtype=torch.float32)
output = F.max_pool2d(input, kernel_size=2, stride=2)
# tensor([[[[ 6.,  8.],
#           [14., 16.]]]])
# Explanation:
# Top-left: max(1,2,5,6) = 6
# Top-right: max(3,4,7,8) = 8
# Bottom-left: max(9,10,13,14) = 14
# Bottom-right: max(11,12,15,16) = 16

# Example 3: Overlapping pooling
output = F.max_pool2d(input, kernel_size=2, stride=1)
# Output: (1, 1, 3, 3)  # Overlapping windows

# Example 4: With return indices (for unpooling)
output, indices = F.max_pool2d(input, kernel_size=2, stride=2,
                               return_indices=True)
# indices stores position of max element for gradient computation

# Example 5: As nn.Module
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
output = pool(input)
```

**Properties**:
- **Translation invariance**: Small shifts don't affect output much
- **Sparse gradients**: Only max element receives gradient
- **No parameters**: No learnable weights

---

### avg_pool2d (Average Pooling 2D)

**Purpose**: Downsample by taking average in local regions

**Signature**:
```python
avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
           int[2] padding=0, bool ceil_mode=False,
           bool count_include_pad=True, int? divisor_override=None) -> Tensor
```

**Behavior**:
```python
# For each spatial window
output[n, c, h, w] = mean(input[n, c, h*stride:h*stride+kernel, w*stride:w*stride+kernel])
```

**Parameters**:
- `count_include_pad`: Whether to include padding in average calculation
- `divisor_override`: Override divisor for averaging (advanced)

**YAML Definition** (`native_functions.yaml:12532-12538`):
```yaml
- func: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
  python_module: nn
  structured_delegate: avg_pool2d.out
  dispatch:
    MkldnnCPU: mkldnn_avg_pool2d
    QuantizedCPU: avg_pool2d_quantized_cpu
  tags: core
```

**CPU Implementation**:
```cpp
void avg_pool2d_kernel(
    const Tensor& input,
    int kH, int kW,
    int dH, int dW,
    int padH, int padW,
    bool count_include_pad,
    Tensor& output) {

  for (int64_t oh = 0; oh < outputHeight; ++oh) {
    for (int64_t ow = 0; ow < outputWidth; ++ow) {
      scalar_t sum = 0;
      int64_t count = 0;

      for (int64_t kh = 0; kh < kH; ++kh) {
        for (int64_t kw = 0; kw < kW; ++kw) {
          int64_t ih = oh * dH - padH + kh;
          int64_t iw = ow * dW - padW + kw;

          if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
            sum += input[n][c][ih][iw];
            count++;
          } else if (count_include_pad) {
            count++;  // Count padding as zero
          }
        }
      }

      output[n][c][oh][ow] = sum / count;
    }
  }
}
```

**MLX Equivalent**:
```python
def avg_pool2d(input, kernel_size, stride=None, padding=0,
               count_include_pad=True):
    """Average pooling 2D"""
    if stride is None:
        stride = kernel_size

    # Convert NCHW → NHWC
    input_nhwc = mx.transpose(input, [0, 2, 3, 1])

    output = mx.avg_pool2d(
        input_nhwc,
        pool_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Convert back to NCHW
    return mx.transpose(output, [0, 3, 1, 2])
```

**Gradient**:
Gradient distributed evenly to all elements in window.

**Usage Examples**:
```python
# Example 1: Standard avg pooling
input = torch.randn(1, 64, 32, 32)
output = F.avg_pool2d(input, kernel_size=2, stride=2)
# Output: (1, 64, 16, 16)

# Example 2: Simple average
input = torch.tensor([[[[1., 2.],
                         [3., 4.]]]])
output = F.avg_pool2d(input, kernel_size=2, stride=2)
# tensor([[[[2.5]]]])  # (1+2+3+4)/4 = 2.5

# Example 3: With padding and count_include_pad
input = torch.ones(1, 1, 2, 2)
output = F.avg_pool2d(input, kernel_size=2, stride=1, padding=1,
                      count_include_pad=True)
# Averages include padding zeros

output_no_pad = F.avg_pool2d(input, kernel_size=2, stride=1, padding=1,
                             count_include_pad=False)
# Averages exclude padding (only real values)

# Example 4: As nn.Module
pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
output = pool(input)
```

**Properties**:
- **Smooth gradients**: All elements in window receive gradient
- **No parameters**: No learnable weights
- **Gentler downsampling**: Preserves more information than max pooling

---

## Pooling Comparison

| Property | Max Pooling | Average Pooling |
|----------|-------------|-----------------|
| Operation | max(window) | mean(window) |
| Gradient | Sparse (only max) | Dense (all elements) |
| Use case | Feature detection | Smooth downsampling |
| Invariance | High | Medium |
| Information preservation | Loses averages | Loses maxima |
| Common in | CNNs (early layers) | CNNs (later layers), GAP |

## Advanced Convolution Variants

### Depthwise Separable Convolution

**Concept**: Factorize standard convolution into depthwise + pointwise

```python
# Standard convolution
standard = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# Parameters: 32 * 64 * 3 * 3 = 18,432

# Depthwise separable (equivalent)
depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
pointwise = nn.Conv2d(32, 64, kernel_size=1)
# Parameters: 32*1*3*3 + 32*64*1*1 = 288 + 2048 = 2,336 (8x reduction!)

# Usage
x = torch.randn(1, 32, 56, 56)
out = pointwise(depthwise(x))
```

### Dilated/Atrous Convolution

**Concept**: Expand receptive field without increasing parameters

```python
# Standard 3x3 conv, receptive field = 3
conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

# Dilated conv, dilation=2, effective kernel=5x5, receptive field = 5
conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)

# Same parameters, larger receptive field!
```

### Transposed Convolution (Deconvolution)

**Purpose**: Upsampling (opposite of strided convolution)

```python
# Upsample 2x
upconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# Input: (1, 64, 16, 16) → Output: (1, 32, 32, 32)
```

---

## Implementation Files

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:1753-1778` (conv1d/2d/3d)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:3957-3965` (max_pool2d)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:12532-12547` (avg_pool2d)

**High-Level Convolution**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/Convolution.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/ConvolutionMM2d.cpp`

**CPU Kernels**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/MaxPoolKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/AvgPoolKernel.cpp`

**MPS Kernels** (Metal):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Convolution.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Pooling.mm`

**cuDNN Wrappers** (CUDA):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cudnn/Conv.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cudnn/Pooling.cpp`

**Gradients**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/tools/autograd/derivatives.yaml`

---

## MLX Porting Summary

**Direct Mappings**:
```python
# PyTorch → MLX
torch.nn.functional.conv1d  → mx.conv1d (with layout conversion)
torch.nn.functional.conv2d  → mx.conv2d (with layout conversion)
torch.nn.functional.conv3d  → mx.conv3d (with layout conversion)
torch.nn.functional.max_pool2d → mx.max_pool2d (with layout conversion)
torch.nn.functional.avg_pool2d → mx.avg_pool2d (with layout conversion)
```

**Critical Differences**:

1. **Data Layout**:
   - **PyTorch**: NCHW (channels-first)
   - **MLX**: NHWC (channels-last)
   - **Solution**: Transpose before/after operations

2. **Weight Layout**:
   - **PyTorch conv2d**: `(C_out, C_in, kH, kW)`
   - **MLX conv2d**: `(kH, kW, C_in, C_out)`
   - **Solution**: Transpose weight tensors

3. **Performance**: MLX NHWC layout can be faster on Apple Silicon (Metal optimizations)

**Example Compatibility Layer**:
```python
import mlx.core as mx

class F:  # torch.nn.functional equivalent
    @staticmethod
    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # PyTorch NCHW → MLX NHWC
        input_nhwc = mx.transpose(input, [0, 2, 3, 1])

        # Weight: (C_out, C_in, kH, kW) → (kH, kW, C_in, C_out)
        weight_mlx = mx.transpose(weight, [2, 3, 1, 0])

        output = mx.conv2d(input_nhwc, weight_mlx, stride, padding, dilation, groups)

        if bias is not None:
            output = output + bias.reshape(1, 1, 1, -1)

        # MLX NHWC → PyTorch NCHW
        return mx.transpose(output, [0, 3, 1, 2])

    @staticmethod
    def max_pool2d(input, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size

        input_nhwc = mx.transpose(input, [0, 2, 3, 1])
        output = mx.max_pool2d(input_nhwc, kernel_size, stride, padding)
        return mx.transpose(output, [0, 3, 1, 2])

    @staticmethod
    def avg_pool2d(input, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size

        input_nhwc = mx.transpose(input, [0, 2, 3, 1])
        output = mx.avg_pool2d(input_nhwc, kernel_size, stride, padding)
        return mx.transpose(output, [0, 3, 1, 2])
```

**Performance Optimization**:
```python
# Store tensors in MLX's preferred NHWC layout
# Avoid repeated transposes in forward pass
class ConvBlock:
    def __init__(self):
        # Store weights in MLX format
        self.weight_mlx_format = ...  # (kH, kW, C_in, C_out)

    def forward(self, x_nhwc):
        # Input already in NHWC
        out = mx.conv2d(x_nhwc, self.weight_mlx_format, ...)
        # Output in NHWC
        return out
```

Convolution and pooling operators are the backbone of CNNs and have efficient MLX implementations, with the main consideration being data layout differences between PyTorch (NCHW) and MLX (NHWC).
