# Type System - ScalarType, Device, Layout, MemoryFormat

## Purpose

PyTorch's type system defines the fundamental properties of tensors: what data they contain (dtype), where they live (device), how they're organized in memory (layout), and how elements are arranged (memory format). Understanding this type system is essential for:
- Implementing multi-backend support (CPU, CUDA, Metal)
- Optimizing memory access patterns
- Supporting quantization and reduced-precision types
- Ensuring type safety in operator implementations

This document covers all four type abstractions that characterize a PyTorch tensor.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       PyTorch Tensor                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TensorImpl                                               │  │
│  │  ┌────────────┬────────────┬────────────┬─────────────┐ │  │
│  │  │ ScalarType │   Device   │   Layout   │ MemoryFormat│ │  │
│  │  │  (dtype)   │  (where)   │   (how)    │  (order)    │ │  │
│  │  ├────────────┼────────────┼────────────┼─────────────┤ │  │
│  │  │ Float32    │ CPU        │ Strided    │ Contiguous  │ │  │
│  │  │ Int64      │ CUDA:0     │ Sparse     │ ChannelsLast│ │  │
│  │  │ BFloat16   │ MPS        │ SparseCsr  │ ...         │ │  │
│  │  │ ...        │ ...        │ ...        │             │ │  │
│  │  └────────────┴────────────┴────────────┴─────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Four Orthogonal Dimensions**:
1. **ScalarType** (dtype): What type of data does each element have?
2. **Device**: Which compute device holds the tensor data?
3. **Layout**: How is the data organized (dense, sparse, etc.)?
4. **MemoryFormat**: What order are elements stored in (row-major, channels-last, etc.)?

## Key Components

### 1. ScalarType - Element Data Type

**File**: [reference/pytorch/torch/headeronly/core/ScalarType.h:258-264](reference/pytorch/torch/headeronly/core/ScalarType.h)

**Purpose**: Enum representing the data type of tensor elements.

**Definition**:

```cpp
enum class ScalarType : int8_t {
  Byte = 0,           // uint8_t
  Char = 1,           // int8_t
  Short = 2,          // int16_t
  Int = 3,            // int32_t
  Long = 4,           // int64_t
  Half = 5,           // float16 (IEEE 754 half precision)
  Float = 6,          // float32
  Double = 7,         // float64
  ComplexHalf = 8,    // complex<float16>
  ComplexFloat = 9,   // complex<float32>
  ComplexDouble = 10, // complex<float64>
  Bool = 11,          // bool

  // Quantized types
  QInt8 = 12,         // Quantized int8
  QUInt8 = 13,        // Quantized uint8
  QInt32 = 14,        // Quantized int32
  QUInt4x2 = 16,      // 4-bit quantized (packed 2 per byte)
  QUInt2x4 = 17,      // 2-bit quantized (packed 4 per byte)

  // Reduced precision floats
  BFloat16 = 15,      // Brain float16 (truncated float32)
  Float8_e5m2 = 23,   // 8-bit float (5 exp, 2 mantissa)
  Float8_e4m3fn = 24, // 8-bit float (4 exp, 3 mantissa)
  Float8_e5m2fnuz = 25,
  Float8_e4m3fnuz = 26,
  Float8_e8m0fnu = 44,
  Float4_e2m1fn_x2 = 45, // 4-bit float packed

  // Unsigned integers
  UInt16 = 27,        // uint16_t
  UInt32 = 28,        // uint32_t
  UInt64 = 29,        // uint64_t

  // Low-bit integers (1-7 bits)
  UInt1 = 30, UInt2 = 31, UInt3 = 32, UInt4 = 33,
  UInt5 = 34, UInt6 = 35, UInt7 = 36,
  Int1 = 37, Int2 = 38, Int3 = 39, Int4 = 40,
  Int5 = 41, Int6 = 42, Int7 = 43,

  // Bit packing types
  Bits1x8 = 18,       // 1-bit values packed 8 per byte
  Bits2x4 = 19,       // 2-bit values packed 4 per byte
  Bits4x2 = 20,       // 4-bit values packed 2 per byte
  Bits8 = 21,         // 8-bit bitfield
  Bits16 = 22,        // 16-bit bitfield

  Undefined,
  NumOptions
};
```

**Categories**:

```cpp
// Standard floating point
isFloatingType(t):
  Float, Double, Half, BFloat16, Float8_*, Float4_*

// Reduced precision (< 32-bit float)
isReducedFloatingType(t):
  Half, BFloat16, Float8_e5m2, Float8_e4m3fn, Float4_e2m1fn_x2, ...

// Complex numbers
isComplexType(t):
  ComplexHalf, ComplexFloat, ComplexDouble

// Integers
isIntegralType(t, includeBool):
  Byte, Char, Short, Int, Long, UInt16, UInt32, UInt64
  (+ Bool if includeBool=true)

// Quantized
isQIntType(t):
  QInt8, QUInt8, QInt32, QUInt4x2, QUInt2x4
```

**Type Mapping**:

```cpp
// C++ type → ScalarType
template<typename T>
struct CppTypeToScalarType;

CppTypeToScalarType<float>::value == ScalarType::Float
CppTypeToScalarType<int64_t>::value == ScalarType::Long
CppTypeToScalarType<c10::Half>::value == ScalarType::Half

// ScalarType → C++ type
template<ScalarType ST>
using ScalarTypeToCPPTypeT = /* corresponding C++ type */;

ScalarTypeToCPPTypeT<ScalarType::Float> == float
ScalarTypeToCPPTypeT<ScalarType::Long> == int64_t
```

**Size Information**:

```cpp
inline size_t elementSize(ScalarType t) {
  switch (t) {
    case ScalarType::Byte:         return sizeof(uint8_t);   // 1
    case ScalarType::Char:         return sizeof(int8_t);    // 1
    case ScalarType::Short:        return sizeof(int16_t);   // 2
    case ScalarType::Int:          return sizeof(int32_t);   // 4
    case ScalarType::Long:         return sizeof(int64_t);   // 8
    case ScalarType::Half:         return sizeof(c10::Half); // 2
    case ScalarType::Float:        return sizeof(float);     // 4
    case ScalarType::Double:       return sizeof(double);    // 8
    case ScalarType::ComplexFloat: return sizeof(c10::complex<float>); // 8
    case ScalarType::Bool:         return sizeof(bool);      // 1
    // ... etc
  }
}
```

### 2. Device - Compute Device Location

**File**: [reference/pytorch/c10/core/Device.h:31-186](reference/pytorch/c10/core/Device.h)

**Purpose**: Identifies which compute device a tensor resides on.

**Structure**:

```cpp
struct Device {
  DeviceType type_;    // CPU, CUDA, MPS, etc.
  DeviceIndex index_;  // Which device of that type (e.g., GPU 0, GPU 1)

  // Constructors
  Device(DeviceType type, DeviceIndex index = -1);
  Device(const std::string& device_string); // e.g., "cuda:0", "mps"

  // Queries
  DeviceType type() const;
  DeviceIndex index() const;
  bool has_index() const;  // true if index != -1

  // Convenience predicates
  bool is_cpu() const;
  bool is_cuda() const;
  bool is_mps() const;
  bool is_metal() const;
  bool is_xla() const;
  // ... more device checks
};
```

**DeviceType Enum** ([reference/pytorch/torch/headeronly/core/DeviceType.h:35-62](reference/pytorch/torch/headeronly/core/DeviceType.h)):

```cpp
enum class DeviceType : int8_t {
  CPU = 0,           // x86/ARM CPU
  CUDA = 1,          // NVIDIA CUDA GPU
  MKLDNN = 2,        // Intel MKL-DNN (reserved, deprecated)
  OPENGL = 3,        // OpenGL
  OPENCL = 4,        // OpenCL
  IDEEP = 5,         // Intel IDEEP
  HIP = 6,           // AMD ROCm/HIP GPU
  FPGA = 7,          // FPGA devices
  MAIA = 8,          // Microsoft MAIA
  XLA = 9,           // Google XLA (TPU)
  Vulkan = 10,       // Vulkan API
  Metal = 11,        // Apple Metal (legacy)
  XPU = 12,          // Intel XPU
  MPS = 13,          // Metal Performance Shaders (Apple GPU)
  Meta = 14,         // Meta tensors (no data)
  HPU = 15,          // Habana HPU
  VE = 16,           // NEC SX-Aurora
  Lazy = 17,         // Lazy evaluation backend
  IPU = 18,          // Graphcore IPU
  MTIA = 19,         // Meta Training/Inference Accelerator
  PrivateUse1 = 20,  // Custom backend slot

  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};
```

**Key Device Types for MLX**:
- **CPU**: Standard CPU execution
- **MPS**: Metal Performance Shaders (Apple GPU, M1/M2/M3/M4)
- **Metal**: Legacy Metal support
- **Meta**: Fake tensors for shape/type analysis without data

**Device Index**:
- `-1`: Current device (abstract)
- `0, 1, 2, ...`: Specific device ordinal
- CPU always has index `-1` or `0`

**String Parsing**:

```python
# Python examples
torch.device("cpu")        # Device(CPU, -1)
torch.device("cuda")       # Device(CUDA, -1) [current CUDA device]
torch.device("cuda:0")     # Device(CUDA, 0)
torch.device("cuda:1")     # Device(CUDA, 1)
torch.device("mps")        # Device(MPS, -1)
```

### 3. Layout - Memory Organization

**File**: [reference/pytorch/c10/core/Layout.h:10-62](reference/pytorch/c10/core/Layout.h)

**Purpose**: Describes how tensor data is organized in memory (dense vs sparse, etc.).

**Enum** ([reference/pytorch/torch/headeronly/core/Layout.h](reference/pytorch/torch/headeronly/core/Layout.h)):

```cpp
enum class Layout : int8_t {
  Strided = 0,    // Dense tensor with strides (default, most common)
  Sparse = 1,     // Sparse COO (coordinate) format
  SparseCsr = 2,  // Sparse compressed sparse row
  SparseCsc = 3,  // Sparse compressed sparse column
  SparseBsr = 4,  // Sparse block sparse row
  SparseBsc = 5,  // Sparse block sparse column
  Mkldnn = 6,     // Intel MKL-DNN optimized layout
  Jagged = 7,     // Jagged/nested tensors (variable-length dimensions)
  NumOptions
};
```

**Strided Layout** (Default):
- Dense tensor with explicit strides
- Element at index `[i, j, k]` located at offset: `i*stride[0] + j*stride[1] + k*stride[2]`
- Supports arbitrary memory layouts via strides (row-major, column-major, transposed, etc.)

**Sparse Layouts**:
- Store only non-zero elements
- Different formats optimize different access patterns:
  - **COO** (Coordinate): List of (index, value) pairs
  - **CSR** (Compressed Sparse Row): Efficient row access
  - **CSC** (Compressed Sparse Column): Efficient column access
  - **BSR/BSC** (Block Sparse): Sparse at block granularity

**Layout Inference**:

```cpp
inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    default:
      return Layout::Strided;
  }
}
```

### 4. MemoryFormat - Element Ordering

**File**: [reference/pytorch/c10/core/MemoryFormat.h:18-264](reference/pytorch/c10/core/MemoryFormat.h)

**Purpose**: Specifies the order in which tensor elements are stored in memory (for strided tensors).

**Enum** ([reference/pytorch/torch/headeronly/core/MemoryFormat.h](reference/pytorch/torch/headeronly/core/MemoryFormat.h)):

```cpp
enum class MemoryFormat : int8_t {
  Contiguous = 0,      // Standard row-major (C-style) ordering
  Preserve = 1,        // Preserve current format (used in operations)
  ChannelsLast = 2,    // NHWC format (channels as fastest-changing dimension)
  ChannelsLast3d = 3,  // NDHWC format (for 3D convolutions)
  NumOptions
};
```

**Contiguous (Default)**:
- Row-major ordering: rightmost index varies fastest
- For `[N, C, H, W]` tensor: `strides = [C*H*W, H*W, W, 1]`
- Standard NumPy/C layout

**ChannelsLast (NHWC)**:
- Optimized for convolutional neural networks
- For `[N, C, H, W]` tensor: `strides = [C*H*W, 1, C*W, C]`
- Memory layout: `[N, H, W, C]` - channels as innermost dimension
- Better cache locality for convolutions (spatial neighbors are close in memory)

**ChannelsLast3d (NDHWC)**:
- 3D version for volumetric data
- For `[N, C, D, H, W]` tensor: `strides = [C*D*H*W, 1, C*H*W, C*W, C]`

**Stride Computation**:

```cpp
// Contiguous strides for [N, C, H, W]
template<typename T>
std::vector<T> get_contiguous_strides(ArrayRef<T> sizes) {
  std::vector<T> strides(sizes.size());
  T stride = 1;
  for (int i = sizes.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }
  return strides;
  // Result: [C*H*W, H*W, W, 1]
}

// ChannelsLast strides for [N, C, H, W]
std::vector<int64_t> get_channels_last_strides_2d(IntArrayRef sizes) {
  // sizes = [N, C, H, W]
  std::vector<int64_t> strides(4);
  strides[1] = 1;            // C (channels fastest)
  strides[3] = sizes[1];     // W
  strides[2] = strides[3] * sizes[3];  // H
  strides[0] = strides[2] * sizes[2];  // N
  return strides;
  // Result: [C*H*W, 1, C*W, C]
}
```

**Detection**:

```cpp
template<typename T>
bool is_channels_last_strides_2d(
    const ArrayRef<T> sizes,
    const ArrayRef<T> strides) {
  // Check if strides follow pattern: C < W < H < N
  // with special handling for size-1 dimensions
  // ...implementation...
}
```

## Data Flow

### Example 1: Type Information from Tensor

```python
import torch

# Create tensor with specific types
x = torch.randn(4, 3, 224, 224, dtype=torch.float32, device='cpu')

# Query type information
print(x.dtype)          # torch.float32 (ScalarType::Float)
print(x.device)         # cpu (Device(CPU, -1))
print(x.layout)         # torch.strided (Layout::Strided)
print(x.is_contiguous())  # True (MemoryFormat::Contiguous)

# Type conversion
y = x.to(dtype=torch.float16)  # Half precision
z = x.to(device='mps')         # Move to Metal GPU
```

### Example 2: Channels-Last Convolution

```python
# Standard contiguous format (NCHW)
x_nchw = torch.randn(1, 3, 224, 224)
print(x_nchw.stride())  # (150528, 50176, 224, 1) = [C*H*W, H*W, W, 1]

# Convert to channels-last (NHWC memory layout, but still indexed as NCHW)
x_nhwc = x_nchw.to(memory_format=torch.channels_last)
print(x_nhwc.stride())  # (150528, 1, 672, 3) = [C*H*W, 1, C*W, C]

# Both represent [1, 3, 224, 224] shape, but different memory order
print(x_nchw.shape)  # torch.Size([1, 3, 224, 224])
print(x_nhwc.shape)  # torch.Size([1, 3, 224, 224])

# Channels-last is faster for convolutions on some hardware
conv = torch.nn.Conv2d(3, 64, kernel_size=3)
conv = conv.to(memory_format=torch.channels_last)

output = conv(x_nhwc)  # Optimized channels-last convolution
```

### Example 3: Device Transfer

```cpp
// C++ example: Moving tensors between devices
at::Tensor cpu_tensor = torch::randn({4, 4});

// Check device
TORCH_CHECK(cpu_tensor.device().is_cpu());

// Move to CUDA
at::Tensor cuda_tensor = cpu_tensor.to(torch::kCUDA);
TORCH_CHECK(cuda_tensor.device().is_cuda());

// Move to MPS (Apple GPU)
at::Tensor mps_tensor = cpu_tensor.to(torch::kMPS);
TORCH_CHECK(mps_tensor.device().is_mps());

// Device index
at::Tensor cuda_0 = cpu_tensor.to(at::Device(at::kCUDA, 0));
at::Tensor cuda_1 = cpu_tensor.to(at::Device(at::kCUDA, 1));
```

## Code Examples

### Example 1: Type Queries

```cpp
#include <ATen/ATen.h>

void inspect_tensor_types(const at::Tensor& t) {
  // ScalarType
  std::cout << "dtype: " << t.scalar_type() << std::endl;
  std::cout << "element size: " << t.element_size() << " bytes" << std::endl;
  std::cout << "is floating point: " << at::isFloatingType(t.scalar_type()) << std::endl;
  std::cout << "is complex: " << at::isComplexType(t.scalar_type()) << std::endl;

  // Device
  std::cout << "device type: " << t.device().type() << std::endl;
  std::cout << "device index: " << t.device().index() << std::endl;
  std::cout << "is CPU: " << t.device().is_cpu() << std::endl;
  std::cout << "is MPS: " << t.device().is_mps() << std::endl;

  // Layout
  std::cout << "layout: " << t.layout() << std::endl;
  std::cout << "is strided: " << (t.layout() == at::kStrided) << std::endl;

  // MemoryFormat
  std::cout << "is contiguous: " << t.is_contiguous() << std::endl;
  std::cout << "is channels last: " << t.is_contiguous(at::MemoryFormat::ChannelsLast) << std::endl;
}
```

### Example 2: Creating Tensors with Specific Types

```cpp
// Create float32 tensor on CPU
auto cpu_f32 = torch::randn({3, 4}, torch::dtype(torch::kFloat32).device(torch::kCPU));

// Create float16 tensor on CUDA
auto cuda_f16 = torch::randn({3, 4}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

// Create bfloat16 tensor on MPS
auto mps_bf16 = torch::randn({3, 4}, torch::dtype(torch::kBFloat16).device(torch::kMPS));

// Create int64 tensor
auto int_tensor = torch::randint(0, 100, {3, 4}, torch::dtype(torch::kInt64));

// Create bool tensor
auto bool_tensor = torch::randint(0, 2, {3, 4}, torch::dtype(torch::kBool));

// Create complex tensor
auto complex_tensor = torch::randn({3, 4}, torch::dtype(torch::kComplexFloat));
```

### Example 3: Channels-Last Optimization

```cpp
// Standard convolution (NCHW)
auto input_nchw = torch::randn({1, 64, 56, 56});
auto conv = torch::nn::Conv2d(
    torch::nn::Conv2dOptions(64, 128, 3).padding(1));

auto output_nchw = conv->forward(input_nchw);

// Channels-last convolution (NHWC memory, faster on some hardware)
auto input_nhwc = input_nchw.to(at::MemoryFormat::ChannelsLast);
auto conv_cl = conv->to(at::MemoryFormat::ChannelsLast);

auto output_nhwc = conv_cl->forward(input_nhwc);

// Output is also channels-last
TORCH_CHECK(output_nhwc.is_contiguous(at::MemoryFormat::ChannelsLast));
```

## MLX Porting Considerations

### What MLX Already Provides

MLX has its own type system:
- **Dtype**: Similar to ScalarType (float32, float16, int32, etc.)
- **Device**: CPU or GPU (unified memory makes device less critical)
- **Array**: Equivalent to PyTorch's Tensor

### What Needs Adaptation

1. **ScalarType Completeness**:
   - PyTorch: 46 scalar types (including quantized, float8, low-bit)
   - MLX: Smaller set focused on ML workloads
   - **Action**: Identify which PyTorch types MLX needs for compatibility
   - **Priority**: Float32, Float16, BFloat16, Int32, Int64, Bool, UInt8, Int8

2. **Device Model**:
   - PyTorch: Explicit device transfers, separate CPU/GPU memory
   - MLX: Unified memory on Apple Silicon
   - **Advantage**: Simpler than PyTorch, no explicit transfers needed
   - **Action**: MLX's device model is already superior for Apple hardware

3. **Layout Support**:
   - PyTorch: Strided, Sparse COO/CSR/CSC, Mkldnn, Jagged
   - MLX: Primarily strided (dense) tensors
   - **Action**: Implement sparse support if needed for specific models

4. **MemoryFormat**:
   - PyTorch: Contiguous, ChannelsLast, ChannelsLast3d
   - MLX: May not need explicit channels-last (Metal handles optimally)
   - **Opportunity**: Metal Performance Shaders may auto-optimize layout

### Metal-Specific Opportunities

1. **Reduced Precision Types**:
   - Metal supports float16 natively
   - PyTorch's BFloat16, Float8 types relevant for quantization
   - **Action**: Map to Metal's native types (MTLDataType)

2. **Unified Memory**:
   - No CPU ↔ GPU transfers needed
   - Simplifies device management significantly
   - **Advantage**: Eliminate entire class of bugs related to device placement

3. **Channels-Last Optimization**:
   - Metal GPU may prefer NHWC layout for convolutions
   - MPSGraph can handle layout transformations internally
   - **Opportunity**: Let Metal compiler handle layout, don't burden MLX users

4. **Quantization**:
   - PyTorch has extensive quantization types (QInt8, QUInt4x2, etc.)
   - Metal supports low-precision compute
   - **Action**: Map PyTorch quantization types to Metal equivalents for model compatibility

### Recommendations for MLX

1. **Core Type Support**:
   - Implement PyTorch's common types: Float32, Float16, BFloat16, Int32, Int64, Bool
   - Add as needed: Int8, UInt8 (for quantization compatibility)
   - Skip: Complex types (unless needed), ultra-low-bit types (Int1-7)

2. **Device Abstraction**:
   - Keep MLX's simple device model (CPU/GPU with unified memory)
   - Provide compatibility shim for PyTorch's explicit `.to(device)` API

3. **Layout Strategy**:
   - Strided tensors are sufficient for most ML workloads
   - Add sparse support only if specific models require it
   - Let Metal handle layout optimization internally

4. **Memory Format**:
   - Support contiguous as default
   - Channels-last can be internal optimization (transparent to user)
   - Avoid exposing complexity to users unless necessary

## Critical File References

**ScalarType**:
- [torch/headeronly/core/ScalarType.h:103-149](reference/pytorch/torch/headeronly/core/ScalarType.h) - Complete list of all 46 scalar types
- [torch/headeronly/core/ScalarType.h:258-264](reference/pytorch/torch/headeronly/core/ScalarType.h) - ScalarType enum definition
- [c10/core/ScalarType.h:43-54](reference/pytorch/c10/core/ScalarType.h) - elementSize() function
- [c10/core/ScalarType.h:56-195](reference/pytorch/c10/core/ScalarType.h) - Type category predicates

**Device**:
- [c10/core/Device.h:31-186](reference/pytorch/c10/core/Device.h) - Device struct with type and index
- [torch/headeronly/core/DeviceType.h:35-62](reference/pytorch/torch/headeronly/core/DeviceType.h) - DeviceType enum with all 21 device types
- [c10/core/DeviceType.h:13-23](reference/pytorch/c10/core/DeviceType.h) - Device type utilities

**Layout**:
- [torch/headeronly/core/Layout.h](reference/pytorch/torch/headeronly/core/Layout.h) - Layout enum
- [c10/core/Layout.h:10-62](reference/pytorch/c10/core/Layout.h) - Layout utilities and backend mapping

**MemoryFormat**:
- [torch/headeronly/core/MemoryFormat.h](reference/pytorch/torch/headeronly/core/MemoryFormat.h) - MemoryFormat enum
- [c10/core/MemoryFormat.h:18-264](reference/pytorch/c10/core/MemoryFormat.h) - Complete implementation with stride computation

## Next Steps

1. Study **memory-model.md** to understand how devices allocate and manage memory
2. Read **operator-overview.md** to see how operators handle different types
3. Examine **metal-mps-backend.md** to see how PyTorch's Metal backend handles type conversions
4. For MLX porting, audit which PyTorch types are actually used by target models (e.g., LLMs typically use Float32, Float16, BFloat16)
