# C++ Extensions (torch.utils.cpp_extension)

## Purpose

PyTorch's C++ extension system allows developers to write custom operators and modules in C++ (and CUDA/ROCm/SYCL for GPU acceleration) while seamlessly integrating them with Python. This enables:
- High-performance custom operators
- Access to low-level hardware features
- Integration with existing C++ libraries
- Custom autograd functions with optimized backward passes

**Reference Files:**
- `torch/utils/cpp_extension.py` - Extension building infrastructure
- `torch/include/` - PyTorch C++ headers
- `torch/csrc/api/` - C++ frontend (libtorch)

---

## Overview

Two primary methods for building C++ extensions:

1. **JIT Compilation** (`load`, `load_inline`): Compile at runtime, cached for subsequent runs
2. **Ahead-of-Time (AOT)** (`setuptools`): Traditional build with `setup.py`

### Extension Types

| Type | Use Case | Backend |
|------|----------|---------|
| `CppExtension` | CPU-only operations | C++ compiler |
| `CUDAExtension` | NVIDIA GPU operations | nvcc + C++ |
| `SyclExtension` | Intel GPU operations | icpx + C++ |

---

## JIT Compilation

### load()

Compile and load a C++ extension at runtime.

```python
from torch.utils.cpp_extension import load

# Compile from source files
my_extension = load(
    name='my_extension',
    sources=['extension.cpp', 'extension_cuda.cu'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    verbose=True
)

# Use the extension
result = my_extension.forward(input_tensor)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Extension module name |
| `sources` | list[str] | C++/CUDA source files |
| `extra_cflags` | list[str] | Additional C++ compiler flags |
| `extra_cuda_cflags` | list[str] | Additional CUDA compiler flags |
| `extra_ldflags` | list[str] | Additional linker flags |
| `extra_include_paths` | list[str] | Include directories |
| `build_directory` | str | Build output directory |
| `verbose` | bool | Print compilation commands |
| `with_cuda` | bool | Enable CUDA support |

### load_inline()

Compile C++ code provided as strings.

```python
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
"""

cuda_source = """
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

my_extension = load_inline(
    name='my_inline_extension',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['my_add'],
    verbose=True
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cpp_sources` | list[str] | C++ source code strings |
| `cuda_sources` | list[str] | CUDA source code strings |
| `functions` | list[str] | Functions to export |
| `extra_cflags` | list[str] | C++ compiler flags |
| `extra_cuda_cflags` | list[str] | CUDA compiler flags |

---

## AOT Compilation with setuptools

### CppExtension

Creates a CPU-only extension.

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='my_extension',
    ext_modules=[
        CppExtension(
            name='my_extension',
            sources=['src/my_extension.cpp'],
            extra_compile_args=['-O3', '-march=native'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### CUDAExtension

Creates a CUDA-enabled extension.

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='my_cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='my_cuda_extension',
            sources=[
                'src/my_extension.cpp',
                'src/cuda_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### BuildExtension

Custom setuptools command with PyTorch-specific features:
- Automatic C++17 standard flag
- Mixed C++/CUDA compilation
- Ninja build system support (faster builds)
- ABI compatibility checking

```python
# Options
BuildExtension.with_options(use_ninja=True)
BuildExtension.with_options(no_python_abi_suffix=True)
```

---

## C++ API

### Tensor Operations

```cpp
#include <torch/extension.h>

torch::Tensor my_function(torch::Tensor input) {
    // Check tensor properties
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.device().is_cpu(), "Input must be on CPU");

    // Create output tensor
    auto output = torch::empty_like(input);

    // Access data
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // Process
    int64_t numel = input.numel();
    for (int64_t i = 0; i < numel; i++) {
        output_data[i] = input_data[i] * 2.0f;
    }

    return output;
}
```

### Tensor Accessors

Type-safe tensor access:

```cpp
// 2D float tensor accessor
auto accessor = tensor.accessor<float, 2>();
float val = accessor[i][j];

// Packed accessor for GPU (contiguous memory)
auto packed = tensor.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
```

### Creating Tensors

```cpp
// Empty tensor
auto t1 = torch::empty({3, 4}, torch::kFloat32);

// Zeros/Ones
auto t2 = torch::zeros({3, 4}, torch::kFloat64);
auto t3 = torch::ones({3, 4}, torch::kInt32);

// From data
std::vector<float> data = {1.0f, 2.0f, 3.0f};
auto t4 = torch::from_blob(data.data(), {3}, torch::kFloat32);

// Copy to specific device
auto t5 = t1.to(torch::kCUDA);

// Clone (deep copy)
auto t6 = t1.clone();
```

### Tensor Options

```cpp
auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .device(torch::kCUDA, 0)
    .requires_grad(true);

auto tensor = torch::randn({3, 4}, options);
```

---

## CUDA Kernels

### Kernel Implementation

```cpp
// cuda_kernels.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void my_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor my_cuda_function(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);

    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return output;
}
```

### Using AT_DISPATCH_FLOATING_TYPES

Type-dispatching macro for generic kernels:

```cpp
#include <ATen/Dispatch.h>

torch::Tensor my_function(torch::Tensor input) {
    auto output = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_function", [&] {
        // scalar_t is the resolved type (float or double)
        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        // ... use input_data and output_data
    });

    return output;
}
```

Available dispatch macros:
- `AT_DISPATCH_FLOATING_TYPES` - float, double
- `AT_DISPATCH_FLOATING_TYPES_AND_HALF` - float, double, half
- `AT_DISPATCH_ALL_TYPES` - all numeric types
- `AT_DISPATCH_ALL_TYPES_AND` - all types plus specified

---

## Custom Autograd Functions

### C++ Autograd Function

```cpp
#include <torch/extension.h>

class MyFunction : public torch::autograd::Function<MyFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        double alpha
    ) {
        // Save for backward
        ctx->save_for_backward({input});
        ctx->saved_data["alpha"] = alpha;

        return input * alpha;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto alpha = ctx->saved_data["alpha"].toDouble();
        auto grad_output = grad_outputs[0];

        // Gradient w.r.t. input
        auto grad_input = grad_output * alpha;

        // Return gradients for each input (None for non-tensor inputs)
        return {grad_input, torch::Tensor()};
    }
};

// Wrapper function
torch::Tensor my_function(torch::Tensor input, double alpha) {
    return MyFunction::apply(input, alpha);
}
```

---

## Binding to Python

### PYBIND11_MODULE

```cpp
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Function declarations
torch::Tensor my_function(torch::Tensor input);
torch::Tensor my_cuda_function(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_function", &my_function, "My custom function");
    m.def("my_cuda_function", &my_cuda_function, "My CUDA function");
}
```

### With Docstrings

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_function",
          &my_function,
          "Multiply input by 2",
          py::arg("input"));
}
```

### Binding Classes

```cpp
class MyModule {
public:
    MyModule(int size) : size_(size) {}
    torch::Tensor forward(torch::Tensor input);
private:
    int size_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MyModule>(m, "MyModule")
        .def(py::init<int>())
        .def("forward", &MyModule::forward);
}
```

---

## Include Paths and Libraries

### Getting Paths

```python
from torch.utils.cpp_extension import include_paths, library_paths

# PyTorch include directories
includes = include_paths()
# Returns: ['.../torch/include', '.../torch/include/torch/csrc/api/include']

# PyTorch library directories
libs = library_paths()
# Returns: ['.../torch/lib']
```

### Linking

```python
CppExtension(
    name='my_extension',
    sources=['my_extension.cpp'],
    include_dirs=include_paths(),
    library_dirs=library_paths(),
    libraries=['c10', 'torch', 'torch_cpu'],  # Core libraries
)
```

---

## Common Patterns

### Checking Device and Contiguity

```cpp
torch::Tensor my_function(torch::Tensor input) {
    // Ensure contiguous (make copy if needed)
    input = input.contiguous();

    // Device check
    if (input.is_cuda()) {
        return my_cuda_impl(input);
    } else {
        return my_cpu_impl(input);
    }
}
```

### Error Handling

```cpp
#include <torch/extension.h>

torch::Tensor my_function(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Expected 2D tensor, got ", input.dim(), "D");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Expected float32 tensor");
    TORCH_INTERNAL_ASSERT(input.numel() > 0);  // Internal assertion

    // ...
}
```

### TensorIterator (Efficient Element-wise)

```cpp
#include <ATen/native/TensorIterator.h>

void my_elementwise(torch::Tensor& output, const torch::Tensor& input) {
    auto iter = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(input)
        .build();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_elementwise", [&] {
        at::native::cpu_kernel(iter, [](scalar_t x) {
            return x * 2;
        });
    });
}
```

---

## Build Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_HOME` | CUDA installation path |
| `TORCH_CUDA_ARCH_LIST` | Target GPU architectures |
| `MAX_JOBS` | Parallel compilation jobs |
| `TORCH_DONT_CHECK_COMPILER_ABI` | Skip ABI check |
| `CC`, `CXX` | C/C++ compilers |

### Compiler Requirements

| Platform | Compiler | Minimum Version |
|----------|----------|-----------------|
| Linux | GCC | 5.0.0 |
| macOS | Clang | Apple Clang |
| Windows | MSVC | 19.0.24215 |

### CUDA Architecture Flags

```python
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0 7.5 8.0 8.6'

# Or in code:
CUDAExtension(
    name='my_extension',
    sources=['my_extension.cu'],
    extra_cuda_cflags=['-arch=sm_80'],
)
```

---

## Ninja Build System

PyTorch extensions use Ninja by default for faster builds.

```python
from torch.utils.cpp_extension import is_ninja_available, verify_ninja_availability

# Check availability
if is_ninja_available():
    print("Ninja is available")

# Verify (raises if not available)
verify_ninja_availability()
```

### Install Ninja

```bash
# pip
pip install ninja

# conda
conda install ninja

# apt (Linux)
sudo apt install ninja-build

# brew (macOS)
brew install ninja
```

---

## MLX Porting Considerations

### Conceptual Mapping

| PyTorch C++ | MLX Equivalent |
|-------------|----------------|
| `torch/extension.h` | N/A (Python-first design) |
| CUDA kernels | Metal compute shaders |
| `CUDAExtension` | Metal extension patterns |
| TensorIterator | MLX vectorized ops |
| pybind11 bindings | Python-native or nanobind |

### Metal Compute Kernels

MLX uses Metal for GPU acceleration on Apple Silicon:

```metal
// Metal shader (my_kernel.metal)
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * 2.0f;
}
```

### MLX Custom Operations

MLX extensions are typically written in Python with compiled kernels:

```python
import mlx.core as mx

# Custom primitive (advanced)
class MyOp(mx.Primitive):
    def __init__(self):
        super().__init__()

    def eval_cpu(self, inputs, outputs):
        # CPU implementation
        pass

    def eval_gpu(self, inputs, outputs):
        # Metal kernel dispatch
        pass
```

### Key Differences

1. **Build System**: MLX doesn't have a cpp_extension equivalent; custom ops are written differently
2. **GPU Backend**: Metal instead of CUDA
3. **ABI**: MLX is Python-first; C++ API is internal
4. **Compilation**: JIT for Metal shaders, not C++

### Implementation Strategy

For porting PyTorch C++ extensions to MLX:

1. **Pure MLX**: Rewrite using MLX primitives where possible
2. **Metal Shaders**: Write custom Metal compute kernels for specialized operations
3. **Python Extensions**: Use standard Python C extensions with NumPy/MLX array protocol
4. **Hybrid**: Combine MLX operations with custom Metal kernels

---

## Summary

| Feature | Function/Class |
|---------|----------------|
| JIT compile files | `load()` |
| JIT compile strings | `load_inline()` |
| CPU extension | `CppExtension` |
| CUDA extension | `CUDAExtension` |
| Build command | `BuildExtension` |
| Include paths | `include_paths()` |
| Library paths | `library_paths()` |
| Ninja check | `is_ninja_available()` |

The C++ extension system is critical for high-performance custom operators in PyTorch. For MLX, custom operations follow different patterns based on Metal compute shaders and Python-first design principles.
