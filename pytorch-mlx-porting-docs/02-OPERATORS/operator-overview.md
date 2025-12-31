# Operator Overview - YAML Schema, Code Generation, and Registration

## Purpose

PyTorch defines over 2,666 operators in a declarative YAML file (`native_functions.yaml`) and uses code generation to create C++ implementations, Python bindings, and autograd formulas. This approach ensures consistency across backends (CPU, CUDA, Metal, XLA) and reduces boilerplate code.

Understanding the operator system is critical for MLX porting because:
- It defines the complete API surface you need to support
- It shows how to register backend implementations
- It demonstrates operator variants (function, method, out, inplace)
- It reveals which operators are core vs decomposable

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│ native_functions.yaml (2,666+ operator definitions)          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - func: add.Tensor(Tensor self, Tensor other) -> Tensor│ │
│  │   variants: function, method                           │  │
│  │   dispatch:                                            │  │
│  │     CPU: add_cpu                                       │  │
│  │     CUDA: add_cuda                                     │  │
│  │     MPS: add_mps                                       │  │
│  │   tags: [core, pointwise]                              │  │
│  │   autogen: add.out                                     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ torchgen (Code Generator)                                    │
│  Reads YAML, generates:                                      │
│  ├─ C++ Declarations (Functions.h)                           │
│  ├─ C++ Dispatch Registration (RegisterCPU.cpp, ...)         │
│  ├─ Python Bindings (python_torch_functions.cpp)             │
│  ├─ Autograd Formulas (derivatives.yaml)                     │
│  └─ Type Stubs (torch/__init__.pyi)                          │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Generated Code + Handwritten Kernels                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ // Generated dispatch call                             │  │
│  │ Tensor add(const Tensor& self, const Tensor& other);  │  │
│  │                                                        │  │
│  │ // Handwritten kernel (aten/src/ATen/native/...)      │  │
│  │ Tensor add_cpu(const Tensor& self, const Tensor& other);│ │
│  │ Tensor add_cuda(const Tensor& self, const Tensor& other);│ │
│  │ Tensor add_mps(const Tensor& self, const Tensor& other); │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Runtime Dispatcher                                            │
│  call add(tensor_a, tensor_b):                               │
│    keyset = tensor_a.key_set() | tensor_b.key_set()          │
│    dispatch to highest priority key (e.g., CPU, CUDA, MPS)   │
└──────────────────────────────────────────────────────────────┘
```

## YAML Schema

### Basic Operator Definition

**File**: [reference/pytorch/aten/src/ATen/native/native_functions.yaml](reference/pytorch/aten/src/ATen/native/native_functions.yaml)

**Example**:

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
    MPS: add_mps
    CompositeExplicitAutograd: add_impl
  tags: [core, pointwise]
  autogen: add.out
```

**Fields**:

1. **func**: Function signature
   - Format: `name.overload(args) -> return_type`
   - Overload distinguishes variants (e.g., `add.Tensor` vs `add.Scalar`)
   - Arguments:
     - Regular: `Tensor self`
     - Keyword-only (after `*`): `*, Scalar alpha=1`
     - Default values: `alpha=1`
     - Mutability annotations: `Tensor(a!)` (modifies in-place)

2. **variants**: Which API styles to generate
   - `function`: `torch.add(a, b)` (namespace function)
   - `method`: `a.add(b)` (tensor method)
   - Default is `function, method` if omitted

3. **dispatch**: Backend implementations
   - Maps `DispatchKey` → kernel function name
   - Special keys:
     - `CompositeExplicitAutograd`: Generic implementation (works for all backends)
     - `CompositeImplicitAutograd`: Decomposes to other ops (autograd propagates)
     - Specific backends: `CPU`, `CUDA`, `MPS`, `XLA`, etc.

4. **tags**: Operator categories (see tags.yaml)
   - `core`: Part of core ATen opset (functional, no aliasing/mutation)
   - `pointwise`: Element-wise operation
   - `reduction`: Aggregates values (sum, mean, max)
   - `inplace_view`: Modifies only metadata (not data)
   - `nondeterministic_seeded`: Random but reproducible with seed
   - ... (see [reference/pytorch/aten/src/ATen/native/tags.yaml](reference/pytorch/aten/src/ATen/native/tags.yaml))

5. **autogen**: Auto-generate variants
   - `add.out`: Generate out-variant (writes to pre-allocated output)
   - Reduces manual YAML entries

### Operator Variants

PyTorch operators come in several flavors:

**1. Function Variant**:
```python
torch.add(a, b)                # Function call
```

**2. Method Variant**:
```python
a.add(b)                       # Tensor method
```

**3. Out Variant** (writes to pre-allocated output):
```yaml
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
```
```python
torch.add(a, b, out=result)    # Reuses 'result' tensor
```

**4. Inplace Variant** (modifies first argument):
```yaml
- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
```
```python
a.add_(b)                      # Modifies 'a' in-place (note underscore)
```

**Complete Example** - All variants of `add`:

```yaml
# Regular
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
    MPS: add_mps
  tags: [core, pointwise]

# Inplace
- func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
  variants: method
  dispatch:
    CPU: add__cpu
    CUDA: add__cuda
    MPS: add__mps
  tags: [inplace_view]

# Out-variant
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU: add_out_cpu
    CUDA: add_out_cuda
    MPS: add_out_mps

# Scalar variant
- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: add_scalar_impl
```

### Type Annotations

**Mutability Annotations**:

```yaml
Tensor self            # Read-only input
Tensor(a!) self        # Mutable (modified in-place)
Tensor(a!) out         # Output argument
Tensor(a) self         # View (shares storage, may alias)
```

**Example**:
```yaml
# Inplace op: modifies 'self'
- func: relu_(Tensor(a!) self) -> Tensor(a!)

# Out-variant: writes to 'out'
- func: relu.out(Tensor self, Tensor(a!) out) -> Tensor(a!)

# View: returns alias of 'self'
- func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
```

### Dispatch Keys

**File**: [reference/pytorch/c10/core/DispatchKey.h](reference/pytorch/c10/core/DispatchKey.h)

Common dispatch keys used in YAML:

```yaml
# Backend-specific implementations
CPU: kernel_cpu                  # CPU implementation
CUDA: kernel_cuda                # NVIDIA CUDA
MPS: kernel_mps                  # Apple Metal Performance Shaders
XLA: kernel_xla                  # Google TPU
Meta: kernel_meta                # Shape inference (no data)

# Composite implementations (generic, works across backends)
CompositeExplicitAutograd: kernel_impl
  # Generic implementation, autograd must be defined separately

CompositeImplicitAutograd: kernel_impl
  # Decomposed into other ops, autograd automatic

# Specialized
SparseCPU: kernel_sparse_cpu     # Sparse tensors on CPU
SparseCUDA: kernel_sparse_cuda   # Sparse tensors on CUDA
QuantizedCPU: kernel_qcpu        # Quantized tensors
```

**Dispatch Priority** (from highest to lowest):
1. Autograd keys (AutogradCPU, AutogradCUDA, AutogradMPS, ...)
2. Backend keys (CPU, CUDA, MPS, ...)
3. Composite fallback

**Example with multiple dispatch entries**:

```yaml
- func: conv2d(Tensor input, Tensor weight, ...) -> Tensor
  dispatch:
    CPU: conv2d_cpu              # Optimized CPU implementation
    CUDA: conv2d_cuda            # cuDNN-based CUDA
    MPS: conv2d_mps              # Metal compute pipeline
    QuantizedCPU: conv2d_qcpu    # Quantized convolution
```

### Tags

**File**: [reference/pytorch/aten/src/ATen/native/tags.yaml:0-99](reference/pytorch/aten/src/ATen/native/tags.yaml)

Tags categorize operators and affect code generation:

**Core Tags**:

```yaml
tags: [core]
  # Core ATen operator
  # - Functional (no in-place or aliasing)
  # - Part of export opset
  # - Survives decomposition passes

tags: [pointwise]
  # Element-wise operation
  # - Output[i] depends only on Input[i]
  # - Supports broadcasting

tags: [reduction]
  # Reduction operation
  # - Aggregates along dimensions (sum, mean, max, min)

tags: [inplace_view]
  # Modifies only tensor metadata (not data)
  # - Example: transpose_, view_

tags: [view_copy]
  # Copy variant of a view operation
  # - Example: transpose_copy is copy version of transpose
```

**Special Behavior Tags**:

```yaml
tags: [nondeterministic_seeded]
  # Random but reproducible
  # - Example: torch.randn (uses Generator)

tags: [nondeterministic_bitwise]
  # Non-deterministic (no bitwise reproducibility)
  # - Example: parallel reductions

tags: [dynamic_output_shape]
  # Output shape depends on input data
  # - Example: torch.nonzero, torch.unique

tags: [data_dependent_output]
  # Non-tensor output depends on tensor data
  # - Example: tensor.item() → Python scalar
```

**Compilation Tags**:

```yaml
tags: [pt2_compliant_tag]
  # Works with torch.compile

tags: [needs_contiguous_strides]
  # Requires contiguous input

tags: [flexible_layout]
  # Can handle arbitrary strides
```

### Auto-Generation

The `autogen` field automatically generates additional variants:

```yaml
- func: add.Tensor(Tensor self, Tensor other) -> Tensor
  autogen: add.out
  # Generates:
  # - func: add.out(Tensor self, Tensor other, Tensor(a!) out) -> Tensor(a!)
```

## Code Generation (torchgen)

**Location**: [reference/pytorch/torchgen/](reference/pytorch/torchgen/)

**Purpose**: Generate boilerplate code from YAML definitions.

**Generated Artifacts**:

```
torchgen reads native_functions.yaml
    │
    ├─► Functions.h                    (Function declarations)
    ├─► TensorBody.h                   (Tensor method declarations)
    ├─► TensorMethods.h                (Tensor method implementations)
    ├─► RegisterCPU.cpp                (CPU kernel registration)
    ├─► RegisterCUDA.cpp               (CUDA kernel registration)
    ├─► RegisterMPS.cpp                (MPS kernel registration)
    ├─► python_torch_functions.cpp     (Python bindings)
    ├─► python_variable_methods.cpp    (Python tensor methods)
    ├─► torch/__init__.pyi             (Type stubs for IDE)
    └─► VariableType.cpp               (Autograd wrapper layer)
```

**Example Generated Code**:

From YAML:
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
```

Generates (conceptual):

```cpp
// Functions.h
namespace at {
  TORCH_API Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha = 1);
}

// TensorBody.h
class Tensor {
 public:
  Tensor add(const Tensor& other, const Scalar& alpha = 1) const;
};

// RegisterCPU.cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(add_cpu));
}

// python_torch_functions.cpp
static PyObject* THPVariable_add(PyObject* self, PyObject* args, PyObject* kwargs) {
  // Parse Python arguments
  // Call at::add
  // Return Python tensor
}
```

## Operator Implementation

**Handwritten Kernels**: [reference/pytorch/aten/src/ATen/native/](reference/pytorch/aten/src/ATen/native/)

**Organization**:

```
aten/src/ATen/native/
├── BinaryOps.cpp              # add, sub, mul, div (generic)
├── cpu/                       # CPU-specific kernels
│   ├── BinaryOpsKernel.cpp    # Vectorized CPU kernels
│   └── ...
├── cuda/                      # CUDA kernels
│   ├── BinaryOpsKernel.cu     # CUDA device code
│   └── ...
├── mps/                       # Metal Performance Shaders
│   ├── operations/
│   │   ├── BinaryOps.mm       # Metal shader invocations
│   │   └── ...
│   └── ...
└── ...
```

**Example CPU Kernel**:

```cpp
// aten/src/ATen/native/BinaryOps.cpp
namespace at::native {

Tensor add_cpu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // Allocate output
  Tensor result = at::empty_like(self);

  // Call vectorized kernel
  add_stub(kCPU, result, self, other, alpha);

  return result;
}

// Kernel stub (defined in cpu/BinaryOpsKernel.cpp)
// Uses CPU vectorization (AVX2, AVX512, NEON, etc.)

} // namespace at::native
```

**Example CUDA Kernel**:

```cpp
// aten/src/ATen/native/cuda/BinaryOpsKernel.cu
namespace at::native {

__global__ void add_kernel(float* out, const float* a, const float* b, float alpha, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + alpha * b[idx];
  }
}

Tensor add_cuda(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor result = at::empty_like(self);

  // Launch CUDA kernel
  int threads = 256;
  int blocks = (self.numel() + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(
      result.data_ptr<float>(),
      self.data_ptr<float>(),
      other.data_ptr<float>(),
      alpha.to<float>(),
      self.numel());

  return result;
}

} // namespace at::native
```

## Critical File References

**YAML Definitions**:
- [aten/src/ATen/native/native_functions.yaml](reference/pytorch/aten/src/ATen/native/native_functions.yaml) - All 2,666 operator definitions (16,119 lines)
- [aten/src/ATen/native/tags.yaml:0-99](reference/pytorch/aten/src/ATen/native/tags.yaml) - Tag definitions and descriptions

**Code Generation**:
- [torchgen/](reference/pytorch/torchgen/) - Code generator implementation
- [torchgen/model.py](reference/pytorch/torchgen/model.py) - YAML schema parsing
- [torchgen/gen.py](reference/pytorch/torchgen/gen.py) - Code generation entry point

**Generated Code** (build artifacts):
- `build/aten/src/ATen/Functions.h` - Function declarations
- `build/aten/src/ATen/RegisterCPU.cpp` - CPU registration
- `build/aten/src/ATen/RegisterCUDA.cpp` - CUDA registration
- `build/aten/src/ATen/RegisterMPS.cpp` - MPS registration

**Kernel Implementations**:
- [aten/src/ATen/native/BinaryOps.cpp](reference/pytorch/aten/src/ATen/native/BinaryOps.cpp) - Generic binary ops
- [aten/src/ATen/native/cpu/](reference/pytorch/aten/src/ATen/native/cpu/) - CPU kernels
- [aten/src/ATen/native/cuda/](reference/pytorch/aten/src/ATen/native/cuda/) - CUDA kernels
- [aten/src/ATen/native/mps/](reference/pytorch/aten/src/ATen/native/mps/) - Metal kernels

**Dispatch Infrastructure**:
- [aten/src/ATen/core/dispatch/Dispatcher.h](reference/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h) - Runtime dispatcher
- [torch/library.h](reference/pytorch/torch/library.h) - TORCH_LIBRARY macros

## MLX Porting Considerations

### What MLX Already Provides

MLX has operators defined in C++ with Python bindings:
- Direct C++ implementation (no YAML intermediate)
- Operators optimized for Metal
- Lazy evaluation graph

### What Needs Adaptation

1. **YAML vs Direct Implementation**:
   - PyTorch: YAML → codegen → C++
   - MLX: Direct C++ implementation
   - **Decision**: For PyTorch compatibility, could adopt YAML approach
   - **Alternative**: Maintain manual implementation but create mapping table

2. **Operator Variants**:
   - PyTorch: function, method, out, inplace
   - MLX: Typically just function variant
   - **Action**: Decide which variants MLX needs for compatibility

3. **Dispatch Complexity**:
   - PyTorch: Complex multi-backend dispatch
   - MLX: Metal-only, simpler dispatch
   - **Advantage**: MLX can skip dispatch overhead

4. **Operator Count**:
   - PyTorch: 2,666 operators (many specialized)
   - MLX: Smaller core set
   - **Strategy**: Tier operators (core → common → specialized)

### Recommendations for MLX

1. **Operator Priority (Tiered Approach)**:
   - **Tier 1** (~50 ops): Essential for basic ML (see operator-reference/arithmetic.md, etc.)
   - **Tier 2** (~200 ops): Common but can wait
   - **Tier 3** (~2,400 ops): Implement on-demand

2. **YAML Adoption** (Optional):
   - Benefits: Consistency with PyTorch, easier maintenance
   - Cost: Additional build complexity
   - **Recommendation**: Only if targeting full PyTorch compatibility

3. **Variant Support**:
   - Implement at least function variant
   - Add method variant for convenience (`array.add(other)` vs `mlx.add(a, b)`)
   - Skip out-variant unless profiling shows allocation overhead
   - Add inplace variants sparingly (MLX's lazy eval may make them less critical)

4. **Tag System**:
   - Adopt `core`, `pointwise`, `reduction` tags minimally
   - Use for optimization hints (e.g., pointwise → elementwise Metal kernel)

## Next Steps

1. Study **operator-categories.md** to see operators grouped by function
2. Read **operator-reference/arithmetic.md** for detailed Tier 1 operator documentation
3. Examine **operator-implementation.md** for implementation patterns
4. For MLX porting, create operator priority list based on target workloads (LLMs, vision, etc.)
