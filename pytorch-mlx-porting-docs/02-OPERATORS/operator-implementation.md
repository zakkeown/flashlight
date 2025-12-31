# Operator Implementation

## Purpose

This document explains how to implement PyTorch operators, covering the complete journey from YAML definition to backend kernel execution. Understanding this implementation pipeline is critical for porting operators to MLX, as it reveals:
- Code generation patterns that must be replicated
- Backend dispatch mechanisms
- Kernel implementation strategies (CPU, CUDA, Metal/MPS)
- Meta function patterns for shape inference

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                 Operator Implementation Pipeline              │
└──────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐      ┌────────────┐     ┌──────────┐
   │  YAML   │      │  torchgen  │     │Generated │
   │Definition│ ───> │ CodeGen    │ ──> │  C++/Py  │
   └─────────┘      └────────────┘     └──────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ Dispatcher   │
                                      │ Registration │
                                      └──────────────┘
                                              │
                 ┌────────────────────────────┼────────────────┐
                 ▼                            ▼                ▼
           ┌──────────┐              ┌──────────┐      ┌──────────┐
           │Meta Func │              │CPU Kernel│      │MPS Kernel│
           │(Shape)   │              │          │      │ (Metal)  │
           └──────────┘              └──────────┘      └──────────┘
```

## Implementation Workflow

### Step 1: YAML Definition

Every operator starts with a definition in `native_functions.yaml`.

**File**: `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml`

**Example**: `add.Tensor` operator

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator handles device checking
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: add_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [core, pointwise]
```

**Key Fields Explained**:

- **`func`**: Signature with name, parameters, return type
- **`device_check`**: When to check device compatibility
- **`structured_delegate`**: Points to out variant for structured kernels
- **`variants`**: Where operator appears (function, method, both)
- **`dispatch`**: Backend-specific implementations
- **`tags`**: Categorization (core, pointwise, reduction, etc.)

### Step 2: Code Generation (torchgen)

**torchgen** parses YAML and generates C++/Python code.

**Directory**: `/Users/zakkeown/Code/flashlight/reference/pytorch/torchgen/`

**Generated Files**:

1. **C++ Declarations** (headers):
   - `build/aten/src/ATen/ops/add.h` - Public C++ API
   - `build/aten/src/ATen/ops/add_native.h` - Native function declarations
   - `build/aten/src/ATen/ops/add_ops.h` - Dispatcher registration

2. **C++ Implementations**:
   - `build/aten/src/ATen/RegisterDispatchKey.cpp` - Dispatcher wiring
   - `build/aten/src/ATen/RegisterSchema.cpp` - Schema registration

3. **Python Bindings**:
   - `build/torch/csrc/autograd/generated/python_torch_functions.cpp`
   - `build/torch/csrc/autograd/generated/VariableType.cpp` - Autograd wrapper

**Example Generated Code for `add.Tensor`**:

```cpp
// Generated in build/aten/src/ATen/ops/add.h
namespace at {

inline Tensor add(
    const Tensor & self,
    const Tensor & other,
    const Scalar & alpha=1) {
  return at::_ops::add_Tensor::call(self, other, alpha);
}

} // namespace at

// Generated in build/aten/src/ATen/ops/add_native.h
namespace at {
namespace native {

TORCH_API Tensor add(
    const Tensor & self,
    const Tensor & other,
    const Scalar & alpha);

} // namespace native
} // namespace at

// Generated dispatcher registration
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(wrapper_add_Tensor));
}
```

### Step 3: Operator Registration

Operators register with the dispatcher using `TORCH_LIBRARY` macros.

**Registration Pattern**:

```cpp
// Register schema
TORCH_LIBRARY(aten, m) {
  m.def("add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
}

// Register backend implementation
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(add_cpu));
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("add.Tensor", TORCH_FN(add_cuda));
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("add.Tensor", TORCH_FN(add_mps));
}
```

**How It Works**:
1. Schema registered with dispatcher
2. Each backend registers its kernel function
3. At runtime, dispatcher selects based on tensor's DispatchKey
4. Calls appropriate kernel (CPU, CUDA, MPS, etc.)

### Step 4: Meta Functions (Shape Inference)

Meta functions compute output shapes and dtypes without touching data.

**File**: `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/BinaryOps.cpp:149-156`

```cpp
namespace at::meta {

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

} // namespace at::meta
```

**Meta Function Responsibilities**:
- Compute output shape via broadcasting
- Determine output dtype via type promotion
- Validate inputs (dimension checks, dtype compatibility)
- **Does NOT access tensor data** (works with meta tensors)

**Common Meta Function Patterns**:

```cpp
// Pointwise binary op (broadcasting)
build_borrowing_binary_op(maybe_get_output(), self, other);

// Pointwise binary op with float promotion
build_borrowing_binary_float_op(maybe_get_output(), self, other);

// Comparison ops (returns bool)
build_borrowing_comparison_op(result, self, other);

// Reduction ops
resize_output(output, reduce_shape(input.sizes(), dims, keepdim));
```

### Step 5: Backend Kernel Implementation

Kernels perform the actual computation for specific backends.

#### CPU Kernel Pattern

**File**: `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:126-170`

```cpp
namespace at::native {

void mul_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (dtype == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else if (dtype == kComplexHalf) {
    cpu_kernel(
        iter,
        [=](c10::complex<at::Half> a, c10::complex<at::Half> b)
            -> c10::complex<at::Half> {
          using comp_t = c10::complex<float>;
          return comp_t{a} * comp_t{b};
        });
  } else {
    // Vectorized implementation for most types
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "mul_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          // Scalar lambda
          [=](scalar_t a, scalar_t b) -> scalar_t {
            return a * b;
          },
          // Vectorized lambda (SIMD)
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a * b;
          });
    });
  }
}

// Register with dispatcher
REGISTER_DISPATCH(mul_stub, &mul_kernel);

} // namespace at::native
```

**CPU Kernel Key Concepts**:

1. **TensorIterator**: Handles broadcasting, type promotion, parallelization
   - Provides efficient iteration over potentially broadcasted tensors
   - Automatically parallelizes across threads
   - Handles contiguous vs strided memory

2. **AT_DISPATCH_***: Type dispatching macros
   - `AT_DISPATCH_ALL_TYPES`: All numeric types
   - `AT_DISPATCH_FLOATING_TYPES`: Float types only
   - `AT_DISPATCH_INTEGRAL_TYPES`: Integer types only
   - Generates template code for each type

3. **Vectorization**: SIMD operations
   - `Vectorized<T>`: SIMD vector type
   - `cpu_kernel_vec`: Accepts both scalar and vectorized lambdas
   - Automatically uses SIMD when possible, falls back to scalar

4. **cpu_kernel vs cpu_kernel_vec**:
   ```cpp
   // Scalar only (no SIMD)
   cpu_kernel(iter, [](scalar_t a, scalar_t b) { return a + b; });

   // Scalar + vectorized
   cpu_kernel_vec(
       iter,
       [](scalar_t a, scalar_t b) { return a + b; },           // Scalar path
       [](Vectorized<T> a, Vectorized<T> b) { return a + b; }  // SIMD path
   );
   ```

#### MPS (Metal) Kernel Pattern

**File**: `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/BinaryOps.mm:48-150`

```cpp
namespace at::native {
namespace mps {

struct BinaryOpCachedGraph : public MPSCachedGraph {
  BinaryOpCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *primaryTensor = nil, *secondaryTensor = nil;
  MPSGraphTensor* outputTensor = nil;
};

typedef MPSGraphTensor* (^BinaryOpBlock)(BinaryOpCachedGraph*, MPSGraphTensor*, MPSGraphTensor*);

static void binaryOpTensor(const Tensor& self,
                           const Tensor& other,
                           const Tensor& output_,
                           std::string op_name,
                           BinaryOpBlock binaryBlock) {
  MPSStream* mpsStream = getCurrentMPSStream();

  // Infer output shape
  auto new_size = at::infer_size(self.sizes(), other.sizes());
  if (!output_.sizes().equals(new_size)) {
    output_.resize_(new_size);
  }

  // Type promotion
  auto inputDataType = self.scalar_type();
  auto otherDataType = other.scalar_type();
  auto outputDataType = output_.scalar_type();
  auto common_dtype = c10::promoteTypes(inputDataType, otherDataType);

  @autoreleasepool {
    // Graph caching: reuse graphs for same operation+shapes
    std::string key = op_name + getTensorsStringKey({self, other, output_});
    auto cachedGraph = LookUpOrCreateCachedGraph<BinaryOpCachedGraph>(key,
      [&](auto mpsGraph, auto newCachedGraph) {
        // Create placeholders for inputs
        newCachedGraph->primaryTensor =
            mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(inputDataType), getMPSShape(self));
        newCachedGraph->secondaryTensor =
            mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(otherDataType), getMPSShape(other));

        // Cast to common dtype if needed
        MPSGraphTensor* primaryCastTensor = newCachedGraph->primaryTensor;
        MPSGraphTensor* secondaryCastTensor = newCachedGraph->secondaryTensor;
        if (inputDataType != common_dtype) {
          primaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->primaryTensor, common_dtype);
        }
        if (otherDataType != common_dtype) {
          secondaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->secondaryTensor, common_dtype);
        }

        // Apply the operation (provided by caller)
        newCachedGraph->outputTensor = binaryBlock(newCachedGraph, primaryCastTensor, secondaryCastTensor);

        // Cast output if needed
        if (outputDataType != common_dtype) {
          newCachedGraph->outputTensor = castMPSTensor(mpsGraph, newCachedGraph->outputTensor, outputDataType);
        }
      });

    // Prepare input feeds
    NSMutableDictionary* feeds = [[NSMutableDictionary new] autorelease];
    // ... populate feeds with actual tensor data ...

    // Run graph
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output_);
    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

// Specific operator implementations use this helper
Tensor& add_out_mps(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& output) {
  binaryOpTensor(self, other, output, "add_out_mps",
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
      MPSGraph* mpsGraph = cachedGraph->graph();
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                           shape:@[@1]
                                                        dataType:primaryCastTensor.dataType];
      MPSGraphTensor* scaledSecondary = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                                                  secondaryTensor:alphaTensor
                                                                             name:nil];
      return [mpsGraph additionWithPrimaryTensor:primaryCastTensor
                                 secondaryTensor:scaledSecondary
                                            name:nil];
    });
  return output;
}

} // namespace mps
} // namespace at::native
```

**MPS Kernel Key Concepts**:

1. **MPSGraph**: Metal Performance Shaders computational graph
   - Build graph once, cache it
   - Reuse for tensors with same shapes/types
   - Graph is JIT-compiled to Metal shaders

2. **Graph Caching**: Critical performance optimization
   ```cpp
   std::string key = op_name + getTensorsStringKey({self, other, output_});
   auto cachedGraph = LookUpOrCreateCachedGraph<T>(key, [&](...) {
       // Build graph if not cached
   });
   ```

3. **Placeholders**: Graph inputs
   - Created during graph construction
   - Actual data bound at execution time
   - Shape must match between graph and runtime

4. **Type Promotion**: Explicit casting
   - MPSGraph requires explicit cast operations
   - Insert cast nodes before operations
   - Cast output back to expected type

5. **Blocks (^)**: Objective-C++ lambda syntax
   ```objective-c
   ^ReturnType(Args) { body }
   ```

**MLX Relevance**: MLX uses similar lazy graph construction, so this pattern translates well.

#### CUDA Kernel Pattern

**File**: `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cuda/BinaryMulDivKernel.cu`

```cpp
namespace at::native {

void mul_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (dtype == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "mul_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "mul_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * b;
      });
    });
  }
}

REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);

} // namespace at::native
```

**CUDA Kernel Key Concepts**:

1. **gpu_kernel**: TensorIterator-based GPU kernel launcher
   - Similar to cpu_kernel, but for GPU
   - Handles grid/block dimension calculation
   - Manages memory coalescing

2. **GPU_LAMBDA**: Device lambda marker
   - Marks lambda for device execution
   - `__device__` qualifier in CUDA

3. **TensorIterator on GPU**: Same abstraction as CPU
   - Broadcasting handled automatically
   - Strided access patterns optimized
   - Parallel across CUDA threads

**MLX Note**: While MLX doesn't use CUDA, the algorithmic patterns (parallelization strategies, reduction techniques) are transferable to Metal.

## Structured Kernels

Many operators use **structured kernels** to reduce code duplication.

### Concept

Instead of implementing `op()`, `op_()`, and `op.out()` separately, implement once:

```cpp
TORCH_META_FUNC(add) (...) {
  // Shape inference
}

TORCH_IMPL_FUNC(add_out_cpu) (...) {
  // CPU computation
}

TORCH_IMPL_FUNC(add_out_cuda) (...) {
  // CUDA computation
}
```

All three variants (`add`, `add_`, `add.out`) automatically generated.

### Example: `add` Structured Kernel

**YAML Declaration**:
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  structured_delegate: add.out
  # ...

- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  # ...
```

**Implementation**:
```cpp
// Meta function (shape inference) - shared by all variants
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

// CPU kernel - called by add(), add_(), add.out()
TORCH_IMPL_FUNC(add_out_cpu) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(Device::CPU, *this, alpha);
}

// CUDA kernel
TORCH_IMPL_FUNC(add_out_cuda) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(Device::CUDA, *this, alpha);
}
```

**Benefits**:
- Write shape logic once
- Backend kernels focus only on computation
- Consistency across variants
- Less boilerplate

## TensorIterator: The Workhorse

`TensorIterator` is PyTorch's abstraction for element-wise and reduction operations.

**File**: `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/TensorIterator.h`

**Purpose**:
- Handles broadcasting
- Manages type promotion
- Provides efficient iteration
- Parallelizes automatically (CPU threads, GPU blocks)
- Handles both contiguous and strided tensors

**Usage Pattern**:

```cpp
// Build iterator
auto iter = TensorIteratorConfig()
    .add_output(output)
    .add_input(input1)
    .add_input(input2)
    .build();

// Dispatch to kernel
cpu_kernel_vec(
    iter,
    [](scalar_t a, scalar_t b) { return a + b; },      // Scalar
    [](Vectorized<T> a, Vectorized<T> b) { return a + b; }  // Vector
);
```

**What TensorIterator Does**:
1. **Broadcasting**: Expands dimensions to compatible shape
2. **Type Promotion**: Computes common dtype
3. **Memory Layout**: Determines iteration strategy (contiguous, strided, etc.)
4. **Parallelization**: Splits work across threads/blocks
5. **Striding**: Computes strides for efficient memory access

**MLX Equivalent**: MLX has similar broadcasting and iteration logic, but more implicit.

## Dispatch Mechanism

### DispatchKey Resolution

When you call `torch.add(tensor_a, tensor_b)`:

1. Dispatcher examines `tensor_a` and `tensor_b` DispatchKeySets
2. Computes highest-priority key (e.g., `Autograd` + `CPU`)
3. Looks up kernel registered for that key
4. Calls appropriate implementation

**Priority Order** (high to low):
```
Autograd > Profiling > Tracing > BackendComponent (CPU, CUDA, MPS, etc.)
```

**Example**:
```cpp
Tensor a = torch::randn({2, 3});  // DispatchKeySet includes CPU, AutogradCPU
Tensor b = torch::randn({2, 3});

// Dispatcher resolution:
// 1. Check DispatchKeySet: {AutogradCPU, CPU}
// 2. Highest priority: AutogradCPU
// 3. Call Autograd wrapper (records in computation graph)
// 4. Autograd wrapper calls CPU implementation
// 5. CPU kernel executes
```

### Fallback Dispatch

Some dispatch keys provide fallback implementations:

```yaml
dispatch:
  CompositeImplicitAutograd: reshape_symint  # Fallback for all backends
  CPU: reshape_cpu                           # CPU-specific override
```

**CompositeImplicitAutograd**: Operator implementation that works for any backend, using only other operators (no direct backend calls).

## Operator Implementation Checklist

To implement a new operator in PyTorch (or port to MLX):

1. **Define in YAML**:
   - [ ] Write function signature
   - [ ] Specify variants (function, method, out, inplace)
   - [ ] List backend dispatch mappings
   - [ ] Add appropriate tags

2. **Implement Meta Function**:
   - [ ] Compute output shape (broadcasting)
   - [ ] Determine output dtype (type promotion)
   - [ ] Validate inputs
   - [ ] Use `build_borrowing_*` helpers

3. **Implement Backend Kernels**:
   - [ ] CPU kernel using TensorIterator
   - [ ] (Optional) CUDA kernel
   - [ ] (Optional) MPS kernel
   - [ ] Register with `REGISTER_DISPATCH` or `TORCH_LIBRARY_IMPL`

4. **Test**:
   - [ ] Shape inference (meta tensors)
   - [ ] Type promotion
   - [ ] Broadcasting
   - [ ] Backward pass (if differentiable)
   - [ ] Backend-specific correctness

5. **Autograd** (if differentiable):
   - [ ] Define backward in derivatives.yaml
   - [ ] Implement gradient formulas

## Common Implementation Patterns

### Pointwise Binary Operation

```cpp
// Meta function
TORCH_META_FUNC2(my_op, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

// CPU kernel
void my_op_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "my_op_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) { return my_operation(a, b); },
        [](Vectorized<T> a, Vectorized<T> b) { return my_operation_vec(a, b); }
    );
  });
}
```

### Reduction Operation

```cpp
// Meta function
TORCH_META_FUNC(my_reduce) (
  const Tensor& self, IntArrayRef dim, bool keepdim
) {
  auto shape = get_reduction_shape(self.sizes(), dim, keepdim);
  set_output_raw_strided(0, shape, {}, self.options());
}

// CPU kernel
void my_reduce_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "my_reduce_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t acc, scalar_t value) {
      return combine(acc, value);  // e.g., acc + value for sum
    });
  });
}
```

### View Operation (Metadata Only)

```cpp
Tensor my_view(const Tensor& self, IntArrayRef size) {
  auto new_size = infer_size(size, self.numel());
  return self.as_strided(new_size, compute_strides(new_size));
}
```

**No kernel needed**: Only manipulates TensorImpl metadata.

## MLX Porting Considerations

### Similarities to MLX

1. **Lazy Evaluation**: MLX uses lazy graphs like MPS
   - Graph construction pattern directly applicable
   - Caching strategies transferable

2. **Unified Memory**: MLX assumes unified memory
   - Simplifies some MPS patterns
   - No explicit host-device transfers

3. **Metal Backend**: MPS reference implementations are gold standard
   - Direct algorithmic reference
   - Metal shader patterns reusable

### Differences from PyTorch

1. **No TensorIterator**: MLX doesn't have equivalent abstraction
   - Broadcasting handled differently
   - Iteration patterns more explicit

2. **Simpler Type System**: MLX has fewer dtypes
   - Less dispatching needed
   - Type promotion simpler

3. **Functional Core**: MLX emphasizes immutability
   - No inplace operations at low level
   - Simplifies some implementation patterns

4. **Python-First API**: MLX operators often implemented in Python
   - Less C++ boilerplate
   - More accessible for rapid development

### Porting Strategy

**For each PyTorch operator**:

1. **Study MPS implementation first** (if available)
   - `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/`
   - Metal shader code directly transferable

2. **Understand broadcasting semantics**
   - PyTorch's meta functions show shape logic
   - Replicate in MLX

3. **Map to MLX primitives**
   - MLX has lower-level ops (e.g., `mlx.core.add`)
   - Compose PyTorch ops from MLX primitives

4. **Implement backward pass**
   - PyTorch's `derivatives.yaml` shows gradients
   - Use `mlx.grad` or manual Jacobian

5. **Test compatibility**
   - Compare PyTorch and MLX outputs
   - Test broadcasting, type promotion, gradients

### Reference Priority

When porting operator `foo`:

1. **Check MPS**: `aten/src/ATen/native/mps/operations/{category}.mm`
2. **Check CPU**: `aten/src/ATen/native/cpu/{category}Kernel.cpp`
3. **Check YAML**: Understand variants and tags
4. **Check Meta**: `aten/src/ATen/native/{category}.cpp` (meta functions)
5. **Check Derivatives**: `tools/autograd/derivatives.yaml` (gradients)

## Critical Files Reference

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/tags.yaml`

**Code Generation**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torchgen/` (entire directory)

**Meta Functions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/BinaryOps.cpp:149-400` (binary ops)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/ReduceOps.cpp` (reductions)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/TensorShape.cpp` (shape ops)

**CPU Kernels**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp`

**MPS Kernels** (Metal - most relevant for MLX):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/BinaryOps.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/ReduceOps.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/PointwiseOps.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Activation.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Convolution.mm`

**CUDA Kernels** (algorithm reference):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cuda/BinaryMulDivKernel.cu`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cuda/Reduce.cuh`

**TensorIterator**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/TensorIterator.h`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/TensorIterator.cpp`

**Dispatcher**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/c10/core/DispatchKey.h`

## Summary

PyTorch operator implementation follows a systematic pipeline:

1. **YAML Definition** → Declares signature, variants, dispatch
2. **Code Generation** → torchgen creates C++/Python bindings
3. **Registration** → TORCH_LIBRARY macros wire to dispatcher
4. **Meta Functions** → Shape inference, dtype promotion (no data)
5. **Backend Kernels** → CPU, CUDA, MPS implementations

**Key Abstractions**:
- **TensorIterator**: Broadcasting, parallelization, type promotion
- **Structured Kernels**: Share meta function across variants
- **Dispatcher**: Runtime backend selection

**For MLX Porting**:
- Study **MPS implementations** first (Metal-based, most similar)
- Understand **meta functions** for shape/type logic
- Map PyTorch ops to **MLX primitives**
- Reference **derivatives.yaml** for backward passes

**Pattern Library**:
- Pointwise ops: TensorIterator + `cpu_kernel_vec`
- Reductions: TensorIterator with accumulation
- Views: Metadata manipulation only
- MPS ops: Cached MPSGraph with placeholders

This implementation knowledge enables systematic porting of PyTorch's 2,666 operators to MLX.
