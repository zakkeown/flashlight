# TensorIterator Framework

## Purpose

The TensorIterator is PyTorch's **most critical infrastructure** for implementing element-wise and reduction operations efficiently. It's a sophisticated multi-tensor iteration engine that handles broadcasting, type promotion, memory layout optimization, and vectorization—all transparently to kernel authors. Understanding TensorIterator is essential for porting PyTorch operators to MLX, as nearly every element-wise operation (arithmetic, comparisons, activations) and many reductions use this framework.

**Key Insight**: TensorIterator allows you to write a simple scalar lambda like `[](float a, float b) { return a + b; }` and automatically handles all the complexity of:
- Broadcasting different-shaped tensors
- Type promotion and conversion
- Stride reordering for cache efficiency
- CPU vectorization (AVX2/AVX512)
- Memory overlap detection
- 32-bit vs 64-bit indexing
- Parallel execution

---

## Architecture Overview

### High-Level Design

TensorIterator is inspired by NumPy's Array Iterator API (`NpyIter`) and provides a unified interface for multi-tensor operations:

```
┌────────────────────────────────────────────────────────────┐
│                   TensorIteratorConfig                     │
│              (Builder Pattern for Setup)                   │
│                                                            │
│  .add_output(tensor)                                       │
│  .add_input(tensor)                                        │
│  .promote_inputs_to_common_dtype()                         │
│  .resize_outputs(false)  // for in-place ops               │
│  .build() ──────────────────────────┐                      │
└─────────────────────────────────────┼──────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────┐
│                    TensorIteratorBase                      │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Shape Inference & Broadcasting                      │ │
│  │  - Compute common shape from all operands            │ │
│  │  - Broadcast each operand to common shape            │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Type Promotion                                      │ │
│  │  - Determine common dtype (if requested)             │ │
│  │  - Insert type conversions as needed                 │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Stride Reordering                                   │ │
│  │  - Reorder dimensions by stride (largest first)      │ │
│  │  - Optimize for cache locality                       │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Dimension Coalescing                                │ │
│  │  - Merge adjacent dimensions with compatible strides │ │
│  │  - Reduce iteration overhead                         │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Iteration State                                     │ │
│  │  - shape_: DimVector (common shape)                  │ │
│  │  - operands_: SmallVector<OperandInfo>               │ │
│  │  - num_outputs_: int                                 │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────┐
│                    Kernel Execution                        │
│                                                            │
│  iter.for_each([](float a, float b) {                     │
│    return a + b;                                           │
│  });                                                       │
│                                                            │
│  OR with explicit vectorization:                          │
│                                                            │
│  cpu_kernel_vec(iter,                                      │
│    [](float a, float b) { return a + b; },                 │
│    [](Vectorized<float> a, Vectorized<float> b) {          │
│      return a * b;                                         │
│    });                                                     │
└────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. TensorIteratorConfig (Builder)

**Location**: `aten/src/ATen/TensorIterator.h`

The configuration builder used to set up the iterator:

```cpp
auto iter = TensorIteratorConfig()
  .add_output(output)        // Outputs must be added first!
  .add_input(input1)
  .add_input(input2)
  .promote_inputs_to_common_dtype()  // Optional type promotion
  .check_all_same_dtype(false)       // Disable dtype check
  .resize_outputs(false)              // For in-place operations
  .build();
```

**Critical Ordering Rule**: Outputs **must** be added before inputs. Adding an output after inputs throws an exception.

**Common Configuration Options**:
- `promote_inputs_to_common_dtype()`: Apply type promotion rules
- `cast_common_dtype_to_outputs(true)`: Cast results to output dtype
- `enforce_safe_casting_to_output(true)`: Prevent precision loss
- `resize_outputs(false)`: Don't resize output (for in-place ops)
- `declare_static_dtype(dtype)`: Override dtype inference
- `declare_static_shape(shape)`: Override shape inference
- `check_all_same_dtype(false)`: Allow mixed types

#### 2. OperandInfo (Per-Tensor Metadata)

**Location**: `aten/src/ATen/TensorIterator.h:117-234`

Stores metadata for each tensor participating in the operation:

```cpp
struct OperandInfo {
  void* data;                       // Data pointer (may differ from tensor->data_ptr())
  StrideVector stride_bytes;        // Strides in BYTES, after broadcasting

  std::optional<Device> device;     // Target device
  ScalarType target_dtype;          // Desired dtype (after promotion)
  ScalarType current_dtype;         // Actual tensor dtype

  bool is_output;                   // True for outputs
  bool will_resize;                 // True if output will be resized
  bool is_read_write;               // True for in-place operations
  bool is_const;                    // True for const inputs

  const Tensor& tensor();           // The underlying tensor
  const Tensor& original_tensor();  // Original before type conversion
};
```

**Key Fields**:
- **`stride_bytes`**: Strides are stored in **bytes**, not elements. This is critical because different operands may have different element sizes during type conversion.
- **`will_resize`**: Indicates three operation modes:
  1. **Functional** (`torch.add(a, b)`): Output is undefined, will be allocated
  2. **In-place** (`torch.add_(a, b)`): Output is same as input, can't resize
  3. **Out variant** (`torch.add(a, b, out=c)`): Output may need resizing if shape mismatches

#### 3. TensorIteratorBase (Core Iterator)

**Location**: `aten/src/ATen/TensorIterator.h:248-600`

The main iterator class providing:

```cpp
struct TensorIteratorBase {
  // Shape and dimension info
  int ndim() const;                  // Number of dimensions
  IntArrayRef shape() const;         // Common shape after broadcasting
  int64_t numel() const;             // Total number of elements

  // Operand access
  int noutputs() const;              // Number of output tensors
  int ninputs() const;               // Number of input tensors
  ScalarType dtype(int64_t arg);     // Dtype of operand
  Device device(int64_t arg);        // Device of operand

  // Iteration queries
  bool is_trivial_1d() const;        // 1D iteration, no buffering/conversion
  bool is_contiguous() const;        // All operands contiguous
  bool can_use_32bit_indexing();     // Can use int32 offsets (GPU optimization)

  // Main iteration methods
  void for_each(loop2d_t loop, int64_t grain_size = GRAIN_SIZE);
  void parallel_reduce(loop2d_t loop);
  void serial_for_each(loop2d_t loop, Range range);

  // Dimension manipulation
  void narrow(int dim, int64_t start, int64_t size);
  void remove_operand(int64_t arg);
  std::unique_ptr<TensorIterator> split(int dim);
};
```

**Important Constants**:
- `GRAIN_SIZE = 32768`: Minimum work size for parallelization

---

## Key Concepts

### Broadcasting Logic

TensorIterator implements NumPy-compatible broadcasting:

```cpp
// Example: (3, 1, 5) + (1, 4, 5) → (3, 4, 5)
auto iter = TensorIteratorConfig()
  .add_output(output)     // Shape will be (3, 4, 5)
  .add_input(a)           // Shape (3, 1, 5)
  .add_input(b)           // Shape (1, 4, 5)
  .build();
```

**Broadcasting Rules**:
1. Align shapes from the **right** (trailing dimensions)
2. Dimensions of size 1 are broadcast to match corresponding dimension
3. Missing dimensions are treated as size 1
4. All operands are virtually expanded to the common shape

**Implementation**:
- Common shape is computed from all inputs
- Each operand's strides are adjusted to reflect broadcasting
- Broadcast dimensions have **stride 0** (reuses same element)

### Type Promotion

TensorIterator can automatically promote operand types:

```cpp
// Example: int32 + float32 → float32
auto iter = TensorIteratorConfig()
  .add_output(output)                       // Will be float32
  .add_input(int_tensor)                    // int32
  .add_input(float_tensor)                  // float32
  .promote_inputs_to_common_dtype()         // Enable promotion
  .build();

ScalarType common = iter.common_dtype();    // Returns float32
```

**Promotion Rules**:
- **Category hierarchy**: `bool → int → float → complex`
- **Within category**: Promote to wider type (e.g., `int32 + int64 → int64`)
- **Unsigned handling**: Unsigned types promote to signed counterparts
- **Cross-category**: Always promote to higher category

**See Also**: `broadcasting-promotion.md` for detailed promotion matrix.

### Stride Reordering

TensorIterator reorders dimensions to optimize cache locality:

```cpp
// Original: shape=[100, 1000], strides=[1000, 1]  (row-major)
// After reorder: dims permuted so largest stride comes first
//
// Goal: Iterate in memory order for better cache performance
```

**Algorithm**:
1. Sort dimensions by stride (largest stride first)
2. Prioritize non-broadcast dimensions (stride > 0)
3. Keep broadcast dimensions (stride = 0) last

**Why It Matters**:
- CPU cache lines load contiguous bytes
- Iterating in stride order minimizes cache misses
- Can improve performance by 2-10x for large tensors

### Dimension Coalescing

Adjacent dimensions with compatible strides are merged:

```cpp
// Original: shape=[10, 20, 30], strides=[600, 30, 1]
// After coalescing: shape=[6000], strides=[1]
//
// This simplifies iteration from 3-nested loops to 1 loop
```

**Coalescing Conditions**:
- Dimensions must be adjacent (after reordering)
- Strides must be compatible: `stride[dim] == stride[dim+1] * shape[dim+1]`
- All operands must satisfy the condition

**Benefits**:
- Reduces loop overhead
- Enables better compiler optimization
- Simplifies vectorization

### Vectorization

TensorIterator provides two vectorization mechanisms:

#### Auto-Vectorization (Compiler-Dependent)

```cpp
cpu_kernel(iter, [](float a, float b) {
  return a + b;
});
```

The compiler may auto-vectorize the scalar lambda using SIMD instructions.

#### Explicit Vectorization

**Location**: `aten/src/ATen/native/cpu/Loops.h`

```cpp
cpu_kernel_vec(iter,
  [](float a, float b) -> float {  // Scalar fallback
    return a + b;
  },
  [](Vectorized<float> a, Vectorized<float> b) -> Vectorized<float> {  // SIMD path
    return a * b;
  }
);
```

**How It Works**:
1. Check if operands are contiguous and same type
2. If yes, use vectorized lambda with `Vectorized<T>` types (wraps `__m256` for AVX2)
3. Process `Vec::size()` elements per iteration (8 floats for AVX2)
4. Fall back to scalar lambda for remainder elements

**Vectorized Type**:
- `Vectorized<float>`: 256-bit vector (8 floats) on AVX2, 512-bit (16 floats) on AVX512
- Supports all standard operations: `+`, `*`, `min`, `max`, `sqrt`, etc.
- See `aten/src/ATen/cpu/vec/vec.h`

---

## Iteration Patterns

### Pattern 1: Basic Element-Wise Operation

**Use Case**: Binary operation like `add`, `mul`, `div`

```cpp
// PyTorch operator implementation
Tensor add_impl(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor output = at::empty_like(self);

  auto iter = TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .add_input(other)
    .build();

  cpu_kernel(iter, [alpha=alpha.to<float>()](float a, float b) -> float {
    return a + alpha * b;
  });

  return output;
}
```

**MLX Equivalent**:
```cpp
// MLX doesn't use TensorIterator - operations are lazy and compiled
array add(const array& a, const array& b) {
  return array(/* lazy graph node representing a + b */);
}
```

### Pattern 2: In-Place Operation

**Use Case**: In-place variant like `add_`

```cpp
Tensor& add_inplace_impl(Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto iter = TensorIteratorConfig()
    .add_output(self)              // Output is same as first input
    .add_input(self)               // Read from same tensor
    .add_input(other)
    .resize_outputs(false)         // Don't resize output!
    .build();

  cpu_kernel(iter, [alpha=alpha.to<float>()](float a, float b) -> float {
    return a + alpha * b;
  });

  return self;
}
```

**Critical**: Set `.resize_outputs(false)` to prevent resizing the output tensor.

### Pattern 3: Type Promotion

**Use Case**: Operations that promote mixed types

```cpp
Tensor mul_impl(const Tensor& self, const Tensor& other) {
  auto iter = TensorIteratorConfig()
    .add_output(Tensor())            // Output will be allocated with common dtype
    .add_input(self)                 // Could be int32
    .add_input(other)                // Could be float32
    .promote_inputs_to_common_dtype()  // Promote to float32
    .cast_common_dtype_to_outputs(true)
    .build();

  ScalarType common = iter.common_dtype();  // Query the promoted type

  AT_DISPATCH_FLOATING_TYPES(common, "mul_cpu", [&] {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
      return a * b;
    });
  });

  return iter.output();
}
```

### Pattern 4: Reduction Operations

**Use Case**: Reductions like `sum`, `mean`, `max`

```cpp
Tensor sum_impl(const Tensor& self, IntArrayRef dim) {
  // Reductions use TensorIterator differently:
  // - Shape reflects the reduced dimensions
  // - Special iteration modes for accumulation

  auto iter = TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .resize_outputs(true)
    .build();

  // Parallel reduction with accumulation
  iter.parallel_reduce([&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
    // Custom reduction kernel
  });

  return output;
}
```

### Pattern 5: Unary Operation

**Use Case**: Unary functions like `exp`, `log`, `relu`

```cpp
Tensor exp_impl(const Tensor& self) {
  Tensor output = at::empty_like(self);

  auto iter = TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .build();

  cpu_kernel(iter, [](float a) -> float {
    return std::exp(a);
  });

  return output;
}
```

### Pattern 6: Comparison Operations

**Use Case**: Comparisons that return `bool` tensors

```cpp
Tensor eq_impl(const Tensor& self, const Tensor& other) {
  Tensor output = at::empty(self.sizes(), self.options().dtype(kBool));

  auto iter = TensorIteratorConfig()
    .add_output(output)
    .add_input(self)
    .add_input(other)
    .check_all_same_dtype(false)  // Output is bool, inputs may be float
    .build();

  cpu_kernel(iter, [](float a, float b) -> bool {
    return a == b;
  });

  return output;
}
```

---

## Build Process

The `.build()` method orchestrates the entire setup:

**Location**: `aten/src/ATen/TensorIterator.cpp`

### Build Steps

```
1. Validate Configuration
   - Check outputs were added before inputs
   - Verify device/dtype consistency

2. Compute Common Shape
   - Broadcast all operand shapes to find common shape
   - Allocate outputs if needed (for functional ops)

3. Type Promotion (if enabled)
   - Determine common dtype from inputs
   - Insert type conversions as needed

4. Compute Strides
   - For each operand, compute broadcasted strides
   - Broadcast dimensions get stride 0

5. Reorder Dimensions
   - Sort dimensions by stride (optimization)
   - Prioritize non-broadcast dimensions

6. Coalesce Dimensions
   - Merge adjacent compatible dimensions
   - Reduce loop nesting depth

7. Detect Special Cases
   - is_trivial_1d: Single element or fully contiguous
   - is_contiguous: All operands contiguous
   - FastSetupType: CONTIGUOUS, CHANNELS_LAST, NON_OVERLAPPING_DENSE

8. Finalize
   - Set up iteration state
   - Prepare data pointers and strides
```

### Computing Output Strides

**See**: `TensorIterator.cpp:194-200` ("NOTE: [Computing output strides]")

**Algorithm**:
1. If output is correctly sized and defined: **respect its strides** (don't change)
2. If output is undefined or incorrect size:
   - Recover permutation from input strides
   - Sort inputs by stride to find memory order
   - Apply same permutation to output
   - **Goal**: Output matches memory layout of inputs for cache efficiency

---

## Loop Execution

### The `for_each` Method

**Signature**:
```cpp
void for_each(loop2d_t loop, int64_t grain_size = GRAIN_SIZE);
```

Where `loop2d_t` is:
```cpp
using loop2d_t = function_ref<void(
  char** data,             // Array of data pointers (one per operand)
  const int64_t* strides,  // Array of strides (flattened)
  int64_t size0,           // Inner loop size
  int64_t size1            // Outer loop size
)>;
```

**How It Works**:
1. Divide iteration space into chunks of `grain_size` elements
2. Use `parallel_for` to distribute chunks to threads
3. For each chunk, call the loop kernel with appropriate pointers/strides

**1D Convenience**:
```cpp
iter.for_each([](char** data, const int64_t* strides, int64_t n) {
  // Simplified 1D loop
  // Automatically wrapped to 2D internally
});
```

### Data Pointer Layout

```cpp
// Example with 1 output, 2 inputs:
char** data = {
  data[0],  // Output pointer
  data[1],  // Input 1 pointer
  data[2]   // Input 2 pointer
};

// Strides are flattened: [dim0_strides..., dim1_strides...]
const int64_t* strides = {
  stride[0][out], stride[0][in1], stride[0][in2],  // Dimension 0
  stride[1][out], stride[1][in1], stride[1][in2]   // Dimension 1
};
```

### Accessing Elements

The `Loops.h` helper provides automatic element access:

```cpp
// User writes simple lambda:
cpu_kernel(iter, [](float a, float b) -> float {
  return a + b;
});

// Loops.h expands to:
for (int64_t i = 0; i < n; i++) {
  float a = *(float*)(data[1] + i * strides[1]);
  float b = *(float*)(data[2] + i * strides[2]);
  float result = lambda(a, b);  // User's lambda
  *(float*)(data[0] + i * strides[0]) = result;
}
```

---

## Advanced Features

### 32-Bit Indexing (GPU Optimization)

**Problem**: GPUs have limited register space. Using 64-bit offsets wastes registers.

**Solution**: Split iteration into sub-iterators that fit in 32-bit indexing.

```cpp
if (!iter.can_use_32bit_indexing()) {
  // Split into multiple sub-iterators
  for (auto& sub_iter : iter.with_32bit_indexing()) {
    gpu_kernel(sub_iter, ...);
  }
} else {
  gpu_kernel(iter, ...);
}
```

**Implementation**:
- `can_use_32bit_indexing()`: Checks if largest offset fits in `int32_t`
- `with_32bit_indexing()`: Returns iterator that lazily splits on demand
- `split(int dim)`: Splits iterator along specified dimension

### Memory Overlap Detection

TensorIterator checks for illegal memory aliasing:

```cpp
// Example: In-place operation with overlapping views
Tensor a = ...;
Tensor b = a[:, :10];  // View into a
a.add_(b);  // ERROR: Overlapping memory access!
```

**Detection**:
- Uses `at::has_internal_overlap()` to check for self-overlap
- Uses `at::assert_no_partial_overlap()` for cross-tensor overlap
- Throws error if unsafe aliasing detected

### Split Operations

For very large tensors, you can manually split iteration:

```cpp
std::unique_ptr<TensorIterator> upper = iter.split(dim);
// Now 'iter' covers lower half, 'upper' covers upper half along 'dim'
```

**Use Cases**:
- Distributing work across multiple GPUs
- Working around device memory limits
- Custom parallelization strategies

### Narrowing

Restrict iteration to a subrange:

```cpp
iter.narrow(dim, start, size);
// Now iterates only over [start:start+size] along 'dim'
```

**Use Case**: Implementing slicing operations or chunked processing.

---

## Performance Considerations

### When TensorIterator Shines

**Good Fit**:
- Element-wise operations (add, mul, exp, relu)
- Broadcasting is frequent
- Mixed strides (e.g., transposed tensors)
- Type promotion needed
- Operations on CPU with AVX2/AVX512

**Performance Wins**:
- Automatic vectorization (2-8x speedup)
- Optimal dimension ordering (2-10x for poor layouts)
- Avoids unnecessary copies
- Efficient broadcasting (stride 0 trick)

### When TensorIterator Adds Overhead

**Poor Fit**:
- Very small tensors (<1000 elements): Setup overhead dominates
- Operations with complex data dependencies (not element-wise)
- Custom memory access patterns
- Operations already implemented with specialized libraries (cuBLAS, cuDNN)

### Optimization Tips

1. **Prefer contiguous tensors**: Check `is_contiguous()` and use fast paths
2. **Vectorize explicitly**: Use `cpu_kernel_vec` for critical kernels
3. **Coalescing**: Ensure dimensions can coalesce (check strides)
4. **Grain size**: Tune `grain_size` for parallelization (default 32768)
5. **Avoid type conversion**: Match output dtype to input when possible

---

## MLX Porting Considerations

### Conceptual Differences

| Aspect | PyTorch TensorIterator | MLX |
|--------|------------------------|-----|
| **Execution** | Eager iteration with immediate execution | Lazy graph construction with delayed execution |
| **Broadcasting** | Runtime broadcasting with stride tricks | Graph-level broadcasting optimization |
| **Vectorization** | Explicit CPU SIMD (AVX2/AVX512) | Metal GPU kernels, automatic vectorization |
| **Type Promotion** | Runtime type promotion | Compile-time type inference in graph |
| **Memory Model** | Separate CPU/GPU memory | Unified memory on Apple Silicon |

### What MLX Already Provides

MLX handles broadcasting and type promotion differently:

```cpp
// MLX broadcasting is automatic at the operation level
array c = a + b;  // Automatically broadcasts if needed
// No need for explicit TensorIterator setup
```

**MLX Benefits**:
- **Simpler API**: No iterator setup needed
- **Lazy evaluation**: Operations are compiled into optimized Metal kernels
- **Unified memory**: Eliminates CPU↔GPU copies
- **Metal Performance Shaders**: Highly optimized for Apple Silicon

### What MLX Lacks (TensorIterator Features)

1. **Explicit stride control**: MLX has simpler memory layout assumptions
2. **CPU vectorization**: MLX targets Metal/GPU primarily
3. **Dimension reordering**: Less critical with unified memory
4. **32-bit indexing optimization**: Metal handles this differently

### Recommendations for MLX Port

#### Don't Port TensorIterator Directly

TensorIterator is deeply tied to PyTorch's eager execution model. Instead:

1. **Understand the pattern**: TensorIterator shows how PyTorch handles:
   - Broadcasting semantics (replicate these rules)
   - Type promotion logic (adopt the same rules)
   - Vectorization strategies (adapt to Metal)

2. **Adopt broadcasting rules**: Implement NumPy-compatible broadcasting in MLX's graph compiler

3. **Type promotion**: Use TensorIterator's promotion rules (see `broadcasting-promotion.md`)

4. **Vectorization**: Let Metal compiler handle SIMD, but understand the patterns

#### Focus on High-Level Concepts

**Port These Ideas**:
- ✅ Broadcasting semantics
- ✅ Type promotion rules
- ✅ Memory layout optimization strategies
- ✅ Parallelization strategies

**Don't Port These Mechanisms**:
- ❌ Explicit TensorIterator class
- ❌ CPU SIMD vectorization (use Metal instead)
- ❌ Runtime dimension reordering (less critical with unified memory)
- ❌ 32-bit indexing tricks (Metal has different limitations)

#### Example: MLX Equivalent Pattern

**PyTorch**:
```cpp
Tensor add_impl(const Tensor& a, const Tensor& b) {
  auto iter = TensorIteratorConfig()
    .add_output(Tensor())
    .add_input(a)
    .add_input(b)
    .promote_inputs_to_common_dtype()
    .build();

  cpu_kernel(iter, [](float x, float y) { return x + y; });
  return iter.output();
}
```

**MLX Equivalent**:
```cpp
array add(const array& a, const array& b) {
  // Broadcasting and type promotion handled automatically
  // Returns lazy graph node that compiles to Metal kernel
  return array(
    /* primitive that encodes: elementwise_add(broadcast(a), broadcast(b)) */
  );
}
```

---

## Common Pitfalls

### Pitfall 1: Adding Inputs Before Outputs

```cpp
// WRONG:
auto iter = TensorIteratorConfig()
  .add_input(input)   // ❌ Input first
  .add_output(output) // ❌ Throws exception!
  .build();

// CORRECT:
auto iter = TensorIteratorConfig()
  .add_output(output) // ✅ Outputs first
  .add_input(input)   // ✅ Then inputs
  .build();
```

### Pitfall 2: Forgetting resize_outputs(false) for In-Place

```cpp
// WRONG (for in-place):
Tensor& add_inplace(Tensor& self, const Tensor& other) {
  auto iter = TensorIteratorConfig()
    .add_output(self)
    .add_input(self)
    .add_input(other)
    // ❌ Missing: .resize_outputs(false)
    .build();
  // May incorrectly resize self!
}
```

### Pitfall 3: Mixing Owned vs Borrowed

```cpp
// WRONG: Mixing ownership models
auto iter = TensorIteratorConfig()
  .add_owned_output(output)      // Owns output
  .add_borrowed_input(input)     // Borrows input
  // ❌ Inconsistent ownership can cause lifetime issues
  .build();
```

**Best Practice**: Use `add_output/add_input` (default) unless you have specific ownership requirements.

### Pitfall 4: Assuming Element Strides

```cpp
// WRONG: Assuming stride is in elements
int64_t stride_elements = operands_[0].stride_bytes[0];  // ❌ Actually in bytes!

// CORRECT:
int64_t stride_bytes = operands_[0].stride_bytes[0];
int64_t stride_elements = stride_bytes / elementSize(dtype());  // ✅
```

### Pitfall 5: Ignoring common_dtype Before Build

```cpp
// WRONG:
auto iter = TensorIteratorConfig()
  .add_output(output)
  .add_input(int_tensor)
  .add_input(float_tensor)
  .promote_inputs_to_common_dtype()
  // ❌ Don't query common_dtype before build()
  .common_dtype();  // ❌ Doesn't exist on config!

// CORRECT:
auto iter = TensorIteratorConfig()
  .add_output(output)
  .add_input(int_tensor)
  .add_input(float_tensor)
  .promote_inputs_to_common_dtype()
  .build();
ScalarType common = iter.common_dtype();  // ✅ Query after build
```

---

## Critical Files Reference

**Core TensorIterator**:
- [aten/src/ATen/TensorIterator.h](../reference/pytorch/aten/src/ATen/TensorIterator.h) - Main header, 800+ lines
- [aten/src/ATen/TensorIterator.cpp](../reference/pytorch/aten/src/ATen/TensorIterator.cpp) - Implementation, 2000+ lines
- [aten/src/ATen/native/TensorIterator.cpp](../reference/pytorch/aten/src/ATen/native/TensorIterator.cpp) - Native ops support

**CPU Kernels**:
- [aten/src/ATen/native/cpu/Loops.h](../reference/pytorch/aten/src/ATen/native/cpu/Loops.h) - `cpu_kernel`, `cpu_kernel_vec`
- [aten/src/ATen/cpu/vec/vec.h](../reference/pytorch/aten/src/ATen/cpu/vec/vec.h) - Vectorized types

**Example Usage**:
- [aten/src/ATen/native/BinaryOps.cpp](../reference/pytorch/aten/src/ATen/native/BinaryOps.cpp) - Binary operations (add, mul, div)
- [aten/src/ATen/native/UnaryOps.cpp](../reference/pytorch/aten/src/ATen/native/UnaryOps.cpp) - Unary operations (exp, log, sqrt)
- [aten/src/ATen/native/Activation.cpp](../reference/pytorch/aten/src/ATen/native/Activation.cpp) - Activation functions

---

## Summary

**TensorIterator is PyTorch's secret sauce for efficient element-wise operations.** It automatically handles:
- ✅ Broadcasting arbitrary shapes
- ✅ Type promotion
- ✅ Memory layout optimization
- ✅ CPU vectorization (AVX2/AVX512)
- ✅ Parallel execution
- ✅ 32-bit indexing for GPUs

**For MLX Porting**:
- Don't port TensorIterator directly—MLX uses lazy evaluation
- **Do** adopt its broadcasting and type promotion semantics
- **Do** understand its optimization strategies for your own Metal kernels
- **Do** reference it when implementing element-wise operations

**Next Steps**:
1. Read [broadcasting-promotion.md](broadcasting-promotion.md) for type promotion rules
2. Study example operators: `BinaryOps.cpp`, `UnaryOps.cpp`
3. Understand how MLX's lazy evaluation handles similar concerns
