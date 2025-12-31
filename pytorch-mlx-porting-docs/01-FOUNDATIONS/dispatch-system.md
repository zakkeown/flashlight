# Dispatch System - Multi-Backend Operator Routing

## Purpose

PyTorch's dispatch system is its **core architectural innovation**, enabling a single operator API to route to different implementations based on tensor properties (device, dtype, layout) and execution context (autograd, tracing, profiling). This allows PyTorch to support multiple backends (CPU, CUDA, Metal, XLA) while maintaining a unified Python/C++ API.

Understanding the dispatch system is critical for MLX porting because it shows how to:
- Register backend-specific implementations (e.g., Metal kernels)
- Handle multiple dispatch (mixed CPU/GPU tensors)
- Integrate autograd with backend kernels
- Add new functionality layers without modifying existing code

## Architecture Overview

```
User calls: torch.add(tensor_a, tensor_b)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Extract DispatchKeySet from tensors                 │
│  - tensor_a.key_set() → {CPU, AutogradCPU}                  │
│  - tensor_b.key_set() → {CPU, AutogradCPU}                  │
│  - Combined: {CPU, AutogradCPU}                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Dispatcher looks up "add" operator                  │
│  - Global operator table: name → OperatorHandle             │
│  - OperatorHandle contains dispatch table for all keys      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Query dispatch table with highest priority key      │
│  - Priority order: AutogradCPU > CPU                        │
│  - Lookup: dispatch_table[AutogradCPU] → autograd_add       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Execute kernel                                       │
│  - autograd_add() runs                                       │
│    - Records operation for backward pass                    │
│    - Redispatches to next key (CPU) via redispatch()        │
│  - cpu_add() runs                                            │
│    - Actual computation happens here                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. DispatchKey - Identifying Handlers

**File**: [reference/pytorch/c10/core/DispatchKey.h:136-434](reference/pytorch/c10/core/DispatchKey.h)

**Purpose**: Enum identifying which kernel/handler should execute for an operation.

**Structure**:

```cpp
enum class DispatchKey : uint16_t {
  Undefined = 0,

  // ~~~~~~~~~~~ Functionality Keys (ordered by priority) ~~~~~~~~~~~

  // Highest priority: Pre-dispatch and Python interception
  PythonDispatcher,         // Bypass C++ dispatcher entirely
  PreDispatch,              // Tracing in make_fx

  // Wrapper layers (before autograd)
  FuncTorchDynamicLayerFrontMode,  // Functorch front mode
  Batched,                  // vmap batching
  VmapMode,                 // Inside vmap context

  // Functionalization (alias/mutation removal)
  Functionalize,            // Remove aliasing and mutations

  // Autograd layer (differentiation)
  ADInplaceOrView,          // Tracks views and in-place ops for autograd
  AutogradCPU,              // CPU autograd (per-backend)
  AutogradCUDA,             // CUDA autograd (per-backend)
  AutogradXLA,              // XLA autograd (per-backend)
  AutogradMPS,              // Metal autograd (per-backend)
  AutogradOther,            // Other backends' autograd
  AutogradFunctionality,    // Generic autograd (building block)

  // Autocasting (automatic mixed precision)
  AutocastCPU,
  AutocastCUDA,
  AutocastMPS,

  // Named tensors
  Named,

  // Conjugate/Negative views
  Conjugate,
  Negative,
  ZeroTensor,

  // Sparse/Quantized/Dense (functionality building blocks)
  Sparse,
  SparseCsr,
  Quantized,
  Dense,                    // Dense tensor functionality

  // Non-extensible backends
  Vulkan,
  Metal,                    // Metal (MPS) backend
  FPGA,

  // Backend selection
  BackendSelect,            // Determine backend for ops without tensor args

  // ~~~~~~~~~~~ Backend Building Blocks ~~~~~~~~~~~
  CPUBit,                   // Building block for CPU
  CUDABit,                  // Building block for CUDA
  XLABit,                   // Building block for XLA
  MPSBit,                   // Building block for Metal Performance Shaders
  IPUBit,                   // Building block for Graphcore IPU
  XPUBit,                   // Building block for Intel XPU
  // ... more backend bits

  // ~~~~~~~~~~~ Runtime Keys (combinations) ~~~~~~~~~~~
  CPU,                      // Dense CPU tensor (CPUBit + Dense)
  CUDA,                     // Dense CUDA tensor (CUDABit + Dense)
  MPS,                      // Dense Metal tensor (MPSBit + Dense)
  XLA,                      // Dense XLA tensor (XLABit + Dense)

  SparseCPU,                // Sparse CPU (CPUBit + Sparse)
  SparseCUDA,               // Sparse CUDA (CUDABit + Sparse)

  QuantizedCPU,             // Quantized CPU (CPUBit + Quantized)
  QuantizedCUDA,            // Quantized CUDA (CUDABit + Quantized)

  // ... many more combinations
};
```

**Priority Order** (highest to lowest):
1. **Python/Tracing layers**: PythonDispatcher, PreDispatch
2. **Functorch layers**: Dynamic modes, batching
3. **Functionalization**: Alias and mutation removal
4. **Autograd**: ADInplaceOrView, then backend-specific autograd (AutogradCPU, AutogradCUDA, AutogradMPS, ...)
5. **Autocasting**: AutocastCPU, AutocastCUDA, AutocastMPS, ...
6. **Backend computation**: CPU, CUDA, MPS, XLA, ...
7. **Fallback**: Default implementations

**Key Insight**: When multiple keys are present, the dispatcher selects the highest priority key. This is how autograd wraps backend implementations - autograd keys have higher priority and redispatch to backend keys after recording the operation.

### 2. DispatchKeySet - Set of Active Keys

**File**: [reference/pytorch/c10/core/DispatchKeySet.h:49-149](reference/pytorch/c10/core/DispatchKeySet.h)

**Purpose**: A 64-bit bitset representing which dispatch keys are active for a tensor or operation.

**Design Philosophy**:

The challenge: With ~12 backends (CPU, CUDA, XLA, MPS, ...) and ~5 functionalities (Dense, Sparse, Quantized, Autograd, ...), a naive approach would need 12 × 5 = 60+ dispatch keys. With variants, this explodes combinatorially.

**Solution**: DispatchKeySet uses two types of bits:
1. **Backend bits** (e.g., CPUBit, CUDABit, MPSBit)
2. **Functionality bits** (e.g., Dense, Sparse, Autograd)

Runtime keys like `CPU` are computed by combining bits:
```cpp
DispatchKey::CPU = CPUBit + Dense
DispatchKey::SparseCPU = CPUBit + Sparse
DispatchKey::AutogradCPU = CPUBit + AutogradFunctionality
```

**Structure**:

```cpp
class DispatchKeySet {
 private:
  uint64_t repr_;  // 64-bit representation

 public:
  // Construct from single key
  DispatchKeySet(DispatchKey k);

  // Construct from multiple keys
  DispatchKeySet(std::initializer_list<DispatchKey> ks);

  // Check if key is in set
  bool has(DispatchKey k) const;

  // Add key to set
  DispatchKeySet add(DispatchKey k) const;

  // Remove key from set
  DispatchKeySet remove(DispatchKey k) const;

  // Get highest priority key
  DispatchKey highestPriorityTypeId() const;

  // Combine with another set (OR operation)
  DispatchKeySet operator|(const DispatchKeySet& other) const;

  // Intersect with another set (AND operation)
  DispatchKeySet operator&(const DispatchKeySet& other) const;
};
```

**Example**:

```cpp
// Tensor on CPU with autograd
auto keyset = DispatchKeySet({DispatchKey::CPU, DispatchKey::AutogradCPU});

// Check membership
keyset.has(DispatchKey::CPU);          // true
keyset.has(DispatchKey::AutogradCPU);  // true
keyset.has(DispatchKey::CUDA);         // false

// Get highest priority (for dispatch)
keyset.highestPriorityTypeId();  // AutogradCPU (higher priority than CPU)

// Combine keysets (for multiple tensor inputs)
auto keyset_a = DispatchKeySet({DispatchKey::CPU});
auto keyset_b = DispatchKeySet({DispatchKey::AutogradCPU});
auto combined = keyset_a | keyset_b;  // {CPU, AutogradCPU}
```

**Extracting DispatchKeySet from Tensors**:

```cpp
// From a single tensor
at::Tensor t = torch::randn({3, 4});
DispatchKeySet ks = t.key_set();

// From multiple tensors (combine with OR)
DispatchKeySet ks = tensor_a.key_set() | tensor_b.key_set() | tensor_c.key_set();
```

### 3. Dispatcher - The Central Routing Table

**File**: [reference/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:70-199](reference/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h)

**Purpose**: Global singleton that maintains the operator registry and routes calls to appropriate kernels.

**Structure**:

```cpp
class Dispatcher {
 public:
  // Get singleton instance
  static Dispatcher& singleton();

  // Look up operator by name
  std::optional<OperatorHandle> findSchema(const OperatorName& name);
  OperatorHandle findSchemaOrThrow(const char* name, const char* overload);

  // Call an operator (extracts keys from tensors, dispatches)
  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return(Args...)>& op, Args... args);

  // Redispatch with explicit keyset (used inside kernels)
  template<class Return, class... Args>
  Return redispatch(
      const TypedOperatorHandle<Return(Args...)>& op,
      DispatchKeySet keyset,
      Args... args);

  // Register operators (called by TORCH_LIBRARY)
  RegistrationHandleRAII registerDef(
      FunctionSchema schema,
      std::string debug_name);

  // Register implementations (called by TORCH_LIBRARY_IMPL)
  RegistrationHandleRAII registerImpl(
      OperatorName op_name,
      DispatchKey dispatch_key,
      KernelFunction kernel);
};
```

**Internal Data Structure**:

```cpp
struct OperatorDef {
  impl::OperatorEntry op;  // Contains dispatch table
  size_t def_count;        // Number of schema registrations
  size_t def_and_impl_count;  // Total registrations
};

// Global lookup table: operator name → OperatorDef
std::unordered_map<OperatorName, OperatorDef> operatorLookupTable_;
```

**OperatorEntry** (the dispatch table for one operator):

```cpp
class OperatorEntry {
 private:
  FunctionSchema schema_;  // Operator signature

  // Dispatch table: DispatchKey → KernelFunction
  // Indexed by DispatchKey enum value
  std::array<KernelFunction, num_runtime_entries> kernels_;

 public:
  // Lookup kernel for a dispatch key
  const KernelFunction& lookup(DispatchKey key) const;

  // Register kernel for a dispatch key
  void registerKernel(DispatchKey key, KernelFunction kernel);

  // Call with automatic key extraction
  template<class... Args>
  auto call(Args&&... args);
};
```

**Dispatch Flow**:

```
1. dispatcher.call(op_handle, tensor_a, tensor_b)
       │
       ▼
2. Extract DispatchKeySet:
   - keyset = tensor_a.key_set() | tensor_b.key_set()
   - keyset = {CPU, AutogradCPU}
       │
       ▼
3. Get highest priority key:
   - key = keyset.highestPriorityTypeId()
   - key = AutogradCPU
       │
       ▼
4. Lookup kernel:
   - kernel = op_handle.lookup(AutogradCPU)
   - kernel = autograd_cpu_add
       │
       ▼
5. Execute kernel:
   - autograd_cpu_add(tensor_a, tensor_b)
```

### 4. Operator Registration - TORCH_LIBRARY Macros

**File**: [reference/pytorch/torch/library.h:972-1079](reference/pytorch/torch/library.h)

**Purpose**: Macros for registering operators and their implementations at static initialization time.

**TORCH_LIBRARY** - Define Operator Schema:

```cpp
// Define operators in namespace "myops"
TORCH_LIBRARY(myops, m) {
  // Register operator schema
  m.def("add(Tensor self, Tensor other) -> Tensor");
  m.def("mul(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");

  // Can also provide default implementation
  m.def("subtract", &subtract_impl);
}
```

**Expands to**:

```cpp
static void TORCH_LIBRARY_init_myops(torch::Library& m);
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_myops(
    torch::Library::DEF,
    &TORCH_LIBRARY_init_myops,
    "myops",
    std::nullopt,
    __FILE__,
    __LINE__);

void TORCH_LIBRARY_init_myops(torch::Library& m) {
  m.def("add(Tensor self, Tensor other) -> Tensor");
  // ... user code ...
}
```

**TORCH_LIBRARY_IMPL** - Register Backend Implementation:

```cpp
// Implement "myops" operators for CPU
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("add", &cpu_add);
  m.impl("mul", &cpu_mul);
}

// Implement "myops" operators for CUDA
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("add", &cuda_add);
  m.impl("mul", &cuda_mul);
}

// Implement "myops" operators for Metal (MPS)
TORCH_LIBRARY_IMPL(myops, MPS, m) {
  m.impl("add", &mps_add);
  m.impl("mul", &mps_mul);
}

// Implement autograd for "myops"
TORCH_LIBRARY_IMPL(myops, AutogradCPU, m) {
  m.impl("add", &autograd_cpu_add);
}
```

**TORCH_LIBRARY_IMPL for Fallbacks**:

```cpp
// Register fallback for all operators in all namespaces on XLA
TORCH_LIBRARY_IMPL(_, XLA, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&xla_fallback>());
}
```

### 5. Kernel Registration Example

**Complete Example**:

```cpp
// 1. Define schema
TORCH_LIBRARY(myops, m) {
  m.def("add(Tensor self, Tensor other) -> Tensor");
}

// 2. Implement for CPU
torch::Tensor add_cpu(const torch::Tensor& self, const torch::Tensor& other) {
  // CPU implementation
  return self + other;  // Uses CPU kernels
}

TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("add", &add_cpu);
}

// 3. Implement for CUDA
torch::Tensor add_cuda(const torch::Tensor& self, const torch::Tensor& other) {
  // CUDA implementation
  // ... CUDA kernel launch ...
  return result;
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("add", &add_cuda);
}

// 4. Implement autograd
torch::Tensor add_autograd(const torch::Tensor& self, const torch::Tensor& other) {
  // Record operation for backward pass
  auto result = [&]() {
    // Redispatch to next key (CPU or CUDA)
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::add(self, other);
  }();

  // Set up gradient function
  if (torch::autograd::compute_requires_grad(self, other)) {
    auto grad_fn = std::make_shared<AddBackward>();
    grad_fn->set_next_edges(/* ... */);
    set_history(result, grad_fn);
  }

  return result;
}

TORCH_LIBRARY_IMPL(myops, AutogradCPU, m) {
  m.impl("add", &add_autograd);
}
```

## Data Flow

### Example: torch.add(cpu_tensor_a, cpu_tensor_b) with requires_grad=True

```
Step 1: User calls torch.add(a, b)
        - a: CPU tensor, requires_grad=True
        - b: CPU tensor, requires_grad=True
        │
        ▼
Step 2: Extract DispatchKeySet
        - a.key_set() = {CPU, AutogradCPU}
        - b.key_set() = {CPU, AutogradCPU}
        - Combined: {CPU, AutogradCPU}
        │
        ▼
Step 3: Dispatcher.call(add_op, a, b)
        - Lookup "add" in operator registry
        - Get OperatorEntry for "add"
        │
        ▼
Step 4: Query dispatch table
        - Highest priority key: AutogradCPU
        - Lookup: dispatch_table[AutogradCPU] → autograd_cpu_add
        │
        ▼
Step 5: Execute autograd_cpu_add(a, b)
        - Creates AddBackward node
        - Redispatches with keyset = {CPU}  (AutogradCPU removed)
        │
        ▼
Step 6: Dispatcher.redispatch(add_op, {CPU}, a, b)
        - Highest priority key: CPU
        - Lookup: dispatch_table[CPU] → cpu_add
        │
        ▼
Step 7: Execute cpu_add(a, b)
        - Actual CPU computation
        - Returns result tensor
        │
        ▼
Step 8: Return to autograd layer
        - Attach grad_fn to result
        - Return result to user
```

### Example: Mixed Device Dispatch (CPU + CUDA)

```
torch.add(cpu_tensor, cuda_tensor)
        │
        ▼
Extract DispatchKeySet:
- cpu_tensor.key_set() = {CPU}
- cuda_tensor.key_set() = {CUDA}
- Combined: {CPU, CUDA}  ← Multiple backends!
        │
        ▼
Dispatcher behavior:
- Some ops have special "mixed device" kernels
- Most ops error: "Expected all tensors on same device"
- Exception: Scalar tensors (0-dim) can cross devices
```

### Example: Redispatch Pattern (Common in Autograd)

```cpp
torch::Tensor my_op_autograd(const torch::Tensor& self) {
  // WRONG: This would recursively call autograd kernel
  // return at::my_op(self);

  // CORRECT: Remove autograd key from dispatch set
  at::AutoDispatchBelowAutograd guard;
  return at::redispatch::my_op(self);
}
```

The guard modifies thread-local state to exclude autograd keys from dispatch, ensuring we dispatch to the backend kernel (CPU, CUDA, etc.) instead of recursing.

## Code Examples

### Example 1: Registering a Custom Operator

```cpp
#include <torch/library.h>

// Define schema
TORCH_LIBRARY(my_custom_ops, m) {
  m.def("relu_squared(Tensor self) -> Tensor");
}

// CPU implementation
torch::Tensor relu_squared_cpu(const torch::Tensor& self) {
  auto relu_out = torch::relu(self);
  return relu_out * relu_out;
}

TORCH_LIBRARY_IMPL(my_custom_ops, CPU, m) {
  m.impl("relu_squared", &relu_squared_cpu);
}

// CUDA implementation
torch::Tensor relu_squared_cuda(const torch::Tensor& self) {
  auto relu_out = torch::relu(self);
  return relu_out * relu_out;
}

TORCH_LIBRARY_IMPL(my_custom_ops, CUDA, m) {
  m.impl("relu_squared", &relu_squared_cuda);
}

// Autograd implementation
torch::Tensor relu_squared_autograd(const torch::Tensor& self) {
  torch::Tensor result;
  {
    at::AutoDispatchBelowAutograd guard;
    result = at::redispatch::my_custom_ops::relu_squared(self);
  }

  if (self.requires_grad()) {
    // Set up backward pass
    auto grad_fn = std::make_shared<ReluSquaredBackward>();
    grad_fn->save_for_backward({self});
    set_history(result, grad_fn);
  }

  return result;
}

TORCH_LIBRARY_IMPL(my_custom_ops, Autograd, m) {
  m.impl("relu_squared", &relu_squared_autograd);
}
```

Usage:
```python
import torch
torch.ops.load_library("my_custom_ops.so")

x = torch.randn(3, 4, requires_grad=True)
y = torch.ops.my_custom_ops.relu_squared(x)
y.sum().backward()
```

### Example 2: Registering a Fallback Kernel

```cpp
// Fallback for all operators on a custom backend
void my_backend_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {

  // Convert inputs from custom backend to CPU
  // ... conversion logic ...

  // Redispatch to CPU
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

  // Convert outputs back to custom backend
  // ... conversion logic ...
}

TORCH_LIBRARY_IMPL(_, MyBackend, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&my_backend_fallback>());
}
```

### Example 3: Inspecting Dispatch Keys

```python
import torch

x = torch.randn(3, 4, requires_grad=True)
print(x._dispatch_key_set())  # DispatchKeySet: {CPU, AutogradCPU}

y = torch.randn(3, 4, device='cuda', requires_grad=True)
print(y._dispatch_key_set())  # DispatchKeySet: {CUDA, AutogradCUDA}

z = torch.randn(3, 4)  # No requires_grad
print(z._dispatch_key_set())  # DispatchKeySet: {CPU}
```

## MLX Porting Considerations

### What MLX Already Provides

MLX has a different execution model:
- **Lazy evaluation**: Operations are queued, then compiled to Metal compute graph
- **Automatic differentiation**: Via `mlx::grad` function transformation, not tape-based
- **Unified dispatch**: Metal-only, no need for multi-backend dispatch

### What Needs Adaptation

1. **Dispatch Complexity**:
   - PyTorch: Complex multi-backend dispatch (CPU, CUDA, Metal, XLA, ...)
   - MLX: Primarily Metal-only
   - **Action**: MLX can use a simpler dispatch system, but still needs layering (e.g., autograd wrapper around Metal kernels)

2. **Operator Registration**:
   - PyTorch: YAML-defined operators + TORCH_LIBRARY macros
   - MLX: Direct C++ registration
   - **Action**: If building PyTorch-compatible API, could adopt YAML approach for consistency

3. **Autograd Integration**:
   - PyTorch: Autograd as a dispatch layer (higher priority than backends)
   - MLX: Autograd via `grad` transformation (functional approach)
   - **Action**: Different paradigms - MLX's approach may be simpler for Metal compilation

4. **Mixed Device Dispatch**:
   - PyTorch: Handles CPU + CUDA mixed inputs (limited support)
   - MLX: Unified memory on Apple Silicon makes this less relevant
   - **Advantage**: MLX can skip complex mixed-device logic

### Metal-Specific Opportunities

1. **Simplified Dispatch**:
   - Only need Metal backend dispatch, not CPU/CUDA/XLA/...
   - Opportunity: Leaner dispatch table, faster lookup

2. **Compile-Time Dispatch**:
   - MLX's lazy evaluation allows compile-time optimization
   - Could inline dispatch for known operator sequences
   - Metal compiler can optimize entire graphs

3. **Unified Memory Benefits**:
   - No device transfer overhead
   - Simplified memory management
   - Can share data structures between "backends" (CPU/GPU views of same memory)

### Recommendations for MLX

1. **Minimal Dispatch Layer**:
   - Don't replicate PyTorch's full complexity
   - Simple dispatch: Operator name → Metal kernel
   - Layering: Autograd transform wraps Metal implementation

2. **Operator Registration**:
   - If targeting PyTorch compatibility: Use YAML schema for consistent API
   - If MLX-native: Direct C++ registration is simpler

3. **Backward Compatibility**:
   - If porting PyTorch models: Implement dispatch interface compatible with `torch.ops.load_library()`
   - Allows loading existing custom operators

4. **Performance**:
   - PyTorch's dispatch has non-trivial overhead (~20-50ns per call)
   - MLX's lazy evaluation can batch operations before dispatch
   - Opportunity: Eliminate per-op dispatch overhead

## Critical File References

**Dispatch Keys**:
- [c10/core/DispatchKey.h:136-434](reference/pytorch/c10/core/DispatchKey.h) - Complete DispatchKey enum with all keys
- [c10/core/DispatchKeySet.h:49-149](reference/pytorch/c10/core/DispatchKeySet.h) - DispatchKeySet bitset implementation

**Dispatcher**:
- [aten/src/ATen/core/dispatch/Dispatcher.h:70-199](reference/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h) - Main Dispatcher class
- [aten/src/ATen/core/dispatch/OperatorEntry.h](reference/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.h) - Per-operator dispatch table
- [aten/src/ATen/core/dispatch/DispatchKeyExtractor.h](reference/pytorch/aten/src/ATen/core/dispatch/DispatchKeyExtractor.h) - Extracting keys from tensors

**Registration**:
- [torch/library.h:972-1079](reference/pytorch/torch/library.h) - TORCH_LIBRARY and TORCH_LIBRARY_IMPL macros
- [aten/src/ATen/core/op_registration/](reference/pytorch/aten/src/ATen/core/op_registration/) - Registration infrastructure

**Backend Examples**:
- [aten/src/ATen/RegisterCPU.cpp](reference/pytorch/aten/src/ATen/RegisterCPU.cpp) - CPU backend registration
- [aten/src/ATen/RegisterCUDA.cpp](reference/pytorch/aten/src/ATen/RegisterCUDA.cpp) - CUDA backend registration
- [aten/src/ATen/mps/MPSHooks.cpp](reference/pytorch/aten/src/ATen/mps/MPSHooks.cpp) - Metal backend hooks

## Next Steps

1. Study **type-system.md** to understand ScalarType, Device, Layout enums
2. Read **operator-overview.md** to see how operators are defined in YAML and how dispatch table is populated
3. Examine **autograd-overview.md** to understand how autograd integrates with dispatch (AutogradCPU, AutogradCUDA, etc.)
4. For MLX porting, decide whether to adopt PyTorch's dispatch complexity or implement a simpler Metal-only dispatch system
