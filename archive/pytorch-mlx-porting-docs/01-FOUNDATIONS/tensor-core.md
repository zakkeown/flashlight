# Tensor Core - TensorImpl, Storage, and Metadata

## Purpose

The tensor core is the fundamental abstraction in PyTorch, providing the low-level representation of multi-dimensional arrays (tensors). Understanding this layer is critical for MLX porting because it establishes the separation between:
- **Data** (the actual bytes in memory)
- **Metadata** (shape, strides, dtype, device)
- **View semantics** (multiple tensors sharing the same data)

This design enables efficient operations like slicing, transposing, and reshaping without copying data.

## Architecture Overview

```
Tensor (user-facing)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ TensorImpl (internal representation)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ View Metadata (per-tensor, not shared)               │  │
│  │  - sizes: [batch, channels, height, width]           │  │
│  │  - strides: [C×H×W, H×W, W, 1]                       │  │
│  │  - storage_offset: int64_t                           │  │
│  │  - dtype: ScalarType (float32, int64, etc.)          │  │
│  │  - device: Device (CPU, CUDA, MPS, etc.)             │  │
│  │  - layout: Layout (strided, sparse, etc.)            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Shared Storage (reference counted)                   │  │
│  │  - intrusive_ptr<StorageImpl> storage_               │ ─┼──┐
│  └──────────────────────────────────────────────────────┘  │  │
│  ┌──────────────────────────────────────────────────────┐  │  │
│  │ Dispatch & Autograd                                  │  │  │
│  │  - key_set_: DispatchKeySet                          │  │  │
│  │  - version_counter_: VariableVersion                 │  │  │
│  │  - autograd_meta_: unique_ptr<AutogradMetaInterface> │  │  │
│  └──────────────────────────────────────────────────────┘  │  │
└─────────────────────────────────────────────────────────────┘  │
                                                                 │
    ┌────────────────────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────────────────────┐
│ StorageImpl (shared data buffer)                            │
│  - data_ptr_: DataPtr (smart pointer to actual memory)      │
│  - nbytes_: size_t (total bytes allocated)                  │
│  - allocator_: Allocator* (CPU, CUDA, Metal, etc.)          │
│  - device_: Device                                           │
│  - refcount: atomic (intrusive_ptr ref counting)            │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    [Actual Memory Buffer]
    [float, float, float, ...]
```

**Key Insight**: Multiple `TensorImpl` objects can point to the same `StorageImpl`, differing only in their metadata. This is how PyTorch implements views, slices, and transposes without copying data.

## Key Components

### 1. TensorImpl - The Core Tensor Representation

**File**: [reference/pytorch/c10/core/TensorImpl.h:511-1500](reference/pytorch/c10/core/TensorImpl.h)

**Purpose**: Internal representation of a tensor containing all metadata and a pointer to storage.

**Key Data Structures**:

```cpp
struct TensorImpl : public c10::intrusive_ptr_target {
  // STORAGE: Shared data buffer
  Storage storage_;

  // VIEW METADATA: Unique to this tensor
  impl::SizesAndStrides sizes_and_strides_;  // Inline for ≤5 dims
  int64_t storage_offset_;                    // Offset into storage
  int64_t numel_;                             // Cached number of elements

  // TYPE INFORMATION
  caffe2::TypeMeta data_type_;                // ScalarType (float32, int64, etc.)
  std::optional<c10::Device> device_opt_;     // Device (CPU, CUDA, MPS)

  // DISPATCH: Which backend kernels to use
  DispatchKeySet key_set_;                    // Runtime dispatch keys

  // AUTOGRAD: Gradient tracking (optional, only if requires_grad=True)
  std::unique_ptr<AutogradMetaInterface> autograd_meta_;
  VariableVersion version_counter_;           // Tracks in-place modifications

  // ADVANCED: Extra metadata for special cases
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_;
  intrusive_ptr<c10::BackendMeta> backend_meta_;
  std::unique_ptr<c10::ExtraMeta> extra_meta_;

  // FLAGS
  bool is_contiguous_ : 1;
  bool is_channels_last_contiguous_ : 1;
  bool is_channels_last_3d_contiguous_ : 1;
  bool is_channels_last_ : 1;
  bool is_channels_last_3d_ : 1;
  bool is_non_overlapping_and_dense_ : 1;
  bool is_wrapped_number_ : 1;
  bool allow_tensor_metadata_change_ : 1;
  // ... more flags
};
```

**Memory Layout Philosophy**:
- **Minimize overhead**: For common cases (≤5 dimensions), sizes and strides are stored inline
- **Lazy allocation**: Autograd metadata only allocated when `requires_grad=True`
- **Reference counting**: Storage is shared via `intrusive_ptr` (atomic refcount)

### 2. Storage - The Data Buffer Abstraction

**File**: [reference/pytorch/c10/core/Storage.h:25-223](reference/pytorch/c10/core/Storage.h)

**Purpose**: Reference-counted wrapper around `StorageImpl`, which manages the actual memory buffer.

**Key APIs**:

```cpp
struct Storage {
  // Access to underlying data
  const void* data() const;            // Read-only pointer
  void* mutable_data() const;          // Mutable pointer

  // Size in bytes
  size_t nbytes() const;
  void set_nbytes(size_t size_bytes);

  // Device and allocator
  at::Device device() const;
  at::Allocator* allocator() const;

  // Reference counting
  size_t use_count() const;
  bool unique() const;
  bool is_alias_of(const Storage& other) const;

  // Underlying implementation
  StorageImpl* unsafeGetStorageImpl() const;

 private:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};
```

**StorageImpl Structure** ([reference/pytorch/c10/core/StorageImpl.h](reference/pytorch/c10/core/StorageImpl.h)):

```cpp
struct StorageImpl : public c10::intrusive_ptr_target {
  DataPtr data_ptr_;        // Smart pointer with custom deleter
  SymInt nbytes_;           // Size in bytes (can be symbolic for tracing)
  bool resizable_;          // Can this storage be resized?
  Allocator* allocator_;    // Allocator for this device

  // Atomic reference count inherited from intrusive_ptr_target
};
```

**DataPtr**: A smart pointer that carries both the data pointer and a deleter function:
```cpp
struct DataPtr {
  void* ptr_;
  void* ctx_;                          // Context for deleter
  DeleterFnPtr deleter_;               // Custom deletion function
  Device device_;
};
```

### 3. SizesAndStrides - Optimized Metadata Storage

**File**: [reference/pytorch/c10/core/impl/SizesAndStrides.h:23-329](reference/pytorch/c10/core/impl/SizesAndStrides.h)

**Purpose**: Efficiently store tensor shape and strides with inline storage for common cases.

**Design**:

```cpp
class SizesAndStrides {
 private:
  size_t size_;  // Number of dimensions
  union {
    // For ≤5 dimensions: inline storage (no heap allocation)
    int64_t inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2];

    // For >5 dimensions: pointer to heap-allocated array
    int64_t* outOfLineStorage_;
  };

 public:
  // Memory layout for inline storage (5 dims):
  // [size0, size1, size2, size3, size4,
  //  stride0, stride1, stride2, stride3, stride4]

  const int64_t* sizes_data() const;
  const int64_t* strides_data() const;

  int64_t size_at(size_t idx) const;
  int64_t stride_at(size_t idx) const;

  void resize(size_t newSize);
  void set_sizes(IntArrayRef newSizes);
  void set_strides(IntArrayRef strides);
};
```

**Optimization**: For tensors with ≤5 dimensions (the vast majority), sizes and strides are stored inline in `TensorImpl`, avoiding a heap allocation. For larger dimensions, the storage falls back to a heap-allocated array.

```
Inline (≤5 dims):
TensorImpl contains: [size₀, size₁, size₂, size₃, size₄ | stride₀, stride₁, stride₂, stride₃, stride₄]
                      ↑─── 5 int64_t ───↑  ↑──── 5 int64_t ────↑

Outline (>5 dims):
TensorImpl contains: [ptr to heap] ──→ [size₀, size₁, ..., sizeₙ, stride₀, stride₁, ..., strideₙ]
```

**MLX Relevance**: MLX's `mlx::array` likely needs similar optimization for small dimension counts.

### 4. VariableVersion - Tracking In-Place Modifications

**File**: [reference/pytorch/c10/core/TensorImpl.h:329-419](reference/pytorch/c10/core/TensorImpl.h)

**Purpose**: Track tensor modifications for autograd safety. When a tensor is saved for backward pass, we record its version. If it's modified in-place afterwards, the version changes, and we can detect the error.

**Structure**:

```cpp
struct VariableVersion {
 private:
  struct VersionCounter : intrusive_ptr_target {
    std::atomic<uint32_t> version_;
  };
  c10::intrusive_ptr<VersionCounter> version_counter_;

 public:
  // For inference tensors, version_counter_ is nullptr (disabled)
  VariableVersion(uint32_t version = 0);
  VariableVersion(Disabled);  // Cheap constructor, no allocation

  bool enabled() const;
  uint32_t current_version() const;
  void bump();  // Increment version on in-place ops
};
```

**Version Counter Sharing** (from NOTE [ Version Counter Sharing ]):
- **Views share version counters** with their base tensor
- `x.detach()` shares the version counter of `x`
- Unpacked saved variables share the version counter

**Not Shared**:
- `x.set_data(...)` creates a new version counter
- `x.data` does not share (intentionally, for historical reasons)

**Why in TensorImpl, not AutogradMeta?**
- Tensors with `requires_grad=False` don't have `AutogradMeta` (optimization)
- But they still need version tracking if saved for backward pass
- Forward pass must be thread-safe, so lazy initialization of `AutogradMeta` would introduce races
- Solution: Always have version counter in `TensorImpl`

### 5. AutogradMeta - Gradient Tracking (Optional)

**File**: [reference/pytorch/c10/core/TensorImpl.h:162-177](reference/pytorch/c10/core/TensorImpl.h)

**Purpose**: Interface for autograd metadata. The actual implementation lives in `torch/csrc/autograd/variable.h` (libtorch.so), while `TensorImpl` lives in c10 (libc10.so). This separation allows c10 to remain minimal.

**Interface**:

```cpp
struct AutogradMetaInterface {
  virtual void set_requires_grad(bool requires_grad, TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& mutable_grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual const at::Tensor& fw_grad(uint64_t level, const at::TensorBase& self) const = 0;
  virtual void set_fw_grad(const at::TensorBase& new_grad, ...) = 0;
  virtual ~AutogradMetaInterface();
};
```

**Actual Implementation** (in `torch/csrc/autograd/variable.h`):

```cpp
struct AutogradMeta : public AutogradMetaInterface {
  Variable grad_;                          // Accumulated gradient
  std::shared_ptr<Node> grad_fn_;         // Backward function (interior nodes)
  std::weak_ptr<Node> grad_accumulator_;  // Gradient accumulator (leaf nodes)
  std::shared_ptr<ForwardGrad> fw_grad_;  // Forward-mode AD support
  bool requires_grad_;
  bool retains_grad_;                      // Keep grad for non-leaf tensors
  uint32_t output_nr_;                     // Output index in parent function
  std::mutex mutex_;                       // Thread safety
};
```

**Factory Pattern**: Since `TensorImpl` (c10) can't directly create `AutogradMeta` (libtorch), there's a factory:

```cpp
struct AutogradMetaFactory {
  virtual std::unique_ptr<AutogradMetaInterface> make() const = 0;
};

// Set once when libtorch.so loads
void SetAutogradMetaFactory(AutogradMetaFactory* factory);
```

## Data Flow

### Creating a Tensor

```
1. User: torch.tensor([1.0, 2.0, 3.0])
         │
         ▼
2. Create StorageImpl:
   - Allocate memory via Allocator (e.g., CPUAllocator)
   - data_ptr_ = DataPtr(malloc(3 * sizeof(float)), CPUDeleter)
   - nbytes_ = 12
   - device_ = Device(CPU)
         │
         ▼
3. Create TensorImpl:
   - storage_ = intrusive_ptr<StorageImpl>(storage_impl)
   - sizes_and_strides_.resize(1)  // 1D tensor
   - sizes_and_strides_.size_at(0) = 3
   - sizes_and_strides_.stride_at(0) = 1
   - storage_offset_ = 0
   - data_type_ = ScalarType::Float
   - device_opt_ = Device(CPU)
   - key_set_ = {CPU}
   - numel_ = 3
         │
         ▼
4. Wrap in Tensor (user-facing):
   - at::Tensor wraps intrusive_ptr<TensorImpl>
```

### Creating a View (e.g., `tensor[:, 0]`)

```
Original Tensor:
  TensorImpl {
    sizes: [4, 5]
    strides: [5, 1]
    offset: 0
    storage: ptr → [20 floats]
  }
         │
         │ view operation: tensor[:, 0]
         ▼
View Tensor:
  TensorImpl {
    sizes: [4]         ← New metadata
    strides: [5]       ← New metadata (step by 5, not 1)
    offset: 0          ← Same offset
    storage: ptr → [20 floats]  ← SAME storage (refcount++)
    version_counter_: shared with original
  }
```

**Key**: View creation is O(1) - just allocate new `TensorImpl` with different metadata, increment storage refcount.

### Storage Lifecycle

```
Tensor A created
  │
  └─→ StorageImpl allocated (refcount = 1)

Tensor B = A.view(...)
  │
  └─→ StorageImpl refcount = 2 (both A and B point to it)

Tensor A destroyed
  │
  └─→ StorageImpl refcount = 1 (B still holds it)

Tensor B destroyed
  │
  └─→ StorageImpl refcount = 0
      │
      └─→ DataPtr destructor called
          │
          └─→ Deleter function called (e.g., free(ptr))
```

### In-Place Modification

```
x = torch.randn(3, 4)
   version = 0

y = x.view(4, 3)
   shares version_counter_ with x

x.add_(1)  # In-place operation
   │
   ├─→ version_counter_.bump()  → version = 1
   │
   └─→ y's version also = 1 (shared counter)
```

**Autograd Safety**: If `x` was saved for backward and then modified, the version mismatch will be detected.

## Code Examples

### Example 1: Understanding Storage Sharing

```python
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"a.data_ptr(): {a.data_ptr()}")
print(f"a.storage().data_ptr(): {a.storage().data_ptr()}")
print(f"a.numel(): {a.numel()}, a.storage().size(): {a.storage().size()}")

# Create a view
b = a[0]  # First row
print(f"\nb.data_ptr(): {b.data_ptr()}")
print(f"b.storage().data_ptr(): {b.storage().data_ptr()}")
print(f"Same storage: {a.storage().data_ptr() == b.storage().data_ptr()}")

# They share storage, but b starts at the same offset
print(f"\na.storage_offset(): {a.storage_offset()}")
print(f"b.storage_offset(): {b.storage_offset()}")

# Different strides
print(f"\na.stride(): {a.stride()}")
print(f"b.stride(): {b.stride()}")
```

Output:
```
a.data_ptr(): 140234567890000
a.storage().data_ptr(): 140234567890000
a.numel(): 4, a.storage().size(): 4

b.data_ptr(): 140234567890000
b.storage().data_ptr(): 140234567890000
Same storage: True

a.storage_offset(): 0
b.storage_offset(): 0

a.stride(): (2, 1)
b.stride(): (1,)
```

### Example 2: Transpose is a View (No Copy)

```python
a = torch.randn(3, 4)
b = a.t()  # Transpose

print(f"Same storage: {a.storage().data_ptr() == b.storage().data_ptr()}")
print(f"a.stride(): {a.stride()}")  # (4, 1) - row-major
print(f"b.stride(): {b.stride()}")  # (1, 4) - column-major (transposed)

# Modifying b affects a (they share storage)
b[0, 0] = 999.0
print(f"a[0, 0] = {a[0, 0]}")  # Also 999.0
```

### Example 3: Slicing with Offset

```python
a = torch.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = a[3:8]            # [3, 4, 5, 6, 7]

print(f"a.storage_offset(): {a.storage_offset()}")  # 0
print(f"b.storage_offset(): {b.storage_offset()}")  # 3 (starts at index 3)

print(f"a.data_ptr(): {a.data_ptr()}")
print(f"b.data_ptr(): {b.data_ptr()}")
print(f"Offset in bytes: {b.data_ptr() - a.data_ptr()}")  # 3 * 8 = 24 bytes (int64)
```

### Example 4: Contiguous vs Non-Contiguous

```python
a = torch.randn(3, 4)
b = a.t()

print(f"a.is_contiguous(): {a.is_contiguous()}")  # True (row-major)
print(f"b.is_contiguous(): {b.is_contiguous()}")  # False (transposed, column-major)

# Making it contiguous creates a copy
c = b.contiguous()
print(f"c.is_contiguous(): {c.is_contiguous()}")  # True
print(f"Same storage: {b.storage().data_ptr() == c.storage().data_ptr()}")  # False (copied)
```

## MLX Porting Considerations

### What MLX Already Provides

MLX has an `mlx::array` type that serves a similar role to PyTorch's Tensor. Key similarities:
- Multi-dimensional arrays with shape and strides
- Unified memory on Apple Silicon (similar to PyTorch's MPS backend)
- Lazy evaluation and graph compilation

### What Needs Adaptation

1. **Storage Abstraction**:
   - PyTorch: Explicit `Storage` and `TensorImpl` separation
   - MLX: Likely more implicit, unified memory means CPU/GPU distinction less critical
   - **Action**: Study MLX's memory model, may not need separate `Storage` class

2. **View Tracking for Autograd**:
   - PyTorch: Version counters, view metadata, shared version counters
   - MLX: Uses function transformations (`mlx::grad`), not tape-based
   - **Action**: Understand if MLX needs similar view tracking or if transform approach eliminates this

3. **Inline Optimization for Sizes/Strides**:
   - PyTorch: Inline for ≤5 dimensions to avoid allocation
   - MLX: Likely beneficial for similar workloads (most ML tensors are 2D-5D)
   - **Action**: Consider adopting similar optimization

4. **Reference Counting**:
   - PyTorch: `intrusive_ptr` for atomic refcounting
   - MLX: C++ modern approach likely uses `std::shared_ptr` or similar
   - **Action**: Profile to see if intrusive pointers provide meaningful benefit on M-series chips

5. **Lazy Initialization**:
   - PyTorch: `AutogradMeta` only allocated when `requires_grad=True`
   - MLX: Different autograd model, but lazy initialization of metadata is still valuable
   - **Action**: Apply lazy initialization pattern for optional metadata

### Metal-Specific Opportunities

1. **Unified Memory**:
   - Apple Silicon has unified memory (CPU and GPU share address space)
   - PyTorch's `Storage` assumes discrete memory (separate CPU/GPU)
   - **MLX Advantage**: Can eliminate expensive data transfers, simplify memory model

2. **Metal Buffer Integration**:
   - Study PyTorch's MPS backend: [reference/pytorch/aten/src/ATen/mps/](reference/pytorch/aten/src/ATen/mps/)
   - Understand how `MTLBuffer` is wrapped in `StorageImpl`
   - **Action**: MLX likely already handles this well, but ensure efficient Metal buffer allocation

3. **Smaller Overhead**:
   - PyTorch's `TensorImpl` is large (~200+ bytes) due to multi-backend support
   - MLX can be leaner since it only targets Metal
   - **Opportunity**: Optimize memory layout for M-series cache hierarchy

### Performance Considerations

1. **Cache Efficiency**:
   - Inline storage for sizes/strides keeps metadata in same cache line as `TensorImpl`
   - Important for operations that iterate over metadata (shape propagation)

2. **Atomic Operations**:
   - Reference counting uses atomic increments/decrements
   - On M-series, test if `std::shared_ptr` (with atomic control block) is competitive with `intrusive_ptr`

3. **View Creation Overhead**:
   - PyTorch: O(1) view creation (allocate new `TensorImpl`, share storage)
   - Ensure MLX has similar O(1) view semantics

## Critical File References

**Core Tensor Abstractions**:
- [c10/core/TensorImpl.h:511-1500](reference/pytorch/c10/core/TensorImpl.h) - Complete `TensorImpl` class
- [c10/core/Storage.h:25-223](reference/pytorch/c10/core/Storage.h) - `Storage` wrapper
- [c10/core/StorageImpl.h](reference/pytorch/c10/core/StorageImpl.h) - Underlying storage implementation
- [c10/core/impl/SizesAndStrides.h:23-329](reference/pytorch/c10/core/impl/SizesAndStrides.h) - Optimized sizes/strides storage

**Related Abstractions**:
- [c10/core/ScalarType.h](reference/pytorch/c10/core/ScalarType.h) - Data type enumeration
- [c10/core/Device.h](reference/pytorch/c10/core/Device.h) - Device abstraction
- [c10/core/Layout.h](reference/pytorch/c10/core/Layout.h) - Tensor layout (strided, sparse, etc.)
- [c10/core/Allocator.h](reference/pytorch/c10/core/Allocator.h) - Memory allocator interface
- [c10/util/intrusive_ptr.h](reference/pytorch/c10/util/intrusive_ptr.h) - Reference counting implementation

**Autograd Integration**:
- [torch/csrc/autograd/variable.h:227-358](reference/pytorch/torch/csrc/autograd/variable.h) - `AutogradMeta` implementation

## Next Steps

1. Study **dispatch-system.md** to understand how `key_set_` determines which backend kernels to call
2. Read **type-system.md** for details on `ScalarType`, `Device`, `Layout`
3. Examine **memory-model.md** for `Allocator` implementations and memory management patterns
4. For MLX porting, compare with MLX's `mlx/core/array.h` to identify gaps and opportunities
