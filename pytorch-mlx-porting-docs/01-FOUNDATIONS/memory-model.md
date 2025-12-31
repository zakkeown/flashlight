# Memory Model - Allocators, Storage, and Memory Management

## Purpose

PyTorch's memory model defines how tensor data is allocated, managed, and deallocated across different devices (CPU, CUDA, Metal). Understanding this model is critical for:
- Implementing efficient memory allocation strategies
- Supporting multiple backends with different memory requirements
- Avoiding memory leaks and use-after-free bugs
- Optimizing for device-specific memory hierarchies
- Implementing custom allocators for specialized hardware

This document covers PyTorch's allocator abstraction, storage lifecycle, and device-specific memory management strategies.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tensor                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TensorImpl                                               │  │
│  │  storage_: intrusive_ptr<StorageImpl> ───────┐           │  │
│  └──────────────────────────────────────────────┼───────────┘  │
└─────────────────────────────────────────────────┼──────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ StorageImpl (reference counted)                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ data_ptr_: DataPtr ──────────────────┐                   │  │
│  │ size_bytes_: SymInt                  │                   │  │
│  │ allocator_: Allocator*               │                   │  │
│  │ resizable_: bool                     │                   │  │
│  │ refcount: atomic (intrusive_ptr)     │                   │  │
│  └──────────────────────────────────────┼───────────────────┘  │
└─────────────────────────────────────────┼──────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ DataPtr (smart pointer with deleter)                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ptr_: void*                          ←───┐               │  │
│  │ ctx_: void* (context for deleter)        │               │  │
│  │ deleter_: DeleterFnPtr                   │               │  │
│  │ device_: Device                          │               │  │
│  └──────────────────────────────────────────┼───────────────┘  │
└─────────────────────────────────────────────┼──────────────────┘
                                               │
                                               ▼
                                    [Actual Memory Buffer]
                                    malloc, cudaMalloc, MTLBuffer
                                               ▲
                                               │
                                               │ allocate/deallocate
                                               │
┌─────────────────────────────────────────────────────────────────┐
│ Allocator (abstract interface)                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ virtual DataPtr allocate(size_t n) = 0;                  │  │
│  │ virtual DeleterFnPtr raw_deleter() const;                │  │
│  │ virtual void copy_data(void* dest, const void* src, ...) │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Implementations:                                                │
│  ├─ CPUAllocator (malloc/free)                                  │
│  ├─ CUDACachingAllocator (cudaMalloc with pool)                 │
│  ├─ MPSAllocator (MTLBuffer for Apple GPU)                      │
│  ├─ XLAAllocator (TPU memory)                                   │
│  └─ Custom allocators...                                         │
└─────────────────────────────────────────────────────────────────┘
```

**Memory Allocation Flow**:
```
1. User creates tensor: torch.randn(3, 4)
2. TensorImpl allocates StorageImpl
3. StorageImpl calls allocator->allocate(size)
4. Allocator returns DataPtr with:
   - Raw pointer to memory
   - Deleter function (e.g., free, cudaFree)
   - Device information
5. When tensor is destroyed:
   - TensorImpl's refcount drops to 0
   - StorageImpl's refcount drops to 0
   - DataPtr destructor calls deleter(ptr)
   - Memory is freed
```

## Key Components

### 1. DataPtr - Smart Pointer with Custom Deleter

**File**: [reference/pytorch/c10/core/Allocator.h:39-138](reference/pytorch/c10/core/Allocator.h)

**Purpose**: A unique pointer to memory with an attached deleter and device information.

**Structure**:

```cpp
class DataPtr {
 private:
  c10::detail::UniqueVoidPtr ptr_;  // Smart pointer (void*, context, deleter)
  Device device_;                    // Which device owns this memory

 public:
  // Constructors
  DataPtr();  // nullptr, defaults to CPU
  DataPtr(void* data, Device device);
  DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device);

  // Access
  void* get() const;                 // Get raw pointer
  void* get_context() const;         // Get deleter context
  DeleterFnPtr get_deleter() const;  // Get deleter function
  Device device() const;             // Get device

  // Ownership
  void* release_context();           // Release ownership, return context
  void clear();                      // Delete and set to null

  // Utilities
  operator bool() const;  // Check if non-null
  bool compare_exchange_deleter(DeleterFnPtr expected, DeleterFnPtr new_deleter);
};
```

**Key Insight**: `DataPtr` is more than `std::unique_ptr<void, Deleter>` - it also carries:
- **Device information**: So we know where the memory lives
- **Context**: For deleters that need additional state (e.g., allocator pointer, event handles)
- **Deleter swapping**: Allows wrapping deleters for instrumentation

**Example**:

```cpp
// CPU allocation with malloc
void cpu_deleter(void* ptr) {
  free(ptr);
}

void* raw_mem = malloc(1024);
DataPtr data_ptr(raw_mem, nullptr, &cpu_deleter, Device(DeviceType::CPU));

// CUDA allocation
void cuda_deleter(void* ptr) {
  cudaFree(ptr);
}

void* cuda_mem = /* cudaMalloc(...) */;
DataPtr cuda_data_ptr(cuda_mem, nullptr, &cuda_deleter, Device(DeviceType::CUDA));

// When data_ptr goes out of scope, cpu_deleter(raw_mem) is called
```

### 2. Allocator - Abstract Allocation Interface

**File**: [reference/pytorch/c10/core/Allocator.h:178-232](reference/pytorch/c10/core/Allocator.h)

**Purpose**: Abstract interface for device-specific memory allocation.

**Interface**:

```cpp
struct Allocator {
  virtual ~Allocator() = default;

  // Allocate memory
  virtual DataPtr allocate(size_t n) = 0;

  // Clone an allocation (calls copy_data)
  DataPtr clone(const void* data, std::size_t n);

  // Check if DataPtr has simple context
  virtual bool is_simple_data_ptr(const DataPtr& data_ptr) const;

  // Get raw deleter (for raw allocate/deallocate API)
  virtual DeleterFnPtr raw_deleter() const {
    return nullptr;  // Default: raw API not supported
  }

  // Raw allocation/deallocation (if supported)
  void* raw_allocate(size_t n);
  void raw_deallocate(void* ptr);

  // Copy data between allocations
  virtual void copy_data(void* dest, const void* src, std::size_t count) const = 0;

 protected:
  // Default implementation using memcpy
  void default_copy_data(void* dest, const void* src, std::size_t count) const;
};
```

**Registration**:

```cpp
// Global allocator registry per device type
C10_API void SetAllocator(DeviceType t, Allocator* alloc, uint8_t priority = 0);
C10_API Allocator* GetAllocator(const DeviceType& t);

// Macro for static registration
#define REGISTER_ALLOCATOR(device_type, allocator_func)
```

### 3. StorageImpl - Memory Buffer Wrapper

**File**: [reference/pytorch/c10/core/StorageImpl.h:51-149](reference/pytorch/c10/core/StorageImpl.h)

**Purpose**: Reference-counted container for tensor data.

**Structure**:

```cpp
struct StorageImpl : public c10::intrusive_ptr_target {
 private:
  at::DataPtr data_ptr_;       // Smart pointer to actual memory
  SymInt size_bytes_;          // Size in bytes (can be symbolic for tracing)
  bool resizable_;             // Can this storage be resized?
  at::Allocator* allocator_;   // Allocator used for this storage

 public:
  // Construction with pre-allocated memory
  StorageImpl(
      use_byte_size_t,
      SymInt size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);

  // Construction with fresh allocation
  StorageImpl(
      use_byte_size_t,
      const SymInt& size_bytes,
      at::Allocator* allocator,
      bool resizable);

  // Access
  const at::DataPtr& data_ptr() const;
  at::DataPtr& mutable_data_ptr();
  const void* data() const;
  void* mutable_data();

  // Size
  size_t nbytes() const;
  SymInt sym_nbytes() const;
  void set_nbytes(size_t size_bytes);

  // Metadata
  bool resizable() const;
  at::Allocator* allocator() const;
  Device device() const;
  DeviceType device_type() const;

  // Lifecycle
  void reset();                 // Clear data, set size to 0
  void release_resources() override;  // intrusive_ptr cleanup
};
```

**Reference Counting**:

```cpp
// StorageImpl inherits from intrusive_ptr_target
// which provides atomic refcounting

c10::intrusive_ptr<StorageImpl> storage1 = /* create */;
// refcount = 1

c10::intrusive_ptr<StorageImpl> storage2 = storage1;
// refcount = 2

storage1.reset();
// refcount = 1

storage2.reset();
// refcount = 0 → StorageImpl destroyed → DataPtr destroyed → deleter called
```

### 4. CPUAllocator - Default CPU Allocator

**File**: [reference/pytorch/c10/core/CPUAllocator.h:41-60](reference/pytorch/c10/core/CPUAllocator.h)

**Purpose**: Standard malloc/free based allocator for CPU tensors.

**Interface**:

```cpp
// Get the global CPU allocator
C10_API at::Allocator* GetCPUAllocator();

// Override the CPU allocator (rarely needed)
C10_API void SetCPUAllocator(at::Allocator* alloc, uint8_t priority = 0);

// Get default implementations
C10_API at::Allocator* GetDefaultCPUAllocator();
C10_API at::Allocator* GetDefaultMobileCPUAllocator();
```

**Implementation** (conceptual):

```cpp
class DefaultCPUAllocator : public Allocator {
 public:
  DataPtr allocate(size_t n) override {
    void* ptr = nullptr;
    if (n > 0) {
      ptr = c10::alloc_cpu(n);  // Aligned malloc
      if (!ptr) {
        throw std::bad_alloc();
      }
    }
    return DataPtr(ptr, ptr, &c10::free_cpu, Device(DeviceType::CPU));
  }

  DeleterFnPtr raw_deleter() const override {
    return &c10::free_cpu;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    default_copy_data(dest, src, count);  // Uses memcpy
  }
};
```

**Memory Alignment**:
- CPU allocations are typically aligned to 64 bytes for cache efficiency
- SIMD operations often require 16/32/64-byte alignment

### 5. MPSAllocator - Metal Performance Shaders (Apple GPU)

**File**: [reference/pytorch/aten/src/ATen/mps/MPSAllocatorInterface.h:15-68](reference/pytorch/aten/src/ATen/mps/MPSAllocatorInterface.h)

**Purpose**: Allocator for Apple's Metal GPU, using MTLBuffer.

**Interface**:

```cpp
class IMPSAllocator : public c10::Allocator {
 public:
  // Standard allocator methods (inherited)
  virtual DataPtr allocate(size_t n) = 0;

  // MPS-specific methods
  virtual void emptyCache() const = 0;
  virtual void freeInactiveBuffers() const = 0;

  // Buffer metadata
  virtual ssize_t getUnalignedBufferSize(const void* ptr) const = 0;
  virtual IntArrayRef getBufferShape(const void* ptr) const = 0;
  virtual id_t getBufferId(const void* ptr) const = 0;
  virtual void setBufferShape(const void* ptr, const IntArrayRef& shape) const = 0;

  // Shared memory support (unified memory)
  virtual bool isSharedBuffer(const void* ptr) const = 0;
  virtual bool isSharedStorageSupported() const = 0;
  virtual std::pair<const void*, uint32_t> getSharedBufferPtr(const void* ptr) const = 0;

  // Scalar buffers (for constants)
  virtual c10::DataPtr allocScalarBufferWithValue(void* value, size_t size) const = 0;

  // Memory statistics
  virtual size_t getTotalAllocatedMemory() const = 0;
  virtual size_t getCurrentAllocatedMemory() const = 0;
  virtual size_t getDriverAllocatedMemory() const = 0;
  virtual size_t getRecommendedMaxMemory() const = 0;

  // Watermark (memory pressure thresholds)
  virtual void setLowWatermarkRatio(double ratio) const = 0;
  virtual void setHighWatermarkRatio(double ratio) const = 0;
  virtual ssize_t getLowWatermarkValue() const = 0;
  virtual size_t getLowWatermarkLimit() const = 0;
  virtual size_t getHighWatermarkLimit() const = 0;

  // Event-based synchronization
  virtual bool recordEvents(c10::ArrayRef<const void*> buffers) const = 0;
  virtual bool waitForEvents(c10::ArrayRef<const void*> buffers) const = 0;
};
```

**Key Features**:
1. **Caching**: MPS allocator pools MTLBuffers to avoid allocation overhead
2. **Shared Memory**: Supports unified memory (CPU can directly access GPU buffers)
3. **Events**: Metal event-based synchronization for async execution
4. **Memory Pressure**: Watermarks trigger cache eviction when memory is low
5. **Buffer Metadata**: Stores shape information for debugging

**Callbacks**:

```cpp
class IMpsAllocatorCallback {
 public:
  enum class EventType {
    ALLOCATED,           // Buffer allocated from Metal
    RECYCLED,            // Buffer reused from pool
    FREED,               // Buffer returned to pool
    RELEASED,            // Buffer released to Metal
    ALLOCATION_FAILED    // Allocation failed
  };

  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// Register callback for profiling/debugging
REGISTER_MPS_ALLOCATOR_CALLBACK(name, callback_class);
```

### 6. CUDACachingAllocator - GPU Memory Pooling

**File**: [reference/pytorch/c10/cuda/CUDACachingAllocator.h:218](reference/pytorch/c10/cuda/CUDACachingAllocator.h)

**Purpose**: Caching allocator for CUDA to avoid expensive cudaMalloc/cudaFree calls.

**Strategy**:
1. **Pool**: Maintain free list of previously allocated buffers
2. **Binning**: Group buffers by size (power-of-2 bins)
3. **Reuse**: Satisfy allocations from pool when possible
4. **Expansion**: Call cudaMalloc only when pool is exhausted
5. **Release**: Return memory to pool on deallocation
6. **Pressure**: Free inactive buffers when memory pressure is high

**Example**:

```
First allocation: cudaMalloc(1024) → buffer A
Deallocation: buffer A → free pool
Second allocation (1024): Reuse buffer A from pool (no cudaMalloc!)
```

**Benefits**:
- **Performance**: cudaMalloc is slow (~10-100 μs), pooling reduces to ~1 μs
- **Fragmentation**: Binning strategy reduces external fragmentation
- **Stream Ordering**: Respects CUDA stream semantics for safety

## Data Flow

### Memory Allocation Lifecycle

```
1. User: torch.randn(1000)
   │
   ▼
2. TensorImpl constructor:
   - Determine size: 1000 * sizeof(float) = 4000 bytes
   - Get allocator: GetAllocator(DeviceType::CPU)
   │
   ▼
3. Create StorageImpl:
   StorageImpl(use_byte_size_t{}, 4000, cpu_allocator, resizable=true)
   │
   ▼
4. StorageImpl calls allocator->allocate(4000)
   │
   ▼
5. CPUAllocator::allocate(4000):
   - ptr = aligned_alloc(64, 4000)  // 64-byte aligned
   - return DataPtr(ptr, ptr, &free_cpu, Device(CPU))
   │
   ▼
6. StorageImpl stores DataPtr
   │
   ▼
7. TensorImpl wraps StorageImpl in intrusive_ptr
   │
   ▼
8. Return Tensor to user
```

### Memory Deallocation Lifecycle

```
1. Tensor goes out of scope
   │
   ▼
2. TensorImpl destructor:
   - storage_.reset() [intrusive_ptr<StorageImpl>]
   - Decrements StorageImpl refcount
   │
   ▼
3. If refcount == 0:
   - StorageImpl::~StorageImpl()
   - data_ptr_.~DataPtr()
   │
   ▼
4. DataPtr destructor:
   - calls deleter_(context_)
   │
   ▼
5. For CPU: free_cpu(ptr) → free(ptr)
   For CUDA: cuda_deleter(ptr) → return to pool or cudaFree
   For MPS: mps_deleter(ptr) → release MTLBuffer
```

### Caching Allocator Flow (CUDA Example)

```
Allocation Request (1024 bytes):
   │
   ▼
Check cache:
   - Lookup 1024-byte bin in free pool
   │
   ├─ Cache hit? → Return cached buffer (fast path)
   │
   └─ Cache miss?
      │
      ▼
   Call cudaMalloc(1024) (slow path)
      │
      ▼
   Return DataPtr with custom deleter:
      deleter = [pool](void* ptr) {
        pool->return_to_cache(ptr, 1024);
      }

Deallocation:
   │
   ▼
Custom deleter called:
   - Instead of cudaFree(ptr), return to cache
   - buffer → 1024-byte bin in free pool
```

## Code Examples

### Example 1: Custom Allocator

```cpp
#include <c10/core/Allocator.h>

// Simple logging allocator that wraps another allocator
class LoggingAllocator : public c10::Allocator {
 private:
  c10::Allocator* base_allocator_;

 public:
  explicit LoggingAllocator(c10::Allocator* base) : base_allocator_(base) {}

  c10::DataPtr allocate(size_t n) override {
    std::cout << "Allocating " << n << " bytes" << std::endl;
    auto data_ptr = base_allocator_->allocate(n);
    std::cout << "Allocated at " << data_ptr.get() << std::endl;
    return data_ptr;
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return base_allocator_->raw_deleter();
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::cout << "Copying " << count << " bytes" << std::endl;
    base_allocator_->copy_data(dest, src, count);
  }
};

// Usage
auto cpu_alloc = c10::GetCPUAllocator();
LoggingAllocator logging_alloc(cpu_alloc);

auto data_ptr = logging_alloc.allocate(1024);
// Output: Allocating 1024 bytes
//         Allocated at 0x7f8a4c000000
```

### Example 2: Manual Storage Management

```cpp
#include <c10/core/StorageImpl.h>

// Create storage manually
auto allocator = c10::GetCPUAllocator();
auto storage = c10::make_intrusive<c10::StorageImpl>(
    c10::StorageImpl::use_byte_size_t{},
    c10::SymInt(4096),  // 4 KB
    allocator,
    /*resizable=*/true);

// Access data
void* ptr = storage->mutable_data();
std::memset(ptr, 0, 4096);

// Storage is reference counted
auto storage2 = storage;  // refcount = 2
storage.reset();          // refcount = 1
storage2.reset();         // refcount = 0 → memory freed
```

### Example 3: Inspecting Memory Usage

```python
import torch

# CPU memory
x = torch.randn(1000, 1000)
print(f"Storage size: {x.storage().size()} elements")
print(f"Storage bytes: {x.storage().size() * x.element_size()} bytes")
print(f"Data pointer: {x.data_ptr()}")

# CUDA memory (if available)
if torch.cuda.is_available():
    y = torch.randn(1000, 1000, device='cuda')
    print(f"CUDA allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"CUDA reserved: {torch.cuda.memory_reserved()} bytes")

    # Empty cache to return memory to CUDA
    torch.cuda.empty_cache()
```

### Example 4: MPS Memory Management

```python
import torch

if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000, device='mps')

    # Get MPS allocator
    from torch.mps import set_per_process_memory_fraction, empty_cache

    # Limit memory usage to 80% of available
    set_per_process_memory_fraction(0.8)

    # Free inactive buffers
    empty_cache()

    # Get memory stats
    print(f"MPS current memory: {torch.mps.current_allocated_memory()}")
    print(f"MPS driver memory: {torch.mps.driver_allocated_memory()}")
```

## MLX Porting Considerations

### What MLX Already Provides

MLX has its own memory management:
- **Unified Memory**: CPU and GPU share address space on Apple Silicon
- **Automatic Management**: Reference counting for `mlx::array`
- **Metal Buffers**: Direct integration with MTLBuffer

### What Needs Adaptation

1. **Allocator Abstraction**:
   - PyTorch: Complex multi-device allocator hierarchy
   - MLX: Simpler unified memory model
   - **Action**: MLX can use simpler allocator (unified memory means no CPU↔GPU transfers)

2. **Caching Strategy**:
   - PyTorch: Explicit caching allocators for CUDA/MPS
   - MLX: May benefit from similar pooling for Metal buffers
   - **Recommendation**: Profile first - unified memory may reduce need for caching

3. **Memory Pressure Handling**:
   - PyTorch: Watermarks, cache eviction strategies
   - MLX: Can leverage Metal's memory pressure APIs
   - **Action**: Implement memory pressure callbacks for iOS/macOS

4. **Reference Counting**:
   - PyTorch: `intrusive_ptr` for atomic refcounting
   - MLX: Likely uses `std::shared_ptr` or similar
   - **Action**: Both approaches work; intrusive_ptr has lower overhead

### Metal-Specific Opportunities

1. **Unified Memory Benefits**:
   - No separate CPU/GPU allocators needed
   - No explicit data transfers
   - **Simplification**: Single allocator can handle both CPU and GPU access

2. **MTLHeap for Allocation**:
   - Use `MTLHeap` for sub-allocating Metal buffers
   - Similar to CUDA caching allocator
   - **Benefit**: Reduce MTLBuffer allocation overhead

3. **Metal Resource Hazard Tracking**:
   - Metal tracks resource dependencies automatically
   - Less manual synchronization needed than CUDA
   - **Advantage**: Simpler than CUDA stream management

4. **Shared Storage Mode**:
   - `MTLResourceStorageModeShared`: CPU-visible GPU memory
   - Zero-copy between CPU and GPU
   - **Use Case**: Weights that are read-only during inference

### Recommendations for MLX

1. **Single Unified Allocator**:
   - Don't replicate PyTorch's separate CPU/CUDA allocators
   - One allocator with unified memory
   - Device parameter becomes less relevant

2. **Buffer Pooling**:
   - Implement MTLBuffer pooling similar to CUDACachingAllocator
   - Bin by size, reuse buffers
   - Measure impact first - may not be needed

3. **Memory Limits**:
   - Respect `recommendedMaxWorkingSetSize` from Metal
   - Implement watermark-based cache eviction
   - Respond to memory pressure notifications

4. **Alignment**:
   - Metal requires specific alignment for certain operations
   - Ensure allocations are at least 16-byte aligned
   - Use 256-byte alignment for optimal performance

5. **Lazy Allocation**:
   - MLX's lazy evaluation can defer allocation
   - Only allocate when tensor is materialized
   - **Opportunity**: Further reduce memory footprint

## Critical File References

**Core Abstractions**:
- [c10/core/Allocator.h:39-138](reference/pytorch/c10/core/Allocator.h) - DataPtr class
- [c10/core/Allocator.h:178-232](reference/pytorch/c10/core/Allocator.h) - Allocator interface
- [c10/core/StorageImpl.h:51-149](reference/pytorch/c10/core/StorageImpl.h) - StorageImpl class
- [c10/util/intrusive_ptr.h](reference/pytorch/c10/util/intrusive_ptr.h) - Reference counting

**CPU Allocator**:
- [c10/core/CPUAllocator.h:41-60](reference/pytorch/c10/core/CPUAllocator.h) - CPU allocator interface
- [c10/core/CPUAllocator.cpp](reference/pytorch/c10/core/CPUAllocator.cpp) - CPU allocator implementation
- [c10/core/alignment.h](reference/pytorch/c10/core/alignment.h) - Memory alignment utilities

**MPS (Metal) Allocator**:
- [aten/src/ATen/mps/MPSAllocatorInterface.h:15-68](reference/pytorch/aten/src/ATen/mps/MPSAllocatorInterface.h) - MPS allocator interface
- [aten/src/ATen/mps/MPSAllocator.h](reference/pytorch/aten/src/ATen/mps/MPSAllocator.h) - MPS allocator implementation
- [aten/src/ATen/mps/MPSAllocator.mm](reference/pytorch/aten/src/ATen/mps/MPSAllocator.mm) - Metal buffer pooling

**CUDA Allocator** (Reference):
- [c10/cuda/CUDACachingAllocator.h:218](reference/pytorch/c10/cuda/CUDACachingAllocator.h) - CUDA caching allocator interface
- [c10/cuda/CUDACachingAllocator.cpp](reference/pytorch/c10/cuda/CUDACachingAllocator.cpp) - CUDA pooling implementation

## Next Steps

1. Study **02-OPERATORS/operator-overview.md** to see how operators use allocators
2. Read **metal-mps-backend.md** for detailed Metal integration patterns
3. Examine **autograd-overview.md** to understand memory management during backward pass
4. For MLX porting, decide on memory pooling strategy (profile first to see if needed with unified memory)

This completes the foundation layer documentation. The next phase covers operator implementation and registration.
