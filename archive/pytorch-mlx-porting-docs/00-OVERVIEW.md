# PyTorch Architecture Overview

## Purpose

This document provides a high-level architectural overview of PyTorch, designed specifically to facilitate porting PyTorch's functionality to MLX (Apple's Metal-based ML framework). Understanding PyTorch's layered architecture is critical for making informed decisions about what to port, how to adapt it, and where MLX's existing capabilities can be leveraged.

## PyTorch Layered Architecture

PyTorch is organized as a multi-layered system, with each layer building on the abstractions provided by the layer below:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API Layer                         │
│                        torch/*.py                               │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │   nn.Module  │  Optimizers  │ DataLoaders  │   Utilities  │ │
│  │  (torch/nn/) │ (torch/optim)│ (torch/utils)│              │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Python Bindings (pybind11)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    torch/csrc/ (C++ Layer)                       │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │   Autograd   │      JIT     │ Distributed  │   Inductor   │ │
│  │   Engine     │  TorchScript │     DDP      │   Compiler   │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ATen (Tensor Library)                         │
│                    aten/src/ATen/                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            Dispatcher & Operator Registry                   │ │
│  │         (Runtime backend/device routing)                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐ │
│  │   CPU    │   CUDA   │    MPS   │   XLA    │   Vulkan     │ │
│  │  Kernels │  Kernels │  (Metal) │  Kernels │   Kernels    │ │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘ │
│           2,666+ Operators (native_functions.yaml)              │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  c10 (Core Tensor Library)                       │
│                      c10/core/                                   │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │  TensorImpl  │   Storage    │ DispatchKey  │  Device      │ │
│  │  (metadata)  │  (data ptr)  │    (enum)    │  (CPU/GPU)   │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
│        Minimal dependencies, platform-independent core          │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### Layer 1: c10 (Core)
**Location**: [c10/core/](reference/pytorch/c10/core/)
**Purpose**: Minimal, platform-independent tensor core

**Key Components**:
- `TensorImpl`: The internal tensor representation (shape, strides, dtype, device)
- `Storage`: Reference-counted memory buffer abstraction
- `DispatchKey`: Enum identifying which backend/kernel to dispatch to
- `Device`: Device abstraction (CPU, CUDA, MPS, XLA, etc.)
- `ScalarType`: Data type enumeration (float32, int64, etc.)
- `Allocator`: Memory allocation interface

**Philosophy**: No heavy dependencies (no protobuf, no backend-specific code). Pure abstractions.

**MLX Relevance**: MLX's `mlx::array` is conceptually similar to `c10::TensorImpl`. Understanding PyTorch's separation of metadata (TensorImpl) from data (Storage) is crucial.

### Layer 2: ATen (A Tensor Library)
**Location**: [aten/src/ATen/](reference/pytorch/aten/src/ATen/)
**Purpose**: Tensor operations and multi-backend dispatch

**Key Components**:
- **Operator Registry**: 2,666+ operators defined in `native_functions.yaml`
- **Dispatcher**: Runtime routing system that selects the correct kernel based on tensor device/dtype
- **Native Implementations**: Backend-specific kernels
  - [aten/src/ATen/native/cpu/](reference/pytorch/aten/src/ATen/native/cpu/) - CPU kernels
  - [aten/src/ATen/native/cuda/](reference/pytorch/aten/src/ATen/native/cuda/) - CUDA kernels
  - [aten/src/ATen/native/mps/](reference/pytorch/aten/src/ATen/native/mps/) - Metal Performance Shaders (Apple GPU)
- **Code Generation**: [torchgen/](reference/pytorch/torchgen/) generates C++/Python bindings from YAML

**Dispatch Flow**:
```
User calls torch.add(tensor_a, tensor_b)
    ↓
Dispatcher extracts DispatchKeySet from tensors
    ↓
Lookup operator "add" in registry
    ↓
Query dispatch table with keys (e.g., CPU, AutogradCPU)
    ↓
Execute corresponding kernel (e.g., cpu_add)
```

**MLX Relevance**: MLX also has multi-backend dispatch (CPU/GPU). PyTorch's `MPS` backend (Metal) is directly relevant as a reference for Metal kernel implementations.

### Layer 3: torch/csrc (C++ Extensions)
**Location**: [torch/csrc/](reference/pytorch/torch/csrc/)
**Purpose**: Advanced features requiring C++ but not core to tensors

**Key Subsystems**:
- **Autograd** ([torch/csrc/autograd/](reference/pytorch/torch/csrc/autograd/)): Automatic differentiation engine
  - Tape-based gradient computation
  - Computational graph (Nodes, Edges)
  - Backward pass execution
- **JIT** ([torch/csrc/jit/](reference/pytorch/torch/csrc/jit/)): TorchScript compilation
- **Distributed** ([torch/csrc/distributed/](reference/pytorch/torch/csrc/distributed/)): Multi-node training (DDP, RPC)
- **Inductor** ([torch/csrc/inductor/](reference/pytorch/torch/csrc/inductor/)): Newer compiler backend

**MLX Relevance**: MLX has `mlx::grad` for automatic differentiation. Studying PyTorch's autograd architecture will help understand tradeoffs between tape-based and transform-based AD.

### Layer 4: Python API
**Location**: [torch/](reference/pytorch/torch/) (Python modules)
**Purpose**: User-facing Python API

**Key Modules**:
- **nn** ([torch/nn/](reference/pytorch/torch/nn/)): Neural network layers
  - `nn.Module`: Base class for all layers
  - `nn.Parameter`: Trainable weights
  - Layers: Linear, Conv2d, BatchNorm, etc.
- **optim** ([torch/optim/](reference/pytorch/torch/optim/)): Optimizers (SGD, Adam, etc.)
- **utils**: DataLoaders, checkpointing, etc.
- **functional**: Stateless function versions of operations

**MLX Relevance**: MLX's Python API is still evolving. PyTorch's `nn.Module` design provides a proven pattern for hierarchical model composition.

## Critical Architectural Patterns

### 1. Dispatch System
**The Core Innovation**: PyTorch's dispatch system allows one operator definition to route to different implementations based on:
- Device (CPU, CUDA, MPS, XLA)
- Dtype (float32, int64, etc.)
- Layout (strided, sparse, nested)
- Autograd mode (training vs inference)

**How It Works**:
```cpp
// Define operator schema
TORCH_LIBRARY(aten, m) {
  m.def("add.Tensor(Tensor self, Tensor other) -> Tensor");
}

// Implement for CPU
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", &cpu_add);
}

// Implement for CUDA
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("add.Tensor", &cuda_add);
}

// Implement for Metal (MPS)
TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("add.Tensor", &mps_add);
}
```

**MLX Approach**: MLX uses a different but related approach with lazy evaluation and graph compilation. Understanding PyTorch's dispatch will help identify where MLX's approach differs.

### 2. View Semantics & Copy-on-Write
**Concept**: Multiple tensors can share the same underlying storage buffer, differing only in metadata (offset, shape, strides).

```
Original Tensor:                     View (e.g., tensor[:, 0]):
┌─────────────────┐                 ┌─────────────────┐
│ TensorImpl      │                 │ TensorImpl      │
│  sizes: [4, 5]  │                 │  sizes: [4]     │
│  strides: [5,1] │                 │  strides: [5]   │
│  offset: 0      │                 │  offset: 0      │
│  storage: ─────┼─────┐            │  storage: ─────┼─────┐
└─────────────────┘     │            └─────────────────┘     │
                        ▼                                    │
                  ┌─────────────────┐                        │
                  │    Storage      │◄───────────────────────┘
                  │  data: [...]    │
                  │  refcount: 2    │
                  └─────────────────┘
```

**MLX Relevance**: MLX also uses view semantics. PyTorch's approach to tracking views for autograd is worth studying.

### 3. Autograd Graph
**Concept**: During the forward pass, PyTorch builds a computational graph of operations. During backward, it traverses this graph in reverse to compute gradients.

```
Forward Pass:                    Backward Pass:
    x (leaf)                        grad_x ←
    │                                       │
    ▼                                       │
  matmul(x, W)  ────────►  MmBackward  ────┘
    │                           │
    ▼                           │
   relu(·)     ────────►  ReluBackward
    │                           │
    ▼                           │
   y (output)                 grad_y (input)
```

**MLX Relevance**: MLX uses function transformations (`mlx::grad`) rather than explicit graph construction. Both approaches have tradeoffs worth understanding.

### 4. Operator Codegen from YAML
**Concept**: Instead of hand-writing 2,666 operators × N backends, PyTorch defines operators once in YAML and generates code.

**Example** from `native_functions.yaml`:
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
    MPS: add_mps
  tags: [core, pointwise]
```

From this, `torchgen` generates:
- C++ function declarations
- Python bindings
- Dispatch registration
- Autograd derivatives (if defined)

**MLX Relevance**: This pattern could be adapted for MLX to maintain API compatibility with PyTorch while generating Metal-optimized implementations.

## Subsystem Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Application                         │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ torch.nn.Module (layers, models)                                 │
│   depends on: Parameters, autograd, operators                    │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ torch.autograd (gradient computation)                            │
│   depends on: operators, computational graph                     │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ ATen Operators (add, matmul, conv2d, etc.)                       │
│   depends on: dispatcher, tensors                                │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ Dispatcher (runtime routing)                                     │
│   depends on: DispatchKey, operator registry                     │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ c10 Core (TensorImpl, Storage, Device, types)                    │
│   no dependencies (minimal core)                                 │
└──────────────────────────────────────────────────────────────────┘
```

**Porting Implication**: You can port PyTorch bottom-up:
1. Start with c10-equivalent (tensor core)
2. Add essential operators
3. Add dispatch/backend infrastructure
4. Add autograd
5. Add nn.Module system
6. Add optimizers and training utilities

## Operator Coverage

PyTorch defines **2,666 operators** in [aten/src/ATen/native/native_functions.yaml](reference/pytorch/aten/src/ATen/native/native_functions.yaml). Not all are equally important for initial MLX porting.

### Tier 1: Critical Operators (~50)
These are essential for basic neural networks:
- **Arithmetic**: add, sub, mul, div, matmul, mm, bmm
- **Activations**: relu, gelu, sigmoid, tanh, softmax
- **Linear Algebra**: mv, dot, addmm
- **Reductions**: sum, mean, max, min, argmax
- **Convolution**: conv1d, conv2d, conv3d
- **Indexing**: gather, scatter, index_select, where

### Tier 2: Important Operators (~200)
Follow patterns, less critical but commonly used:
- Shape manipulation: view, reshape, transpose, permute
- Element-wise: exp, log, sqrt, pow, abs
- Comparison: eq, lt, gt
- Statistical: var, std, median

### Tier 3: Long Tail (~2,400)
Specialized variants, backward compatibility, edge cases. Implement on-demand.

## Metal Performance Shaders (MPS) Backend

PyTorch already has a Metal backend, which is **highly relevant** for MLX porting:

**Location**:
- [aten/src/ATen/mps/](reference/pytorch/aten/src/ATen/mps/) - MPS integration layer
- [aten/src/ATen/native/mps/](reference/pytorch/aten/src/ATen/native/mps/) - Metal kernel implementations

**What to Study**:
- How PyTorch wraps `MPSGraph` APIs
- Memory management with Metal buffers
- Synchronization between CPU and GPU
- Kernel implementation patterns

**MLX Advantage**: MLX is designed for Metal from the ground up, so it should handle this more elegantly than PyTorch's bolt-on approach.

## Reading Guide

To understand PyTorch for MLX porting, read the documentation in this order:

### Week 1: Foundations
1. **01-FOUNDATIONS/tensor-core.md** - Understand TensorImpl, Storage, metadata
2. **01-FOUNDATIONS/dispatch-system.md** - Learn the dispatch mechanism
3. **01-FOUNDATIONS/type-system.md** - Device, dtype, layout abstractions
4. **01-FOUNDATIONS/memory-model.md** - Memory management, allocators

### Week 2: Operators & Autograd
5. **02-OPERATORS/operator-overview.md** - YAML schema, codegen workflow
6. **02-OPERATORS/operator-reference/arithmetic.md** - Study key operators
7. **03-AUTOGRAD/autograd-overview.md** - Automatic differentiation architecture
8. **03-AUTOGRAD/computational-graph.md** - Graph construction and traversal

### Week 3: Neural Networks
9. **04-NEURAL-NETWORKS/module-system.md** - nn.Module pattern
10. **04-NEURAL-NETWORKS/layers-reference/** - Study common layers
11. **05-TRAINING/optimizer-base.md** - Optimizer architecture
12. **05-TRAINING/optimizers/adam.md** - Modern optimizer example

### Week 4: Metal & Porting Strategy
13. **06-BACKENDS/metal-mps-backend.md** - PyTorch's Metal implementation
14. **08-PORTING-GUIDE/mlx-mapping.md** - PyTorch → MLX concept mapping
15. **08-PORTING-GUIDE/implementation-roadmap.md** - Phased porting plan

### Optional: Advanced Topics
16. **07-ADVANCED/** - Compilation, quantization, sparse tensors (as needed)

## Key Differences: PyTorch vs MLX

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Execution** | Eager + JIT compilation | Lazy evaluation + graph compilation |
| **Memory** | Discrete (CPU/GPU separate) | Unified memory (Apple Silicon) |
| **Backend** | Multi-platform (CPU, CUDA, Metal, etc.) | Metal-only (Apple GPUs) |
| **Autograd** | Tape-based (build graph during forward) | Transform-based (`grad` function transform) |
| **Language** | C++ core, Python API | C++ core, Python & Swift APIs |
| **Operator Count** | 2,666+ operators | Smaller, focused set |
| **Target** | Research & production on any hardware | Apple Silicon optimization |

## Critical Files Reference

**Core Abstractions**:
- [c10/core/TensorImpl.h](reference/pytorch/c10/core/TensorImpl.h) - Tensor internal representation
- [c10/core/Storage.h](reference/pytorch/c10/core/Storage.h) - Memory buffer abstraction
- [c10/core/DispatchKey.h](reference/pytorch/c10/core/DispatchKey.h) - Backend dispatch keys

**Operator System**:
- [aten/src/ATen/native/native_functions.yaml](reference/pytorch/aten/src/ATen/native/native_functions.yaml) - All 2,666 operators defined here
- [aten/src/ATen/core/dispatch/Dispatcher.h](reference/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h) - Runtime dispatcher

**Autograd**:
- [torch/csrc/autograd/variable.h](reference/pytorch/torch/csrc/autograd/variable.h) - Variable and AutogradMeta
- [torch/csrc/autograd/function.h](reference/pytorch/torch/csrc/autograd/function.h) - Gradient functions (Nodes)
- [torch/csrc/autograd/engine.h](reference/pytorch/torch/csrc/autograd/engine.h) - Backward pass engine

**Neural Networks**:
- [torch/nn/modules/module.py](reference/pytorch/torch/nn/modules/module.py) - nn.Module base class
- [torch/nn/parameter.py](reference/pytorch/torch/nn/parameter.py) - Parameter wrapper

**Metal Backend**:
- [aten/src/ATen/mps/](reference/pytorch/aten/src/ATen/mps/) - MPS integration
- [aten/src/ATen/native/mps/](reference/pytorch/aten/src/ATen/native/mps/) - Metal kernels

## Next Steps

1. Read through the Foundation documents (01-FOUNDATIONS/)
2. Study the Metal backend implementation (06-BACKENDS/metal-mps-backend.md)
3. Review the MLX mapping guide (08-PORTING-GUIDE/mlx-mapping.md)
4. Identify which PyTorch operators MLX already supports
5. Create a porting roadmap based on your specific use case (LLMs, computer vision, etc.)

This documentation is designed to be read non-linearly. Each document is self-contained with cross-references to related topics.
