# torch.compile and TorchDynamo

## Overview

`torch.compile` is PyTorch's flagship model optimization API, introduced in PyTorch 2.0. It provides JIT compilation that can significantly improve model performance by capturing and optimizing computational graphs at runtime.

**Architecture Stack**:
```
torch.compile (User API)
        ↓
TorchDynamo (Graph Capture via Python bytecode transformation)
        ↓
AOTAutograd (Ahead-of-time autograd decomposition)
        ↓
Backend Compiler (inductor, triton, etc.)
        ↓
Optimized Kernel Code
```

---

## torch.compile API

### Basic Usage

```python
import torch

# Method 1: Decorator
@torch.compile
def my_function(x):
    return torch.sin(x) + torch.cos(x)

# Method 2: Direct call
model = MyModel()
optimized_model = torch.compile(model)

# Method 3: As context manager (via wrapper)
compiled_fn = torch.compile(lambda x: model(x))
```

### Full Signature

```python
torch.compile(
    model: Callable = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | Callable = "inductor",
    mode: str | None = None,
    options: dict | None = None,
    disable: bool = False,
) -> Callable
```

**Source**: `torch/__init__.py:2559-2673`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Callable | None | Function or nn.Module to optimize |
| `fullgraph` | bool | False | Require entire function as single graph |
| `dynamic` | bool/None | None | Enable dynamic shape tracing |
| `backend` | str/Callable | "inductor" | Compilation backend |
| `mode` | str/None | None | Optimization mode preset |
| `options` | dict/None | None | Backend-specific options |
| `disable` | bool | False | Make torch.compile a no-op |

### Modes

| Mode | Description |
|------|-------------|
| `"default"` | Balanced performance/overhead |
| `"reduce-overhead"` | Minimize Python overhead (CUDA graphs) |
| `"max-autotune"` | Maximum kernel tuning (Triton templates, CUDA graphs) |
| `"max-autotune-no-cudagraphs"` | Max tuning without CUDA graphs |

**Example**:
```python
# Default mode - balanced
model = torch.compile(model)

# Reduce overhead - small batches, inference
model = torch.compile(model, mode="reduce-overhead")

# Max performance - training, large models
model = torch.compile(model, mode="max-autotune")
```

### Dynamic Shapes

```python
# Auto-detect dynamism (default)
model = torch.compile(model, dynamic=None)

# Force dynamic shapes - avoid recompilations
model = torch.compile(model, dynamic=True)

# Force static shapes - maximum optimization
model = torch.compile(model, dynamic=False)
```

---

## TorchDynamo

TorchDynamo is the frontend that captures PyTorch operations by transforming Python bytecode at runtime.

### How Dynamo Works

1. **Bytecode Interception**: Dynamo intercepts Python frame execution
2. **Symbolic Tracing**: Executes code symbolically, tracking operations
3. **Guard Generation**: Creates runtime checks for assumptions
4. **Graph Construction**: Builds FX graph of operations
5. **Compilation**: Passes graph to backend compiler

**Source Files**:
- `torch/_dynamo/eval_frame.py` - Frame evaluation hook
- `torch/_dynamo/convert_frame.py` - Frame conversion
- `torch/_dynamo/symbolic_convert.py` - Symbolic execution
- `torch/_dynamo/guards.py` - Guard generation
- `torch/_dynamo/output_graph.py` - Graph construction

### Graph Breaks

A graph break occurs when Dynamo cannot trace through a piece of code. This splits the computation into multiple subgraphs.

**Common Causes**:
- Unsupported Python operations (e.g., `print()`)
- Data-dependent control flow
- Calling untraceable functions
- Accessing non-tensor data structures

**Example**:
```python
def fn(x):
    x = x + 1
    print(f"x = {x}")  # GRAPH BREAK - print is a side effect
    x = x * 2
    return x

# Results in two subgraphs:
# Graph 1: x = x + 1
# (break for print)
# Graph 2: x = x * 2
```

**Forcing Full Graph**:
```python
@torch.compile(fullgraph=True)
def fn(x):
    return x + 1

# Will raise error if any graph break occurs
```

### Guards

Guards are runtime checks that verify assumptions made during compilation.

**Guard Types**:
- **Tensor Guards**: Shape, dtype, device, stride
- **Global Guards**: Global variable values
- **Module Guards**: Module attribute checks
- **Shape Guards**: Symbolic shape relationships

**Example**:
```python
@torch.compile
def fn(x):
    return x + 1

# On first call with shape (32, 64):
# Guards generated:
# - x.shape == (32, 64)
# - x.dtype == torch.float32
# - x.device == 'cuda:0'
# - etc.
```

When guards fail, Dynamo recompiles with new assumptions.

---

## Backends

### Inductor (Default)

Inductor is PyTorch's native compiler backend that generates optimized kernels.

**Key Features**:
- Triton-based GPU kernel generation
- C++ CPU kernel generation
- Fusion of operations
- Memory optimization

**Example**:
```python
model = torch.compile(model, backend="inductor")
```

### Available Backends

```python
import torch._dynamo as dynamo

# List stable backends
print(dynamo.list_backends())
# ['inductor', 'eager', 'aot_eager', ...]

# List all backends (including experimental)
print(dynamo.list_backends(None))
```

| Backend | Description |
|---------|-------------|
| `inductor` | Default - generates optimized kernels |
| `eager` | Immediate execution (no optimization) |
| `aot_eager` | AOT decomposition + eager execution |
| `aot_ts` | AOT + TorchScript |
| `cudagraphs` | CUDA graph capture |

### Custom Backends

```python
from torch._dynamo import register_backend

@register_backend
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    gm: FX GraphModule containing the traced operations
    example_inputs: Sample inputs used during tracing
    """
    # Transform the graph
    # ...
    return gm.forward  # Return callable
```

---

## Compiler Utilities

### torch.compiler Module

```python
import torch.compiler

# Reset compilation caches
torch.compiler.reset()

# Allow function in graph (escape hatch)
torch.compiler.allow_in_graph(my_custom_fn)

# Check if compiling
if torch.compiler.is_compiling():
    pass  # In compilation context

# Disable compilation for a function
@torch.compiler.disable
def eager_only_fn(x):
    return x + 1
```

**Source**: `torch/compiler/__init__.py`

### Compilation Stances

Control compilation behavior dynamically:

```python
# Force eager mode
with torch.compiler.set_stance("force_eager"):
    result = compiled_model(x)  # Runs eagerly

# Fail on recompile
torch.compiler.set_stance("fail_on_recompile")

# Eager then compile (dynamic shapes)
torch.compiler.set_stance("eager_then_compile")
```

**Available Stances**:
| Stance | Description |
|--------|-------------|
| `"default"` | Normal compilation |
| `"force_eager"` | Ignore all torch.compile |
| `"eager_on_recompile"` | Eager if recompile needed |
| `"fail_on_recompile"` | Error on recompile |
| `"eager_then_compile"` | First call eager, then compile |

### Debugging

```python
# Enable logging
import torch._dynamo.config as config
config.verbose = True

# Or via environment variable
# TORCH_LOGS=+dynamo python script.py
# TORCH_LOGS=guards python script.py  # Debug guard failures

# Explain compilation
@torch.compile
def fn(x):
    return x + 1

torch._dynamo.explain(fn)(torch.randn(10))
```

---

## Configuration

### Dynamo Config

```python
import torch._dynamo.config as dynamo_config

# Recompile limit (default 8)
dynamo_config.recompile_limit = 8

# Cache size limit
dynamo_config.cache_size_limit = 64

# Suppress errors (fall back to eager)
dynamo_config.suppress_errors = False

# Verbose logging
dynamo_config.verbose = True
```

### Inductor Config

```python
import torch._inductor.config as inductor_config

# Kernel fusion
inductor_config.fusion_enabled = True

# Max autotune
inductor_config.max_autotune = True

# CUDA graphs
inductor_config.triton.cudagraphs = True

# Epilogue fusion (fuse pointwise ops)
inductor_config.epilogue_fusion = True
```

**List All Options**:
```python
print(torch._inductor.list_options())
```

---

## Performance Patterns

### Best Practices

1. **Warm-up Phase**: First call triggers compilation
```python
# Warm-up (compilation happens here)
with torch.no_grad():
    _ = model(sample_input)

# Actual inference (uses compiled code)
for batch in dataloader:
    output = model(batch)
```

2. **Avoid Graph Breaks**
```python
# Bad - causes graph break
def fn(x):
    print(x.shape)  # Graph break
    return x + 1

# Good - use torch operations
def fn(x):
    # No print in hot path
    return x + 1
```

3. **Handle Dynamic Shapes**
```python
# For variable sequence lengths
model = torch.compile(model, dynamic=True)

# Or mark specific dimensions as dynamic
from torch.export import Dim
batch = Dim("batch", min=1, max=128)
# Use with export, not compile
```

4. **CUDA Graphs for Inference**
```python
# Reduce overhead with CUDA graphs
model = torch.compile(model, mode="reduce-overhead")

# Mark iteration boundaries
for batch in dataloader:
    torch.compiler.cudagraph_mark_step_begin()
    output = model(batch)
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Slow first call | Expected - compilation overhead |
| Frequent recompiles | Check guards, use `dynamic=True` |
| Memory issues | Reduce `triton.cudagraphs`, check fusion |
| Graph breaks | Use `fullgraph=True` to debug |
| Wrong outputs | Check `options={"fallback_random": True}` |

---

## AOTAutograd

AOTAutograd (Ahead-of-Time Autograd) decomposes high-level operations into primitive operations before compilation.

**Benefits**:
- Exposes more fusion opportunities
- Enables gradient computation at compile time
- Simplifies backend implementation

**Source**: `torch/_functorch/aot_autograd.py`

```python
# AOT eager for debugging
model = torch.compile(model, backend="aot_eager")

# AOT with custom backend
from torch._functorch.aot_autograd import aot_function

def my_compiler(gm, example_inputs):
    return gm.forward

aot_fn = aot_function(fn, my_compiler)
```

---

## MLX Implications

### Comparison

| Feature | PyTorch torch.compile | MLX |
|---------|----------------------|-----|
| Compilation | JIT via Dynamo + backends | No JIT compilation |
| Graphs | FX graphs | Lazy evaluation graphs |
| Fusion | Kernel fusion via Inductor | Automatic fusion in lazy eval |
| Dynamic shapes | Supported via guards | Native dynamic shapes |
| CUDA graphs | Supported | N/A (no CUDA) |

### MLX Approach

MLX uses lazy evaluation instead of JIT compilation:

```python
import mlx.core as mx

def fn(x):
    return mx.sin(x) + mx.cos(x)

# Operations are lazy - graph is built automatically
x = mx.array([1, 2, 3])
y = fn(x)  # Graph built, not executed

# Explicit evaluation
mx.eval(y)  # Graph executed on Metal
```

**Key Differences**:
1. **No torch.compile equivalent**: MLX is always "compiled"
2. **Lazy evaluation**: Operations form graphs automatically
3. **Metal optimization**: Graph optimized for Apple Silicon
4. **No guards/recompilation**: Dynamic by nature

### Porting Considerations

```python
# PyTorch compiled code
@torch.compile
def pytorch_fn(x):
    return torch.sin(x) + torch.cos(x)

# MLX equivalent (no compilation needed)
def mlx_fn(x):
    return mx.sin(x) + mx.cos(x)

# Both achieve similar optimization via different mechanisms
```

For complex models:
- Remove `@torch.compile` decorators
- Rely on MLX's lazy evaluation for optimization
- Use `mx.eval()` strategically to control evaluation points

---

## Implementation Files

**Core API**:
- `torch/__init__.py:2559-2673` - `torch.compile` definition
- `torch/compiler/__init__.py` - Compiler utilities

**TorchDynamo**:
- `torch/_dynamo/__init__.py` - Main entry points
- `torch/_dynamo/eval_frame.py` - Frame evaluation
- `torch/_dynamo/convert_frame.py` - Frame conversion
- `torch/_dynamo/symbolic_convert.py` - Symbolic execution
- `torch/_dynamo/guards.py` - Guard generation
- `torch/_dynamo/output_graph.py` - Graph construction
- `torch/_dynamo/bytecode_transformation.py` - Bytecode manipulation

**Inductor**:
- `torch/_inductor/__init__.py` - Backend entry
- `torch/_inductor/compile_fx.py` - FX compilation
- `torch/_inductor/graph.py` - Graph representation
- `torch/_inductor/codegen/` - Code generation

**AOTAutograd**:
- `torch/_functorch/aot_autograd.py` - AOT decomposition

**Configuration**:
- `torch/_dynamo/config.py` - Dynamo configuration
- `torch/_inductor/config.py` - Inductor configuration

---

## Summary

`torch.compile` provides significant performance improvements through:
1. **Graph Capture**: TorchDynamo traces Python code
2. **Guard System**: Enables safe caching and reuse
3. **Backend Flexibility**: Inductor or custom backends
4. **Automatic Optimization**: Fusion, memory optimization, kernel tuning

For MLX porting:
- Replace with lazy evaluation patterns
- Remove compilation decorators
- Trust MLX's automatic graph optimization
- Focus on strategic `mx.eval()` placement
