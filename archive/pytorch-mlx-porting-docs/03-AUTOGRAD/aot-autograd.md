# AOT Autograd (Ahead-of-Time Autograd)

## Purpose

AOT Autograd captures both forward and backward computation graphs ahead of time, before execution. This enables:
- Whole-graph optimizations across forward and backward passes
- Custom compiler backend integration
- Memory-efficient activation checkpointing via intelligent graph partitioning
- Debugging and visualization of autograd behavior

AOT Autograd is a core component of PyTorch 2.0's compilation stack and underlies `torch.compile`.

**Reference Files:**
- `torch/_functorch/aot_autograd.py` - Core AOT implementation
- `torch/_functorch/compilers.py` - Built-in compiler backends
- `torch/_functorch/partitioners.py` - Graph partitioning strategies
- `functorch/compile/__init__.py` - Public API exports

---

## Overview

Traditional PyTorch uses **eager autograd** - gradients are computed on-the-fly during `backward()`. AOT Autograd instead:

1. **Traces** the forward function to capture an FX graph
2. **Differentiates** the FX graph symbolically to produce a backward graph
3. **Partitions** the combined (joint) graph into optimized forward/backward graphs
4. **Compiles** each graph with a user-specified backend
5. **Wraps** compiled graphs in `autograd.Function` for seamless integration

```
User Function
     │
     ▼
┌────────────────┐
│ make_fx trace  │ ──► Joint FX Graph (forward + backward)
└────────────────┘
     │
     ▼
┌────────────────┐
│  Partitioner   │ ──► Separate forward/backward graphs
└────────────────┘            │
     │                        │
     ▼                        ▼
┌────────────────┐    ┌────────────────┐
│  fw_compiler   │    │  bw_compiler   │
└────────────────┘    └────────────────┘
     │                        │
     ▼                        ▼
┌────────────────────────────────────────┐
│     autograd.Function wrapper          │
└────────────────────────────────────────┘
```

---

## Core API

### aot_function

Compiles a Python function with AOT Autograd.

```python
from functorch.compile import aot_function

def aot_function(
    fn: Callable,
    fw_compiler: Callable,           # Compiles forward graph
    bw_compiler: Optional[Callable] = None,  # Compiles backward (defaults to fw_compiler)
    partition_fn: Callable = default_partition,
    decompositions: Optional[dict] = None,
    num_params_buffers: int = 0,     # Number of params (not differentiated)
    hasher_type: str = "StaticShapeHasher",
    static_argnums: Optional[tuple[int]] = None,
    keep_inference_input_mutations: bool = False,
) -> Callable
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `fn` | Function to compile |
| `fw_compiler` | Receives forward FX graph, returns callable |
| `bw_compiler` | Receives backward FX graph, returns callable |
| `partition_fn` | Splits joint graph into forward/backward |
| `decompositions` | Operator decomposition rules |
| `num_params_buffers` | Leading args that are parameters (no grad tracking) |
| `static_argnums` | Arguments with static shapes (caching optimization) |

**Example:**

```python
from functorch.compile import aot_function
import torch.fx as fx

# Simple compiler that prints the graph
def my_compiler(fx_g: fx.GraphModule, example_inputs):
    print("Compiling graph:")
    print(fx_g.code)
    return fx_g  # Return compiled callable

def f(x):
    return x.cos().cos()

# Create AOT-compiled version
compiled_f = aot_function(
    f,
    fw_compiler=my_compiler,
    bw_compiler=my_compiler
)

# Use like normal - first call triggers compilation
x = torch.randn(3, requires_grad=True)
y = compiled_f(x)
y.sum().backward()  # Backward graph also compiled
```

### aot_module

Compiles an `nn.Module` with AOT Autograd.

```python
from functorch.compile import aot_module

def aot_module(
    mod: nn.Module,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[dict] = None,
    keep_inference_input_mutations: bool = False,
) -> nn.Module
```

**Example:**

```python
from functorch.compile import aot_module
from torchvision.models import resnet18

model = resnet18()
compiled_model = aot_module(
    model,
    fw_compiler=my_compiler,
    bw_compiler=my_compiler
)

# Forward/backward now use compiled graphs
output = compiled_model(torch.randn(1, 3, 224, 224))
output.sum().backward()
```

### aot_module_simplified

A simplified version for cases where you don't need the full autograd.Function wrapper.

```python
from functorch.compile import aot_module_simplified

compiled = aot_module_simplified(
    mod,
    example_inputs,
    fw_compiler=...,
    bw_compiler=...,
)
```

---

## Compiler Backends

AOT Autograd provides several built-in compiler backends.

### nop (No-op)

Returns the FX graph unchanged. Useful for debugging.

```python
from functorch.compile import nop

compiled_f = aot_function(f, fw_compiler=nop)
```

### ts_compile (TorchScript)

Compiles to TorchScript for optimized execution.

```python
from functorch.compile import ts_compile

compiled_f = aot_function(f, fw_compiler=ts_compile)
```

**Implementation:**
```python
@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps) -> Callable:
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    f = torch.jit.optimize_for_inference(f)
    return f
```

### print_compile

Prints the graph code, useful for debugging.

```python
from functorch.compile import print_compile

compiled_f = aot_function(f, fw_compiler=print_compile)
```

### draw_graph_compile

Visualizes the graph as SVG files.

```python
from functorch.compile import draw_graph_compile

compiled_f = aot_function(
    f,
    fw_compiler=draw_graph_compile("forward"),
    bw_compiler=draw_graph_compile("backward")
)
```

### debug_nop

A slow interpreter that validates traced metadata against real execution.

```python
from functorch.compile import debug_nop

compiled_f = aot_function(f, fw_compiler=debug_nop)
```

### Custom Compiler

Write your own compiler backend:

```python
from functorch.compile import make_boxed_compiler

@make_boxed_compiler
def my_custom_compiler(fx_g: fx.GraphModule, example_inputs):
    # Transform the graph
    # fx_g.graph contains the FX nodes
    # example_inputs are sample inputs for shape inference

    # Optionally modify the graph
    for node in fx_g.graph.nodes:
        # ... transformations ...
        pass
    fx_g.recompile()

    # Return a callable
    return fx_g  # or return optimized version
```

The `@make_boxed_compiler` decorator handles input/output boxing for autograd compatibility.

---

## Partitioners

Partitioners split the joint forward+backward graph into separate graphs, determining what gets saved for backward (activations) vs recomputed.

### default_partition

Simple partitioner that saves all forward outputs needed by backward.

```python
from functorch.compile import default_partition

compiled_f = aot_function(
    f,
    fw_compiler=my_compiler,
    partition_fn=default_partition
)
```

### min_cut_rematerialization_partition

**Memory-efficient partitioner** that uses min-cut algorithms to optimize the memory/compute tradeoff.

```python
from functorch.compile import min_cut_rematerialization_partition

compiled_f = aot_function(
    f,
    fw_compiler=my_compiler,
    partition_fn=min_cut_rematerialization_partition
)
```

**How it works:**

1. Analyzes the joint graph to find all nodes required for backward
2. Builds a cost model: memory cost (tensor sizes) vs compute cost (FLOPs)
3. Uses min-cut algorithm to find optimal partition:
   - Nodes above the cut → forward graph (activations saved)
   - Nodes below the cut → recomputed during backward
4. Cheap operations (element-wise ops) prefer recomputation
5. Expensive operations (matmuls, convolutions) prefer saving

**Key heuristics:**

```python
@dataclass
class OpTypes:
    fusible_ops: OrderedSet[Callable]         # Cheap, prefer recompute
    compute_intensive_ops: OrderedSet[Callable]  # Expensive, prefer save
    random_ops: OrderedSet[Callable]          # Must save (non-deterministic)
    view_ops: OrderedSet[Callable]            # Free, no memory cost
    recomputable_ops: OrderedSet[Callable]    # Safe to recompute
```

**Configuration options:**

```python
@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool      # Don't recompute if used much later
    ban_if_long_fusible_chains: bool # Avoid recomputing long chains
    ban_if_materialized_backward: bool
    ban_if_not_in_allowlist: bool
    ban_if_reduction: bool           # Reductions often memory-bound
```

### memory_efficient_fusion

Convenience wrapper combining min-cut partitioning with TorchScript compilation:

```python
from functorch.compile import memory_efficient_fusion

# Automatically uses min_cut_rematerialization_partition
compiled_f = memory_efficient_fusion(f)

# Also works with modules
compiled_model = memory_efficient_fusion(model)
```

---

## Graph Structure

### Joint Graph

The joint graph contains both forward and backward computation:

```python
def f(x):
    return x.cos().cos()

# Joint graph structure:
# Inputs:  [x, tangent_for_output]
# Outputs: [forward_output, gradient_for_x]

# Pseudocode:
def joint_graph(x, tangent):
    # Forward
    t1 = x.cos()
    out = t1.cos()

    # Backward (interleaved with forward)
    grad_t1 = tangent * (-t1.sin())
    grad_x = grad_t1 * (-x.sin())

    return out, grad_x
```

### Forward Graph

After partitioning, forward graph returns outputs + saved tensors:

```python
def forward_graph(x):
    t1 = x.cos()
    out = t1.cos()
    return out, t1, x  # Output + saved activations
```

### Backward Graph

Backward graph receives tangents + saved tensors:

```python
def backward_graph(tangent, saved_t1, saved_x):
    grad_t1 = tangent * (-saved_t1.sin())
    grad_x = grad_t1 * (-saved_x.sin())
    return grad_x
```

---

## Handling Mutations and Aliases

AOT Autograd carefully handles input mutations and output aliasing.

### Input Data Mutations

Functions that mutate inputs are converted to functional form:

```python
# Original
def f(x):
    x.mul_(2)
    return x.mul(3)

# After AOT Autograd transformation:
def compiled_forward(x):
    x_updated = x.mul(2)  # Mutation → new tensor
    out = x_updated.mul(3)
    return x_updated, out

# Wrapper performs the mutation after:
def wrapper(x):
    x_updated, out = compiled_forward(x)
    x.copy_(x_updated)  # Apply mutation
    return out
```

### Input Metadata Mutations

Shape/stride mutations (like `x.t_()`) are handled similarly:

```python
# Original
def f(x):
    x.t_()  # Transpose in-place
    return x.mul(3)

# After AOT Autograd:
def compiled_forward(x):
    x_updated = x.t()  # Metadata mutation → view
    out = x_updated.mul(3)
    return x_updated, out

def wrapper(x):
    x_updated, out = compiled_forward(x)
    x.as_strided_(x_updated)  # Apply metadata change
    return out
```

### Output Aliasing

Outputs that alias inputs or intermediates are regenerated:

```python
# Original
def f(x):
    out1 = x.t()            # Aliases input
    intermediate = x.mul(2)
    out2 = intermediate.view(-1)  # Aliases intermediate
    return out1, out2

# Compiled forward returns intermediates:
def compiled_forward(x):
    out1 = x.t()
    intermediate = x.mul(2)
    out2 = intermediate.view(-1)
    return out1, out2, intermediate

# Wrapper regenerates aliases:
def wrapper(x):
    out1, out2, intermediate = compiled_forward(x)
    out1_regen = out1._view_func(x)        # Regenerate from input
    out2_regen = out2._view_func(intermediate)  # Regenerate from intermediate
    return out1_regen, out2_regen
```

### Synthetic Bases

When mutated inputs alias each other, AOT Autograd creates "synthetic bases":

```python
# Original
def f(x, x_view):  # x_view = x.view(-1)
    x.mul_(2)
    return x * x_view

# Wrapper creates synthetic base:
def wrapper(x, x_view):
    base = merge_view_inputs(x, x_view)  # Single base tensor
    x_updated, out = compiled_forward(base)
    x.copy_(x_updated)  # Mutation propagates to x_view
    return out
```

---

## Integration with torch.compile

AOT Autograd is the autograd component of `torch.compile`:

```
torch.compile
     │
     ├── TorchDynamo (graph capture)
     │        │
     │        ▼
     │   FX Graph (forward only)
     │        │
     │        ▼
     ├── AOT Autograd (backward generation)
     │        │
     │        ▼
     │   Forward + Backward graphs
     │        │
     │        ▼
     └── Inductor (code generation)
              │
              ▼
         Optimized kernels
```

When using `torch.compile`, AOT Autograd is invoked automatically:

```python
@torch.compile
def f(x):
    return x.cos().cos()

# Internally uses AOT Autograd to generate backward graph
```

---

## Decompositions

Decompositions break complex operators into simpler ones for easier optimization:

```python
from torch._decomp import get_decompositions

default_decompositions = {
    torch.ops.aten.gelu_backward,
    torch.ops.aten.leaky_relu_backward,
    torch.ops.aten.sigmoid_backward,
    torch.ops.aten.tanh_backward,
    torch.ops.aten.silu_backward,
    # ... more
}

decompositions = get_decompositions(default_decompositions)

compiled_f = aot_function(
    f,
    fw_compiler=my_compiler,
    decompositions=decompositions
)
```

---

## Debugging

### Visualize Graphs

```python
from functorch.compile import aot_function, draw_graph

def visualize_compiler(name):
    def compiler(fx_g, inps):
        print(f"\n{name} graph:")
        print(fx_g.code)
        draw_graph(fx_g, name)  # Creates SVG file
        return fx_g
    return compiler

compiled_f = aot_function(
    f,
    fw_compiler=visualize_compiler("forward"),
    bw_compiler=visualize_compiler("backward")
)
```

### Get Compilation Context

```python
from functorch.compile import (
    get_aot_compilation_context,
    get_aot_graph_name,
    get_graph_being_compiled,
)

# Inside a compiler:
def my_compiler(fx_g, inps):
    ctx = get_aot_compilation_context()
    name = get_aot_graph_name()
    print(f"Compiling: {name}")
    return fx_g
```

### FX Minifier

For debugging compilation failures:

```python
from functorch.compile import minifier

# Reduce failing graph to minimal reproducer
minifier(fx_graph, example_inputs, check_fn)
```

---

## Effect Tokens

AOT Autograd handles side effects (prints, TorchBind ops) via "effect tokens":

```python
# Side-effectful graph with tokens:
def graph_with_effects(token0, reader):
    token1, frame = with_effects(side_effect_op, (reader,), token0)
    frame = frame * 2
    token2, frame2 = with_effects(side_effect_op, (reader,), token1)
    return token2, frame, frame2
```

Tokens ensure ordering of side effects while maintaining functional semantics.

---

## MLX Porting Considerations

### Conceptual Mapping

| PyTorch AOT Autograd | MLX Equivalent |
|---------------------|----------------|
| Eager tracing → FX graph | MLX is lazy by default |
| Joint graph (fwd+bwd) | `mx.grad` traces backward automatically |
| Partitioner (memory optimization) | MLX's lazy eval + `mx.compile` |
| Compiler backends | `mx.compile` with Metal backend |
| Effect tokens | N/A (MLX is purely functional) |

### Key Differences

1. **Default Behavior**: MLX is lazy by default, so "ahead of time" graph capture is the norm, not an optimization
2. **Backward Generation**: MLX's `mx.grad` and `mx.vjp` handle backward graph creation
3. **Memory Optimization**: MLX's lazy evaluation naturally enables memory optimizations without explicit partitioning
4. **Compilation**: `mx.compile` fuses operations and generates optimized Metal kernels

### MLX Approach

```python
import mlx.core as mx

def f(x):
    return mx.cos(mx.cos(x))

# MLX grad captures forward+backward lazily
grad_f = mx.grad(f)

# Compile for optimization (similar to AOT + compiler backend)
compiled_grad_f = mx.compile(grad_f)
```

### Implementation Strategy

For MLX porting, AOT Autograd concepts map to:

1. **Graph Capture**: Already inherent in MLX's lazy evaluation
2. **Forward/Backward Separation**: Use `mx.vjp` for explicit control
3. **Memory Optimization**: Leverage lazy evaluation; MLX compiler handles fusion
4. **Custom Transformations**: Build FX-like graph utilities if needed for advanced transformations

---

## Summary

| Component | Purpose |
|-----------|---------|
| `aot_function` | Compile function with custom backends |
| `aot_module` | Compile nn.Module |
| `default_partition` | Simple forward/backward split |
| `min_cut_rematerialization_partition` | Memory-efficient partitioning |
| `ts_compile` | TorchScript backend |
| `memory_efficient_fusion` | Combined min-cut + TS |
| `nop`, `print_compile` | Debugging backends |
| `make_boxed_compiler` | Create custom backends |

AOT Autograd enables whole-graph optimization and is the foundation of PyTorch 2.0's compilation infrastructure. For MLX, the lazy evaluation model provides similar benefits by default, with `mx.compile` serving as the optimization backend.
