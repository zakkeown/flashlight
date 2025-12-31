# FX Graph Capture and Transformations

## Overview

PyTorch FX is a framework for capturing and transforming PyTorch programs at the graph level. It provides symbolic tracing to capture a program's operations as a graph (IR), enabling analysis, optimization, and code generation.

**Reference Files:**
- `torch/fx/graph.py` - Graph data structure
- `torch/fx/node.py` - Node representation
- `torch/fx/_symbolic_trace.py` - Symbolic tracer
- `torch/fx/graph_module.py` - GraphModule wrapper
- `torch/fx/interpreter.py` - Graph interpreter
- `torch/fx/subgraph_rewriter.py` - Pattern matching/replacement

## Architecture

```
FX Pipeline
├── Symbolic Tracing (symbolic_trace)
│   └── Tracer → Proxy → Graph
├── Graph IR
│   ├── Node (operation representation)
│   └── Graph (DAG of nodes)
├── GraphModule
│   └── nn.Module + Graph
├── Transformations
│   ├── Graph passes
│   ├── Subgraph rewriting
│   └── Node manipulation
└── Code Generation
    └── Graph → Python code
```

---

## Core Components

### Node

A Node represents a single operation in the graph.

```python
class Node:
    op: str              # Operation type
    name: str            # Unique name
    target: Target       # What to call (function, method, attr)
    args: tuple          # Positional arguments
    kwargs: dict         # Keyword arguments
    users: dict          # Nodes that use this node
    _prev: Node          # Previous node in graph order
    _next: Node          # Next node in graph order
```

### Node Operations (op types)

| Op | Description | Target | Example |
|----|-------------|--------|---------|
| `placeholder` | Input to the graph | Input name | `x = placeholder('x')` |
| `call_function` | Call a Python function | Function ref | `torch.relu(x)` |
| `call_method` | Call a method on a value | Method name | `x.sum()` |
| `call_module` | Call a submodule | Module name | `self.conv(x)` |
| `get_attr` | Access an attribute | Attr name | `self.weight` |
| `output` | Return value(s) | 'output' | `return x` |

### Graph

Container for nodes with insertion/iteration APIs.

```python
class Graph:
    def __init__(self):
        self._root: Node  # Sentinel node
        self._len: int    # Number of nodes

    # Insertion
    def placeholder(self, name: str) -> Node
    def call_function(self, target: Callable, args, kwargs) -> Node
    def call_method(self, method_name: str, args, kwargs) -> Node
    def call_module(self, module_name: str, args, kwargs) -> Node
    def get_attr(self, qualified_name: str) -> Node
    def output(self, result) -> Node

    # Iteration
    @property
    def nodes(self) -> Iterator[Node]

    # Code generation
    def python_code(self, root_module: str) -> PythonCode
```

### GraphModule

An nn.Module that wraps a Graph.

```python
class GraphModule(nn.Module):
    def __init__(self, root: nn.Module, graph: Graph):
        self._graph: Graph
        self._code: str  # Generated Python code

    def forward(self, *args) -> Any:
        # Executes the graph
        ...

    @property
    def graph(self) -> Graph

    @property
    def code(self) -> str

    def recompile(self) -> PythonCode
```

---

## Symbolic Tracing

### symbolic_trace()

Captures a module's forward pass as a Graph.

```python
import torch
import torch.fx as fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModule()
traced = fx.symbolic_trace(model)

print(traced.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %linear : [num_users=1] = call_module[target=linear](args = (%x,), kwargs = {})
#     %relu : [num_users=1] = call_function[target=torch.relu](args = (%linear,), kwargs = {})
#     %output : [num_users=0] = output(args = (%relu,), kwargs = {})

print(traced.code)
# def forward(self, x):
#     linear = self.linear(x);  x = None
#     relu = torch.relu(linear);  linear = None
#     return relu
```

### Tracer Class

Custom tracers for specialized behavior.

```python
from torch.fx import Tracer, Proxy

class CustomTracer(Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        # Don't trace into certain modules
        if isinstance(m, torch.nn.BatchNorm2d):
            return True
        return super().is_leaf_module(m, module_qualified_name)

    def create_arg(self, a):
        # Custom argument handling
        return super().create_arg(a)

tracer = CustomTracer()
graph = tracer.trace(model)
```

### Proxy

Proxies track operations during tracing.

```python
# During tracing, tensors are replaced with Proxies
# Operations on Proxies are recorded in the graph

def traced_function(x):
    # x is a Proxy during tracing
    y = x * 2      # Records call_function(operator.mul, (x, 2))
    z = y.sum()    # Records call_method('sum', (y,))
    return z       # Records output((z,))
```

---

## Graph Manipulation

### Iterating Nodes

```python
for node in traced.graph.nodes:
    print(f"{node.name}: {node.op} -> {node.target}")
    print(f"  args: {node.args}")
    print(f"  kwargs: {node.kwargs}")
    print(f"  users: {list(node.users.keys())}")
```

### Inserting Nodes

```python
graph = traced.graph

# Find insertion point
for node in graph.nodes:
    if node.op == 'call_function' and node.target == torch.relu:
        # Insert after this node
        with graph.inserting_after(node):
            # Add a new operation
            new_node = graph.call_function(torch.sigmoid, args=(node,))

            # Replace uses of old node with new node
            node.replace_all_uses_with(new_node)
            # But new_node still uses the original node
            new_node.args = (node,)

# Recompile to update code
traced.recompile()
```

### Removing Nodes

```python
# Find and remove dead code
for node in reversed(list(graph.nodes)):
    if len(node.users) == 0 and node.op != 'output':
        graph.erase_node(node)

traced.recompile()
```

### Replacing Targets

```python
# Replace all relu with gelu
for node in graph.nodes:
    if node.op == 'call_function' and node.target == torch.relu:
        node.target = torch.nn.functional.gelu

traced.recompile()
```

---

## Subgraph Rewriter

Pattern matching and replacement for common transformations.

```python
from torch.fx import subgraph_rewriter

def pattern(x):
    # Pattern to match
    y = torch.relu(x)
    return y

def replacement(x):
    # Replacement pattern
    y = torch.nn.functional.gelu(x)
    return y

# Apply replacement
subgraph_rewriter.replace_pattern(traced, pattern, replacement)
```

### More Complex Patterns

```python
def fused_bias_relu_pattern(x, bias):
    y = x + bias
    return torch.relu(y)

def fused_bias_relu_replacement(x, bias):
    return torch.nn.functional.relu(x + bias)

replaced = subgraph_rewriter.replace_pattern(
    traced,
    fused_bias_relu_pattern,
    fused_bias_relu_replacement
)
print(f"Replaced {len(replaced)} patterns")
```

---

## Graph Interpreter

Execute a graph node by node (useful for debugging or custom execution).

```python
from torch.fx import Interpreter

class DebugInterpreter(Interpreter):
    def run_node(self, n):
        result = super().run_node(n)
        print(f"{n.name}: {type(result)}")
        if isinstance(result, torch.Tensor):
            print(f"  shape: {result.shape}, dtype: {result.dtype}")
        return result

interp = DebugInterpreter(traced)
output = interp.run(torch.randn(10, 10))
```

### Custom Execution

```python
class ProfilingInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.timings = {}

    def run_node(self, n):
        import time
        start = time.perf_counter()
        result = super().run_node(n)
        self.timings[n.name] = time.perf_counter() - start
        return result

interp = ProfilingInterpreter(traced)
output = interp.run(input_tensor)
print(interp.timings)
```

---

## Common Transformations

### Dead Code Elimination

```python
def eliminate_dead_code(graph):
    """Remove nodes with no users (except output)."""
    changed = True
    while changed:
        changed = False
        for node in reversed(list(graph.nodes)):
            if node.op != 'output' and len(node.users) == 0:
                graph.erase_node(node)
                changed = True
```

### Constant Folding

```python
def fold_constants(graph_module):
    """Evaluate operations on constant inputs."""
    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            # Check if all args are constants
            if all(isinstance(arg, (int, float, torch.Tensor))
                   for arg in node.args):
                # Evaluate at trace time
                with torch.no_grad():
                    result = node.target(*node.args, **node.kwargs)

                # Replace with constant
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.get_attr(f'_const_{node.name}')
                    setattr(graph_module, f'_const_{node.name}', result)
                    node.replace_all_uses_with(new_node)
                    graph_module.graph.erase_node(node)

    graph_module.recompile()
```

### Operator Fusion

```python
def fuse_conv_bn(graph_module):
    """Fuse Conv2d + BatchNorm2d patterns."""
    patterns = []

    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            module = getattr(graph_module, node.target)
            if isinstance(module, torch.nn.BatchNorm2d):
                # Check if input is a Conv2d
                prev = node.args[0]
                if prev.op == 'call_module':
                    prev_module = getattr(graph_module, prev.target)
                    if isinstance(prev_module, torch.nn.Conv2d):
                        patterns.append((prev, node))

    for conv_node, bn_node in patterns:
        # Fuse weights and biases
        conv = getattr(graph_module, conv_node.target)
        bn = getattr(graph_module, bn_node.target)
        fused = fuse_conv_bn_weights(conv, bn)

        # Replace modules
        setattr(graph_module, conv_node.target, fused)
        bn_node.replace_all_uses_with(conv_node)
        graph_module.graph.erase_node(bn_node)

    graph_module.recompile()
```

---

## Shape Propagation

```python
from torch.fx.passes.shape_prop import ShapeProp

# Propagate shapes through the graph
sp = ShapeProp(traced)
sp.run(torch.randn(1, 3, 224, 224))

# Access shape information
for node in traced.graph.nodes:
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        meta = node.meta['tensor_meta']
        print(f"{node.name}: shape={meta.shape}, dtype={meta.dtype}")
```

---

## Graph Visualization

```python
# Print graph in readable format
print(traced.graph)

# Generate Python code
print(traced.code)

# Visualize with GraphViz (if installed)
from torch.fx.passes.graph_drawer import FxGraphDrawer

drawer = FxGraphDrawer(traced, "my_model")
dot = drawer.get_dot_graph()
dot.render("graph", format="png")  # Saves graph.png
```

---

## Common Patterns

### Custom Pass

```python
def my_graph_pass(graph_module):
    """Template for a graph transformation pass."""
    graph = graph_module.graph

    for node in list(graph.nodes):  # Copy list since we modify
        if should_transform(node):
            transform_node(graph, node)

    graph_module.recompile()
    return graph_module

def should_transform(node):
    return (node.op == 'call_function' and
            node.target == some_target)

def transform_node(graph, node):
    with graph.inserting_after(node):
        new_node = graph.call_function(
            new_target,
            args=node.args,
            kwargs=node.kwargs
        )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
```

### Leaf Module Configuration

```python
class MyTracer(Tracer):
    # Don't trace into these module types
    LEAF_MODULES = {
        torch.nn.BatchNorm2d,
        torch.nn.LayerNorm,
        CustomModule,
    }

    def is_leaf_module(self, m, qualified_name):
        return type(m) in self.LEAF_MODULES
```

---

## MLX Mapping

### MLX Graph Concepts

MLX uses lazy evaluation which implicitly builds computation graphs:

```python
import mlx.core as mx

# MLX builds graphs lazily
def mlx_function(x, y):
    z = x @ y           # Operation recorded
    w = mx.softmax(z)   # Operation recorded
    return w            # Graph not executed yet

# Graph executed on eval
result = mlx_function(mx.random.normal((10, 10)), mx.random.normal((10, 10)))
mx.eval(result)  # Now executed
```

### Compilation

```python
# MLX compilation optimizes the graph
@mx.compile
def optimized_function(x, y):
    return mx.softmax(x @ y, axis=-1)
```

### Key Differences

| Aspect | PyTorch FX | MLX |
|--------|------------|-----|
| Graph capture | Explicit `symbolic_trace()` | Implicit (lazy evaluation) |
| IR access | Full Graph/Node API | Internal (not exposed) |
| Transformations | User-defined passes | Internal optimizations |
| Code generation | Explicit `graph.python_code()` | Not exposed |
| Execution | GraphModule.forward() | `mx.eval()` |

### Conceptual Mapping

| FX Concept | MLX Equivalent |
|------------|----------------|
| `symbolic_trace()` | Implicit graph building |
| `GraphModule` | Compiled function |
| `Node` | Internal operation |
| `Graph.nodes` | Not exposed |
| `subgraph_rewriter` | Internal optimization passes |
| `Interpreter` | `mx.eval()` |

---

## Integration with torch.compile

FX is the foundation for torch.compile (PyTorch 2.0+):

```python
# torch.compile uses FX internally
@torch.compile
def compiled_function(x):
    return torch.relu(x @ x.T)

# Access the FX graph (debugging)
torch._dynamo.config.log_level = logging.DEBUG
compiled_function(torch.randn(10, 10))
```

---

## Best Practices

1. **Use symbolic_trace for simple modules** - More reliable than Dynamo for basic cases

2. **Handle data-dependent control flow** - Use `torch.cond` or concrete_args

3. **Mark leaf modules appropriately** - Prevent tracing into complex/opaque modules

4. **Always recompile after modifications** - `graph_module.recompile()`

5. **Verify transformations** - Compare outputs before/after

6. **Use shape propagation** - Enables shape-aware optimizations

7. **Preserve node ordering** - Maintain topological order

8. **Clean up dead code** - Run DCE after transformations

---

## Summary

| Component | Purpose |
|-----------|---------|
| `symbolic_trace()` | Capture module as graph |
| `Graph` | Container for nodes |
| `Node` | Single operation |
| `GraphModule` | Executable graph wrapper |
| `Tracer` | Customizable tracing logic |
| `Proxy` | Tracks operations during trace |
| `Interpreter` | Step-by-step execution |
| `subgraph_rewriter` | Pattern matching/replacement |
| `ShapeProp` | Shape inference pass |
| `recompile()` | Regenerate code after changes |
