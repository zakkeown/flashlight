# torch.export

## Overview

`torch.export` provides ahead-of-time (AOT) tracing and serialization of PyTorch models. Unlike `torch.compile` which is JIT, export creates a static graph representation that can be saved, loaded, and deployed independently of Python.

**Key Differences from torch.compile**:
| Aspect | torch.compile | torch.export |
|--------|--------------|--------------|
| Timing | JIT (at runtime) | AOT (ahead of time) |
| Output | Optimized callable | ExportedProgram |
| Serializable | No | Yes |
| Dynamic shapes | Via guards | Via Dim specifications |
| Use case | Training/inference speedup | Deployment, interop |

---

## Core API

### torch.export.export

```python
torch.export.export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: dict | tuple | list | None = None,
    strict: bool = False,
    preserve_module_call_signature: tuple[str, ...] = (),
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram
```

**Source**: `torch/export/__init__.py:165-286`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `mod` | nn.Module | Module to trace |
| `args` | tuple | Example positional inputs |
| `kwargs` | dict | Example keyword inputs |
| `dynamic_shapes` | dict/tuple/list | Dynamic dimension specs |
| `strict` | bool | Use TorchDynamo for stricter tracing |
| `preserve_module_call_signature` | tuple | Preserve submodule signatures |

### Basic Usage

```python
import torch
from torch.export import export, Dim

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
example_input = (torch.randn(32, 10),)

# Export with static shapes
exported = export(model, example_input)

# Export with dynamic batch dimension
batch = Dim("batch", min=1, max=128)
exported_dynamic = export(
    model,
    example_input,
    dynamic_shapes={"x": {0: batch}}
)
```

---

## ExportedProgram

The `ExportedProgram` class encapsulates the exported model.

**Source**: `torch/export/exported_program.py`

### Structure

```python
@dataclasses.dataclass
class ExportedProgram:
    graph_module: torch.fx.GraphModule  # The traced graph
    graph_signature: ExportGraphSignature  # Input/output signatures
    state_dict: dict[str, Any]  # Parameters and buffers
    range_constraints: dict  # Shape constraints
    module_call_graph: list[ModuleCallEntry]  # Module hierarchy
    constants: dict  # Constant values
```

### Key Methods

```python
# Execute the exported program
result = exported_program(*args, **kwargs)

# Get the FX graph
graph = exported_program.graph_module.graph

# Access parameters
params = exported_program.state_dict

# Run with different inputs
output = exported_program.module()(new_input)

# Apply decompositions
decomposed = exported_program.run_decompositions()
```

### Serialization

```python
from torch.export import save, load

# Save to file
save(exported_program, "model.pt2")

# Save to file-like object
with open("model.pt2", "wb") as f:
    save(exported_program, f)

# Load from file
loaded = load("model.pt2")

# Load with additional globals
loaded = load("model.pt2", f_globals={"custom_op": my_op})
```

---

## Dynamic Shapes

### Dim API

The `Dim` function creates dynamic dimension specifications:

```python
from torch.export import Dim

# Unbounded dynamic dimension
batch = Dim("batch")

# Bounded dimension
seq_len = Dim("seq_len", min=1, max=512)

# Multiple dimensions with relationship
B = Dim("B")
# B2 = B * 2  # Derived dimensions
```

**Source**: `torch/export/dynamic_shapes.py`

### Specifying Dynamic Shapes

```python
# Method 1: Dictionary mapping arg names to dim specs
dynamic_shapes = {
    "x": {0: batch, 1: None},  # batch dynamic, second static
    "y": {0: batch},  # Same batch dimension
}

# Method 2: Tuple/list matching input order
dynamic_shapes = (
    {0: batch},  # For first arg
    {0: batch},  # For second arg
)

# Method 3: Full specification with tuples
dynamic_shapes = {
    "x": (batch, None, None),  # (dynamic, static, static)
}
```

### Example with Dynamic Shapes

```python
import torch
from torch.export import export, Dim

class Transformer(torch.nn.Module):
    def forward(self, tokens, mask):
        # tokens: (batch, seq_len, hidden)
        # mask: (batch, seq_len)
        return tokens * mask.unsqueeze(-1)

model = Transformer()

# Define dynamic dimensions
batch = Dim("batch", min=1, max=64)
seq = Dim("seq", min=1, max=1024)

# Export with dynamic shapes
exported = export(
    model,
    (torch.randn(1, 16, 256), torch.ones(1, 16)),
    dynamic_shapes={
        "tokens": {0: batch, 1: seq},
        "mask": {0: batch, 1: seq},
    }
)

# Now works with any batch/seq within bounds
result = exported(torch.randn(32, 128, 256), torch.ones(32, 128))
```

---

## Graph Signature

The `ExportGraphSignature` describes the inputs and outputs of the exported graph.

```python
@dataclasses.dataclass
class ExportGraphSignature:
    input_specs: list[InputSpec]
    output_specs: list[OutputSpec]
```

### Input Types

```python
class InputKind(Enum):
    USER_INPUT = "user_input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT_TENSOR = "constant_tensor"
    CUSTOM_OBJ = "custom_obj"
    TOKEN = "token"

@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: str | None = None
    persistent: bool | None = None
```

### Output Types

```python
class OutputKind(Enum):
    USER_OUTPUT = "user_output"
    LOSS_OUTPUT = "loss_output"
    BUFFER_MUTATION = "buffer_mutation"
    GRADIENT_TO_PARAMETER = "gradient_to_parameter"
    GRADIENT_TO_USER_INPUT = "gradient_to_user_input"
    USER_INPUT_MUTATION = "user_input_mutation"
    TOKEN = "token"
```

### Inspecting Signatures

```python
exported = export(model, (x,))

# Print input specs
for spec in exported.graph_signature.input_specs:
    print(f"{spec.kind}: {spec.target}")

# Print output specs
for spec in exported.graph_signature.output_specs:
    print(f"{spec.kind}: {spec.arg}")
```

---

## Unflattening

Export flattens module hierarchies. `unflatten` restores the original structure:

```python
from torch.export import unflatten, UnflattenedModule

# Export flattens the module
exported = export(model, (x,))
# Graph has no module hierarchy

# Unflatten restores structure
unflat_model = unflatten(exported)
# Now has original nn.Module structure
```

### UnflattenedModule

```python
class UnflattenedModule(torch.nn.Module):
    """Module with restored hierarchy from ExportedProgram."""

    def forward(self, *args, **kwargs):
        # Calls the underlying exported program
        return self._graph_module(*args, **kwargs)
```

### Preserving Module Signatures

```python
# Preserve specific submodule signatures during export
exported = export(
    model,
    (x,),
    preserve_module_call_signature=("encoder", "decoder"),
)

# These submodules maintain their calling conventions
unflat = unflatten(exported)
```

---

## Decompositions

Export can decompose high-level operators into primitive operations:

```python
# Get default decompositions
from torch.export import default_decompositions

decomp_table = default_decompositions()

# Apply decompositions to exported program
decomposed = exported.run_decompositions(decomp_table)
```

### Custom Decomposition Table

```python
from torch.export import CustomDecompTable

# Create custom decomposition table
decomp = CustomDecompTable()

# Register custom decomposition
@decomp.register(torch.ops.aten.my_op)
def my_op_decomp(x):
    return x + 1

# Use with export
exported = export(model, (x,))
decomposed = exported.run_decompositions(decomp)
```

---

## Strict vs Non-Strict Mode

### Non-Strict (Default)

```python
exported = export(model, (x,), strict=False)
```

- Traces through Python runtime
- More permissive
- May miss some constraint violations
- Recommended for most cases

### Strict Mode

```python
exported = export(model, (x,), strict=True)
```

- Uses TorchDynamo for tracing
- Stricter validation
- Catches more potential issues
- May have limited Python feature coverage

---

## Draft Export

For debugging export issues, use `draft_export`:

```python
from torch.export import draft_export

# Get detailed error information
draft_result = draft_export(model, (x,))

# Shows all potential issues, not just the first
for error in draft_result.errors:
    print(error)
```

---

## Custom Operators

### Registering Custom Ops for Export

```python
from torch.library import Library, impl

# Define custom op
lib = Library("mylib", "DEF")
lib.define("my_op(Tensor x) -> Tensor")

@impl(lib, "my_op", "CPU")
def my_op_cpu(x):
    return x + 1

@impl(lib, "my_op", "Meta")
def my_op_meta(x):
    return torch.empty_like(x)

# Now can be used in export
model.my_op = torch.ops.mylib.my_op
exported = export(model, (x,))
```

### Custom Objects

```python
from torch.export import register_dataclass
from dataclasses import dataclass

@dataclass
class MyConfig:
    hidden_size: int
    num_layers: int

# Register for export
register_dataclass(MyConfig)

# Can now use in exported models
```

---

## Interoperability

### ONNX Export

```python
# Export to ONNX from ExportedProgram
import torch.onnx

torch.onnx.export(
    exported.module(),
    example_input,
    "model.onnx",
    export_params=True,
)
```

### TensorRT/Other Runtimes

```python
# ExportedProgram can be converted to other formats
# via the FX graph representation

graph = exported.graph_module.graph
for node in graph.nodes:
    print(node.op, node.target, node.args)
```

---

## MLX Implications

### Model Export for MLX

Since MLX uses a different execution model, exported PyTorch models need conversion:

```python
# PyTorch export
exported = export(model, (x,))

# For MLX, traverse the FX graph and convert operations
def convert_to_mlx(exported_program):
    import mlx.core as mx
    import mlx.nn as nn

    graph = exported_program.graph_module.graph
    mlx_ops = {}

    for node in graph.nodes:
        if node.op == "call_function":
            # Map PyTorch ops to MLX ops
            if node.target == torch.ops.aten.add:
                mlx_ops[node.name] = lambda a, b: mx.add(a, b)
            elif node.target == torch.ops.aten.matmul:
                mlx_ops[node.name] = lambda a, b: mx.matmul(a, b)
            # ... more mappings

    return mlx_ops
```

### Weight Conversion

```python
def convert_weights_to_mlx(exported_program):
    import mlx.core as mx

    mlx_weights = {}
    for name, param in exported_program.state_dict.items():
        # Convert PyTorch tensor to MLX array
        numpy_array = param.detach().cpu().numpy()
        mlx_weights[name] = mx.array(numpy_array)

    return mlx_weights
```

### Dynamic Shapes in MLX

MLX handles dynamic shapes natively:

```python
import mlx.core as mx

# MLX arrays are naturally dynamic
def mlx_model(x):
    # Works with any shape
    return mx.matmul(x, weights)

# No explicit Dim specifications needed
result = mlx_model(mx.random.normal((32, 10)))
result2 = mlx_model(mx.random.normal((64, 10)))
```

---

## Common Patterns

### Export with Preprocessing

```python
class ModelWithPreprocess(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, raw_input):
        # Preprocessing that should be part of export
        processed = raw_input / 255.0
        processed = processed - 0.5
        return self.model(processed)

# Export includes preprocessing
exported = export(ModelWithPreprocess(model), (raw_input,))
```

### Export Ensemble

```python
class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        outputs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Export entire ensemble
exported = export(Ensemble(models), (x,))
```

### Conditional Export

```python
# Export different configurations
def export_for_device(model, device_type):
    if device_type == "mobile":
        # Quantized export
        quantized = torch.quantization.quantize_dynamic(model)
        return export(quantized, (x,))
    else:
        return export(model, (x,))
```

---

## Implementation Files

**Core Export**:
- `torch/export/__init__.py` - Main API
- `torch/export/_trace.py` - Tracing implementation
- `torch/export/exported_program.py` - ExportedProgram class
- `torch/export/graph_signature.py` - Signature definitions

**Dynamic Shapes**:
- `torch/export/dynamic_shapes.py` - Dim and shape handling

**Utilities**:
- `torch/export/unflatten.py` - Module unflattening
- `torch/export/decomp_utils.py` - Decomposition utilities

**Serialization**:
- `torch/_export/serde/` - Serialization/deserialization

---

## Summary

`torch.export` provides:
1. **AOT graph capture** for deployment scenarios
2. **Dynamic shape support** via Dim specifications
3. **Serialization** for model persistence
4. **Interoperability** with other runtimes

For MLX porting:
- Use exported FX graphs as reference for operation mapping
- Convert state_dict to MLX arrays
- MLX's lazy evaluation provides similar graph benefits
- No explicit export step needed in MLX (dynamic by nature)
