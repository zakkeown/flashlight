# TorchScript and JIT Compilation

## Overview

TorchScript is a statically-typed subset of Python that can be compiled and optimized for deployment. It provides two main mechanisms: **scripting** (source code analysis) and **tracing** (recording operations). The JIT compiler optimizes TorchScript programs for faster execution.

**Reference Files:**
- `torch/jit/_script.py` - torch.jit.script implementation
- `torch/jit/_trace.py` - torch.jit.trace implementation
- `torch/jit/_freeze.py` - Model freezing
- `torch/jit/_serialization.py` - Save/load functionality

## Architecture

```
TorchScript Pipeline
├── Frontend
│   ├── script()     - Parse Python source → TorchScript AST
│   └── trace()      - Record operations → Graph
├── IR (Intermediate Representation)
│   └── TorchScript Graph - SSA-based IR
├── Optimization Passes
│   ├── Constant folding
│   ├── Dead code elimination
│   ├── Operator fusion
│   └── Common subexpression elimination
└── Execution
    └── TorchScript Interpreter
```

---

## torch.jit.script

Compiles Python source code to TorchScript by analyzing the source.

### Function Signature

```python
def script(
    obj: Any,                      # Function, nn.Module, class, dict, or list
    optimize: None = None,         # Deprecated
    _frames_up: int = 0,           # Internal
    _rcb: Callable | None = None,  # Resolution callback
    example_inputs: list[tuple] | dict | None = None,  # Type hints
) -> ScriptModule | ScriptFunction
```

### Scripting a Function

```python
import torch

@torch.jit.script
def compute(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.sum() > y.sum():
        return x * 2
    else:
        return y * 2

# View compiled code
print(compute.code)

# View graph
print(compute.graph)

# Execute
result = compute(torch.ones(3), torch.zeros(3))
```

### Scripting an nn.Module

```python
class MyModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.relu(x)
        return x

# Script the module
model = MyModule(64)
scripted = torch.jit.script(model)

# Access code and graph
print(scripted.code)
print(scripted.graph)
```

### Type Annotations

TorchScript requires explicit types for function arguments and returns:

```python
from typing import List, Tuple, Optional, Dict

@torch.jit.script
def typed_function(
    x: torch.Tensor,
    values: List[int],
    mapping: Dict[str, torch.Tensor],
    optional_y: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, int]:
    result = x.sum()
    if optional_y is not None:
        result = result + optional_y.sum()
    return result, len(values)
```

### Supported Python Features

| Feature | Supported | Notes |
|---------|-----------|-------|
| Control flow (if/else, for, while) | ✅ | Fully supported |
| List/Dict/Tuple | ✅ | Must be homogeneous types |
| Classes | ✅ | Use `@torch.jit.script` decorator |
| Type annotations | ✅ | Required for function args |
| Lambda functions | ❌ | Use named functions |
| *args, **kwargs | ⚠️ | Limited support |
| Generators | ❌ | Not supported |
| Global variables | ⚠️ | Captured at compile time |

---

## torch.jit.trace

Records operations by executing the model with example inputs.

### Function Signature

```python
def trace(
    func: Callable | nn.Module,      # Function or module to trace
    example_inputs: tuple | Tensor,  # Example inputs for tracing
    check_trace: bool = True,        # Verify trace accuracy
    check_inputs: list[tuple] = None, # Additional inputs to check
    check_tolerance: float = 1e-5,   # Tolerance for check
    strict: bool = True,             # Strict mode
    example_kwarg_inputs: dict = None, # Keyword argument inputs
) -> ScriptModule | ScriptFunction
```

### Basic Tracing

```python
import torch

def simple_function(x):
    return x * 2 + 1

# Trace with example input
example_input = torch.rand(3, 4)
traced = torch.jit.trace(simple_function, example_input)

# View the traced graph
print(traced.graph)

# Execute
result = traced(torch.rand(5, 6))  # Works with different shapes
```

### Tracing an nn.Module

```python
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x

model = ConvNet()
model.eval()  # Set to eval mode before tracing

example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
```

### trace_module()

Trace specific methods of a module:

```python
class MultiMethodModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.softmax(self.forward(x), dim=-1)

model = MultiMethodModule()

# Trace multiple methods
traced = torch.jit.trace_module(
    model,
    inputs={
        'forward': (torch.rand(1, 10),),
        'predict': (torch.rand(1, 10),),
    }
)
```

---

## Script vs Trace Comparison

| Aspect | script() | trace() |
|--------|----------|---------|
| Control flow | ✅ Preserved | ❌ Flattened |
| Dynamic shapes | ✅ Supported | ⚠️ Shape recorded |
| Data-dependent ops | ✅ Supported | ❌ Not captured |
| Type annotations | Required | Not required |
| Source code access | Required | Not required |
| Python subset | TorchScript only | Full Python |

### When to Use Script

- Control flow depends on data (if/else based on tensor values)
- Dynamic loops (varying iteration count)
- Complex Python logic

```python
@torch.jit.script
def dynamic_computation(x: torch.Tensor) -> torch.Tensor:
    # Control flow based on data - must use script
    for i in range(x.size(0)):  # Dynamic loop
        if x[i].sum() > 0:      # Data-dependent condition
            x[i] = x[i] * 2
    return x
```

### When to Use Trace

- Simple forward passes
- Pre-trained models without source access
- When source contains unsupported Python features

```python
# Tracing works even with complex internal operations
# as long as the trace is representative
model = torchvision.models.resnet18(pretrained=True)
traced = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
```

---

## ScriptModule

The result of scripting or tracing an nn.Module.

### Attributes and Methods

```python
scripted = torch.jit.script(model)

# View source code representation
print(scripted.code)

# View IR graph
print(scripted.graph)

# Access parameters
for name, param in scripted.named_parameters():
    print(name, param.shape)

# Access submodules
for name, module in scripted.named_modules():
    print(name, type(module))

# Save to file
scripted.save('model.pt')

# Load from file
loaded = torch.jit.load('model.pt')
```

---

## Saving and Loading

### Save ScriptModule

```python
# Script or trace the model
scripted = torch.jit.script(model)

# Save to file
scripted.save('model.pt')

# Save with extra files
extra_files = {'config.json': '{"version": "1.0"}'}
scripted.save('model.pt', _extra_files=extra_files)
```

### Load ScriptModule

```python
# Load on CPU
model = torch.jit.load('model.pt', map_location='cpu')

# Load on specific device
model = torch.jit.load('model.pt', map_location='cuda:0')

# Load with extra files
extra_files = {'config.json': ''}
model = torch.jit.load('model.pt', _extra_files=extra_files)
config = extra_files['config.json']
```

---

## Optimization and Freezing

### freeze()

Freeze a ScriptModule for inference optimization:

```python
import torch.jit

model = torch.jit.script(my_model)
model.eval()  # Must be in eval mode

# Freeze the model
frozen = torch.jit.freeze(model)

# Frozen model has:
# - Inlined submodules
# - Folded constants
# - Removed training-only code
```

### optimize_for_inference()

Apply inference-specific optimizations:

```python
optimized = torch.jit.optimize_for_inference(frozen)
```

### Fusion

```python
# Enable/disable fusion
torch.jit.set_fusion_strategy([('STATIC', 3), ('DYNAMIC', 10)])

# Check fusion status
with torch.jit.fuser('fuser1'):  # or 'fuser2', 'none'
    result = model(input)
```

---

## Script Classes

Define custom classes usable in TorchScript:

```python
@torch.jit.script
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance(self, other: 'Point') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx ** 2 + dy ** 2) ** 0.5

@torch.jit.script
def use_point():
    p1 = Point(0.0, 0.0)
    p2 = Point(3.0, 4.0)
    return p1.distance(p2)  # Returns 5.0
```

---

## Annotations

### @torch.jit.ignore

Exclude methods from scripting:

```python
class MyModule(torch.nn.Module):
    def forward(self, x):
        return self.helper(x)

    @torch.jit.ignore
    def helper(self, x):
        # This method won't be scripted
        # Can use any Python feature
        return x * 2
```

### @torch.jit.unused

Mark methods as unused (won't be called):

```python
class MyModule(torch.nn.Module):
    def forward(self, x):
        return x * 2

    @torch.jit.unused
    def training_helper(self, x):
        # This is only used during training
        # Error if called in TorchScript
        pass
```

### @torch.jit.export

Explicitly export a method:

```python
class MyModule(torch.nn.Module):
    def forward(self, x):
        return x

    @torch.jit.export
    def custom_method(self, x):
        # This method is also exported
        return x * 2
```

---

## Common Patterns

### Conditional Module Selection

```python
class ConditionalModel(torch.nn.Module):
    def __init__(self, use_complex: bool):
        super().__init__()
        self.use_complex = use_complex
        self.simple = torch.nn.Linear(10, 10)
        self.complex = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_complex:
            return self.complex(x)
        else:
            return self.simple(x)

# Script captures the conditional
scripted = torch.jit.script(ConditionalModel(True))
```

### Optional Inputs

```python
from typing import Optional

class OptionalInput(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        return x.sum()

scripted = torch.jit.script(OptionalInput())
scripted(torch.rand(3, 4))  # Without mask
scripted(torch.rand(3, 4), torch.ones(3, 4))  # With mask
```

---

## Debugging

### Print Graph

```python
scripted = torch.jit.script(model)
print(scripted.graph)
```

### Print Code

```python
print(scripted.code)
```

### Check Trace

```python
# Enable trace checking warnings
traced = torch.jit.trace(
    model,
    example_input,
    check_trace=True,
    check_inputs=[input1, input2, input3]
)
```

### Last Graph

```python
# After running a scripted function
result = scripted(input)
print(scripted.graph_for(input))  # Optimized graph
```

---

## MLX Mapping

### MLX Compilation

MLX uses `mx.compile()` for JIT compilation, which differs from TorchScript:

```python
import mlx.core as mx

# MLX compilation
@mx.compile
def mlx_function(x, y):
    return x @ y + mx.softmax(x, axis=-1)

# Or compile existing function
def regular_function(x, y):
    return x @ y

compiled = mx.compile(regular_function)
```

### Key Differences

| Aspect | TorchScript | MLX |
|--------|-------------|-----|
| Approach | Source analysis / tracing | Lazy graph compilation |
| Language | TorchScript subset | Full Python |
| Control flow | Captured in IR | Traced dynamically |
| Type system | Static types required | Dynamic |
| Serialization | .pt files | Separate weight serialization |
| Optimization | JIT passes | Graph-level optimization |

### Conceptual Mapping

| TorchScript | MLX Equivalent |
|-------------|----------------|
| `torch.jit.script(fn)` | `mx.compile(fn)` |
| `torch.jit.trace(fn, input)` | Implicit lazy evaluation |
| `scripted.save('model.pt')` | `mx.save()` + model code |
| `torch.jit.freeze(model)` | Not needed (lazy eval) |
| `ScriptModule` | Compiled function + weights |

### MLX Example

```python
import mlx.core as mx
import mlx.nn as nn

class MLXModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def __call__(self, x):
        return mx.relu(self.linear(x))

model = MLXModel(64)

# Compile for efficient execution
@mx.compile
def forward(model, x):
    return model(x)

# Execute - compilation happens on first call
result = forward(model, mx.random.normal((10, 64)))

# Save weights (not the compiled function)
mx.save('weights.npz', dict(model.parameters()))
```

---

## Best Practices

1. **Use script for control flow** - Data-dependent branches need scripting

2. **Use trace for simple models** - Faster and works with any Python

3. **Add type annotations** - Required for script, helpful for debugging

4. **Set eval mode before trace** - Avoids training/eval mode issues

5. **Test with check_trace=True** - Catches tracing issues early

6. **Freeze for deployment** - Optimizes and removes training code

7. **Save extra files** - Include config/metadata with model

8. **Use @torch.jit.ignore** - For Python-only helper methods

---

## Summary

| Function | Purpose |
|----------|---------|
| `torch.jit.script()` | Compile from source code |
| `torch.jit.trace()` | Record operations with example input |
| `torch.jit.trace_module()` | Trace specific module methods |
| `torch.jit.save()` | Serialize ScriptModule |
| `torch.jit.load()` | Load ScriptModule |
| `torch.jit.freeze()` | Optimize for inference |
| `torch.jit.optimize_for_inference()` | Apply inference optimizations |
| `@torch.jit.script` | Decorator for functions/classes |
| `@torch.jit.ignore` | Exclude from scripting |
| `@torch.jit.export` | Include method in script |
