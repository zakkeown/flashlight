# TorchGen Code Generation Pipeline

## Purpose

PyTorch's code generation system (`torchgen`) is a sophisticated infrastructure that generates **thousands of lines of C++ and Python code** from a single YAML specification file. Understanding this system is crucial for:
1. Comprehending how PyTorch operators are actually implemented
2. Understanding the relationship between YAML definitions and runtime code
3. Deciding whether to adopt a similar approach for MLX

**Key Insight**: PyTorch defines **2,666 operators** in `native_functions.yaml`. Instead of hand-writing implementations for each operator across multiple backends (CPU, CUDA, MPS, etc.), PyTorch uses code generation to:
- Generate dispatcher registration code
- Generate Python bindings
- Generate autograd derivatives
- Generate method/function variants
- Generate type-checking and argument validation

This documentation explains the entire pipeline from YAML → Generated Code.

---

## Architecture Overview

### The Pipeline

```
┌────────────────────────────────────────────────────────┐
│  native_functions.yaml (16,119 lines)                  │
│  - 2,666 operator definitions                          │
│  - Schemas, dispatch info, tags, metadata              │
└──────────────┬─────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│  torchgen/gen.py (Main Entry Point)                    │
│  - Parses YAML using model.py                          │
│  - Builds NativeFunction objects                       │
│  - Coordinates code generation                         │
└──────────────┬─────────────────────────────────────────┘
               │
               ├──────────────────────┬──────────────────┬─────────────────┐
               ▼                      ▼                  ▼                 ▼
┌─────────────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌─────────────┐
│  Dispatcher Code    │  │ Python Bindings  │  │ Autograd Code  │  │  Variants   │
│  (register_*.cpp)   │  │ (python_*.cpp)   │  │ (derivatives)  │  │ (method/fn) │
└─────────────────────┘  └──────────────────┘  └────────────────┘  └─────────────┘
               │                      │                  │                 │
               └──────────────────────┴──────────────────┴─────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │   Build System (CMake)       │
                        │   Compiles Generated Code    │
                        └──────────────────────────────┘
```

### Key Components

1. **native_functions.yaml**: Single source of truth for operator definitions
2. **torchgen/model.py**: Data model (classes representing YAML structure)
3. **torchgen/gen.py**: Main code generation orchestrator
4. **torchgen/api/**: APIs for different code generation targets (dispatcher, python, autograd)
5. **torchgen/dest/**: "Destinations" that write generated code to files

---

## YAML Schema Deep-Dive

### Basic Operator Definition

**Location**: `aten/src/ATen/native/native_functions.yaml`

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
    MPS: add_mps
  tags: [core, pointwise]
```

### Schema Components

#### 1. `func`: Function Signature

**Format**: `name.overload(arguments) -> return_type`

```yaml
# Basic example:
- func: add(Tensor self, Tensor other) -> Tensor

# With overload name:
- func: add.Tensor(Tensor self, Tensor other) -> Tensor
- func: add.Scalar(Tensor self, Scalar other) -> Tensor

# With keyword-only args (after *):
- func: add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

# Multiple returns:
- func: topk(Tensor self, int k) -> (Tensor values, Tensor indices)
```

**Overload Names**:
- Used to distinguish multiple versions of the same operator
- Common overloads: `Tensor`, `Scalar`, `out`, `_`
- Becomes part of the C++ function name: `at::add_Tensor(...)`

**Argument Annotations**:
- `Tensor(a!)`: In-place modification (output alias)
- `Tensor?`: Optional tensor (may be undefined)
- `Tensor[]`: List of tensors
- `int?`: Optional integer
- `*`: Keyword-only arguments follow

#### 2. `variants`: Calling Conventions

Specifies how the operator can be called:

```yaml
variants: function, method
```

- **`function`**: `torch.add(a, b)` (namespace function)
- **`method`**: `a.add(b)` (tensor method)
- **Both**: Generates both calling conventions

**Generated Code**:
```cpp
// function variant:
namespace at {
  Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha);
}

// method variant:
class TensorBase {
  Tensor add(const Tensor& other, const Scalar& alpha) const;
};
```

#### 3. `dispatch`: Backend Routing

Maps backends to implementation functions:

```yaml
dispatch:
  CPU: add_cpu           # aten/src/ATen/native/BinaryOps.cpp
  CUDA: add_cuda         # aten/src/ATen/native/cuda/BinaryOps.cu
  MPS: add_mps           # aten/src/ATen/native/mps/operations/Binary.mm
  Meta: add_meta         # For tracing without execution
```

**Special Dispatch Keys**:
- `CompositeExplicitAutograd`: Composite kernel, manually defined backward
- `CompositeImplicitAutograd`: Composite kernel, autograd from decomposition
- `Math`: Dispatches to math kernel (used for reductions)

**Multiple Dispatch Keys**:
```yaml
dispatch:
  CPU, CUDA, MPS: binary_op_impl  # Same impl for multiple backends
```

#### 4. `tags`: Operator Categorization

Tags help with selective compilation and optimization:

```yaml
tags: [core, pointwise, canonical]
```

**Common Tags**:
- `core`: Essential operator (always included)
- `pointwise`: Element-wise operation
- `canonical`: Canonical version (not an alias)
- `view_copy`: Creates a view
- `inplace_view`: In-place operation that returns a view
- `generated`: Automatically generated operator

#### 5. `structured`: Structured Kernels

For operators with common patterns:

```yaml
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: add_stub
```

**Structured Kernels** provide:
- Automatic shape inference
- Output allocation (for functional variant)
- TensorIterator setup
- Reduces boilerplate significantly

#### 6. `autogen`: Automatic Variant Generation

Automatically generate related variants:

```yaml
- func: add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  autogen: add.out  # Auto-generate the out variant
```

Torchgen creates:
- `add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)`

### Advanced Schema Features

#### Aliasing Annotations

Control memory aliasing semantics:

```yaml
# In-place operation:
- func: add_(Tensor(a!) self, Tensor other) -> Tensor(a!)

# Out parameter:
- func: add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)

# View operation:
- func: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
```

**Annotations**:
- `(a!)`: Mutates and returns aliased tensor
- `(a)`: Returns view of input tensor (no mutation)
- No annotation: Returns new tensor (no aliasing)

#### Python-Specific Features

```yaml
# Default values:
- func: add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

# Python name override:
python_module: special
- func: bessel_j0(Tensor self) -> Tensor
  # Becomes: torch.special.bessel_j0()
```

#### Manual Bindings

For special cases requiring hand-written code:

```yaml
- func: _backward(Tensor self, Tensor[] inputs, Tensor? gradient=None) -> ()
  manual_cpp_binding: True  # Skip C++ binding generation
  manual_kernel_registration: True  # Skip dispatcher registration
```

---

## Code Generation Process

### Step 1: Parse YAML

**File**: `torchgen/gen.py:170-200`

```python
def parse_native_yaml(path: str) -> ParsedYaml:
    with open(path) as f:
        es = yaml.load(f, Loader=LineLoader)  # Custom loader tracks line numbers

    native_functions = []
    backend_indices = defaultdict(dict)

    for e in es:
        # Create NativeFunction object from YAML entry
        func, backend_metadata = NativeFunction.from_yaml(
            e, location, valid_tags, ignore_keys
        )
        native_functions.append(func)
        BackendIndex.grow_index(backend_indices, backend_metadata)

    return ParsedYaml(native_functions, backend_indices)
```

**NativeFunction Object**:
```python
@dataclass(frozen=True)
class NativeFunction:
    func: FunctionSchema          # Parsed function signature
    use_const_ref_for_mutable_tensors: bool
    variants: set[Variant]        # function, method
    structured: bool
    structured_delegate: str | None
    structured_inherits: str | None
    precomputed: Sequence[str]
    autogen: Sequence[str]
    manual_cpp_binding: bool
    manual_kernel_registration: bool
    python_module: str | None
    category_override: str | None
    device_guard: bool
    device_check: DeviceCheckType
    ufunc_inner_loop: dict[str, str]
    tags: set[str]
    loc: Location                 # File + line number
```

### Step 2: Build Code Generation Context

**API Modules** translate schemas to different C++ signatures:

#### dispatcher.py: Dispatcher API

```python
@dataclass(frozen=True)
class DispatcherSignature:
    """
    Signature for the dispatcher interface:
      Tensor add(const Tensor& self, const Tensor& other, const Scalar& alpha);
    """
    func: FunctionSchema
    def arguments(self) -> list[Binding]:
        return dispatcher.arguments(self.func)
```

#### native.py: Native Kernel API

```python
@dataclass(frozen=True)
class NativeSignature:
    """
    Signature for native kernels:
      Tensor add_cpu(const Tensor& self, const Tensor& other, const Scalar& alpha);
    """
    func: FunctionSchema
    prefix: str = ""  # e.g., "add_cpu"
```

#### python.py: Python Binding API

```python
@dataclass(frozen=True)
class PythonSignature:
    """
    Signature for Python bindings:
      def add(self: Tensor, other: Tensor, alpha: Number = 1) -> Tensor
    """
    func: FunctionSchema
    method: bool  # True for tensor methods
```

### Step 3: Generate Code

#### Dispatcher Registration

**Template**: `torchgen/dest/register_dispatch_key.py`

**Generated File**: `build/aten/src/ATen/RegisterCPU.cpp` (example)

```cpp
// Generated registration for add.Tensor on CPU
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor",
    TORCH_FN(wrapper_CPU_add_Tensor));  // Wrapper calls add_cpu
}

// Generated wrapper:
at::Tensor wrapper_CPU_add_Tensor(
    const at::Tensor & self,
    const at::Tensor & other,
    const at::Scalar & alpha) {
  // Setup (profiling, device guard, etc.)
  auto _r = at::native::add_cpu(self, other, alpha);
  return _r;
}
```

**Key Files**:
- `RegisterCPU.cpp`: CPU dispatcher registrations
- `RegisterCUDA.cpp`: CUDA dispatcher registrations
- `RegisterMPS.cpp`: MPS (Metal) dispatcher registrations
- One file per dispatch key

#### Python Bindings

**Template**: `tools/autograd/gen_python_functions.py`

**Generated File**: `build/aten/src/ATen/core/PythonFunctions.cpp`

```cpp
// Generated Python binding for torch.add
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
    "add(Tensor input, Scalar other, Scalar alpha=1)",
  });

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // Tensor + Tensor variant
      if (_r.isNone(3)) {
        // Functional form
        return wrap(dispatch_add(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // Out form
        return wrap(dispatch_add_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
    case 1: {
      // Tensor + Scalar variant
      return wrap(dispatch_add(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

**Argument Parsing**:
- `PythonArgParser`: Parses Python args/kwargs
- Type checking and conversion
- Overload resolution (picks correct variant)
- Error messages with schema info

#### Autograd Derivatives

**Input**: `tools/autograd/derivatives.yaml`

```yaml
- name: add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
  self: grad
  other: grad * alpha.conj()
```

**Generated**: Backward function in `build/torch/csrc/autograd/generated/Functions.cpp`

```cpp
variable_list AddBackward::apply(variable_list&& grads) {
  auto grad = grads[0];
  variable_list grad_inputs(2);

  // self gradient:
  if (should_compute_output(0)) {
    grad_inputs[0] = grad;
  }

  // other gradient:
  if (should_compute_output(1)) {
    grad_inputs[1] = grad * alpha_.conj();
  }

  return grad_inputs;
}
```

### Step 4: Template Engine

**Code Template System**: `torchgen/code_template.py`

Templates use `${variable}` substitution:

```python
REGISTRATION_TEMPLATE = CodeTemplate("""\
TORCH_LIBRARY_IMPL(${namespace}, ${dispatch_key}, m) {
  ${registrations}
}
""")

generated = REGISTRATION_TEMPLATE.substitute(
    namespace="aten",
    dispatch_key="CPU",
    registrations="\n  ".join(reg_lines)
)
```

**Advanced Features**:
- Conditional blocks: `$if condition ... $endif`
- Loops: `$for item in items ... $endfor`
- Indentation preservation

---

## Key Modules

### torchgen/model.py

Defines the data model for YAML:

```python
@dataclass(frozen=True)
class FunctionSchema:
    """Parsed function signature"""
    name: OperatorName
    arguments: Arguments
    returns: tuple[Return, ...]

@dataclass(frozen=True)
class OperatorName:
    """Operator name with optional overload"""
    name: BaseOperatorName  # e.g., "add"
    overload_name: str      # e.g., "Tensor"

@dataclass(frozen=True)
class Argument:
    """Function argument"""
    name: str
    type: Type
    default: str | None
    annotation: Annotation | None  # (a!), (a), etc.
```

### torchgen/api/

Converts `FunctionSchema` to different C++ API signatures:

- **dispatcher.py**: Dispatcher calling convention
- **native.py**: Native kernel calling convention
- **python.py**: Python binding signature
- **autograd.py**: Autograd-specific signatures
- **structured.py**: Structured kernel signatures

### torchgen/dest/

"Destinations" that write generated code:

- **register_dispatch_key.py**: Dispatcher registration code
- **native_functions.py**: Native function declarations
- **lazy_ir.py**: Lazy tensor IR
- **ufunc.py**: Universal function wrappers

---

## Build Integration

### CMake Integration

**Location**: `cmake/public/LoadHIP.cmake`, `cmake/Dependencies.cmake`

```cmake
# Run code generation during CMake configure
execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE}
    -m torchgen.gen
    --source-path ${CMAKE_CURRENT_SOURCE_DIR}/aten/src/ATen
    --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
  WORKING_DIRECTORY ${TORCH_ROOT}
)

# Add generated files to build
set(ATen_CPU_SRCS
  ${CMAKE_BINARY_DIR}/aten/src/ATen/RegisterCPU.cpp
  ${CMAKE_BINARY_DIR}/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
  # ... more generated files
)

add_library(torch_cpu SHARED ${ATen_CPU_SRCS})
```

### Generated Files Output

**Build Directory Structure**:
```
build/
├── aten/src/ATen/
│   ├── RegisterCPU.cpp          # CPU dispatcher registration
│   ├── RegisterCUDA.cpp         # CUDA dispatcher registration
│   ├── RegisterMPS.cpp          # MPS dispatcher registration
│   ├── Functions.h              # Function declarations
│   ├── NativeFunctions.h        # Native kernel declarations
│   └── ...
├── torch/csrc/autograd/generated/
│   ├── Functions.cpp            # Autograd backward functions
│   ├── python_functions.cpp     # Python bindings
│   └── variable_factories.h    # Tensor creation functions
```

---

## Custom Operator Registration

For operators **not** in `native_functions.yaml`, use manual registration:

### TORCH_LIBRARY Macro

```cpp
// Define operator schema
TORCH_LIBRARY(my_ops, m) {
  m.def("my_add(Tensor self, Tensor other) -> Tensor");
}

// Implement for CPU
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("my_add", [](const Tensor& self, const Tensor& other) {
    return self + other;  // Implementation
  });
}

// Implement for CUDA
TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("my_add", my_cuda_add);  // Custom CUDA kernel
}
```

**Usage**:
```python
import torch
torch.ops.my_ops.my_add(a, b)
```

---

## MLX Porting Considerations

### Should MLX Adopt Code Generation?

**Pros**:
- ✅ **Reduces boilerplate**: Avoid hand-writing Python bindings for 1000+ ops
- ✅ **Consistency**: Same schema drives C++, Python, docs
- ✅ **Maintainability**: Single source of truth
- ✅ **Automatic variants**: Generate out/inplace variants automatically

**Cons**:
- ❌ **Complexity**: Adds build-time dependency and complexity
- ❌ **Debug difficulty**: Generated code harder to debug
- ❌ **Learning curve**: Team must understand code generation

### Recommendations for MLX

#### Option 1: Simplified YAML Schema (Recommended)

Adopt a **minimal** YAML schema for MLX:

```yaml
# mlx_ops.yaml
- name: add
  signature: "array add(const array& a, const array& b)"
  dispatch:
    metal: add_metal
    cpu: add_cpu
  python_name: add
  grad:
    a: grad
    b: grad
```

**Generate**:
- Python bindings (via pybind11)
- Dispatcher registration
- Basic gradient functions

**Skip**:
- Multiple backends (MLX targets Metal primarily)
- Complex alias analysis (MLX has simpler semantics)
- Structured kernels (less critical with lazy evaluation)

#### Option 2: Manual Registration with Macros

Use C++ macros for registration, skip YAML:

```cpp
// mlx_ops.h
#define MLX_OP(name, signature) \
  m.def(#name, &name, signature);

// Registration:
MLX_OP(add, "array add(const array&, const array&)");
MLX_OP(mul, "array mul(const array&, const array&)");
```

**Pros**: Simpler, no build-time code generation
**Cons**: More manual work, less consistent

#### Option 3: Hybrid Approach

- Use YAML for **high-level operator list**
- Generate **Python bindings only**
- Hand-write C++ kernels and dispatch

This is the **sweet spot** for MLX:
- Automate the tedious part (Python bindings)
- Keep control over performance-critical kernels
- Maintain flexibility

### What to Port from TorchGen

**Port These Ideas**:
1. ✅ **YAML schema** for operator definitions
2. ✅ **Python binding generation** (major time saver)
3. ✅ **Gradient formula specification** (derivatives.yaml)
4. ✅ **Variant generation** (out, inplace)

**Don't Port**:
1. ❌ Complex structured kernel system (overkill for MLX)
2. ❌ Multiple backend dispatch (MLX primarily Metal)
3. ❌ TorchScript integration (MLX doesn't have equivalent)
4. ❌ Alias analysis complexity (MLX semantics simpler)

### Example MLX Code Generation

**Input** (`mlx_ops.yaml`):
```yaml
- name: add
  cpp_signature: "array(const array&, const array&)"
  python_signature: "add(a: array, b: array) -> array"
  grad:
    a: grad_out
    b: grad_out
```

**Generated Python Binding**:
```cpp
// Generated by mlxgen
m.def("add",
  [](const array& a, const array& b) { return mlx::core::add(a, b); },
  py::arg("a"), py::arg("b"),
  "Element-wise addition");
```

**Generated Gradient**:
```cpp
// Generated gradient function
std::vector<array> add_vjp(const array& grad_out) {
  return {grad_out, grad_out};  // Both inputs get full gradient
}
```

---

## Critical Files Reference

**Main Generation**:
- [torchgen/gen.py](../reference/pytorch/torchgen/gen.py) - Main entry point
- [torchgen/model.py](../reference/pytorch/torchgen/model.py) - Data model
- [aten/src/ATen/native/native_functions.yaml](../reference/pytorch/aten/src/ATen/native/native_functions.yaml) - Operator definitions

**APIs**:
- [torchgen/api/dispatcher.py](../reference/pytorch/torchgen/api/dispatcher.py) - Dispatcher signatures
- [torchgen/api/python.py](../reference/pytorch/torchgen/api/python.py) - Python bindings
- [torchgen/api/native.py](../reference/pytorch/torchgen/api/native.py) - Native kernels

**Destinations**:
- [torchgen/dest/register_dispatch_key.py](../reference/pytorch/torchgen/dest/register_dispatch_key.py) - Dispatcher registration
- [tools/autograd/gen_python_functions.py](../reference/pytorch/tools/autograd/gen_python_functions.py) - Python bindings

**Derivatives**:
- [tools/autograd/derivatives.yaml](../reference/pytorch/tools/autograd/derivatives.yaml) - Gradient formulas

---

## Summary

**TorchGen** is PyTorch's solution to the **2,666 operator problem**: How to implement thousands of operators across multiple backends without writing millions of lines of boilerplate code.

**Key Mechanisms**:
- ✅ **Single YAML schema** (`native_functions.yaml`) defines all operators
- ✅ **Code generation** produces dispatcher registration, Python bindings, autograd
- ✅ **Template system** enables flexible code generation
- ✅ **Build integration** ensures generated code is always up-to-date

**For MLX**:
- **Recommended**: Adopt simplified YAML + Python binding generation
- **Skip**: Complex structured kernels, multi-backend dispatch
- **Focus**: Automate Python bindings, maintain control over kernels

**Next Steps**:
1. Read [python-bindings.md](python-bindings.md) for pybind11 integration details
2. Study `native_functions.yaml` schema for specific operator examples
3. Decide on MLX code generation strategy (YAML vs manual)
