# Function Transforms (torch.func)

## Overview

PyTorch's function transforms provide JAX-style composable transformations for vectorization, automatic differentiation, and Jacobian computation. These transforms operate on pure functions and can be composed together for complex operations like per-sample gradients, Hessians, and batched Jacobians.

**Reference Files:**
- `torch/_functorch/apis.py` - Public API (grad, vmap, etc.)
- `torch/_functorch/eager_transforms.py` - Implementation (vjp, jacrev, jvp, jacfwd)
- `torch/_functorch/vmap.py` - Vectorized mapping implementation

## Transform Hierarchy

```
torch.func Transforms
├── Vectorization
│   └── vmap          - Vectorized mapping (batched execution)
├── Gradient Transforms
│   ├── grad          - Compute gradients (scalar output)
│   ├── grad_and_value - Gradients + function value
│   ├── vjp           - Vector-Jacobian product (reverse-mode)
│   └── jvp           - Jacobian-vector product (forward-mode)
├── Jacobian Transforms
│   ├── jacrev        - Jacobian via reverse-mode (vjp + vmap)
│   └── jacfwd        - Jacobian via forward-mode (jvp + vmap)
└── Higher-Order
    └── hessian       - Second-order derivatives
```

---

## vmap (Vectorized Map)

Transforms a function to operate over a batch dimension without explicit loops.

### Function Signature

```python
def vmap(
    func: Callable,
    in_dims: int | tuple = 0,        # Which dimensions are batched in inputs
    out_dims: int | tuple = 0,       # Where to place batch dim in outputs
    randomness: str = "error",       # How to handle random ops
    *,
    chunk_size: int | None = None    # Process in chunks for memory
) -> Callable
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `in_dims` | Input batch dimension(s). Single int or tuple matching arg structure. `None` = not batched |
| `out_dims` | Output batch dimension(s). Where batch dim appears in outputs |
| `randomness` | `"error"` (default), `"same"`, or `"different"` for random ops |
| `chunk_size` | Process batch in chunks to reduce memory |

### Basic Usage

```python
from torch.func import vmap
import torch

# Apply sin to each element in batch
x = torch.randn(64, 10)
result = vmap(torch.sin)(x)  # Shape: (64, 10)

# Matrix-vector multiply for each sample
def mv(matrix, vec):
    return matrix @ vec

batch_matrices = torch.randn(32, 4, 5)
batch_vectors = torch.randn(32, 5)

# Without vmap: need loop
# results = torch.stack([mv(m, v) for m, v in zip(batch_matrices, batch_vectors)])

# With vmap: vectorized
results = vmap(mv)(batch_matrices, batch_vectors)  # Shape: (32, 4)
```

### in_dims Specification

```python
# All inputs batched on dim 0
vmap(fn)(x, y)  # Default: in_dims=0

# First input batched, second not
vmap(fn, in_dims=(0, None))(batch_x, single_y)

# Different batch dimensions
vmap(fn, in_dims=(0, 1))(x, y.T)

# Nested structures
vmap(fn, in_dims=((0, 0), None))(nested_input, config)
```

### Nested vmap (Multi-dimensional Batching)

```python
# Batch over two dimensions
x = torch.randn(B1, B2, N)

# Outer vmap batches over B1, inner over B2
result = vmap(vmap(torch.sin))(x)
```

### Randomness Handling

```python
# Error on random ops (default)
# vmap(lambda x: x + torch.randn_like(x))(batch)  # Raises error

# Same random values for all batch elements
vmap(lambda x: x + torch.randn_like(x), randomness="same")(batch)

# Different random values (requires explicit key management)
vmap(lambda x: x + torch.randn_like(x), randomness="different")(batch)
```

---

## grad

Computes gradients of a scalar-valued function with respect to specified arguments.

### Function Signature

```python
def grad(
    func: Callable,
    argnums: int | tuple[int, ...] = 0,  # Which args to differentiate
    has_aux: bool = False                 # Function returns (output, aux)?
) -> Callable
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `argnums` | Indices of arguments to compute gradients for |
| `has_aux` | If True, func returns `(loss, aux_data)` tuple |

### Basic Usage

```python
from torch.func import grad

# Simple gradient
x = torch.randn(5)
grad_fn = grad(lambda x: x.sin().sum())
gradient = grad_fn(x)  # Same as x.cos()

# Multiple inputs
def f(x, y):
    return (x * y).sum()

x, y = torch.randn(5), torch.randn(5)

# Gradient w.r.t. first arg (default)
dx = grad(f)(x, y)  # = y

# Gradient w.r.t. second arg
dy = grad(f, argnums=1)(x, y)  # = x

# Gradient w.r.t. both
dx, dy = grad(f, argnums=(0, 1))(x, y)
```

### Auxiliary Return Values

```python
from torch.func import grad

def loss_with_metrics(x, target):
    pred = x.sin()
    loss = ((pred - target) ** 2).sum()
    metrics = {"pred_mean": pred.mean(), "loss": loss}
    return loss, metrics  # (output, aux)

x = torch.randn(5, requires_grad=True)
target = torch.randn(5)

# Get gradient and auxiliary data
grad_fn = grad(loss_with_metrics, has_aux=True)
gradient, metrics = grad_fn(x, target)
```

### Higher-Order Gradients

```python
from torch.func import grad

def f(x):
    return x.sin().sum()

x = torch.randn(5)

# First derivative
first = grad(f)(x)  # cos(x)

# Second derivative (Hessian diagonal)
second = grad(grad(f))(x)  # -sin(x)

# Third derivative
third = grad(grad(grad(f)))(x)  # -cos(x)
```

---

## grad_and_value

Returns both the gradient and the function value.

### Function Signature

```python
def grad_and_value(
    func: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False
) -> Callable
```

### Usage

```python
from torch.func import grad_and_value

def loss_fn(x):
    return x.pow(2).sum()

x = torch.randn(5)

# Get both gradient and loss value
gradient, loss_value = grad_and_value(loss_fn)(x)
# gradient = 2 * x
# loss_value = x.pow(2).sum()
```

---

## vjp (Vector-Jacobian Product)

Computes the vector-Jacobian product (reverse-mode AD). Returns both the function output and a function to compute VJPs.

### Function Signature

```python
def vjp(
    func: Callable,
    *primals,                    # Input values
    has_aux: bool = False
) -> tuple[output, vjp_fn] | tuple[output, vjp_fn, aux]
```

### Relationship to Jacobian

For `y = f(x)` with Jacobian `J = ∂y/∂x`:
- `vjp(f, x)` returns `(y, vjp_fn)`
- `vjp_fn(v)` computes `v @ J` (row of Jacobian if v is one-hot)

### Usage

```python
from torch.func import vjp

def f(x):
    return x.sin()

x = torch.randn(5)
output, vjp_fn = vjp(f, x)

# Compute VJP with cotangent vector
cotangent = torch.ones_like(output)
(grad_x,) = vjp_fn(cotangent)  # Same as grad(f)(x) when cotangent is 1
```

### Multiple Outputs

```python
from torch.func import vjp

def f(x):
    return x.sin(), x.cos()

x = torch.randn(5)
(sin_x, cos_x), vjp_fn = vjp(f, x)

# VJP needs cotangents for each output
cotangents = (torch.ones(5), torch.ones(5))
(grad_x,) = vjp_fn(cotangents)
# grad_x = cos(x) + (-sin(x)) = cos(x) - sin(x)
```

### Multiple Inputs

```python
from torch.func import vjp

def matmul(x, y):
    return x @ y

x = torch.randn(5, 4)
y = torch.randn(4, 5)
output, vjp_fn = vjp(matmul, x, y)

cotangent = torch.randn(5, 5)
grad_x, grad_y = vjp_fn(cotangent)
# grad_x = cotangent @ y.T
# grad_y = x.T @ cotangent
```

---

## jvp (Jacobian-Vector Product)

Computes the Jacobian-vector product (forward-mode AD). Evaluates function and tangent simultaneously.

### Function Signature

```python
def jvp(
    func: Callable,
    primals: tuple,      # Input values
    tangents: tuple,     # Tangent vectors (directional derivatives)
    *,
    strict: bool = False,
    has_aux: bool = False
) -> tuple[outputs, jvp_outputs]
```

### Relationship to Jacobian

For `y = f(x)` with Jacobian `J = ∂y/∂x`:
- `jvp(f, (x,), (v,))` returns `(y, J @ v)` (column of Jacobian if v is one-hot)

### Usage

```python
from torch.func import jvp

def f(x):
    return x.sin()

x = torch.randn(5)
tangent = torch.randn(5)  # Direction to differentiate

output, jvp_out = jvp(f, (x,), (tangent,))
# output = sin(x)
# jvp_out = cos(x) * tangent (directional derivative)
```

### Computing Jacobian Columns

```python
from torch.func import jvp

def f(x):
    return x ** 2

x = torch.randn(3)

# Get column 0 of Jacobian
e0 = torch.tensor([1., 0., 0.])
_, col0 = jvp(f, (x,), (e0,))

# Get column 1 of Jacobian
e1 = torch.tensor([0., 1., 0.])
_, col1 = jvp(f, (x,), (e1,))
```

---

## jacrev (Jacobian via Reverse Mode)

Computes the full Jacobian using reverse-mode AD (vjp + vmap).

### Function Signature

```python
def jacrev(
    func: Callable,
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: bool = False,
    chunk_size: int | None = None      # Memory-efficient chunked computation
) -> Callable
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `argnums` | Which arguments to compute Jacobian w.r.t. |
| `chunk_size` | Compute Jacobian in chunks (reduces memory, `1` = row-by-row) |

### Usage

```python
from torch.func import jacrev

def f(x):
    return x.sin()

x = torch.randn(5)
jacobian = jacrev(f)(x)  # Shape: (5, 5) - diagonal matrix of cos(x)

# Verify it's diagonal
expected = torch.diag(x.cos())
assert torch.allclose(jacobian, expected)
```

### When to Use jacrev

**jacrev is efficient when:**
- Output dimension < Input dimension
- `m << n` for function `f: R^n -> R^m`

**Complexity:** O(m) backward passes, where m = output dimension

### Batched Jacobians

```python
from torch.func import jacrev, vmap

def f(x):
    return x @ x  # x^2

batch_x = torch.randn(32, 5)

# Jacobian for each sample in batch
batched_jacobian = vmap(jacrev(f))(batch_x)  # Shape: (32, 5, 5)
```

---

## jacfwd (Jacobian via Forward Mode)

Computes the full Jacobian using forward-mode AD (jvp + vmap).

### Function Signature

```python
def jacfwd(
    func: Callable,
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: bool = False,
    randomness: str = "error"
) -> Callable
```

### When to Use jacfwd

**jacfwd is efficient when:**
- Input dimension < Output dimension
- `n << m` for function `f: R^n -> R^m`

**Complexity:** O(n) forward passes, where n = input dimension

### Comparison: jacrev vs jacfwd

| Scenario | Preferred Transform |
|----------|-------------------|
| Scalar output (loss) | `jacrev` or `grad` |
| Many outputs, few inputs | `jacfwd` |
| Few outputs, many inputs | `jacrev` |
| Square Jacobian | Either (similar cost) |

---

## hessian

Computes the Hessian (second-order derivatives) of a scalar-valued function.

### Function Signature

```python
def hessian(
    func: Callable,
    argnums: int = 0
) -> Callable
```

### Implementation

Internally implemented as `jacrev(jacrev(func))`:

```python
def hessian(func, argnums=0):
    return jacrev(jacrev(func, argnums=argnums), argnums=argnums)
```

### Usage

```python
from torch.func import hessian

def f(x):
    return x.sin().sum()

x = torch.randn(5)
H = hessian(f)(x)  # Shape: (5, 5)

# For this function, Hessian is diagonal: -diag(sin(x))
expected = torch.diag(-x.sin())
assert torch.allclose(H, expected)
```

---

## Composing Transforms

The power of function transforms comes from composition.

### Per-Sample Gradients

```python
from torch.func import grad, vmap

def loss_fn(params, x, y):
    pred = x @ params
    return ((pred - y) ** 2).mean()

# Gradient w.r.t. params for single sample
grad_fn = grad(loss_fn)

# Per-sample gradients (no reduce over batch)
def per_sample_loss(params, x, y):
    pred = x @ params
    return (pred - y) ** 2  # No mean!

# vmap over samples, grad over params
per_sample_grads = vmap(grad(per_sample_loss), in_dims=(None, 0, 0))

params = torch.randn(10, 1)
batch_x = torch.randn(32, 10)
batch_y = torch.randn(32, 1)

grads = per_sample_grads(params, batch_x, batch_y)  # Shape: (32, 10, 1)
```

### Hessian-Vector Product

```python
from torch.func import grad, jvp

def hvp(f, x, v):
    """Hessian-vector product: H @ v"""
    return jvp(grad(f), (x,), (v,))[1]

def f(x):
    return x.pow(3).sum()

x = torch.randn(5)
v = torch.randn(5)

hvp_result = hvp(f, x, v)  # 6 * x * v (diagonal Hessian)
```

### Fisher-Vector Product

```python
from torch.func import grad, jvp, vmap

def fvp(log_likelihood, params, data, v):
    """Fisher-vector product."""
    def loss(p):
        return -vmap(log_likelihood, in_dims=(None, 0))(p, data).mean()

    # Fisher = E[grad @ grad.T], FVP computed via JVP of gradient
    _, jvp_out = jvp(grad(loss), (params,), (v,))
    return jvp_out
```

---

## Functional API for Modules

### make_functional (Deprecated)

For working with `nn.Module` in a functional style:

```python
from torch.func import functional_call

model = nn.Linear(10, 5)

# Get parameters as a dict
params = dict(model.named_parameters())

# Functional forward pass
def func_forward(params, x):
    return functional_call(model, params, (x,))

# Now can use with transforms
x = torch.randn(32, 10)
jacobian = vmap(jacrev(lambda p: functional_call(model, p, (x[0],))))(params)
```

---

## MLX Mapping

### Direct Conceptual Alignment

MLX's design is heavily inspired by JAX and uses the same transform-based approach:

| PyTorch (`torch.func`) | MLX |
|------------------------|-----|
| `vmap` | `mx.vmap` |
| `grad` | `mx.grad` |
| `grad_and_value` | `mx.value_and_grad` |
| `vjp` | `mx.vjp` |
| `jvp` | `mx.jvp` |
| `jacrev` | Compose `mx.vjp` + `mx.vmap` |
| `jacfwd` | Compose `mx.jvp` + `mx.vmap` |

### Key MLX Differences

1. **Primary paradigm**: MLX is designed around transforms from the ground up
2. **Lazy evaluation**: MLX arrays are lazy; transforms work on computation graphs
3. **Compilation**: `mx.compile` can be combined with transforms for optimization

### Example Translations

```python
# PyTorch
from torch.func import grad, vmap

@torch.func.grad
def pytorch_loss_grad(params, x, y):
    return loss(params, x, y)

batched_grads = vmap(pytorch_loss_grad, in_dims=(None, 0, 0))

# MLX equivalent
import mlx.core as mx

@mx.grad
def mlx_loss_grad(params, x, y):
    return loss(params, x, y)

batched_grads = mx.vmap(mlx_loss_grad, in_axes=(None, 0, 0))
```

---

## Limitations and Considerations

1. **Pure functions**: Transforms work best on pure functions without side effects
2. **Tensor outputs**: Functions must return Tensors (not Python scalars)
3. **In-place operations**: Avoid in-place ops in transformed functions
4. **Random operations**: Require explicit `randomness` parameter in vmap
5. **torch.no_grad**: Transforms respect inner no_grad, ignore outer

### no_grad Behavior

```python
from torch.func import grad

def f(x):
    with torch.no_grad():
        c = x ** 2  # Not differentiated
    return x - c

# grad respects inner no_grad
gradient = grad(f)(x)  # Gradient of (x - c) = 1, not (x - x^2)

# Outer no_grad is ignored by transforms
with torch.no_grad():
    gradient = grad(f)(x)  # Still computes gradient!
```

---

## Summary

| Transform | Input | Output | Use Case |
|-----------|-------|--------|----------|
| `vmap` | `f(x)` | `f(batch_x)` | Batched execution |
| `grad` | `f: R^n -> R` | `∇f: R^n -> R^n` | Scalar gradients |
| `vjp` | `f, x, v` | `f(x), v @ J` | Reverse-mode AD |
| `jvp` | `f, x, v` | `f(x), J @ v` | Forward-mode AD |
| `jacrev` | `f: R^n -> R^m` | `J: R^n -> R^(m×n)` | Full Jacobian (m < n) |
| `jacfwd` | `f: R^n -> R^m` | `J: R^n -> R^(m×n)` | Full Jacobian (n < m) |
| `hessian` | `f: R^n -> R` | `H: R^n -> R^(n×n)` | Second derivatives |
