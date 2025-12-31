# Custom Functions

## Purpose

`torch.autograd.Function` allows users to define **custom backward passes** for operations not covered by PyTorch's built-in derivatives or requiring special gradient logic. Custom functions enable:
- Implementing operations with custom gradients
- Interfacing with non-differentiable code (C++/CUDA extensions)
- Memory optimization (gradient checkpointing)
- Numerical stability improvements
- Implementing algorithms with non-standard backpropagation

## Basic API

**File**: [torch/autograd/function.py](../reference/pytorch/torch/autograd/function.py)

```python
import torch
from torch.autograd import Function

class MyFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        # 1. Compute forward pass
        output = input1 * input2

        # 2. Save tensors for backward
        ctx.save_for_backward(input1, input2)

        # 3. Save non-tensor data as attributes
        ctx.some_value = 42

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Retrieve saved tensors
        input1, input2 = ctx.saved_tensors

        # 2. Compute input gradients
        grad_input1 = grad_output * input2
        grad_input2 = grad_output * input1

        # 3. Return one gradient per input (or None if non-differentiable)
        return grad_input1, grad_input2

# Usage
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = MyFunction.apply(x, y)
z.backward()

print(x.grad)  # tensor([3.])
print(y.grad)  # tensor([2.])
```

## Context Methods

### save_for_backward()

Save tensors needed during backward:

```python
@staticmethod
def forward(ctx, x, y):
    ctx.save_for_backward(x, y)  # Efficient, uses SavedVariable
    ctx.non_tensor_value = 5     # Save non-tensors as attributes
    return x + y
```

**Why use `save_for_backward`?**:
- Applies saved tensor hooks (for gradient checkpointing)
- Tracks version for in-place operation detection
- Memory-efficient storage
- Required for proper double backward support

### mark_non_differentiable()

Mark outputs that don't need gradients:

```python
class Sort(Function):
    @staticmethod
    def forward(ctx, x):
        sorted_values, indices = x.sort()
        ctx.mark_non_differentiable(indices)  # Indices don't have gradients
        ctx.save_for_backward(indices)
        return sorted_values, indices

    @staticmethod
    def backward(ctx, grad_sorted, grad_indices):
        indices, = ctx.saved_tensors
        grad_input = torch.zeros_like(grad_sorted)
        grad_input.scatter_(0, indices, grad_sorted)
        return grad_input  # Only one gradient (for x)
```

### mark_dirty()

Mark inputs modified in-place:

```python
class InplaceOp(Function):
    @staticmethod
    def forward(ctx, x):
        x.add_(1)  # In-place modification
        ctx.mark_dirty(x)  # Required!
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
```

### set_materialize_grads()

Control whether undefined gradients become zero tensors:

```python
@staticmethod
def forward(ctx, x):
    ctx.set_materialize_grads(False)  # Don't create zero tensors
    return x.clone(), x.clone()

@staticmethod
def backward(ctx, g1, g2):
    # Must check for None explicitly
    grad = torch.zeros_like(ctx.saved_tensors[0])
    if g1 is not None:
        grad += g1
    if g2 is not None:
        grad += g2
    return grad
```

## Advanced Examples

### Example 1: ReLU with Custom Gradient

```python
class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.clamp(min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0  # Zero gradient where output is zero
        return grad_input
```

### Example 2: Straight-Through Estimator

```python
class BinaryQuantize(Function):
    """Quantize to {-1, 1} in forward, use identity in backward"""

    @staticmethod
    def forward(ctx, input):
        return input.sign()  # Forward: quantize to {-1, 1}

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Backward: straight-through (identity)
```

### Example 3: Gradient Clipping

```python
class GradientClip(Function):
    @staticmethod
    def forward(ctx, input, max_norm):
        ctx.max_norm = max_norm
        return input

    @staticmethod
    def backward(ctx, grad_output):
        max_norm = ctx.max_norm
        norm = grad_output.norm()
        if norm > max_norm:
            grad_output = grad_output * (max_norm / norm)
        return grad_output, None  # None for max_norm (non-differentiable)
```

### Example 4: Checkpoint (Memory Optimization)

```python
class Checkpoint(Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        # Recompute forward to get intermediate values
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)
        # Compute gradients using recomputed values
        torch.autograd.backward(outputs, grad_outputs)
        grads = tuple(inp.grad for inp in inputs)
        return (None,) + grads  # None for run_function
```

## Double Backward

For second-order derivatives, omit `@once_differentiable`:

```python
class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        result = x.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        # This expression is differentiable, enabling double backward
        return grad_output * result
```

To disable double backward:

```python
from torch.autograd.function import once_differentiable

class MyFunc(Function):
    @staticmethod
    def forward(ctx, x):
        ...

    @staticmethod
    @once_differentiable  # Error if double backward attempted
    def backward(ctx, grad_output):
        ...
```

## Forward-Mode AD

Implement `jvp` for forward-mode automatic differentiation:

```python
class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        ctx.save_for_forward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x

    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        x, y = ctx.saved_tensors
        # Jacobian-vector product: ∂(x*y)/∂x * x_tangent + ∂(x*y)/∂y * y_tangent
        return x_tangent * y + x * y_tangent
```

## MLX Equivalent

MLX uses `@mx.custom_function` with `@vjp` and `@jvp` decorators:

```python
import mlx.core as mx

@mx.custom_function
def my_function(x, y):
    # Forward pass
    return x * y

@my_function.vjp
def my_function_vjp(primals, cotangents):
    x, y = primals
    g, = cotangents
    # Vector-Jacobian product (backward)
    return (g * y, g * x)

@my_function.jvp
def my_function_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    # Jacobian-vector product (forward)
    primal_out = x * y
    tangent_out = x_dot * y + x * y_dot
    return primal_out, tangent_out
```

### Comparison

| Feature | PyTorch | MLX |
|---------|---------|-----|
| **Class-based** | Yes (`Function` class) | No (decorator-based) |
| **Context** | `ctx.save_for_backward` | Closure captures |
| **Forward** | `@staticmethod forward(ctx, ...)` | Function definition |
| **Backward** | `@staticmethod backward(ctx, grad)` | `@func.vjp` decorator |
| **Forward AD** | `@staticmethod jvp(ctx, tangent)` | `@func.jvp` decorator |
| **Apply** | `MyFunc.apply(x, y)` | `my_function(x, y)` |

### MLX Porting Example

```python
# PyTorch
class CustomOp(Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return x @ weight

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        return grad @ weight.T, x.T @ grad

# MLX
@mx.custom_function
def custom_op(x, weight):
    return x @ weight

@custom_op.vjp
def custom_op_vjp(primals, cotangents):
    x, weight = primals
    g, = cotangents
    return (g @ weight.T, x.T @ g)
```

## Critical File References

- [torch/autograd/function.py](../reference/pytorch/torch/autograd/function.py): Function base class, FunctionCtx
- [torch/csrc/autograd/python_function.cpp](../reference/pytorch/torch/csrc/autograd/python_function.cpp): C++ implementation
- [torch/autograd/__init__.py](../reference/pytorch/torch/autograd/__init__.py): Public API exports
