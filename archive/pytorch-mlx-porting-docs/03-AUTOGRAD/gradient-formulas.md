# Gradient Formulas

## Purpose

PyTorch defines gradient formulas (backward derivatives) for all differentiable operations in [`derivatives.yaml`](../reference/pytorch/tools/autograd/derivatives.yaml). This file specifies:
- How to compute gradients for each operation's inputs
- Forward-mode automatic differentiation (forward AD) formulas
- Which outputs are differentiable
- Saved variables needed for backward

The code generator (`torchgen`) parses this YAML file and generates C++ backward functions that the autograd engine executes during `.backward()`.

## derivatives.yaml Structure

**File**: [tools/autograd/derivatives.yaml](../reference/pytorch/tools/autograd/derivatives.yaml)

### Basic Format

```yaml
- name: <function_signature>
  <input_name>: <gradient_formula>
  <input_name>: <gradient_formula>
  result: <forward_derivative_formula>  # Optional: forward AD
  output_differentiability: [bool, ...]  # Optional: which outputs are differentiable
```

### Simple Examples

```yaml
# Example 1: abs() - absolute value
- name: abs(Tensor self) -> Tensor
  self: grad * self.sgn()
  result: handle_r_to_c(result.scalar_type(), self_t.conj() * self_p.sgn())

# Example 2: add() - element-wise addition
- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(grad, alpha.conj()))
  result: self_t + maybe_multiply(other_t, alpha)

# Example 3: mul() - element-wise multiplication
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
  result: self_t * other_p + self_p * other_t
```

## Available Variables in Formulas

### Output Gradients

```yaml
# Single differentiable output:
self: grad * some_function()  # Use 'grad'

# Multiple differentiable outputs:
input1: grads[0] * ...  # Use 'grads[idx]'
input2: grads[1] * ...

# Named outputs:
values, indices: func() -> (Tensor values, Tensor indices)
  self: grad_values * ...  # Use 'grad_{output_name}'
```

### Input Arguments

All input arguments from `native_functions.yaml` are available:

```yaml
- name: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  mat1: mm_mat1_backward(grad, mat2, mat1.sym_sizes(), ...)
  mat2: mm_mat2_backward(grad, mat1, mat2.sym_sizes(), ...)
```

### Forward Results

```yaml
# Access forward computation result
- name: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
  self: grad * exponent * self.pow(exponent - 1)  # Uses 'self' from forward

# Or explicitly save 'result'
- name: relu(Tensor self) -> Tensor
  self: grad * (result > 0).type_as(grad)  # Uses 'result' from forward
```

### Special Variables

```cpp
// grad_input_mask: std::array<bool, N> indicating which inputs need gradients
self, other: foo(grad_input_mask)
// grad_input_mask[0] == true if self needs grad
// grad_input_mask[1] == true if other needs grad

// wrap_opt_if: Conditional saving to avoid unnecessary storage
saved_var: wrap_opt_if(tensor, grad_input_mask[0])
// Saves 'tensor' only if first input needs gradient

// retain_variables: bool, true if keeping graph for multiple backwards
```

## Common Gradient Patterns

### Element-wise Operations

**Pattern**: Gradient flows through unchanged or multiplied by local derivative

```yaml
# Unary element-wise
- name: exp(Tensor self) -> Tensor
  self: grad * result  # d/dx exp(x) = exp(x)
  result: auto_element_wise

- name: log(Tensor self) -> Tensor
  self: grad / self  # d/dx log(x) = 1/x
  result: auto_element_wise

- name: sin(Tensor self) -> Tensor
  self: grad * self.cos().conj()  # d/dx sin(x) = cos(x)
  result: auto_element_wise

# Binary element-wise
- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: grad  # d/dx (x + y) = 1
  other: grad * alpha  # d/dy (x + alpha*y) = alpha

- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad * other  # d/dx (x * y) = y
  other: grad * self  # d/dy (x * y) = x
```

### Reduction Operations

**Pattern**: Broadcast gradient back to original shape

```yaml
- name: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: sum_backward(grad.to(self.scalar_type()), self.sym_sizes(), dim, keepdim)
  result: auto_linear

# sum_backward implementation:
Tensor sum_backward(Tensor grad, sizes, dims, keepdim) {
  if (!keepdim && sizes.size() > 0) {
    grad = grad.unsqueeze(dims);  // Add dimensions back
  }
  return grad.expand(sizes);  // Broadcast to original shape
}
```

```yaml
- name: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: mean_backward(grad.to(self.scalar_type()), self.sym_sizes(), dim, keepdim)
  result: auto_linear

# mean_backward: same as sum but divide by count
Tensor mean_backward(Tensor grad, sizes, dims, keepdim) {
  auto sum_grad = sum_backward(grad, sizes, dims, keepdim);
  auto numel = compute_numel(sizes, dims);
  return sum_grad / numel;
}
```

### Matrix Operations

**Pattern**: Use matrix transpose and multiplication

```yaml
# Matrix multiplication: C = A @ B
- name: mm(Tensor self, Tensor mat2) -> Tensor
  self: grad.mm(mat2.t().conj())     # dL/dA = dL/dC @ B^T
  mat2: self.t().conj().mm(grad)     # dL/dB = A^T @ dL/dC
  result: self_t.mm(mat2_p) + self_p.mm(mat2_t)

# Batched matrix multiplication: C = A @ B (with batch dims)
- name: bmm(Tensor self, Tensor mat2) -> Tensor
  self: grad.bmm(mat2.transpose(1, 2).conj())
  mat2: self.transpose(1, 2).conj().bmm(grad)
  result: self_t.bmm(mat2_p) + self_p.bmm(mat2_t)

# Matrix-vector multiplication: y = A @ x
- name: mv(Tensor self, Tensor vec) -> Tensor
  self: grad.ger(vec.conj())         # dL/dA = dL/dy ⊗ x^T
  vec: self.t().conj().mv(grad)      # dL/dx = A^T @ dL/dy
  result: self_t.mv(vec_p) + self_p.mv(vec_t)
```

### Activation Functions

```yaml
# ReLU: max(0, x)
- name: relu(Tensor self) -> Tensor
  self: grad * (result > 0).type_as(grad)
  result: self_t * (self_p > 0).type_as(self_t)

# Sigmoid: σ(x) = 1 / (1 + e^(-x))
- name: sigmoid(Tensor self) -> Tensor
  self: sigmoid_backward(grad, result)
  result: auto_element_wise

# sigmoid_backward: σ'(x) = σ(x) * (1 - σ(x))
Tensor sigmoid_backward(Tensor grad, Tensor output) {
  return grad * output * (1 - output);
}

# Tanh: tanh(x)
- name: tanh(Tensor self) -> Tensor
  self: tanh_backward(grad, result)
  result: auto_element_wise

# tanh_backward: tanh'(x) = 1 - tanh(x)^2
Tensor tanh_backward(Tensor grad, Tensor output) {
  return grad * (1 - output * output);
}

# Softmax: exp(x_i) / sum(exp(x_j))
- name: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  self: _softmax_backward_data(grad, result, dim, self.scalar_type())
  result: auto_linear

# _softmax_backward_data: Uses Jacobian matrix multiplication
Tensor _softmax_backward_data(Tensor grad, Tensor output, int dim) {
  // grad_input = output * (grad - sum(grad * output))
  return output * (grad - (grad * output).sum(dim, true));
}
```

### Convolution

```yaml
- name: convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
  input: convolution_backward(grad, input, weight, bias.defined() ? bias.sym_sizes() : std::vector<c10::SymInt>{}, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask)[0]
  weight: convolution_backward(...)[1]
  bias: convolution_backward(...)[2]
  result: convolution_backward_overrideable(...)

# Implementation uses cudnn/mkldnn for actual backward convolution
```

## Advanced Features

### Non-Differentiable Inputs

```yaml
# Explicitly mark inputs as non-differentiable
- name: argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  self: non_differentiable
  output_differentiability: [False]  # Output is also non-differentiable

# Or simply omit gradient formula (implicitly non-differentiable)
- name: any(Tensor self) -> Tensor
  output_differentiability: [False]
```

### Complex Number Support

```yaml
# handle_r_to_c: Convert real gradient to complex if needed
- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(grad, alpha.conj()))

# .conj(): Conjugate for complex gradients (Wirtinger calculus)
- name: acos(Tensor self) -> Tensor
  self: grad * -((-self * self + 1).rsqrt()).conj()
```

### Conditional Gradient Computation

```yaml
# maybe_multiply: Only multiply if alpha != 1
Tensor maybe_multiply(Tensor grad, Scalar alpha) {
  return alpha.equal(1) ? grad : grad * alpha;
}

# Conditional based on grad_input_mask
- name: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: "grad_input_mask[0] ? maybe_multiply(grad, beta.conj()) : Tensor()"
  mat1: "grad_input_mask[1] ? mm_mat1_backward(...) : Tensor()"
  mat2: "grad_input_mask[2] ? mm_mat2_backward(...) : Tensor()"
```

### Not Implemented Gradients

```yaml
# Placeholder for future implementation
- name: acosh_(Tensor(a!) self) -> Tensor(a!)
  self: not_implemented("inplace version of acosh")

# Will raise error if backward is called
```

### Custom Backward Functions

For complex derivatives, implement in C++:

```yaml
- name: max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
  self: max_pool2d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, indices)
  output_differentiability: [True, False]

# Implementation in torch/csrc/autograd/FunctionsManual.cpp
Tensor max_pool2d_with_indices_backward(...) {
  // Complex logic here
}
```

## Forward-Mode AD

Forward derivatives compute Jacobian-vector products in the forward direction:

```yaml
# 'result' specifies forward derivative
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad * other  # Backward: reverse mode
  other: grad * self
  result: self_t * other_p + self_p * other_t  # Forward: JVP

# self_p: primal value of self
# self_t: tangent value of self (forward gradient)
# other_p: primal value of other
# other_t: tangent value of other
# result: forward derivative = ∂(self*other)/∂self * self_t + ∂(self*other)/∂other * other_t
```

### Auto-generation

```yaml
# auto_linear: Automatically generate for linear functions
- name: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  result: auto_linear  # Forward deriv = sum(self_t, dim, keepdim)

# auto_element_wise: Automatically generate for element-wise functions
- name: exp(Tensor self) -> Tensor
  result: auto_element_wise  # Forward deriv = self_t * exp(self_p)
```

## Double Backward (Higher-Order Gradients)

To support `grad(grad(...))`, define derivatives of backward functions:

```yaml
# First-order backward
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad * other
  other: grad * self

# To enable double backward, the formulas above are themselves differentiable
# This happens automatically if you use differentiable operations
```

Example of double backward:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2          # y = x^2
dy_dx = grad(y, x, create_graph=True)[0]  # dy/dx = 2x
d2y_dx2 = grad(dy_dx, x)[0]  # d²y/dx² = 2
```

Generated backward for `pow`:
```yaml
- name: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
  self: grad * exponent * self.pow(exponent - 1)
```

When `create_graph=True`, the expression `grad * exponent * self.pow(exponent - 1)` itself builds a graph, allowing backward through backward.

## Gradient Formula Examples

### Example 1: Log

```yaml
- name: log(Tensor self) -> Tensor
  self: grad / self
  result: auto_element_wise
```

**Forward**: `y = log(x)`
**Backward**: `grad_x = grad_y / x`
**Math**: `∂log(x)/∂x = 1/x`

### Example 2: Matrix Multiply

```yaml
- name: mm(Tensor self, Tensor mat2) -> Tensor
  self: grad.mm(mat2.t().conj())
  mat2: self.t().conj().mm(grad)
```

**Forward**: `C = A @ B`
**Backward**:
- `grad_A = grad_C @ B^T`
- `grad_B = A^T @ grad_C`

**Math**:
```
dL/dA_ij = Σ_k (dL/dC_ik * ∂C_ik/∂A_ij)
         = Σ_k (dL/dC_ik * B_kj)
         = (dL/dC @ B^T)_ij
```

### Example 3: Softmax

```yaml
- name: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  self: _softmax_backward_data(grad, result, dim, self.scalar_type())
```

**Forward**: `y_i = exp(x_i) / Σ_j exp(x_j)`
**Backward**: `grad_x = y * (grad_y - Σ_j (grad_y_j * y_j))`

**Math (Jacobian)**:
```
∂y_i/∂x_j = y_i * (δ_ij - y_j)

grad_x_i = Σ_j (grad_y_j * ∂y_j/∂x_i)
         = Σ_j (grad_y_j * y_j * (δ_ji - y_i))
         = grad_y_i * y_i - y_i * Σ_j (grad_y_j * y_j)
         = y_i * (grad_y_i - Σ_j (grad_y_j * y_j))
```

### Example 4: Batch Normalization

```yaml
- name: native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
  input, weight, bias: "training ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, save_mean, save_invstd, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>()"
  output_differentiability: [True, False, False]
```

Complex backward delegated to C++ function `native_batch_norm_backward`.

## MLX Porting Considerations

### MLX Gradient Definition

MLX uses **function transformations** instead of YAML:

```python
import mlx.core as mx

# PyTorch: YAML derivative
# - name: mul.Tensor(Tensor self, Tensor other) -> Tensor
#   self: grad * other
#   other: grad * self

# MLX: Automatic differentiation
def mul(x, y):
    return x * y

# MLX infers gradient automatically
grad_fn = mx.grad(mul, argnums=[0, 1])
```

### Custom Gradients in MLX

```python
# For custom gradients, use mx.custom_function
@mx.custom_function
def my_function(x):
    # Forward pass
    return x ** 2

@my_function.vjp
def my_function_vjp(primals, cotangents):
    (x,) = primals
    (g,) = cotangents
    # Backward pass
    return (2 * x * g,)

@my_function.jvp
def my_function_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    # Forward pass derivative
    return x ** 2, 2 * x * x_dot
```

### Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Definition** | YAML file | Python decorators or automatic |
| **When Computed** | Code generation (compile time) | Runtime or JIT compilation |
| **Customization** | C++ functions in FunctionsManual.cpp | Python @vjp/@jvp decorators |
| **Auto-diff** | Must explicitly define all derivatives | Automatic for standard ops |
| **Complex Numbers** | Wirtinger calculus explicit | Automatic |

### Porting Strategy

**Don't port derivatives.yaml directly**. Instead:

1. **Use MLX's automatic differentiation** for most operations
2. **Define custom gradients** only when needed:
   - Non-standard mathematical operations
   - Performance optimizations
   - Numerical stability improvements

3. **Example port**:

```python
# PyTorch derivative YAML:
# - name: softmax(Tensor self, int dim) -> Tensor
#   self: _softmax_backward_data(grad, result, dim, ...)

# MLX: Let it auto-differentiate
def softmax(x, dim):
    exp_x = mx.exp(x - mx.max(x, axis=dim, keepdims=True))
    return exp_x / mx.sum(exp_x, axis=dim, keepdims=True)

# MLX will compute correct gradient automatically through chain rule

# Only if needed for performance:
@mx.custom_function
def softmax_optimized(x, dim):
    return softmax(x, dim)

@softmax_optimized.vjp
def softmax_vjp(primals, cotangents):
    (x,) = primals
    (g,) = cotangents
    y = softmax(x, dim)
    # Optimized backward: y * (g - sum(g * y))
    return (y * (g - mx.sum(g * y, axis=dim, keepdims=True)),)
```

## Critical File References

- [tools/autograd/derivatives.yaml](../reference/pytorch/tools/autograd/derivatives.yaml): All gradient definitions (~5000 lines)
- [torch/csrc/autograd/FunctionsManual.cpp](../reference/pytorch/torch/csrc/autograd/FunctionsManual.cpp): Complex backward implementations
- [torch/csrc/autograd/generated/](../reference/pytorch/torch/csrc/autograd/generated/): Generated backward functions
- [torchgen/api/autograd.py](../reference/pytorch/torchgen/api/autograd.py): Code generator for derivatives

## Next Steps

The final autograd document covers:
- **custom-functions.md**: torch.autograd.Function API for user-defined gradients
