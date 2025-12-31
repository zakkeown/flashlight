# PyTorch to MLX Mapping Guide

## Purpose

This document provides comprehensive mappings between PyTorch and MLX (Apple's Machine Learning framework for Metal). It covers:
- Core tensor operations and their MLX equivalents
- Architecture differences and design philosophy
- API compatibility strategies
- Common porting patterns and pitfalls
- Performance considerations specific to Metal

This is the primary reference for developers porting PyTorch code to MLX or building compatibility layers.

## Philosophical Differences

### Execution Model

**PyTorch**:
- **Eager execution**: Operations execute immediately when called
- **Explicit graph construction**: Autograd builds graph during forward pass
- **Imperative style**: Standard Python control flow works naturally

**MLX**:
- **Lazy evaluation**: Operations are queued and fused for optimization
- **Implicit graph**: Graph built when needed (e.g., during `mx.grad()`)
- **Functional style**: Encourages pure functions and immutability

**Example**:
```python
# PyTorch: Eager
x = torch.randn(100, 100)
y = x + 1          # Executes immediately
z = y * 2          # Executes immediately
print(z.sum())     # Result available

# MLX: Lazy
x = mx.random.normal((100, 100))
y = x + 1          # Queued
z = y * 2          # Queued
mx.eval(z)         # Now execute the fused operation
print(z.sum())     # Forces evaluation
```

### Memory Model

**PyTorch**:
- **Discrete memory**: Separate CPU and GPU memory
- **Explicit transfers**: `.to(device)`, `.cuda()`, `.cpu()`
- **Reference counting**: Tensors manage their own memory

**MLX**:
- **Unified memory**: Single address space shared between CPU and Metal GPU
- **No transfers needed**: Arrays accessible from both CPU and GPU
- **Automatic management**: Memory handled by Metal framework

**Example**:
```python
# PyTorch: Explicit device management
x_cpu = torch.randn(100, 100)
x_gpu = x_cpu.cuda()                # Copy to GPU
y = x_gpu + 1                       # GPU operation
result = y.cpu()                    # Copy back to CPU

# MLX: Unified memory
x = mx.random.normal((100, 100))    # Already on Metal
y = x + 1                           # Automatically on Metal
result = np.array(y)                # Access from CPU when needed
```

### Mutability

**PyTorch**:
- **In-place operations**: `x.add_(1)`, `x[0] = 5`
- **Version tracking**: Detects illegal in-place modifications
- **Mixed mutability**: Some operations in-place, some not

**MLX**:
- **Immutable arrays**: All arrays are immutable
- **Functional updates**: Operations return new arrays
- **No version tracking needed**: Cannot modify arrays after creation

**Example**:
```python
# PyTorch: In-place
x = torch.randn(5, requires_grad=True)
x.add_(1)  # Modifies x in-place
x[0] = 10  # Direct assignment

# MLX: Functional
x = mx.random.normal((5,))
x = x + 1  # Creates new array, reassigns variable
# x[0] = 10  # ERROR: Cannot modify immutable array
x = mx.concatenate([mx.array([10]), x[1:]])  # Must reconstruct
```

## Tensor/Array Creation

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.tensor([1, 2, 3])` | `mx.array([1, 2, 3])` | From Python list |
| `torch.from_numpy(arr)` | `mx.array(arr)` | From NumPy array |
| `torch.zeros(3, 4)` | `mx.zeros((3, 4))` | Note: tuple for shape |
| `torch.ones(3, 4)` | `mx.ones((3, 4))` | |
| `torch.empty(3, 4)` | Not available | MLX doesn't support uninitialized arrays |
| `torch.full((3, 4), 7)` | `mx.full((3, 4), 7)` | |
| `torch.arange(0, 10, 2)` | `mx.arange(0, 10, 2)` | |
| `torch.linspace(0, 1, 100)` | `mx.linspace(0, 1, 100)` | |
| `torch.eye(5)` | `mx.eye(5)` | Identity matrix |
| `torch.randn(3, 4)` | `mx.random.normal((3, 4))` | Note: different API |
| `torch.rand(3, 4)` | `mx.random.uniform(shape=(3, 4))` | |
| `torch.randint(0, 10, (3, 4))` | `mx.random.randint(0, 10, (3, 4))` | |

### Random Number Generation Differences

**PyTorch**:
```python
# Generator object for reproducibility
gen = torch.Generator().manual_seed(42)
x = torch.randn(10, generator=gen)

# Global seed
torch.manual_seed(42)
x = torch.randn(10)
```

**MLX**:
```python
# Key-based PRNG (like JAX)
key = mx.random.key(42)
x = mx.random.normal((10,), key=key)

# Global seed (simpler but not composable)
mx.random.seed(42)
x = mx.random.normal((10,))
```

## Data Types

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.float32` / `torch.float` | `mx.float32` | Default floating point |
| `torch.float64` / `torch.double` | Not supported | MLX doesn't have float64 |
| `torch.float16` / `torch.half` | `mx.float16` | Half precision |
| `torch.bfloat16` | `mx.bfloat16` | Brain floating point |
| `torch.int32` / `torch.int` | `mx.int32` | |
| `torch.int64` / `torch.long` | `mx.int64` | |
| `torch.int16` / `torch.short` | `mx.int16` | |
| `torch.int8` | `mx.int8` | |
| `torch.uint8` | `mx.uint8` | |
| `torch.bool` | `mx.bool_` | Note: underscore in MLX |
| `torch.complex64` | `mx.complex64` | |
| `torch.complex128` | Not supported | No float64 → no complex128 |

### Type Conversion

```python
# PyTorch
x = torch.randn(10)
x_int = x.to(torch.int32)
x_half = x.half()
x_double = x.double()

# MLX
x = mx.random.normal((10,))
x_int = x.astype(mx.int32)
x_half = x.astype(mx.float16)
# x_double = x.astype(mx.float64)  # ERROR: Not supported
```

## Basic Operations

### Arithmetic

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `a + b` | `a + b` | Element-wise addition |
| `a - b` | `a - b` | Element-wise subtraction |
| `a * b` | `a * b` | Element-wise multiplication |
| `a / b` | `a / b` | Element-wise division |
| `a ** b` | `a ** b` | Element-wise power |
| `a @ b` | `a @ b` | Matrix multiplication |
| `torch.add(a, b, alpha=2)` | `a + 2 * b` | MLX doesn't have alpha parameter |
| `torch.addcdiv(a, b, c, value=0.1)` | `a + 0.1 * (b / c)` | No fused operation |
| `torch.addcmul(a, b, c, value=0.1)` | `a + 0.1 * (b * c)` | No fused operation |

### Mathematical Functions

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.abs(x)` | `mx.abs(x)` | |
| `torch.exp(x)` | `mx.exp(x)` | |
| `torch.log(x)` | `mx.log(x)` | Natural log |
| `torch.log10(x)` | `mx.log10(x)` | |
| `torch.log2(x)` | `mx.log2(x)` | |
| `torch.sqrt(x)` | `mx.sqrt(x)` | |
| `torch.rsqrt(x)` | `1 / mx.sqrt(x)` | No direct rsqrt |
| `torch.sin(x)` | `mx.sin(x)` | |
| `torch.cos(x)` | `mx.cos(x)` | |
| `torch.tan(x)` | `mx.tan(x)` | |
| `torch.sigmoid(x)` | `mx.sigmoid(x)` | |
| `torch.tanh(x)` | `mx.tanh(x)` | |

### Activation Functions

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `F.relu(x)` | `mx.maximum(x, 0)` or `nn.relu(x)` | |
| `F.gelu(x)` | `nn.gelu(x)` | |
| `F.silu(x)` / `F.swish(x)` | `nn.silu(x)` | |
| `F.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` | Note: axis not dim |
| `F.log_softmax(x, dim=-1)` | `mx.log_softmax(x, axis=-1)` | |
| `F.softplus(x)` | `mx.softplus(x)` | |
| `F.leaky_relu(x, 0.01)` | `nn.leaky_relu(x, 0.01)` | |

## Indexing and Slicing

### Basic Indexing

```python
# PyTorch
x = torch.randn(10, 20, 30)
a = x[0]          # First element along dim 0
b = x[:, 5]       # All of dim 0, 5th of dim 1
c = x[..., -1]    # Last element along dim 2

# MLX: Same syntax
x = mx.random.normal((10, 20, 30))
a = x[0]
b = x[:, 5]
c = x[..., -1]
```

### Advanced Indexing

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `x[mask]` | `x[mask]` | Boolean indexing |
| `x[[0, 2, 4]]` | `x[[0, 2, 4]]` | Integer array indexing |
| `x[torch.tensor([0, 2, 4])]` | `x[mx.array([0, 2, 4])]` | Tensor indexing |
| `x[0, [1, 2], :]` | `x[0, mx.array([1, 2]), :]` | Mixed indexing |

### Gather/Scatter

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.gather(x, 1, idx)` | `mx.take_along_axis(x, idx, axis=1)` | Different name |
| `torch.scatter(x, 1, idx, src)` | No direct equivalent | Need custom implementation |
| `torch.index_select(x, 0, idx)` | `mx.take(x, idx, axis=0)` | |
| `x.masked_fill(mask, value)` | `mx.where(mask, value, x)` | Functional, not in-place |
| `torch.where(cond, a, b)` | `mx.where(cond, a, b)` | Same API |

## Shape Manipulation

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `x.view(2, -1)` | `mx.reshape(x, (2, -1))` | MLX requires tuple |
| `x.reshape(2, -1)` | `mx.reshape(x, (2, -1))` | |
| `x.transpose(0, 1)` | `mx.transpose(x, (1, 0))` | MLX uses permutation tuple |
| `x.permute(2, 0, 1)` | `mx.transpose(x, (2, 0, 1))` | Same as transpose in MLX |
| `x.squeeze()` | `mx.squeeze(x)` | |
| `x.squeeze(0)` | `mx.squeeze(x, axis=0)` | Note: axis not dim |
| `x.unsqueeze(0)` | `mx.expand_dims(x, axis=0)` | Different name |
| `x.flatten()` | `mx.flatten(x)` | |
| `x.flatten(1)` | `mx.reshape(x, (x.shape[0], -1))` | MLX doesn't support start_dim |
| `x.repeat(2, 3, 4)` | `mx.tile(x, (2, 3, 4))` | Different name |
| `x.expand(10, -1, -1)` | `mx.broadcast_to(x, (10, ) + x.shape[1:])` | More explicit |

### Concatenation and Stacking

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.cat([a, b], dim=0)` | `mx.concatenate([a, b], axis=0)` | Note: axis not dim |
| `torch.stack([a, b], dim=0)` | `mx.stack([a, b], axis=0)` | |
| `torch.split(x, 3, dim=0)` | `mx.split(x, 3, axis=0)` | |
| `torch.chunk(x, 3, dim=0)` | No direct equivalent | Use split with computed sizes |
| `torch.unbind(x, dim=0)` | No direct equivalent | Use list comprehension with indexing |

## Reduction Operations

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `x.sum()` | `mx.sum(x)` | |
| `x.sum(dim=0)` | `mx.sum(x, axis=0)` | Note: axis not dim |
| `x.sum(dim=0, keepdim=True)` | `mx.sum(x, axis=0, keepdims=True)` | Note: keepdims |
| `x.mean()` | `mx.mean(x)` | |
| `x.mean(dim=0)` | `mx.mean(x, axis=0)` | |
| `x.max()` | `mx.max(x)` | |
| `x.max(dim=0)` | `mx.max(x, axis=0)` | Returns values only (no indices) |
| `torch.max(x, dim=0)` | `mx.argmax(x, axis=0)` for indices | Different API |
| `x.min()` | `mx.min(x)` | |
| `x.argmax()` | `mx.argmax(x)` | |
| `x.argmax(dim=0)` | `mx.argmax(x, axis=0)` | |
| `x.std()` | `mx.std(x)` | |
| `x.var()` | `mx.var(x)` | |
| `x.prod()` | `mx.prod(x)` | |
| `x.cumsum(dim=0)` | `mx.cumsum(x, axis=0)` | |
| `x.cumprod(dim=0)` | `mx.cumprod(x, axis=0)` | |

## Linear Algebra

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.mm(a, b)` | `mx.matmul(a, b)` or `a @ b` | Matrix multiplication |
| `torch.bmm(a, b)` | `mx.matmul(a, b)` | MLX matmul handles batches |
| `torch.matmul(a, b)` | `mx.matmul(a, b)` | |
| `torch.mv(mat, vec)` | `mx.matmul(mat, vec)` | Matrix-vector product |
| `torch.dot(a, b)` | `mx.sum(a * b)` | Dot product (no direct function) |
| `torch.inner(a, b)` | `mx.inner(a, b)` | |
| `torch.outer(a, b)` | `mx.outer(a, b)` | |
| `torch.addmm(bias, a, b)` | `bias + a @ b` | No fused operation |
| `torch.baddbmm(bias, a, b)` | `bias + mx.matmul(a, b)` | |
| `torch.linalg.inv(a)` | `mx.linalg.inv(a)` | Matrix inverse |
| `torch.linalg.solve(A, b)` | `mx.linalg.solve(A, b)` | Solve linear system |
| `torch.linalg.cholesky(a)` | `mx.linalg.cholesky(a)` | |
| `torch.linalg.qr(a)` | `mx.linalg.qr(a)` | |
| `torch.linalg.svd(a)` | `mx.linalg.svd(a)` | |
| `torch.linalg.eig(a)` | Not available | No eigenvalue decomposition yet |
| `torch.linalg.norm(x, ord=2)` | `mx.linalg.norm(x, ord=2)` | |

## Convolution and Pooling

### Convolution

**PyTorch**:
```python
import torch.nn.functional as F

# Conv2d: (N, C_in, H, W) format (channels-first/NCHW)
x = torch.randn(1, 3, 32, 32)
weight = torch.randn(64, 3, 3, 3)
out = F.conv2d(x, weight, stride=1, padding=1)  # (1, 64, 32, 32)
```

**MLX**:
```python
import mlx.nn as nn

# Conv2d: (N, H, W, C_in) format (channels-last/NHWC)
x = mx.random.normal((1, 32, 32, 3))
weight = mx.random.normal((3, 3, 3, 64))  # (kH, kW, C_in, C_out)
out = mx.conv2d(x, weight, stride=1, padding=1)  # (1, 32, 32, 64)
```

**Critical Layout Difference**:
```python
# Converting PyTorch NCHW to MLX NHWC
def torch_to_mlx_layout(x_torch):
    # PyTorch: (N, C, H, W)
    # MLX:     (N, H, W, C)
    return mx.transpose(x_torch, (0, 2, 3, 1))

def mlx_to_torch_layout(x_mlx):
    # MLX:     (N, H, W, C)
    # PyTorch: (N, C, H, W)
    return mx.transpose(x_mlx, (0, 3, 1, 2))

# Weight conversion
def torch_weight_to_mlx(weight_torch):
    # PyTorch: (C_out, C_in, kH, kW)
    # MLX:     (kH, kW, C_in, C_out)
    return mx.transpose(weight_torch, (2, 3, 1, 0))
```

### Pooling

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `F.max_pool2d(x, 2)` | `nn.max_pool2d(x, 2)` | NCHW vs NHWC layout |
| `F.avg_pool2d(x, 2)` | `nn.avg_pool2d(x, 2)` | |
| `F.adaptive_avg_pool2d(x, 1)` | No direct equivalent | Implement with mean |

## Normalization

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `F.batch_norm(x, mean, var, weight, bias)` | `nn.batch_norm(x, weight, bias, mean, var)` | Different arg order |
| `F.layer_norm(x, shape, weight, bias)` | `nn.layer_norm(x, shape, weight, bias)` | Similar |
| `F.group_norm(x, groups, weight, bias)` | `nn.group_norm(x, groups, weight, bias)` | |
| `F.instance_norm(x)` | No built-in | Implement with layer_norm |

## Autograd

### Basic Gradient Computation

**PyTorch**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])

# Multiple outputs
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = x * y
grads = torch.autograd.grad(z, [x, y])
print(grads)  # (tensor([3.]), tensor([2.]))
```

**MLX**:
```python
def f(x):
    return x ** 2

grad_f = mx.grad(f)
x = mx.array([2.0])
print(grad_f(x))  # array([4.])

# Multiple arguments
def f(x, y):
    return x * y

grad_f = mx.grad(f, argnums=[0, 1])  # Gradients for both args
x, y = mx.array([2.0]), mx.array([3.0])
grads = grad_f(x, y)
print(grads)  # (array([3.]), array([2.]))
```

### Value and Gradient

**PyTorch**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
value = y.item()
grad = x.grad.item()
```

**MLX**:
```python
def f(x):
    return (x ** 2).sum()

value_and_grad_f = mx.value_and_grad(f)
x = mx.array([2.0])
value, grad = value_and_grad_f(x)
print(value, grad)  # (array(4.), array([4.]))
```

### Custom Gradients

**PyTorch**:
```python
from torch.autograd import Function

class MyFunc(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return 2 * x * grad_output
```

**MLX**:
```python
@mx.custom_function
def my_func(x):
    return x ** 2

@my_func.vjp
def my_func_vjp(primals, cotangents):
    x, = primals
    g, = cotangents
    return (2 * x * g,)
```

## Neural Network Modules

### Simple Layer

**PyTorch**:
```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

model = Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training step
loss = criterion(model(input), target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**MLX**:
```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        scale = (in_features) ** -0.5
        self.weight = mx.random.uniform(-scale, scale, (out_features, in_features))
        self.bias = mx.zeros(out_features)

    def __call__(self, x):
        return x @ self.weight.T + self.bias

model = Linear(784, 10)
optimizer = optim.Adam(learning_rate=0.001)

# Training step
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, input, target)
optimizer.update(model, grads)
```

### Model Composition

**PyTorch**:
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
```

**MLX**:
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

## Optimizers

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.optim.SGD(params, lr=0.01)` | `optim.SGD(learning_rate=0.01)` | Different API |
| `torch.optim.Adam(params, lr=0.001)` | `optim.Adam(learning_rate=0.001)` | |
| `torch.optim.AdamW(params, lr=0.001)` | `optim.AdamW(learning_rate=0.001)` | |

**PyTorch Update**:
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**MLX Update**:
```python
loss, grads = loss_and_grad_fn(model, x, y)
optimizer.update(model, grads)
```

## Loss Functions

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `F.mse_loss(pred, target)` | `mx.mean((pred - target) ** 2)` | No built-in |
| `F.cross_entropy(logits, target)` | `nn.losses.cross_entropy(logits, target)` | |
| `F.binary_cross_entropy(pred, target)` | Custom implementation needed | |
| `F.nll_loss(log_probs, target)` | Use cross_entropy | |

## Device Management

### PyTorch: Explicit

```python
# Move to GPU
model = model.cuda()
x = x.to('cuda')

# Move to CPU
model = model.cpu()
x = x.cpu()

# Specific device
device = torch.device('cuda:0')
model = model.to(device)
x = x.to(device)
```

### MLX: Automatic (Metal)

```python
# Everything is automatically on Metal
model = MyModel()
x = mx.random.normal((10, 784))

# No explicit device management needed
# Unified memory means CPU can access Metal arrays directly
```

## Data Loading

### PyTorch DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    output = model(batch_x)
    loss = criterion(output, batch_y)
    ...
```

### MLX: Manual Batching

```python
def batch_iterate(x, y, batch_size, shuffle=True):
    indices = mx.arange(len(x))
    if shuffle:
        indices = mx.random.permutation(indices)

    for i in range(0, len(x), batch_size):
        batch_indices = indices[i:i+batch_size]
        yield x[batch_indices], y[batch_indices]

for batch_x, batch_y in batch_iterate(x_train, y_train, 32):
    output = model(batch_x)
    loss = loss_fn(output, batch_y)
    ...
```

## Common Pitfalls and Solutions

### 1. Layout Confusion (NCHW vs NHWC)

**Problem**:
```python
# PyTorch model expects (N, C, H, W)
# MLX uses (N, H, W, C)
```

**Solution**:
```python
def convert_pytorch_conv_to_mlx(pytorch_weight):
    # PyTorch: (out_channels, in_channels, kH, kW)
    # MLX: (kH, kW, in_channels, out_channels)
    return mx.transpose(pytorch_weight, (2, 3, 1, 0))

def convert_input_to_mlx(pytorch_input):
    # PyTorch: (N, C, H, W)
    # MLX: (N, H, W, C)
    return mx.transpose(pytorch_input, (0, 2, 3, 1))
```

### 2. In-Place Operations

**Problem**:
```python
# PyTorch
x.add_(1)  # Works

# MLX
x.add_(1)  # ERROR: Arrays are immutable
```

**Solution**:
```python
# Use functional updates
x = x + 1

# Or for complex updates
x = mx.where(condition, new_value, x)
```

### 3. Dimension vs Axis

**Problem**:
```python
# PyTorch uses 'dim'
x.sum(dim=0)

# MLX uses 'axis'
x.sum(axis=0)
```

**Solution**: Create compatibility wrapper
```python
def torch_sum(x, dim=None, keepdim=False):
    axis = dim  # Rename parameter
    keepdims = keepdim  # Rename parameter
    return mx.sum(x, axis=axis, keepdims=keepdims)
```

### 4. Gradient Accumulation

**Problem**:
```python
# PyTorch: Gradients accumulate automatically
loss1.backward()
loss2.backward()
optimizer.step()  # Updates with sum of gradients

# MLX: Must accumulate manually
```

**Solution**:
```python
# Manually accumulate gradients
total_grads = None
for batch in batches:
    loss, grads = loss_and_grad_fn(model, batch)
    if total_grads is None:
        total_grads = grads
    else:
        total_grads = tree_map(lambda a, b: a + b, total_grads, grads)

optimizer.update(model, total_grads)
```

### 5. Missing float64

**Problem**:
```python
# PyTorch
x = torch.randn(10, dtype=torch.float64)

# MLX: No float64 support
```

**Solution**:
```python
# Use float32 instead
x = mx.random.normal((10,), dtype=mx.float32)

# Be aware of potential precision differences in some algorithms
```

## Performance Considerations

### 1. Lazy Evaluation

MLX operations are lazy and fused for efficiency:

```python
# This creates a single fused kernel
y = mx.exp(mx.log(x) + 1)

# Force evaluation when needed
mx.eval(y)
```

### 2. Unified Memory

Avoid unnecessary NumPy conversions:

```python
# Bad: Forces evaluation and copy
x_np = np.array(x)
y_np = process_in_numpy(x_np)
y = mx.array(y_np)

# Good: Stay in MLX
y = process_in_mlx(x)
```

### 3. Metal-Specific Optimizations

```python
# Use Metal-optimized operations when available
# Conv2d, matmul, etc. are highly optimized for Metal

# Prefer built-in ops over custom implementations
x = mx.conv2d(...)  # Fast Metal implementation
# vs
x = custom_conv2d(...)  # Slower unless heavily optimized
```

## Summary of Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Execution** | Eager | Lazy + fusion |
| **Memory** | Discrete (CPU/GPU) | Unified (Metal) |
| **Mutability** | Mutable tensors | Immutable arrays |
| **Layout** | NCHW (channels-first) | NHWC (channels-last) |
| **Parameters** | dim, keepdim | axis, keepdims |
| **Gradients** | Automatic accumulation | Manual accumulation |
| **Precision** | float16, float32, float64 | float16, float32, bfloat16 |
| **Autograd** | Tape-based | Function transformation |
| **Modules** | Class-based (nn.Module) | Simpler base class |
| **Optimizers** | Take parameters at init | Update called with grads |

This comprehensive mapping should serve as your primary reference when porting PyTorch code to MLX or building compatibility layers.
