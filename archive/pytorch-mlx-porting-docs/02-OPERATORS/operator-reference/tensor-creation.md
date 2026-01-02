# Tensor Creation Operations

## Overview

Tensor creation operations construct new tensors with specified shapes, data types, and initial values. These are fundamental building blocks for all PyTorch programs.

**Reference:** `aten/src/ATen/native/native_functions.yaml` (creation ops section)

## Categories

```
Tensor Creation Operations
├── Constant Fill
│   ├── zeros, ones
│   ├── full
│   └── empty
├── Sequences
│   ├── arange
│   ├── linspace, logspace
│   └── range (deprecated)
├── Identity/Diagonal
│   ├── eye
│   └── diag
├── Random
│   ├── rand, randn
│   ├── randint, randperm
│   └── bernoulli
└── From Existing Data
    ├── tensor, as_tensor
    ├── from_numpy
    └── clone
```

---

## Constant Fill Operations

### zeros()

Creates a tensor filled with zeros.

```python
torch.zeros(
    *size: int,                    # Shape dimensions
    out: Tensor = None,            # Output tensor
    dtype: torch.dtype = None,     # Data type
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False
) -> Tensor
```

**Examples:**

```python
# Basic usage
x = torch.zeros(3, 4)           # Shape (3, 4)
x = torch.zeros(2, 3, 4)        # Shape (2, 3, 4)

# With options
x = torch.zeros(3, 4, dtype=torch.float16)
x = torch.zeros(3, 4, device='cuda')
x = torch.zeros(3, 4, requires_grad=True)

# From tuple/list
shape = (2, 3)
x = torch.zeros(shape)          # Error! Use *shape
x = torch.zeros(*shape)         # Correct
```

### ones()

Creates a tensor filled with ones.

```python
torch.ones(
    *size: int,
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False
) -> Tensor
```

**Examples:**

```python
x = torch.ones(3, 4)
x = torch.ones(2, 3, dtype=torch.int32)
x = torch.ones(5, device='mps')
```

### full()

Creates a tensor filled with a specified value.

```python
torch.full(
    size: tuple[int],              # Shape as tuple
    fill_value: Number,            # Value to fill
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False
) -> Tensor
```

**Examples:**

```python
# Fill with specific value
x = torch.full((3, 4), 3.14)         # All elements = 3.14
x = torch.full((2, 3), -1)           # All elements = -1
x = torch.full((5,), float('inf'))   # All elements = inf
```

### empty()

Creates an uninitialized tensor (faster, contains garbage values).

```python
torch.empty(
    *size: int,
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format
) -> Tensor
```

**Examples:**

```python
# Uninitialized (fast allocation)
x = torch.empty(3, 4)  # Contents undefined!

# Use when you'll overwrite all values
output = torch.empty(batch_size, num_classes)
model.forward(input, out=output)
```

---

## Sequence Operations

### arange()

Creates a 1D tensor with values from start to end (exclusive).

```python
torch.arange(
    start: Number = 0,             # Start value (inclusive)
    end: Number,                   # End value (exclusive)
    step: Number = 1,              # Step size
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False
) -> Tensor
```

**Examples:**

```python
# Basic sequences
torch.arange(5)              # tensor([0, 1, 2, 3, 4])
torch.arange(1, 5)           # tensor([1, 2, 3, 4])
torch.arange(0, 5, 2)        # tensor([0, 2, 4])
torch.arange(0, 1, 0.1)      # tensor([0.0, 0.1, ..., 0.9])

# Float range
torch.arange(0.0, 1.0, 0.25)  # tensor([0.0, 0.25, 0.5, 0.75])
```

### linspace()

Creates a 1D tensor with evenly spaced values.

```python
torch.linspace(
    start: Number,                 # Start value
    end: Number,                   # End value (inclusive!)
    steps: int,                    # Number of points
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False
) -> Tensor
```

**Examples:**

```python
# 5 points from 0 to 1 (inclusive)
torch.linspace(0, 1, 5)       # tensor([0.0, 0.25, 0.5, 0.75, 1.0])

# Useful for coordinate grids
x = torch.linspace(-1, 1, 100)
y = torch.sin(x * 3.14159)
```

### logspace()

Creates a 1D tensor with logarithmically spaced values.

```python
torch.logspace(
    start: Number,                 # base^start is first value
    end: Number,                   # base^end is last value
    steps: int,
    base: float = 10.0,            # Logarithm base
    out: Tensor = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    requires_grad: bool = False
) -> Tensor
```

**Examples:**

```python
# Powers of 10: 10^0, 10^0.5, 10^1, 10^1.5, 10^2
torch.logspace(0, 2, 5)       # tensor([1, 3.16, 10, 31.6, 100])

# Powers of 2
torch.logspace(0, 4, 5, base=2)  # tensor([1, 2, 4, 8, 16])
```

---

## Identity and Diagonal Operations

### eye()

Creates a 2D identity matrix.

```python
torch.eye(
    n: int,                        # Number of rows
    m: int = None,                 # Number of columns (default: n)
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False
) -> Tensor
```

**Examples:**

```python
# Square identity
torch.eye(3)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# Rectangular
torch.eye(3, 4)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.]])
```

### diag()

Creates a diagonal matrix or extracts diagonal.

```python
torch.diag(
    input: Tensor,                 # 1D or 2D tensor
    diagonal: int = 0              # Diagonal offset
) -> Tensor
```

**Examples:**

```python
# Create diagonal matrix from 1D
v = torch.tensor([1, 2, 3])
torch.diag(v)
# tensor([[1, 0, 0],
#         [0, 2, 0],
#         [0, 0, 3]])

# Extract diagonal from 2D
m = torch.arange(9).reshape(3, 3)
torch.diag(m)        # tensor([0, 4, 8])
torch.diag(m, 1)     # tensor([1, 5])  - above main diagonal
torch.diag(m, -1)    # tensor([3, 7])  - below main diagonal
```

---

## Random Operations

### rand()

Creates a tensor with uniform random values in [0, 1).

```python
torch.rand(
    *size: int,
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    generator: torch.Generator = None
) -> Tensor
```

**Examples:**

```python
torch.rand(3, 4)       # 3x4 uniform [0, 1)
torch.rand(2, 3, 4)    # 2x3x4 tensor

# Reproducible
g = torch.Generator().manual_seed(42)
torch.rand(3, generator=g)
```

### randn()

Creates a tensor with standard normal (Gaussian) random values (mean=0, std=1).

```python
torch.randn(
    *size: int,
    out: Tensor = None,
    dtype: torch.dtype = None,
    layout: torch.layout = torch.strided,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    generator: torch.Generator = None
) -> Tensor
```

**Examples:**

```python
torch.randn(3, 4)      # 3x4 standard normal

# Custom mean and std
mean, std = 5.0, 2.0
x = mean + std * torch.randn(3, 4)
```

### randint()

Creates a tensor with random integers.

```python
torch.randint(
    low: int = 0,                  # Minimum (inclusive)
    high: int,                     # Maximum (exclusive)
    size: tuple[int],              # Shape
    out: Tensor = None,
    dtype: torch.dtype = torch.int64,
    device: torch.device = None,
    requires_grad: bool = False,
    generator: torch.Generator = None
) -> Tensor
```

**Examples:**

```python
torch.randint(10, (3, 4))       # [0, 10) integers, shape (3, 4)
torch.randint(5, 10, (3,))      # [5, 10) integers, shape (3,)
```

### randperm()

Creates a random permutation of integers [0, n).

```python
torch.randperm(
    n: int,                        # Range [0, n)
    out: Tensor = None,
    dtype: torch.dtype = torch.int64,
    device: torch.device = None,
    requires_grad: bool = False,
    generator: torch.Generator = None
) -> Tensor
```

**Examples:**

```python
torch.randperm(5)     # e.g., tensor([2, 4, 0, 3, 1])

# Shuffle a tensor
x = torch.arange(10)
shuffled = x[torch.randperm(10)]
```

---

## From Existing Data

### tensor()

Creates a tensor from Python data (always copies).

```python
torch.tensor(
    data: array_like,              # Input data
    dtype: torch.dtype = None,
    device: torch.device = None,
    requires_grad: bool = False,
    pin_memory: bool = False
) -> Tensor
```

**Examples:**

```python
# From list
torch.tensor([1, 2, 3])           # tensor([1, 2, 3])
torch.tensor([[1, 2], [3, 4]])    # 2D tensor

# With dtype
torch.tensor([1, 2, 3], dtype=torch.float32)
```

### as_tensor()

Creates a tensor, sharing data when possible.

```python
torch.as_tensor(
    data: array_like,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> Tensor
```

**Examples:**

```python
import numpy as np

arr = np.array([1, 2, 3])
t = torch.as_tensor(arr)  # Shares memory with numpy array!

# Modifying t affects arr (if same device/dtype)
t[0] = 100
print(arr)  # [100, 2, 3]
```

### from_numpy()

Creates a tensor sharing memory with a numpy array.

```python
torch.from_numpy(ndarray: numpy.ndarray) -> Tensor
```

**Examples:**

```python
import numpy as np

arr = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(arr)  # Shares memory!

# Changes in one affect the other
arr[0] = 999
print(t)  # tensor([999., 2., 3.])
```

### clone()

Creates a copy of a tensor.

```python
Tensor.clone(memory_format: torch.memory_format = torch.preserve_format) -> Tensor
```

**Examples:**

```python
x = torch.tensor([1, 2, 3])
y = x.clone()  # Independent copy

# With gradient tracking
x = torch.randn(3, requires_grad=True)
y = x.clone()  # y also requires_grad=True
```

---

## Like Variants

Each creation function has a `_like` variant that copies shape/dtype/device from an existing tensor.

### zeros_like(), ones_like(), etc.

```python
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor
torch.ones_like(input, *, ...) -> Tensor
torch.empty_like(input, *, ...) -> Tensor
torch.full_like(input, fill_value, *, ...) -> Tensor
torch.rand_like(input, *, ...) -> Tensor
torch.randn_like(input, *, ...) -> Tensor
torch.randint_like(input, low=0, high, *, ...) -> Tensor
```

**Examples:**

```python
x = torch.randn(3, 4, device='cuda', dtype=torch.float16)

# Same shape, dtype, device
zeros = torch.zeros_like(x)
ones = torch.ones_like(x)
noise = torch.randn_like(x)

# Override specific attributes
y = torch.zeros_like(x, dtype=torch.float32)
```

---

## New Variants

Tensor methods to create new tensors with same dtype/device.

```python
Tensor.new_zeros(size, *, dtype=None, device=None, requires_grad=False) -> Tensor
Tensor.new_ones(size, *, ...) -> Tensor
Tensor.new_empty(size, *, ...) -> Tensor
Tensor.new_full(size, fill_value, *, ...) -> Tensor
Tensor.new_tensor(data, *, ...) -> Tensor
```

**Examples:**

```python
x = torch.randn(3, 4, device='cuda', dtype=torch.float16)

# Create with same dtype/device as x
zeros = x.new_zeros(5, 6)        # Shape (5, 6), same dtype/device
ones = x.new_ones(2, 3)          # Shape (2, 3)
filled = x.new_full((4, 4), 7)   # Shape (4, 4), all 7s
```

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `torch.zeros(shape)` | `mx.zeros(shape)` |
| `torch.ones(shape)` | `mx.ones(shape)` |
| `torch.full(shape, val)` | `mx.full(shape, val)` |
| `torch.empty(shape)` | `mx.zeros(shape)` (no empty) |
| `torch.arange(start, end, step)` | `mx.arange(start, end, step)` |
| `torch.linspace(start, end, n)` | `mx.linspace(start, end, n)` |
| `torch.eye(n)` | `mx.eye(n)` |
| `torch.rand(shape)` | `mx.random.uniform(shape=shape)` |
| `torch.randn(shape)` | `mx.random.normal(shape=shape)` |
| `torch.randint(low, high, shape)` | `mx.random.randint(low, high, shape)` |

### MLX Examples

```python
import mlx.core as mx

# Constant fill
x = mx.zeros((3, 4))
x = mx.ones((3, 4))
x = mx.full((3, 4), 3.14)

# Sequences
x = mx.arange(0, 10, 2)
x = mx.linspace(0, 1, 5)

# Identity
x = mx.eye(3)

# Random (requires key for reproducibility)
key = mx.random.key(42)
x = mx.random.uniform(key=key, shape=(3, 4))
x = mx.random.normal(key=key, shape=(3, 4))

# From array
x = mx.array([1, 2, 3])
x = mx.array([[1, 2], [3, 4]])
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| empty() | Uninitialized | No equivalent (use zeros) |
| Random state | Generator object | Key-based PRNG |
| Device | device='cuda' | Unified memory |
| Gradient tracking | requires_grad=True | Separate gradient system |
| Memory sharing | from_numpy shares | mx.array always copies |

---

## Best Practices

1. **Use `_like` variants** - Inherit dtype/device automatically

2. **Avoid empty()** - Unless you'll definitely overwrite all values

3. **Specify dtype explicitly** - Avoid unexpected type inference

4. **Use generators for reproducibility** - `torch.Generator().manual_seed(seed)`

5. **Prefer as_tensor() over tensor()** - When data sharing is acceptable

6. **Match devices** - Create tensors on same device as computation

7. **Consider memory layout** - Use `memory_format` for optimal layout

---

## Summary

| Function | Creates |
|----------|---------|
| `zeros()` | All zeros |
| `ones()` | All ones |
| `full()` | All same value |
| `empty()` | Uninitialized |
| `arange()` | Arithmetic sequence |
| `linspace()` | Evenly spaced (inclusive) |
| `logspace()` | Logarithmically spaced |
| `eye()` | Identity matrix |
| `diag()` | Diagonal matrix / extract diagonal |
| `rand()` | Uniform [0, 1) |
| `randn()` | Standard normal |
| `randint()` | Random integers |
| `randperm()` | Random permutation |
| `tensor()` | From Python data (copy) |
| `as_tensor()` | From array (share if possible) |
| `from_numpy()` | From numpy (share memory) |
| `clone()` | Copy existing tensor |
