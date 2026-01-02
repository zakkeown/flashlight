# Tensor Operations

Tensors are the fundamental data structure in Flashlight, just as they are in PyTorch. This guide covers everything you need to know about creating and manipulating tensors.

## Creating Tensors

### From Python Data

The most direct way to create a tensor is from Python lists or numbers:

```python
import flashlight

# From a list
x = flashlight.tensor([1.0, 2.0, 3.0])

# From nested lists (2D tensor)
matrix = flashlight.tensor([[1, 2, 3], [4, 5, 6]])

# With gradient tracking
x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

### Creation Functions

Flashlight provides familiar tensor creation functions:

```python
# Zeros and ones
zeros = flashlight.zeros(3, 4)           # 3x4 tensor of zeros
ones = flashlight.ones(2, 3, 4)          # 2x3x4 tensor of ones

# Random tensors
rand = flashlight.rand(3, 4)             # Uniform [0, 1)
randn = flashlight.randn(3, 4)           # Standard normal

# Sequences
arange = flashlight.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = flashlight.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

# Identity and diagonal
eye = flashlight.eye(3)                  # 3x3 identity matrix
diag = flashlight.diag(flashlight.tensor([1, 2, 3]))

# Empty (uninitialized) and full
empty = flashlight.empty(3, 4)
full = flashlight.full((3, 4), 7.0)      # Filled with 7.0
```

### From NumPy

```python
import numpy as np

np_array = np.array([1.0, 2.0, 3.0])
tensor = flashlight.tensor(np_array)

# Convert back to NumPy
back_to_numpy = tensor.numpy()
```

## Tensor Properties

Every tensor has these key properties:

```python
x = flashlight.randn(2, 3, 4)

x.shape       # (2, 3, 4)
x.ndim        # 3
x.numel()     # 24 (total elements)
x.dtype       # flashlight.float32
x.device      # 'mps' (Apple Silicon GPU)
x.requires_grad  # False (unless set)
```

### Data Types

Flashlight supports these dtypes:

| Dtype | Description |
|-------|-------------|
| `flashlight.float32` | 32-bit floating point (default) |
| `flashlight.float16` | 16-bit floating point |
| `flashlight.bfloat16` | Brain floating point |
| `flashlight.int32` | 32-bit integer |
| `flashlight.int64` | 64-bit integer |
| `flashlight.bool` | Boolean |

```python
# Specify dtype at creation
x = flashlight.zeros(3, 4, dtype=flashlight.float16)

# Cast to different dtype
y = x.to(flashlight.float32)
# or
y = x.float()
```

!!! warning "No float64 Support"
    MLX does not support float64. If you need double precision, consider staying on PyTorch for that computation.

## Basic Operations

### Arithmetic

```python
a = flashlight.tensor([1.0, 2.0, 3.0])
b = flashlight.tensor([4.0, 5.0, 6.0])

# Element-wise operations
c = a + b      # Addition
c = a - b      # Subtraction
c = a * b      # Multiplication
c = a / b      # Division
c = a ** 2     # Power

# In-place operations (note: these create copies internally)
a.add_(b)      # a = a + b
a.mul_(2)      # a = a * 2
```

### Matrix Operations

```python
A = flashlight.randn(3, 4)
B = flashlight.randn(4, 5)

# Matrix multiplication
C = A @ B                    # or flashlight.matmul(A, B)

# Batch matrix multiplication
batch_A = flashlight.randn(10, 3, 4)
batch_B = flashlight.randn(10, 4, 5)
batch_C = batch_A @ batch_B  # (10, 3, 5)
```

### Reductions

```python
x = flashlight.randn(3, 4)

x.sum()           # Sum all elements
x.mean()          # Mean of all elements
x.max()           # Maximum value
x.min()           # Minimum value
x.std()           # Standard deviation

# Along specific dimensions
x.sum(dim=0)      # Sum along rows -> shape (4,)
x.mean(dim=1)     # Mean along columns -> shape (3,)
x.max(dim=1)      # Returns (values, indices)
```

### Broadcasting

Flashlight follows NumPy/PyTorch broadcasting rules:

```python
a = flashlight.randn(3, 4)
b = flashlight.randn(4)      # Will broadcast to (3, 4)

c = a + b  # Works! b is broadcast across dim 0

# Explicit broadcasting
b_expanded = b.unsqueeze(0).expand(3, 4)
```

## Indexing and Slicing

### Basic Indexing

```python
x = flashlight.randn(3, 4, 5)

# Single element
element = x[0, 1, 2]

# Slicing
row = x[0]           # First "row" -> shape (4, 5)
slice = x[:, 1:3]    # Columns 1-2 -> shape (3, 2, 5)
stepped = x[::2]     # Every other -> shape (2, 4, 5)
```

### Advanced Indexing

```python
x = flashlight.randn(5, 4)

# Index with tensor
indices = flashlight.tensor([0, 2, 4])
selected = x[indices]  # Rows 0, 2, 4 -> shape (3, 4)

# Boolean indexing
mask = x > 0
positives = x[mask]    # All positive values (1D)
```

### Assignment

```python
x = flashlight.zeros(3, 4)

x[0] = 1.0                    # Set first row to 1
x[:, 0] = flashlight.ones(3)  # Set first column
x[x < 0] = 0                  # Clamp negatives to 0
```

## Shape Manipulation

### Reshaping

```python
x = flashlight.randn(12)

# Reshape
y = x.reshape(3, 4)
y = x.view(3, 4)      # Same as reshape for contiguous tensors

# Infer one dimension
y = x.reshape(3, -1)  # -> (3, 4)
y = x.reshape(-1, 2)  # -> (6, 2)
```

### Adding/Removing Dimensions

```python
x = flashlight.randn(3, 4)

# Add dimension
y = x.unsqueeze(0)    # -> (1, 3, 4)
y = x.unsqueeze(-1)   # -> (3, 4, 1)

# Remove dimension
z = y.squeeze()       # Remove all size-1 dims
z = y.squeeze(0)      # Remove specific dim if size 1
```

### Transposing

```python
x = flashlight.randn(3, 4)

# 2D transpose
y = x.T               # -> (4, 3)
y = x.transpose(0, 1) # Same thing

# Multi-dimensional permute
z = flashlight.randn(2, 3, 4, 5)
w = z.permute(0, 2, 1, 3)  # -> (2, 4, 3, 5)
```

### Concatenation and Stacking

```python
a = flashlight.randn(3, 4)
b = flashlight.randn(3, 4)

# Concatenate along existing dimension
c = flashlight.cat([a, b], dim=0)  # -> (6, 4)
c = flashlight.cat([a, b], dim=1)  # -> (3, 8)

# Stack creates new dimension
s = flashlight.stack([a, b], dim=0)  # -> (2, 3, 4)
```

### Splitting

```python
x = flashlight.randn(6, 4)

# Split into equal parts
parts = flashlight.chunk(x, 2, dim=0)  # Two (3, 4) tensors

# Split at specific sizes
parts = flashlight.split(x, [2, 4], dim=0)  # (2, 4) and (4, 4)
```

## Device and Memory

### Unified Memory

Unlike PyTorch with CUDA, MLX uses Apple's unified memory architecture. There's no explicit memory transfer between CPU and GPU:

```python
x = flashlight.randn(1000, 1000)
# x is automatically available on both CPU and GPU
# No need for .to('cuda') or .to('cpu')
```

### Device Property

For PyTorch compatibility, tensors have a `device` property:

```python
x = flashlight.randn(3, 4)
print(x.device)  # 'mps' on Apple Silicon

# These are compatibility shims (no actual transfer)
y = x.to('cpu')
z = x.to('mps')
```

## PyTorch Migration Notes

### What Works the Same

- Tensor creation functions (`zeros`, `ones`, `randn`, etc.)
- Indexing and slicing syntax
- Most arithmetic and reduction operations
- Shape manipulation (`reshape`, `view`, `transpose`, etc.)
- Autograd with `requires_grad` and `backward()`

### Key Differences

1. **No float64**: Use float32 instead
2. **No explicit device transfers**: Unified memory handles this
3. **Immutability**: In-place operations create internal copies
4. **Layout for convolutions**: Automatic NCHW ↔ NHWC conversion

### Common Migration Pattern

```python
# PyTorch
import torch
x = torch.randn(3, 4, device='cuda')
x = x.to('cpu')

# Flashlight
import flashlight
x = flashlight.randn(3, 4)  # No device needed
# No transfer needed - unified memory
```
