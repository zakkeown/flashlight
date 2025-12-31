# Nested Tensors (torch.nested)

## Purpose

Nested tensors (also called jagged/ragged tensors) represent batches of tensors with variable sizes in one or more dimensions. They solve the problem of efficiently handling variable-length sequences without wasteful padding.

**Key Use Cases:**
- Variable-length sequences in NLP (different sentence lengths)
- Variable-size images in vision tasks
- Efficient attention computation (Flash Attention with variable lengths)
- KV-cache representation in transformer inference

**Reference Files:**
- `torch/nested/__init__.py` - Public API
- `torch/nested/_internal/nested_tensor.py` - Core NestedTensor class
- `torch/nested/_internal/ops.py` - Operations on nested tensors (~100KB)
- `torch/nested/_internal/sdpa.py` - SDPA integration

---

## Overview

### The Problem with Padding

Traditional approach for variable-length sequences:
```python
# Sequences of different lengths
seq1 = torch.randn(5, 64)   # length 5
seq2 = torch.randn(10, 64)  # length 10
seq3 = torch.randn(3, 64)   # length 3

# Padded batch (wasteful)
batch = torch.zeros(3, 10, 64)  # max_len = 10
batch[0, :5] = seq1
batch[1, :10] = seq2
batch[2, :3] = seq3
# 50% of memory is padding zeros!
```

### Nested Tensor Solution

```python
# Efficient representation without padding
nt = torch.nested.nested_tensor([seq1, seq2, seq3])
# Only stores 18 * 64 = 1152 elements (not 30 * 64 = 1920)
```

### Layouts

PyTorch supports two nested tensor layouts:

| Layout | Storage | Use Case |
|--------|---------|----------|
| `torch.strided` | Separate storage per element | General use |
| `torch.jagged` | Packed contiguous storage | SDPA, efficient ops |

**Jagged Layout** is the preferred layout for performance:
```
Values buffer: [seq1_data, seq2_data, seq3_data]  (packed)
Offsets:       [0, 5, 15, 18]  (cumulative lengths)
```

---

## Core API

### nested_tensor (Create from List)

**Purpose**: Create a nested tensor from a list of tensors (leaf tensor, no autograd history).

**Signature**:
```python
torch.nested.nested_tensor(
    tensor_list: List[Tensor],
    *,
    dtype: DType = None,
    layout: Layout = None,
    device: Device = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor_list` | List[Tensor] | List of tensors with same ndim |
| `dtype` | torch.dtype | Output dtype (default: inferred) |
| `layout` | torch.layout | `torch.strided` or `torch.jagged` |
| `device` | torch.device | Output device |
| `requires_grad` | bool | Enable gradient tracking |
| `pin_memory` | bool | Pin memory for GPU transfer |

**Usage Examples**:
```python
import torch

# Basic creation (strided layout)
a = torch.arange(3, dtype=torch.float)
b = torch.arange(5, dtype=torch.float)
nt = torch.nested.nested_tensor([a, b])
nt.is_leaf  # True

# Jagged layout (recommended for performance)
nt_jagged = torch.nested.nested_tensor([a, b], layout=torch.jagged)

# With gradient tracking
nt = torch.nested.nested_tensor([a, b], requires_grad=True)
```

**Constraints**:
- All tensors must have the same `ndim`
- All tensors must have compatible dtypes
- Jagged layout supports only one ragged dimension

---

### as_nested_tensor (Create with Autograd History)

**Purpose**: Create a nested tensor preserving autograd history from inputs.

**Signature**:
```python
torch.nested.as_nested_tensor(
    ts: Tensor | List[Tensor] | Tuple[Tensor],
    dtype: DType = None,
    device: Device = None,
    layout: Layout = None,
) -> Tensor
```

**Key Difference from nested_tensor**:
- `nested_tensor`: Creates leaf tensor, breaks autograd
- `as_nested_tensor`: Preserves autograd history

**Usage Examples**:
```python
# Preserves gradients
a = torch.arange(3, dtype=torch.float, requires_grad=True)
b = torch.arange(5, dtype=torch.float, requires_grad=True)

nt = torch.nested.as_nested_tensor([a, b])
nt.is_leaf  # False (preserves computation graph)

# Backward pass works
fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)])
nt.backward(fake_grad)
a.grad  # tensor([1., 1., 1.])
b.grad  # tensor([0., 0., 0., 0., 0.])

# From regular tensor (batch of same-size elements)
c = torch.randn(3, 5, requires_grad=True)
nt = torch.nested.as_nested_tensor(c)  # 3 elements of shape (5,)
```

---

### nested_tensor_from_jagged (Low-Level Constructor)

**Purpose**: Create a jagged nested tensor directly from packed values and offsets/lengths.

**Signature**:
```python
torch.nested.nested_tensor_from_jagged(
    values: Tensor,
    offsets: Tensor | None = None,
    lengths: Tensor | None = None,
    jagged_dim: int | None = None,
    min_seqlen: int | None = None,
    max_seqlen: int | None = None,
) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | Tensor | Packed data buffer of shape (sum_B(*), D_1, ..., D_N) |
| `offsets` | Tensor | Offsets into packed dim, shape (B+1,) |
| `lengths` | Tensor | Lengths of each element, shape (B,) |
| `jagged_dim` | int | Which dimension is ragged (default: 1) |
| `min_seqlen` | int | Cached minimum sequence length |
| `max_seqlen` | int | Cached maximum sequence length |

**Understanding Offsets and Lengths**:
```python
# offsets example: [0, 2, 3, 6]
# This means:
#   - Element 0: indices [0, 2) → length 2
#   - Element 1: indices [2, 3) → length 1
#   - Element 2: indices [3, 6) → length 3

# Equivalent lengths: [2, 1, 3]
```

**Usage Examples**:
```python
# Create from packed values
values = torch.randn(12, 5)  # 12 total rows, 5 features
offsets = torch.tensor([0, 3, 5, 6, 10, 12])  # 5 batch elements

nt = torch.nested.nested_tensor_from_jagged(values, offsets)
nt.shape  # torch.Size([5, j2, 5]) - j2 denotes ragged dimension

# Sequence lengths
offsets.diff()  # tensor([3, 2, 1, 4, 2])

# With caching for SDPA performance
nt = torch.nested.nested_tensor_from_jagged(
    values, offsets,
    min_seqlen=1,
    max_seqlen=4
)

# Nested tensor with "holes" (using both offsets and lengths)
values = torch.randn(6, 5)
offsets = torch.tensor([0, 2, 3, 6])
lengths = torch.tensor([1, 1, 2])  # Actual lengths within each offset range

nt = torch.nested.nested_tensor_from_jagged(values, offsets, lengths)
# Element 0: values[0:1]   (not values[0:2])
# Element 1: values[2:3]   (not values[2:3])
# Element 2: values[3:5]   (not values[3:6])
```

**Returns**: A view of the input values tensor.

---

### to_padded_tensor (Convert to Regular Tensor)

**Purpose**: Convert nested tensor to regular tensor with padding.

**Signature**:
```python
torch.nested.to_padded_tensor(
    input: Tensor,
    padding: float,
    output_size: Tuple[int] = None,
    *,
    out: Tensor = None,
) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Nested tensor to convert |
| `padding` | float | Value to use for padding |
| `output_size` | Tuple[int] | Output size (inferred if None) |
| `out` | Tensor | Optional output tensor |

**Usage Examples**:
```python
nt = torch.nested.nested_tensor([
    torch.randn(2, 5),
    torch.randn(3, 4)
])

# Infer output size (max of each dimension)
pt = torch.nested.to_padded_tensor(nt, padding=0.0)
pt.shape  # torch.Size([2, 3, 5])

# Specify larger output size
pt = torch.nested.to_padded_tensor(nt, padding=1.0, output_size=(2, 4, 6))
pt.shape  # torch.Size([2, 4, 6])

# Cannot truncate
pt = torch.nested.to_padded_tensor(nt, padding=2.0, output_size=(2, 2, 2))
# RuntimeError: Value in output_size is less than NestedTensor padded size
```

**Note**: Always copies data (nested and regular tensors have different memory layouts).

---

### narrow (Create NT from Slices)

**Purpose**: Create a nested tensor by narrowing a regular tensor with variable start/length.

**Signature**:
```python
torch.nested.narrow(
    tensor: Tensor,
    dim: int,
    start: int | Tensor,
    length: int | Tensor,
    layout: Layout = torch.strided,
) -> Tensor
```

**Use Case**: KV-cache representation in transformer inference.

**Usage Examples**:
```python
# Create KV-cache style nested tensor
base = torch.randn(5, 10, 20)  # batch=5, max_len=10, features=20

# Different start positions and lengths for each batch element
starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
lengths = torch.tensor([3, 2, 2, 1, 5], dtype=torch.int64)

# Jagged layout (view, no copy, good for SDPA)
nt = torch.nested.narrow(base, dim=1, start=starts, length=lengths, layout=torch.jagged)
nt.is_contiguous()  # False (it's a view)

# Strided layout (copies data)
nt = torch.nested.narrow(base, dim=1, start=starts, length=lengths, layout=torch.strided)
```

**Jagged vs Strided**:
- `torch.jagged`: Creates non-contiguous view (no copy, efficient for SDPA)
- `torch.strided`: Copies narrowed data into contiguous storage

---

### masked_select (Create NT from Mask)

**Purpose**: Create a nested tensor by selecting elements based on a mask.

**Signature**:
```python
torch.nested.masked_select(tensor: Tensor, mask: Tensor) -> Tensor
```

**Usage Examples**:
```python
tensor = torch.randn(3, 3)
mask = torch.tensor([
    [False, False, True],
    [True, False, True],
    [False, False, True]
])

nt = torch.nested.masked_select(tensor, mask)
nt.shape  # torch.Size([3, j4])

# Lengths per batch element
nt.offsets().diff()  # tensor([1, 2, 1])
```

---

## Internal Architecture

### NestedTensor Class

The `NestedTensor` class is a tensor subclass with specialized storage:

```python
class NestedTensor(torch.Tensor):
    _values: torch.Tensor      # Packed data buffer
    _offsets: torch.Tensor     # Offsets into packed dimension
    _lengths: torch.Tensor     # Optional: explicit lengths (for "holes")
    _ragged_idx: int           # Which dimension is ragged (usually 1)
    _size: tuple[int, ...]     # Symbolic shape with nested ints
    _strides: tuple[int, ...]  # Symbolic strides
    _metadata_cache: Dict      # Cached min/max seqlen
```

### Nested Ints (Symbolic Ragged Dimensions)

Ragged dimensions are represented symbolically:
```python
nt = torch.nested.nested_tensor([
    torch.randn(3, 5),
    torch.randn(7, 5),
])
nt.shape  # torch.Size([2, j1, 5])  - j1 represents the ragged dimension
```

The `j1` (or `*`) notation indicates a dimension with variable size per batch element.

### Shape Representation

For a nested tensor with outer shape `[B, *, D]`:
- `B`: Batch size (known)
- `*`: Ragged dimension (represented by NestedIntNode)
- `D`: Feature dimension (known)

Internal storage is `[sum(*), D]` with offsets tracking batch boundaries.

### Stride Representation

Strides are also symbolic:
```
Shape:   [B, *, D]
Strides: [*D, D, 1]  where *D = ragged_size * D
```

---

## SDPA Integration

Nested tensors integrate with Scaled Dot-Product Attention for efficient variable-length attention.

### Benefits

1. **No Padding Overhead**: Compute attention only on actual tokens
2. **Memory Efficiency**: No wasted memory on padding
3. **Flash Attention**: Compatible with optimized attention kernels

### Usage with SDPA

```python
import torch
import torch.nn.functional as F

# Variable-length queries, keys, values
batch_size = 4
d_model = 64

# Create nested tensors for variable-length sequences
q_list = [torch.randn(seq_len, d_model) for seq_len in [10, 15, 8, 20]]
k_list = [torch.randn(seq_len, d_model) for seq_len in [10, 15, 8, 20]]
v_list = [torch.randn(seq_len, d_model) for seq_len in [10, 15, 8, 20]]

q = torch.nested.nested_tensor(q_list, layout=torch.jagged)
k = torch.nested.nested_tensor(k_list, layout=torch.jagged)
v = torch.nested.nested_tensor(v_list, layout=torch.jagged)

# Reshape for multi-head attention
q = q.unflatten(-1, (8, 8)).transpose(1, 2)  # [B, heads, *, head_dim]
k = k.unflatten(-1, (8, 8)).transpose(1, 2)
v = v.unflatten(-1, (8, 8)).transpose(1, 2)

# SDPA with nested tensors (uses optimized kernels)
output = F.scaled_dot_product_attention(q, k, v)
```

### Caching min/max seqlen

For optimal SDPA performance, cache sequence length metadata:

```python
values = torch.randn(100, 64)
offsets = torch.tensor([0, 10, 25, 33, 60, 100])

# Without caching (GPU->CPU sync to compute)
nt = torch.nested.nested_tensor_from_jagged(values, offsets)

# With caching (avoids sync)
nt = torch.nested.nested_tensor_from_jagged(
    values, offsets,
    min_seqlen=10,
    max_seqlen=40
)
```

---

## Autograd Support

Nested tensors fully support automatic differentiation:

```python
a = torch.randn(3, 5, requires_grad=True)
b = torch.randn(7, 5, requires_grad=True)

nt = torch.nested.as_nested_tensor([a, b])

# Forward operations
output = nt.sum()

# Backward pass
output.backward()

# Gradients flow to original tensors
print(a.grad.shape)  # torch.Size([3, 5])
print(b.grad.shape)  # torch.Size([7, 5])
```

---

## Supported Operations

### Core Tensor Operations

```python
nt.values()    # Get underlying packed values buffer
nt.offsets()   # Get offsets tensor
nt.lengths()   # Get lengths tensor (if present)
nt.unbind()    # Unpack into list of tensors
```

### Mathematical Operations

Most element-wise operations work on nested tensors:
- Arithmetic: `+`, `-`, `*`, `/`
- Activations: `relu`, `gelu`, `sigmoid`, etc.
- Reductions: `sum`, `mean` (with dimension restrictions)

### Shape Operations

```python
nt.transpose(dim0, dim1)  # Transpose (with restrictions)
nt.unflatten(dim, sizes)  # Unflatten a dimension
nt.flatten(start_dim, end_dim)  # Flatten dimensions
```

---

## MLX Porting Considerations

### Key Challenges

1. **No Native Ragged Tensor Support**: MLX doesn't have built-in nested tensor support
2. **SDPA Integration**: Need custom attention kernels for variable lengths
3. **Autograd**: Gradient tracking through ragged operations

### Potential Approaches

**Approach 1: Explicit Padding**
```python
# Convert to padded for MLX, use mask for operations
padded = mx.zeros((batch_size, max_len, features))
mask = mx.zeros((batch_size, max_len), dtype=mx.bool_)

for i, (seq, length) in enumerate(zip(sequences, lengths)):
    padded[i, :length] = seq
    mask[i, :length] = True
```

**Approach 2: Custom NestedArray Class**
```python
class NestedArray:
    """MLX equivalent of PyTorch NestedTensor"""

    def __init__(self, values: mx.array, offsets: mx.array):
        self.values = values
        self.offsets = offsets

    @property
    def batch_size(self):
        return len(self.offsets) - 1

    def to_padded(self, padding_value=0.0):
        # Implementation...
        pass

    def unbind(self):
        result = []
        for i in range(self.batch_size):
            start, end = self.offsets[i], self.offsets[i + 1]
            result.append(self.values[start:end])
        return result
```

**Approach 3: Operation-Level Handling**
```python
def nested_matmul(nt_a, nt_b):
    """Process each element separately"""
    results = []
    for a, b in zip(nt_a.unbind(), nt_b.unbind()):
        results.append(a @ b)
    return NestedArray.from_list(results)
```

### Performance Considerations

1. **Batch Operations**: MLX prefers uniform batches; consider grouping similar lengths
2. **Memory Layout**: MLX uses row-major; ensure efficient access patterns
3. **Kernel Fusion**: Custom kernels may be needed for fused nested operations

---

## Common Patterns

### Transformer with Variable-Length Inputs

```python
def transformer_block(q, k, v, mask=None):
    """
    Args:
        q, k, v: Nested tensors of shape [B, *, D]
    """
    # Project to multi-head
    q = linear_q(q).unflatten(-1, (num_heads, head_dim)).transpose(1, 2)
    k = linear_k(k).unflatten(-1, (num_heads, head_dim)).transpose(1, 2)
    v = linear_v(v).unflatten(-1, (num_heads, head_dim)).transpose(1, 2)

    # Attention (uses optimized nested tensor kernels)
    attn_output = F.scaled_dot_product_attention(q, k, v)

    # Project back
    attn_output = attn_output.transpose(1, 2).flatten(-2, -1)
    return linear_out(attn_output)
```

### Batch Processing with Variable Lengths

```python
def process_variable_batch(sequences):
    """Process batch of variable-length sequences"""
    # Create nested tensor
    nt = torch.nested.nested_tensor(sequences, layout=torch.jagged)

    # Apply operations
    nt = F.layer_norm(nt, normalized_shape=(nt.shape[-1],))
    nt = F.relu(nt)

    # Convert back if needed
    return nt.unbind()
```

---

## Summary Table

| Function | Purpose | Returns |
|----------|---------|---------|
| `nested_tensor` | Create from list (leaf) | NestedTensor |
| `as_nested_tensor` | Create from list (with autograd) | NestedTensor |
| `nested_tensor_from_jagged` | Create from packed values/offsets | NestedTensor (view) |
| `to_padded_tensor` | Convert to regular tensor | Tensor (copy) |
| `narrow` | Create from variable slices | NestedTensor |
| `masked_select` | Create from masked selection | NestedTensor |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| `.values()` | Underlying packed data buffer |
| `.offsets()` | Offsets into packed dimension |
| `.lengths()` | Explicit lengths (optional) |
| `.shape` | Symbolic shape with nested ints |

### Layout Comparison

| Aspect | `torch.strided` | `torch.jagged` |
|--------|-----------------|----------------|
| Storage | Separate per element | Packed contiguous |
| Memory | Higher overhead | More efficient |
| SDPA | Less optimized | Highly optimized |
| Views | Full support | View of values |
| Creation | Default | Recommended for performance |
