# Masked Tensors (torch.masked)

## Purpose

Masked tensors provide a way to perform operations on tensors while ignoring certain elements based on a boolean mask. This enables:
- Padding-free computation on variable-length sequences
- Efficient attention with padding masks
- Selective reduction operations
- Sparse tensor compatibility

**Status**: The PyTorch MaskedTensor API is in **prototype stage** and subject to change.

**Reference Files:**
- `torch/masked/__init__.py` - Public API
- `torch/masked/_ops.py` - Masked operations (~1,800 lines)
- `torch/masked/maskedtensor/core.py` - MaskedTensor class
- `torch/masked/maskedtensor/creation.py` - Creation functions

---

## Overview

### The Masking Concept

A masked tensor consists of:
- **Data tensor**: The actual values
- **Mask tensor**: Boolean tensor indicating which values are "valid"

```python
import torch
from torch.masked import MaskedTensor

data = torch.tensor([1.0, 2.0, 3.0, 4.0])
mask = torch.tensor([True, True, False, True])

# Elements where mask=True are included
# Elements where mask=False are ignored
mt = MaskedTensor(data, mask)
# Conceptually: [1.0, 2.0, --, 4.0]
```

### Mask Semantics

| Mask Value | Meaning |
|------------|---------|
| `True` | Include element in computation |
| `False` | Exclude/ignore element |

---

## Core API

### MaskedTensor (Class)

**Purpose**: A tensor subclass that pairs data with a boolean mask.

**Construction**:
```python
torch.masked.MaskedTensor(data, mask, requires_grad=False)
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | Tensor | The data tensor |
| `mask` | Tensor | Boolean mask tensor (same shape as data) |
| `requires_grad` | bool | Enable gradient tracking |

**Constraints**:
- `data` and `mask` must have the same layout
- `mask` must be boolean dtype
- Supports `torch.strided`, `torch.sparse_coo`, and `torch.sparse_csr` layouts

**Usage Examples**:
```python
from torch.masked import MaskedTensor

# Basic creation
data = torch.arange(6).reshape(2, 3).float()
mask = torch.tensor([[True, False, False],
                     [True, True, False]])

mt = MaskedTensor(data, mask)
print(mt)
# MaskedTensor(
#   [[ 0.0,   --,   --],
#    [ 3.0,  4.0,   --]]
# )

# Access underlying data and mask
mt.get_data()  # Returns the data tensor
mt.get_mask()  # Returns the mask tensor
```

---

### masked_tensor (Function)

**Purpose**: Create a MaskedTensor from data and mask.

**Signature**:
```python
torch.masked.masked_tensor(data, mask, requires_grad=False) -> MaskedTensor
```

**Usage**:
```python
from torch.masked import masked_tensor

mt = masked_tensor(data, mask)
```

---

### as_masked_tensor (Function)

**Purpose**: Create a MaskedTensor preserving autograd history.

**Signature**:
```python
torch.masked.as_masked_tensor(data, mask) -> MaskedTensor
```

**Usage**:
```python
from torch.masked import as_masked_tensor

data = torch.randn(3, 4, requires_grad=True)
mask = torch.ones(3, 4, dtype=torch.bool)

mt = as_masked_tensor(data, mask)
# Gradients flow through to original data tensor
```

---

### is_masked_tensor (Function)

**Purpose**: Check if an object is a MaskedTensor.

**Signature**:
```python
torch.masked.is_masked_tensor(obj) -> bool
```

**Usage**:
```python
from torch.masked import is_masked_tensor, MaskedTensor

mt = MaskedTensor(data, mask)
is_masked_tensor(mt)  # True
is_masked_tensor(data)  # False
```

---

## Masked Operations

### Reduction Operations

All reduction operations ignore masked-out elements.

#### sum

**Purpose**: Sum of unmasked elements.

**Signature**:
```python
torch.masked.sum(input, dim=None, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor |
| `dim` | int/tuple | Dimension(s) to reduce |
| `keepdim` | bool | Keep reduced dimensions |
| `dtype` | torch.dtype | Output dtype |
| `mask` | Tensor | Boolean mask |

**Usage Examples**:
```python
import torch
from torch.masked import sum as masked_sum

input = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])
mask = torch.tensor([[True, False, True],
                     [True, True, False]])

# Sum with mask
result = masked_sum(input, dim=1, mask=mask)
# tensor([4., 9.])  # [1+3, 4+5]

# Sum all unmasked elements
result = masked_sum(input, mask=mask)
# tensor(13.)  # 1+3+4+5
```

---

#### mean

**Purpose**: Mean of unmasked elements.

**Signature**:
```python
torch.masked.mean(input, dim=None, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

**Usage**:
```python
from torch.masked import mean as masked_mean

result = masked_mean(input, dim=1, mask=mask)
# tensor([2., 4.5])  # [(1+3)/2, (4+5)/2]
```

---

#### prod

**Purpose**: Product of unmasked elements.

**Signature**:
```python
torch.masked.prod(input, dim=None, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

---

#### amax / amin

**Purpose**: Maximum/minimum of unmasked elements.

**Signatures**:
```python
torch.masked.amax(input, dim=None, *, keepdim=False, dtype=None, mask=None) -> Tensor
torch.masked.amin(input, dim=None, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

**Usage**:
```python
from torch.masked import amax, amin

max_vals = amax(input, dim=1, mask=mask)
# tensor([3., 5.])

min_vals = amin(input, dim=1, mask=mask)
# tensor([1., 4.])
```

---

#### argmax / argmin

**Purpose**: Indices of maximum/minimum unmasked elements.

**Signatures**:
```python
torch.masked.argmax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
torch.masked.argmin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

---

#### median

**Purpose**: Median of unmasked elements.

**Signature**:
```python
torch.masked.median(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

---

#### var / std

**Purpose**: Variance/standard deviation of unmasked elements.

**Signatures**:
```python
torch.masked.var(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
torch.masked.std(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

---

### Cumulative Operations

#### cumsum

**Purpose**: Cumulative sum over unmasked elements.

**Signature**:
```python
torch.masked.cumsum(input, dim, *, dtype=None, mask=None) -> Tensor
```

**Usage**:
```python
from torch.masked import cumsum

input = torch.tensor([1., 2., 3., 4.])
mask = torch.tensor([True, False, True, True])

result = cumsum(input, dim=0, mask=mask)
# tensor([1., 1., 4., 8.])  # 1, (skip 2), 1+3, 1+3+4
```

---

#### cumprod

**Purpose**: Cumulative product over unmasked elements.

**Signature**:
```python
torch.masked.cumprod(input, dim, *, dtype=None, mask=None) -> Tensor
```

---

### Normalization Operations

#### softmax

**Purpose**: Softmax over unmasked elements only.

**Signature**:
```python
torch.masked.softmax(input, dim, *, dtype=None, mask=None) -> Tensor
```

**Usage**:
```python
from torch.masked import softmax

input = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.]])
mask = torch.tensor([[True, False, True],
                     [True, True, True]])

result = softmax(input, dim=1, mask=mask)
# Row 0: softmax([1, 3]) with middle element undefined
# Row 1: softmax([4, 5, 6])
```

**Important**: Masked-out positions have **undefined** values in the output.

---

#### log_softmax

**Purpose**: Log-softmax over unmasked elements.

**Signature**:
```python
torch.masked.log_softmax(input, dim, *, dtype=None, mask=None) -> Tensor
```

---

#### softmin

**Purpose**: Softmin over unmasked elements.

**Signature**:
```python
torch.masked.softmin(input, dim, *, dtype=None, mask=None) -> Tensor
```

---

#### logsumexp

**Purpose**: Log-sum-exp over unmasked elements.

**Signature**:
```python
torch.masked.logsumexp(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

---

#### logaddexp

**Purpose**: Element-wise log-add-exp with masking.

**Signature**:
```python
torch.masked.logaddexp(input1, input2, *, mask=None) -> Tensor
```

---

#### norm

**Purpose**: Norm over unmasked elements.

**Signature**:
```python
torch.masked.norm(input, ord, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
```

---

#### normalize

**Purpose**: Normalize over unmasked elements.

**Signature**:
```python
torch.masked.normalize(input, ord, dim, *, eps=1e-12, dtype=None, mask=None) -> Tensor
```

---

## Mask Broadcasting

The mask doesn't need to match the input shape exactly—it must be **broadcastable**.

```python
input = torch.randn(3, 4, 5)

# Mask per batch element (broadcasts over features)
mask = torch.tensor([True, True, False])[:, None, None]  # Shape: (3, 1, 1)

# Mask per feature (broadcasts over batch and middle dim)
mask = torch.tensor([True, False, True, True, False])  # Shape: (5,)

# Full mask
mask = torch.ones(3, 4, 5, dtype=torch.bool)
```

**Output Mask Computation**:
```python
output_mask = torch.any(
    torch.broadcast_to(mask, input.shape),
    dim=dim,
    keepdim=keepdim
)
```

---

## Sparse Layout Support

MaskedTensor works with sparse layouts:

```python
# COO sparse
data_coo = torch.sparse_coo_tensor(
    indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
    values=torch.tensor([1., 2., 3.]),
    size=(3, 3)
)
mask_coo = torch.sparse_coo_tensor(
    indices=torch.tensor([[0, 1, 2], [0, 1, 2]]),
    values=torch.tensor([True, False, True]),
    size=(3, 3)
)

mt = MaskedTensor(data_coo, mask_coo)

# CSR sparse
data_csr = data_coo.to_sparse_csr()
mask_csr = mask_coo.to_sparse_csr()

mt = MaskedTensor(data_csr, mask_csr)
```

---

## Undefined Values in Output

For fully masked-out dimensions, output values are **undefined**:

```python
input = torch.tensor([[1., 2., 3.]])
mask = torch.tensor([[False, False, False]])  # All masked out!

result = masked_sum(input, dim=1, mask=mask)
# result[0] is UNDEFINED (may be 0, nan, or anything)
```

**Rationale**: Allowing undefined values enables more efficient storage and computation.

---

## Common Patterns

### Attention with Padding Mask

```python
def masked_attention(query, key, value, padding_mask):
    """
    Args:
        query: (batch, seq_q, dim)
        key: (batch, seq_k, dim)
        value: (batch, seq_k, dim)
        padding_mask: (batch, seq_k) - True for valid positions
    """
    from torch.masked import softmax as masked_softmax

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
    # scores: (batch, seq_q, seq_k)

    # Expand mask for attention
    # mask: (batch, 1, seq_k) -> broadcasts to (batch, seq_q, seq_k)
    mask = padding_mask.unsqueeze(1)

    # Masked softmax ignores padding positions
    attn_weights = masked_softmax(scores, dim=-1, mask=mask)

    # Apply attention
    return torch.matmul(attn_weights, value)
```

### Sequence Mean Pooling

```python
def masked_mean_pool(sequences, lengths, max_len):
    """
    Mean pool variable-length sequences.

    Args:
        sequences: (batch, max_len, dim)
        lengths: (batch,) - actual lengths
    """
    from torch.masked import mean as masked_mean

    # Create mask from lengths
    batch_size = sequences.size(0)
    positions = torch.arange(max_len, device=sequences.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    # mask: (batch, max_len)

    # Expand mask for features
    mask = mask.unsqueeze(-1)  # (batch, max_len, 1)

    # Mean over sequence dimension, ignoring padding
    return masked_mean(sequences, dim=1, mask=mask)
```

### Selective Loss Computation

```python
def masked_cross_entropy(logits, targets, mask):
    """
    Cross entropy ignoring masked positions.

    Args:
        logits: (batch, seq, vocab)
        targets: (batch, seq)
        mask: (batch, seq) - True for valid positions
    """
    from torch.masked import sum as masked_sum, mean as masked_mean

    # Compute per-position loss
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    ).view(targets.shape)

    # Masked mean
    total_loss = masked_sum(ce_loss, mask=mask)
    num_valid = mask.sum()

    return total_loss / num_valid
```

---

## MLX Porting Considerations

### No Native MaskedTensor

MLX doesn't have a MaskedTensor type. Use explicit masking:

```python
import mlx.core as mx

def masked_sum_mlx(input, mask, dim=None, keepdim=False):
    """MLX implementation of masked sum"""
    # Zero out masked elements
    masked_input = mx.where(mask, input, mx.zeros_like(input))

    return mx.sum(masked_input, axis=dim, keepdims=keepdim)

def masked_mean_mlx(input, mask, dim=None, keepdim=False):
    """MLX implementation of masked mean"""
    masked_input = mx.where(mask, input, mx.zeros_like(input))

    # Sum of values
    total = mx.sum(masked_input, axis=dim, keepdims=keepdim)

    # Count of valid elements
    count = mx.sum(mask.astype(input.dtype), axis=dim, keepdims=keepdim)

    return total / count
```

### Masked Softmax for MLX

```python
def masked_softmax_mlx(input, mask, dim=-1):
    """MLX implementation of masked softmax"""
    # Set masked positions to -inf before softmax
    masked_input = mx.where(mask, input, mx.full_like(input, float('-inf')))

    return mx.softmax(masked_input, axis=dim)

def masked_log_softmax_mlx(input, mask, dim=-1):
    """MLX implementation of masked log softmax"""
    masked_input = mx.where(mask, input, mx.full_like(input, float('-inf')))

    # Stable log softmax
    max_val = mx.max(masked_input, axis=dim, keepdims=True)
    shifted = masked_input - max_val
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=dim, keepdims=True))

    return shifted - log_sum_exp
```

### MaskedArray Wrapper for MLX

```python
class MaskedArray:
    """MLX equivalent of PyTorch MaskedTensor"""

    def __init__(self, data: mx.array, mask: mx.array):
        if data.shape != mask.shape:
            raise ValueError("data and mask must have same shape")
        self.data = data
        self.mask = mask  # True = valid, False = masked

    def sum(self, axis=None, keepdims=False):
        masked_data = mx.where(self.mask, self.data, 0)
        return mx.sum(masked_data, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        masked_data = mx.where(self.mask, self.data, 0)
        total = mx.sum(masked_data, axis=axis, keepdims=keepdims)
        count = mx.sum(self.mask.astype(self.data.dtype), axis=axis, keepdims=keepdims)
        return total / count

    def max(self, axis=None, keepdims=False):
        masked_data = mx.where(self.mask, self.data, float('-inf'))
        return mx.max(masked_data, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        masked_data = mx.where(self.mask, self.data, float('inf'))
        return mx.min(masked_data, axis=axis, keepdims=keepdims)

    def softmax(self, axis=-1):
        masked_data = mx.where(self.mask, self.data, float('-inf'))
        return mx.softmax(masked_data, axis=axis)
```

---

## Summary Table

### Creation Functions

| Function | Purpose |
|----------|---------|
| `MaskedTensor(data, mask)` | Create from tensors |
| `masked_tensor(data, mask)` | Functional creation |
| `as_masked_tensor(data, mask)` | Preserve autograd |
| `is_masked_tensor(obj)` | Type check |

### Reduction Operations

| Function | Identity | Description |
|----------|----------|-------------|
| `sum` | 0 | Sum of unmasked elements |
| `prod` | 1 | Product of unmasked elements |
| `mean` | N/A | Mean of unmasked elements |
| `amax` | -inf | Maximum of unmasked elements |
| `amin` | +inf | Minimum of unmasked elements |
| `argmax` | N/A | Index of maximum |
| `argmin` | N/A | Index of minimum |
| `median` | N/A | Median of unmasked elements |
| `var` | N/A | Variance of unmasked elements |
| `std` | N/A | Standard deviation |

### Cumulative Operations

| Function | Description |
|----------|-------------|
| `cumsum` | Cumulative sum over unmasked |
| `cumprod` | Cumulative product over unmasked |

### Normalization Operations

| Function | Description |
|----------|-------------|
| `softmax` | Softmax over unmasked elements |
| `log_softmax` | Log-softmax over unmasked |
| `softmin` | Softmin over unmasked |
| `logsumexp` | Log-sum-exp over unmasked |
| `logaddexp` | Element-wise log-add-exp |
| `norm` | Norm over unmasked elements |
| `normalize` | Normalize over unmasked |

### Key Semantics

| Aspect | Behavior |
|--------|----------|
| Mask True | Element included in computation |
| Mask False | Element ignored/excluded |
| Broadcasting | Mask can be smaller than input |
| Fully masked dim | Output value is undefined |
| Sparse support | COO and CSR layouts supported |
