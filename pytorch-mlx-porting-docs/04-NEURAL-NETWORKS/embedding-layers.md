# PyTorch Embedding Layers

## Purpose

This document provides comprehensive documentation of PyTorch's embedding modules. Embeddings are fundamental for:

1. Natural Language Processing (word embeddings, token embeddings)
2. Large Language Models (vocabulary representations)
3. Recommendation Systems (user/item embeddings)
4. Categorical Feature Encoding (learned representations)

**Source**: [torch/nn/modules/sparse.py](../../reference/pytorch/torch/nn/modules/sparse.py)

## Architecture Overview

### Embedding Module Hierarchy

```
                        Module
                          |
            ┌─────────────┴─────────────┐
       Embedding                  EmbeddingBag
    (simple lookup)          (lookup + aggregation)
```

### Embedding vs EmbeddingBag

| Feature | Embedding | EmbeddingBag |
|---------|-----------|--------------|
| **Operation** | Lookup only | Lookup + reduction |
| **Output shape** | `(*, H)` | `(B, H)` |
| **Use case** | Sequences | Aggregated bags |
| **Memory** | Returns full tensors | More memory efficient |
| **Modes** | N/A | sum, mean, max |

---

## 1. Embedding

A simple lookup table that stores embeddings of a fixed dictionary and size.

### How It Works

```
Vocabulary (num_embeddings=5):        Input indices:      Output:
┌───────────────────────────┐         [1, 3, 0]           ┌────────────────┐
│ 0: [0.1, 0.2, 0.3, 0.4]   │                             │ [0.5, 0.6,...] │ <- idx 1
│ 1: [0.5, 0.6, 0.7, 0.8]   │   ──────────────────────>   │ [1.3, 1.4,...] │ <- idx 3
│ 2: [0.9, 1.0, 1.1, 1.2]   │                             │ [0.1, 0.2,...] │ <- idx 0
│ 3: [1.3, 1.4, 1.5, 1.6]   │                             └────────────────┘
│ 4: [1.7, 1.8, 1.9, 2.0]   │
└───────────────────────────┘
     weight: (5, 4)                                        output: (3, 4)
```

### Constructor

```python
nn.Embedding(
    num_embeddings: int,       # Size of vocabulary (V)
    embedding_dim: int,        # Dimension of embeddings (H)
    padding_idx: int = None,   # Index that returns zeros, no gradient
    max_norm: float = None,    # Renormalize if norm > max_norm
    norm_type: float = 2.0,    # p-norm type for max_norm
    scale_grad_by_freq: bool = False,  # Scale gradients by word frequency
    sparse: bool = False,      # Use sparse gradient updates
    _weight: Tensor = None,    # Custom initial weights
    _freeze: bool = False,     # Freeze weights (no training)
    device=None,
    dtype=None,
)
```

### Parameters Explained

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_embeddings` | int | Size of the dictionary (vocabulary size) |
| `embedding_dim` | int | Size of each embedding vector |
| `padding_idx` | int | Index whose embedding stays zero, excluded from gradients |
| `max_norm` | float | If set, renormalize embeddings exceeding this norm |
| `norm_type` | float | p-norm type for max_norm (default: 2 = L2 norm) |
| `scale_grad_by_freq` | bool | Scale gradients by inverse word frequency |
| `sparse` | bool | Use sparse gradients (memory efficient for large vocabs) |

### Attributes

```python
embedding.weight  # Tensor of shape (num_embeddings, embedding_dim)
embedding.num_embeddings  # V
embedding.embedding_dim   # H
embedding.padding_idx     # Index or None
```

### Shape

- **Input**: `(*)` - IntTensor or LongTensor of arbitrary shape containing indices
- **Output**: `(*, H)` - Same shape as input with embedding dimension appended

### Weight Initialization

```python
def reset_parameters(self) -> None:
    init.normal_(self.weight)  # N(0, 1) initialization
    self._fill_padding_idx_with_zero()

def _fill_padding_idx_with_zero(self) -> None:
    if self.padding_idx is not None:
        with torch.no_grad():
            self.weight[self.padding_idx].fill_(0)
```

### Basic Example

```python
# Vocabulary of 10 words, 3-dimensional embeddings
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# Look up embeddings for a batch of indices
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])  # (batch=2, seq_len=4)
output = embedding(input)  # (2, 4, 3)

print(output.shape)  # torch.Size([2, 4, 3])
```

### Padding Index Example

```python
# Word at index 0 is padding (e.g., <PAD> token)
embedding = nn.Embedding(10, 3, padding_idx=0)

input = torch.LongTensor([[0, 2, 0, 5]])
output = embedding(input)
# output[0, 0] and output[0, 2] are zeros (padding positions)
# Gradients do not flow through padding_idx
```

### Max Norm Example

```python
# Constrain embedding norms to 1.0
embedding = nn.Embedding(10, 3, max_norm=1.0)

# Note: max_norm modifies weights in-place during forward!
# If using weight for other operations, clone first:
W = embedding.weight.clone() @ other_matrix  # Safe
```

### Sparse Gradients

For large vocabularies, sparse gradients are more memory efficient:

```python
# Large vocabulary with sparse gradients
embedding = nn.Embedding(50000, 300, sparse=True)

# Only these optimizers support sparse gradients:
# - optim.SGD (CUDA and CPU)
# - optim.SparseAdam (CUDA and CPU)
# - optim.Adagrad (CPU)

optimizer = torch.optim.SparseAdam(embedding.parameters())
```

### Loading Pretrained Embeddings

```python
# From a tensor (e.g., GloVe, Word2Vec)
pretrained = torch.FloatTensor([
    [1.0, 2.3, 3.0],  # Word 0
    [4.0, 5.1, 6.3],  # Word 1
])

embedding = nn.Embedding.from_pretrained(
    pretrained,
    freeze=True,           # Don't update during training
    padding_idx=None,      # Optional padding index
)

# Get embedding for word 1
input = torch.LongTensor([1])
output = embedding(input)  # tensor([[4.0, 5.1, 6.3]])
```

---

## 2. EmbeddingBag

Computes sums, means, or max of bags of embeddings without instantiating intermediate tensors. More efficient than Embedding + reduction for aggregated representations.

### How It Works

```
Input (1D with offsets):                    Output (after reduction):
indices: [2, 1, 4, 0, 3, 2, 1]              ┌────────────────┐
offsets: [0, 3, 5]                          │ bag 0: reduce([2,1,4]) │
                                            │ bag 1: reduce([0,3])   │
Bag 0: indices[0:3] = [2, 1, 4]             │ bag 2: reduce([2,1])   │
Bag 1: indices[3:5] = [0, 3]                └────────────────┘
Bag 2: indices[5:7] = [2, 1]                     output: (3, H)
```

### Constructor

```python
nn.EmbeddingBag(
    num_embeddings: int,         # Size of vocabulary
    embedding_dim: int,          # Dimension of embeddings
    max_norm: float = None,      # Max norm constraint
    norm_type: float = 2.0,      # Norm type
    scale_grad_by_freq: bool = False,
    mode: str = 'mean',          # 'sum', 'mean', or 'max'
    sparse: bool = False,        # Sparse gradients
    include_last_offset: bool = False,  # CSR-style offsets
    padding_idx: int = None,     # Padding index
    device=None,
    dtype=None,
)
```

### Modes

| Mode | Operation | Equivalent To |
|------|-----------|---------------|
| `'sum'` | Sum of embeddings | `Embedding(x).sum(dim=1)` |
| `'mean'` | Mean of embeddings | `Embedding(x).mean(dim=1)` |
| `'max'` | Max of embeddings | `Embedding(x).max(dim=1)` |

### Shape

**2D Input** (fixed-length bags):
- Input: `(B, N)` - B bags, each with N indices
- Output: `(B, H)` - One vector per bag

**1D Input with offsets** (variable-length bags):
- Input: `(total_indices,)` - Concatenated indices
- Offsets: `(B,)` - Starting position of each bag
- Output: `(B, H)` - One vector per bag

### Basic Example

```python
# Sum mode for text classification (bag-of-words style)
embedding_bag = nn.EmbeddingBag(10, 3, mode='sum')

# 2D input: 2 bags, 4 indices each
input_2d = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
output = embedding_bag(input_2d)  # (2, 3)

# 1D input with offsets: variable-length bags
input_1d = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])  # Concatenated
offsets = torch.LongTensor([0, 4])  # Bag 0 starts at 0, Bag 1 at 4
output = embedding_bag(input_1d, offsets)  # (2, 3)
```

### Per-Sample Weights

For weighted sums (only with `mode='sum'`):

```python
embedding_bag = nn.EmbeddingBag(10, 3, mode='sum')

input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
offsets = torch.LongTensor([0, 4])
weights = torch.FloatTensor([0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5])

output = embedding_bag(input, offsets, per_sample_weights=weights)
# Each embedding is scaled by its weight before summing
```

### Padding Index

```python
embedding_bag = nn.EmbeddingBag(10, 3, mode='sum', padding_idx=2)

# Indices equal to padding_idx don't contribute to the result
input = torch.LongTensor([2, 2, 2, 2, 4, 3, 2, 9])
offsets = torch.LongTensor([0, 4])

output = embedding_bag(input, offsets)
# Bag 0: all padding -> zeros
# Bag 1: sum of embeddings for [4, 3, 9] (2 is excluded)
```

### include_last_offset (CSR Format)

```python
# Standard offsets: [0, 3, 5] means bags at [0:3] and [3:5]
# CSR-style offsets: [0, 3, 5, 8] includes end position

embedding_bag = nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)

input = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8])
offsets = torch.LongTensor([0, 3, 5, 8])  # 3 bags + end marker

output = embedding_bag(input, offsets)  # (3, 3)
```

### Loading Pretrained

```python
pretrained = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])

embedding_bag = nn.EmbeddingBag.from_pretrained(
    pretrained,
    freeze=True,
    mode='sum',
    padding_idx=None,
)
```

---

## Comparison: Embedding vs EmbeddingBag

### Memory Usage

```python
# For text classification with 1000-word documents:
vocab_size = 50000
embed_dim = 300
doc_len = 1000
batch_size = 32

# Embedding approach:
embedding = nn.Embedding(vocab_size, embed_dim)
# Memory: batch_size * doc_len * embed_dim = 32 * 1000 * 300 = 9.6M floats

# EmbeddingBag approach:
embedding_bag = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
# Memory: batch_size * embed_dim = 32 * 300 = 9.6K floats
# ~1000x more memory efficient!
```

### When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Sequence models (RNN, Transformer) | Embedding |
| Word/token sequences | Embedding |
| Bag-of-words models | EmbeddingBag |
| Document classification | EmbeddingBag |
| User/item aggregation | EmbeddingBag |
| Memory-constrained | EmbeddingBag |

---

## Common Patterns

### Positional Embeddings

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.token_embed(x) + self.pos_embed(positions)
```

### Shared Embeddings (Tied Weights)

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        # Tie weights: decoder uses same weights as embedding
        self.decoder.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        # ... transformer layers ...
        return self.decoder(x)
```

### Multi-Hot Encoding with EmbeddingBag

```python
class MultiLabelEmbedding(nn.Module):
    """For items with multiple categorical features."""
    def __init__(self, num_categories, embed_dim):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(num_categories, embed_dim, mode='sum')

    def forward(self, indices, offsets):
        # indices: all category IDs, offsets: start of each sample
        return self.embedding_bag(indices, offsets)

# Example: Movie genres
# Movie 1: [Action, Comedy] = indices [0, 1]
# Movie 2: [Drama, Romance, Comedy] = indices [2, 3, 1]
indices = torch.LongTensor([0, 1, 2, 3, 1])
offsets = torch.LongTensor([0, 2])  # Movie 1 at 0, Movie 2 at 2
```

---

## MLX Mapping

### Direct Mapping

```python
# PyTorch                          # MLX
nn.Embedding(V, H)                 mlx.nn.Embedding(V, H)
```

### MLX Embedding Implementation

```python
import mlx.core as mx
import mlx.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialize from N(0, 1)
        self.weight = mx.random.normal(shape=(num_embeddings, embedding_dim))

    def __call__(self, x):
        # Simple table lookup using advanced indexing
        return self.weight[x]
```

### MLX EmbeddingBag (Not Built-in)

```python
class EmbeddingBag(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = 'mean'):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.mode = mode

    def __call__(self, x, offsets=None):
        if x.ndim == 2:
            # 2D input: (B, N) -> aggregate over dim 1
            embeddings = self.embedding(x)  # (B, N, H)
            if self.mode == 'sum':
                return mx.sum(embeddings, axis=1)
            elif self.mode == 'mean':
                return mx.mean(embeddings, axis=1)
            elif self.mode == 'max':
                return mx.max(embeddings, axis=1)
        else:
            # 1D with offsets: requires manual segmentation
            raise NotImplementedError("1D input with offsets not yet implemented")
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| `padding_idx` | Built-in | Manual implementation |
| `max_norm` | Built-in (in-place) | Manual implementation |
| `sparse` | Built-in | Not available |
| `EmbeddingBag` | Built-in | Manual implementation |
| `from_pretrained` | Class method | Load weights manually |

### Converting PyTorch Embeddings to MLX

```python
def convert_embedding(pt_embedding):
    """Convert PyTorch Embedding to MLX format."""
    weight = pt_embedding.weight.detach().numpy()
    mlx_embedding = Embedding(
        pt_embedding.num_embeddings,
        pt_embedding.embedding_dim
    )
    mlx_embedding.weight = mx.array(weight)
    return mlx_embedding
```

---

## Gradient Considerations

### Sparse vs Dense Gradients

```python
# Dense gradients (default): Full gradient tensor
embedding = nn.Embedding(50000, 300)
# grad.shape = (50000, 300) - large!

# Sparse gradients: Only non-zero entries stored
embedding = nn.Embedding(50000, 300, sparse=True)
# grad is a sparse tensor with only accessed indices
```

### Optimizer Support for Sparse

| Optimizer | Sparse Support |
|-----------|----------------|
| SGD | Yes (CUDA, CPU) |
| SparseAdam | Yes (CUDA, CPU) |
| Adagrad | Yes (CPU only) |
| Adam | No |
| AdamW | No |
| RMSprop | No |

### Gradient Scaling by Frequency

```python
# Rare words get larger updates
embedding = nn.Embedding(vocab_size, embed_dim, scale_grad_by_freq=True)

# Gradient for word i is scaled by 1/count(i) in the mini-batch
# Helps rare words learn faster
```

---

## Summary

### Quick Reference

| Module | Input | Output | Use Case |
|--------|-------|--------|----------|
| Embedding | `(*)` indices | `(*, H)` | Token sequences |
| EmbeddingBag | `(B, N)` or 1D+offsets | `(B, H)` | Aggregated representations |

### Parameter Count

```
Embedding: num_embeddings * embedding_dim
         = V * H

Example: Embedding(50000, 300) = 15,000,000 parameters = 60MB (float32)
```

### Memory Tips

1. Use `sparse=True` for large vocabularies (>100K)
2. Use `EmbeddingBag` instead of `Embedding + reduce` when possible
3. Consider vocabulary pruning for very large vocabs
4. Use half-precision (`dtype=torch.float16`) for memory savings

---

## Implementation Files

**Source**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/torch/nn/modules/sparse.py`

**Functional API**:
- `F.embedding(input, weight, ...)`
- `F.embedding_bag(input, weight, offsets, ...)`

**YAML Definitions** (`aten/src/ATen/native/native_functions.yaml`):
- `embedding`, `embedding_dense_backward`, `embedding_sparse_backward`
- `embedding_bag`, `_embedding_bag`, `_embedding_bag_forward_only`
