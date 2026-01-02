# Distance and Similarity Functions

## Overview

PyTorch provides modules and functions for computing distances and similarities between vectors. These are commonly used in metric learning, embedding comparisons, nearest neighbor search, and loss computation.

**Reference File:** `torch/nn/modules/distance.py`

## Available Operations

```
Distance and Similarity
├── nn.PairwiseDistance      - p-norm distance between pairs
├── nn.CosineSimilarity      - Cosine similarity between pairs
├── F.pairwise_distance      - Functional pairwise distance
├── F.cosine_similarity      - Functional cosine similarity
├── F.pdist                  - Pairwise distances within a batch
└── torch.cdist              - Cross-batch pairwise distances
```

---

## nn.PairwiseDistance

Computes the p-norm distance between corresponding vectors in two inputs.

### Class Definition

```python
class PairwiseDistance(nn.Module):
    def __init__(
        self,
        p: float = 2.0,           # Norm degree (p-norm)
        eps: float = 1e-6,        # Epsilon for numerical stability
        keepdim: bool = False     # Keep output dimensions
    )
```

### Mathematical Formula

```
dist(x, y) = ||x - y + ε||_p

where ||z||_p = (Σ|z_i|^p)^(1/p)
```

### Basic Usage

```python
import torch
import torch.nn as nn

# L2 (Euclidean) distance
pdist = nn.PairwiseDistance(p=2)

x1 = torch.randn(100, 128)  # 100 vectors, 128 dimensions
x2 = torch.randn(100, 128)

distances = pdist(x1, x2)   # Shape: (100,)
print(distances.shape)      # torch.Size([100])
```

### Different p-norms

```python
# L1 distance (Manhattan)
l1_dist = nn.PairwiseDistance(p=1)

# L2 distance (Euclidean)
l2_dist = nn.PairwiseDistance(p=2)

# L-infinity distance (Chebyshev)
linf_dist = nn.PairwiseDistance(p=float('inf'))

x1 = torch.tensor([[1., 2., 3.]])
x2 = torch.tensor([[4., 5., 6.]])

print(l1_dist(x1, x2))      # tensor([9.])      # |1-4| + |2-5| + |3-6|
print(l2_dist(x1, x2))      # tensor([5.1962])  # sqrt(9+9+9)
print(linf_dist(x1, x2))    # tensor([3.])      # max(|1-4|, |2-5|, |3-6|)
```

### Shape Behavior

```python
pdist = nn.PairwiseDistance(p=2)

# Batched inputs
x1 = torch.randn(32, 64)   # (N, D)
x2 = torch.randn(32, 64)
out = pdist(x1, x2)        # (N,) = (32,)

# Unbatched inputs
x1 = torch.randn(64)       # (D,)
x2 = torch.randn(64)
out = pdist(x1, x2)        # () scalar

# With keepdim
pdist_keep = nn.PairwiseDistance(p=2, keepdim=True)
out = pdist_keep(torch.randn(32, 64), torch.randn(32, 64))
print(out.shape)           # torch.Size([32, 1])
```

---

## nn.CosineSimilarity

Computes the cosine similarity between corresponding vectors.

### Class Definition

```python
class CosineSimilarity(nn.Module):
    def __init__(
        self,
        dim: int = 1,             # Dimension for similarity computation
        eps: float = 1e-8         # Epsilon to avoid division by zero
    )
```

### Mathematical Formula

```
similarity = (x₁ · x₂) / max(||x₁||₂ × ||x₂||₂, ε)
```

### Basic Usage

```python
import torch
import torch.nn as nn

cos_sim = nn.CosineSimilarity(dim=1)

x1 = torch.randn(100, 128)
x2 = torch.randn(100, 128)

similarities = cos_sim(x1, x2)  # Range: [-1, 1]
print(similarities.shape)       # torch.Size([100])
```

### Understanding Cosine Similarity

```python
cos_sim = nn.CosineSimilarity(dim=0)

# Same direction → 1
a = torch.tensor([1., 2., 3.])
b = torch.tensor([2., 4., 6.])
print(cos_sim(a, b))  # tensor(1.)

# Opposite direction → -1
c = torch.tensor([-1., -2., -3.])
print(cos_sim(a, c))  # tensor(-1.)

# Orthogonal → 0
d = torch.tensor([1., 0., 0.])
e = torch.tensor([0., 1., 0.])
print(cos_sim(d, e))  # tensor(0.)
```

### Dimension Selection

```python
# Compute along different dimensions
x1 = torch.randn(4, 3, 5)
x2 = torch.randn(4, 3, 5)

# Along dim=1 (default)
cos1 = nn.CosineSimilarity(dim=1)
out1 = cos1(x1, x2)
print(out1.shape)  # torch.Size([4, 5])

# Along dim=2
cos2 = nn.CosineSimilarity(dim=2)
out2 = cos2(x1, x2)
print(out2.shape)  # torch.Size([4, 3])

# Along dim=0
cos0 = nn.CosineSimilarity(dim=0)
out0 = cos0(x1, x2)
print(out0.shape)  # torch.Size([3, 5])
```

---

## Functional Interface

### F.pairwise_distance()

```python
import torch.nn.functional as F

x1 = torch.randn(32, 128)
x2 = torch.randn(32, 128)

# L2 distance
dist = F.pairwise_distance(x1, x2, p=2)

# L1 distance with keepdim
dist = F.pairwise_distance(x1, x2, p=1, keepdim=True)
```

### F.cosine_similarity()

```python
x1 = torch.randn(32, 128)
x2 = torch.randn(32, 128)

sim = F.cosine_similarity(x1, x2, dim=1)
```

---

## F.pdist

Computes pairwise distances between all pairs in a batch.

```python
# All pairs of distances within a batch
x = torch.randn(10, 128)  # 10 vectors

# Returns (N*(N-1)/2,) = (45,) distances
distances = F.pdist(x, p=2)
print(distances.shape)  # torch.Size([45])

# Order: (0,1), (0,2), ..., (0,9), (1,2), ..., (8,9)
```

---

## torch.cdist

Computes pairwise distances between two batches of vectors.

```python
# Cross-batch distances
x1 = torch.randn(32, 10, 128)  # 32 batches, 10 vectors each
x2 = torch.randn(32, 15, 128)  # 32 batches, 15 vectors each

# All pairs: (32, 10, 15) - distance from each x1 to each x2
distances = torch.cdist(x1, x2, p=2)
print(distances.shape)  # torch.Size([32, 10, 15])
```

---

## Common Use Cases

### 1. Contrastive Loss

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, x1, x2, label):
        # label: 1 = same class, 0 = different class
        distance = self.pdist(x1, x2)

        # Same class: minimize distance
        # Different class: push apart by margin
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)

        return loss.mean()
```

### 2. Triplet Loss

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist(anchor, positive)
        neg_dist = self.pdist(anchor, negative)

        # Push positive closer, negative farther
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
```

### 3. Nearest Neighbor Search

```python
def find_nearest_neighbors(query, database, k=5):
    """Find k nearest neighbors using cosine similarity."""
    # query: (N, D), database: (M, D)

    # Normalize for cosine similarity
    query_norm = F.normalize(query, dim=1)
    db_norm = F.normalize(database, dim=1)

    # Compute similarities: (N, M)
    similarities = query_norm @ db_norm.T

    # Top-k most similar
    values, indices = similarities.topk(k, dim=1)

    return indices, values
```

### 4. Embedding Comparison

```python
class EmbeddingSimilarity(nn.Module):
    def __init__(self, embed_dim, use_cosine=True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.use_cosine = use_cosine
        if use_cosine:
            self.similarity = nn.CosineSimilarity(dim=1)
        else:
            self.distance = nn.PairwiseDistance(p=2)

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)

        if self.use_cosine:
            return self.similarity(e1, e2)
        else:
            return -self.distance(e1, e2)  # Negative for similarity
```

---

## MLX Mapping

### Implementing in MLX

```python
import mlx.core as mx

def pairwise_distance_mlx(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """Compute pairwise p-norm distance."""
    diff = x1 - x2

    if p == float('inf'):
        dist = mx.max(mx.abs(diff), axis=-1, keepdims=keepdim)
    elif p == 1:
        dist = mx.sum(mx.abs(diff), axis=-1, keepdims=keepdim)
    elif p == 2:
        dist = mx.sqrt(mx.sum(diff ** 2 + eps, axis=-1, keepdims=keepdim))
    else:
        dist = mx.sum(mx.abs(diff) ** p + eps, axis=-1, keepdims=keepdim) ** (1/p)

    return dist


def cosine_similarity_mlx(x1, x2, dim=1, eps=1e-8):
    """Compute cosine similarity."""
    # Dot product
    dot = mx.sum(x1 * x2, axis=dim)

    # Norms
    norm1 = mx.sqrt(mx.sum(x1 ** 2, axis=dim))
    norm2 = mx.sqrt(mx.sum(x2 ** 2, axis=dim))

    # Similarity
    return dot / mx.maximum(norm1 * norm2, eps)


def cdist_mlx(x1, x2, p=2.0):
    """Compute cross-batch pairwise distances."""
    # x1: (B, N, D), x2: (B, M, D)
    # Output: (B, N, M)

    # Expand for broadcasting
    x1_exp = mx.expand_dims(x1, axis=2)  # (B, N, 1, D)
    x2_exp = mx.expand_dims(x2, axis=1)  # (B, 1, M, D)

    diff = x1_exp - x2_exp  # (B, N, M, D)

    if p == 2:
        return mx.sqrt(mx.sum(diff ** 2, axis=-1))
    else:
        return mx.sum(mx.abs(diff) ** p, axis=-1) ** (1/p)
```

### MLX Usage Example

```python
import mlx.core as mx

# Pairwise distance
x1 = mx.random.normal((100, 128))
x2 = mx.random.normal((100, 128))

dist = pairwise_distance_mlx(x1, x2, p=2)
print(dist.shape)  # (100,)

# Cosine similarity
sim = cosine_similarity_mlx(x1, x2, dim=1)
print(sim.shape)  # (100,)
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Module API | `nn.PairwiseDistance`, `nn.CosineSimilarity` | Manual implementation |
| Functional | `F.pairwise_distance`, `F.cosine_similarity` | Custom functions |
| pdist | `F.pdist` | Manual with broadcasting |
| cdist | `torch.cdist` | Manual with broadcasting |
| Negative p | Supported | Manual handling |

---

## Performance Considerations

1. **Batch processing** - Use vectorized operations over loops

2. **Memory efficiency** - cdist can be memory-intensive for large batches

3. **Numerical stability** - Use eps to avoid division by zero

4. **Normalized inputs** - Pre-normalize for faster repeated cosine similarity

```python
# Pre-normalize for multiple comparisons
query_norm = F.normalize(query, dim=1)
db_norm = F.normalize(database, dim=1)

# Now cosine similarity is just dot product
similarity = query_norm @ db_norm.T
```

---

## Summary

| Function | Description | Output Range |
|----------|-------------|--------------|
| `PairwiseDistance` | p-norm distance | [0, ∞) |
| `CosineSimilarity` | Cosine angle similarity | [-1, 1] |
| `F.pdist` | All-pairs distances in batch | [0, ∞) |
| `torch.cdist` | Cross-batch distances | [0, ∞) |

### Common p-norms

| p | Name | Formula |
|---|------|---------|
| 1 | Manhattan/L1 | Σ\|x_i - y_i\| |
| 2 | Euclidean/L2 | √(Σ(x_i - y_i)²) |
| ∞ | Chebyshev/L∞ | max\|x_i - y_i\| |

### Shape Reference

```
PairwiseDistance:
  Input:  (N, D) and (N, D)
  Output: (N,) or (N, 1) if keepdim

CosineSimilarity (dim=1):
  Input:  (N, D) and (N, D)
  Output: (N,)

pdist:
  Input:  (N, D)
  Output: (N*(N-1)/2,)

cdist:
  Input:  (B, N, D) and (B, M, D)
  Output: (B, N, M)
```
