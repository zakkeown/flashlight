# Samplers

## Overview

Samplers control how indices are generated for dataset access. They determine the order of data loading, enabling shuffling, weighted sampling, subset selection, and distributed data partitioning. Samplers are the strategy layer between DataLoader and Dataset.

**Reference Files:**
- `torch/utils/data/sampler.py`
- `torch/utils/data/distributed.py`

## Sampler Hierarchy

```
Sampler[T_co] (Abstract Base)
├── SequentialSampler[int]      - Sequential indices 0, 1, 2, ...
├── RandomSampler[int]          - Random permutation or with replacement
├── SubsetRandomSampler[int]    - Random from specified indices
├── WeightedRandomSampler[int]  - Probability-weighted selection
├── BatchSampler[list[int]]     - Wraps sampler to yield batches
└── DistributedSampler[T_co]    - Shards data across processes
```

---

## Base Sampler

Abstract base class defining the sampler interface.

### Class Definition

```python
class Sampler(Generic[T_co]):
    """Base class for all Samplers."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # __len__ is optional - not required but expected by DataLoader
```

### Design Notes

- **Type parameter**: `T_co` is typically `int` (single index) or `list[int]` (batch)
- **No default `__len__`**: Intentionally omitted to allow infinite samplers
- **Iterator protocol**: Must implement `__iter__`, optionally `__len__`

---

## SequentialSampler

Yields indices in order: 0, 1, 2, ..., len(data_source) - 1.

### Class Definition

```python
class SequentialSampler(Sampler[int]):
    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
```

### Use Case

- Default sampler when `shuffle=False` in DataLoader
- Validation/test loops where order matters
- Reproducible iteration

### Example

```python
from torch.utils.data import SequentialSampler

dataset = list(range(100))
sampler = SequentialSampler(dataset)
list(sampler)  # [0, 1, 2, ..., 99]
```

---

## RandomSampler

Yields indices in random order, with or without replacement.

### Class Definition

```python
class RandomSampler(Sampler[int]):
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,  # With or without replacement
        num_samples: int | None = None,  # Number to draw
        generator = None  # torch.Generator for reproducibility
    ) -> None
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `data_source` | Dataset to sample from (must have `__len__`) |
| `replacement` | If True, sample with replacement (can repeat indices) |
| `num_samples` | Total samples to generate (default: len(data_source)) |
| `generator` | Random generator for reproducibility |

### Iteration Behavior

**Without replacement (default):**
```python
# Yields permutation of [0, N-1]
# If num_samples > N, yields multiple complete permutations
for _ in range(num_samples // n):
    yield from torch.randperm(n, generator=generator).tolist()
yield from torch.randperm(n, generator=generator)[:num_samples % n].tolist()
```

**With replacement:**
```python
# Uses randint for efficiency (batched in chunks of 32)
for _ in range(num_samples // 32):
    yield from torch.randint(high=n, size=(32,), generator=generator).tolist()
yield from torch.randint(high=n, size=(num_samples % 32,), generator=generator).tolist()
```

### Use Cases

- Default sampler when `shuffle=True` in DataLoader
- Oversampling with `replacement=True` and `num_samples > len(dataset)`
- Reproducible training with explicit `generator`

### Example

```python
from torch.utils.data import RandomSampler
import torch

dataset = list(range(10))

# Basic shuffled sampling
sampler = RandomSampler(dataset)
print(list(sampler))  # e.g., [3, 7, 0, 5, 2, 8, 1, 6, 4, 9]

# Reproducible sampling
g = torch.Generator()
g.manual_seed(42)
sampler = RandomSampler(dataset, generator=g)

# Oversampling with replacement
sampler = RandomSampler(dataset, replacement=True, num_samples=50)
len(sampler)  # 50
```

---

## SubsetRandomSampler

Randomly samples from a specified list of indices (without replacement).

### Class Definition

```python
class SubsetRandomSampler(Sampler[int]):
    def __init__(
        self,
        indices: Sequence[int],  # Indices to sample from
        generator = None
    ) -> None
```

### Iteration Behavior

```python
def __iter__(self) -> Iterator[int]:
    for i in torch.randperm(len(self.indices), generator=self.generator).tolist():
        yield self.indices[i]
```

### Use Cases

- Train/validation splits
- K-fold cross-validation
- Sampling specific subsets

### Example

```python
from torch.utils.data import SubsetRandomSampler, DataLoader

# Split dataset
full_indices = list(range(1000))
train_indices = full_indices[:800]
val_indices = full_indices[800:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, sampler=train_sampler)
val_loader = DataLoader(dataset, sampler=val_sampler)
```

---

## WeightedRandomSampler

Samples indices with probabilities proportional to given weights.

### Class Definition

```python
class WeightedRandomSampler(Sampler[int]):
    def __init__(
        self,
        weights: Sequence[float],  # Sampling weights (not normalized)
        num_samples: int,          # Number of samples to draw
        replacement: bool = True,  # Must be True if num_samples > len(weights)
        generator = None
    ) -> None
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `weights` | Per-sample weights (1D, any positive values) |
| `num_samples` | Total samples to generate |
| `replacement` | Whether to sample with replacement |
| `generator` | Random generator |

### Iteration Behavior

Uses `torch.multinomial` for efficient weighted sampling:

```python
def __iter__(self) -> Iterator[int]:
    rand_tensor = torch.multinomial(
        self.weights, self.num_samples, self.replacement, generator=self.generator
    )
    yield from iter(rand_tensor.tolist())
```

### Use Cases

- **Class imbalance**: Oversample minority classes
- **Importance sampling**: Prioritize certain samples
- **Curriculum learning**: Weight samples by difficulty

### Example: Handling Class Imbalance

```python
from torch.utils.data import WeightedRandomSampler, DataLoader

# Dataset with imbalanced classes
labels = [0, 0, 0, 0, 0, 1, 1, 2]  # 5 class-0, 2 class-1, 1 class-2
class_counts = [5, 2, 1]

# Weight inversely proportional to class frequency
class_weights = [1.0 / count for count in class_counts]
sample_weights = [class_weights[label] for label in labels]
# sample_weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 1.0]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(labels),
    replacement=True
)

dataloader = DataLoader(dataset, sampler=sampler)
```

### Example: Sampling Specific Distribution

```python
# Sample indices 0-5 with custom probabilities
weights = [0.1, 0.9, 0.4, 0.7, 3.0, 0.6]
sampler = WeightedRandomSampler(weights, num_samples=5, replacement=True)
list(sampler)  # e.g., [4, 4, 1, 4, 5] (index 4 has highest weight)
```

---

## BatchSampler

Wraps another sampler to yield batches of indices instead of individual indices.

### Class Definition

```python
class BatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],  # Base sampler
        batch_size: int,
        drop_last: bool  # Drop incomplete final batch
    ) -> None
```

### Iteration Behavior

```python
def __iter__(self) -> Iterator[list[int]]:
    sampler_iter = iter(self.sampler)
    if self.drop_last:
        # Use zip trick for efficient batching
        args = [sampler_iter] * self.batch_size
        for batch_droplast in zip(*args, strict=False):
            yield list(batch_droplast)
    else:
        batch = list(itertools.islice(sampler_iter, self.batch_size))
        while batch:
            yield batch
            batch = list(itertools.islice(sampler_iter, self.batch_size))
```

### Length Calculation

```python
def __len__(self) -> int:
    if self.drop_last:
        return len(self.sampler) // self.batch_size
    else:
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

### Use Cases

- Custom batch composition
- Variable batch sizes
- Wrapping custom samplers

### Example

```python
from torch.utils.data import BatchSampler, SequentialSampler

base_sampler = SequentialSampler(range(10))

# With drop_last=False
batch_sampler = BatchSampler(base_sampler, batch_size=3, drop_last=False)
list(batch_sampler)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

# With drop_last=True
batch_sampler = BatchSampler(base_sampler, batch_size=3, drop_last=True)
list(batch_sampler)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

### DataLoader Integration

```python
# Using batch_sampler replaces batch_size, shuffle, sampler, drop_last
dataloader = DataLoader(
    dataset,
    batch_sampler=custom_batch_sampler
    # Cannot specify: batch_size, shuffle, sampler, drop_last
)
```

---

## DistributedSampler

Partitions dataset across multiple processes for distributed training.

### Class Definition

```python
class DistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,  # World size (auto-detected)
        rank: int | None = None,           # Process rank (auto-detected)
        shuffle: bool = True,
        seed: int = 0,                     # Base seed for shuffling
        drop_last: bool = False            # Drop tail for even division
    ) -> None
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset to partition |
| `num_replicas` | Total number of processes (default: `dist.get_world_size()`) |
| `rank` | Current process rank (default: `dist.get_rank()`) |
| `shuffle` | Shuffle indices each epoch |
| `seed` | Base random seed (must be same across all processes) |
| `drop_last` | Drop samples to ensure equal partition sizes |

### Partitioning Algorithm

```python
def __iter__(self) -> Iterator:
    if self.shuffle:
        # Deterministic shuffle based on epoch + seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
    else:
        indices = list(range(len(self.dataset)))

    if not self.drop_last:
        # Pad to make evenly divisible
        padding_size = self.total_size - len(indices)
        indices += indices[:padding_size]  # Wrap around
    else:
        # Truncate
        indices = indices[:self.total_size]

    # Subsample: rank 0 gets [0, R, 2R, ...], rank 1 gets [1, R+1, 2R+1, ...]
    indices = indices[self.rank : self.total_size : self.num_replicas]
    return iter(indices)
```

### Critical: set_epoch()

**Must call `set_epoch()` before each epoch** to ensure different shuffling:

```python
def set_epoch(self, epoch: int) -> None:
    """Ensure different ordering per epoch when shuffle=True."""
    self.epoch = epoch
```

### Use Case: Distributed Training

```python
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='nccl')

dataset = MyDataset(...)
sampler = DistributedSampler(dataset)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    # shuffle must be False when using sampler
    shuffle=False
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # CRITICAL: ensures proper shuffling
    for batch in dataloader:
        # Each process gets different samples
        train_step(batch)
```

### Partition Visualization

For 10 samples across 3 processes:

```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

With drop_last=False (pad to 12):
  Padded: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
  Rank 0: [0, 3, 6, 9]
  Rank 1: [1, 4, 7, 0]  <- Note: padding wraps
  Rank 2: [2, 5, 8, 1]

With drop_last=True (truncate to 9):
  Truncated: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  Rank 0: [0, 3, 6]
  Rank 1: [1, 4, 7]
  Rank 2: [2, 5, 8]
```

---

## Custom Sampler Patterns

### Length-based Batching

Group samples by length for efficient padding:

```python
class SortedBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Sort by length
        sorted_indices = torch.argsort(torch.tensor(self.lengths)).tolist()

        # Create batches of similar lengths
        batches = [
            sorted_indices[i:i+self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]

        # Shuffle batch order (not within batches)
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
```

### Stratified Sampler

Ensure each batch has balanced class representation:

```python
class StratifiedBatchSampler(Sampler[list[int]]):
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __iter__(self):
        # Interleave classes
        all_indices = []
        class_iters = {
            label: iter(random.sample(indices, len(indices)) if self.shuffle else indices)
            for label, indices in self.class_indices.items()
        }

        # Round-robin from each class
        while class_iters:
            for label in list(class_iters.keys()):
                try:
                    all_indices.append(next(class_iters[label]))
                except StopIteration:
                    del class_iters[label]

        # Batch the interleaved indices
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i+self.batch_size]
```

### Curriculum Sampler

Start with easy samples, gradually include harder ones:

```python
class CurriculumSampler(Sampler[int]):
    def __init__(self, difficulties, num_samples, epoch, max_epochs):
        self.difficulties = difficulties
        self.num_samples = num_samples
        self.epoch = epoch
        self.max_epochs = max_epochs

    def __iter__(self):
        # Progress from 0 to 1 over training
        progress = self.epoch / self.max_epochs

        # Threshold increases over time
        max_difficulty = progress  # Start easy, include harder samples

        # Filter by difficulty
        valid_indices = [
            i for i, d in enumerate(self.difficulties)
            if d <= max_difficulty
        ]

        # Sample from valid indices
        if len(valid_indices) < self.num_samples:
            # Oversample if needed
            indices = random.choices(valid_indices, k=self.num_samples)
        else:
            indices = random.sample(valid_indices, self.num_samples)

        yield from indices
```

---

## DataLoader Integration

### Sampler Selection in DataLoader

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,         # Creates RandomSampler if True, SequentialSampler if False
    sampler=None,         # Custom sampler overrides shuffle
    batch_sampler=None,   # Custom batch sampler overrides batch_size/shuffle/sampler/drop_last
    drop_last=False
)
```

**Mutual exclusivity rules:**

1. `shuffle=True` implies `sampler=RandomSampler(dataset)`
2. If `sampler` provided, `shuffle` must be False
3. If `batch_sampler` provided, cannot specify `batch_size`, `shuffle`, `sampler`, or `drop_last`

### Internal Sampler Creation

```python
# Simplified DataLoader logic
if batch_sampler is not None:
    batch_size = None
    drop_last = False
    sampler = None
elif sampler is None:
    if shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

if batch_sampler is None:
    batch_sampler = BatchSampler(sampler, batch_size, drop_last)
```

---

## MLX Mapping

### Implementation Approach

MLX doesn't have built-in samplers, but the patterns are straightforward to implement:

```python
# Basic sampler pattern for MLX
def sequential_sampler(dataset_size):
    """Sequential indices."""
    return list(range(dataset_size))

def random_sampler(dataset_size, replacement=False, num_samples=None, key=None):
    """Random indices using MLX random."""
    import mlx.core as mx

    num_samples = num_samples or dataset_size

    if replacement:
        return mx.random.randint(0, dataset_size, (num_samples,), key=key).tolist()
    else:
        # For permutation, use argsort of random values
        indices = mx.argsort(mx.random.uniform(shape=(dataset_size,), key=key))
        if num_samples < dataset_size:
            indices = indices[:num_samples]
        return indices.tolist()

def weighted_sampler(weights, num_samples, replacement=True, key=None):
    """Weighted sampling using categorical distribution."""
    import mlx.core as mx

    # Normalize weights to probabilities
    probs = mx.array(weights) / mx.sum(mx.array(weights))

    # Sample from categorical
    indices = mx.random.categorical(mx.log(probs), num_samples, key=key)
    return indices.tolist()
```

### Distributed Sampling for MLX

```python
def distributed_indices(dataset_size, num_replicas, rank, shuffle=True, seed=0, epoch=0):
    """Get indices for this rank in distributed setting."""
    import mlx.core as mx

    if shuffle:
        key = mx.random.key(seed + epoch)
        indices = mx.argsort(mx.random.uniform(shape=(dataset_size,), key=key))
        indices = indices.tolist()
    else:
        indices = list(range(dataset_size))

    # Pad to make evenly divisible
    total_size = ((dataset_size + num_replicas - 1) // num_replicas) * num_replicas
    while len(indices) < total_size:
        indices.extend(indices[:total_size - len(indices)])

    # Subsample for this rank
    return indices[rank::num_replicas]
```

### Batching Utility

```python
def batch_indices(indices, batch_size, drop_last=False):
    """Yield batches of indices."""
    for i in range(0, len(indices), batch_size):
        batch = indices[i:i+batch_size]
        if len(batch) == batch_size or not drop_last:
            yield batch
```

---

## Summary Table

| Sampler | Output Type | Use Case |
|---------|-------------|----------|
| `SequentialSampler` | `int` | Validation, deterministic iteration |
| `RandomSampler` | `int` | Training, shuffling |
| `SubsetRandomSampler` | `int` | Train/val splits, cross-validation |
| `WeightedRandomSampler` | `int` | Class imbalance, importance sampling |
| `BatchSampler` | `list[int]` | Custom batch composition |
| `DistributedSampler` | `int` | Multi-GPU/multi-node training |

---

## Best Practices

1. **Always use `set_epoch()`** with DistributedSampler for proper shuffling
2. **Use generators** for reproducible sampling
3. **Consider weighted sampling** for imbalanced datasets
4. **Use SubsetRandomSampler** for splits instead of creating new datasets
5. **Set `drop_last=True`** during training with batch normalization
6. **Match world size and rank** exactly in distributed settings
