# DataLoader and Dataset

## Overview

PyTorch's data loading infrastructure provides efficient, flexible mechanisms for feeding data to training loops. The two core abstractions are `Dataset` (data representation) and `DataLoader` (batch iteration).

**Reference Files:**
- `torch/utils/data/dataloader.py`
- `torch/utils/data/dataset.py`

## Architecture

```
Data Loading Pipeline
├── Dataset (data source abstraction)
│   ├── Map-style Dataset (indexed access)
│   └── Iterable-style Dataset (streaming access)
├── Sampler (index generation strategy)
│   ├── SequentialSampler, RandomSampler
│   └── BatchSampler, DistributedSampler
├── DataLoader (orchestrates loading)
│   ├── Single-process loading
│   └── Multi-process loading (workers)
└── Collate Function (batch assembly)
```

---

## Dataset

### Map-style Dataset

Implements `__getitem__` and optionally `__len__`. Supports random access by index.

```python
class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    # Optional but recommended
    def __len__(self) -> int:
        ...
```

### Example Implementation

```python
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

### Iterable-style Dataset

Implements `__iter__`. Suited for streaming data or when random access is expensive.

```python
class IterableDataset(Dataset[T_co], Iterable[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
```

### Example Implementation

```python
class StreamDataset(IterableDataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        for item in self.data_source.stream():
            yield self.process(item)
```

### Multi-worker Handling for IterableDataset

Each worker gets a **copy** of the dataset. Without coordination, all workers return duplicates.

```python
class ShardedIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process: return full range
            iter_start, iter_end = self.start, self.end
        else:
            # Multi-process: shard the range
            per_worker = (self.end - self.start) // worker_info.num_workers
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = iter_start + per_worker
            if worker_id == worker_info.num_workers - 1:
                iter_end = self.end  # Last worker gets remainder

        return iter(range(iter_start, iter_end))
```

---

## Built-in Dataset Classes

### TensorDataset

Wraps tensors; indexes along first dimension.

```python
class TensorDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, *tensors: Tensor) -> None:
        # All tensors must have same size in dimension 0
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
```

**Usage:**
```python
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
```

### ConcatDataset

Concatenates multiple datasets sequentially.

```python
dataset = ConcatDataset([dataset1, dataset2, dataset3])
# len(dataset) == len(dataset1) + len(dataset2) + len(dataset3)
```

### Subset

View into a dataset at specified indices.

```python
indices = list(range(0, 100))
subset = Subset(full_dataset, indices)
# len(subset) == 100
```

### ChainDataset

Chains iterable datasets (for streaming).

```python
dataset = ChainDataset([iterable_dataset1, iterable_dataset2])
```

### random_split

Splits dataset into non-overlapping subsets.

```python
train_set, val_set = random_split(
    dataset,
    [0.8, 0.2],  # 80% train, 20% val
    generator=torch.Generator().manual_seed(42)
)
```

---

## DataLoader

The main interface for iterating over datasets in batches.

### Constructor

```python
DataLoader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler: Sampler = None,
    batch_sampler: Sampler = None,
    num_workers: int = 0,
    collate_fn: Callable = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable = None,
    multiprocessing_context = None,
    generator: torch.Generator = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
    in_order: bool = True
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `batch_size` | Samples per batch (default: 1) |
| `shuffle` | Reshuffle at every epoch (default: False) |
| `num_workers` | Subprocess count for loading (0 = main process) |
| `collate_fn` | Merges samples into batches |
| `pin_memory` | Copy to CUDA pinned memory for faster transfer |
| `drop_last` | Drop incomplete final batch |
| `prefetch_factor` | Batches prefetched per worker (default: 2) |
| `persistent_workers` | Keep workers alive across epochs |

### Basic Usage

```python
# Simple training loop
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Iteration Flow

1. **Sampler** generates indices (or batches of indices)
2. **Fetcher** retrieves samples from dataset using indices
3. **Collate function** combines samples into a batch
4. **Pin memory** (optional) copies batch to pinned memory
5. **Batch** is returned to user

---

## Collate Functions

### Default Collate

Automatically stacks tensors, converts numpy arrays, handles nested structures.

```python
from torch.utils.data import default_collate

samples = [
    {'x': torch.tensor([1, 2]), 'y': 0},
    {'x': torch.tensor([3, 4]), 'y': 1},
]

batch = default_collate(samples)
# {'x': tensor([[1, 2], [3, 4]]), 'y': tensor([0, 1])}
```

### Custom Collate

```python
def custom_collate(batch):
    """Handle variable-length sequences."""
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences to max length
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    labels = torch.tensor(labels)

    return data, labels

dataloader = DataLoader(dataset, collate_fn=custom_collate)
```

---

## Multi-process Loading

### Worker Configuration

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,           # 4 worker processes
    prefetch_factor=2,       # Each worker prefetches 2 batches
    persistent_workers=True  # Keep workers alive
)
```

### Worker Info

Access worker context within dataset or worker_init_fn:

```python
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    # worker_info.id - worker ID (0 to num_workers-1)
    # worker_info.num_workers - total number of workers
    # worker_info.seed - random seed for this worker
    # worker_info.dataset - dataset copy in this worker

    # Example: seed numpy differently per worker
    np.random.seed(worker_info.seed % (2**32))
```

### Prefetching

With `num_workers=4` and `prefetch_factor=2`:
- Total prefetched batches: 4 × 2 = 8
- Workers load batches in parallel while main process trains

---

## Memory Pinning

Pinned (page-locked) memory enables faster CPU→GPU transfer.

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True  # Enable pinning
)

for data, target in dataloader:
    # Data is already in pinned memory
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
```

### Custom Pin Memory

For custom types, implement `pin_memory()` method:

```python
class CustomBatch:
    def __init__(self, data):
        self.data = data

    def pin_memory(self):
        self.data = self.data.pin_memory()
        return self
```

---

## Shuffling and Sampling

### Automatic Shuffling

```python
# shuffle=True uses RandomSampler internally
dataloader = DataLoader(dataset, shuffle=True)
```

### Custom Sampler

```python
from torch.utils.data import WeightedRandomSampler

# Sample inversely proportional to class frequency
weights = [1.0/count for count in class_counts]
sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)

dataloader = DataLoader(dataset, sampler=sampler)
# Note: shuffle must be False when using custom sampler
```

### Batch Sampler

Control batch composition directly:

```python
from torch.utils.data import BatchSampler, SequentialSampler

# Create batches that group similar samples
batch_sampler = BatchSampler(
    SequentialSampler(dataset),
    batch_size=32,
    drop_last=False
)

dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
# Note: batch_size, shuffle, sampler, drop_last must not be specified
```

---

## Common Patterns

### Training/Validation Split

```python
from torch.utils.data import random_split

full_dataset = MyDataset(...)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Cross-validation Folds

```python
from torch.utils.data import Subset

def k_fold_split(dataset, k=5):
    fold_size = len(dataset) // k
    indices = list(range(len(dataset)))

    for fold in range(k):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        yield (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices)
        )
```

### Infinite DataLoader

```python
def cycle(dataloader):
    """Infinitely cycle through dataloader."""
    while True:
        for batch in dataloader:
            yield batch

infinite_loader = cycle(train_loader)
batch = next(infinite_loader)
```

---

## MLX Mapping

### Key Differences

1. **No multi-process loading**: MLX typically uses single-process data loading
2. **Lazy evaluation**: MLX's lazy execution may affect batching patterns
3. **Unified memory**: No need for `pin_memory` on Apple Silicon

### Porting Approach

```python
# PyTorch
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# MLX equivalent (simpler)
def mlx_dataloader(dataset, batch_size, shuffle=False):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = [dataset[j] for j in batch_indices]
        yield default_collate(batch)  # Or custom collate
```

### Using mx.data (if available)

MLX may provide native data loading utilities. Check MLX documentation for current API.

---

## Best Practices

1. **Set num_workers appropriately**: Usually 4-8 workers, experiment for your system
2. **Use pin_memory=True** for GPU training
3. **Enable persistent_workers** for small datasets (reduces worker startup overhead)
4. **Drop last incomplete batch** during training if batch norm is used
5. **Use separate generators** for reproducibility:
   ```python
   g = torch.Generator()
   g.manual_seed(0)
   DataLoader(dataset, generator=g, shuffle=True)
   ```
6. **Handle IterableDataset carefully** with multiple workers to avoid duplicates
