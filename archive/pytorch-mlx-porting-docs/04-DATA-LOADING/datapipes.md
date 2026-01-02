# DataPipes

## Overview

DataPipes are composable data processing primitives that provide a functional approach to data loading. They extend the traditional `Dataset` API with chainable operations similar to functional programming.

**Two Types**:
- **IterDataPipe**: For iterable-style data (streaming, large datasets)
- **MapDataPipe**: For map-style data (random access)

**Source**: `torch/utils/data/datapipes/`

**Note**: The core DataPipes have moved to `torchdata` package, but basic implementations remain in PyTorch core.

---

## Architecture

### IterDataPipe

```python
class IterDataPipe(IterableDataset[_T_co], metaclass=_IterDataPipeMeta):
    """Base class for iterable-style DataPipes."""

    functions: dict[str, Callable] = {}  # Registered functional methods

    def __iter__(self) -> Iterator[_T_co]:
        return self

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        """Register a DataPipe as a chainable method."""
        cls.functions[function_name] = ...
```

**Source**: `torch/utils/data/datapipes/datapipe.py:55-125`

**Key Properties**:
- Single iterator constraint (only one active iterator at a time)
- Lazy evaluation (elements computed on-demand)
- Chainable via functional API
- Serializable with `dill`

### MapDataPipe

```python
class MapDataPipe(Dataset[_T_co], metaclass=_DataPipeMeta):
    """Base class for map-style DataPipes."""

    def __getitem__(self, index) -> _T_co:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError(...)
```

**Source**: `torch/utils/data/datapipes/datapipe.py:230-280`

---

## Built-in IterDataPipes

### Source DataPipes

#### IterableWrapper

Wrap any iterable as a DataPipe.

```python
from torch.utils.data.datapipes.iter import IterableWrapper

dp = IterableWrapper([1, 2, 3, 4, 5])
list(dp)  # [1, 2, 3, 4, 5]

# With deep copy
dp = IterableWrapper(range(10), deepcopy=True)
```

#### FileLister

List files matching a pattern.

```python
from torch.utils.data.datapipes.iter import FileLister

dp = FileLister(root="data/", masks="*.txt")
```

#### FileOpener

Open files for reading.

```python
from torch.utils.data.datapipes.iter import FileOpener

dp = FileLister("data/", "*.txt").open_files(mode="r")
```

### Transformation DataPipes

#### Mapper

Apply a function to each element.

```python
from torch.utils.data.datapipes.iter import Mapper

dp = IterableWrapper([1, 2, 3])
mapped = Mapper(dp, fn=lambda x: x * 2)
# Or functional form:
mapped = dp.map(lambda x: x * 2)
list(mapped)  # [2, 4, 6]
```

#### Filter

Filter elements by predicate.

```python
from torch.utils.data.datapipes.iter import Filter

dp = IterableWrapper(range(10))
filtered = Filter(dp, filter_fn=lambda x: x % 2 == 0)
# Or functional form:
filtered = dp.filter(lambda x: x % 2 == 0)
list(filtered)  # [0, 2, 4, 6, 8]
```

#### Collator

Apply collation function to batches.

```python
from torch.utils.data.datapipes.iter import Collator

dp = IterableWrapper([[1, 2], [3, 4]])
collated = Collator(dp, collate_fn=torch.tensor)
```

### Grouping DataPipes

#### Batcher

Group elements into batches.

```python
from torch.utils.data.datapipes.iter import Batcher

dp = IterableWrapper(range(10))
batched = Batcher(dp, batch_size=3)
# Or functional form:
batched = dp.batch(3)
list(batched)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
```

#### UnBatcher

Flatten batches back to individual elements.

```python
from torch.utils.data.datapipes.iter import UnBatcher

dp = IterableWrapper([[1, 2], [3, 4, 5]])
unbatched = UnBatcher(dp)
list(unbatched)  # [1, 2, 3, 4, 5]
```

#### Grouper

Group elements by key function.

```python
from torch.utils.data.datapipes.iter import Grouper

dp = IterableWrapper([("a", 1), ("b", 2), ("a", 3)])
grouped = Grouper(dp, group_key_fn=lambda x: x[0])
```

### Combinatorics DataPipes

#### Shuffler

Shuffle elements with a buffer.

```python
from torch.utils.data.datapipes.iter import Shuffler

dp = IterableWrapper(range(10))
shuffled = Shuffler(dp, buffer_size=5)
# Or functional form:
shuffled = dp.shuffle(buffer_size=5)
```

#### Sampler

Sample from a DataPipe using a sampler.

```python
from torch.utils.data.datapipes.iter import Sampler
from torch.utils.data import RandomSampler

dp = IterableWrapper(range(10))
sampled = Sampler(dp, sampler=RandomSampler(range(10)))
```

### Combining DataPipes

#### Zipper

Zip multiple DataPipes together.

```python
from torch.utils.data.datapipes.iter import Zipper

dp1 = IterableWrapper([1, 2, 3])
dp2 = IterableWrapper(["a", "b", "c"])
zipped = Zipper(dp1, dp2)
# Or functional form:
zipped = dp1.zip(dp2)
list(zipped)  # [(1, "a"), (2, "b"), (3, "c")]
```

#### Concater

Concatenate multiple DataPipes.

```python
from torch.utils.data.datapipes.iter import Concater

dp1 = IterableWrapper([1, 2, 3])
dp2 = IterableWrapper([4, 5, 6])
concatenated = Concater(dp1, dp2)
# Or functional form:
concatenated = dp1.concat(dp2)
list(concatenated)  # [1, 2, 3, 4, 5, 6]
```

#### Forker

Fork a DataPipe into multiple copies.

```python
from torch.utils.data.datapipes.iter import Forker

dp = IterableWrapper(range(5))
dp1, dp2 = Forker(dp, num_instances=2)
```

#### Demultiplexer

Split a DataPipe based on classifier.

```python
from torch.utils.data.datapipes.iter import Demultiplexer

dp = IterableWrapper(range(10))
even_dp, odd_dp = Demultiplexer(
    dp,
    num_instances=2,
    classifier_fn=lambda x: x % 2
)
```

#### Multiplexer

Interleave multiple DataPipes.

```python
from torch.utils.data.datapipes.iter import Multiplexer

dp1 = IterableWrapper([1, 2, 3])
dp2 = IterableWrapper(["a", "b", "c"])
muxed = Multiplexer(dp1, dp2)
list(muxed)  # [1, "a", 2, "b", 3, "c"]
```

### Sharding DataPipes

#### ShardingFilter

Filter for distributed data loading.

```python
from torch.utils.data.datapipes.iter import ShardingFilter

dp = IterableWrapper(range(10))
sharded = ShardingFilter(dp)

# In DataLoader worker
# Automatically filters based on worker_id
```

---

## Built-in MapDataPipes

### SequenceWrapper

Wrap a sequence as MapDataPipe.

```python
from torch.utils.data.datapipes.map import SequenceWrapper

dp = SequenceWrapper([1, 2, 3, 4, 5])
dp[2]  # 3
len(dp)  # 5
```

### Mapper (Map-style)

```python
from torch.utils.data.datapipes.map import Mapper

dp = SequenceWrapper([1, 2, 3])
mapped = Mapper(dp, fn=lambda x: x * 2)
mapped[1]  # 4
```

### Concater (Map-style)

```python
from torch.utils.data.datapipes.map import Concater

dp1 = SequenceWrapper([1, 2])
dp2 = SequenceWrapper([3, 4])
concatenated = Concater(dp1, dp2)
concatenated[2]  # 3 (from dp2)
```

### Zipper (Map-style)

```python
from torch.utils.data.datapipes.map import Zipper

dp1 = SequenceWrapper([1, 2, 3])
dp2 = SequenceWrapper(["a", "b", "c"])
zipped = Zipper(dp1, dp2)
zipped[1]  # (2, "b")
```

---

## Functional API

DataPipes support a functional/chainable API through registered methods:

```python
dp = IterableWrapper(range(100))

# Chain operations
result = (
    dp
    .filter(lambda x: x % 2 == 0)   # Keep even numbers
    .map(lambda x: x ** 2)           # Square each
    .batch(10)                       # Batch into groups of 10
    .shuffle(buffer_size=5)          # Shuffle batches
)
```

### Registering Custom DataPipes

```python
from torch.utils.data.datapipes.iter import IterDataPipe
from torch.utils.data.datapipes._decorator import functional_datapipe

@functional_datapipe("double")
class DoublerIterDataPipe(IterDataPipe):
    def __init__(self, source_dp):
        self.source_dp = source_dp

    def __iter__(self):
        for item in self.source_dp:
            yield item * 2

# Now available as functional method
dp = IterableWrapper([1, 2, 3])
doubled = dp.double()
list(doubled)  # [2, 4, 6]
```

---

## DataChunk

Container for batched data returned by grouping operations.

```python
class DataChunk(list[_T]):
    """Container for a batch of elements."""

    def __init__(self, items: Iterable[_T]):
        super().__init__(list(items))

    def raw_iterator(self) -> Iterator[_T]:
        yield from self.items
```

**Source**: `torch/utils/data/datapipes/datapipe.py:39-52`

---

## Integration with DataLoader

```python
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

# Create DataPipe
dp = (
    IterableWrapper(file_paths)
    .open_files()
    .map(parse_file)
    .batch(32)
    .shuffle()
)

# Use with DataLoader
loader = DataLoader(dp, num_workers=4)

for batch in loader:
    process(batch)
```

---

## Serialization

DataPipes support serialization for checkpointing and distributed training:

```python
import pickle

dp = IterableWrapper(range(10)).map(lambda x: x * 2)

# Serialize
serialized = pickle.dumps(dp)

# Deserialize
restored = pickle.loads(serialized)
```

For lambda functions, `dill` is required:
```python
# Install dill for lambda serialization
# pip install dill

import dill
serialized = dill.dumps(dp)
```

---

## Sharding for Multi-Process/Distributed

```python
from torch.utils.data.datapipes.iter import ShardingFilter

def build_datapipe(root_dir):
    dp = FileLister(root_dir, masks="*.txt")
    dp = dp.sharding_filter()  # Apply sharding
    dp = dp.map(process_file)
    dp = dp.batch(32)
    return dp

# DataLoader handles sharding across workers
loader = DataLoader(build_datapipe("data/"), num_workers=4)
```

---

## MLX Implementation

MLX doesn't have a DataPipe equivalent, but you can create similar patterns:

### Basic DataPipe Pattern

```python
import mlx.core as mx

class IterDataPipe:
    """Base class for MLX-compatible DataPipes."""
    def __iter__(self):
        raise NotImplementedError

    def map(self, fn):
        return MapperPipe(self, fn)

    def filter(self, fn):
        return FilterPipe(self, fn)

    def batch(self, batch_size):
        return BatcherPipe(self, batch_size)


class MapperPipe(IterDataPipe):
    def __init__(self, source, fn):
        self.source = source
        self.fn = fn

    def __iter__(self):
        for item in self.source:
            yield self.fn(item)


class FilterPipe(IterDataPipe):
    def __init__(self, source, fn):
        self.source = source
        self.fn = fn

    def __iter__(self):
        for item in self.source:
            if self.fn(item):
                yield item


class BatcherPipe(IterDataPipe):
    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for item in self.source:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class IterableWrapper(IterDataPipe):
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable
```

### Usage

```python
# Create pipeline
dp = (
    IterableWrapper(range(100))
    .filter(lambda x: x % 2 == 0)
    .map(lambda x: x ** 2)
    .batch(10)
)

for batch in dp:
    # Convert to MLX arrays
    batch_array = mx.array(batch)
    # Process...
```

### Shuffler Implementation

```python
import random

class ShufflerPipe(IterDataPipe):
    def __init__(self, source, buffer_size=100):
        self.source = source
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for item in self.source:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for i in range(len(buffer) // 2):
                    yield buffer.pop()

        # Flush remaining
        random.shuffle(buffer)
        yield from buffer
```

---

## Comparison: Dataset vs DataPipe

| Feature | Dataset | DataPipe |
|---------|---------|----------|
| Style | Class-based | Functional/composable |
| Chaining | Manual | Built-in |
| Lazy evaluation | No | Yes (Iter) |
| Streaming | No | Yes |
| Serialization | Manual | Built-in |
| Sharding | Manual | Built-in |

---

## Implementation Files

**Core**:
- `torch/utils/data/datapipes/datapipe.py` - Base classes
- `torch/utils/data/datapipes/_decorator.py` - Registration decorator
- `torch/utils/data/datapipes/_typing.py` - Type metaclasses

**Iter DataPipes**:
- `torch/utils/data/datapipes/iter/__init__.py` - Exports
- `torch/utils/data/datapipes/iter/callable.py` - Mapper, Collator
- `torch/utils/data/datapipes/iter/combinatorics.py` - Sampler, Shuffler
- `torch/utils/data/datapipes/iter/combining.py` - Zipper, Concater, Forker
- `torch/utils/data/datapipes/iter/grouping.py` - Batcher, Grouper
- `torch/utils/data/datapipes/iter/selecting.py` - Filter
- `torch/utils/data/datapipes/iter/sharding.py` - ShardingFilter

**Map DataPipes**:
- `torch/utils/data/datapipes/map/__init__.py` - Exports
- `torch/utils/data/datapipes/map/callable.py` - Mapper
- `torch/utils/data/datapipes/map/combining.py` - Zipper, Concater

---

## Summary

DataPipes provide:
1. **Composable data processing** with functional API
2. **Lazy evaluation** for memory efficiency
3. **Built-in sharding** for distributed training
4. **Serialization support** for checkpointing

For MLX:
- Implement simple pipe pattern for chaining
- Use Python generators for lazy evaluation
- Consider `mx.data.Dataset` for structured loading
- Manual sharding for distributed scenarios
