"""
Data Loading Module

Implements PyTorch-compatible data loading utilities for MLX.
"""

from .dataset import (
    Dataset, TensorDataset, IterableDataset,
    ConcatDataset, ChainDataset, Subset, StackDataset,
    random_split,
)
from .sampler import (
    Sampler, SequentialSampler, RandomSampler, BatchSampler,
    SubsetRandomSampler, WeightedRandomSampler,
)
from .dataloader import DataLoader, default_collate


# Worker info placeholder (not applicable to MLX's single-threaded execution)
_worker_info = None


def get_worker_info():
    """Get worker info for current DataLoader worker.

    Returns None in MLX since workers are not used (no multiprocessing).
    """
    return _worker_info


def default_convert(data):
    """Convert data to default format.

    Simply returns the data as-is for MLX compatibility.
    """
    return data


class IterDataPipe(IterableDataset):
    """Iterable-style DataPipe base class.

    DataPipes are a PyTorch 2.0+ feature for composable data loading.
    This is a basic implementation for compatibility.
    """
    pass


class MapDataPipe(Dataset):
    """Map-style DataPipe base class.

    DataPipes are a PyTorch 2.0+ feature for composable data loading.
    This is a basic implementation for compatibility.
    """
    pass


class DFIterDataPipe(IterDataPipe):
    """DataFrame IterDataPipe base class.

    DataPipe for iterating over DataFrame-like data structures.
    This is a compatibility stub for PyTorch's DataPipe infrastructure.
    """
    pass


class DataChunk:
    """Container for a chunk of data.

    Used by DataPipes to represent batches of data items.
    This is a compatibility class matching PyTorch's interface.
    """

    def __init__(self, items=None):
        if items is None:
            items = []
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def as_str(self, indent=''):
        return indent + str(self.items)

    def raw_iterator(self):
        return iter(self.items)


class _DatasetKind:
    """Enum for dataset types used by DataLoader.

    Internal class matching PyTorch's _DatasetKind.
    """
    Map = 0
    Iterable = 1


def functional_datapipe(*args, **kwargs):
    """Decorator to register a function as a DataPipe.

    This is a compatibility stub for PyTorch's functional datapipe registration.

    Args:
        *args: Positional arguments (typically name to register the datapipe under).
        **kwargs: Keyword arguments.

    Returns:
        Decorator function
    """
    name = args[0] if args else kwargs.get('name', None)
    def decorator(cls):
        # In PyTorch, this registers the class as a functional datapipe
        # For MLX compatibility, we just return the class unchanged
        if name is not None:
            cls._datapipe_name = name
        return cls
    return decorator


def argument_validation(*args, **kwargs):
    """Decorator for validating datapipe arguments.

    This is a compatibility stub for PyTorch's argument validation decorator.

    Args:
        *args: Positional arguments (typically function to wrap).
        **kwargs: Keyword arguments.

    Returns:
        Wrapped function (unchanged in MLX)
    """
    if args and callable(args[0]):
        return args[0]
    def decorator(fn):
        return fn
    return decorator


def runtime_validation(*args, **kwargs):
    """Decorator for runtime validation of datapipe operations.

    This is a compatibility stub for PyTorch's runtime validation decorator.

    Args:
        *args: Positional arguments (typically function to wrap).
        **kwargs: Keyword arguments.

    Returns:
        Wrapped function (unchanged in MLX)
    """
    if args and callable(args[0]):
        return args[0]
    def decorator(fn):
        return fn
    return decorator


class runtime_validation_disabled:
    """Context manager to disable runtime validation.

    This is a compatibility stub for PyTorch's runtime_validation_disabled.
    In MLX, runtime validation is not implemented, so this is a no-op.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class guaranteed_datapipes_determinism:
    """Context manager to guarantee datapipe determinism.

    This is a compatibility stub for PyTorch's guaranteed_datapipes_determinism.
    In MLX, datapipes follow standard Python iteration semantics.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class non_deterministic:
    """Decorator/context manager marking operations as non-deterministic.

    This is a compatibility stub for PyTorch's non_deterministic marker.
    Can be used as a decorator or context manager.
    """

    def __init__(self, *args, **kwargs):
        self.fn = args[0] if args and callable(args[0]) else None
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self.fn is not None:
            return self.fn(*args, **kwargs)
        # Used as decorator
        def wrapper(fn):
            return fn
        return wrapper(*args) if args else wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class DistributedSampler(Sampler):
    """Sampler for distributed training.

    In MLX, this is a no-op since distributed training is not supported.
    It simply returns indices for the full dataset.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False):
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Total samples per replica
        if self.drop_last and len(dataset) % self.num_replicas != 0:
            self.num_samples = len(dataset) // self.num_replicas
        else:
            self.num_samples = (len(dataset) + self.num_replicas - 1) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        from ._random import mlx_permutation, mlx_seeded_key

        dataset_len = len(self.dataset)

        if self.shuffle:
            # Use MLX random with seeded key for reproducibility
            key = mlx_seeded_key(self.seed, self.epoch)
            indices = mlx_permutation(dataset_len, key=key)
        else:
            indices = list(range(dataset_len))

        # Add extra samples for even distribution using efficient slicing
        if len(indices) < self.total_size:
            padding_size = self.total_size - len(indices)
            # Use modular indexing for padding (handles padding > dataset_len)
            if padding_size <= dataset_len:
                indices = indices + indices[:padding_size]
            else:
                # Rare case: need to repeat indices multiple times
                full_repeats = padding_size // dataset_len
                remainder = padding_size % dataset_len
                indices = indices * (full_repeats + 1) + indices[:remainder]

        # Subsample for this replica
        offset = self.rank * self.num_samples
        indices = indices[offset:offset + self.num_samples]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """Set epoch for shuffling."""
        self.epoch = epoch


__all__ = [
    # Datasets
    'Dataset',
    'TensorDataset',
    'IterableDataset',
    'ConcatDataset',
    'ChainDataset',
    'Subset',
    'StackDataset',
    'random_split',
    # Samplers
    'Sampler',
    'SequentialSampler',
    'RandomSampler',
    'BatchSampler',
    'SubsetRandomSampler',
    'WeightedRandomSampler',
    'DistributedSampler',
    # DataLoader
    'DataLoader',
    'default_collate',
    # Utility functions
    'get_worker_info',
    'default_convert',
    # DataPipes
    'IterDataPipe',
    'MapDataPipe',
    'DFIterDataPipe',
    'DataChunk',
    'functional_datapipe',
    # Validation and determinism
    'argument_validation',
    'runtime_validation',
    'runtime_validation_disabled',
    'guaranteed_datapipes_determinism',
    'non_deterministic',
    # Internal
    '_DatasetKind',
]
