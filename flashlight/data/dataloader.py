"""
DataLoader

Implements PyTorch-compatible DataLoader for MLX.
"""

from typing import Any, Callable, Iterator, List, Optional, Sequence, TypeVar, Union

# Module-level imports for performance (avoid repeated imports per batch)
import flashlight as _flashlight

from ..tensor import Tensor as _MLXTensor
from .dataset import Dataset, IterableDataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler

try:
    import mlx.core as _mx

    _MLX_AVAILABLE = True
except ImportError:
    _mx = None
    _MLX_AVAILABLE = False

try:
    import numpy as _np

    _NUMPY_AVAILABLE = True
except ImportError:
    _np = None
    _NUMPY_AVAILABLE = False

T = TypeVar("T")

# Type dispatch cache for faster collation
_COLLATE_TYPE_HANDLERS = {}


def _get_type_handler(elem_type):
    """Get cached type handler for collation."""
    return _COLLATE_TYPE_HANDLERS.get(elem_type)


def _register_type_handler(elem_type, handler):
    """Register a type handler for collation."""
    _COLLATE_TYPE_HANDLERS[elem_type] = handler


def default_collate(batch: List[Any]) -> Any:
    """
    Default collate function that handles batching of samples.

    Takes a list of samples and collates them into a batch. Supports:
    - Tensors: stacked along a new first dimension
    - Tuples/Lists: recursively collated element-wise
    - Dicts: recursively collated by key
    - Numbers: converted to tensors
    - Strings: kept as list

    Args:
        batch: List of samples to collate.

    Returns:
        Collated batch.

    Example:
        >>> import flashlight
        >>> samples = [flashlight.tensor([1, 2]), flashlight.tensor([3, 4])]
        >>> batch = default_collate(samples)
        >>> batch.shape
        (2, 2)
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]
    elem_type = type(elem)

    # Fast path: check cached type handlers first
    handler = _get_type_handler(elem_type)
    if handler is not None:
        return handler(batch, elem, elem_type)

    # Handle flashlight Tensor - use module-level import
    if isinstance(elem, _MLXTensor):
        _register_type_handler(elem_type, lambda b, e, t: _flashlight.stack(b, dim=0))
        return _flashlight.stack(batch, dim=0)

    # Handle MLX arrays directly - use module-level import
    if _MLX_AVAILABLE and isinstance(elem, _mx.array):

        def _handle_mx_array(b, e, t):
            tensors = [_flashlight.tensor(x) for x in b]
            return _flashlight.stack(tensors, dim=0)

        _register_type_handler(elem_type, _handle_mx_array)
        return _handle_mx_array(batch, elem, elem_type)

    # Handle numpy arrays - use module-level import
    if _NUMPY_AVAILABLE and isinstance(elem, _np.ndarray):

        def _handle_numpy(b, e, t):
            # Batch convert: stack numpy arrays first, then convert once
            stacked = _np.stack(b, axis=0)
            return _flashlight.tensor(stacked)

        _register_type_handler(elem_type, _handle_numpy)
        return _handle_numpy(batch, elem, elem_type)

    # Handle tuples - collate each element position separately
    if isinstance(elem, tuple):

        def _handle_tuple(b, e, t):
            transposed = list(zip(*b))
            # Check if it's a namedtuple
            if hasattr(e, "_fields"):
                return t(*(default_collate(list(samples)) for samples in transposed))
            return tuple(default_collate(list(samples)) for samples in transposed)

        _register_type_handler(elem_type, _handle_tuple)
        return _handle_tuple(batch, elem, elem_type)

    # Handle lists - collate each element position separately
    if isinstance(elem, list):

        def _handle_list(b, e, t):
            transposed = list(zip(*b))
            return [default_collate(list(samples)) for samples in transposed]

        _register_type_handler(elem_type, _handle_list)
        return _handle_list(batch, elem, elem_type)

    # Handle dicts
    if isinstance(elem, dict):
        # Don't cache dict handler since keys may vary
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    # Handle numbers (int, float)
    if isinstance(elem, (int, float)):
        _register_type_handler(elem_type, lambda b, e, t: _flashlight.tensor(b))
        return _flashlight.tensor(batch)

    # Handle strings - keep as list
    if isinstance(elem, str):
        _register_type_handler(str, lambda b, e, t: b)
        return batch

    # Handle bytes
    if isinstance(elem, bytes):
        _register_type_handler(bytes, lambda b, e, t: b)
        return batch

    # Default: try to convert to tensor
    try:
        result = _flashlight.tensor(batch)
        _register_type_handler(elem_type, lambda b, e, t: _flashlight.tensor(b))
        return result
    except Exception:
        # If all else fails, return as-is
        return batch


class DataLoader:
    """
    Data loader that combines a dataset and a sampler, providing an iterable
    over the dataset with batching, shuffling, and collation.

    Args:
        dataset: Dataset from which to load data.
        batch_size: How many samples per batch to load (default: 1).
        shuffle: Set to True to have data reshuffled at every epoch (default: False).
        sampler: Defines the strategy to draw samples from the dataset. Can be
                 any Sampler with __len__ and __iter__. Mutually exclusive with
                 shuffle.
        batch_sampler: Like sampler, but returns a batch of indices at a time.
                       Mutually exclusive with batch_size, shuffle, sampler,
                       and drop_last.
        collate_fn: Merges a list of samples to form a mini-batch of Tensor(s).
                    Default: default_collate.
        drop_last: Set to True to drop the last incomplete batch if the dataset
                   size is not divisible by batch_size (default: False).

    Example:
        >>> import flashlight
        >>> from flashlight.data import TensorDataset, DataLoader
        >>> x = flashlight.randn(100, 10)
        >>> y = flashlight.randint(0, 2, (100,))
        >>> dataset = TensorDataset(x, y)
        >>> loader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch_x, batch_y in loader:
        ...     print(batch_x.shape, batch_y.shape)
        ...     break
        (16, 10) (16,)
    """

    def __init__(
        self,
        dataset: Union[Dataset[T], IterableDataset[T]],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[T]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context: Optional[Any] = None,
        generator: Optional[Any] = None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        """
        Initialize DataLoader.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size (default: 1).
            shuffle: Whether to shuffle (default: False).
            sampler: Custom sampler (optional).
            batch_sampler: Custom batch sampler (optional).
            num_workers: Number of workers for loading (ignored in MLX, default: 0).
            collate_fn: Collation function (default: default_collate).
            pin_memory: If True, copies tensors to pinned memory (ignored in MLX).
            drop_last: Whether to drop last incomplete batch (default: False).
            timeout: Timeout for collecting a batch (ignored in MLX).
            worker_init_fn: Function called on each worker subprocess (ignored in MLX).
            multiprocessing_context: Multiprocessing context (ignored in MLX).
            generator: Generator for random sampling (optional).
            prefetch_factor: Number of batches to prefetch per worker (ignored in MLX).
            persistent_workers: Keep workers alive between epochs (ignored in MLX).
            pin_memory_device: Device to pin memory to (ignored in MLX).
            in_order: Whether to return batches in order (default: True, ignored in MLX).

        Note:
            MLX uses unified memory, so num_workers, pin_memory, and related
            multiprocessing options are accepted for compatibility but ignored.
        """
        # Store compatibility params (they're ignored but available for inspection)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device
        self.in_order = in_order
        self.dataset = dataset
        self._is_iterable = isinstance(dataset, IterableDataset)

        # Validate arguments
        if batch_sampler is not None:
            # batch_sampler is mutually exclusive with batch_size, shuffle, sampler, drop_last
            if batch_size != 1:
                raise ValueError("batch_size must be 1 when batch_sampler is provided")
            if shuffle:
                raise ValueError("shuffle must be False when batch_sampler is provided")
            if sampler is not None:
                raise ValueError("sampler must be None when batch_sampler is provided")
            if drop_last:
                raise ValueError("drop_last must be False when batch_sampler is provided")
            self.batch_sampler = batch_sampler
            self.batch_size = None
        else:
            if sampler is not None and shuffle:
                raise ValueError("sampler and shuffle are mutually exclusive")

            if batch_size is None or batch_size <= 0:
                raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

            self.batch_size = batch_size

            # Create sampler if not provided
            if sampler is None:
                if self._is_iterable:
                    # For iterable datasets, we don't use a sampler
                    sampler = None
                elif shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

            # Create batch sampler
            if sampler is not None:
                self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
            else:
                self.batch_sampler = None

        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Any]:
        """
        Return an iterator over batches.

        Yields:
            Batches of data, collated according to collate_fn.
        """
        if self._is_iterable:
            # Handle iterable datasets
            yield from self._iter_iterable()
        else:
            # Handle map-style datasets
            yield from self._iter_map_style()

    def _iter_map_style(self) -> Iterator[Any]:
        """Iterate over map-style dataset."""
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def _iter_iterable(self) -> Iterator[Any]:
        """Iterate over iterable dataset."""
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # Handle the last incomplete batch
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        """
        Return the number of batches.

        Returns:
            Number of batches.

        Raises:
            TypeError: If the dataset is an IterableDataset (length unknown).
        """
        if self._is_iterable:
            raise TypeError(
                "IterableDataset does not have a length. "
                "Use a map-style Dataset if you need __len__."
            )
        return len(self.batch_sampler)


class _DataLoaderIter:
    """
    Iterator for DataLoader.

    This is a simple single-process iterator. Multi-process loading
    is not yet implemented.
    """

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._index = 0

        if loader.batch_sampler is not None:
            self._sampler_iter = iter(loader.batch_sampler)
        else:
            self._sampler_iter = None

    def __next__(self) -> Any:
        if self._sampler_iter is not None:
            try:
                batch_indices = next(self._sampler_iter)
            except StopIteration:
                raise StopIteration

            batch = [self.loader.dataset[idx] for idx in batch_indices]
            return self.loader.collate_fn(batch)
        else:
            raise StopIteration

    def __iter__(self) -> "_DataLoaderIter":
        return self


__all__ = [
    "DataLoader",
    "default_collate",
]
