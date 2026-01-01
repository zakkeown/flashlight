"""
Dataset Classes

Implements PyTorch-compatible Dataset classes for MLX.
"""

from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Any, Generic, Iterator, List, Optional, Tuple, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(ABC, Generic[T_co]):
    """
    Abstract base class for all map-style datasets.

    All datasets that represent a map from keys to data samples should subclass
    this class. Subclasses must override __getitem__ and __len__.

    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...     def __getitem__(self, index):
        ...         return self.data[index]
        ...     def __len__(self):
        ...         return len(self.data)
    """

    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            The sample at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            Number of samples.
        """
        raise NotImplementedError

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        """Concatenate two datasets."""
        return ConcatDataset([self, other])


class IterableDataset(ABC, Generic[T_co]):
    """
    Abstract base class for all iterable-style datasets.

    All datasets that represent an iterable of data samples should subclass
    this class. Subclasses must override __iter__.

    This is particularly useful for streaming data or when the full dataset
    cannot fit in memory.

    Example:
        >>> class MyIterableDataset(IterableDataset):
        ...     def __init__(self, start, end):
        ...         self.start = start
        ...         self.end = end
        ...     def __iter__(self):
        ...         for i in range(self.start, self.end):
        ...             yield i
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """
        Return an iterator over the dataset.

        Returns:
            Iterator yielding samples.
        """
        raise NotImplementedError

    def __add__(self, other: "IterableDataset[T_co]") -> "ChainDataset[T_co]":
        """Chain two iterable datasets."""
        return ChainDataset([self, other])


class TensorDataset(Dataset[Tuple[Any, ...]]):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors: Tensors that have the same size of the first dimension.

    Example:
        >>> x = flashlight.randn(100, 10)
        >>> y = flashlight.randint(0, 2, (100,))
        >>> dataset = TensorDataset(x, y)
        >>> len(dataset)
        100
        >>> sample = dataset[0]
        >>> sample[0].shape  # x[0]
        (10,)
    """

    def __init__(self, *tensors: Any) -> None:
        """
        Initialize TensorDataset.

        Args:
            *tensors: Tensors to wrap. All tensors must have the same
                     first dimension size.

        Raises:
            ValueError: If tensors don't have the same first dimension.
        """
        if len(tensors) == 0:
            raise ValueError("At least one tensor is required")

        # Get the first dimension size from the first tensor
        first_dim = self._get_len(tensors[0])

        # Verify all tensors have the same first dimension
        for i, tensor in enumerate(tensors):
            tensor_len = self._get_len(tensor)
            if tensor_len != first_dim:
                raise ValueError(
                    f"Tensor at index {i} has size {tensor_len} in first dimension, "
                    f"but expected {first_dim}"
                )

        self.tensors = tensors

    def _get_len(self, tensor: Any) -> int:
        """Get the length (first dimension) of a tensor."""
        if hasattr(tensor, "shape"):
            return tensor.shape[0]
        elif hasattr(tensor, "__len__"):
            return len(tensor)
        else:
            raise TypeError(f"Cannot determine length of {type(tensor)}")

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample.

        Returns:
            Tuple of tensor values at the given index.
        """
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self._get_len(self.tensors[0])


class ConcatDataset(Dataset[T_co]):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets: List of datasets to concatenate.

    Example:
        >>> dataset1 = TensorDataset(flashlight.randn(10, 5))
        >>> dataset2 = TensorDataset(flashlight.randn(20, 5))
        >>> combined = ConcatDataset([dataset1, dataset2])
        >>> len(combined)
        30
    """

    def __init__(self, datasets: List[Dataset[T_co]]) -> None:
        """
        Initialize ConcatDataset.

        Args:
            datasets: List of datasets to concatenate.

        Raises:
            ValueError: If no datasets are provided.
        """
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")

        self.datasets = list(datasets)
        self.cumulative_sizes = self._cumsum([len(d) for d in self.datasets])

    def _cumsum(self, sequence: List[int]) -> List[int]:
        """Compute cumulative sum."""
        result = []
        total = 0
        for x in sequence:
            total += x
            result.append(total)
        return result

    def __len__(self) -> int:
        """Return total length of concatenated datasets."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> T_co:
        """
        Get a sample from the concatenated dataset.

        Args:
            index: Index of the sample.

        Returns:
            Sample at the given index.
        """
        if index < 0:
            if -index > len(self):
                raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
            index = len(self) + index

        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        # Find which dataset contains this index using O(log n) binary search
        dataset_idx = bisect_right(self.cumulative_sizes, index)

        # Calculate index within that dataset
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]


class ChainDataset(IterableDataset[T_co]):
    """
    Dataset for chaining multiple IterableDatasets.

    Args:
        datasets: List of iterable datasets to chain.

    Example:
        >>> class NumberDataset(IterableDataset):
        ...     def __init__(self, start, end):
        ...         self.start, self.end = start, end
        ...     def __iter__(self):
        ...         yield from range(self.start, self.end)
        >>> d1 = NumberDataset(0, 5)
        >>> d2 = NumberDataset(10, 15)
        >>> chained = ChainDataset([d1, d2])
        >>> list(chained)
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    """

    def __init__(self, datasets: List[IterableDataset[T_co]]) -> None:
        """
        Initialize ChainDataset.

        Args:
            datasets: List of iterable datasets to chain.
        """
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator[T_co]:
        """Iterate over all chained datasets."""
        for dataset in self.datasets:
            yield from dataset


class Subset(Dataset[T_co]):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset: The whole dataset.
        indices: Indices in the whole set selected for subset.

    Example:
        >>> dataset = TensorDataset(flashlight.arange(100))
        >>> subset = Subset(dataset, [0, 10, 20, 30])
        >>> len(subset)
        4
    """

    def __init__(self, dataset: Dataset[T_co], indices: List[int]) -> None:
        """
        Initialize Subset.

        Args:
            dataset: The full dataset.
            indices: List of indices to include in the subset.
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> T_co:
        """Get a sample from the subset."""
        if isinstance(index, list):
            return [self.dataset[self.indices[i]] for i in index]
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        """Return the size of the subset."""
        return len(self.indices)


class StackDataset(Dataset[Tuple[Any, ...]]):
    """
    Dataset that stacks multiple datasets.

    Each sample is a dictionary containing samples from each dataset,
    keyed by their position or provided keys.

    Args:
        datasets: Either a dict of datasets or positional dataset arguments.

    Example:
        >>> ds1 = TensorDataset(flashlight.randn(10, 5))
        >>> ds2 = TensorDataset(flashlight.randn(10, 3))
        >>> stacked = StackDataset(images=ds1, labels=ds2)
        >>> sample = stacked[0]
        >>> 'images' in sample and 'labels' in sample
        True
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize StackDataset.

        Can be called with positional datasets or keyword datasets.
        """
        if args and kwargs:
            raise ValueError("Cannot mix positional and keyword arguments")

        if kwargs:
            self.datasets = kwargs
            self._keys = list(kwargs.keys())
        elif args:
            self.datasets = {str(i): ds for i, ds in enumerate(args)}
            self._keys = [str(i) for i in range(len(args))]
        else:
            raise ValueError("At least one dataset is required")

        # Verify all datasets have the same length
        lengths = [len(ds) for ds in self.datasets.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All datasets must have the same length")

        self._length = lengths[0] if lengths else 0

    def __getitem__(self, index: int) -> dict:
        """Get a sample from all datasets."""
        return {key: self.datasets[key][index] for key in self._keys}

    def __len__(self) -> int:
        """Return the common length of all datasets."""
        return self._length


def random_split(*args, **kwargs) -> List[Subset]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset: Dataset to be split.
        lengths: Lengths of splits to be produced.
        generator: Generator used for the random permutation (optional).

    Returns:
        List of Subset objects.

    Example:
        >>> dataset = TensorDataset(flashlight.arange(100))
        >>> train, val = random_split(dataset, [80, 20])
        >>> len(train), len(val)
        (80, 20)
    """
    from ._random import mlx_permutation, mlx_seeded_key

    # Parse arguments flexibly to match PyTorch's signature
    if len(args) >= 1:
        dataset = args[0]
    else:
        dataset = kwargs.get("dataset")

    if len(args) >= 2:
        lengths = args[1]
    else:
        lengths = kwargs.get("lengths")

    generator = args[2] if len(args) >= 3 else kwargs.get("generator", None)

    if dataset is None or lengths is None:
        raise TypeError("random_split() missing required arguments: 'dataset' and 'lengths'")

    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Sum of input lengths ({sum(lengths)}) does not equal "
            f"the dataset length ({len(dataset)})"
        )

    # Generate random permutation using MLX
    n = len(dataset)
    if generator is not None:
        # If generator has a seed attribute, use it for reproducibility
        seed = getattr(generator, "seed", 0) if hasattr(generator, "seed") else 0
        key = mlx_seeded_key(seed)
        indices = mlx_permutation(n, key=key)
    else:
        indices = mlx_permutation(n)

    # Split indices according to lengths
    subsets = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, indices[offset : offset + length]))
        offset += length

    return subsets


__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "StackDataset",
    "random_split",
]
