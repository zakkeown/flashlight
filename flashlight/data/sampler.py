"""
Sampler Classes

Implements PyTorch-compatible sampling strategies for DataLoader.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional, Sized

from ._random import mlx_permutation, mlx_randint, mlx_shuffle_list, mlx_weighted_sample


class Sampler(ABC):
    """
    Base class for all Samplers.

    Every Sampler subclass must provide an __iter__ method that returns
    an iterator over indices of dataset elements, and a __len__ method
    that returns the length of the returned iterators.

    Args:
        data_source: Dataset to sample from (optional, for compatibility).
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        self.data_source = data_source

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        """Return an iterator over indices."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError


class SequentialSampler(Sampler):
    """
    Samples elements sequentially, always in the same order.

    Args:
        data_source: Dataset to sample from.

    Example:
        >>> sampler = SequentialSampler(range(10))
        >>> list(sampler)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, data_source: Sized) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        """Return sequential indices."""
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data_source)


class RandomSampler(Sampler):
    """
    Samples elements randomly.

    If without replacement, then samples from a shuffled dataset.
    If with replacement, then the user can specify num_samples to draw.

    Args:
        data_source: Dataset to sample from.
        replacement: If True, samples are drawn with replacement.
        num_samples: Number of samples to draw. If None, defaults to len(data_source).
        generator: Generator used for random permutation (optional).

    Example:
        >>> sampler = RandomSampler(range(10))
        >>> indices = list(sampler)
        >>> len(indices)
        10
        >>> set(indices) == set(range(10))  # All indices present
        True
    """

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[Any] = None,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not replacement and num_samples is not None:
            if num_samples > len(data_source):
                raise ValueError(
                    f"num_samples ({num_samples}) should be less than or equal to "
                    f"len(data_source) ({len(data_source)}) when replacement=False"
                )

    @property
    def num_samples(self) -> int:
        """Return the number of samples to draw."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        """Return randomly sampled indices."""
        n = len(self.data_source)

        if self.replacement:
            # Sample with replacement using MLX random
            indices = mlx_randint(0, n, self.num_samples)
            yield from indices
        else:
            # Sample without replacement (shuffle) using MLX random
            indices = mlx_permutation(n)
            if self._num_samples is not None:
                indices = indices[: self._num_samples]
            yield from indices

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """
    Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices: A sequence of indices.
        generator: Generator used for random permutation (optional).

    Example:
        >>> sampler = SubsetRandomSampler([0, 2, 4, 6, 8])
        >>> list(sampler)  # Some permutation of [0, 2, 4, 6, 8]
    """

    def __init__(self, indices: List[int], generator: Optional[Any] = None) -> None:
        super().__init__()
        self.indices = list(indices)
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        """Return randomly shuffled subset indices."""
        shuffled = mlx_shuffle_list(self.indices)
        return iter(shuffled)

    def __len__(self) -> int:
        """Return the number of indices."""
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    """
    Samples elements from a weighted distribution.

    Args:
        weights: A sequence of weights, not necessarily summing to 1.
        num_samples: Number of samples to draw.
        replacement: If True, samples are drawn with replacement.
        generator: Generator used in sampling (optional).

    Example:
        >>> weights = [0.1, 0.9]  # Second element much more likely
        >>> sampler = WeightedRandomSampler(weights, num_samples=10, replacement=True)
    """

    def __init__(
        self,
        weights: List[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if not replacement and num_samples > len(weights):
            raise ValueError(
                f"num_samples ({num_samples}) should be less than or equal to "
                f"len(weights) ({len(weights)}) when replacement=False"
            )

        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        """Return weighted random indices."""
        # Use MLX-based weighted sampling (with Gumbel-top-k for without replacement)
        indices = mlx_weighted_sample(self.weights, self.num_samples, replacement=self.replacement)
        yield from indices

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples


class BatchSampler(Sampler):
    """
    Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler: Base sampler to draw indices from.
        batch_size: Size of mini-batch.
        drop_last: If True, drop the last incomplete batch if the dataset
                   size is not divisible by batch_size.

    Example:
        >>> base_sampler = SequentialSampler(range(10))
        >>> batch_sampler = BatchSampler(base_sampler, batch_size=3, drop_last=False)
        >>> list(batch_sampler)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, got {batch_size}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices."""
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Handle the last incomplete batch
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


__all__ = [
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
    "BatchSampler",
]
