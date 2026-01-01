"""
Test Phase 4: Edge Cases and Error Handling

Tests for boundary conditions, error cases, and unusual scenarios.
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    from flashlight.data import (
        Dataset, IterableDataset,
        TensorDataset, ConcatDataset, Subset, StackDataset,
        DataLoader, default_collate,
        SequentialSampler, RandomSampler, SubsetRandomSampler,
        WeightedRandomSampler, BatchSampler,
        random_split
    )
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestEmptyDatasets(TestCase):
    """Test behavior with empty datasets."""

    def test_empty_tensor_dataset_error(self):
        """Test that TensorDataset requires at least one tensor."""
        with self.assertRaises(ValueError):
            TensorDataset()

    def test_empty_concat_dataset_error(self):
        """Test that ConcatDataset requires at least one dataset."""
        with self.assertRaises(ValueError):
            ConcatDataset([])

    def test_empty_subset(self):
        """Test Subset with empty indices."""
        x = flashlight.randn(10, 5)
        dataset = TensorDataset(x)
        subset = Subset(dataset, [])

        self.assertEqual(len(subset), 0)

    def test_dataloader_with_empty_subset(self):
        """Test DataLoader with empty dataset."""
        x = flashlight.randn(10, 5)
        dataset = TensorDataset(x)
        subset = Subset(dataset, [])

        loader = DataLoader(subset, batch_size=4)
        batches = list(loader)
        self.assertEqual(len(batches), 0)


@skipIfNoMLX
class TestSingleSampleDataset(TestCase):
    """Test behavior with single-sample datasets."""

    def test_single_sample_tensor_dataset(self):
        """Test TensorDataset with single sample."""
        x = flashlight.randn(1, 10)
        dataset = TensorDataset(x)

        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample[0].shape, (10,))

    def test_single_sample_dataloader(self):
        """Test DataLoader with single sample."""
        x = flashlight.randn(1, 10)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=1)

        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape, (1, 10))

    def test_single_sample_shuffle(self):
        """Test shuffling with single sample."""
        x = flashlight.tensor([[1.0, 2.0, 3.0]])
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        batches = list(loader)
        self.assertEqual(len(batches), 1)

    def test_single_sample_random_split(self):
        """Test random_split with single sample."""
        x = flashlight.randn(1, 5)
        dataset = TensorDataset(x)

        [train] = random_split(dataset, [1])
        self.assertEqual(len(train), 1)


@skipIfNoMLX
class TestBatchSizeLargerThanDataset(TestCase):
    """Test when batch_size > len(dataset)."""

    def test_batch_larger_no_drop_last(self):
        """Test batch_size larger than dataset with drop_last=False."""
        x = flashlight.randn(5, 10)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=10, drop_last=False)

        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], 5)

    def test_batch_larger_drop_last(self):
        """Test batch_size larger than dataset with drop_last=True."""
        x = flashlight.randn(5, 10)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=10, drop_last=True)

        batches = list(loader)
        self.assertEqual(len(batches), 0)


@skipIfNoMLX
class TestNegativeIndexing(TestCase):
    """Test negative indexing behavior."""

    def test_concat_dataset_negative_index(self):
        """Test ConcatDataset with negative indices."""
        x1 = flashlight.tensor([[1.0], [2.0], [3.0]])
        x2 = flashlight.tensor([[4.0], [5.0]])
        ds1 = TensorDataset(x1)
        ds2 = TensorDataset(x2)
        concat = ConcatDataset([ds1, ds2])

        # Test negative indexing
        self.assertEqual(float(concat[-1][0].item()), 5.0)
        self.assertEqual(float(concat[-2][0].item()), 4.0)
        self.assertEqual(float(concat[-5][0].item()), 1.0)

    def test_subset_negative_index(self):
        """Test Subset with negative indices."""
        x = flashlight.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        dataset = TensorDataset(x)
        subset = Subset(dataset, [0, 2, 4])  # Indices: 0->1.0, 2->3.0, 4->5.0

        # Test negative indexing
        self.assertEqual(float(subset[-1][0].item()), 5.0)  # Last element
        self.assertEqual(float(subset[-2][0].item()), 3.0)  # Second to last
        self.assertEqual(float(subset[-3][0].item()), 1.0)  # First element via negative

    def test_subset_negative_index_out_of_range(self):
        """Test Subset raises IndexError for out-of-range negative indices."""
        x = flashlight.randn(10, 3)
        dataset = TensorDataset(x)
        subset = Subset(dataset, [0, 1, 2])

        with self.assertRaises(IndexError):
            _ = subset[-4]  # Only 3 elements, -4 is out of range

        with self.assertRaises(IndexError):
            _ = subset[-100]

    def test_concat_dataset_out_of_range(self):
        """Test ConcatDataset with out-of-range indices."""
        x = flashlight.randn(5, 3)
        ds = TensorDataset(x)
        concat = ConcatDataset([ds])

        with self.assertRaises(IndexError):
            _ = concat[10]

        with self.assertRaises(IndexError):
            _ = concat[-10]


@skipIfNoMLX
class TestCollateEdgeCases(TestCase):
    """Test collate function edge cases."""

    def test_collate_empty_batch(self):
        """Test collating empty batch."""
        result = default_collate([])
        self.assertEqual(result, [])

    def test_collate_strings(self):
        """Test that strings are kept as list."""
        samples = ["hello", "world", "test"]
        result = default_collate(samples)
        self.assertEqual(result, samples)

    def test_collate_bytes(self):
        """Test that bytes are kept as list."""
        samples = [b"hello", b"world", b"test"]
        result = default_collate(samples)
        self.assertEqual(result, samples)

    def test_collate_mixed_shapes_error(self):
        """Test that mixed shapes cause error."""
        samples = [
            flashlight.tensor([1.0, 2.0]),
            flashlight.tensor([3.0, 4.0, 5.0]),  # Different shape
        ]
        # This should raise an error during stacking
        with self.assertRaises(Exception):
            default_collate(samples)

    def test_collate_namedtuple(self):
        """Test collating namedtuples preserves type."""
        from collections import namedtuple

        Sample = namedtuple('Sample', ['x', 'y'])
        samples = [
            Sample(x=flashlight.tensor([1.0]), y=flashlight.tensor([0])),
            Sample(x=flashlight.tensor([2.0]), y=flashlight.tensor([1])),
        ]
        result = default_collate(samples)

        self.assertIsInstance(result, Sample)
        self.assertEqual(result.x.shape, (2, 1))
        self.assertEqual(result.y.shape, (2, 1))


@skipIfNoMLX
class TestSamplerEdgeCases(TestCase):
    """Test sampler edge cases."""

    def test_random_sampler_zero_samples(self):
        """Test RandomSampler with num_samples specified."""
        data = list(range(10))
        # Test with explicit num_samples less than data size
        sampler = RandomSampler(data, replacement=True, num_samples=5)
        indices = list(sampler)
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(0 <= i < 10 for i in indices))

    def test_random_sampler_more_samples_with_replacement(self):
        """Test RandomSampler with num_samples > len(data) requires replacement."""
        data = list(range(5))
        sampler = RandomSampler(data, replacement=True, num_samples=20)
        indices = list(sampler)
        self.assertEqual(len(indices), 20)
        self.assertTrue(all(0 <= i < 5 for i in indices))

    def test_weighted_sampler_all_zero_weights(self):
        """Test WeightedRandomSampler with all zero weights."""
        # With all zero weights, should use uniform sampling
        weights = [0.0, 0.0, 0.0, 0.0]
        sampler = WeightedRandomSampler(weights, 100, replacement=True)
        indices = list(sampler)

        # Should still produce 100 samples
        self.assertEqual(len(indices), 100)
        # All indices should be valid
        self.assertTrue(all(0 <= i < 4 for i in indices))

    def test_batch_sampler_batch_size_equals_dataset(self):
        """Test BatchSampler when batch_size equals dataset size."""
        base = SequentialSampler(range(10))
        batch_sampler = BatchSampler(base, batch_size=10, drop_last=False)

        batches = list(batch_sampler)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 10)


@skipIfNoMLX
class TestDataLoaderEdgeCases(TestCase):
    """Test DataLoader edge cases."""

    def test_dataloader_conflicting_params_error(self):
        """Test DataLoader raises error on conflicting parameters."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        # batch_sampler with batch_size != 1
        base = SequentialSampler(dataset)
        batch_sampler = BatchSampler(base, batch_size=4, drop_last=False)

        with self.assertRaises(ValueError):
            DataLoader(dataset, batch_size=8, batch_sampler=batch_sampler)

    def test_dataloader_sampler_with_shuffle_error(self):
        """Test DataLoader raises error when both sampler and shuffle provided."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)
        sampler = SequentialSampler(dataset)

        with self.assertRaises(ValueError):
            DataLoader(dataset, sampler=sampler, shuffle=True)

    def test_dataloader_invalid_batch_size(self):
        """Test DataLoader with invalid batch_size."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        with self.assertRaises(ValueError):
            DataLoader(dataset, batch_size=0)

        with self.assertRaises(ValueError):
            DataLoader(dataset, batch_size=-1)

    def test_dataloader_multiple_iterations(self):
        """Test DataLoader can be iterated multiple times."""
        x = flashlight.arange(100).reshape(100, 1)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)

        batches1 = [b[0].numpy().flatten().tolist() for b in loader]
        batches2 = [b[0].numpy().flatten().tolist() for b in loader]

        # Without shuffle, should be the same
        self.assertEqual(batches1, batches2)


@skipIfNoMLX
class TestRandomSplitEdgeCases(TestCase):
    """Test random_split edge cases."""

    def test_split_zero_length_error(self):
        """Test that zero-length splits are allowed."""
        x = flashlight.randn(10, 5)
        dataset = TensorDataset(x)

        # Split with a zero-length subset
        train, empty, val = random_split(dataset, [8, 0, 2])

        self.assertEqual(len(train), 8)
        self.assertEqual(len(empty), 0)
        self.assertEqual(len(val), 2)

    def test_split_missing_args_error(self):
        """Test random_split raises error on missing arguments."""
        with self.assertRaises(TypeError):
            random_split()

        x = flashlight.randn(10, 5)
        dataset = TensorDataset(x)

        with self.assertRaises(TypeError):
            random_split(dataset)  # Missing lengths


@skipIfNoMLX
class TestMLXRandomUtilities(TestCase):
    """Test the MLX random utilities module."""

    def test_mlx_permutation_basic(self):
        """Test basic permutation generation."""
        from flashlight.data._random import mlx_permutation

        perm = mlx_permutation(10)
        self.assertEqual(len(perm), 10)
        self.assertEqual(set(perm), set(range(10)))

    def test_mlx_permutation_empty(self):
        """Test permutation with n=0."""
        from flashlight.data._random import mlx_permutation

        perm = mlx_permutation(0)
        self.assertEqual(perm, [])

    def test_mlx_shuffle_list(self):
        """Test list shuffling."""
        from flashlight.data._random import mlx_shuffle_list

        original = [1, 2, 3, 4, 5]
        shuffled = mlx_shuffle_list(original)

        self.assertEqual(set(shuffled), set(original))
        self.assertEqual(len(shuffled), len(original))

    def test_mlx_randint(self):
        """Test random integer generation."""
        from flashlight.data._random import mlx_randint

        values = mlx_randint(0, 10, 100)
        self.assertEqual(len(values), 100)
        self.assertTrue(all(0 <= v < 10 for v in values))

    def test_mlx_weighted_sample_with_replacement(self):
        """Test weighted sampling with replacement."""
        from flashlight.data._random import mlx_weighted_sample

        weights = [0.1, 0.9]
        samples = mlx_weighted_sample(weights, 1000, replacement=True)

        self.assertEqual(len(samples), 1000)
        # All samples should be valid indices
        self.assertTrue(all(s in [0, 1] for s in samples))

    def test_mlx_weighted_sample_without_replacement(self):
        """Test weighted sampling without replacement."""
        from flashlight.data._random import mlx_weighted_sample

        weights = [0.5, 0.3, 0.2]
        samples = mlx_weighted_sample(weights, 2, replacement=False)

        self.assertEqual(len(samples), 2)
        self.assertEqual(len(set(samples)), 2)  # No duplicates

    def test_mlx_seeded_key(self):
        """Test seeded key generation."""
        from flashlight.data._random import mlx_seeded_key, mlx_permutation

        key1 = mlx_seeded_key(42, epoch=0)
        key2 = mlx_seeded_key(42, epoch=0)

        perm1 = mlx_permutation(100, key=key1)
        perm2 = mlx_permutation(100, key=key2)

        # Same seed + epoch should produce same permutation
        self.assertEqual(perm1, perm2)

        # Different epoch should produce different permutation
        key3 = mlx_seeded_key(42, epoch=1)
        perm3 = mlx_permutation(100, key=key3)
        self.assertNotEqual(perm1, perm3)


@skipIfNoMLX
class TestStubFunctions(TestCase):
    """Test stub functions provided for PyTorch compatibility."""

    def test_get_worker_info_returns_none(self):
        """Test get_worker_info returns None (single-threaded MLX)."""
        from flashlight.data import get_worker_info

        result = get_worker_info()
        self.assertIsNone(result)

    def test_default_convert_identity(self):
        """Test default_convert returns input unchanged."""
        from flashlight.data import default_convert

        # Test with various types
        test_inputs = [
            42,
            "hello",
            [1, 2, 3],
            {"a": 1, "b": 2},
            flashlight.tensor([1.0, 2.0]),
        ]

        for inp in test_inputs:
            result = default_convert(inp)
            if isinstance(inp, flashlight.Tensor):
                # Tensor comparison
                self.assertTrue((result == inp).all())
            else:
                self.assertEqual(result, inp)


@skipIfNoMLX
class TestDataChunkContainer(TestCase):
    """Test DataChunk container class."""

    def test_datachunk_basic(self):
        """Test DataChunk creation and iteration."""
        from flashlight.data import DataChunk

        items = [1, 2, 3, 4, 5]
        chunk = DataChunk(items)

        self.assertEqual(len(chunk), 5)
        self.assertEqual(list(chunk), items)

    def test_datachunk_getitem(self):
        """Test DataChunk indexing."""
        from flashlight.data import DataChunk

        items = ["a", "b", "c", "d"]
        chunk = DataChunk(items)

        self.assertEqual(chunk[0], "a")
        self.assertEqual(chunk[2], "c")
        self.assertEqual(chunk[-1], "d")

    def test_datachunk_as_str(self):
        """Test DataChunk string representation."""
        from flashlight.data import DataChunk

        items = [1, 2, 3]
        chunk = DataChunk(items)

        result = chunk.as_str()
        self.assertIsInstance(result, str)

    def test_datachunk_raw_iterator(self):
        """Test DataChunk raw iterator."""
        from flashlight.data import DataChunk

        items = [10, 20, 30]
        chunk = DataChunk(items)

        raw_items = list(chunk.raw_iterator())
        self.assertEqual(raw_items, items)

    def test_datachunk_empty(self):
        """Test DataChunk with empty list."""
        from flashlight.data import DataChunk

        chunk = DataChunk([])
        self.assertEqual(len(chunk), 0)
        self.assertEqual(list(chunk), [])


@skipIfNoMLX
class TestSequentialSamplerParity(TestCase):
    """Test SequentialSampler explicit parity with PyTorch."""

    def test_sequential_sampler_order(self):
        """Test SequentialSampler produces ordered indices."""
        data = list(range(100))
        sampler = SequentialSampler(data)

        indices = list(sampler)
        expected = list(range(100))

        self.assertEqual(indices, expected)

    def test_sequential_sampler_length(self):
        """Test SequentialSampler __len__."""
        data = list(range(50))
        sampler = SequentialSampler(data)

        self.assertEqual(len(sampler), 50)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
