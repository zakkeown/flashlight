"""
Test Phase 4: Additional Dataset Classes

Tests for ChainDataset, StackDataset, and random_split.
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX, skipIfNoTorch

try:
    import flashlight
    from flashlight.data import (
        Dataset, IterableDataset, TensorDataset,
        ConcatDataset, ChainDataset, Subset, StackDataset,
        random_split
    )
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


class SimpleIterableDataset(IterableDataset):
    """Simple iterable dataset for testing."""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield i


@skipIfNoMLX
class TestChainDataset(TestCase):
    """Test ChainDataset class."""

    def test_chain_iteration(self):
        """Test iterating through chained datasets."""
        d1 = SimpleIterableDataset(0, 5)
        d2 = SimpleIterableDataset(10, 15)
        chained = ChainDataset([d1, d2])

        result = list(chained)
        expected = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        self.assertEqual(result, expected)

    def test_chain_single_dataset(self):
        """Test chaining a single dataset."""
        d1 = SimpleIterableDataset(5, 10)
        chained = ChainDataset([d1])

        result = list(chained)
        self.assertEqual(result, [5, 6, 7, 8, 9])

    def test_chain_empty_dataset(self):
        """Test chaining with empty dataset."""
        d1 = SimpleIterableDataset(0, 0)  # Empty
        d2 = SimpleIterableDataset(1, 3)
        chained = ChainDataset([d1, d2])

        result = list(chained)
        self.assertEqual(result, [1, 2])

    def test_chain_add_operator(self):
        """Test using + operator to chain IterableDatasets."""
        d1 = SimpleIterableDataset(0, 3)
        d2 = SimpleIterableDataset(3, 6)
        chained = d1 + d2

        self.assertIsInstance(chained, ChainDataset)
        result = list(chained)
        self.assertEqual(result, [0, 1, 2, 3, 4, 5])

    def test_chain_multiple_iterations(self):
        """Test that chained dataset can be iterated multiple times."""
        d1 = SimpleIterableDataset(0, 3)
        d2 = SimpleIterableDataset(3, 5)
        chained = ChainDataset([d1, d2])

        first = list(chained)
        second = list(chained)
        self.assertEqual(first, second)


@skipIfNoMLX
class TestStackDataset(TestCase):
    """Test StackDataset class."""

    def test_stacked_sample_format(self):
        """Verify samples are dicts with correct keys."""
        x = flashlight.randn(10, 5)
        y = flashlight.randint(0, 2, (10,))
        ds_x = TensorDataset(x)
        ds_y = TensorDataset(y)

        stacked = StackDataset(images=ds_x, labels=ds_y)
        sample = stacked[0]

        self.assertIsInstance(sample, dict)
        self.assertIn('images', sample)
        self.assertIn('labels', sample)

    def test_keyword_keys(self):
        """Verify keyword argument names become keys."""
        x = flashlight.tensor([[1.0], [2.0], [3.0]])
        y = flashlight.tensor([0, 1, 0])
        ds_x = TensorDataset(x)
        ds_y = TensorDataset(y)

        stacked = StackDataset(features=ds_x, targets=ds_y)
        sample = stacked[0]

        self.assertEqual(set(sample.keys()), {'features', 'targets'})

    def test_positional_keys(self):
        """Verify positional args get string index keys."""
        x = flashlight.tensor([[1.0], [2.0]])
        y = flashlight.tensor([0, 1])
        ds_x = TensorDataset(x)
        ds_y = TensorDataset(y)

        stacked = StackDataset(ds_x, ds_y)
        sample = stacked[0]

        self.assertEqual(set(sample.keys()), {'0', '1'})

    def test_length_mismatch_error(self):
        """Verify error on mismatched lengths."""
        x = flashlight.randn(10, 5)
        y = flashlight.randn(5, 3)  # Different length
        ds_x = TensorDataset(x)
        ds_y = TensorDataset(y)

        with self.assertRaises(ValueError):
            StackDataset(a=ds_x, b=ds_y)

    def test_stack_length(self):
        """Test __len__ of stacked dataset."""
        x = flashlight.randn(20, 5)
        y = flashlight.randn(20, 3)
        ds_x = TensorDataset(x)
        ds_y = TensorDataset(y)

        stacked = StackDataset(x=ds_x, y=ds_y)
        self.assertEqual(len(stacked), 20)

    def test_cannot_mix_positional_and_keyword(self):
        """Test error when mixing positional and keyword args."""
        x = flashlight.randn(10, 5)
        ds = TensorDataset(x)

        with self.assertRaises(ValueError):
            StackDataset(ds, extra=ds)

    def test_empty_error(self):
        """Test error when no datasets provided."""
        with self.assertRaises(ValueError):
            StackDataset()


@skipIfNoMLX
class TestRandomSplit(TestCase):
    """Test random_split function."""

    def test_split_sizes(self):
        """Verify splits have correct lengths."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        train, val, test = random_split(dataset, [70, 20, 10])

        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(test), 10)

    def test_no_overlap(self):
        """Verify split subsets don't share indices."""
        x = flashlight.arange(100).reshape(100, 1)
        dataset = TensorDataset(x)

        train, val = random_split(dataset, [80, 20])

        # Get all values from each split
        train_vals = set(float(train[i][0].item()) for i in range(len(train)))
        val_vals = set(float(val[i][0].item()) for i in range(len(val)))

        # No overlap
        self.assertEqual(len(train_vals & val_vals), 0)

    def test_complete_coverage(self):
        """Verify all indices are in some split."""
        x = flashlight.arange(50).reshape(50, 1)
        dataset = TensorDataset(x)

        train, val = random_split(dataset, [30, 20])

        # Get all values from both splits
        train_vals = set(float(train[i][0].item()) for i in range(len(train)))
        val_vals = set(float(val[i][0].item()) for i in range(len(val)))

        all_vals = train_vals | val_vals
        self.assertEqual(all_vals, set(range(50)))

    def test_wrong_sum_error(self):
        """Verify error when lengths don't sum to dataset size."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        with self.assertRaises(ValueError):
            random_split(dataset, [60, 30])  # Sum is 90, not 100

    def test_randomness(self):
        """Verify splits are random (different each time)."""
        x = flashlight.arange(100).reshape(100, 1)
        dataset = TensorDataset(x)

        splits = []
        for _ in range(5):
            train, val = random_split(dataset, [80, 20])
            train_vals = tuple(sorted(float(train[i][0].item()) for i in range(len(train))))
            splits.append(train_vals)

        unique_splits = set(splits)
        # Should see different splits
        self.assertGreater(len(unique_splits), 1)

    def test_single_split(self):
        """Test splitting into a single subset (entire dataset)."""
        x = flashlight.randn(50, 10)
        dataset = TensorDataset(x)

        [full] = random_split(dataset, [50])
        self.assertEqual(len(full), 50)

    def test_keyword_arguments(self):
        """Test calling with keyword arguments."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        train, val = random_split(dataset=dataset, lengths=[80, 20])

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestDatasetParity(TestCase):
    """Parity tests comparing MLX datasets with PyTorch."""

    def test_random_split_coverage_parity(self):
        """Compare random_split behavior with PyTorch."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data import random_split as torch_random_split

        # MLX version
        x_mlx = flashlight.arange(100).reshape(100, 1)
        dataset_mlx = TensorDataset(x_mlx)
        train_mlx, val_mlx = random_split(dataset_mlx, [80, 20])

        # PyTorch version
        x_torch = torch.arange(100).reshape(100, 1).float()
        dataset_torch = TorchTensorDataset(x_torch)
        train_torch, val_torch = torch_random_split(dataset_torch, [80, 20])

        # Both should have same sizes
        self.assertEqual(len(train_mlx), len(train_torch))
        self.assertEqual(len(val_mlx), len(val_torch))

        # Both should have complete coverage (no overlap, all elements present)
        mlx_train_vals = set(float(train_mlx[i][0].item()) for i in range(len(train_mlx)))
        mlx_val_vals = set(float(val_mlx[i][0].item()) for i in range(len(val_mlx)))

        torch_train_vals = set(float(train_torch[i][0].item()) for i in range(len(train_torch)))
        torch_val_vals = set(float(val_torch[i][0].item()) for i in range(len(val_torch)))

        # No overlap in either
        self.assertEqual(len(mlx_train_vals & mlx_val_vals), 0)
        self.assertEqual(len(torch_train_vals & torch_val_vals), 0)

        # Complete coverage in both
        self.assertEqual(mlx_train_vals | mlx_val_vals, set(range(100)))
        self.assertEqual(torch_train_vals | torch_val_vals, set(range(100)))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
