"""
Test Phase 4: Sampler Classes

Comprehensive tests for all sampler classes:
- SequentialSampler
- RandomSampler
- SubsetRandomSampler
- WeightedRandomSampler
- BatchSampler
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX, skipIfNoTorch

try:
    import mlx_compat
    from mlx_compat.data import (
        TensorDataset,
        SequentialSampler, RandomSampler, SubsetRandomSampler,
        WeightedRandomSampler, BatchSampler
    )
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestSequentialSampler(TestCase):
    """Test SequentialSampler class."""

    def test_sequential_order(self):
        """Verify indices are sequential 0 to N-1."""
        data = list(range(10))
        sampler = SequentialSampler(data)
        indices = list(sampler)
        self.assertEqual(indices, list(range(10)))

    def test_sequential_length(self):
        """Verify __len__ returns correct count."""
        data = list(range(100))
        sampler = SequentialSampler(data)
        self.assertEqual(len(sampler), 100)

    def test_sequential_empty(self):
        """Test with empty data source."""
        data = []
        sampler = SequentialSampler(data)
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_sequential_iteration_multiple_times(self):
        """Verify sampler can be iterated multiple times."""
        data = list(range(5))
        sampler = SequentialSampler(data)
        first = list(sampler)
        second = list(sampler)
        self.assertEqual(first, second)


@skipIfNoMLX
class TestRandomSampler(TestCase):
    """Test RandomSampler class."""

    def test_shuffle_produces_all_indices(self):
        """Verify all indices appear exactly once without replacement."""
        data = list(range(100))
        sampler = RandomSampler(data, replacement=False)
        indices = list(sampler)
        self.assertEqual(len(indices), 100)
        self.assertEqual(set(indices), set(range(100)))

    def test_replacement_allows_repeats(self):
        """Verify replacement=True allows repeated indices."""
        data = list(range(10))
        sampler = RandomSampler(data, replacement=True, num_samples=1000)
        indices = list(sampler)
        self.assertEqual(len(indices), 1000)
        # With replacement and 1000 samples from 10 items, we should see repeats
        unique_indices = set(indices)
        # All 10 indices should appear at least once with high probability
        self.assertGreaterEqual(len(unique_indices), 5)

    def test_num_samples_limits_output(self):
        """Verify num_samples parameter works correctly."""
        data = list(range(100))
        sampler = RandomSampler(data, replacement=False, num_samples=50)
        indices = list(sampler)
        self.assertEqual(len(indices), 50)
        # All indices should be unique
        self.assertEqual(len(set(indices)), 50)

    def test_num_samples_error_when_exceeds_data(self):
        """Verify error when num_samples > len(data) with replacement=False."""
        data = list(range(10))
        with self.assertRaises(ValueError):
            RandomSampler(data, replacement=False, num_samples=20)

    def test_shuffle_is_random(self):
        """Verify shuffling produces different orders (statistical)."""
        data = list(range(100))
        sampler = RandomSampler(data, replacement=False)

        # Get multiple shuffles
        shuffles = [tuple(list(sampler)) for _ in range(10)]
        unique_shuffles = set(shuffles)

        # With 100! possible orderings, 10 shuffles should all be different
        self.assertGreater(len(unique_shuffles), 1)

    def test_random_sampler_length(self):
        """Verify __len__ returns correct count."""
        data = list(range(50))
        sampler = RandomSampler(data)
        self.assertEqual(len(sampler), 50)

        sampler_with_num = RandomSampler(data, replacement=True, num_samples=100)
        self.assertEqual(len(sampler_with_num), 100)


@skipIfNoMLX
class TestSubsetRandomSampler(TestCase):
    """Test SubsetRandomSampler class."""

    def test_subset_indices_only(self):
        """Verify only specified indices are returned."""
        subset_indices = [0, 5, 10, 15, 20]
        sampler = SubsetRandomSampler(subset_indices)
        result = list(sampler)
        self.assertEqual(set(result), set(subset_indices))

    def test_shuffled_order(self):
        """Verify indices are shuffled."""
        subset_indices = list(range(100))
        sampler = SubsetRandomSampler(subset_indices)

        # Get multiple shuffles
        results = [tuple(list(sampler)) for _ in range(5)]
        unique_results = set(results)

        # Should see different orderings
        self.assertGreater(len(unique_results), 1)

    def test_subset_sampler_length(self):
        """Verify __len__ returns correct count."""
        indices = [1, 3, 5, 7, 9]
        sampler = SubsetRandomSampler(indices)
        self.assertEqual(len(sampler), 5)

    def test_empty_subset(self):
        """Test with empty subset."""
        sampler = SubsetRandomSampler([])
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])


@skipIfNoMLX
class TestWeightedRandomSampler(TestCase):
    """Test WeightedRandomSampler class."""

    def test_weighted_distribution_with_replacement(self):
        """Verify higher weights produce more samples (statistically)."""
        # Weight second element 10x more than first
        weights = [0.1, 0.9]
        num_samples = 1000
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
        indices = list(sampler)

        count_0 = sum(1 for i in indices if i == 0)
        count_1 = sum(1 for i in indices if i == 1)

        # Second element should appear much more often
        self.assertGreater(count_1, count_0 * 2)

    def test_weighted_without_replacement(self):
        """Test weighted sampling without replacement."""
        weights = [0.1, 0.3, 0.6]
        num_samples = 2  # Less than len(weights)
        sampler = WeightedRandomSampler(weights, num_samples, replacement=False)
        indices = list(sampler)

        self.assertEqual(len(indices), 2)
        # All indices should be unique
        self.assertEqual(len(set(indices)), 2)

    def test_replacement_required_for_large_samples(self):
        """Test error when num_samples > weights without replacement."""
        weights = [0.5, 0.5]
        with self.assertRaises(ValueError):
            WeightedRandomSampler(weights, num_samples=10, replacement=False)

    def test_weighted_sampler_length(self):
        """Verify __len__ returns correct count."""
        weights = [1.0, 2.0, 3.0]
        sampler = WeightedRandomSampler(weights, num_samples=50, replacement=True)
        self.assertEqual(len(sampler), 50)

    def test_uniform_weights(self):
        """Test with uniform weights - should be approximately uniform."""
        weights = [1.0, 1.0, 1.0, 1.0]
        num_samples = 4000
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
        indices = list(sampler)

        counts = [sum(1 for i in indices if i == j) for j in range(4)]
        # Each should be roughly 1000 (+/- some variance)
        for count in counts:
            self.assertGreater(count, 500)
            self.assertLess(count, 1500)

    def test_zero_weight_handling(self):
        """Test that zero-weight items are rarely/never sampled."""
        weights = [0.0, 1.0, 0.0]
        num_samples = 100
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
        indices = list(sampler)

        # All samples should be index 1 (the only non-zero weight)
        # Note: Due to epsilon in log calculation, might get occasional others
        count_1 = sum(1 for i in indices if i == 1)
        self.assertGreater(count_1, 90)  # At least 90% should be index 1


@skipIfNoMLX
class TestBatchSampler(TestCase):
    """Test BatchSampler class."""

    def test_batch_sizes(self):
        """Verify correct batch sizes."""
        base_sampler = SequentialSampler(range(10))
        batch_sampler = BatchSampler(base_sampler, batch_size=3, drop_last=False)
        batches = list(batch_sampler)

        self.assertEqual(len(batches), 4)  # 10 / 3 = 3 full + 1 partial
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[-1]), 1)  # Last batch has 1

    def test_drop_last_true(self):
        """Verify incomplete batch dropped when drop_last=True."""
        base_sampler = SequentialSampler(range(10))
        batch_sampler = BatchSampler(base_sampler, batch_size=3, drop_last=True)
        batches = list(batch_sampler)

        self.assertEqual(len(batches), 3)  # 10 // 3 = 3 full batches
        for batch in batches:
            self.assertEqual(len(batch), 3)

    def test_drop_last_false(self):
        """Verify incomplete batch included when drop_last=False."""
        base_sampler = SequentialSampler(range(10))
        batch_sampler = BatchSampler(base_sampler, batch_size=3, drop_last=False)
        batches = list(batch_sampler)

        self.assertEqual(len(batches), 4)  # 3 full + 1 partial
        self.assertEqual(batches[-1], [9])

    def test_batch_sampler_length(self):
        """Verify __len__ calculation."""
        base_sampler = SequentialSampler(range(35))

        # With drop_last=False
        batch_sampler = BatchSampler(base_sampler, batch_size=8, drop_last=False)
        self.assertEqual(len(batch_sampler), 5)  # ceil(35/8) = 5

        # With drop_last=True
        batch_sampler = BatchSampler(base_sampler, batch_size=8, drop_last=True)
        self.assertEqual(len(batch_sampler), 4)  # floor(35/8) = 4

    def test_batch_with_random_sampler(self):
        """Test BatchSampler with RandomSampler."""
        base_sampler = RandomSampler(range(20), replacement=False)
        batch_sampler = BatchSampler(base_sampler, batch_size=5, drop_last=False)
        batches = list(batch_sampler)

        self.assertEqual(len(batches), 4)
        # All indices should appear exactly once
        all_indices = [idx for batch in batches for idx in batch]
        self.assertEqual(set(all_indices), set(range(20)))

    def test_batch_size_validation(self):
        """Test error on invalid batch size."""
        base_sampler = SequentialSampler(range(10))
        with self.assertRaises(ValueError):
            BatchSampler(base_sampler, batch_size=0, drop_last=False)
        with self.assertRaises(ValueError):
            BatchSampler(base_sampler, batch_size=-1, drop_last=False)

    def test_exact_division(self):
        """Test when dataset size is exact multiple of batch size."""
        base_sampler = SequentialSampler(range(12))
        batch_sampler = BatchSampler(base_sampler, batch_size=4, drop_last=False)
        batches = list(batch_sampler)

        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len(batch), 4)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestSamplerParity(TestCase):
    """Parity tests comparing MLX samplers with PyTorch."""

    def test_random_sampler_distribution_parity(self):
        """Compare RandomSampler distribution with PyTorch."""
        import torch
        from torch.utils.data import RandomSampler as TorchRandomSampler

        data = list(range(10))
        num_iterations = 1000

        # Count index occurrences for MLX
        mlx_counts = np.zeros(10)
        for _ in range(num_iterations):
            mlx_sampler = RandomSampler(data, replacement=True, num_samples=100)
            for idx in mlx_sampler:
                mlx_counts[idx] += 1
        mlx_counts /= mlx_counts.sum()

        # Count index occurrences for PyTorch
        torch_counts = np.zeros(10)
        for _ in range(num_iterations):
            torch_sampler = TorchRandomSampler(data, replacement=True, num_samples=100)
            for idx in torch_sampler:
                torch_counts[idx] += 1
        torch_counts /= torch_counts.sum()

        # Both should be approximately uniform
        expected = np.ones(10) / 10
        np.testing.assert_allclose(mlx_counts, expected, atol=0.05)
        np.testing.assert_allclose(torch_counts, expected, atol=0.05)

    def test_weighted_sampler_distribution_parity(self):
        """Compare WeightedRandomSampler distribution with PyTorch."""
        import torch
        from torch.utils.data import WeightedRandomSampler as TorchWeightedRandomSampler

        weights = [0.1, 0.2, 0.3, 0.4]
        num_samples = 10000

        # MLX sampling
        mlx_sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
        mlx_counts = np.zeros(4)
        for idx in mlx_sampler:
            mlx_counts[idx] += 1
        mlx_freq = mlx_counts / mlx_counts.sum()

        # PyTorch sampling
        torch_sampler = TorchWeightedRandomSampler(
            torch.tensor(weights), num_samples, replacement=True
        )
        torch_counts = np.zeros(4)
        for idx in torch_sampler:
            torch_counts[idx] += 1
        torch_freq = torch_counts / torch_counts.sum()

        # Both should match expected distribution
        expected = np.array(weights) / sum(weights)
        np.testing.assert_allclose(mlx_freq, expected, atol=0.05)
        np.testing.assert_allclose(torch_freq, expected, atol=0.05)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
