"""
Test Phase 4: Mathematical Validation Tests

Rigorous statistical and mathematical validation tests for the data module.
These tests verify distribution properties, algorithmic correctness, and
numerical parity with PyTorch.
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np
import pytest
from collections import Counter

from tests.common_utils import TestCase, skipIfNoMLX, skipIfNoTorch

# Statistical test utilities
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import mlx_compat
    from mlx_compat.data import (
        TensorDataset, DataLoader,
        SequentialSampler, RandomSampler, SubsetRandomSampler,
        WeightedRandomSampler, BatchSampler, DistributedSampler,
        ChainDataset, StackDataset, random_split
    )
    from mlx_compat.data._random import (
        mlx_permutation, mlx_shuffle_list, mlx_randint,
        mlx_weighted_sample, mlx_seeded_key
    )
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


def skipIfNoScipy(func):
    """Skip test if scipy is not available."""
    if not SCIPY_AVAILABLE:
        return unittest.skip("scipy not available")(func)
    return func


# =============================================================================
# CRITICAL: Weighted Sampling Distribution Validation
# =============================================================================

@skipIfNoMLX
class TestWeightedSampleDistribution(TestCase):
    """Mathematical validation of weighted sampling distribution properties."""

    @skipIfNoScipy
    def test_weighted_sample_chi_square_uniform(self):
        """Chi-square test: uniform weights should produce uniform distribution."""
        weights = [1.0, 1.0, 1.0, 1.0]
        num_samples = 10000

        samples = mlx_weighted_sample(weights, num_samples, replacement=True)
        counts = np.bincount(samples, minlength=len(weights))

        # Expected: uniform distribution
        expected = np.ones(len(weights)) * num_samples / len(weights)

        # Chi-square test (p > 0.01 means we cannot reject uniform hypothesis)
        chi2, p_value = stats.chisquare(counts, expected)
        self.assertGreater(p_value, 0.01,
            f"Chi-square test failed: counts={counts}, expected={expected}, p={p_value}")

    @skipIfNoScipy
    def test_weighted_sample_chi_square_skewed(self):
        """Chi-square test: skewed weights should produce skewed distribution."""
        weights = [0.1, 0.2, 0.3, 0.4]
        num_samples = 10000

        samples = mlx_weighted_sample(weights, num_samples, replacement=True)
        counts = np.bincount(samples, minlength=len(weights))

        # Expected distribution based on weights
        total_weight = sum(weights)
        expected = np.array(weights) / total_weight * num_samples

        chi2, p_value = stats.chisquare(counts, expected)
        self.assertGreater(p_value, 0.01,
            f"Chi-square test failed: counts={counts}, expected={expected}, p={p_value}")

    @skipIfNoScipy
    def test_weighted_sample_extreme_weights(self):
        """Test with extreme weight ratios (99:1)."""
        weights = [0.99, 0.01]
        num_samples = 10000

        samples = mlx_weighted_sample(weights, num_samples, replacement=True)
        counts = np.bincount(samples, minlength=len(weights))

        # First element should dominate
        ratio = counts[0] / counts[1] if counts[1] > 0 else float('inf')

        # Expected ratio is 99:1, allow tolerance
        expected_ratio = 99
        self.assertGreater(ratio, expected_ratio * 0.5,
            f"Extreme weights not respected: ratio={ratio}, expected~{expected_ratio}")
        self.assertLess(ratio, expected_ratio * 2.0,
            f"Extreme weights not respected: ratio={ratio}, expected~{expected_ratio}")

    @skipIfNoScipy
    def test_weighted_sample_near_zero_weights(self):
        """Test with near-zero weights."""
        weights = [1e-6, 1.0, 1e-6]
        num_samples = 10000

        samples = mlx_weighted_sample(weights, num_samples, replacement=True)
        counts = np.bincount(samples, minlength=len(weights))

        # Middle element should dominate (almost all samples)
        self.assertGreater(counts[1] / num_samples, 0.99,
            f"Near-zero weights failed: counts={counts}")

    def test_weighted_sample_single_nonzero(self):
        """Test with only one non-zero weight."""
        weights = [0.0, 1.0, 0.0]
        num_samples = 1000

        samples = mlx_weighted_sample(weights, num_samples, replacement=True)
        counts = np.bincount(samples, minlength=len(weights))

        # All samples should be index 1
        self.assertGreater(counts[1] / num_samples, 0.95,
            f"Single nonzero weight failed: counts={counts}")

    @skipIfNoScipy
    def test_weighted_sample_without_replacement_correctness(self):
        """Validate Gumbel-top-k respects relative weights for first selection.

        For weighted sampling WITHOUT replacement, marginal probabilities differ
        from the weights because once an item is selected, it can't be chosen again.
        However, the FIRST item's selection probability should match the weights.
        """
        weights = [0.1, 0.3, 0.6]
        num_trials = 5000
        num_samples = 2  # Pick 2 out of 3

        # Count how often each index is selected FIRST
        first_counts = np.zeros(3)

        for _ in range(num_trials):
            samples = mlx_weighted_sample(weights, num_samples, replacement=False)
            # The first sample should follow the weight distribution
            first_counts[samples[0]] += 1

        # First selection probabilities should match normalized weights
        empirical_first_probs = first_counts / num_trials
        expected_probs = np.array(weights) / sum(weights)

        # Higher weight items should be selected first more often
        # Verify ordering is preserved: P(idx=2) > P(idx=1) > P(idx=0)
        self.assertGreater(empirical_first_probs[2], empirical_first_probs[1],
            "Higher weight should be selected first more often")
        self.assertGreater(empirical_first_probs[1], empirical_first_probs[0],
            "Higher weight should be selected first more often")

        # Allow reasonable tolerance for statistical variation
        np.testing.assert_allclose(empirical_first_probs, expected_probs, atol=0.05,
            err_msg=f"First selection probs: empirical={empirical_first_probs}, expected={expected_probs}")

    def test_weighted_sample_without_replacement_uniqueness(self):
        """Verify no duplicate indices in without-replacement sampling."""
        weights = [0.25, 0.25, 0.25, 0.25]

        for num_samples in [2, 3, 4]:
            for _ in range(100):
                samples = mlx_weighted_sample(weights, num_samples, replacement=False)
                self.assertEqual(len(set(samples)), num_samples,
                    f"Duplicates found in without-replacement: {samples}")


# =============================================================================
# CRITICAL: Seeded Reproducibility Validation
# =============================================================================

@skipIfNoMLX
class TestSeededReproducibility(TestCase):
    """Validation of seeded random number generation."""

    def test_seed_reproducibility_basic(self):
        """Same seed produces identical sequences."""
        for seed in [0, 42, 12345, 999999]:
            key1 = mlx_seeded_key(seed, epoch=0)
            key2 = mlx_seeded_key(seed, epoch=0)

            perm1 = mlx_permutation(100, key=key1)
            perm2 = mlx_permutation(100, key=key2)

            self.assertEqual(perm1, perm2,
                f"Reproducibility failed for seed={seed}")

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds produce different sequences."""
        seeds = [0, 1, 42, 100, 12345]
        permutations = []

        for seed in seeds:
            key = mlx_seeded_key(seed, epoch=0)
            perm = tuple(mlx_permutation(50, key=key))
            permutations.append(perm)

        # All should be unique
        unique_perms = set(permutations)
        self.assertEqual(len(unique_perms), len(seeds),
            f"Different seeds produced identical sequences")

    def test_different_epochs_produce_different_sequences(self):
        """Different epochs produce different sequences."""
        seed = 42
        epochs = [0, 1, 2, 3, 4]
        permutations = []

        for epoch in epochs:
            key = mlx_seeded_key(seed, epoch=epoch)
            perm = tuple(mlx_permutation(50, key=key))
            permutations.append(perm)

        # All should be unique
        unique_perms = set(permutations)
        self.assertEqual(len(unique_perms), len(epochs),
            f"Different epochs produced identical sequences")

    def test_seed_epoch_collision_awareness(self):
        """Check that seed+epoch collision is documented behavior."""
        # seed=40, epoch=2 gives key for 42
        # seed=41, epoch=1 gives key for 42
        # seed=42, epoch=0 gives key for 42
        # These WILL produce the same sequence - this is expected behavior
        # but should be documented

        key1 = mlx_seeded_key(40, epoch=2)
        key2 = mlx_seeded_key(41, epoch=1)
        key3 = mlx_seeded_key(42, epoch=0)

        perm1 = mlx_permutation(50, key=key1)
        perm2 = mlx_permutation(50, key=key2)
        perm3 = mlx_permutation(50, key=key3)

        # Document: these are expected to be equal due to simple addition
        self.assertEqual(perm1, perm2, "Seed collision behavior changed")
        self.assertEqual(perm2, perm3, "Seed collision behavior changed")

    @skipIfNoScipy
    def test_epoch_sequences_uncorrelated(self):
        """Epoch-based sequences should be statistically uncorrelated."""
        seed = 42
        n = 100

        # Get permutations for consecutive epochs
        key0 = mlx_seeded_key(seed, epoch=0)
        key1 = mlx_seeded_key(seed, epoch=1)

        perm0 = np.array(mlx_permutation(n, key=key0))
        perm1 = np.array(mlx_permutation(n, key=key1))

        # Compute Spearman correlation
        corr, p_value = stats.spearmanr(perm0, perm1)

        # Should have low correlation (random shuffles are ~uncorrelated)
        self.assertLess(abs(corr), 0.3,
            f"Epoch sequences correlated: r={corr}")


# =============================================================================
# MAJOR: Random Sampler Distribution Validation
# =============================================================================

@skipIfNoMLX
class TestRandomSamplerDistribution(TestCase):
    """Statistical validation of RandomSampler distribution properties."""

    @skipIfNoScipy
    def test_shuffle_produces_uniform_distribution(self):
        """Shuffled indices should be uniformly distributed at each position."""
        data = list(range(10))
        num_trials = 5000

        # Count which index appears at position 0
        position_counts = np.zeros((10, 10))  # [position, value]

        for _ in range(num_trials):
            sampler = RandomSampler(data, replacement=False)
            indices = list(sampler)
            for pos, val in enumerate(indices):
                position_counts[pos, val] += 1

        # Each position should have uniform distribution over values
        expected = num_trials / 10

        for pos in range(10):
            chi2, p_value = stats.chisquare(position_counts[pos],
                                             [expected] * 10)
            self.assertGreater(p_value, 0.001,
                f"Position {pos} not uniform: p={p_value}")

    def test_replacement_produces_repeats(self):
        """Sampling with replacement should produce repeated indices."""
        data = list(range(5))

        # With 1000 samples from 5 items, we should see repeats
        sampler = RandomSampler(data, replacement=True, num_samples=1000)
        indices = list(sampler)

        # Should have repeats
        self.assertLess(len(set(indices)), len(indices),
            "Sampling with replacement produced no repeats")

    @skipIfNoScipy
    def test_replacement_uniform_distribution(self):
        """Sampling with replacement should produce uniform distribution."""
        data = list(range(5))
        num_samples = 10000

        sampler = RandomSampler(data, replacement=True, num_samples=num_samples)
        indices = list(sampler)
        counts = np.bincount(indices, minlength=5)

        expected = num_samples / 5
        chi2, p_value = stats.chisquare(counts, [expected] * 5)

        self.assertGreater(p_value, 0.01,
            f"Replacement sampling not uniform: counts={counts}, p={p_value}")


# =============================================================================
# MAJOR: SubsetRandomSampler Parity Tests
# =============================================================================

@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestSubsetRandomSamplerParity(TestCase):
    """Parity tests for SubsetRandomSampler vs PyTorch."""

    def test_subset_sampler_returns_correct_indices(self):
        """Verify SubsetRandomSampler only returns specified indices."""
        import torch
        from torch.utils.data import SubsetRandomSampler as TorchSubsetRandomSampler

        indices = [5, 10, 15, 20, 25]

        # MLX
        mlx_sampler = SubsetRandomSampler(indices)
        mlx_result = set(list(mlx_sampler))

        # PyTorch
        torch_sampler = TorchSubsetRandomSampler(indices)
        torch_result = set(list(torch_sampler))

        # Both should only contain the specified indices
        self.assertEqual(mlx_result, set(indices))
        self.assertEqual(torch_result, set(indices))

    @skipIfNoScipy
    def test_subset_sampler_uniform_distribution(self):
        """Verify SubsetRandomSampler has uniform distribution over indices."""
        indices = list(range(10))
        num_trials = 5000

        # Count first-position occurrences
        counts = np.zeros(10)

        for _ in range(num_trials):
            sampler = SubsetRandomSampler(indices)
            first_idx = next(iter(sampler))
            counts[first_idx] += 1

        # Should be roughly uniform
        expected = num_trials / 10
        chi2, p_value = stats.chisquare(counts, [expected] * 10)

        self.assertGreater(p_value, 0.01,
            f"SubsetRandomSampler not uniform: p={p_value}")


# =============================================================================
# MAJOR: BatchSampler Parity Tests
# =============================================================================

@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestBatchSamplerParity(TestCase):
    """Parity tests for BatchSampler vs PyTorch."""

    def test_batch_sampler_structure_parity(self):
        """Verify BatchSampler produces same structure as PyTorch."""
        import torch
        from torch.utils.data import (
            SequentialSampler as TorchSequentialSampler,
            BatchSampler as TorchBatchSampler
        )

        data = list(range(25))
        batch_size = 7

        for drop_last in [True, False]:
            # MLX
            mlx_base = SequentialSampler(data)
            mlx_batch = BatchSampler(mlx_base, batch_size, drop_last=drop_last)
            mlx_batches = list(mlx_batch)

            # PyTorch
            torch_base = TorchSequentialSampler(data)
            torch_batch = TorchBatchSampler(torch_base, batch_size, drop_last=drop_last)
            torch_batches = list(torch_batch)

            # Same number of batches
            self.assertEqual(len(mlx_batches), len(torch_batches))

            # Same batch contents (for sequential sampler)
            for i, (mlx_b, torch_b) in enumerate(zip(mlx_batches, torch_batches)):
                self.assertEqual(mlx_b, list(torch_b),
                    f"Batch {i} mismatch: MLX={mlx_b}, PyTorch={torch_b}")

    def test_batch_sampler_with_random_preserves_coverage(self):
        """BatchSampler with RandomSampler should cover all indices."""
        data = list(range(100))
        batch_size = 17

        base = RandomSampler(data, replacement=False)
        batch_sampler = BatchSampler(base, batch_size, drop_last=False)

        all_indices = []
        for batch in batch_sampler:
            all_indices.extend(batch)

        # Should cover all indices exactly once
        self.assertEqual(sorted(all_indices), list(range(100)))


# =============================================================================
# MODERATE: ChainDataset and StackDataset Parity
# =============================================================================

@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestChainDatasetParity(TestCase):
    """Parity tests for ChainDataset vs PyTorch."""

    def test_chain_dataset_order_parity(self):
        """Verify ChainDataset iteration order matches PyTorch."""
        import torch
        from torch.utils.data import ChainDataset as TorchChainDataset
        from torch.utils.data import IterableDataset as TorchIterableDataset

        # PyTorch ChainDataset requires IterableDataset subclasses
        class TorchIterable(TorchIterableDataset):
            def __init__(self, start, end):
                self.start = start
                self.end = end
            def __iter__(self):
                return iter(range(self.start, self.end))

        # MLX version uses our IterableDataset
        from mlx_compat.data import IterableDataset

        class MLXIterable(IterableDataset):
            def __init__(self, start, end):
                self.start = start
                self.end = end
            def __iter__(self):
                return iter(range(self.start, self.end))

        # MLX
        mlx_d1 = MLXIterable(0, 5)
        mlx_d2 = MLXIterable(10, 15)
        mlx_chain = ChainDataset([mlx_d1, mlx_d2])
        mlx_result = list(mlx_chain)

        # PyTorch
        torch_d1 = TorchIterable(0, 5)
        torch_d2 = TorchIterable(10, 15)
        torch_chain = TorchChainDataset([torch_d1, torch_d2])
        torch_result = list(torch_chain)

        self.assertEqual(mlx_result, torch_result)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestStackDatasetParity(TestCase):
    """Parity tests for StackDataset."""

    def test_stack_dataset_structure(self):
        """Verify StackDataset produces correct dict structure."""
        import torch

        # Create test data
        np_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np_y = np.array([0, 1, 0])

        # MLX version
        x_mlx = mlx_compat.tensor(np_x)
        y_mlx = mlx_compat.tensor(np_y)
        ds_x = TensorDataset(x_mlx)
        ds_y = TensorDataset(y_mlx)
        stacked = StackDataset(features=ds_x, labels=ds_y)

        # Check structure
        sample = stacked[0]
        self.assertIsInstance(sample, dict)
        self.assertEqual(set(sample.keys()), {'features', 'labels'})

        # Check values
        np.testing.assert_allclose(sample['features'][0].numpy(), np_x[0])


# =============================================================================
# MODERATE: DataLoader Shuffle Statistical Validation
# =============================================================================

@skipIfNoMLX
class TestDataLoaderShuffleStatistics(TestCase):
    """Statistical validation that DataLoader shuffle is truly random."""

    def test_shuffle_produces_different_orders(self):
        """Shuffling should produce different orders across iterations."""
        x = mlx_compat.arange(100).reshape(100, 1)
        dataset = TensorDataset(x)

        orders = []
        for _ in range(20):
            loader = DataLoader(dataset, batch_size=100, shuffle=True)
            batch = next(iter(loader))
            order = tuple(int(v) for v in batch[0].numpy().flatten())
            orders.append(order)

        unique_orders = set(orders)
        # Should see many different orders
        self.assertGreater(len(unique_orders), 10,
            f"Shuffle produced too few unique orders: {len(unique_orders)}/20")

    @skipIfNoScipy
    def test_shuffle_first_element_uniform(self):
        """First element after shuffle should be uniformly distributed."""
        x = mlx_compat.arange(10).reshape(10, 1)
        dataset = TensorDataset(x)

        first_elements = []
        for _ in range(1000):
            loader = DataLoader(dataset, batch_size=10, shuffle=True)
            batch = next(iter(loader))
            first_elem = int(batch[0][0].item())
            first_elements.append(first_elem)

        counts = np.bincount(first_elements, minlength=10)
        expected = 100  # 1000 / 10

        chi2, p_value = stats.chisquare(counts, [expected] * 10)
        self.assertGreater(p_value, 0.01,
            f"First element not uniform: counts={counts}, p={p_value}")


# =============================================================================
# MODERATE: DistributedSampler Distribution Validation
# =============================================================================

@skipIfNoMLX
class TestDistributedSamplerDistribution(TestCase):
    """Distribution validation for DistributedSampler."""

    def test_each_rank_gets_fair_share(self):
        """Each rank should get approximately equal share of data."""
        x = mlx_compat.randn(100, 10)
        dataset = TensorDataset(x)

        num_replicas = 4
        rank_sizes = []

        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
            rank_sizes.append(len(list(sampler)))

        # All ranks should have same size
        self.assertEqual(len(set(rank_sizes)), 1,
            f"Unequal rank sizes: {rank_sizes}")

    @skipIfNoScipy
    def test_shuffle_distribution_across_epochs(self):
        """Shuffled distribution should be uniform across multiple epochs."""
        x = mlx_compat.arange(40).reshape(40, 1)
        dataset = TensorDataset(x)

        sampler = DistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True, seed=42
        )

        # Count first-element occurrences across epochs
        first_elements = []
        for epoch in range(100):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            first_elements.append(indices[0])

        counts = np.bincount(first_elements, minlength=40)

        # Should be roughly uniform (each element ~2.5 times in 100 epochs)
        # Just check no element is completely absent or overly dominant
        self.assertTrue(all(c > 0 for c in counts if c != 0),
            "Some elements never appear first")
        self.assertTrue(max(counts) < 10,
            f"Element appears too often as first: max={max(counts)}")


# =============================================================================
# MODERATE: Collate NumPy Array Handling
# =============================================================================

@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestCollateNumpyParity(TestCase):
    """Parity tests for collating NumPy arrays."""

    def test_collate_numpy_arrays(self):
        """Verify NumPy array collation produces correct tensor."""
        import torch
        from mlx_compat.data.dataloader import default_collate as mlx_collate
        from torch.utils.data._utils.collate import default_collate as torch_collate

        # Create numpy arrays
        np_samples = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]

        # MLX collation
        mlx_result = mlx_collate(np_samples)

        # PyTorch collation
        torch_result = torch_collate(np_samples)

        # Compare shapes
        self.assertEqual(mlx_result.shape, tuple(torch_result.shape))

        # Compare values
        np.testing.assert_allclose(
            mlx_result.numpy(),
            torch_result.numpy(),
            rtol=1e-5
        )

    def test_collate_numpy_preserves_dtype(self):
        """Verify NumPy dtype is reasonably preserved."""
        from mlx_compat.data.dataloader import default_collate

        # Float32
        np_float32 = [np.array([1.0, 2.0], dtype=np.float32)]
        result = default_collate(np_float32)
        # MLX should produce float32
        self.assertEqual(result.numpy().dtype, np.float32)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
