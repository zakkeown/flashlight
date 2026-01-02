"""
Test Phase 4: DistributedSampler

Tests for the DistributedSampler class.
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX, skipIfNoTorch

try:
    import flashlight
    from flashlight.data import DistributedSampler, TensorDataset

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestDistributedSampler(TestCase):
    """Test DistributedSampler class."""

    def test_single_replica(self):
        """Test with single replica (default behavior)."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
        indices = list(sampler)

        self.assertEqual(len(indices), 100)

    def test_partition_coverage(self):
        """Verify all indices covered across replicas."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        num_replicas = 4
        all_indices = []

        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
            indices = list(sampler)
            all_indices.extend(indices)

        # All indices should be covered (with possible padding)
        self.assertEqual(len(set(all_indices)), len(dataset))

    def test_partition_sizes(self):
        """Verify each replica gets same number of samples."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        num_replicas = 4
        sizes = []

        for rank in range(num_replicas):
            sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)
            sizes.append(len(sampler))

        # All replicas should have same size
        self.assertEqual(len(set(sizes)), 1)

    def test_set_epoch_changes_order(self):
        """Verify set_epoch produces different shuffle."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        sampler = DistributedSampler(dataset, shuffle=True, seed=42)

        sampler.set_epoch(0)
        indices_epoch0 = list(sampler)

        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # Different epochs should produce different orders
        self.assertNotEqual(indices_epoch0, indices_epoch1)

    def test_shuffle_reproducibility(self):
        """Verify same seed + epoch produces same shuffle."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        sampler1 = DistributedSampler(dataset, shuffle=True, seed=42)
        sampler2 = DistributedSampler(dataset, shuffle=True, seed=42)

        sampler1.set_epoch(5)
        sampler2.set_epoch(5)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        self.assertEqual(indices1, indices2)

    def test_no_shuffle(self):
        """Verify shuffle=False produces sequential order."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        indices = list(sampler)

        self.assertEqual(indices, list(range(100)))

    def test_drop_last_behavior(self):
        """Verify drop_last truncates correctly."""
        # 103 samples, 4 replicas
        # Without drop_last: each gets 26 (104 total, with padding)
        # With drop_last: each gets 25 (100 total)
        x = flashlight.randn(103, 10)
        dataset = TensorDataset(x)

        sampler_no_drop = DistributedSampler(dataset, num_replicas=4, rank=0, drop_last=False)
        sampler_drop = DistributedSampler(dataset, num_replicas=4, rank=0, drop_last=True)

        self.assertEqual(len(sampler_no_drop), 26)
        self.assertEqual(len(sampler_drop), 25)

    def test_padding_when_needed(self):
        """Verify padding added when dataset size not divisible by replicas."""
        # 10 samples, 4 replicas = need 12 samples total (3 per replica)
        x = flashlight.randn(10, 5)
        dataset = TensorDataset(x)

        num_replicas = 4
        all_indices = []

        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
            indices = list(sampler)
            all_indices.extend(indices)

        # Should have 12 indices total (with 2 padded)
        self.assertEqual(len(all_indices), 12)

    def test_length(self):
        """Test __len__ method."""
        x = flashlight.randn(100, 10)
        dataset = TensorDataset(x)

        sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
        self.assertEqual(len(sampler), 25)

        sampler = DistributedSampler(dataset, num_replicas=3, rank=0)
        self.assertEqual(len(sampler), 34)  # ceil(100/3) = 34

    def test_default_values(self):
        """Test default num_replicas and rank."""
        x = flashlight.randn(50, 10)
        dataset = TensorDataset(x)

        # Default should be 1 replica, rank 0
        sampler = DistributedSampler(dataset)
        self.assertEqual(len(sampler), 50)
        indices = list(sampler)
        self.assertEqual(len(indices), 50)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestDistributedSamplerParity(TestCase):
    """Parity tests comparing MLX DistributedSampler with PyTorch."""

    def test_distributed_coverage_parity(self):
        """Compare DistributedSampler coverage with PyTorch."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler

        # MLX version
        x_mlx = flashlight.randn(100, 10)
        dataset_mlx = TensorDataset(x_mlx)

        # PyTorch version
        x_torch = torch.randn(100, 10)
        dataset_torch = TorchTensorDataset(x_torch)

        num_replicas = 4

        # Test that both cover all indices
        mlx_all = []
        torch_all = []

        for rank in range(num_replicas):
            mlx_sampler = DistributedSampler(
                dataset_mlx, num_replicas=num_replicas, rank=rank, shuffle=False
            )
            torch_sampler = TorchDistributedSampler(
                dataset_torch, num_replicas=num_replicas, rank=rank, shuffle=False
            )

            mlx_all.extend(list(mlx_sampler))
            torch_all.extend(list(torch_sampler))

        # Both should cover all indices
        self.assertEqual(len(set(mlx_all)), len(dataset_mlx))
        self.assertEqual(len(set(torch_all)), len(dataset_torch))

    def test_distributed_length_parity(self):
        """Compare DistributedSampler lengths with PyTorch."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler

        for dataset_size in [97, 100, 103]:
            for num_replicas in [1, 2, 4, 8]:
                # MLX
                x_mlx = flashlight.randn(dataset_size, 10)
                dataset_mlx = TensorDataset(x_mlx)
                mlx_sampler = DistributedSampler(dataset_mlx, num_replicas=num_replicas, rank=0)

                # PyTorch
                x_torch = torch.randn(dataset_size, 10)
                dataset_torch = TorchTensorDataset(x_torch)
                torch_sampler = TorchDistributedSampler(
                    dataset_torch, num_replicas=num_replicas, rank=0
                )

                self.assertEqual(
                    len(mlx_sampler),
                    len(torch_sampler),
                    f"Length mismatch for size={dataset_size}, replicas={num_replicas}",
                )


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
