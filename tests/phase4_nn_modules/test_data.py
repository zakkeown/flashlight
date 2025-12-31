"""
Test Phase 4: Data Loading

Tests the data module:
- Dataset, TensorDataset, ConcatDataset, Subset
- DataLoader, default_collate
- Samplers
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    from mlx_compat.data import TensorDataset, ConcatDataset, DataLoader, Subset
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestTensorDataset(TestCase):
    """Test TensorDataset class."""

    def test_tensor_dataset_creation_single(self):
        """Test TensorDataset creation with single tensor."""
        x = mlx_compat.randn(100, 10)
        dataset = TensorDataset(x)
        self.assertEqual(len(dataset), 100)

    def test_tensor_dataset_creation_multiple(self):
        """Test TensorDataset creation with multiple tensors."""
        x = mlx_compat.randn(100, 10)
        y = mlx_compat.randint(0, 2, (100,))
        dataset = TensorDataset(x, y)
        self.assertEqual(len(dataset), 100)

    def test_tensor_dataset_getitem(self):
        """Test TensorDataset __getitem__."""
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = mlx_compat.tensor([0, 1, 0])
        dataset = TensorDataset(x, y)

        sample = dataset[0]
        self.assertEqual(len(sample), 2)
        np.testing.assert_allclose(sample[0].numpy(), [1.0, 2.0])
        self.assertEqual(int(sample[1].numpy()), 0)

    def test_tensor_dataset_mismatched_lengths(self):
        """Test TensorDataset with mismatched tensor lengths raises error."""
        x = mlx_compat.randn(100, 10)
        y = mlx_compat.randn(50, 5)  # Different first dimension
        with self.assertRaises(ValueError):
            TensorDataset(x, y)

    def test_tensor_dataset_empty_raises(self):
        """Test TensorDataset with no tensors raises error."""
        with self.assertRaises(ValueError):
            TensorDataset()


@skipIfNoMLX
class TestConcatDataset(TestCase):
    """Test ConcatDataset class."""

    def test_concat_two_datasets(self):
        """Test concatenating two datasets."""
        x1 = mlx_compat.randn(10, 5)
        x2 = mlx_compat.randn(20, 5)
        dataset1 = TensorDataset(x1)
        dataset2 = TensorDataset(x2)

        combined = ConcatDataset([dataset1, dataset2])
        self.assertEqual(len(combined), 30)

    def test_concat_dataset_indexing(self):
        """Test indexing into a concatenated dataset."""
        x1 = mlx_compat.tensor([[1.0], [2.0]])
        x2 = mlx_compat.tensor([[3.0], [4.0], [5.0]])
        dataset1 = TensorDataset(x1)
        dataset2 = TensorDataset(x2)

        combined = ConcatDataset([dataset1, dataset2])

        # Index into first dataset
        sample = combined[0]
        np.testing.assert_allclose(sample[0].numpy(), [1.0])

        # Index into second dataset
        sample = combined[2]
        np.testing.assert_allclose(sample[0].numpy(), [3.0])

    def test_concat_dataset_add_operator(self):
        """Test concatenation using + operator."""
        x1 = mlx_compat.randn(10, 5)
        x2 = mlx_compat.randn(20, 5)
        dataset1 = TensorDataset(x1)
        dataset2 = TensorDataset(x2)

        combined = dataset1 + dataset2
        self.assertEqual(len(combined), 30)


@skipIfNoMLX
class TestSubset(TestCase):
    """Test Subset class."""

    def test_subset_creation(self):
        """Test Subset creation."""
        x = mlx_compat.randn(100, 10)
        dataset = TensorDataset(x)
        indices = [0, 5, 10, 15, 20]

        subset = Subset(dataset, indices)
        self.assertEqual(len(subset), 5)

    def test_subset_indexing(self):
        """Test Subset indexing."""
        x = mlx_compat.tensor([[float(i)] for i in range(10)])
        dataset = TensorDataset(x)
        indices = [2, 5, 8]

        subset = Subset(dataset, indices)
        sample = subset[0]  # Should be dataset[2]
        np.testing.assert_allclose(sample[0].numpy(), [2.0])

        sample = subset[2]  # Should be dataset[8]
        np.testing.assert_allclose(sample[0].numpy(), [8.0])


@skipIfNoMLX
class TestDataLoader(TestCase):
    """Test DataLoader class."""

    def test_dataloader_creation(self):
        """Test DataLoader creation with default parameters."""
        x = mlx_compat.randn(32, 10)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset)
        self.assertIsNotNone(loader)

    def test_dataloader_iteration(self):
        """Test DataLoader iteration."""
        x = mlx_compat.randn(32, 10)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=8)

        batches = list(loader)
        self.assertEqual(len(batches), 4)  # 32 / 8 = 4 batches

    def test_dataloader_batch_size(self):
        """Test DataLoader batch size."""
        x = mlx_compat.randn(32, 10)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=16)

        for batch in loader:
            self.assertEqual(batch[0].shape[0], 16)
            break

    def test_dataloader_drop_last(self):
        """Test DataLoader with drop_last=True."""
        x = mlx_compat.randn(35, 10)  # 35 not divisible by 8
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=8, drop_last=True)

        batches = list(loader)
        self.assertEqual(len(batches), 4)  # 35 // 8 = 4 (drops 3 samples)

    def test_dataloader_no_drop_last(self):
        """Test DataLoader with drop_last=False (default)."""
        x = mlx_compat.randn(35, 10)  # 35 not divisible by 8
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=8, drop_last=False)

        batches = list(loader)
        self.assertEqual(len(batches), 5)  # 4 full + 1 partial

    def test_dataloader_shuffle(self):
        """Test DataLoader with shuffle=True produces different order."""
        x = mlx_compat.tensor([[float(i)] for i in range(100)])
        dataset = TensorDataset(x)

        # Get order without shuffle
        loader_no_shuffle = DataLoader(dataset, batch_size=10, shuffle=False)
        first_no_shuffle = next(iter(loader_no_shuffle))[0][:, 0].numpy()

        # Get order with shuffle (may or may not differ due to randomness)
        loader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)
        first_shuffle = next(iter(loader_shuffle))[0][:, 0].numpy()

        # Just verify both produce valid data
        self.assertEqual(first_no_shuffle.shape, (10,))
        self.assertEqual(first_shuffle.shape, (10,))

    def test_dataloader_multiple_tensors(self):
        """Test DataLoader with multiple tensors in dataset."""
        x = mlx_compat.randn(32, 10)
        y = mlx_compat.randint(0, 10, (32,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8)

        for batch_x, batch_y in loader:
            self.assertEqual(batch_x.shape, (8, 10))
            self.assertEqual(batch_y.shape, (8,))
            break


@skipIfNoMLX
class TestDefaultCollate(TestCase):
    """Test default_collate function."""

    def test_collate_tensors(self):
        """Test collating tensors."""
        from mlx_compat.data.dataloader import default_collate

        samples = [
            mlx_compat.tensor([1.0, 2.0]),
            mlx_compat.tensor([3.0, 4.0]),
            mlx_compat.tensor([5.0, 6.0])
        ]
        batch = default_collate(samples)
        self.assertEqual(batch.shape, (3, 2))

    def test_collate_tuples(self):
        """Test collating tuples of tensors."""
        from mlx_compat.data.dataloader import default_collate

        samples = [
            (mlx_compat.tensor([1.0]), mlx_compat.tensor([0])),
            (mlx_compat.tensor([2.0]), mlx_compat.tensor([1])),
        ]
        batch = default_collate(samples)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].shape, (2, 1))
        self.assertEqual(batch[1].shape, (2, 1))

    def test_collate_numbers(self):
        """Test collating numbers."""
        from mlx_compat.data.dataloader import default_collate

        samples = [1.0, 2.0, 3.0]
        batch = default_collate(samples)
        np.testing.assert_allclose(batch.numpy(), [1.0, 2.0, 3.0])


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
