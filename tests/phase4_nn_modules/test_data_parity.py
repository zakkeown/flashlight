"""
Test Phase 4: PyTorch Parity Tests

Numerical parity comparison tests for data loading components.
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
        TensorDataset, ConcatDataset, Subset,
        DataLoader, default_collate
    )
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestDataLoaderParity(TestCase):
    """Parity tests comparing MLX DataLoader with PyTorch."""

    def test_batch_output_shape_parity(self):
        """Compare batch output shapes with PyTorch DataLoader."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data import DataLoader as TorchDataLoader

        batch_size = 16

        # MLX version
        x_mlx = mlx_compat.randn(100, 64)
        y_mlx = mlx_compat.randint(0, 10, (100,))
        dataset_mlx = TensorDataset(x_mlx, y_mlx)
        loader_mlx = DataLoader(dataset_mlx, batch_size=batch_size, shuffle=False)

        # PyTorch version
        x_torch = torch.randn(100, 64)
        y_torch = torch.randint(0, 10, (100,))
        dataset_torch = TorchTensorDataset(x_torch, y_torch)
        loader_torch = TorchDataLoader(dataset_torch, batch_size=batch_size, shuffle=False)

        # Compare batch shapes
        mlx_batches = list(loader_mlx)
        torch_batches = list(loader_torch)

        self.assertEqual(len(mlx_batches), len(torch_batches))

        for i, (mlx_batch, torch_batch) in enumerate(zip(mlx_batches, torch_batches)):
            mlx_x, mlx_y = mlx_batch
            torch_x, torch_y = torch_batch

            self.assertEqual(mlx_x.shape, tuple(torch_x.shape), f"Batch {i} x shape mismatch")
            self.assertEqual(mlx_y.shape, tuple(torch_y.shape), f"Batch {i} y shape mismatch")

    def test_num_batches_parity(self):
        """Compare number of batches with PyTorch DataLoader."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data import DataLoader as TorchDataLoader

        for dataset_size in [97, 100, 128]:
            for batch_size in [8, 16, 32]:
                for drop_last in [True, False]:
                    # MLX version
                    x_mlx = mlx_compat.randn(dataset_size, 10)
                    dataset_mlx = TensorDataset(x_mlx)
                    loader_mlx = DataLoader(
                        dataset_mlx, batch_size=batch_size, drop_last=drop_last
                    )

                    # PyTorch version
                    x_torch = torch.randn(dataset_size, 10)
                    dataset_torch = TorchTensorDataset(x_torch)
                    loader_torch = TorchDataLoader(
                        dataset_torch, batch_size=batch_size, drop_last=drop_last
                    )

                    mlx_count = len(list(loader_mlx))
                    torch_count = len(list(loader_torch))

                    self.assertEqual(
                        mlx_count, torch_count,
                        f"Batch count mismatch for size={dataset_size}, "
                        f"batch={batch_size}, drop_last={drop_last}"
                    )

    def test_sequential_order_parity(self):
        """Compare sequential ordering with PyTorch DataLoader."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data import DataLoader as TorchDataLoader

        # Create ordered data
        x_mlx = mlx_compat.arange(100).reshape(100, 1).float()
        dataset_mlx = TensorDataset(x_mlx)
        loader_mlx = DataLoader(dataset_mlx, batch_size=10, shuffle=False)

        x_torch = torch.arange(100).reshape(100, 1).float()
        dataset_torch = TorchTensorDataset(x_torch)
        loader_torch = TorchDataLoader(dataset_torch, batch_size=10, shuffle=False)

        mlx_values = np.concatenate([b[0].numpy() for b in loader_mlx])
        torch_values = np.concatenate([b[0].numpy() for b in loader_torch])

        np.testing.assert_allclose(mlx_values, torch_values)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestCollateParity(TestCase):
    """Parity tests for collate functions."""

    def test_collate_tensors_parity(self):
        """Compare tensor collation with PyTorch."""
        import torch
        from torch.utils.data._utils.collate import default_collate as torch_collate

        # MLX tensors
        mlx_samples = [
            mlx_compat.tensor([1.0, 2.0, 3.0]),
            mlx_compat.tensor([4.0, 5.0, 6.0]),
            mlx_compat.tensor([7.0, 8.0, 9.0]),
        ]
        mlx_batch = default_collate(mlx_samples)

        # PyTorch tensors
        torch_samples = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor([7.0, 8.0, 9.0]),
        ]
        torch_batch = torch_collate(torch_samples)

        # Compare shapes and values
        self.assertEqual(mlx_batch.shape, tuple(torch_batch.shape))
        np.testing.assert_allclose(
            mlx_batch.numpy(),
            torch_batch.numpy(),
            rtol=1e-5
        )

    def test_collate_tuples_parity(self):
        """Compare tuple collation with PyTorch."""
        import torch
        from torch.utils.data._utils.collate import default_collate as torch_collate

        # MLX tuples
        mlx_samples = [
            (mlx_compat.tensor([1.0, 2.0]), mlx_compat.tensor([0])),
            (mlx_compat.tensor([3.0, 4.0]), mlx_compat.tensor([1])),
        ]
        mlx_batch = default_collate(mlx_samples)

        # PyTorch tuples
        torch_samples = [
            (torch.tensor([1.0, 2.0]), torch.tensor([0])),
            (torch.tensor([3.0, 4.0]), torch.tensor([1])),
        ]
        torch_batch = torch_collate(torch_samples)

        self.assertEqual(len(mlx_batch), len(torch_batch))

        for mlx_elem, torch_elem in zip(mlx_batch, torch_batch):
            self.assertEqual(mlx_elem.shape, tuple(torch_elem.shape))
            np.testing.assert_allclose(
                mlx_elem.numpy(),
                torch_elem.numpy(),
                rtol=1e-5
            )

    def test_collate_dicts_parity(self):
        """Compare dict collation with PyTorch."""
        import torch
        from torch.utils.data._utils.collate import default_collate as torch_collate

        # MLX dicts
        mlx_samples = [
            {'x': mlx_compat.tensor([1.0, 2.0]), 'y': mlx_compat.tensor([0])},
            {'x': mlx_compat.tensor([3.0, 4.0]), 'y': mlx_compat.tensor([1])},
        ]
        mlx_batch = default_collate(mlx_samples)

        # PyTorch dicts
        torch_samples = [
            {'x': torch.tensor([1.0, 2.0]), 'y': torch.tensor([0])},
            {'x': torch.tensor([3.0, 4.0]), 'y': torch.tensor([1])},
        ]
        torch_batch = torch_collate(torch_samples)

        self.assertEqual(set(mlx_batch.keys()), set(torch_batch.keys()))

        for key in mlx_batch:
            self.assertEqual(mlx_batch[key].shape, tuple(torch_batch[key].shape))
            np.testing.assert_allclose(
                mlx_batch[key].numpy(),
                torch_batch[key].numpy(),
                rtol=1e-5
            )

    def test_collate_numbers_parity(self):
        """Compare number collation with PyTorch."""
        import torch
        from torch.utils.data._utils.collate import default_collate as torch_collate

        samples = [1.0, 2.0, 3.0, 4.0, 5.0]

        mlx_batch = default_collate(samples)
        torch_batch = torch_collate(samples)

        self.assertEqual(mlx_batch.shape, tuple(torch_batch.shape))
        np.testing.assert_allclose(
            mlx_batch.numpy(),
            torch_batch.numpy(),
            rtol=1e-5
        )

    def test_collate_nested_parity(self):
        """Compare nested structure collation with PyTorch."""
        import torch
        from torch.utils.data._utils.collate import default_collate as torch_collate

        # MLX nested
        mlx_samples = [
            {'inputs': (mlx_compat.tensor([1.0]), mlx_compat.tensor([2.0]))},
            {'inputs': (mlx_compat.tensor([3.0]), mlx_compat.tensor([4.0]))},
        ]
        mlx_batch = default_collate(mlx_samples)

        # PyTorch nested
        torch_samples = [
            {'inputs': (torch.tensor([1.0]), torch.tensor([2.0]))},
            {'inputs': (torch.tensor([3.0]), torch.tensor([4.0]))},
        ]
        torch_batch = torch_collate(torch_samples)

        # Check nested structure
        self.assertIn('inputs', mlx_batch)
        self.assertIsInstance(mlx_batch['inputs'], tuple)
        self.assertEqual(len(mlx_batch['inputs']), 2)

        for i in range(2):
            np.testing.assert_allclose(
                mlx_batch['inputs'][i].numpy(),
                torch_batch['inputs'][i].numpy(),
                rtol=1e-5
            )


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestDatasetParity(TestCase):
    """Parity tests for dataset classes."""

    def test_tensor_dataset_indexing_parity(self):
        """Compare TensorDataset indexing with PyTorch."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset

        # Create identical data
        np_x = np.random.randn(100, 10).astype(np.float32)
        np_y = np.random.randint(0, 5, 100)

        # MLX
        x_mlx = mlx_compat.tensor(np_x)
        y_mlx = mlx_compat.tensor(np_y)
        dataset_mlx = TensorDataset(x_mlx, y_mlx)

        # PyTorch
        x_torch = torch.from_numpy(np_x)
        y_torch = torch.from_numpy(np_y)
        dataset_torch = TorchTensorDataset(x_torch, y_torch)

        # Compare indexing
        for i in [0, 10, 50, 99]:
            mlx_sample = dataset_mlx[i]
            torch_sample = dataset_torch[i]

            np.testing.assert_allclose(
                mlx_sample[0].numpy(),
                torch_sample[0].numpy(),
                rtol=1e-5
            )
            np.testing.assert_array_equal(
                mlx_sample[1].numpy(),
                torch_sample[1].numpy()
            )

    def test_concat_dataset_parity(self):
        """Compare ConcatDataset with PyTorch."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data import ConcatDataset as TorchConcatDataset

        np_x1 = np.random.randn(30, 5).astype(np.float32)
        np_x2 = np.random.randn(20, 5).astype(np.float32)

        # MLX
        ds1_mlx = TensorDataset(mlx_compat.tensor(np_x1))
        ds2_mlx = TensorDataset(mlx_compat.tensor(np_x2))
        concat_mlx = ConcatDataset([ds1_mlx, ds2_mlx])

        # PyTorch
        ds1_torch = TorchTensorDataset(torch.from_numpy(np_x1))
        ds2_torch = TorchTensorDataset(torch.from_numpy(np_x2))
        concat_torch = TorchConcatDataset([ds1_torch, ds2_torch])

        self.assertEqual(len(concat_mlx), len(concat_torch))

        # Compare indices across boundary
        for i in [0, 29, 30, 49]:
            mlx_sample = concat_mlx[i]
            torch_sample = concat_torch[i]

            np.testing.assert_allclose(
                mlx_sample[0].numpy(),
                torch_sample[0].numpy(),
                rtol=1e-5
            )

    def test_subset_parity(self):
        """Compare Subset with PyTorch."""
        import torch
        from torch.utils.data import TensorDataset as TorchTensorDataset
        from torch.utils.data import Subset as TorchSubset

        np_x = np.random.randn(100, 10).astype(np.float32)
        indices = [5, 15, 25, 35, 45]

        # MLX
        ds_mlx = TensorDataset(mlx_compat.tensor(np_x))
        subset_mlx = Subset(ds_mlx, indices)

        # PyTorch
        ds_torch = TorchTensorDataset(torch.from_numpy(np_x))
        subset_torch = TorchSubset(ds_torch, indices)

        self.assertEqual(len(subset_mlx), len(subset_torch))

        for i in range(len(indices)):
            mlx_sample = subset_mlx[i]
            torch_sample = subset_torch[i]

            np.testing.assert_allclose(
                mlx_sample[0].numpy(),
                torch_sample[0].numpy(),
                rtol=1e-5
            )


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
