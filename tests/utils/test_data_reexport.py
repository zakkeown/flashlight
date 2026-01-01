"""
Tests for flashlight.utils.data re-exports.

Verifies that all data loading utilities are correctly re-exported
from the utils.data namespace for PyTorch API compatibility.
"""

import unittest


class TestDataReexports(unittest.TestCase):
    """Test that utils.data re-exports all expected items."""

    def test_dataset_imports(self):
        """Test Dataset class imports."""
        from flashlight.utils.data import (
            Dataset,
            TensorDataset,
            IterableDataset,
            ConcatDataset,
            ChainDataset,
            Subset,
            StackDataset,
        )

        self.assertIsNotNone(Dataset)
        self.assertIsNotNone(TensorDataset)
        self.assertIsNotNone(IterableDataset)
        self.assertIsNotNone(ConcatDataset)
        self.assertIsNotNone(ChainDataset)
        self.assertIsNotNone(Subset)
        self.assertIsNotNone(StackDataset)

    def test_sampler_imports(self):
        """Test Sampler class imports."""
        from flashlight.utils.data import (
            Sampler,
            SequentialSampler,
            RandomSampler,
            BatchSampler,
            SubsetRandomSampler,
            WeightedRandomSampler,
            DistributedSampler,
        )

        self.assertIsNotNone(Sampler)
        self.assertIsNotNone(SequentialSampler)
        self.assertIsNotNone(RandomSampler)
        self.assertIsNotNone(BatchSampler)
        self.assertIsNotNone(SubsetRandomSampler)
        self.assertIsNotNone(WeightedRandomSampler)
        self.assertIsNotNone(DistributedSampler)

    def test_dataloader_imports(self):
        """Test DataLoader imports."""
        from flashlight.utils.data import DataLoader, default_collate

        self.assertIsNotNone(DataLoader)
        self.assertIsNotNone(default_collate)

    def test_utility_function_imports(self):
        """Test utility function imports."""
        from flashlight.utils.data import (
            get_worker_info,
            default_convert,
            random_split,
        )

        self.assertIsNotNone(get_worker_info)
        self.assertIsNotNone(default_convert)
        self.assertIsNotNone(random_split)

    def test_datapipe_imports(self):
        """Test DataPipe imports."""
        from flashlight.utils.data import (
            IterDataPipe,
            MapDataPipe,
            DFIterDataPipe,
            DataChunk,
            functional_datapipe,
        )

        self.assertIsNotNone(IterDataPipe)
        self.assertIsNotNone(MapDataPipe)
        self.assertIsNotNone(DFIterDataPipe)
        self.assertIsNotNone(DataChunk)
        self.assertIsNotNone(functional_datapipe)

    def test_validation_imports(self):
        """Test validation decorator imports."""
        from flashlight.utils.data import (
            argument_validation,
            runtime_validation,
            runtime_validation_disabled,
            guaranteed_datapipes_determinism,
            non_deterministic,
        )

        self.assertIsNotNone(argument_validation)
        self.assertIsNotNone(runtime_validation)
        self.assertIsNotNone(runtime_validation_disabled)
        self.assertIsNotNone(guaranteed_datapipes_determinism)
        self.assertIsNotNone(non_deterministic)

    def test_internal_imports(self):
        """Test internal class imports."""
        from flashlight.utils.data import _DatasetKind

        self.assertIsNotNone(_DatasetKind)

    def test_all_exports(self):
        """Test that __all__ contains expected items."""
        from flashlight.utils import data

        expected_exports = [
            'Dataset', 'TensorDataset', 'IterableDataset',
            'ConcatDataset', 'ChainDataset', 'Subset', 'StackDataset',
            'random_split',
            'Sampler', 'SequentialSampler', 'RandomSampler',
            'BatchSampler', 'SubsetRandomSampler', 'WeightedRandomSampler',
            'DistributedSampler',
            'DataLoader', 'default_collate',
            'get_worker_info', 'default_convert',
        ]

        for item in expected_exports:
            self.assertIn(item, data.__all__, f"{item} not in __all__")

    def test_same_as_flashlight_data(self):
        """Test that re-exports are identical to flashlight.data."""
        from flashlight import data as direct_data
        from flashlight.utils import data as utils_data

        # Key classes should be the same object
        self.assertIs(utils_data.Dataset, direct_data.Dataset)
        self.assertIs(utils_data.DataLoader, direct_data.DataLoader)
        self.assertIs(utils_data.Sampler, direct_data.Sampler)
        self.assertIs(utils_data.TensorDataset, direct_data.TensorDataset)


class TestDataFunctionality(unittest.TestCase):
    """Test that re-exported items work correctly."""

    def test_tensor_dataset_works(self):
        """Test TensorDataset functionality through re-export."""
        from flashlight.utils.data import TensorDataset
        import flashlight

        x = flashlight.randn(10, 5)
        y = flashlight.randint(0, 2, (10,))

        dataset = TensorDataset(x, y)

        self.assertEqual(len(dataset), 10)
        sample_x, sample_y = dataset[0]
        self.assertEqual(sample_x.shape, (5,))

    def test_dataloader_works(self):
        """Test DataLoader functionality through re-export."""
        from flashlight.utils.data import DataLoader, TensorDataset
        import flashlight

        x = flashlight.randn(100, 10)
        y = flashlight.randint(0, 2, (100,))

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16)

        batch_count = 0
        for batch_x, batch_y in loader:
            batch_count += 1
            if batch_count == 1:
                self.assertEqual(batch_x.shape[0], 16)

        self.assertGreater(batch_count, 0)

    def test_random_sampler_works(self):
        """Test RandomSampler functionality through re-export."""
        from flashlight.utils.data import RandomSampler, TensorDataset
        import flashlight

        x = flashlight.randn(50, 5)
        dataset = TensorDataset(x)

        sampler = RandomSampler(dataset)
        indices = list(sampler)

        self.assertEqual(len(indices), 50)
        # Check randomness (indices should not be in order)
        self.assertNotEqual(indices, list(range(50)))

    def test_sequential_sampler_works(self):
        """Test SequentialSampler functionality through re-export."""
        from flashlight.utils.data import SequentialSampler, TensorDataset
        import flashlight

        x = flashlight.randn(20, 5)
        dataset = TensorDataset(x)

        sampler = SequentialSampler(dataset)
        indices = list(sampler)

        self.assertEqual(indices, list(range(20)))

    def test_random_split_works(self):
        """Test random_split functionality through re-export."""
        from flashlight.utils.data import random_split, TensorDataset
        import flashlight

        x = flashlight.randn(100, 5)
        dataset = TensorDataset(x)

        train, val = random_split(dataset, [80, 20])

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)


if __name__ == "__main__":
    unittest.main()
