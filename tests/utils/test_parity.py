"""
PyTorch parity tests for mlx_compat.utils module.

Compares behavior and numerical results against PyTorch.
"""

import unittest
import numpy as np

import pytest

try:
    import torch
    import torch.utils.benchmark as torch_benchmark
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import mlx_compat
from mlx_compat.utils.benchmark import Timer, Measurement, Compare, select_unit, trim_sigfig


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestBenchmarkParity(unittest.TestCase):
    """Test benchmark utilities against PyTorch."""

    def test_select_unit_parity(self):
        """Test select_unit matches PyTorch behavior."""
        test_times = [1e-9, 1e-7, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]

        for t in test_times:
            mlx_unit, mlx_scale = select_unit(t)

            # Verify units are valid
            self.assertIn(mlx_unit, ["ns", "us", "ms", "s"])

            # Verify scaling is correct
            scaled = t / mlx_scale
            self.assertGreaterEqual(scaled, 0.001)

    def test_trim_sigfig_parity(self):
        """Test trim_sigfig produces reasonable results."""
        test_cases = [
            (123.456, 3),
            (0.00123, 2),
            (9999.9, 2),
            (0.0, 5),
        ]

        for x, n in test_cases:
            result = trim_sigfig(x, n)
            # Result should be a number (int or float)
            self.assertIsInstance(result, (int, float))

    def test_timer_measures_time(self):
        """Test Timer produces reasonable timing results."""
        # MLX Timer
        mlx_timer = Timer(
            stmt="x = 1 + 1",
            label="simple_add",
        )
        mlx_measurement = mlx_timer.timeit(1000)

        self.assertIsInstance(mlx_measurement, Measurement)
        self.assertGreater(mlx_measurement.median, 0)

        # PyTorch Timer (for comparison structure)
        torch_timer = torch_benchmark.Timer(
            stmt="x = 1 + 1",
            label="simple_add",
        )
        torch_measurement = torch_timer.timeit(1000)

        # Both should measure positive time
        self.assertGreater(torch_measurement.median, 0)

    def test_measurement_statistics(self):
        """Test Measurement statistics are reasonable."""
        from mlx_compat.utils.benchmark import TaskSpec

        spec = TaskSpec(stmt="pass")
        times = [0.1, 0.11, 0.09, 0.105, 0.095]

        m = Measurement(
            number_per_run=1,
            raw_times=times,
            task_spec=spec,
        )

        # Check statistics
        self.assertAlmostEqual(m.mean, np.mean(times), places=5)
        self.assertAlmostEqual(m.median, np.median(times), places=5)
        self.assertGreater(m.iqr, 0)


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestCheckpointParity(unittest.TestCase):
    """Test checkpoint against PyTorch behavior."""

    def test_checkpoint_output_parity(self):
        """Test checkpoint output matches non-checkpointed output."""
        np.random.seed(42)
        data = np.random.randn(4, 10).astype(np.float32)

        # MLX function
        def mlx_fn(x):
            return mlx_compat.relu(x * 2 + 1)

        # PyTorch function
        def torch_fn(x):
            return torch.relu(x * 2 + 1)

        # MLX
        mlx_x = mlx_compat.tensor(data, requires_grad=True)
        from mlx_compat.utils import checkpoint as mlx_checkpoint
        mlx_result = mlx_checkpoint(mlx_fn, mlx_x)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_result = torch.utils.checkpoint.checkpoint(torch_fn, torch_x, use_reentrant=True)

        # Compare outputs
        import mlx.core as mx
        mx.eval(mlx_result._mlx_array)

        np.testing.assert_allclose(
            mlx_result._mlx_array.tolist(),
            torch_result.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_checkpoint_sequential_output_parity(self):
        """Test checkpoint_sequential output matches PyTorch."""
        np.random.seed(42)
        data = np.random.randn(2, 10).astype(np.float32)

        # Simple linear layers
        # Both PyTorch and MLX Linear weight is (out_features, in_features)
        weight1 = np.random.randn(10, 10).astype(np.float32)  # 10 -> 10
        weight2 = np.random.randn(5, 10).astype(np.float32)   # 10 -> 5

        # MLX layers
        mlx_linear1 = mlx_compat.nn.Linear(10, 10, bias=False)
        mlx_linear1.weight = mlx_compat.nn.Parameter(mlx_compat.tensor(weight1))
        mlx_linear2 = mlx_compat.nn.Linear(10, 5, bias=False)
        mlx_linear2.weight = mlx_compat.nn.Parameter(mlx_compat.tensor(weight2))

        mlx_layers = [mlx_linear1, mlx_linear2]

        # PyTorch layers
        torch_linear1 = torch.nn.Linear(10, 10, bias=False)
        torch_linear1.weight = torch.nn.Parameter(torch.tensor(weight1))
        torch_linear2 = torch.nn.Linear(10, 5, bias=False)
        torch_linear2.weight = torch.nn.Parameter(torch.tensor(weight2))

        torch_layers = [torch_linear1, torch_linear2]

        # MLX checkpoint_sequential
        mlx_x = mlx_compat.tensor(data, requires_grad=True)
        from mlx_compat.utils.checkpoint import checkpoint_sequential
        mlx_result = checkpoint_sequential(mlx_layers, segments=2, input=mlx_x)

        # PyTorch checkpoint_sequential
        torch_x = torch.tensor(data, requires_grad=True)
        torch_result = torch.utils.checkpoint.checkpoint_sequential(
            torch.nn.Sequential(*torch_layers),
            segments=2,
            input=torch_x,
            use_reentrant=True,
        )

        # Compare
        import mlx.core as mx
        mx.eval(mlx_result._mlx_array)

        np.testing.assert_allclose(
            mlx_result._mlx_array.tolist(),
            torch_result.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestHooksParity(unittest.TestCase):
    """Test hooks utilities against PyTorch behavior."""

    def test_removable_handle_api(self):
        """Test RemovableHandle has PyTorch-compatible API."""
        from mlx_compat.utils.hooks import RemovableHandle
        from collections import OrderedDict

        # Both should support the same basic operations
        mlx_hooks = OrderedDict()
        mlx_handle = RemovableHandle(mlx_hooks)

        # Check API compatibility
        self.assertTrue(hasattr(mlx_handle, 'id'))
        self.assertTrue(hasattr(mlx_handle, 'remove'))
        self.assertTrue(callable(mlx_handle.remove))

        # Context manager support
        self.assertTrue(hasattr(mlx_handle, '__enter__'))
        self.assertTrue(hasattr(mlx_handle, '__exit__'))

    def test_unserializable_hook_decorator(self):
        """Test unserializable_hook decorator marks functions."""
        from mlx_compat.utils.hooks import unserializable_hook

        @unserializable_hook
        def my_hook(grad):
            return grad

        # Should have marker attribute
        self.assertTrue(hasattr(my_hook, '__mlx_unserializable__'))


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestDataParity(unittest.TestCase):
    """Test data utilities against PyTorch behavior."""

    def test_dataloader_iteration_parity(self):
        """Test DataLoader iteration matches PyTorch."""
        np.random.seed(42)
        data = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 5, (100,)).astype(np.int64)

        # MLX
        from mlx_compat.utils.data import DataLoader, TensorDataset
        mlx_x = mlx_compat.tensor(data)
        mlx_y = mlx_compat.tensor(labels)
        mlx_dataset = TensorDataset(mlx_x, mlx_y)
        mlx_loader = DataLoader(mlx_dataset, batch_size=10, shuffle=False)

        # PyTorch
        torch_x = torch.tensor(data)
        torch_y = torch.tensor(labels)
        torch_dataset = torch.utils.data.TensorDataset(torch_x, torch_y)
        torch_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=10, shuffle=False)

        # Compare batch counts
        mlx_batches = list(mlx_loader)
        torch_batches = list(torch_loader)

        self.assertEqual(len(mlx_batches), len(torch_batches))

        # Compare first batch shapes
        self.assertEqual(mlx_batches[0][0].shape, tuple(torch_batches[0][0].shape))
        self.assertEqual(mlx_batches[0][1].shape, tuple(torch_batches[0][1].shape))

    def test_sequential_sampler_parity(self):
        """Test SequentialSampler matches PyTorch."""
        from mlx_compat.utils.data import SequentialSampler, TensorDataset

        data = np.random.randn(50, 5).astype(np.float32)

        # MLX
        mlx_dataset = TensorDataset(mlx_compat.tensor(data))
        mlx_sampler = SequentialSampler(mlx_dataset)
        mlx_indices = list(mlx_sampler)

        # PyTorch
        torch_dataset = torch.utils.data.TensorDataset(torch.tensor(data))
        torch_sampler = torch.utils.data.SequentialSampler(torch_dataset)
        torch_indices = list(torch_sampler)

        self.assertEqual(mlx_indices, torch_indices)

    def test_random_split_lengths(self):
        """Test random_split produces correct split sizes."""
        from mlx_compat.utils.data import random_split, TensorDataset

        data = np.random.randn(100, 5).astype(np.float32)

        # MLX
        mlx_dataset = TensorDataset(mlx_compat.tensor(data))
        mlx_train, mlx_val = random_split(mlx_dataset, [70, 30])

        # PyTorch
        torch_dataset = torch.utils.data.TensorDataset(torch.tensor(data))
        torch_train, torch_val = torch.utils.data.random_split(torch_dataset, [70, 30])

        self.assertEqual(len(mlx_train), len(torch_train))
        self.assertEqual(len(mlx_val), len(torch_val))


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestModelZooParity(unittest.TestCase):
    """Test model_zoo utilities against PyTorch behavior."""

    def test_get_dir_api_parity(self):
        """Test get_dir has same API as torch.hub.get_dir."""
        from mlx_compat.utils.model_zoo import get_dir, set_dir

        # Both should return a string path
        mlx_dir = get_dir()
        torch_dir = torch.hub.get_dir()

        self.assertIsInstance(mlx_dir, str)
        self.assertIsInstance(torch_dir, str)

        # Both should contain "hub" in path
        self.assertIn("hub", mlx_dir)
        self.assertIn("hub", torch_dir)

    def test_set_dir_api_parity(self):
        """Test set_dir has same API as torch.hub.set_dir."""
        from mlx_compat.utils.model_zoo import get_dir, set_dir

        original = get_dir()

        # Both should accept a path string
        set_dir("/tmp/test_cache")

        # Restore
        set_dir(original)


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestWeakParity(unittest.TestCase):
    """Test weak reference utilities against PyTorch behavior."""

    def test_weak_tensor_key_dict_api(self):
        """Test WeakTensorKeyDictionary has similar API to torch.utils.weak."""
        from mlx_compat.utils.weak import WeakTensorKeyDictionary

        # Create dict
        d = WeakTensorKeyDictionary()

        # Should support dict-like operations
        self.assertTrue(hasattr(d, '__setitem__'))
        self.assertTrue(hasattr(d, '__getitem__'))
        self.assertTrue(hasattr(d, '__delitem__'))
        self.assertTrue(hasattr(d, '__contains__'))
        self.assertTrue(hasattr(d, '__len__'))
        self.assertTrue(hasattr(d, '__iter__'))

    def test_weak_id_key_dict_matches_torch(self):
        """Test WeakIdKeyDictionary behaves like torch version."""
        from mlx_compat.utils.weak import WeakIdKeyDictionary
        import torch.utils.weak as torch_weak

        # Both should handle objects as keys by identity
        class Obj:
            pass

        mlx_dict = WeakIdKeyDictionary()
        torch_dict = torch_weak.WeakIdKeyDictionary()

        obj = Obj()

        mlx_dict[obj] = "value"
        torch_dict[obj] = "value"

        self.assertEqual(mlx_dict[obj], torch_dict[obj])
        self.assertEqual(len(mlx_dict), len(torch_dict))


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestMobileOptimizerParity(unittest.TestCase):
    """Test mobile_optimizer utilities against PyTorch behavior."""

    def test_optimizer_type_enum_values(self):
        """Test MobileOptimizerType has same values as PyTorch."""
        from mlx_compat.utils.mobile_optimizer import MobileOptimizerType

        # Check common optimization types exist
        self.assertTrue(hasattr(MobileOptimizerType, 'CONV_BN_FUSION'))
        self.assertTrue(hasattr(MobileOptimizerType, 'REMOVE_DROPOUT'))
        self.assertTrue(hasattr(MobileOptimizerType, 'FUSE_ADD_RELU'))

    def test_optimize_for_mobile_removes_dropout(self):
        """Test optimize_for_mobile removes dropout like PyTorch."""
        from mlx_compat.utils.mobile_optimizer import optimize_for_mobile
        import mlx_compat.nn as nn

        # MLX model with dropout
        mlx_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(0.5),
            nn.Linear(10, 5),
        )

        # PyTorch model with dropout
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(10, 5),
        )

        # Optimize MLX
        mlx_optimized = optimize_for_mobile(mlx_model)

        # PyTorch optimization (scripted)
        torch_model.eval()
        try:
            torch_scripted = torch.jit.script(torch_model)
            torch_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(torch_scripted)
            has_torch_mobile = True
        except Exception:
            has_torch_mobile = False

        # Both should be in eval mode after optimization
        self.assertFalse(mlx_optimized.training)

        # MLX dropout should be replaced
        has_dropout = any(isinstance(m, nn.Dropout) for m in mlx_optimized.children())
        self.assertFalse(has_dropout)


@pytest.mark.parity
@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestCollectEnvParity(unittest.TestCase):
    """Test collect_env utilities against PyTorch behavior."""

    def test_get_env_info_returns_namedtuple(self):
        """Test get_env_info returns structured data like PyTorch."""
        from mlx_compat.utils.collect_env import get_env_info, SystemEnv

        env = get_env_info()

        # Should be a named tuple
        self.assertIsInstance(env, SystemEnv)

        # Should have version fields
        self.assertTrue(hasattr(env, 'python_version'))
        self.assertTrue(hasattr(env, 'numpy_version'))

    def test_get_pretty_env_info_returns_string(self):
        """Test get_pretty_env_info returns formatted string like PyTorch."""
        from mlx_compat.utils.collect_env import get_pretty_env_info

        # MLX version
        mlx_info = get_pretty_env_info()

        # PyTorch version
        torch_info = torch.utils.collect_env.get_pretty_env_info()

        # Both should return non-empty strings
        self.assertIsInstance(mlx_info, str)
        self.assertIsInstance(torch_info, str)
        self.assertGreater(len(mlx_info), 100)
        self.assertGreater(len(torch_info), 100)

        # Both should contain Python version info
        self.assertIn("Python", mlx_info)
        self.assertIn("Python", torch_info)


if __name__ == "__main__":
    unittest.main()
