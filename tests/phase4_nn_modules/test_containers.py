"""
Test Phase 4: Container Modules

Tests the nn.containers module:
- Sequential
- ModuleList
- ModuleDict
- ParameterList
- ParameterDict
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestSequential(TestCase):
    """Test nn.Sequential container."""

    def test_creation_from_modules(self):
        """Test Sequential creation from module list."""
        seq = mlx_compat.nn.Sequential(
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.ReLU(),
            mlx_compat.nn.Linear(20, 5)
        )
        self.assertEqual(len(seq), 3)

    def test_forward(self):
        """Test Sequential forward pass."""
        seq = mlx_compat.nn.Sequential(
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.ReLU(),
            mlx_compat.nn.Linear(20, 5)
        )
        x = mlx_compat.randn(3, 10)
        output = seq(x)
        self.assertEqual(output.shape, (3, 5))

    def test_indexing(self):
        """Test indexing into Sequential."""
        linear1 = mlx_compat.nn.Linear(10, 20)
        linear2 = mlx_compat.nn.Linear(20, 5)
        seq = mlx_compat.nn.Sequential(linear1, linear2)
        self.assertIs(seq[0], linear1)
        self.assertIs(seq[1], linear2)

    def test_append(self):
        """Test appending to Sequential."""
        seq = mlx_compat.nn.Sequential()
        seq.append(mlx_compat.nn.Linear(10, 20))
        seq.append(mlx_compat.nn.ReLU())
        self.assertEqual(len(seq), 2)

    def test_parameters(self):
        """Test parameter iteration in Sequential."""
        seq = mlx_compat.nn.Sequential(
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.Linear(20, 5)
        )
        params = list(seq.parameters())
        self.assertEqual(len(params), 4)  # 2 weights + 2 biases


@skipIfNoMLX
class TestModuleList(TestCase):
    """Test nn.ModuleList container."""

    def test_creation(self):
        """Test ModuleList creation."""
        modules = mlx_compat.nn.ModuleList([
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.Linear(20, 30)
        ])
        self.assertEqual(len(modules), 2)

    def test_append(self):
        """Test appending to ModuleList."""
        modules = mlx_compat.nn.ModuleList()
        modules.append(mlx_compat.nn.Linear(10, 20))
        modules.append(mlx_compat.nn.Linear(20, 30))
        self.assertEqual(len(modules), 2)

    def test_extend(self):
        """Test extending ModuleList."""
        modules = mlx_compat.nn.ModuleList([mlx_compat.nn.Linear(10, 20)])
        modules.extend([mlx_compat.nn.Linear(20, 30), mlx_compat.nn.Linear(30, 40)])
        self.assertEqual(len(modules), 3)

    def test_indexing(self):
        """Test indexing ModuleList."""
        linear1 = mlx_compat.nn.Linear(10, 20)
        linear2 = mlx_compat.nn.Linear(20, 30)
        modules = mlx_compat.nn.ModuleList([linear1, linear2])
        self.assertIs(modules[0], linear1)
        self.assertIs(modules[1], linear2)

    def test_iteration(self):
        """Test iterating over ModuleList."""
        modules = mlx_compat.nn.ModuleList([
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.ReLU(),
            mlx_compat.nn.Linear(20, 5)
        ])
        count = sum(1 for _ in modules)
        self.assertEqual(count, 3)

    def test_parameters(self):
        """Test parameter iteration in ModuleList."""
        modules = mlx_compat.nn.ModuleList([
            mlx_compat.nn.Linear(10, 20),
            mlx_compat.nn.Linear(20, 5)
        ])
        params = list(modules.parameters())
        self.assertEqual(len(params), 4)


@skipIfNoMLX
class TestModuleDict(TestCase):
    """Test nn.ModuleDict container."""

    def test_creation(self):
        """Test ModuleDict creation."""
        modules = mlx_compat.nn.ModuleDict({
            'linear1': mlx_compat.nn.Linear(10, 20),
            'linear2': mlx_compat.nn.Linear(20, 30)
        })
        self.assertEqual(len(modules), 2)

    def test_key_access(self):
        """Test accessing modules by key."""
        linear1 = mlx_compat.nn.Linear(10, 20)
        modules = mlx_compat.nn.ModuleDict({'linear1': linear1})
        self.assertIs(modules['linear1'], linear1)

    def test_update(self):
        """Test updating ModuleDict."""
        modules = mlx_compat.nn.ModuleDict()
        modules['linear1'] = mlx_compat.nn.Linear(10, 20)
        modules['linear2'] = mlx_compat.nn.Linear(20, 30)
        self.assertEqual(len(modules), 2)

    def test_keys_values_items(self):
        """Test keys(), values(), items() methods."""
        modules = mlx_compat.nn.ModuleDict({
            'a': mlx_compat.nn.Linear(10, 20),
            'b': mlx_compat.nn.Linear(20, 30)
        })
        self.assertEqual(set(modules.keys()), {'a', 'b'})
        self.assertEqual(len(list(modules.values())), 2)
        self.assertEqual(len(list(modules.items())), 2)


@skipIfNoMLX
class TestParameterList(TestCase):
    """Test nn.ParameterList container."""

    def test_creation(self):
        """Test ParameterList creation."""
        params = mlx_compat.nn.ParameterList([
            mlx_compat.nn.Parameter(mlx_compat.randn(3, 3)),
            mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))
        ])
        self.assertEqual(len(params), 2)

    def test_append(self):
        """Test appending to ParameterList."""
        params = mlx_compat.nn.ParameterList()
        params.append(mlx_compat.nn.Parameter(mlx_compat.randn(3, 3)))
        self.assertEqual(len(params), 1)

    def test_indexing(self):
        """Test indexing ParameterList."""
        p1 = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))
        params = mlx_compat.nn.ParameterList([p1])
        self.assertIs(params[0], p1)


@skipIfNoMLX
class TestParameterDict(TestCase):
    """Test nn.ParameterDict container."""

    def test_creation(self):
        """Test ParameterDict creation."""
        params = mlx_compat.nn.ParameterDict({
            'weight': mlx_compat.nn.Parameter(mlx_compat.randn(3, 3)),
            'bias': mlx_compat.nn.Parameter(mlx_compat.randn(3))
        })
        self.assertEqual(len(params), 2)

    def test_key_access(self):
        """Test accessing parameters by key."""
        weight = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))
        params = mlx_compat.nn.ParameterDict({'weight': weight})
        self.assertIs(params['weight'], weight)


if __name__ == '__main__':
    unittest.main()
