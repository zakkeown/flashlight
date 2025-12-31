"""
Test Phase 4: Neural Network Modules

Tests the nn.Module system:
- Module base class functionality
- Parameter registration and management
- train/eval mode switching
- Parameter iteration
- Module composition
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
class TestModule(TestCase):
    """Test nn.Module base class functionality."""

    def test_module_creation(self):
        """Test basic module creation."""
        class SimpleModule(mlx_compat.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))
                self.bias = mlx_compat.nn.Parameter(mlx_compat.randn(3))

            def forward(self, x):
                return x

        module = SimpleModule()
        self.assertIsNotNone(module)
        self.assertTrue(module.training)

    def test_parameter_registration(self):
        """Test that parameters are automatically registered."""
        class SimpleModule(mlx_compat.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))
                self.bias = mlx_compat.nn.Parameter(mlx_compat.randn(3))

            def forward(self, x):
                return x

        module = SimpleModule()
        params = list(module.parameters())
        self.assertEqual(len(params), 2)

        # Check named parameters
        named_params = dict(module.named_parameters())
        self.assertIn('weight', named_params)
        self.assertIn('bias', named_params)

    def test_train_eval_mode(self):
        """Test train/eval mode switching."""
        module = mlx_compat.nn.Module()

        # Default is training mode
        self.assertTrue(module.training)

        # Switch to eval mode
        module.eval()
        self.assertFalse(module.training)

        # Switch back to train mode
        module.train()
        self.assertTrue(module.training)

        # Explicit train(False) should set eval mode
        module.train(False)
        self.assertFalse(module.training)

    def test_nested_modules(self):
        """Test that nested modules are tracked."""
        class InnerModule(mlx_compat.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mlx_compat.nn.Parameter(mlx_compat.randn(2, 2))

            def forward(self, x):
                return x

        class OuterModule(mlx_compat.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()
                self.weight = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))

            def forward(self, x):
                return x

        module = OuterModule()

        # Should have 3 parameters total (2 from inner modules + 1 from outer)
        params = list(module.parameters())
        self.assertEqual(len(params), 3)

        # Check named parameters include nested
        named_params = dict(module.named_parameters())
        self.assertIn('weight', named_params)
        self.assertIn('inner1.weight', named_params)
        self.assertIn('inner2.weight', named_params)

    def test_train_mode_propagates(self):
        """Test that train/eval mode propagates to child modules."""
        class InnerModule(mlx_compat.nn.Module):
            def forward(self, x):
                return x

        class OuterModule(mlx_compat.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModule()

            def forward(self, x):
                return x

        module = OuterModule()

        # Set to eval mode
        module.eval()
        self.assertFalse(module.training)
        self.assertFalse(module.inner.training)

        # Set back to train mode
        module.train()
        self.assertTrue(module.training)
        self.assertTrue(module.inner.training)

    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        class SimpleModule(mlx_compat.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))

            def forward(self, x):
                return mlx_compat.sum(self.weight * x)

        module = SimpleModule()
        x = mlx_compat.randn(3, 3)

        # Forward + backward to create gradients
        output = module(x)
        output.backward()

        # Check gradient exists
        self.assertIsNotNone(module.weight.grad)

        # Zero gradients
        module.zero_grad()
        self.assertIsNone(module.weight.grad)


@skipIfNoMLX
class TestParameter(TestCase):
    """Test nn.Parameter class."""

    def test_parameter_creation(self):
        """Test parameter creation from data."""
        data = mlx_compat.randn(3, 3)
        param = mlx_compat.nn.Parameter(data)

        self.assertIsInstance(param, mlx_compat.nn.Parameter)
        self.assertTrue(param.requires_grad)
        self.assertEqual(param.shape, (3, 3))

    def test_parameter_requires_grad(self):
        """Test that parameters require gradients by default."""
        param = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3))
        self.assertTrue(param.requires_grad)

    def test_parameter_no_grad(self):
        """Test creating parameter without gradients."""
        param = mlx_compat.nn.Parameter(mlx_compat.randn(3, 3), requires_grad=False)
        self.assertFalse(param.requires_grad)

    def test_parameter_repr(self):
        """Test parameter string representation."""
        param = mlx_compat.nn.Parameter(mlx_compat.tensor([1.0, 2.0, 3.0]))
        repr_str = repr(param)
        self.assertIn('Parameter containing:', repr_str)


@skipIfNoMLX
class TestLinear(TestCase):
    """Test nn.Linear layer."""

    def test_linear_creation(self):
        """Test linear layer creation."""
        linear = mlx_compat.nn.Linear(10, 5)

        self.assertEqual(linear.in_features, 10)
        self.assertEqual(linear.out_features, 5)
        self.assertEqual(linear.weight.shape, (5, 10))
        self.assertEqual(linear.bias.shape, (5,))

    def test_linear_no_bias(self):
        """Test linear layer without bias."""
        linear = mlx_compat.nn.Linear(10, 5, bias=False)

        self.assertIsNone(linear.bias)
        self.assertEqual(linear.weight.shape, (5, 10))

    def test_linear_forward(self):
        """Test linear layer forward pass."""
        linear = mlx_compat.nn.Linear(10, 5)
        x = mlx_compat.randn(3, 10)  # batch_size=3, in_features=10

        output = linear(x)

        # Output should be [3, 5]
        self.assertEqual(output.shape, (3, 5))

    def test_linear_parameters(self):
        """Test that linear layer has correct parameters."""
        linear = mlx_compat.nn.Linear(10, 5)

        params = list(linear.parameters())
        self.assertEqual(len(params), 2)  # weight and bias

        # With no bias
        linear_no_bias = mlx_compat.nn.Linear(10, 5, bias=False)
        params_no_bias = list(linear_no_bias.parameters())
        self.assertEqual(len(params_no_bias), 1)  # only weight

    def test_linear_backward(self):
        """Test backward pass through linear layer."""
        linear = mlx_compat.nn.Linear(10, 5)
        x = mlx_compat.tensor(np.random.randn(3, 10).astype(np.float32), requires_grad=True)

        # Forward pass
        output = linear(x)
        loss = mlx_compat.sum(output)

        # Backward pass
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias.grad)
        self.assertIsNotNone(x.grad)

    def test_linear_gradient_shape(self):
        """Test that gradients have correct shapes."""
        linear = mlx_compat.nn.Linear(10, 5)
        x = mlx_compat.randn(3, 10, requires_grad=True)

        output = linear(x)
        loss = mlx_compat.sum(output)
        loss.backward()

        # Gradient shapes should match parameter shapes
        self.assertEqual(linear.weight.grad.shape, linear.weight.shape)
        self.assertEqual(linear.bias.grad.shape, linear.bias.shape)
        self.assertEqual(x.grad.shape, x.shape)

    def test_linear_initialization(self):
        """Test that linear layer weights are initialized reasonably."""
        linear = mlx_compat.nn.Linear(100, 50)

        # Weights should be in a reasonable range (Kaiming init)
        weight_std = float(linear.weight.std().numpy())
        expected_std = np.sqrt(1.0 / 100)  # Kaiming uniform approximation

        # Should be within 3x of expected (loose check for random init)
        self.assertLess(weight_std, expected_std * 3)
        self.assertGreater(weight_std, expected_std / 3)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
