"""
PyTorch Parity Tests

Tests numerical parity between flashlight and PyTorch implementations.
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import numpy as np

# Import test utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    import flashlight.nn as nn
    import flashlight.optim as optim
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def skip_if_no_torch(func):
    """Skip test if PyTorch is not available."""
    return unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")(func)


@skipIfNoMLX
@skip_if_no_torch
class TestLinearParity(TestCase):
    """Test Linear layer parity with PyTorch."""

    def test_linear_forward(self):
        """Test Linear layer forward pass matches PyTorch."""
        # Create layers
        torch_linear = torch_nn.Linear(10, 5)
        mlx_linear = nn.Linear(10, 5)

        # Copy weights from PyTorch to MLX
        mlx_linear.weight._mlx_array[:] = torch_linear.weight.detach().numpy()
        mlx_linear.bias._mlx_array[:] = torch_linear.bias.detach().numpy()

        # Random input
        x_np = np.random.randn(3, 10).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = flashlight.tensor(x_np)

        # Forward pass
        y_torch = torch_linear(x_torch)
        y_mlx = mlx_linear(x_mlx)

        # Check parity
        np.testing.assert_allclose(
            y_mlx.numpy(),
            y_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6
        )


@skipIfNoMLX
@skip_if_no_torch
class TestActivationParity(TestCase):
    """Test activation function parity with PyTorch."""

    def test_relu_forward(self):
        """Test ReLU matches PyTorch."""
        x_np = np.random.randn(4, 10).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = flashlight.tensor(x_np)

        y_torch = torch.relu(x_torch)
        y_mlx = flashlight.relu(x_mlx)

        np.testing.assert_allclose(
            y_mlx.numpy(),
            y_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6
        )

    def test_gelu_forward(self):
        """Test GELU matches PyTorch."""
        x_np = np.random.randn(4, 10).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = flashlight.tensor(x_np)

        y_torch = torch.nn.functional.gelu(x_torch)
        y_mlx = flashlight.gelu(x_mlx)

        np.testing.assert_allclose(
            y_mlx.numpy(),
            y_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5
        )


@skipIfNoMLX
@skip_if_no_torch
class TestConvParity(TestCase):
    """Test Conv2d layer parity with PyTorch."""

    def test_conv2d_forward(self):
        """Test Conv2d forward pass matches PyTorch."""
        # Create layers
        torch_conv = torch_nn.Conv2d(3, 16, kernel_size=3, padding=1)
        mlx_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # Copy weights - note the transpose for MLX format
        mlx_conv.weight._mlx_array[:] = torch_conv.weight.detach().numpy()
        if torch_conv.bias is not None:
            mlx_conv.bias._mlx_array[:] = torch_conv.bias.detach().numpy()

        # Random input
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = flashlight.tensor(x_np)

        # Forward pass
        y_torch = torch_conv(x_torch)
        y_mlx = mlx_conv(x_mlx)

        # Check parity (conv can have slightly larger tolerance)
        np.testing.assert_allclose(
            y_mlx.numpy(),
            y_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5
        )


@skipIfNoMLX
@skip_if_no_torch
class TestLossParity(TestCase):
    """Test loss function parity with PyTorch."""

    def test_mse_loss(self):
        """Test MSE loss matches PyTorch."""
        pred_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randn(4, 10).astype(np.float32)

        pred_torch = torch.from_numpy(pred_np)
        target_torch = torch.from_numpy(target_np)
        pred_mlx = flashlight.tensor(pred_np)
        target_mlx = flashlight.tensor(target_np)

        loss_torch = torch_nn.MSELoss()(pred_torch, target_torch)
        loss_mlx = nn.MSELoss()(pred_mlx, target_mlx)

        np.testing.assert_allclose(
            loss_mlx.numpy(),
            loss_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6
        )

    def test_cross_entropy_loss(self):
        """Test CrossEntropy loss matches PyTorch."""
        logits_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randint(0, 10, size=(4,)).astype(np.int32)

        logits_torch = torch.from_numpy(logits_np)
        target_torch = torch.from_numpy(target_np).long()
        logits_mlx = flashlight.tensor(logits_np)
        target_mlx = flashlight.tensor(target_np)

        loss_torch = torch_nn.CrossEntropyLoss()(logits_torch, target_torch)
        loss_mlx = nn.CrossEntropyLoss()(logits_mlx, target_mlx)

        np.testing.assert_allclose(
            loss_mlx.numpy(),
            loss_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )


@skipIfNoMLX
@skip_if_no_torch
class TestOptimizerParity(TestCase):
    """Test optimizer parity with PyTorch."""

    def test_sgd_step(self):
        """Test SGD optimizer step matches PyTorch."""
        # Create parameters
        param_np = np.random.randn(5, 3).astype(np.float32)
        grad_np = np.random.randn(5, 3).astype(np.float32)

        # PyTorch
        param_torch = torch.from_numpy(param_np.copy()).requires_grad_(True)
        param_torch.grad = torch.from_numpy(grad_np)
        opt_torch = torch_optim.SGD([param_torch], lr=0.01)
        opt_torch.step()

        # MLX
        param_mlx = nn.Parameter(flashlight.tensor(param_np.copy()))
        param_mlx.grad = flashlight.tensor(grad_np)
        opt_mlx = optim.SGD([param_mlx], lr=0.01)
        opt_mlx.step()

        # Check parity
        np.testing.assert_allclose(
            param_mlx.numpy(),
            param_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-6
        )

    def test_adam_step(self):
        """Test Adam optimizer step matches PyTorch."""
        # Create parameters
        param_np = np.random.randn(5, 3).astype(np.float32)
        grad_np = np.random.randn(5, 3).astype(np.float32)

        # PyTorch
        param_torch = torch.from_numpy(param_np.copy()).requires_grad_(True)
        param_torch.grad = torch.from_numpy(grad_np)
        opt_torch = torch_optim.Adam([param_torch], lr=0.001)
        opt_torch.step()

        # MLX
        param_mlx = nn.Parameter(flashlight.tensor(param_np.copy()))
        param_mlx.grad = flashlight.tensor(grad_np)
        opt_mlx = optim.Adam([param_mlx], lr=0.001)
        opt_mlx.step()

        # Check parity
        np.testing.assert_allclose(
            param_mlx.numpy(),
            param_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )


@skipIfNoMLX
@skip_if_no_torch
class TestModelParity(TestCase):
    """Test full model parity with PyTorch."""

    def test_simple_mlp_forward(self):
        """Test simple MLP forward pass matches PyTorch."""
        # Create models
        torch_model = torch_nn.Sequential(
            torch_nn.Linear(784, 128),
            torch_nn.ReLU(),
            torch_nn.Linear(128, 10)
        )

        mlx_model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # Copy weights
        mlx_model[0].weight._mlx_array[:] = torch_model[0].weight.detach().numpy()
        mlx_model[0].bias._mlx_array[:] = torch_model[0].bias.detach().numpy()
        mlx_model[2].weight._mlx_array[:] = torch_model[2].weight.detach().numpy()
        mlx_model[2].bias._mlx_array[:] = torch_model[2].bias.detach().numpy()

        # Random input
        x_np = np.random.randn(4, 784).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_mlx = flashlight.tensor(x_np)

        # Forward pass
        y_torch = torch_model(x_torch)
        y_mlx = mlx_model(x_mlx)

        # Check parity
        np.testing.assert_allclose(
            y_mlx.numpy(),
            y_torch.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )


if __name__ == '__main__':
    from common_utils import run_tests
    run_tests()
