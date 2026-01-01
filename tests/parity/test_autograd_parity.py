"""
Test Autograd Parity with PyTorch

Comprehensive numerical parity tests comparing flashlight autograd
to PyTorch autograd. All tests verify gradient computation matches
within tolerance (1e-5 by default, 1e-4 for some operations).
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX, skipIfNoTorch

try:
    import flashlight
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestMaxBackwardParity(TestCase):
    """Test max backward parity with PyTorch."""

    def test_max_global_parity(self):
        """Test global max gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_y = torch_x.max()
        torch_y.backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        mlx_y = mlx_x.max()
        mlx_y.backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_max_dim_parity(self):
        """Test max along dimension gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(3, 4, 5).astype(np.float32)

        for dim in [0, 1, 2]:
            # PyTorch
            torch_x = torch.tensor(data, requires_grad=True)
            torch_y = torch_x.max(dim=dim).values
            torch_y.sum().backward()

            # MLX Compat
            mlx_x = flashlight.tensor(data, requires_grad=True)
            mlx_y = mlx_x.max(dim=dim).values
            flashlight.sum(mlx_y).backward()

            np.testing.assert_allclose(
                mlx_x.grad.numpy(), torch_x.grad.numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Max dim={dim} gradient mismatch"
            )

    def test_max_ties_parity(self):
        """Test max gradient with ties - gradient distributed equally among tied values."""
        import torch

        # Data with ties
        data = np.array([[1.0, 3.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_x.max().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        mlx_x.max().backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestSumMeanBackwardParity(TestCase):
    """Test sum/mean backward parity with PyTorch."""

    def test_sum_global_parity(self):
        """Test global sum gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_x.sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_x).backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_sum_dim_parity(self):
        """Test sum along dimension gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(3, 4, 5).astype(np.float32)

        for dim in [0, 1, 2]:
            # PyTorch
            torch_x = torch.tensor(data, requires_grad=True)
            torch_x.sum(dim=dim).sum().backward()

            # MLX Compat
            mlx_x = flashlight.tensor(data, requires_grad=True)
            flashlight.sum(flashlight.sum(mlx_x, dim=dim)).backward()

            np.testing.assert_allclose(
                mlx_x.grad.numpy(), torch_x.grad.numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Sum dim={dim} gradient mismatch"
            )

    def test_mean_parity(self):
        """Test mean gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_x.mean().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.mean(mlx_x).backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestArithmeticBackwardParity(TestCase):
    """Test arithmetic backward parity with PyTorch."""

    def test_add_parity(self):
        """Test add gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)

        # PyTorch
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)
        (torch_a + torch_b).sum().backward()

        # MLX Compat
        mlx_a = flashlight.tensor(a, requires_grad=True)
        mlx_b = flashlight.tensor(b, requires_grad=True)
        flashlight.sum(mlx_a + mlx_b).backward()

        np.testing.assert_allclose(mlx_a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mlx_b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_mul_parity(self):
        """Test mul gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)

        # PyTorch
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)
        (torch_a * torch_b).sum().backward()

        # MLX Compat
        mlx_a = flashlight.tensor(a, requires_grad=True)
        mlx_b = flashlight.tensor(b, requires_grad=True)
        flashlight.sum(mlx_a * mlx_b).backward()

        np.testing.assert_allclose(mlx_a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mlx_b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_matmul_parity(self):
        """Test matmul gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)
        (torch_a @ torch_b).sum().backward()

        # MLX Compat
        mlx_a = flashlight.tensor(a, requires_grad=True)
        mlx_b = flashlight.tensor(b, requires_grad=True)
        flashlight.sum(mlx_a @ mlx_b).backward()

        np.testing.assert_allclose(mlx_a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mlx_b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_pow_parity(self):
        """Test pow gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.abs(np.random.randn(3, 4).astype(np.float32)) + 0.1

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        (torch_x ** 2.5).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_x ** 2.5).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-4, atol=1e-4)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestActivationBackwardParity(TestCase):
    """Test activation backward parity with PyTorch."""

    def test_relu_parity(self):
        """Test ReLU gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.relu(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.relu(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_sigmoid_parity(self):
        """Test sigmoid gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.sigmoid(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.sigmoid(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_tanh_parity(self):
        """Test tanh gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.tanh(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.tanh(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_softmax_parity(self):
        """Test softmax gradient matches PyTorch."""
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        F.softmax(torch_x, dim=-1).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.softmax(mlx_x, dim=-1)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_gelu_parity(self):
        """Test GELU gradient matches PyTorch."""
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch with tanh approximation
        torch_x = torch.tensor(data, requires_grad=True)
        F.gelu(torch_x, approximate='tanh').sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.gelu(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-4, atol=1e-4)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestTrigBackwardParity(TestCase):
    """Test trigonometric backward parity with PyTorch."""

    def test_sin_parity(self):
        """Test sin gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.sin(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.sin(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_cos_parity(self):
        """Test cos gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.cos(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.cos(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestExpLogBackwardParity(TestCase):
    """Test exp/log backward parity with PyTorch."""

    def test_exp_parity(self):
        """Test exp gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32) * 0.5  # Avoid large values

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.exp(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.exp(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_log_parity(self):
        """Test log gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.abs(np.random.randn(4, 5).astype(np.float32)) + 0.1

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.log(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.log(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestTransposeBackwardParity(TestCase):
    """Test transpose backward parity with PyTorch."""

    def test_transpose_2d_parity(self):
        """Test 2D transpose gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_x.transpose(0, 1).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_x.transpose(0, 1)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_transpose_3d_parity(self):
        """Test 3D transpose gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(2, 3, 4).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_x.transpose(0, 2).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_x.transpose(0, 2)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestConv2dBackwardParity(TestCase):
    """Test Conv2d backward parity with PyTorch."""

    def test_conv2d_parity(self):
        """Test Conv2d gradient matches PyTorch."""
        import torch
        import torch.nn as torch_nn

        np.random.seed(42)
        data = np.random.randn(1, 3, 8, 8).astype(np.float32)
        weight = np.random.randn(4, 3, 3, 3).astype(np.float32)
        bias = np.random.randn(4).astype(np.float32)

        # PyTorch
        torch_conv = torch_nn.Conv2d(3, 4, kernel_size=3, padding=1)
        with torch.no_grad():
            torch_conv.weight.copy_(torch.tensor(weight))
            torch_conv.bias.copy_(torch.tensor(bias))

        torch_x = torch.tensor(data, requires_grad=True)
        torch_conv(torch_x).sum().backward()

        # MLX Compat
        mlx_conv = flashlight.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        with flashlight.no_grad():
            mlx_conv.weight = flashlight.tensor(weight)
            mlx_conv.bias = flashlight.tensor(bias)

        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_conv(mlx_x)).backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestComplexGraphParity(TestCase):
    """Test complex computation graph backward parity with PyTorch."""

    def test_multi_path_parity(self):
        """Test gradient accumulation from multiple paths."""
        import torch

        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)

        # PyTorch: y = x * x + x
        torch_x = torch.tensor(data, requires_grad=True)
        torch_y = torch_x * torch_x + torch_x
        torch_y.sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        mlx_y = mlx_x * mlx_x + mlx_x
        flashlight.sum(mlx_y).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_chain_ops_parity(self):
        """Test chained operations gradient."""
        import torch

        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)

        # PyTorch: y = relu(sigmoid(x * 2))
        torch_x = torch.tensor(data, requires_grad=True)
        torch_y = torch.relu(torch.sigmoid(torch_x * 2))
        torch_y.sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        mlx_y = flashlight.relu(flashlight.sigmoid(mlx_x * 2))
        flashlight.sum(mlx_y).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestSubDivBackwardParity(TestCase):
    """Test subtraction and division backward parity with PyTorch."""

    def test_sub_parity(self):
        """Test sub gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(3, 4).astype(np.float32)

        # PyTorch
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)
        (torch_a - torch_b).sum().backward()

        # MLX Compat
        mlx_a = flashlight.tensor(a, requires_grad=True)
        mlx_b = flashlight.tensor(b, requires_grad=True)
        flashlight.sum(mlx_a - mlx_b).backward()

        np.testing.assert_allclose(mlx_a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mlx_b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_div_parity(self):
        """Test div gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.abs(np.random.randn(3, 4).astype(np.float32)) + 0.5  # Avoid division by zero

        # PyTorch
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)
        (torch_a / torch_b).sum().backward()

        # MLX Compat
        mlx_a = flashlight.tensor(a, requires_grad=True)
        mlx_b = flashlight.tensor(b, requires_grad=True)
        flashlight.sum(mlx_a / mlx_b).backward()

        np.testing.assert_allclose(mlx_a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mlx_b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestSqrtAbsNegBackwardParity(TestCase):
    """Test sqrt, abs, neg backward parity with PyTorch."""

    def test_sqrt_parity(self):
        """Test sqrt gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.abs(np.random.randn(4, 5).astype(np.float32)) + 0.1

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.sqrt(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.sqrt(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_abs_parity(self):
        """Test abs gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        # Avoid zero where gradient is undefined
        data = np.random.randn(4, 5).astype(np.float32)
        data = np.where(np.abs(data) < 0.1, 0.2 * np.sign(data), data)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch.abs(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.abs(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_neg_parity(self):
        """Test negation gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        (-torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(-mlx_x).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestMoreActivationBackwardParity(TestCase):
    """Test additional activation backward parity with PyTorch."""

    def test_log_softmax_parity(self):
        """Test log_softmax gradient matches PyTorch."""
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        F.log_softmax(torch_x, dim=-1).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.log_softmax(mlx_x, dim=-1)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_silu_parity(self):
        """Test SiLU/Swish gradient matches PyTorch."""
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        F.silu(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.silu(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_leaky_relu_parity(self):
        """Test leaky_relu gradient matches PyTorch."""
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        F.leaky_relu(torch_x, negative_slope=0.01).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.leaky_relu(mlx_x, negative_slope=0.01)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_elu_parity(self):
        """Test ELU gradient matches PyTorch."""
        import torch
        import torch.nn.functional as F

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        F.elu(torch_x, alpha=1.0).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(flashlight.elu(mlx_x, alpha=1.0)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestShapeBackwardParity(TestCase):
    """Test shape operation backward parity with PyTorch."""

    def test_cat_parity(self):
        """Test cat gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        a = np.random.randn(2, 3).astype(np.float32)
        b = np.random.randn(2, 3).astype(np.float32)

        # PyTorch
        torch_a = torch.tensor(a, requires_grad=True)
        torch_b = torch.tensor(b, requires_grad=True)
        torch.cat([torch_a, torch_b], dim=0).sum().backward()

        # MLX Compat
        mlx_a = flashlight.tensor(a, requires_grad=True)
        mlx_b = flashlight.tensor(b, requires_grad=True)
        flashlight.sum(flashlight.cat([mlx_a, mlx_b], dim=0)).backward()

        np.testing.assert_allclose(mlx_a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mlx_b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5)

    def test_view_parity(self):
        """Test view/reshape gradient matches PyTorch."""
        import torch

        np.random.seed(42)
        data = np.random.randn(2, 3, 4).astype(np.float32)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        torch_x.view(6, 4).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_x.view(6, 4)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestPoolingBackwardParity(TestCase):
    """Test pooling operation backward parity with PyTorch."""

    def test_max_pool2d_parity(self):
        """Test MaxPool2d gradient matches PyTorch."""
        import torch
        import torch.nn as torch_nn

        np.random.seed(42)
        data = np.random.randn(1, 2, 6, 6).astype(np.float32)

        # PyTorch
        torch_pool = torch_nn.MaxPool2d(kernel_size=2, stride=2)
        torch_x = torch.tensor(data, requires_grad=True)
        torch_pool(torch_x).sum().backward()

        # MLX Compat
        mlx_pool = flashlight.nn.MaxPool2d(kernel_size=2, stride=2)
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_pool(mlx_x)).backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_avg_pool2d_parity(self):
        """Test AvgPool2d gradient matches PyTorch."""
        import torch
        import torch.nn as torch_nn

        np.random.seed(42)
        data = np.random.randn(1, 2, 6, 6).astype(np.float32)

        # PyTorch
        torch_pool = torch_nn.AvgPool2d(kernel_size=2, stride=2)
        torch_x = torch.tensor(data, requires_grad=True)
        torch_pool(torch_x).sum().backward()

        # MLX Compat
        mlx_pool = flashlight.nn.AvgPool2d(kernel_size=2, stride=2)
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(mlx_pool(mlx_x)).backward()

        np.testing.assert_allclose(
            mlx_x.grad.numpy(), torch_x.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestEmbeddingBackwardParity(TestCase):
    """Test Embedding backward parity with PyTorch."""

    def test_embedding_parity(self):
        """Test Embedding gradient matches PyTorch."""
        import torch
        import torch.nn as torch_nn

        np.random.seed(42)
        weight = np.random.randn(10, 5).astype(np.float32)
        indices = np.array([1, 3, 5, 3, 1])

        # PyTorch
        torch_emb = torch_nn.Embedding(10, 5)
        with torch.no_grad():
            torch_emb.weight.copy_(torch.tensor(weight))
        torch_indices = torch.tensor(indices, dtype=torch.long)
        torch_out = torch_emb(torch_indices)
        torch_out.sum().backward()

        # MLX Compat - use Parameter to ensure requires_grad=True
        from flashlight.nn.parameter import Parameter
        mlx_emb = flashlight.nn.Embedding(10, 5)
        mlx_emb.weight = Parameter(flashlight.tensor(weight, requires_grad=True))
        mlx_indices = flashlight.tensor(indices, dtype=flashlight.int32)
        mlx_out = mlx_emb(mlx_indices)
        flashlight.sum(mlx_out).backward()

        np.testing.assert_allclose(
            mlx_emb.weight.grad.numpy(), torch_emb.weight.grad.numpy(),
            rtol=1e-5, atol=1e-5
        )


@skipIfNoMLX
@skipIfNoTorch
@pytest.mark.parity
class TestCustomFunctionBackwardParity(TestCase):
    """Test custom Function backward parity with PyTorch."""

    def test_custom_function_parity(self):
        """Test custom Function gradient matches PyTorch custom function."""
        import torch
        from flashlight.autograd import Function

        np.random.seed(42)
        data = np.random.randn(4, 5).astype(np.float32)

        # PyTorch custom function
        class TorchSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x ** 2

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return grad_output * 2 * x

        # MLX Compat custom function
        class MLXSquare(Function):
            @staticmethod
            def forward(ctx, x):
                import mlx.core as mx
                ctx.save_for_backward(x)
                return x._mlx_array ** 2

            @staticmethod
            def backward(ctx, grad_output):
                import mlx.core as mx
                x, = ctx.saved_tensors
                return flashlight.tensor(grad_output._mlx_array * 2 * x._mlx_array)

        # PyTorch
        torch_x = torch.tensor(data, requires_grad=True)
        TorchSquare.apply(torch_x).sum().backward()

        # MLX Compat
        mlx_x = flashlight.tensor(data, requires_grad=True)
        flashlight.sum(MLXSquare.apply(mlx_x)).backward()

        np.testing.assert_allclose(mlx_x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
