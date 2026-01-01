"""
Convolution Layer Parity Tests

Tests numerical parity between flashlight conv layers and PyTorch.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import flashlight
import flashlight.nn as nn


def copy_conv_weights(mlx_layer, torch_layer):
    """Copy weights from PyTorch conv layer to flashlight."""
    mlx_layer.weight = nn.Parameter(
        flashlight.tensor(torch_layer.weight.detach().numpy())
    )
    if torch_layer.bias is not None:
        mlx_layer.bias = nn.Parameter(
            flashlight.tensor(torch_layer.bias.detach().numpy())
        )


class TestConv1dParity:
    """Test Conv1d parity with PyTorch."""

    @pytest.mark.parity
    def test_conv1d_basic_parity(self):
        """Test basic Conv1d matches PyTorch."""
        batch, in_ch, length = 2, 3, 16
        out_ch, kernel = 8, 3

        torch_conv = torch.nn.Conv1d(in_ch, out_ch, kernel)
        mlx_conv = nn.Conv1d(in_ch, out_ch, kernel)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, length).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"Conv1d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_conv1d_with_stride_padding(self):
        """Test Conv1d with stride and padding."""
        batch, in_ch, length = 2, 4, 32
        out_ch, kernel, stride, padding = 8, 5, 2, 2

        torch_conv = torch.nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        mlx_conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, length).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"Conv1d stride/padding mismatch: {max_diff}"


class TestConv2dParity:
    """Test Conv2d parity with PyTorch."""

    @pytest.mark.parity
    def test_conv2d_basic_parity(self):
        """Test basic Conv2d matches PyTorch."""
        batch, in_ch, h, w = 2, 3, 16, 16
        out_ch, kernel = 8, 3

        torch_conv = torch.nn.Conv2d(in_ch, out_ch, kernel)
        mlx_conv = nn.Conv2d(in_ch, out_ch, kernel)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"Conv2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_conv2d_with_stride_padding(self):
        """Test Conv2d with stride and padding."""
        batch, in_ch, h, w = 2, 4, 32, 32
        out_ch, kernel, stride, padding = 16, 3, 2, 1

        torch_conv = torch.nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        mlx_conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"Conv2d stride/padding mismatch: {max_diff}"

    @pytest.mark.parity
    def test_conv2d_asymmetric_kernel(self):
        """Test Conv2d with asymmetric kernel."""
        batch, in_ch, h, w = 2, 3, 16, 16
        out_ch, kernel = 8, (3, 5)

        torch_conv = torch.nn.Conv2d(in_ch, out_ch, kernel)
        mlx_conv = nn.Conv2d(in_ch, out_ch, kernel)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"Conv2d asymmetric kernel mismatch: {max_diff}"

    @pytest.mark.parity
    def test_conv2d_groups(self):
        """Test Conv2d with groups (depthwise-separable style)."""
        batch, in_ch, h, w = 2, 8, 16, 16
        out_ch, kernel, groups = 8, 3, 4

        torch_conv = torch.nn.Conv2d(in_ch, out_ch, kernel, groups=groups)
        mlx_conv = nn.Conv2d(in_ch, out_ch, kernel, groups=groups)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"Conv2d groups mismatch: {max_diff}"


class TestConv3dParity:
    """Test Conv3d parity with PyTorch."""

    @pytest.mark.parity
    def test_conv3d_basic_parity(self):
        """Test basic Conv3d matches PyTorch."""
        batch, in_ch, d, h, w = 1, 2, 8, 8, 8
        out_ch, kernel = 4, 3

        torch_conv = torch.nn.Conv3d(in_ch, out_ch, kernel)
        mlx_conv = nn.Conv3d(in_ch, out_ch, kernel)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, d, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-4, f"Conv3d mismatch: {max_diff}"


class TestConvTransposeParity:
    """Test ConvTranspose parity with PyTorch."""

    @pytest.mark.parity
    def test_conv_transpose2d_basic_parity(self):
        """Test basic ConvTranspose2d matches PyTorch."""
        batch, in_ch, h, w = 2, 8, 8, 8
        out_ch, kernel = 4, 3

        torch_conv = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel)
        mlx_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"ConvTranspose2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_conv_transpose2d_stride(self):
        """Test ConvTranspose2d with stride (upsampling)."""
        batch, in_ch, h, w = 2, 8, 8, 8
        out_ch, kernel, stride = 4, 4, 2

        torch_conv = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride)
        mlx_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride)
        copy_conv_weights(mlx_conv, torch_conv)

        np.random.seed(42)
        x_np = np.random.randn(batch, in_ch, h, w).astype(np.float32)

        torch_out = torch_conv(torch.tensor(x_np))
        mlx_out = mlx_conv(flashlight.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"ConvTranspose2d stride mismatch: {max_diff}"
