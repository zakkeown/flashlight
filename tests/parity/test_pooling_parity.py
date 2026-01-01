"""
Pooling Layer Parity Tests

Tests numerical parity between mlx_compat pooling layers and PyTorch.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import mlx_compat
import mlx_compat.nn as nn


class TestMaxPoolParity:
    """Test MaxPool parity with PyTorch."""

    @pytest.mark.parity
    def test_maxpool1d_basic_parity(self):
        """Test MaxPool1d matches PyTorch."""
        batch, channels, length = 2, 4, 16
        kernel_size = 2

        torch_pool = torch.nn.MaxPool1d(kernel_size)
        mlx_pool = nn.MaxPool1d(kernel_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-6, f"MaxPool1d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_maxpool2d_basic_parity(self):
        """Test MaxPool2d matches PyTorch."""
        batch, channels, h, w = 2, 4, 16, 16
        kernel_size = 2

        torch_pool = torch.nn.MaxPool2d(kernel_size)
        mlx_pool = nn.MaxPool2d(kernel_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-6, f"MaxPool2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_maxpool2d_stride_padding(self):
        """Test MaxPool2d with stride and padding."""
        batch, channels, h, w = 2, 4, 32, 32
        kernel_size, stride, padding = 3, 2, 1

        torch_pool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        mlx_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-6, f"MaxPool2d stride/padding mismatch: {max_diff}"

    @pytest.mark.parity
    def test_maxpool3d_basic_parity(self):
        """Test MaxPool3d matches PyTorch."""
        batch, channels, d, h, w = 1, 2, 8, 8, 8
        kernel_size = 2

        torch_pool = torch.nn.MaxPool3d(kernel_size)
        mlx_pool = nn.MaxPool3d(kernel_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, d, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-6, f"MaxPool3d mismatch: {max_diff}"


class TestAvgPoolParity:
    """Test AvgPool parity with PyTorch."""

    @pytest.mark.parity
    def test_avgpool1d_basic_parity(self):
        """Test AvgPool1d matches PyTorch."""
        batch, channels, length = 2, 4, 16
        kernel_size = 2

        torch_pool = torch.nn.AvgPool1d(kernel_size)
        mlx_pool = nn.AvgPool1d(kernel_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"AvgPool1d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_avgpool2d_basic_parity(self):
        """Test AvgPool2d matches PyTorch."""
        batch, channels, h, w = 2, 4, 16, 16
        kernel_size = 2

        torch_pool = torch.nn.AvgPool2d(kernel_size)
        mlx_pool = nn.AvgPool2d(kernel_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"AvgPool2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_avgpool2d_count_include_pad(self):
        """Test AvgPool2d with count_include_pad=False."""
        batch, channels, h, w = 2, 4, 16, 16
        kernel_size, padding = 3, 1

        torch_pool = torch.nn.AvgPool2d(kernel_size, padding=padding, count_include_pad=False)
        mlx_pool = nn.AvgPool2d(kernel_size, padding=padding, count_include_pad=False)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"AvgPool2d count_include_pad mismatch: {max_diff}"


class TestAdaptivePoolParity:
    """Test AdaptivePool parity with PyTorch."""

    @pytest.mark.parity
    def test_adaptive_avgpool2d_parity(self):
        """Test AdaptiveAvgPool2d matches PyTorch."""
        batch, channels, h, w = 2, 4, 16, 16
        output_size = (4, 4)

        torch_pool = torch.nn.AdaptiveAvgPool2d(output_size)
        mlx_pool = nn.AdaptiveAvgPool2d(output_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"AdaptiveAvgPool2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_adaptive_avgpool2d_single_output(self):
        """Test AdaptiveAvgPool2d with output_size=1 (global pooling)."""
        batch, channels, h, w = 2, 64, 7, 7
        output_size = 1

        torch_pool = torch.nn.AdaptiveAvgPool2d(output_size)
        mlx_pool = nn.AdaptiveAvgPool2d(output_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"AdaptiveAvgPool2d global mismatch: {max_diff}"

    @pytest.mark.parity
    def test_adaptive_maxpool2d_parity(self):
        """Test AdaptiveMaxPool2d matches PyTorch."""
        batch, channels, h, w = 2, 4, 16, 16
        output_size = (4, 4)

        torch_pool = torch.nn.AdaptiveMaxPool2d(output_size)
        mlx_pool = nn.AdaptiveMaxPool2d(output_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-6, f"AdaptiveMaxPool2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_adaptive_avgpool1d_parity(self):
        """Test AdaptiveAvgPool1d matches PyTorch."""
        batch, channels, length = 2, 4, 32
        output_size = 8

        torch_pool = torch.nn.AdaptiveAvgPool1d(output_size)
        mlx_pool = nn.AdaptiveAvgPool1d(output_size)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        torch_out = torch_pool(torch.tensor(x_np))
        mlx_out = mlx_pool(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"AdaptiveAvgPool1d mismatch: {max_diff}"
