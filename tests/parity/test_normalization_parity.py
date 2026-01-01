"""
Normalization Layer Parity Tests

Tests numerical parity between mlx_compat normalization layers and PyTorch.
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import mlx_compat
import mlx_compat.nn as nn


def copy_norm_weights(mlx_layer, torch_layer):
    """Copy weights from PyTorch normalization layer to mlx_compat."""
    if hasattr(torch_layer, 'weight') and torch_layer.weight is not None:
        mlx_layer.weight = nn.Parameter(
            mlx_compat.tensor(torch_layer.weight.detach().numpy())
        )
    if hasattr(torch_layer, 'bias') and torch_layer.bias is not None:
        mlx_layer.bias = nn.Parameter(
            mlx_compat.tensor(torch_layer.bias.detach().numpy())
        )
    # Copy running stats for BatchNorm
    if hasattr(torch_layer, 'running_mean') and torch_layer.running_mean is not None:
        mlx_layer.running_mean = mlx_compat.tensor(torch_layer.running_mean.detach().numpy())
    if hasattr(torch_layer, 'running_var') and torch_layer.running_var is not None:
        mlx_layer.running_var = mlx_compat.tensor(torch_layer.running_var.detach().numpy())


class TestBatchNormParity:
    """Test BatchNorm parity with PyTorch."""

    @pytest.mark.parity
    def test_batchnorm1d_parity(self):
        """Test BatchNorm1d matches PyTorch in eval mode."""
        batch, features = 8, 64

        torch_bn = torch.nn.BatchNorm1d(features)
        torch_bn.eval()
        mlx_bn = nn.BatchNorm1d(features)
        mlx_bn.eval()
        copy_norm_weights(mlx_bn, torch_bn)

        np.random.seed(42)
        x_np = np.random.randn(batch, features).astype(np.float32)

        torch_out = torch_bn(torch.tensor(x_np))
        mlx_out = mlx_bn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"BatchNorm1d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_batchnorm2d_parity(self):
        """Test BatchNorm2d matches PyTorch in eval mode."""
        batch, channels, h, w = 4, 32, 16, 16

        torch_bn = torch.nn.BatchNorm2d(channels)
        torch_bn.eval()
        mlx_bn = nn.BatchNorm2d(channels)
        mlx_bn.eval()
        copy_norm_weights(mlx_bn, torch_bn)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_bn(torch.tensor(x_np))
        mlx_out = mlx_bn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"BatchNorm2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_batchnorm2d_training_parity(self):
        """Test BatchNorm2d matches PyTorch in training mode."""
        batch, channels, h, w = 8, 16, 8, 8

        torch_bn = torch.nn.BatchNorm2d(channels)
        torch_bn.train()
        mlx_bn = nn.BatchNorm2d(channels)
        mlx_bn.train()
        copy_norm_weights(mlx_bn, torch_bn)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_bn(torch.tensor(x_np))
        mlx_out = mlx_bn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"BatchNorm2d training mismatch: {max_diff}"

    @pytest.mark.parity
    def test_batchnorm3d_parity(self):
        """Test BatchNorm3d matches PyTorch."""
        batch, channels, d, h, w = 2, 8, 4, 8, 8

        torch_bn = torch.nn.BatchNorm3d(channels)
        torch_bn.eval()
        mlx_bn = nn.BatchNorm3d(channels)
        mlx_bn.eval()
        copy_norm_weights(mlx_bn, torch_bn)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, d, h, w).astype(np.float32)

        torch_out = torch_bn(torch.tensor(x_np))
        mlx_out = mlx_bn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"BatchNorm3d mismatch: {max_diff}"


class TestLayerNormParity:
    """Test LayerNorm parity with PyTorch."""

    @pytest.mark.parity
    def test_layernorm_1d_parity(self):
        """Test LayerNorm with 1D normalized_shape."""
        batch, seq, features = 4, 10, 64

        torch_ln = torch.nn.LayerNorm(features)
        mlx_ln = nn.LayerNorm(features)
        copy_norm_weights(mlx_ln, torch_ln)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq, features).astype(np.float32)

        torch_out = torch_ln(torch.tensor(x_np))
        mlx_out = mlx_ln(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"LayerNorm 1D mismatch: {max_diff}"

    @pytest.mark.parity
    def test_layernorm_2d_parity(self):
        """Test LayerNorm with 2D normalized_shape."""
        batch, h, w = 4, 10, 64

        torch_ln = torch.nn.LayerNorm([h, w])
        mlx_ln = nn.LayerNorm([h, w])
        copy_norm_weights(mlx_ln, torch_ln)

        np.random.seed(42)
        x_np = np.random.randn(batch, h, w).astype(np.float32)

        torch_out = torch_ln(torch.tensor(x_np))
        mlx_out = mlx_ln(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"LayerNorm 2D mismatch: {max_diff}"

    @pytest.mark.parity
    def test_layernorm_different_eps(self):
        """Test LayerNorm with different epsilon values."""
        batch, features = 4, 64

        for eps in [1e-5, 1e-6, 1e-3]:
            torch_ln = torch.nn.LayerNorm(features, eps=eps)
            mlx_ln = nn.LayerNorm(features, eps=eps)
            copy_norm_weights(mlx_ln, torch_ln)

            np.random.seed(42)
            x_np = np.random.randn(batch, features).astype(np.float32)

            torch_out = torch_ln(torch.tensor(x_np))
            mlx_out = mlx_ln(mlx_compat.tensor(x_np))

            max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
            assert max_diff < 1e-5, f"LayerNorm eps={eps} mismatch: {max_diff}"


class TestGroupNormParity:
    """Test GroupNorm parity with PyTorch."""

    @pytest.mark.parity
    def test_groupnorm_parity(self):
        """Test GroupNorm matches PyTorch."""
        batch, channels, h, w = 4, 32, 16, 16
        num_groups = 8

        torch_gn = torch.nn.GroupNorm(num_groups, channels)
        mlx_gn = nn.GroupNorm(num_groups, channels)
        copy_norm_weights(mlx_gn, torch_gn)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_gn(torch.tensor(x_np))
        mlx_out = mlx_gn(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"GroupNorm mismatch: {max_diff}"

    @pytest.mark.parity
    def test_groupnorm_different_groups(self):
        """Test GroupNorm with different group counts."""
        batch, channels, h, w = 4, 32, 8, 8

        for num_groups in [1, 4, 8, 32]:
            torch_gn = torch.nn.GroupNorm(num_groups, channels)
            mlx_gn = nn.GroupNorm(num_groups, channels)
            copy_norm_weights(mlx_gn, torch_gn)

            np.random.seed(42)
            x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

            torch_out = torch_gn(torch.tensor(x_np))
            mlx_out = mlx_gn(mlx_compat.tensor(x_np))

            max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
            assert max_diff < 1e-5, f"GroupNorm groups={num_groups} mismatch: {max_diff}"


class TestInstanceNormParity:
    """Test InstanceNorm parity with PyTorch."""

    @pytest.mark.parity
    def test_instancenorm2d_parity(self):
        """Test InstanceNorm2d matches PyTorch."""
        batch, channels, h, w = 4, 16, 16, 16

        torch_in = torch.nn.InstanceNorm2d(channels, affine=True)
        mlx_in = nn.InstanceNorm2d(channels, affine=True)
        copy_norm_weights(mlx_in, torch_in)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, h, w).astype(np.float32)

        torch_out = torch_in(torch.tensor(x_np))
        mlx_out = mlx_in(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"InstanceNorm2d mismatch: {max_diff}"

    @pytest.mark.parity
    def test_instancenorm1d_parity(self):
        """Test InstanceNorm1d matches PyTorch."""
        batch, channels, length = 4, 16, 32

        torch_in = torch.nn.InstanceNorm1d(channels, affine=True)
        mlx_in = nn.InstanceNorm1d(channels, affine=True)
        copy_norm_weights(mlx_in, torch_in)

        np.random.seed(42)
        x_np = np.random.randn(batch, channels, length).astype(np.float32)

        torch_out = torch_in(torch.tensor(x_np))
        mlx_out = mlx_in(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"InstanceNorm1d mismatch: {max_diff}"


class TestRMSNormParity:
    """Test RMSNorm parity (PyTorch added RMSNorm in 2.4+)."""

    @pytest.mark.parity
    def test_rmsnorm_manual_parity(self):
        """Test RMSNorm matches manual PyTorch implementation."""
        batch, features = 4, 64

        mlx_rms = nn.RMSNorm(features)

        np.random.seed(42)
        x_np = np.random.randn(batch, features).astype(np.float32)

        # Manual RMSNorm in PyTorch
        x_torch = torch.tensor(x_np)
        rms = torch.sqrt(torch.mean(x_torch ** 2, dim=-1, keepdim=True) + 1e-5)
        torch_out = x_torch / rms

        mlx_out = mlx_rms(mlx_compat.tensor(x_np))

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-4, f"RMSNorm mismatch: {max_diff}"
