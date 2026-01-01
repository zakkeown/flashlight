"""
Parity tests for nn.functional APIs.

Tests that flashlight.nn.functional produces identical outputs to torch.nn.functional.
"""

import numpy as np
import pytest

# Check if PyTorch is available
try:
    import torch
    import torch.nn.functional as F_torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import flashlight
import flashlight.nn.functional as F_mlx


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestActivationParity:
    """Parity tests for activation functions."""

    def test_relu_parity(self):
        """Test relu produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.relu(x_torch)
        out_mlx = F_mlx.relu(x_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_relu_negative_values(self):
        """Test relu correctly zeros negative values."""
        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.relu(x_torch)
        out_mlx = F_mlx.relu(x_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_gelu_parity(self):
        """Test gelu produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.gelu(x_torch)
        out_mlx = F_mlx.gelu(x_mlx)

        # GELU has slight numerical differences, use looser tolerance
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)

    def test_sigmoid_parity(self):
        """Test sigmoid produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.sigmoid(x_torch)
        out_mlx = F_mlx.sigmoid(x_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_tanh_parity(self):
        """Test tanh produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.tanh(x_torch)
        out_mlx = F_mlx.tanh(x_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_softmax_parity(self):
        """Test softmax produces matching outputs."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.softmax(x_torch, dim=-1)
        out_mlx = F_mlx.softmax(x_mlx, dim=-1)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_softmax_sums_to_one(self):
        """Test softmax outputs sum to 1."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        out_mlx = F_mlx.softmax(x_mlx, dim=-1)
        sums = out_mlx.numpy().sum(axis=-1)

        np.testing.assert_allclose(sums, np.ones(5), rtol=1e-5)

    def test_log_softmax_parity(self):
        """Test log_softmax produces matching outputs."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.log_softmax(x_torch, dim=-1)
        out_mlx = F_mlx.log_softmax(x_mlx, dim=-1)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_silu_parity(self):
        """Test silu (swish) produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.silu(x_torch)
        out_mlx = F_mlx.silu(x_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_leaky_relu_parity(self):
        """Test leaky_relu produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        for negative_slope in [0.01, 0.1, 0.3]:
            out_torch = F_torch.leaky_relu(x_torch, negative_slope=negative_slope)
            out_mlx = F_mlx.leaky_relu(x_mlx, negative_slope=negative_slope)

            np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_elu_parity(self):
        """Test elu produces matching outputs."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        for alpha in [0.5, 1.0, 2.0]:
            out_torch = F_torch.elu(x_torch, alpha=alpha)
            out_mlx = F_mlx.elu(x_mlx, alpha=alpha)

            np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestLinearParity:
    """Parity tests for linear operations."""

    def test_linear_parity(self):
        """Test linear produces matching outputs."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        w_np = np.random.randn(20, 10).astype(np.float32)
        b_np = np.random.randn(20).astype(np.float32)

        x_torch = torch.tensor(x_np)
        w_torch = torch.tensor(w_np)
        b_torch = torch.tensor(b_np)

        x_mlx = flashlight.tensor(x_np)
        w_mlx = flashlight.tensor(w_np)
        b_mlx = flashlight.tensor(b_np)

        out_torch = F_torch.linear(x_torch, w_torch, b_torch)
        out_mlx = F_mlx.linear(x_mlx, w_mlx, b_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_linear_no_bias_parity(self):
        """Test linear without bias produces matching outputs."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        w_np = np.random.randn(20, 10).astype(np.float32)

        x_torch = torch.tensor(x_np)
        w_torch = torch.tensor(w_np)

        x_mlx = flashlight.tensor(x_np)
        w_mlx = flashlight.tensor(w_np)

        out_torch = F_torch.linear(x_torch, w_torch)
        out_mlx = F_mlx.linear(x_mlx, w_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestNormalizationParity:
    """Parity tests for normalization operations."""

    def test_layer_norm_parity(self):
        """Test layer_norm produces matching outputs."""
        x_np = np.random.randn(2, 5, 10).astype(np.float32)
        w_np = np.random.randn(10).astype(np.float32)
        b_np = np.random.randn(10).astype(np.float32)

        x_torch = torch.tensor(x_np)
        w_torch = torch.tensor(w_np)
        b_torch = torch.tensor(b_np)

        x_mlx = flashlight.tensor(x_np)
        w_mlx = flashlight.tensor(w_np)
        b_mlx = flashlight.tensor(b_np)

        out_torch = F_torch.layer_norm(x_torch, [10], weight=w_torch, bias=b_torch)
        out_mlx = F_mlx.layer_norm(x_mlx, [10], weight=w_mlx, bias=b_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)

    def test_layer_norm_no_affine_parity(self):
        """Test layer_norm without affine parameters."""
        x_np = np.random.randn(2, 5, 10).astype(np.float32)

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.layer_norm(x_torch, [10])
        out_mlx = F_mlx.layer_norm(x_mlx, [10])

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestLossFunctionParity:
    """Parity tests for loss functions."""

    def test_mse_loss_parity(self):
        """Test mse_loss produces matching outputs."""
        pred_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randn(10, 5).astype(np.float32)

        pred_torch = torch.tensor(pred_np)
        target_torch = torch.tensor(target_np)

        pred_mlx = flashlight.tensor(pred_np)
        target_mlx = flashlight.tensor(target_np)

        for reduction in ["mean", "sum", "none"]:
            out_torch = F_torch.mse_loss(pred_torch, target_torch, reduction=reduction)
            out_mlx = F_mlx.mse_loss(pred_mlx, target_mlx, reduction=reduction)

            np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_l1_loss_parity(self):
        """Test l1_loss produces matching outputs."""
        pred_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randn(10, 5).astype(np.float32)

        pred_torch = torch.tensor(pred_np)
        target_torch = torch.tensor(target_np)

        pred_mlx = flashlight.tensor(pred_np)
        target_mlx = flashlight.tensor(target_np)

        for reduction in ["mean", "sum", "none"]:
            out_torch = F_torch.l1_loss(pred_torch, target_torch, reduction=reduction)
            out_mlx = F_mlx.l1_loss(pred_mlx, target_mlx, reduction=reduction)

            np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_cross_entropy_parity(self):
        """Test cross_entropy produces matching outputs."""
        logits_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randint(0, 5, size=(10,)).astype(np.int64)

        logits_torch = torch.tensor(logits_np)
        target_torch = torch.tensor(target_np)

        logits_mlx = flashlight.tensor(logits_np)
        target_mlx = flashlight.tensor(target_np)

        out_torch = F_torch.cross_entropy(logits_torch, target_torch)
        out_mlx = F_mlx.cross_entropy(logits_mlx, target_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)

    def test_binary_cross_entropy_with_logits_parity(self):
        """Test binary_cross_entropy_with_logits produces matching outputs."""
        logits_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randint(0, 2, size=(10, 5)).astype(np.float32)

        logits_torch = torch.tensor(logits_np)
        target_torch = torch.tensor(target_np)

        logits_mlx = flashlight.tensor(logits_np)
        target_mlx = flashlight.tensor(target_np)

        out_torch = F_torch.binary_cross_entropy_with_logits(logits_torch, target_torch)
        out_mlx = F_mlx.binary_cross_entropy_with_logits(logits_mlx, target_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_smooth_l1_loss_parity(self):
        """Test smooth_l1_loss produces matching outputs."""
        pred_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randn(10, 5).astype(np.float32)

        pred_torch = torch.tensor(pred_np)
        target_torch = torch.tensor(target_np)

        pred_mlx = flashlight.tensor(pred_np)
        target_mlx = flashlight.tensor(target_np)

        for beta in [0.5, 1.0, 2.0]:
            out_torch = F_torch.smooth_l1_loss(pred_torch, target_torch, beta=beta)
            out_mlx = F_mlx.smooth_l1_loss(pred_mlx, target_mlx, beta=beta)

            np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestPoolingParity:
    """Parity tests for pooling operations."""

    def test_max_pool2d_parity(self):
        """Test max_pool2d produces matching outputs."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.max_pool2d(x_torch, kernel_size=2, stride=2)
        out_mlx = F_mlx.max_pool2d(x_mlx, kernel_size=2, stride=2)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_avg_pool2d_parity(self):
        """Test avg_pool2d produces matching outputs."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.avg_pool2d(x_torch, kernel_size=2, stride=2)
        out_mlx = F_mlx.avg_pool2d(x_mlx, kernel_size=2, stride=2)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_adaptive_avg_pool2d_global_parity(self):
        """Test adaptive_avg_pool2d with global pooling."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.adaptive_avg_pool2d(x_torch, (1, 1))
        out_mlx = F_mlx.adaptive_avg_pool2d(x_mlx, (1, 1))

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestEmbeddingParity:
    """Parity tests for embedding operations."""

    def test_embedding_parity(self):
        """Test embedding produces matching outputs."""
        num_embeddings = 100
        embedding_dim = 32

        weight_np = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        indices_np = np.random.randint(0, num_embeddings, size=(5, 10)).astype(np.int64)

        weight_torch = torch.tensor(weight_np)
        indices_torch = torch.tensor(indices_np)

        weight_mlx = flashlight.tensor(weight_np)
        indices_mlx = flashlight.tensor(indices_np)

        out_torch = F_torch.embedding(indices_torch, weight_torch)
        out_mlx = F_mlx.embedding(indices_mlx, weight_mlx)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestPaddingParity:
    """Parity tests for padding operations."""

    def test_pad_constant_parity(self):
        """Test constant padding produces matching outputs."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        # Pad last two dimensions: (left, right, top, bottom)
        out_torch = F_torch.pad(x_torch, (1, 2, 3, 4), mode="constant", value=0.0)
        out_mlx = F_mlx.pad(x_mlx, (1, 2, 3, 4), mode="constant", value=0.0)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestNormalizeParity:
    """Parity tests for normalize operation."""

    def test_normalize_l2_parity(self):
        """Test L2 normalize produces matching outputs."""
        x_np = np.random.randn(5, 10).astype(np.float32)

        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.normalize(x_torch, p=2, dim=1)
        out_mlx = F_mlx.normalize(x_mlx, p=2, dim=1)

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestSignatureCompatibility:
    """Test that function signatures match PyTorch for compatibility."""

    def test_mse_loss_deprecated_size_average(self):
        """Test mse_loss accepts deprecated size_average parameter."""
        pred_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randn(10, 5).astype(np.float32)

        pred_mlx = flashlight.tensor(pred_np)
        target_mlx = flashlight.tensor(target_np)

        # Should emit deprecation warning but work
        with pytest.warns(DeprecationWarning):
            loss = F_mlx.mse_loss(pred_mlx, target_mlx, size_average=True)

        # size_average=True should be equivalent to reduction='mean'
        expected = F_mlx.mse_loss(pred_mlx, target_mlx, reduction="mean")
        np.testing.assert_allclose(loss.numpy(), expected.numpy())

    def test_mse_loss_deprecated_reduce(self):
        """Test mse_loss accepts deprecated reduce parameter."""
        pred_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randn(10, 5).astype(np.float32)

        pred_mlx = flashlight.tensor(pred_np)
        target_mlx = flashlight.tensor(target_np)

        # reduce=False should be equivalent to reduction='none'
        with pytest.warns(DeprecationWarning):
            loss = F_mlx.mse_loss(pred_mlx, target_mlx, reduce=False)

        expected = F_mlx.mse_loss(pred_mlx, target_mlx, reduction="none")
        np.testing.assert_allclose(loss.numpy(), expected.numpy())

    def test_cross_entropy_deprecated_params(self):
        """Test cross_entropy accepts deprecated parameters."""
        logits_np = np.random.randn(10, 5).astype(np.float32)
        target_np = np.random.randint(0, 5, size=(10,)).astype(np.int64)

        logits_mlx = flashlight.tensor(logits_np)
        target_mlx = flashlight.tensor(target_np)

        with pytest.warns(DeprecationWarning):
            loss = F_mlx.cross_entropy(logits_mlx, target_mlx, size_average=True)

        expected = F_mlx.cross_entropy(logits_mlx, target_mlx, reduction="mean")
        np.testing.assert_allclose(loss.numpy(), expected.numpy(), rtol=1e-5)

    def test_softmax_dtype_param(self):
        """Test softmax accepts dtype parameter."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        # Should accept dtype parameter without error
        out = F_mlx.softmax(x_mlx, dim=-1, dtype=None)
        assert out.shape == x_mlx.shape

    def test_log_softmax_dtype_param(self):
        """Test log_softmax accepts dtype parameter."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        # Should accept dtype parameter without error
        out = F_mlx.log_softmax(x_mlx, dim=-1, dtype=None)
        assert out.shape == x_mlx.shape

    def test_silu_inplace_param(self):
        """Test silu accepts inplace parameter."""
        x_np = np.random.randn(10, 20).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        # Should accept inplace parameter without error (even though it's ignored)
        out = F_mlx.silu(x_mlx, inplace=False)
        assert out.shape == x_mlx.shape

    def test_interpolate_antialias_param(self):
        """Test interpolate accepts antialias parameter."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        # Should accept antialias parameter (with warning)
        out = F_mlx.interpolate(x_mlx, size=(4, 4), antialias=False)
        assert out.shape == (2, 3, 4, 4)

    def test_normalize_out_param(self):
        """Test normalize raises for out parameter."""
        x_np = np.random.randn(5, 10).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        # out=None should work fine
        out = F_mlx.normalize(x_mlx, p=2, dim=1, out=None)
        assert out.shape == x_mlx.shape

        # out=something should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            F_mlx.normalize(x_mlx, p=2, dim=1, out=x_mlx)

    def test_pad_value_none(self):
        """Test pad accepts value=None (defaults to 0)."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_mlx = flashlight.tensor(x_np)

        # value=None should default to 0.0
        out = F_mlx.pad(x_mlx, (1, 1, 1, 1), mode="constant", value=None)
        assert out.shape == (2, 3, 10, 10)

    def test_triplet_margin_loss_deprecated_params(self):
        """Test triplet_margin_loss accepts deprecated parameters."""
        anchor = flashlight.randn(10, 128)
        positive = flashlight.randn(10, 128)
        negative = flashlight.randn(10, 128)

        with pytest.warns(DeprecationWarning):
            loss = F_mlx.triplet_margin_loss(anchor, positive, negative, size_average=True)

        expected = F_mlx.triplet_margin_loss(anchor, positive, negative, reduction="mean")
        np.testing.assert_allclose(loss.numpy(), expected.numpy(), rtol=1e-5)

    def test_margin_ranking_loss_deprecated_params(self):
        """Test margin_ranking_loss accepts deprecated parameters."""
        x1 = flashlight.randn(10)
        x2 = flashlight.randn(10)
        target = flashlight.tensor(np.random.choice([-1, 1], size=10).astype(np.float32))

        with pytest.warns(DeprecationWarning):
            loss = F_mlx.margin_ranking_loss(x1, x2, target, size_average=True)

        expected = F_mlx.margin_ranking_loss(x1, x2, target, reduction="mean")
        np.testing.assert_allclose(loss.numpy(), expected.numpy(), rtol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestGridSample3DParity:
    """Parity tests for 3D grid sampling operations."""

    def test_grid_sample_3d_bilinear_zeros_basic(self):
        """Test 3D grid_sample with bilinear (trilinear) interpolation and zeros padding."""
        # Input: (N, C, D_in, H_in, W_in)
        N, C, D_in, H_in, W_in = 2, 3, 4, 5, 6
        D_out, H_out, W_out = 3, 4, 5

        x_np = np.random.randn(N, C, D_in, H_in, W_in).astype(np.float32)
        # Grid: (N, D_out, H_out, W_out, 3) with values in [-1, 1]
        grid_np = np.random.randn(N, D_out, H_out, W_out, 3).astype(np.float32) * 0.5

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)

    def test_grid_sample_3d_bilinear_zeros_align_corners(self):
        """Test 3D grid_sample with align_corners=True."""
        N, C, D_in, H_in, W_in = 2, 2, 3, 4, 5
        D_out, H_out, W_out = 2, 3, 4

        x_np = np.random.randn(N, C, D_in, H_in, W_in).astype(np.float32)
        grid_np = np.random.randn(N, D_out, H_out, W_out, 3).astype(np.float32) * 0.5

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)

    def test_grid_sample_3d_bilinear_border(self):
        """Test 3D grid_sample with border padding mode."""
        N, C, D_in, H_in, W_in = 2, 2, 4, 4, 4
        D_out, H_out, W_out = 3, 3, 3

        x_np = np.random.randn(N, C, D_in, H_in, W_in).astype(np.float32)
        # Use values outside [-1, 1] to test border clamping
        grid_np = np.random.randn(N, D_out, H_out, W_out, 3).astype(np.float32) * 2.0

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="bilinear", padding_mode="border", align_corners=False
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="bilinear", padding_mode="border", align_corners=False
        )

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)

    def test_grid_sample_3d_bilinear_reflection(self):
        """Test 3D grid_sample with reflection padding mode."""
        N, C, D_in, H_in, W_in = 2, 2, 4, 4, 4
        D_out, H_out, W_out = 3, 3, 3

        x_np = np.random.randn(N, C, D_in, H_in, W_in).astype(np.float32)
        # Use values outside [-1, 1] to test reflection
        grid_np = np.random.randn(N, D_out, H_out, W_out, 3).astype(np.float32) * 1.5

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="bilinear", padding_mode="reflection", align_corners=False
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="bilinear", padding_mode="reflection", align_corners=False
        )

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-4)

    def test_grid_sample_3d_nearest_zeros(self):
        """Test 3D grid_sample with nearest neighbor interpolation."""
        N, C, D_in, H_in, W_in = 2, 3, 4, 5, 6
        D_out, H_out, W_out = 3, 4, 5

        x_np = np.random.randn(N, C, D_in, H_in, W_in).astype(np.float32)
        grid_np = np.random.randn(N, D_out, H_out, W_out, 3).astype(np.float32) * 0.5

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="nearest", padding_mode="zeros", align_corners=False
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="nearest", padding_mode="zeros", align_corners=False
        )

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_grid_sample_3d_nearest_border(self):
        """Test 3D grid_sample with nearest neighbor and border padding."""
        N, C, D_in, H_in, W_in = 2, 2, 4, 4, 4
        D_out, H_out, W_out = 3, 3, 3

        x_np = np.random.randn(N, C, D_in, H_in, W_in).astype(np.float32)
        grid_np = np.random.randn(N, D_out, H_out, W_out, 3).astype(np.float32) * 2.0

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="nearest", padding_mode="border", align_corners=False
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="nearest", padding_mode="border", align_corners=False
        )

        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_grid_sample_3d_identity_grid(self):
        """Test 3D grid_sample with an identity grid (should reproduce input)."""
        N, C, D, H, W = 1, 2, 3, 4, 5

        x_np = np.random.randn(N, C, D, H, W).astype(np.float32)

        # Create identity grid
        d_coords = np.linspace(-1, 1, D)
        h_coords = np.linspace(-1, 1, H)
        w_coords = np.linspace(-1, 1, W)
        ww, hh, dd = np.meshgrid(w_coords, h_coords, d_coords, indexing="xy")
        # Note: PyTorch expects (x, y, z) = (w, h, d)
        grid_np = np.stack([ww, hh, dd], axis=-1).transpose(2, 0, 1, 3).astype(np.float32)
        grid_np = np.expand_dims(grid_np, axis=0)  # (1, D, H, W, 3)

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # With identity grid and align_corners=True, output should match input
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5)
        # Also check that PyTorch output matches input (sanity check)
        np.testing.assert_allclose(out_torch.numpy(), x_np, rtol=1e-4, atol=1e-5)

    def test_grid_sample_3d_out_of_bounds_zeros(self):
        """Test 3D grid_sample with out-of-bounds coordinates and zeros padding."""
        N, C, D_in, H_in, W_in = 1, 1, 3, 3, 3
        D_out, H_out, W_out = 2, 2, 2

        x_np = np.ones((N, C, D_in, H_in, W_in), dtype=np.float32)
        # All grid points are way out of bounds
        grid_np = np.ones((N, D_out, H_out, W_out, 3), dtype=np.float32) * 5.0

        x_torch = torch.tensor(x_np)
        grid_torch = torch.tensor(grid_np)

        x_mlx = flashlight.tensor(x_np)
        grid_mlx = flashlight.tensor(grid_np)

        out_torch = F_torch.grid_sample(
            x_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        out_mlx = F_mlx.grid_sample(
            x_mlx, grid_mlx, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # With zeros padding and out-of-bounds, output should be all zeros
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out_torch.numpy(), 0.0, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
class TestInterpolateParity:
    """Parity tests for interpolation functions."""

    # =====================================================
    # Nearest Interpolation Tests (3D, 4D, 5D)
    # =====================================================

    def test_nearest_3d_upsample(self):
        """Test nearest interpolation upsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16,), mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, size=(16,), mode="nearest")

        assert out_mlx.shape == (2, 3, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_3d_downsample(self):
        """Test nearest interpolation downsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8,), mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8,), mode="nearest")

        assert out_mlx.shape == (2, 3, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_4d_upsample(self):
        """Test nearest interpolation upsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16, 16), mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, size=(16, 16), mode="nearest")

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_4d_downsample(self):
        """Test nearest interpolation downsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 8), mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 8), mode="nearest")

        assert out_mlx.shape == (2, 3, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_5d_upsample(self):
        """Test nearest interpolation upsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 16, 16), mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 16, 16), mode="nearest")

        assert out_mlx.shape == (2, 3, 8, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_5d_downsample(self):
        """Test nearest interpolation downsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 8, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(4, 8, 8), mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, size=(4, 8, 8), mode="nearest")

        assert out_mlx.shape == (2, 3, 4, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    # =====================================================
    # Nearest-Exact Interpolation Tests (3D, 4D, 5D)
    # =====================================================

    def test_nearest_exact_3d_upsample(self):
        """Test nearest-exact interpolation upsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16,), mode="nearest-exact")
        out_mlx = F_mlx.interpolate(x_mlx, size=(16,), mode="nearest-exact")

        assert out_mlx.shape == (2, 3, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_exact_3d_downsample(self):
        """Test nearest-exact interpolation downsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8,), mode="nearest-exact")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8,), mode="nearest-exact")

        assert out_mlx.shape == (2, 3, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_exact_4d_upsample(self):
        """Test nearest-exact interpolation upsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16, 16), mode="nearest-exact")
        out_mlx = F_mlx.interpolate(x_mlx, size=(16, 16), mode="nearest-exact")

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_exact_4d_downsample(self):
        """Test nearest-exact interpolation downsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 8), mode="nearest-exact")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 8), mode="nearest-exact")

        assert out_mlx.shape == (2, 3, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_exact_5d_upsample(self):
        """Test nearest-exact interpolation upsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 16, 16), mode="nearest-exact")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 16, 16), mode="nearest-exact")

        assert out_mlx.shape == (2, 3, 8, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_nearest_exact_5d_downsample(self):
        """Test nearest-exact interpolation downsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 8, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(4, 8, 8), mode="nearest-exact")
        out_mlx = F_mlx.interpolate(x_mlx, size=(4, 8, 8), mode="nearest-exact")

        assert out_mlx.shape == (2, 3, 4, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    # =====================================================
    # Linear Interpolation Tests (3D only)
    # =====================================================

    def test_linear_3d_upsample(self):
        """Test linear interpolation upsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16,), mode="linear", align_corners=False)
        out_mlx = F_mlx.interpolate(x_mlx, size=(16,), mode="linear", align_corners=False)

        assert out_mlx.shape == (2, 3, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_linear_3d_align_corners(self):
        """Test linear interpolation with align_corners=True for 3D input."""
        x_np = np.random.randn(2, 3, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16,), mode="linear", align_corners=True)
        out_mlx = F_mlx.interpolate(x_mlx, size=(16,), mode="linear", align_corners=True)

        assert out_mlx.shape == (2, 3, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_linear_3d_downsample(self):
        """Test linear interpolation downsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8,), mode="linear", align_corners=False)
        out_mlx = F_mlx.interpolate(x_mlx, size=(8,), mode="linear", align_corners=False)

        assert out_mlx.shape == (2, 3, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    # =====================================================
    # Bilinear Interpolation Tests (4D only)
    # =====================================================

    def test_bilinear_4d_upsample(self):
        """Test bilinear interpolation upsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(
            x_torch, size=(16, 16), mode="bilinear", align_corners=False
        )
        out_mlx = F_mlx.interpolate(x_mlx, size=(16, 16), mode="bilinear", align_corners=False)

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_bilinear_4d_align_corners(self):
        """Test bilinear interpolation with align_corners=True for 4D input."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16, 16), mode="bilinear", align_corners=True)
        out_mlx = F_mlx.interpolate(x_mlx, size=(16, 16), mode="bilinear", align_corners=True)

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_bilinear_4d_downsample(self):
        """Test bilinear interpolation downsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 8), mode="bilinear", align_corners=False)
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 8), mode="bilinear", align_corners=False)

        assert out_mlx.shape == (2, 3, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    # =====================================================
    # Bicubic Interpolation Tests (4D only)
    # =====================================================

    def test_bicubic_4d_upsample(self):
        """Test bicubic interpolation upsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16, 16), mode="bicubic", align_corners=False)
        out_mlx = F_mlx.interpolate(x_mlx, size=(16, 16), mode="bicubic", align_corners=False)

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-4)

    def test_bicubic_4d_align_corners(self):
        """Test bicubic interpolation with align_corners=True for 4D input."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(16, 16), mode="bicubic", align_corners=True)
        out_mlx = F_mlx.interpolate(x_mlx, size=(16, 16), mode="bicubic", align_corners=True)

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-4)

    def test_bicubic_4d_downsample(self):
        """Test bicubic interpolation downsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 8), mode="bicubic", align_corners=False)
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 8), mode="bicubic", align_corners=False)

        assert out_mlx.shape == (2, 3, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-4)

    # =====================================================
    # Trilinear Interpolation Tests (5D only)
    # =====================================================

    def test_trilinear_5d_upsample(self):
        """Test trilinear interpolation upsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(
            x_torch, size=(8, 16, 16), mode="trilinear", align_corners=False
        )
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 16, 16), mode="trilinear", align_corners=False)

        assert out_mlx.shape == (2, 3, 8, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_trilinear_5d_align_corners(self):
        """Test trilinear interpolation with align_corners=True for 5D input."""
        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(
            x_torch, size=(8, 16, 16), mode="trilinear", align_corners=True
        )
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 16, 16), mode="trilinear", align_corners=True)

        assert out_mlx.shape == (2, 3, 8, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_trilinear_5d_downsample(self):
        """Test trilinear interpolation downsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 8, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(
            x_torch, size=(4, 8, 8), mode="trilinear", align_corners=False
        )
        out_mlx = F_mlx.interpolate(x_mlx, size=(4, 8, 8), mode="trilinear", align_corners=False)

        assert out_mlx.shape == (2, 3, 4, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    # =====================================================
    # Area Interpolation Tests (3D, 4D, 5D)
    # =====================================================

    def test_area_3d_downsample(self):
        """Test area interpolation downsampling for 3D input (NCL)."""
        x_np = np.random.randn(2, 3, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8,), mode="area")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8,), mode="area")

        assert out_mlx.shape == (2, 3, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_area_4d_downsample(self):
        """Test area interpolation downsampling for 4D input (NCHW)."""
        x_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(8, 8), mode="area")
        out_mlx = F_mlx.interpolate(x_mlx, size=(8, 8), mode="area")

        assert out_mlx.shape == (2, 3, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    def test_area_5d_downsample(self):
        """Test area interpolation downsampling for 5D input (NCDHW)."""
        x_np = np.random.randn(2, 3, 8, 16, 16).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, size=(4, 8, 8), mode="area")
        out_mlx = F_mlx.interpolate(x_mlx, size=(4, 8, 8), mode="area")

        assert out_mlx.shape == (2, 3, 4, 8, 8)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-5)

    # =====================================================
    # Scale Factor Tests
    # =====================================================

    def test_interpolate_scale_factor_3d(self):
        """Test interpolation with scale_factor for 3D input."""
        x_np = np.random.randn(2, 3, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, scale_factor=2.0, mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, scale_factor=2.0, mode="nearest")

        assert out_mlx.shape == (2, 3, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_interpolate_scale_factor_4d(self):
        """Test interpolation with scale_factor for 4D input."""
        x_np = np.random.randn(2, 3, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, scale_factor=2.0, mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, scale_factor=2.0, mode="nearest")

        assert out_mlx.shape == (2, 3, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)

    def test_interpolate_scale_factor_5d(self):
        """Test interpolation with scale_factor for 5D input."""
        x_np = np.random.randn(2, 3, 4, 8, 8).astype(np.float32)
        x_torch = torch.tensor(x_np)
        x_mlx = flashlight.tensor(x_np)

        out_torch = F_torch.interpolate(x_torch, scale_factor=2.0, mode="nearest")
        out_mlx = F_mlx.interpolate(x_mlx, scale_factor=2.0, mode="nearest")

        assert out_mlx.shape == (2, 3, 8, 16, 16)
        np.testing.assert_allclose(out_torch.numpy(), out_mlx.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
