"""
Parity tests for nn.Module layers.

Tests that mlx_compat.nn layers produce identical outputs to torch.nn layers.
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import mlx_compat
import mlx_compat.nn as nn_mlx


def copy_linear_weights(mlx_layer, torch_layer):
    """Copy weights from PyTorch Linear to MLX Linear."""
    mlx_layer.weight._mlx_array = mlx_compat.tensor(
        torch_layer.weight.detach().numpy()
    )._mlx_array
    if torch_layer.bias is not None:
        mlx_layer.bias._mlx_array = mlx_compat.tensor(
            torch_layer.bias.detach().numpy()
        )._mlx_array


def copy_conv_weights(mlx_layer, torch_layer):
    """Copy weights from PyTorch Conv to MLX Conv."""
    mlx_layer.weight._mlx_array = mlx_compat.tensor(
        torch_layer.weight.detach().numpy()
    )._mlx_array
    if torch_layer.bias is not None:
        mlx_layer.bias._mlx_array = mlx_compat.tensor(
            torch_layer.bias.detach().numpy()
        )._mlx_array


def copy_bn_weights(mlx_layer, torch_layer):
    """Copy weights from PyTorch BatchNorm to MLX BatchNorm."""
    if torch_layer.weight is not None:
        mlx_layer.weight._mlx_array = mlx_compat.tensor(
            torch_layer.weight.detach().numpy()
        )._mlx_array
    if torch_layer.bias is not None:
        mlx_layer.bias._mlx_array = mlx_compat.tensor(
            torch_layer.bias.detach().numpy()
        )._mlx_array
    mlx_layer.running_mean._mlx_array = mlx_compat.tensor(
        torch_layer.running_mean.detach().numpy()
    )._mlx_array
    mlx_layer.running_var._mlx_array = mlx_compat.tensor(
        torch_layer.running_var.detach().numpy()
    )._mlx_array


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestLinearParity:
    """Parity tests for Linear layer."""

    def test_linear_forward_parity(self):
        """Test Linear forward pass matches PyTorch."""
        np.random.seed(42)
        in_features, out_features = 64, 128
        x_np = np.random.randn(4, in_features).astype(np.float32)

        # Create layers
        torch_layer = nn_torch.Linear(in_features, out_features)
        mlx_layer = nn_mlx.Linear(in_features, out_features)
        copy_linear_weights(mlx_layer, torch_layer)

        # Forward pass
        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_linear_no_bias_parity(self):
        """Test Linear without bias matches PyTorch."""
        np.random.seed(42)
        in_features, out_features = 64, 128
        x_np = np.random.randn(4, in_features).astype(np.float32)

        torch_layer = nn_torch.Linear(in_features, out_features, bias=False)
        mlx_layer = nn_mlx.Linear(in_features, out_features, bias=False)
        copy_linear_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_linear_3d_input_parity(self):
        """Test Linear with 3D input matches PyTorch."""
        np.random.seed(42)
        in_features, out_features = 64, 128
        x_np = np.random.randn(4, 10, in_features).astype(np.float32)

        torch_layer = nn_torch.Linear(in_features, out_features)
        mlx_layer = nn_mlx.Linear(in_features, out_features)
        copy_linear_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestConv2dParity:
    """Parity tests for Conv2d layer."""

    def test_conv2d_forward_parity(self):
        """Test Conv2d forward pass matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.Conv2d(16, 32, kernel_size=3, padding=1)
        mlx_layer = nn_mlx.Conv2d(16, 32, kernel_size=3, padding=1)
        copy_conv_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_conv2d_stride_parity(self):
        """Test Conv2d with stride matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 16, 16).astype(np.float32)

        torch_layer = nn_torch.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        mlx_layer = nn_mlx.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        copy_conv_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_conv2d_no_bias_parity(self):
        """Test Conv2d without bias matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        mlx_layer = nn_mlx.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        copy_conv_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestConv1dParity:
    """Parity tests for Conv1d layer."""

    def test_conv1d_forward_parity(self):
        """Test Conv1d forward pass matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 100).astype(np.float32)

        torch_layer = nn_torch.Conv1d(16, 32, kernel_size=3, padding=1)
        mlx_layer = nn_mlx.Conv1d(16, 32, kernel_size=3, padding=1)
        copy_conv_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestBatchNormParity:
    """Parity tests for BatchNorm layers."""

    def test_batchnorm2d_eval_parity(self):
        """Test BatchNorm2d in eval mode matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 32, 8, 8).astype(np.float32)

        torch_layer = nn_torch.BatchNorm2d(32)
        torch_layer.eval()
        mlx_layer = nn_mlx.BatchNorm2d(32)
        mlx_layer.eval()
        copy_bn_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_batchnorm1d_eval_parity(self):
        """Test BatchNorm1d in eval mode matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 32, 100).astype(np.float32)

        torch_layer = nn_torch.BatchNorm1d(32)
        torch_layer.eval()
        mlx_layer = nn_mlx.BatchNorm1d(32)
        mlx_layer.eval()
        copy_bn_weights(mlx_layer, torch_layer)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestLayerNormParity:
    """Parity tests for LayerNorm."""

    def test_layernorm_parity(self):
        """Test LayerNorm matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 10, 64).astype(np.float32)

        torch_layer = nn_torch.LayerNorm(64)
        mlx_layer = nn_mlx.LayerNorm(64)

        # Copy weights
        mlx_layer.weight._mlx_array = mlx_compat.tensor(
            torch_layer.weight.detach().numpy()
        )._mlx_array
        mlx_layer.bias._mlx_array = mlx_compat.tensor(
            torch_layer.bias.detach().numpy()
        )._mlx_array

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_layernorm_tuple_shape_parity(self):
        """Test LayerNorm with tuple normalized_shape matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 10, 64).astype(np.float32)

        torch_layer = nn_torch.LayerNorm([10, 64])
        mlx_layer = nn_mlx.LayerNorm([10, 64])

        mlx_layer.weight._mlx_array = mlx_compat.tensor(
            torch_layer.weight.detach().numpy()
        )._mlx_array
        mlx_layer.bias._mlx_array = mlx_compat.tensor(
            torch_layer.bias.detach().numpy()
        )._mlx_array

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestPoolingParity:
    """Parity tests for pooling layers."""

    def test_maxpool2d_parity(self):
        """Test MaxPool2d matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.MaxPool2d(kernel_size=2, stride=2)
        mlx_layer = nn_mlx.MaxPool2d(kernel_size=2, stride=2)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_avgpool2d_parity(self):
        """Test AvgPool2d matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.AvgPool2d(kernel_size=2, stride=2)
        mlx_layer = nn_mlx.AvgPool2d(kernel_size=2, stride=2)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_adaptiveavgpool2d_parity(self):
        """Test AdaptiveAvgPool2d matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.AdaptiveAvgPool2d((1, 1))
        mlx_layer = nn_mlx.AdaptiveAvgPool2d((1, 1))

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestActivationLayersParity:
    """Parity tests for activation layers."""

    def test_relu_parity(self):
        """Test ReLU matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 64).astype(np.float32)

        torch_layer = nn_torch.ReLU()
        mlx_layer = nn_mlx.ReLU()

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_gelu_parity(self):
        """Test GELU matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 64).astype(np.float32)

        torch_layer = nn_torch.GELU()
        mlx_layer = nn_mlx.GELU()

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_silu_parity(self):
        """Test SiLU matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 64).astype(np.float32)

        torch_layer = nn_torch.SiLU()
        mlx_layer = nn_mlx.SiLU()

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_leakyrelu_parity(self):
        """Test LeakyReLU matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 64).astype(np.float32)

        torch_layer = nn_torch.LeakyReLU(negative_slope=0.01)
        mlx_layer = nn_mlx.LeakyReLU(negative_slope=0.01)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_softmax_parity(self):
        """Test Softmax matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(4, 10).astype(np.float32)

        torch_layer = nn_torch.Softmax(dim=-1)
        mlx_layer = nn_mlx.Softmax(dim=-1)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestEmbeddingParity:
    """Parity tests for Embedding layer."""

    def test_embedding_parity(self):
        """Test Embedding matches PyTorch."""
        np.random.seed(42)
        num_embeddings, embedding_dim = 100, 64
        indices = np.array([0, 10, 20, 30], dtype=np.int64)

        torch_layer = nn_torch.Embedding(num_embeddings, embedding_dim)
        mlx_layer = nn_mlx.Embedding(num_embeddings, embedding_dim)

        # Copy weights
        mlx_layer.weight._mlx_array = mlx_compat.tensor(
            torch_layer.weight.detach().numpy()
        )._mlx_array

        out_torch = torch_layer(torch.tensor(indices))
        out_mlx = mlx_layer(mlx_compat.tensor(indices))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_embedding_2d_indices_parity(self):
        """Test Embedding with 2D indices matches PyTorch."""
        np.random.seed(42)
        num_embeddings, embedding_dim = 100, 64
        indices = np.array([[0, 10], [20, 30]], dtype=np.int64)

        torch_layer = nn_torch.Embedding(num_embeddings, embedding_dim)
        mlx_layer = nn_mlx.Embedding(num_embeddings, embedding_dim)

        mlx_layer.weight._mlx_array = mlx_compat.tensor(
            torch_layer.weight.detach().numpy()
        )._mlx_array

        out_torch = torch_layer(torch.tensor(indices))
        out_mlx = mlx_layer(mlx_compat.tensor(indices))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestDropoutParity:
    """Parity tests for Dropout layers in eval mode."""

    def test_dropout_eval_parity(self):
        """Test Dropout in eval mode passes through unchanged."""
        np.random.seed(42)
        x_np = np.random.randn(4, 64).astype(np.float32)

        torch_layer = nn_torch.Dropout(p=0.5)
        torch_layer.eval()
        mlx_layer = nn_mlx.Dropout(p=0.5)
        mlx_layer.eval()

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestLossParity:
    """Parity tests for loss functions."""

    def test_mseloss_parity(self):
        """Test MSELoss matches PyTorch."""
        np.random.seed(42)
        pred_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randn(4, 10).astype(np.float32)

        torch_loss = nn_torch.MSELoss()
        mlx_loss = nn_mlx.MSELoss()

        out_torch = torch_loss(torch.tensor(pred_np), torch.tensor(target_np))
        out_mlx = mlx_loss(mlx_compat.tensor(pred_np), mlx_compat.tensor(target_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_l1loss_parity(self):
        """Test L1Loss matches PyTorch."""
        np.random.seed(42)
        pred_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randn(4, 10).astype(np.float32)

        torch_loss = nn_torch.L1Loss()
        mlx_loss = nn_mlx.L1Loss()

        out_torch = torch_loss(torch.tensor(pred_np), torch.tensor(target_np))
        out_mlx = mlx_loss(mlx_compat.tensor(pred_np), mlx_compat.tensor(target_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_crossentropyloss_parity(self):
        """Test CrossEntropyLoss matches PyTorch."""
        np.random.seed(42)
        logits_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.array([0, 3, 5, 9], dtype=np.int64)

        torch_loss = nn_torch.CrossEntropyLoss()
        mlx_loss = nn_mlx.CrossEntropyLoss()

        out_torch = torch_loss(torch.tensor(logits_np), torch.tensor(target_np))
        out_mlx = mlx_loss(
            mlx_compat.tensor(logits_np),
            mlx_compat.tensor(target_np, dtype=mlx_compat.int32)
        )

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_bceloss_parity(self):
        """Test BCELoss matches PyTorch."""
        np.random.seed(42)
        # Use sigmoid to get values in (0, 1)
        pred_np = 1 / (1 + np.exp(-np.random.randn(4, 10).astype(np.float32)))
        target_np = np.random.rand(4, 10).astype(np.float32)

        torch_loss = nn_torch.BCELoss()
        mlx_loss = nn_mlx.BCELoss()

        out_torch = torch_loss(torch.tensor(pred_np), torch.tensor(target_np))
        out_mlx = mlx_loss(mlx_compat.tensor(pred_np), mlx_compat.tensor(target_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_nllloss_parity(self):
        """Test NLLLoss matches PyTorch."""
        np.random.seed(42)
        # Log probabilities
        log_probs_np = np.log(np.random.dirichlet(np.ones(10), size=4).astype(np.float32))
        target_np = np.array([0, 3, 5, 9], dtype=np.int64)

        torch_loss = nn_torch.NLLLoss()
        mlx_loss = nn_mlx.NLLLoss()

        out_torch = torch_loss(torch.tensor(log_probs_np), torch.tensor(target_np))
        out_mlx = mlx_loss(
            mlx_compat.tensor(log_probs_np),
            mlx_compat.tensor(target_np, dtype=mlx_compat.int32)
        )

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestRNNParity:
    """Parity tests for RNN layers."""

    def test_lstm_parity(self):
        """Test LSTM matches PyTorch."""
        np.random.seed(42)
        input_size, hidden_size = 32, 64
        batch_size, seq_len = 4, 10
        x_np = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        torch_layer = nn_torch.LSTM(input_size, hidden_size, batch_first=False)
        mlx_layer = nn_mlx.LSTM(input_size, hidden_size, batch_first=False)

        # Copy weights - LSTM has weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        mlx_layer.weight_ih_l0._mlx_array = mlx_compat.tensor(
            torch_layer.weight_ih_l0.detach().numpy()
        )._mlx_array
        mlx_layer.weight_hh_l0._mlx_array = mlx_compat.tensor(
            torch_layer.weight_hh_l0.detach().numpy()
        )._mlx_array
        mlx_layer.bias_ih_l0._mlx_array = mlx_compat.tensor(
            torch_layer.bias_ih_l0.detach().numpy()
        )._mlx_array
        mlx_layer.bias_hh_l0._mlx_array = mlx_compat.tensor(
            torch_layer.bias_hh_l0.detach().numpy()
        )._mlx_array

        out_torch, (h_torch, c_torch) = torch_layer(torch.tensor(x_np))
        out_mlx, (h_mlx, c_mlx) = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )

    def test_gru_parity(self):
        """Test GRU matches PyTorch."""
        np.random.seed(42)
        input_size, hidden_size = 32, 64
        batch_size, seq_len = 4, 10
        x_np = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        torch_layer = nn_torch.GRU(input_size, hidden_size, batch_first=False)
        mlx_layer = nn_mlx.GRU(input_size, hidden_size, batch_first=False)

        # Copy weights
        mlx_layer.weight_ih_l0._mlx_array = mlx_compat.tensor(
            torch_layer.weight_ih_l0.detach().numpy()
        )._mlx_array
        mlx_layer.weight_hh_l0._mlx_array = mlx_compat.tensor(
            torch_layer.weight_hh_l0.detach().numpy()
        )._mlx_array
        mlx_layer.bias_ih_l0._mlx_array = mlx_compat.tensor(
            torch_layer.bias_ih_l0.detach().numpy()
        )._mlx_array
        mlx_layer.bias_hh_l0._mlx_array = mlx_compat.tensor(
            torch_layer.bias_hh_l0.detach().numpy()
        )._mlx_array

        out_torch, h_torch = torch_layer(torch.tensor(x_np))
        out_mlx, h_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-4, atol=1e-5
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for parity tests")
@pytest.mark.parity
class TestPaddingParity:
    """Parity tests for padding layers."""

    def test_zeropad2d_parity(self):
        """Test ZeroPad2d matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.ZeroPad2d(padding=(1, 2, 3, 4))
        mlx_layer = nn_mlx.ZeroPad2d(padding=(1, 2, 3, 4))

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_reflectionpad2d_parity(self):
        """Test ReflectionPad2d matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.ReflectionPad2d(padding=2)
        mlx_layer = nn_mlx.ReflectionPad2d(padding=2)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    def test_replicationpad2d_parity(self):
        """Test ReplicationPad2d matches PyTorch."""
        np.random.seed(42)
        x_np = np.random.randn(2, 16, 8, 8).astype(np.float32)

        torch_layer = nn_torch.ReplicationPad2d(padding=2)
        mlx_layer = nn_mlx.ReplicationPad2d(padding=2)

        out_torch = torch_layer(torch.tensor(x_np))
        out_mlx = mlx_layer(mlx_compat.tensor(x_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'parity'])
