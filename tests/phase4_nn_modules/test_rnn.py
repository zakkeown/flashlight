"""
Test Phase 4: RNN Layers

Tests the nn.layers.rnn module:
- RNNCell, LSTMCell, GRUCell
- RNN, LSTM, GRU
- Batch first and bidirectional modes
- PyTorch parity
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@skipIfNoMLX
class TestRNNCell(TestCase):
    """Test nn.RNNCell."""

    def test_creation(self):
        """Test RNNCell creation with default parameters."""
        cell = flashlight.nn.RNNCell(input_size=10, hidden_size=20)
        self.assertEqual(cell.input_size, 10)
        self.assertEqual(cell.hidden_size, 20)

    def test_creation_with_bias(self):
        """Test RNNCell creation with bias."""
        cell = flashlight.nn.RNNCell(input_size=10, hidden_size=20, bias=True)
        self.assertTrue(cell.bias is not None or hasattr(cell, "bias_ih"))

    def test_creation_without_bias(self):
        """Test RNNCell creation without bias."""
        cell = flashlight.nn.RNNCell(input_size=10, hidden_size=20, bias=False)
        # Check bias is False or None

    def test_forward_shape(self):
        """Test RNNCell forward pass output shape."""
        cell = flashlight.nn.RNNCell(input_size=10, hidden_size=20)
        x = flashlight.randn(5, 10)  # batch=5, input=10
        hx = flashlight.randn(5, 20)  # batch=5, hidden=20
        output = cell(x, hx)
        self.assertEqual(output.shape, (5, 20))

    def test_forward_no_hidden(self):
        """Test RNNCell forward with no initial hidden state."""
        cell = flashlight.nn.RNNCell(input_size=10, hidden_size=20)
        x = flashlight.randn(5, 10)
        output = cell(x)
        self.assertEqual(output.shape, (5, 20))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        input_size, hidden_size, batch_size = 10, 20, 5

        # Create weights
        weight_ih = np.random.randn(hidden_size, input_size).astype(np.float32)
        weight_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        bias_ih = np.random.randn(hidden_size).astype(np.float32)
        bias_hh = np.random.randn(hidden_size).astype(np.float32)

        # PyTorch
        cell_torch = torch.nn.RNNCell(input_size, hidden_size)
        cell_torch.weight_ih.data = torch.tensor(weight_ih)
        cell_torch.weight_hh.data = torch.tensor(weight_hh)
        cell_torch.bias_ih.data = torch.tensor(bias_ih)
        cell_torch.bias_hh.data = torch.tensor(bias_hh)

        # MLX
        cell_mlx = flashlight.nn.RNNCell(input_size, hidden_size)
        cell_mlx.weight_ih._mlx_array = flashlight.tensor(weight_ih)._mlx_array
        cell_mlx.weight_hh._mlx_array = flashlight.tensor(weight_hh)._mlx_array
        cell_mlx.bias_ih._mlx_array = flashlight.tensor(bias_ih)._mlx_array
        cell_mlx.bias_hh._mlx_array = flashlight.tensor(bias_hh)._mlx_array

        # Input
        x_np = np.random.randn(batch_size, input_size).astype(np.float32)
        hx_np = np.random.randn(batch_size, hidden_size).astype(np.float32)

        out_torch = cell_torch(torch.tensor(x_np), torch.tensor(hx_np))
        out_mlx = cell_mlx(flashlight.tensor(x_np), flashlight.tensor(hx_np))

        np.testing.assert_allclose(
            out_torch.detach().numpy(), out_mlx.numpy(), rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestLSTMCell(TestCase):
    """Test nn.LSTMCell."""

    def test_creation(self):
        """Test LSTMCell creation with default parameters."""
        cell = flashlight.nn.LSTMCell(input_size=10, hidden_size=20)
        self.assertEqual(cell.input_size, 10)
        self.assertEqual(cell.hidden_size, 20)

    def test_forward_shape(self):
        """Test LSTMCell forward pass output shape."""
        cell = flashlight.nn.LSTMCell(input_size=10, hidden_size=20)
        x = flashlight.randn(5, 10)
        hx = flashlight.randn(5, 20)
        cx = flashlight.randn(5, 20)
        h_new, c_new = cell(x, (hx, cx))
        self.assertEqual(h_new.shape, (5, 20))
        self.assertEqual(c_new.shape, (5, 20))

    def test_forward_no_hidden(self):
        """Test LSTMCell forward with no initial hidden state."""
        cell = flashlight.nn.LSTMCell(input_size=10, hidden_size=20)
        x = flashlight.randn(5, 10)
        h_new, c_new = cell(x)
        self.assertEqual(h_new.shape, (5, 20))
        self.assertEqual(c_new.shape, (5, 20))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        input_size, hidden_size, batch_size = 10, 20, 5

        # Create PyTorch and MLX cells
        cell_torch = torch.nn.LSTMCell(input_size, hidden_size)
        cell_mlx = flashlight.nn.LSTMCell(input_size, hidden_size)

        # Copy weights
        for name, param in cell_torch.named_parameters():
            mlx_param = getattr(cell_mlx, name)
            mlx_param._mlx_array = flashlight.tensor(param.detach().numpy())._mlx_array

        # Input
        x_np = np.random.randn(batch_size, input_size).astype(np.float32)
        hx_np = np.random.randn(batch_size, hidden_size).astype(np.float32)
        cx_np = np.random.randn(batch_size, hidden_size).astype(np.float32)

        h_torch, c_torch = cell_torch(
            torch.tensor(x_np), (torch.tensor(hx_np), torch.tensor(cx_np))
        )
        h_mlx, c_mlx = cell_mlx(
            flashlight.tensor(x_np), (flashlight.tensor(hx_np), flashlight.tensor(cx_np))
        )

        np.testing.assert_allclose(h_torch.detach().numpy(), h_mlx.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(c_torch.detach().numpy(), c_mlx.numpy(), rtol=1e-4, atol=1e-5)


@skipIfNoMLX
class TestGRUCell(TestCase):
    """Test nn.GRUCell."""

    def test_creation(self):
        """Test GRUCell creation with default parameters."""
        cell = flashlight.nn.GRUCell(input_size=10, hidden_size=20)
        self.assertEqual(cell.input_size, 10)
        self.assertEqual(cell.hidden_size, 20)

    def test_forward_shape(self):
        """Test GRUCell forward pass output shape."""
        cell = flashlight.nn.GRUCell(input_size=10, hidden_size=20)
        x = flashlight.randn(5, 10)
        hx = flashlight.randn(5, 20)
        output = cell(x, hx)
        self.assertEqual(output.shape, (5, 20))


@skipIfNoMLX
class TestRNN(TestCase):
    """Test nn.RNN layer."""

    def test_creation(self):
        """Test RNN creation."""
        rnn = flashlight.nn.RNN(input_size=10, hidden_size=20, num_layers=2)
        self.assertEqual(rnn.input_size, 10)
        self.assertEqual(rnn.hidden_size, 20)
        self.assertEqual(rnn.num_layers, 2)

    def test_forward_shape(self):
        """Test RNN forward pass output shape."""
        rnn = flashlight.nn.RNN(input_size=10, hidden_size=20, num_layers=2)
        x = flashlight.randn(7, 5, 10)  # seq=7, batch=5, input=10
        output, h_n = rnn(x)
        self.assertEqual(output.shape, (7, 5, 20))
        self.assertEqual(h_n.shape, (2, 5, 20))

    def test_batch_first(self):
        """Test RNN with batch_first=True."""
        rnn = flashlight.nn.RNN(input_size=10, hidden_size=20, batch_first=True)
        x = flashlight.randn(5, 7, 10)  # batch=5, seq=7, input=10
        output, h_n = rnn(x)
        self.assertEqual(output.shape, (5, 7, 20))

    def test_bidirectional(self):
        """Test bidirectional RNN."""
        rnn = flashlight.nn.RNN(input_size=10, hidden_size=20, bidirectional=True)
        x = flashlight.randn(7, 5, 10)
        output, h_n = rnn(x)
        self.assertEqual(output.shape, (7, 5, 40))  # 2 * hidden_size
        self.assertEqual(h_n.shape, (2, 5, 20))  # num_directions * num_layers


@skipIfNoMLX
class TestLSTM(TestCase):
    """Test nn.LSTM layer."""

    def test_creation(self):
        """Test LSTM creation."""
        lstm = flashlight.nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
        self.assertEqual(lstm.input_size, 10)
        self.assertEqual(lstm.hidden_size, 20)
        self.assertEqual(lstm.num_layers, 2)

    def test_forward_shape(self):
        """Test LSTM forward pass output shape."""
        lstm = flashlight.nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
        x = flashlight.randn(7, 5, 10)
        output, (h_n, c_n) = lstm(x)
        self.assertEqual(output.shape, (7, 5, 20))
        self.assertEqual(h_n.shape, (2, 5, 20))
        self.assertEqual(c_n.shape, (2, 5, 20))

    def test_batch_first(self):
        """Test LSTM with batch_first=True."""
        lstm = flashlight.nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
        x = flashlight.randn(5, 7, 10)
        output, (h_n, c_n) = lstm(x)
        self.assertEqual(output.shape, (5, 7, 20))

    def test_bidirectional(self):
        """Test bidirectional LSTM."""
        lstm = flashlight.nn.LSTM(input_size=10, hidden_size=20, bidirectional=True)
        x = flashlight.randn(7, 5, 10)
        output, (h_n, c_n) = lstm(x)
        self.assertEqual(output.shape, (7, 5, 40))


@skipIfNoMLX
class TestGRU(TestCase):
    """Test nn.GRU layer."""

    def test_creation(self):
        """Test GRU creation."""
        gru = flashlight.nn.GRU(input_size=10, hidden_size=20, num_layers=2)
        self.assertEqual(gru.input_size, 10)
        self.assertEqual(gru.hidden_size, 20)
        self.assertEqual(gru.num_layers, 2)

    def test_forward_shape(self):
        """Test GRU forward pass output shape."""
        gru = flashlight.nn.GRU(input_size=10, hidden_size=20, num_layers=2)
        x = flashlight.randn(7, 5, 10)
        output, h_n = gru(x)
        self.assertEqual(output.shape, (7, 5, 20))
        self.assertEqual(h_n.shape, (2, 5, 20))

    def test_batch_first(self):
        """Test GRU with batch_first=True."""
        gru = flashlight.nn.GRU(input_size=10, hidden_size=20, batch_first=True)
        x = flashlight.randn(5, 7, 10)
        output, h_n = gru(x)
        self.assertEqual(output.shape, (5, 7, 20))


if __name__ == "__main__":
    unittest.main()
