"""
RNN Layer Parity Tests

Tests numerical parity between flashlight RNN layers and PyTorch.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import flashlight
import flashlight.nn as nn


def copy_rnn_weights(mlx_layer, torch_layer):
    """Copy weights from PyTorch RNN to flashlight RNN."""
    import mlx.core as mx

    # For LSTM/GRU, weight names follow pattern: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
    if hasattr(torch_layer, "weight_ih_l0"):
        mlx_layer.weight_ih_l0 = nn.Parameter(
            flashlight.tensor(torch_layer.weight_ih_l0.detach().numpy())
        )
        mlx_layer.weight_hh_l0 = nn.Parameter(
            flashlight.tensor(torch_layer.weight_hh_l0.detach().numpy())
        )
        if torch_layer.bias:
            mlx_layer.bias_ih_l0 = nn.Parameter(
                flashlight.tensor(torch_layer.bias_ih_l0.detach().numpy())
            )
            mlx_layer.bias_hh_l0 = nn.Parameter(
                flashlight.tensor(torch_layer.bias_hh_l0.detach().numpy())
            )


class TestLSTMParity:
    """Test LSTM parity with PyTorch."""

    @pytest.mark.parity
    def test_lstm_forward_parity(self):
        """Test LSTM forward pass matches PyTorch."""
        batch, seq_len, input_size, hidden_size = 4, 10, 32, 64

        # Create layers
        torch_lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        mlx_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Copy weights
        copy_rnn_weights(mlx_lstm, torch_lstm)

        # Create input
        np.random.seed(42)
        x_np = np.random.randn(batch, seq_len, input_size).astype(np.float32)

        torch_x = torch.tensor(x_np)
        mlx_x = flashlight.tensor(x_np)

        # Forward pass
        torch_out, (torch_h, torch_c) = torch_lstm(torch_x)
        mlx_out, (mlx_h, mlx_c) = mlx_lstm(mlx_x)

        # Compare outputs
        torch_out_np = torch_out.detach().numpy()
        mlx_out_np = np.array(mlx_out.tolist())

        max_diff = np.max(np.abs(torch_out_np - mlx_out_np))
        assert max_diff < 1e-4, f"LSTM output mismatch: max diff = {max_diff}"

    @pytest.mark.parity
    def test_lstm_different_seq_lengths(self):
        """Test LSTM with various sequence lengths."""
        input_size, hidden_size = 16, 32

        torch_lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        mlx_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        copy_rnn_weights(mlx_lstm, torch_lstm)

        for seq_len in [1, 5, 20, 50]:
            np.random.seed(123)
            x_np = np.random.randn(2, seq_len, input_size).astype(np.float32)

            torch_out, _ = torch_lstm(torch.tensor(x_np))
            mlx_out, _ = mlx_lstm(flashlight.tensor(x_np))

            max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
            assert max_diff < 1e-4, f"LSTM seq_len={seq_len} mismatch: {max_diff}"


class TestGRUParity:
    """Test GRU parity with PyTorch."""

    @pytest.mark.parity
    def test_gru_forward_parity(self):
        """Test GRU forward pass matches PyTorch."""
        batch, seq_len, input_size, hidden_size = 4, 10, 32, 64

        torch_gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        mlx_gru = nn.GRU(input_size, hidden_size, batch_first=True)

        copy_rnn_weights(mlx_gru, torch_gru)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq_len, input_size).astype(np.float32)

        torch_out, torch_h = torch_gru(torch.tensor(x_np))
        mlx_out, mlx_h = mlx_gru(flashlight.tensor(x_np))

        torch_out_np = torch_out.detach().numpy()
        mlx_out_np = np.array(mlx_out.tolist())

        max_diff = np.max(np.abs(torch_out_np - mlx_out_np))
        assert max_diff < 1e-4, f"GRU output mismatch: max diff = {max_diff}"

    @pytest.mark.parity
    def test_gru_different_hidden_sizes(self):
        """Test GRU with various hidden sizes."""
        input_size = 16

        for hidden_size in [8, 32, 128]:
            torch_gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
            mlx_gru = nn.GRU(input_size, hidden_size, batch_first=True)
            copy_rnn_weights(mlx_gru, torch_gru)

            np.random.seed(456)
            x_np = np.random.randn(2, 10, input_size).astype(np.float32)

            torch_out, _ = torch_gru(torch.tensor(x_np))
            mlx_out, _ = mlx_gru(flashlight.tensor(x_np))

            max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
            assert max_diff < 1e-4, f"GRU hidden_size={hidden_size} mismatch: {max_diff}"


class TestRNNCellParity:
    """Test RNN cell parity with PyTorch."""

    @pytest.mark.parity
    def test_lstm_cell_parity(self):
        """Test LSTMCell matches PyTorch."""
        batch, input_size, hidden_size = 4, 32, 64

        torch_cell = torch.nn.LSTMCell(input_size, hidden_size)
        mlx_cell = nn.LSTMCell(input_size, hidden_size)

        # Copy weights
        mlx_cell.weight_ih = nn.Parameter(flashlight.tensor(torch_cell.weight_ih.detach().numpy()))
        mlx_cell.weight_hh = nn.Parameter(flashlight.tensor(torch_cell.weight_hh.detach().numpy()))
        mlx_cell.bias_ih = nn.Parameter(flashlight.tensor(torch_cell.bias_ih.detach().numpy()))
        mlx_cell.bias_hh = nn.Parameter(flashlight.tensor(torch_cell.bias_hh.detach().numpy()))

        np.random.seed(42)
        x_np = np.random.randn(batch, input_size).astype(np.float32)
        h_np = np.random.randn(batch, hidden_size).astype(np.float32)
        c_np = np.random.randn(batch, hidden_size).astype(np.float32)

        torch_h, torch_c = torch_cell(torch.tensor(x_np), (torch.tensor(h_np), torch.tensor(c_np)))
        mlx_h, mlx_c = mlx_cell(
            flashlight.tensor(x_np), (flashlight.tensor(h_np), flashlight.tensor(c_np))
        )

        h_diff = np.max(np.abs(torch_h.detach().numpy() - np.array(mlx_h.tolist())))
        c_diff = np.max(np.abs(torch_c.detach().numpy() - np.array(mlx_c.tolist())))

        assert h_diff < 1e-5, f"LSTMCell h mismatch: {h_diff}"
        assert c_diff < 1e-5, f"LSTMCell c mismatch: {c_diff}"

    @pytest.mark.parity
    def test_gru_cell_parity(self):
        """Test GRUCell matches PyTorch."""
        batch, input_size, hidden_size = 4, 32, 64

        torch_cell = torch.nn.GRUCell(input_size, hidden_size)
        mlx_cell = nn.GRUCell(input_size, hidden_size)

        # Copy weights
        mlx_cell.weight_ih = nn.Parameter(flashlight.tensor(torch_cell.weight_ih.detach().numpy()))
        mlx_cell.weight_hh = nn.Parameter(flashlight.tensor(torch_cell.weight_hh.detach().numpy()))
        mlx_cell.bias_ih = nn.Parameter(flashlight.tensor(torch_cell.bias_ih.detach().numpy()))
        mlx_cell.bias_hh = nn.Parameter(flashlight.tensor(torch_cell.bias_hh.detach().numpy()))

        np.random.seed(42)
        x_np = np.random.randn(batch, input_size).astype(np.float32)
        h_np = np.random.randn(batch, hidden_size).astype(np.float32)

        torch_h = torch_cell(torch.tensor(x_np), torch.tensor(h_np))
        mlx_h = mlx_cell(flashlight.tensor(x_np), flashlight.tensor(h_np))

        h_diff = np.max(np.abs(torch_h.detach().numpy() - np.array(mlx_h.tolist())))
        assert h_diff < 1e-5, f"GRUCell h mismatch: {h_diff}"
