"""
Recurrent Neural Network Layers

Implements RNN, LSTM, and GRU layers for neural networks.
"""

import math
from typing import Any, Optional, Tuple, Union

import mlx.core as mx

from ...tensor import Tensor
from ..module import Module
from ..parameter import Parameter


def _lstm_cell_impl(input, h, c, weight_ih, weight_hh, bias_ih, bias_hh, use_bias):
    """Core LSTM cell computation."""
    igates = mx.matmul(input, weight_ih.T)
    hgates = mx.matmul(h, weight_hh.T)

    if use_bias:
        gates = igates + bias_ih + hgates + bias_hh
    else:
        gates = igates + hgates

    # Split gates - PyTorch order is i, f, g, o
    i, f, g, o = mx.split(gates, 4, axis=1)

    # Apply gate activations
    i = mx.sigmoid(i)
    f = mx.sigmoid(f)
    g = mx.tanh(g)
    o = mx.sigmoid(o)

    # Update cell and hidden state
    c_new = f * c + i * g
    h_new = o * mx.tanh(c_new)

    return h_new, c_new


def _gru_cell_impl(input, hx, weight_ih, weight_hh, bias_ih, bias_hh, use_bias, hidden_size):
    """Core GRU cell computation."""
    igates = mx.matmul(input, weight_ih.T)
    hgates = mx.matmul(hx, weight_hh.T)

    if use_bias:
        igates = igates + bias_ih
        hgates = hgates + bias_hh

    # Split gates - PyTorch order is r, z, n
    H = hidden_size
    ir, iz, in_ = igates[:, :H], igates[:, H : 2 * H], igates[:, 2 * H :]
    hr, hz, hn = hgates[:, :H], hgates[:, H : 2 * H], hgates[:, 2 * H :]

    # Apply gate activations
    r = mx.sigmoid(ir + hr)
    z = mx.sigmoid(iz + hz)
    n = mx.tanh(in_ + r * hn)

    # Update hidden state
    hy = (1 - z) * n + z * hx

    return hy


# Compiled versions for better performance
_lstm_cell_compiled = mx.compile(_lstm_cell_impl)
_gru_cell_compiled = mx.compile(_gru_cell_impl)


class RNNCellBase(Module):
    """
    Base class for RNN cells.

    This is an abstract base class that RNNCell, LSTMCell, and GRUCell inherit from.
    It provides common functionality and interface for all cell types.
    """

    __constants__ = ["input_size", "hidden_size", "bias"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, bias={self.bias}"

    def reset_parameters(self) -> None:
        """Reset parameters to initial values."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight._mlx_array = mx.random.uniform(-stdv, stdv, weight.shape)


class RNNBase(Module):
    """
    Base class for RNN modules (RNN, LSTM, GRU).

    This is an abstract base class that provides common functionality for
    multi-layer recurrent neural networks.
    """

    __constants__ = [
        "mode",
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
    ]

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1

    def reset_parameters(self) -> None:
        """Reset parameters to initial values."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight._mlx_array = mx.random.uniform(-stdv, stdv, weight.shape)

    def flatten_parameters(self) -> None:
        """
        Flatten parameters for efficient computation.

        This is a no-op in MLX as we don't need to flatten parameters
        for GPU optimization like in CUDA.
        """
        pass

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional}"
        )


class RNNCell(Module):
    """
    An Elman RNN cell.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        bias: If False, the layer does not use bias weights (default: True)
        nonlinearity: Non-linearity to use ('tanh' or 'relu')

    Shape:
        - Input: [batch, input_size]
        - Hidden: [batch, hidden_size]
        - Output: [batch, hidden_size]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        # Initialize weights
        stdv = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(
            Tensor._from_mlx_array(mx.random.uniform(-stdv, stdv, (hidden_size, input_size)))
        )
        self.weight_hh = Parameter(
            Tensor._from_mlx_array(mx.random.uniform(-stdv, stdv, (hidden_size, hidden_size)))
        )

        if bias:
            self.bias_ih = Parameter(Tensor._from_mlx_array(mx.zeros(hidden_size)))
            self.bias_hh = Parameter(Tensor._from_mlx_array(mx.zeros(hidden_size)))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """Apply RNN cell."""
        if hx is None:
            hx = Tensor._from_mlx_array(mx.zeros((input.shape[0], self.hidden_size)))

        # h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
        igates = mx.matmul(input._mlx_array, self.weight_ih._mlx_array.T)
        hgates = mx.matmul(hx._mlx_array, self.weight_hh._mlx_array.T)

        if self.bias:
            igates = igates + self.bias_ih._mlx_array
            hgates = hgates + self.bias_hh._mlx_array

        if self.nonlinearity == "tanh":
            hy = mx.tanh(igates + hgates)
        else:
            hy = mx.maximum(igates + hgates, 0)  # relu

        return Tensor._from_mlx_array(hy)

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, bias={self.bias}, nonlinearity={self.nonlinearity}"


class LSTMCell(Module):
    """
    A long short-term memory (LSTM) cell.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        bias: If False, the layer does not use bias weights (default: True)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [batch, input_size]
        - Hidden: tuple of (h, c), each [batch, hidden_size]
        - Output: tuple of (h', c'), each [batch, hidden_size]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize weights (4 gates: i, f, g, o)
        stdv = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(
            Tensor._from_mlx_array(mx.random.uniform(-stdv, stdv, (4 * hidden_size, input_size)))
        )
        self.weight_hh = Parameter(
            Tensor._from_mlx_array(mx.random.uniform(-stdv, stdv, (4 * hidden_size, hidden_size)))
        )

        if bias:
            self.bias_ih = Parameter(Tensor._from_mlx_array(mx.zeros(4 * hidden_size)))
            self.bias_hh = Parameter(Tensor._from_mlx_array(mx.zeros(4 * hidden_size)))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """Apply LSTM cell."""
        batch_size = input.shape[0]

        if hx is None:
            zeros = mx.zeros((batch_size, self.hidden_size))
            h = Tensor._from_mlx_array(zeros)
            c = Tensor._from_mlx_array(zeros)
        else:
            h, c = hx

        # Compute gates
        igates = mx.matmul(input._mlx_array, self.weight_ih._mlx_array.T)
        hgates = mx.matmul(h._mlx_array, self.weight_hh._mlx_array.T)

        if self.bias:
            gates = igates + self.bias_ih._mlx_array + hgates + self.bias_hh._mlx_array
        else:
            gates = igates + hgates

        # Split gates
        i, f, g, o = mx.split(gates, 4, axis=1)

        # Apply gate activations
        i = mx.sigmoid(i)  # input gate
        f = mx.sigmoid(f)  # forget gate
        g = mx.tanh(g)  # cell gate
        o = mx.sigmoid(o)  # output gate

        # Update cell and hidden state
        c_new = f * c._mlx_array + i * g
        h_new = o * mx.tanh(c_new)

        return Tensor._from_mlx_array(h_new), Tensor._from_mlx_array(c_new)

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, bias={self.bias}"


class GRUCell(Module):
    """
    A gated recurrent unit (GRU) cell.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        bias: If False, the layer does not use bias weights (default: True)
        device: Device to place the layer on (ignored, MLX uses unified memory)
        dtype: Data type for the layer parameters (ignored, uses default)

    Shape:
        - Input: [batch, input_size]
        - Hidden: [batch, hidden_size]
        - Output: [batch, hidden_size]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
    ):
        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize weights (3 gates: r, z, n)
        stdv = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(
            Tensor._from_mlx_array(mx.random.uniform(-stdv, stdv, (3 * hidden_size, input_size)))
        )
        self.weight_hh = Parameter(
            Tensor._from_mlx_array(mx.random.uniform(-stdv, stdv, (3 * hidden_size, hidden_size)))
        )

        if bias:
            self.bias_ih = Parameter(Tensor._from_mlx_array(mx.zeros(3 * hidden_size)))
            self.bias_hh = Parameter(Tensor._from_mlx_array(mx.zeros(3 * hidden_size)))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """Apply GRU cell."""
        if hx is None:
            hx = Tensor._from_mlx_array(mx.zeros((input.shape[0], self.hidden_size)))

        # Compute gates
        igates = mx.matmul(input._mlx_array, self.weight_ih._mlx_array.T)
        hgates = mx.matmul(hx._mlx_array, self.weight_hh._mlx_array.T)

        if self.bias:
            igates = igates + self.bias_ih._mlx_array
            hgates = hgates + self.bias_hh._mlx_array

        # Split gates
        ir, iz, in_ = mx.split(igates, 3, axis=1)
        hr, hz, hn = mx.split(hgates, 3, axis=1)

        # Apply gate activations
        r = mx.sigmoid(ir + hr)  # reset gate
        z = mx.sigmoid(iz + hz)  # update gate
        n = mx.tanh(in_ + r * hn)  # new gate

        # Update hidden state
        hy = (1 - z) * n + z * hx._mlx_array

        return Tensor._from_mlx_array(hy)

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, bias={self.bias}"


class RNN(Module):
    """
    Multi-layer Elman RNN.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers (default: 1)
        nonlinearity: Non-linearity to use ('tanh' or 'relu')
        bias: If False, the layer does not use bias weights (default: True)
        batch_first: If True, input/output shape is [batch, seq, feature] (default: False)
        dropout: Dropout probability (default: 0)
        bidirectional: If True, becomes a bidirectional RNN (default: False)

    Shape:
        - Input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        - Hidden: [num_layers * num_directions, batch, hidden_size]
        - Output: [seq_len, batch, hidden_size * num_directions] or [batch, seq, hidden_size * num_directions]
    """

    def __init__(self, *args, **kwargs):
        # Parse args/kwargs to match PyTorch's flexible signature
        if len(args) >= 1:
            input_size = args[0]
            args = args[1:]
        else:
            input_size = kwargs.pop("input_size")

        if len(args) >= 1:
            hidden_size = args[0]
            args = args[1:]
        else:
            hidden_size = kwargs.pop("hidden_size")

        num_layers = kwargs.pop("num_layers", 1)
        nonlinearity = kwargs.pop("nonlinearity", "tanh")
        bias = kwargs.pop("bias", True)
        batch_first = kwargs.pop("batch_first", False)
        dropout = kwargs.pop("dropout", 0.0)
        bidirectional = kwargs.pop("bidirectional", False)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create weights directly with PyTorch-compatible names
        # instead of using cells (for proper weight naming)
        self._cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            for direction in range(self.num_directions):
                suffix = f"_l{layer}" if direction == 0 else f"_l{layer}_reverse"

                # Create weights
                stdv = 1.0 / math.sqrt(hidden_size)
                weight_ih = Parameter(
                    Tensor._from_mlx_array(
                        mx.random.uniform(-stdv, stdv, (hidden_size, layer_input_size))
                    )
                )
                weight_hh = Parameter(
                    Tensor._from_mlx_array(
                        mx.random.uniform(-stdv, stdv, (hidden_size, hidden_size))
                    )
                )
                setattr(self, f"weight_ih{suffix}", weight_ih)
                setattr(self, f"weight_hh{suffix}", weight_hh)

                if bias:
                    bias_ih = Parameter(Tensor._from_mlx_array(mx.zeros(hidden_size)))
                    bias_hh = Parameter(Tensor._from_mlx_array(mx.zeros(hidden_size)))
                    setattr(self, f"bias_ih{suffix}", bias_ih)
                    setattr(self, f"bias_hh{suffix}", bias_hh)

    def _get_weights(self, layer: int, direction: int):
        """Get weights for a specific layer and direction."""
        suffix = f"_l{layer}" if direction == 0 else f"_l{layer}_reverse"
        weight_ih = getattr(self, f"weight_ih{suffix}")
        weight_hh = getattr(self, f"weight_hh{suffix}")
        if self.bias:
            bias_ih = getattr(self, f"bias_ih{suffix}")
            bias_hh = getattr(self, f"bias_hh{suffix}")
        else:
            bias_ih = None
            bias_hh = None
        return weight_ih, weight_hh, bias_ih, bias_hh

    def _rnn_cell_forward(self, input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
        """Single RNN cell forward pass."""
        # h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)
        igates = mx.matmul(input, weight_ih._mlx_array.T)
        hgates = mx.matmul(hx, weight_hh._mlx_array.T)

        if self.bias:
            igates = igates + bias_ih._mlx_array
            hgates = hgates + bias_hh._mlx_array

        if self.nonlinearity == "tanh":
            hy = mx.tanh(igates + hgates)
        else:
            hy = mx.maximum(igates + hgates, 0)  # relu

        return hy

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Apply multi-layer RNN."""
        if self.batch_first:
            # [batch, seq, feature] -> [seq, batch, feature]
            input = Tensor._from_mlx_array(mx.transpose(input._mlx_array, [1, 0, 2]))

        seq_len, batch_size, _ = input.shape

        if hx is None:
            num_directions = self.num_directions
            hx = Tensor._from_mlx_array(
                mx.zeros((self.num_layers * num_directions, batch_size, self.hidden_size))
            )

        # Process each layer
        output = input._mlx_array
        hidden_states = []

        for layer in range(self.num_layers):
            # Get weights for forward direction
            weight_ih, weight_hh, bias_ih, bias_hh = self._get_weights(layer, 0)

            layer_output_fwd = []
            h_fwd = hx._mlx_array[layer * self.num_directions]

            # Forward direction
            for t in range(seq_len):
                x_t = output[t]
                h_fwd = self._rnn_cell_forward(x_t, h_fwd, weight_ih, weight_hh, bias_ih, bias_hh)
                layer_output_fwd.append(h_fwd)

            hidden_states.append(h_fwd)

            if self.bidirectional:
                # Get weights for backward direction
                weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r = self._get_weights(layer, 1)

                layer_output_bwd = []
                h_bwd = hx._mlx_array[layer * self.num_directions + 1]

                # Backward direction
                for t in range(seq_len - 1, -1, -1):
                    x_t = output[t]
                    h_bwd = self._rnn_cell_forward(
                        x_t, h_bwd, weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r
                    )
                    layer_output_bwd.insert(0, h_bwd)

                hidden_states.append(h_bwd)

                # Concatenate forward and backward
                output = mx.concatenate(
                    [mx.stack(layer_output_fwd, axis=0), mx.stack(layer_output_bwd, axis=0)], axis=2
                )
            else:
                output = mx.stack(layer_output_fwd, axis=0)

        # Stack hidden states
        h_n = Tensor._from_mlx_array(mx.stack(hidden_states, axis=0))
        output = Tensor._from_mlx_array(output)

        if self.batch_first:
            # [seq, batch, feature] -> [batch, seq, feature]
            output = Tensor._from_mlx_array(mx.transpose(output._mlx_array, [1, 0, 2]))

        return output, h_n

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"nonlinearity={self.nonlinearity}, batch_first={self.batch_first}, "
            f"bidirectional={self.bidirectional}"
        )


class LSTM(Module):
    """
    Multi-layer Long Short-Term Memory (LSTM) RNN.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers (default: 1)
        bias: If False, the layer does not use bias weights (default: True)
        batch_first: If True, input/output shape is [batch, seq, feature] (default: False)
        dropout: Dropout probability (default: 0)
        bidirectional: If True, becomes a bidirectional LSTM (default: False)

    Shape:
        - Input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        - Hidden: tuple of (h, c), each [num_layers * num_directions, batch, hidden_size]
        - Output: [seq_len, batch, hidden_size * num_directions]

    Note:
        For single-layer unidirectional LSTM without initial hidden state, this
        implementation can use MLX's native LSTM for better performance.
    """

    def __init__(self, *args, **kwargs):
        # Parse args/kwargs to match PyTorch's flexible signature
        if len(args) >= 1:
            input_size = args[0]
            args = args[1:]
        else:
            input_size = kwargs.pop("input_size")

        if len(args) >= 1:
            hidden_size = args[0]
            args = args[1:]
        else:
            hidden_size = kwargs.pop("hidden_size")

        num_layers = kwargs.pop("num_layers", 1)
        bias = kwargs.pop("bias", True)
        batch_first = kwargs.pop("batch_first", False)
        dropout = kwargs.pop("dropout", 0.0)
        bidirectional = kwargs.pop("bidirectional", False)
        proj_size = kwargs.pop("proj_size", 0)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        # device and dtype are accepted for PyTorch compatibility but ignored
        if proj_size < 0:
            raise ValueError(f"proj_size should be a non-negative integer, got {proj_size}")
        if proj_size >= hidden_size:
            raise ValueError(
                f"proj_size ({proj_size}) must be smaller than hidden_size ({hidden_size})"
            )

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1

        # When proj_size > 0, the output hidden state is projected to proj_size
        # The real hidden state size used in recurrence is still hidden_size
        # but the output h is projected to proj_size before being used for:
        # 1. The next layer's input
        # 2. The recurrent connection (weight_hh connects from proj_size, not hidden_size)
        self._real_hidden_size = proj_size if proj_size > 0 else hidden_size

        # Create weights directly with PyTorch-compatible names
        for layer in range(num_layers):
            # Input size for this layer
            if layer == 0:
                layer_input_size = input_size
            else:
                # After first layer, input comes from previous layer's output
                # which has size (proj_size if proj_size > 0 else hidden_size) * num_directions
                layer_input_size = self._real_hidden_size * self.num_directions

            for direction in range(self.num_directions):
                suffix = f"_l{layer}" if direction == 0 else f"_l{layer}_reverse"

                # Create weights (4 gates: i, f, g, o)
                stdv = 1.0 / math.sqrt(hidden_size)
                weight_ih = Parameter(
                    Tensor._from_mlx_array(
                        mx.random.uniform(-stdv, stdv, (4 * hidden_size, layer_input_size))
                    )
                )
                # weight_hh connects from projected hidden state (or full hidden if no projection)
                weight_hh = Parameter(
                    Tensor._from_mlx_array(
                        mx.random.uniform(-stdv, stdv, (4 * hidden_size, self._real_hidden_size))
                    )
                )
                setattr(self, f"weight_ih{suffix}", weight_ih)
                setattr(self, f"weight_hh{suffix}", weight_hh)

                if bias:
                    bias_ih = Parameter(Tensor._from_mlx_array(mx.zeros(4 * hidden_size)))
                    bias_hh = Parameter(Tensor._from_mlx_array(mx.zeros(4 * hidden_size)))
                    setattr(self, f"bias_ih{suffix}", bias_ih)
                    setattr(self, f"bias_hh{suffix}", bias_hh)

                # Projection weight: projects hidden_size -> proj_size
                if proj_size > 0:
                    weight_hr = Parameter(
                        Tensor._from_mlx_array(
                            mx.random.uniform(-stdv, stdv, (proj_size, hidden_size))
                        )
                    )
                    setattr(self, f"weight_hr{suffix}", weight_hr)

    def _get_weights(self, layer: int, direction: int):
        """Get weights for a specific layer and direction."""
        suffix = f"_l{layer}" if direction == 0 else f"_l{layer}_reverse"
        weight_ih = getattr(self, f"weight_ih{suffix}")
        weight_hh = getattr(self, f"weight_hh{suffix}")
        if self.bias:
            bias_ih = getattr(self, f"bias_ih{suffix}")
            bias_hh = getattr(self, f"bias_hh{suffix}")
        else:
            bias_ih = None
            bias_hh = None
        # Get projection weight if proj_size > 0
        if self.proj_size > 0:
            weight_hr = getattr(self, f"weight_hr{suffix}")
        else:
            weight_hr = None
        return weight_ih, weight_hh, bias_ih, bias_hh, weight_hr

    def _lstm_cell_forward(
        self, input, h, c, weight_ih, weight_hh, bias_ih, bias_hh, weight_hr=None
    ):
        """Single LSTM cell forward pass with optional projection."""
        # Use compiled version for better performance
        h_new, c_new = _lstm_cell_compiled(
            input,
            h,
            c,
            weight_ih._mlx_array,
            weight_hh._mlx_array,
            bias_ih._mlx_array if bias_ih is not None else None,
            bias_hh._mlx_array if bias_hh is not None else None,
            self.bias,
        )

        # Apply projection if proj_size > 0
        if weight_hr is not None:
            # Project h from hidden_size to proj_size
            h_new = mx.matmul(h_new, weight_hr._mlx_array.T)

        return h_new, c_new

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Apply multi-layer LSTM."""

        if self.batch_first:
            input = Tensor._from_mlx_array(mx.transpose(input._mlx_array, [1, 0, 2]))

        seq_len, batch_size, _ = input.shape

        if hx is None:
            num_directions = self.num_directions
            # h has shape [num_layers * num_directions, batch, real_hidden_size]
            # where real_hidden_size = proj_size if proj_size > 0 else hidden_size
            h_zeros = mx.zeros(
                (self.num_layers * num_directions, batch_size, self._real_hidden_size)
            )
            # c always has shape [num_layers * num_directions, batch, hidden_size]
            c_zeros = mx.zeros((self.num_layers * num_directions, batch_size, self.hidden_size))
            hx = (Tensor._from_mlx_array(h_zeros), Tensor._from_mlx_array(c_zeros))

        h0, c0 = hx
        output = input._mlx_array
        h_states = []
        c_states = []

        for layer in range(self.num_layers):
            # Get weights for forward direction (now includes weight_hr)
            weight_ih, weight_hh, bias_ih, bias_hh, weight_hr = self._get_weights(layer, 0)

            layer_output_fwd = []
            h_fwd = h0._mlx_array[layer * self.num_directions]
            c_fwd = c0._mlx_array[layer * self.num_directions]

            for t in range(seq_len):
                x_t = output[t]
                h_fwd, c_fwd = self._lstm_cell_forward(
                    x_t, h_fwd, c_fwd, weight_ih, weight_hh, bias_ih, bias_hh, weight_hr
                )
                layer_output_fwd.append(h_fwd)

            h_states.append(h_fwd)
            c_states.append(c_fwd)

            if self.bidirectional:
                # Get weights for backward direction (now includes weight_hr)
                weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r, weight_hr_r = self._get_weights(
                    layer, 1
                )

                layer_output_bwd = []
                h_bwd = h0._mlx_array[layer * self.num_directions + 1]
                c_bwd = c0._mlx_array[layer * self.num_directions + 1]

                for t in range(seq_len - 1, -1, -1):
                    x_t = output[t]
                    h_bwd, c_bwd = self._lstm_cell_forward(
                        x_t,
                        h_bwd,
                        c_bwd,
                        weight_ih_r,
                        weight_hh_r,
                        bias_ih_r,
                        bias_hh_r,
                        weight_hr_r,
                    )
                    layer_output_bwd.insert(0, h_bwd)

                h_states.append(h_bwd)
                c_states.append(c_bwd)

                output = mx.concatenate(
                    [mx.stack(layer_output_fwd, axis=0), mx.stack(layer_output_bwd, axis=0)], axis=2
                )
            else:
                output = mx.stack(layer_output_fwd, axis=0)

        h_n = Tensor._from_mlx_array(mx.stack(h_states, axis=0))
        c_n = Tensor._from_mlx_array(mx.stack(c_states, axis=0))
        output = Tensor._from_mlx_array(output)

        if self.batch_first:
            output = Tensor._from_mlx_array(mx.transpose(output._mlx_array, [1, 0, 2]))

        return output, (h_n, c_n)

    def extra_repr(self) -> str:
        s = (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional}"
        )
        if self.proj_size > 0:
            s += f", proj_size={self.proj_size}"
        return s


class GRU(Module):
    """
    Multi-layer Gated Recurrent Unit (GRU) RNN.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers (default: 1)
        bias: If False, the layer does not use bias weights (default: True)
        batch_first: If True, input/output shape is [batch, seq, feature] (default: False)
        dropout: Dropout probability (default: 0)
        bidirectional: If True, becomes a bidirectional GRU (default: False)

    Shape:
        - Input: [seq_len, batch, input_size] or [batch, seq_len, input_size] if batch_first
        - Hidden: [num_layers * num_directions, batch, hidden_size]
        - Output: [seq_len, batch, hidden_size * num_directions]
    """

    def __init__(self, *args, **kwargs):
        # Parse args/kwargs to match PyTorch's flexible signature
        if len(args) >= 1:
            input_size = args[0]
            args = args[1:]
        else:
            input_size = kwargs.pop("input_size")

        if len(args) >= 1:
            hidden_size = args[0]
            args = args[1:]
        else:
            hidden_size = kwargs.pop("hidden_size")

        num_layers = kwargs.pop("num_layers", 1)
        bias = kwargs.pop("bias", True)
        batch_first = kwargs.pop("batch_first", False)
        dropout = kwargs.pop("dropout", 0.0)
        bidirectional = kwargs.pop("bidirectional", False)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        # device and dtype are accepted for PyTorch compatibility but ignored
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create weights directly with PyTorch-compatible names
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            for direction in range(self.num_directions):
                suffix = f"_l{layer}" if direction == 0 else f"_l{layer}_reverse"

                # Create weights (3 gates: r, z, n)
                stdv = 1.0 / math.sqrt(hidden_size)
                weight_ih = Parameter(
                    Tensor._from_mlx_array(
                        mx.random.uniform(-stdv, stdv, (3 * hidden_size, layer_input_size))
                    )
                )
                weight_hh = Parameter(
                    Tensor._from_mlx_array(
                        mx.random.uniform(-stdv, stdv, (3 * hidden_size, hidden_size))
                    )
                )
                setattr(self, f"weight_ih{suffix}", weight_ih)
                setattr(self, f"weight_hh{suffix}", weight_hh)

                if bias:
                    bias_ih = Parameter(Tensor._from_mlx_array(mx.zeros(3 * hidden_size)))
                    bias_hh = Parameter(Tensor._from_mlx_array(mx.zeros(3 * hidden_size)))
                    setattr(self, f"bias_ih{suffix}", bias_ih)
                    setattr(self, f"bias_hh{suffix}", bias_hh)

    def _get_weights(self, layer: int, direction: int):
        """Get weights for a specific layer and direction."""
        suffix = f"_l{layer}" if direction == 0 else f"_l{layer}_reverse"
        weight_ih = getattr(self, f"weight_ih{suffix}")
        weight_hh = getattr(self, f"weight_hh{suffix}")
        if self.bias:
            bias_ih = getattr(self, f"bias_ih{suffix}")
            bias_hh = getattr(self, f"bias_hh{suffix}")
        else:
            bias_ih = None
            bias_hh = None
        return weight_ih, weight_hh, bias_ih, bias_hh

    def _gru_cell_forward(self, input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
        """Single GRU cell forward pass."""
        # Use compiled version for better performance
        return _gru_cell_compiled(
            input,
            hx,
            weight_ih._mlx_array,
            weight_hh._mlx_array,
            bias_ih._mlx_array if bias_ih is not None else None,
            bias_hh._mlx_array if bias_hh is not None else None,
            self.bias,
            self.hidden_size,
        )

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Apply multi-layer GRU."""
        if self.batch_first:
            input = Tensor._from_mlx_array(mx.transpose(input._mlx_array, [1, 0, 2]))

        seq_len, batch_size, _ = input.shape

        if hx is None:
            num_directions = self.num_directions
            hx = Tensor._from_mlx_array(
                mx.zeros((self.num_layers * num_directions, batch_size, self.hidden_size))
            )

        output = input._mlx_array
        hidden_states = []

        for layer in range(self.num_layers):
            # Get weights for forward direction
            weight_ih, weight_hh, bias_ih, bias_hh = self._get_weights(layer, 0)

            layer_output_fwd = []
            h_fwd = hx._mlx_array[layer * self.num_directions]

            for t in range(seq_len):
                x_t = output[t]
                h_fwd = self._gru_cell_forward(x_t, h_fwd, weight_ih, weight_hh, bias_ih, bias_hh)
                layer_output_fwd.append(h_fwd)

            hidden_states.append(h_fwd)

            if self.bidirectional:
                # Get weights for backward direction
                weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r = self._get_weights(layer, 1)

                layer_output_bwd = []
                h_bwd = hx._mlx_array[layer * self.num_directions + 1]

                for t in range(seq_len - 1, -1, -1):
                    x_t = output[t]
                    h_bwd = self._gru_cell_forward(
                        x_t, h_bwd, weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r
                    )
                    layer_output_bwd.insert(0, h_bwd)

                hidden_states.append(h_bwd)

                output = mx.concatenate(
                    [mx.stack(layer_output_fwd, axis=0), mx.stack(layer_output_bwd, axis=0)], axis=2
                )
            else:
                output = mx.stack(layer_output_fwd, axis=0)

        h_n = Tensor._from_mlx_array(mx.stack(hidden_states, axis=0))
        output = Tensor._from_mlx_array(output)

        if self.batch_first:
            output = Tensor._from_mlx_array(mx.transpose(output._mlx_array, [1, 0, 2]))

        return output, h_n

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional}"
        )


__all__ = [
    "RNNCellBase",
    "RNNBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    "LSTM",
    "GRU",
]
