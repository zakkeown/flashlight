# PyTorch Recurrent Neural Network Layers

## Purpose

This document provides comprehensive documentation of PyTorch's recurrent neural network (RNN) modules. RNNs are essential for sequence modeling tasks including:

1. Natural language processing (text generation, translation)
2. Time series forecasting
3. Speech recognition
4. Video analysis

**Source**: [torch/nn/modules/rnn.py](../../reference/pytorch/torch/nn/modules/rnn.py)

## Architecture Overview

### RNN Module Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Module                                   │
│                            ↑                                     │
│               ┌────────────┴────────────┐                       │
│            RNNBase                   RNNCellBase                │
│     (multi-layer sequences)       (single step)                 │
│               ↑                         ↑                       │
│    ┌──────┬──────┬──────┐     ┌────────┼────────┐              │
│   RNN   LSTM   GRU     │   RNNCell LSTMCell GRUCell            │
└─────────────────────────────────────────────────────────────────┘
```

### Sequence vs Cell Modules

| Type | Class | Use Case |
|------|-------|----------|
| **Sequence** | RNN, LSTM, GRU | Process entire sequences at once |
| **Cell** | RNNCell, LSTMCell, GRUCell | Manual step-by-step control |

---

## 1. Base Classes

### RNNBase

The base class for all sequence-processing RNN modules.

**Common Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_size` | int | Features in input x |
| `hidden_size` | int | Features in hidden state h |
| `num_layers` | int | Number of stacked RNN layers (default: 1) |
| `bias` | bool | Use bias weights (default: True) |
| `batch_first` | bool | Input shape `(batch, seq, feature)` (default: False) |
| `dropout` | float | Dropout between layers except last (default: 0) |
| `bidirectional` | bool | Bidirectional RNN (default: False) |

**Weight Initialization**:
All weights and biases are initialized from U(-√k, √k) where k = 1/hidden_size.

```python
def reset_parameters(self) -> None:
    stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
    for weight in self.parameters():
        init.uniform_(weight, -stdv, stdv)
```

---

## 2. Sequence Modules

### RNN (Elman RNN)

The basic recurrent neural network with tanh or ReLU nonlinearity.

**Formula**:
```
h_t = tanh(W_ih × x_t + b_ih + W_hh × h_(t-1) + b_hh)
```

Or with ReLU:
```
h_t = relu(W_ih × x_t + b_ih + W_hh × h_(t-1) + b_hh)
```

**Constructor**:
```python
nn.RNN(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    nonlinearity: str = 'tanh',  # 'tanh' or 'relu'
    bias: bool = True,
    batch_first: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
)
```

**Inputs**:
- `input`: `(L, N, H_in)` or `(N, L, H_in)` if batch_first
- `h_0`: `(D×num_layers, N, H_out)` - optional initial hidden state

**Outputs**:
- `output`: `(L, N, D×H_out)` or `(N, L, D×H_out)` - hidden states for all timesteps
- `h_n`: `(D×num_layers, N, H_out)` - final hidden state

Where D = 2 if bidirectional else 1.

**Example**:
```python
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2)
input = torch.randn(5, 3, 10)  # (seq_len, batch, features)
h0 = torch.randn(2, 3, 20)     # (num_layers, batch, hidden)
output, hn = rnn(input, h0)
# output: (5, 3, 20), hn: (2, 3, 20)
```

**Attributes**:
- `weight_ih_l[k]`: Input-hidden weights for layer k, shape `(hidden_size, input_size)` for k=0
- `weight_hh_l[k]`: Hidden-hidden weights, shape `(hidden_size, hidden_size)`
- `bias_ih_l[k]`, `bias_hh_l[k]`: Biases, shape `(hidden_size,)`

---

### LSTM (Long Short-Term Memory)

LSTM addresses the vanishing gradient problem with gated memory cells.

**Gate Formulas**:
```
i_t = σ(W_ii × x_t + b_ii + W_hi × h_(t-1) + b_hi)    # Input gate
f_t = σ(W_if × x_t + b_if + W_hf × h_(t-1) + b_hf)    # Forget gate
g_t = tanh(W_ig × x_t + b_ig + W_hg × h_(t-1) + b_hg) # Cell gate
o_t = σ(W_io × x_t + b_io + W_ho × h_(t-1) + b_ho)    # Output gate

c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t                       # Cell state
h_t = o_t ⊙ tanh(c_t)                                  # Hidden state
```

Where σ is sigmoid and ⊙ is element-wise multiplication.

**Constructor**:
```python
nn.LSTM(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bias: bool = True,
    batch_first: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
    proj_size: int = 0,  # LSTM-specific: projection layer
)
```

**Projection** (`proj_size > 0`):
Adds a projection layer to reduce output dimensionality:
```
h_t = W_hr × h_t  # Projects from hidden_size to proj_size
```

**Inputs**:
- `input`: `(L, N, H_in)` or `(N, L, H_in)` if batch_first
- `(h_0, c_0)`: Tuple of:
  - `h_0`: `(D×num_layers, N, H_out)` - initial hidden state
  - `c_0`: `(D×num_layers, N, H_cell)` - initial cell state

**Outputs**:
- `output`: `(L, N, D×H_out)`
- `(h_n, c_n)`: Final hidden and cell states

**Example**:
```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
input = torch.randn(5, 3, 10)   # (seq_len, batch, features)
h0 = torch.randn(2, 3, 20)      # (num_layers, batch, hidden)
c0 = torch.randn(2, 3, 20)      # (num_layers, batch, hidden)
output, (hn, cn) = lstm(input, (h0, c0))
# output: (5, 3, 20), hn: (2, 3, 20), cn: (2, 3, 20)
```

**Attributes**:
- `weight_ih_l[k]`: Shape `(4×hidden_size, input_size)` - concatenated [W_ii|W_if|W_ig|W_io]
- `weight_hh_l[k]`: Shape `(4×hidden_size, hidden_size)`
- `bias_ih_l[k]`, `bias_hh_l[k]`: Shape `(4×hidden_size,)`
- `weight_hr_l[k]`: Shape `(proj_size, hidden_size)` if proj_size > 0

---

### GRU (Gated Recurrent Unit)

GRU simplifies LSTM with fewer gates while maintaining effectiveness.

**Gate Formulas**:
```
r_t = σ(W_ir × x_t + b_ir + W_hr × h_(t-1) + b_hr)    # Reset gate
z_t = σ(W_iz × x_t + b_iz + W_hz × h_(t-1) + b_hz)    # Update gate
n_t = tanh(W_in × x_t + b_in + r_t ⊙ (W_hn × h_(t-1) + b_hn))  # New gate

h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_(t-1)                 # Hidden state
```

**Note**: PyTorch's GRU implementation differs slightly from the original paper for efficiency. The reset gate is applied after the W_hn multiplication.

**Constructor**:
```python
nn.GRU(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    bias: bool = True,
    batch_first: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
)
```

**Inputs/Outputs**: Same as RNN.

**Example**:
```python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = gru(input, h0)
```

**Attributes**:
- `weight_ih_l[k]`: Shape `(3×hidden_size, input_size)` - concatenated [W_ir|W_iz|W_in]
- `weight_hh_l[k]`: Shape `(3×hidden_size, hidden_size)`
- `bias_ih_l[k]`, `bias_hh_l[k]`: Shape `(3×hidden_size,)`

---

## 3. Cell Modules

Cell modules process one timestep at a time, giving fine-grained control.

### RNNCell

**Formula**:
```
h' = tanh(W_ih × x + b_ih + W_hh × h + b_hh)
```

**Constructor**:
```python
nn.RNNCell(
    input_size: int,
    hidden_size: int,
    bias: bool = True,
    nonlinearity: str = 'tanh',  # 'tanh' or 'relu'
)
```

**Inputs**:
- `input`: `(N, H_in)` or `(H_in)`
- `hidden`: `(N, H_out)` or `(H_out)` - defaults to zeros

**Output**: `h'` with same shape as hidden

**Example**:
```python
cell = nn.RNNCell(10, 20)
input = torch.randn(6, 3, 10)  # 6 timesteps, batch=3
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = cell(input[i], hx)
    output.append(hx)
```

---

### LSTMCell

**Formula**:
```
i = σ(W_ii × x + b_ii + W_hi × h + b_hi)
f = σ(W_if × x + b_if + W_hf × h + b_hf)
g = tanh(W_ig × x + b_ig + W_hg × h + b_hg)
o = σ(W_io × x + b_io + W_ho × h + b_ho)
c' = f ⊙ c + i ⊙ g
h' = o ⊙ tanh(c')
```

**Constructor**:
```python
nn.LSTMCell(
    input_size: int,
    hidden_size: int,
    bias: bool = True,
)
```

**Inputs**:
- `input`: `(N, H_in)` or `(H_in)`
- `(h_0, c_0)`: Tuple of hidden and cell states

**Output**: `(h', c')` tuple

**Example**:
```python
cell = nn.LSTMCell(10, 20)
input = torch.randn(2, 3, 10)  # 2 timesteps
hx = torch.randn(3, 20)
cx = torch.randn(3, 20)
output = []
for i in range(2):
    hx, cx = cell(input[i], (hx, cx))
    output.append(hx)
```

**Attributes**:
- `weight_ih`: Shape `(4×hidden_size, input_size)`
- `weight_hh`: Shape `(4×hidden_size, hidden_size)`
- `bias_ih`, `bias_hh`: Shape `(4×hidden_size,)`

---

### GRUCell

**Formula**:
```
r = σ(W_ir × x + b_ir + W_hr × h + b_hr)
z = σ(W_iz × x + b_iz + W_hz × h + b_hz)
n = tanh(W_in × x + b_in + r ⊙ (W_hn × h + b_hn))
h' = (1 - z) ⊙ n + z ⊙ h
```

**Constructor**:
```python
nn.GRUCell(
    input_size: int,
    hidden_size: int,
    bias: bool = True,
)
```

**Inputs/Outputs**: Same as RNNCell.

**Attributes**:
- `weight_ih`: Shape `(3×hidden_size, input_size)`
- `weight_hh`: Shape `(3×hidden_size, hidden_size)`
- `bias_ih`, `bias_hh`: Shape `(3×hidden_size,)`

---

## 4. Variable Length Sequences

PyTorch supports packed sequences for efficient handling of variable-length inputs.

### Packing and Padding

```python
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pack_sequence,
    pad_sequence,
)

# Example: Variable length sequences
sequences = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(7, 10)]
lengths = torch.tensor([5, 3, 7])

# Pad to same length
padded = pad_sequence(sequences, batch_first=True)  # (3, 7, 10)

# Pack for RNN
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)

# Process with RNN
lstm = nn.LSTM(10, 20, batch_first=True)
output_packed, (hn, cn) = lstm(packed)

# Unpack
output, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
```

### PackedSequence Structure

```python
@dataclass
class PackedSequence:
    data: Tensor           # Concatenated sequence data
    batch_sizes: Tensor    # Batch size at each timestep
    sorted_indices: Tensor # Indices to sort by length
    unsorted_indices: Tensor  # Indices to restore original order
```

---

## 5. Bidirectional RNNs

Bidirectional RNNs process sequences in both directions.

```python
# Bidirectional LSTM
lstm = nn.LSTM(10, 20, bidirectional=True)
input = torch.randn(5, 3, 10)
output, (hn, cn) = lstm(input)

# output shape: (5, 3, 40) - concatenated forward and backward
# hn shape: (2, 3, 20) - [forward_final, backward_final]

# Split output into forward and backward
output_fwd = output[:, :, :20]  # Forward hidden states
output_bwd = output[:, :, 20:]  # Backward hidden states

# Or reshape
output_split = output.view(5, 3, 2, 20)  # (seq, batch, directions, hidden)
```

---

## 6. Common Patterns

### Stacked RNNs

```python
# 3-layer stacked LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=3, dropout=0.2)

# Hidden state shape: (3, batch, 20)
# Each layer takes output from previous layer
```

### Sequence-to-Sequence

```python
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        _, (h, c) = self.encoder(src)
        output, _ = self.decoder(tgt, (h, c))
        return self.fc(output)
```

### Forget Gate Bias Initialization

For LSTMs, setting forget gate bias to 1 helps gradient flow:

```python
lstm = nn.LSTM(10, 20)
for name, param in lstm.named_parameters():
    if 'bias_hh' in name:
        # bias_hh is [b_hi, b_hf, b_hg, b_ho]
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)  # Set forget gate bias to 1
```

---

## MLX Mapping

### MLX RNN Support

MLX has limited built-in RNN support. Custom implementations are required for most RNN architectures.

### Manual LSTM Implementation in MLX

```python
import mlx.core as mx
import mlx.nn as nn
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined input-hidden weights for all gates
        scale = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = mx.random.uniform(
            low=-scale, high=scale,
            shape=(4 * hidden_size, input_size)
        )
        self.weight_hh = mx.random.uniform(
            low=-scale, high=scale,
            shape=(4 * hidden_size, hidden_size)
        )
        self.bias_ih = mx.zeros((4 * hidden_size,))
        self.bias_hh = mx.zeros((4 * hidden_size,))

    def __call__(self, x, hx=None):
        if hx is None:
            h = mx.zeros((x.shape[0], self.hidden_size))
            c = mx.zeros((x.shape[0], self.hidden_size))
        else:
            h, c = hx

        # Combined gates computation
        gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh

        # Split into individual gates
        i, f, g, o = mx.split(gates, 4, axis=-1)

        # Apply activations
        i = mx.sigmoid(i)  # Input gate
        f = mx.sigmoid(f)  # Forget gate
        g = mx.tanh(g)     # Cell gate
        o = mx.sigmoid(o)  # Output gate

        # Update cell and hidden state
        c_new = f * c + i * g
        h_new = o * mx.tanh(c_new)

        return h_new, c_new


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ]

    def __call__(self, x, hx=None):
        """
        x: (seq_len, batch, input_size)
        Returns: output (seq_len, batch, hidden_size), (h_n, c_n)
        """
        seq_len, batch_size, _ = x.shape

        if hx is None:
            h = [mx.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
            c = [mx.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            h, c = [hx[0][i] for i in range(self.num_layers)], [hx[1][i] for i in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            inp = x[t]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, (h[layer], c[layer]))
                inp = h[layer]
            outputs.append(h[-1])

        output = mx.stack(outputs, axis=0)
        h_n = mx.stack(h, axis=0)
        c_n = mx.stack(c, axis=0)

        return output, (h_n, c_n)


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = mx.random.uniform(low=-scale, high=scale, shape=(3 * hidden_size, input_size))
        self.weight_hh = mx.random.uniform(low=-scale, high=scale, shape=(3 * hidden_size, hidden_size))
        self.bias_ih = mx.zeros((3 * hidden_size,))
        self.bias_hh = mx.zeros((3 * hidden_size,))

    def __call__(self, x, h=None):
        if h is None:
            h = mx.zeros((x.shape[0], self.hidden_size))

        # Input gates
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh

        i_r, i_z, i_n = mx.split(gi, 3, axis=-1)
        h_r, h_z, h_n = mx.split(gh, 3, axis=-1)

        r = mx.sigmoid(i_r + h_r)  # Reset gate
        z = mx.sigmoid(i_z + h_z)  # Update gate
        n = mx.tanh(i_n + r * h_n) # New gate

        h_new = (1 - z) * n + z * h

        return h_new
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Built-in RNNs** | RNN, LSTM, GRU, Cells | Limited/None |
| **cuDNN optimization** | Yes (flatten_parameters) | Metal optimized |
| **PackedSequence** | Supported | Manual implementation |
| **Bidirectional** | Built-in | Manual implementation |
| **Projection layers** | LSTM proj_size | Manual |
| **Dropout** | Between layers | Manual |

---

## Summary

### When to Use Each

| Module | Use Case |
|--------|----------|
| **LSTM** | Long sequences, complex dependencies, memory required |
| **GRU** | Similar to LSTM but faster, fewer parameters |
| **RNN** | Simple sequences, short dependencies |
| **Cells** | Custom control flow, attention mechanisms |

### Quick Reference

| Property | RNN | LSTM | GRU |
|----------|-----|------|-----|
| Gates | 0 | 4 (i, f, g, o) | 3 (r, z, n) |
| States | h | h, c | h |
| Parameters per layer | 1× | 4× | 3× |
| Vanishing gradient | Yes | No | No |
| Computation | Fastest | Slowest | Middle |
