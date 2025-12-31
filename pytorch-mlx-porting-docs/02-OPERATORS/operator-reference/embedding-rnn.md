# PyTorch Embeddings and RNN Operations

## Overview

Embeddings and Recurrent Neural Networks (RNNs) are fundamental components for processing sequential data and handling discrete token inputs. PyTorch provides **highly optimized** implementations of embedding lookups, EmbeddingBag (with pooling), and RNN cells (LSTM, GRU, vanilla RNN).

**Key Operations**:
1. **Embedding**: Lookup table mapping discrete tokens to dense vectors
2. **EmbeddingBag**: Embedding with automatic pooling (sum/mean/max)
3. **LSTM**: Long Short-Term Memory cells
4. **GRU**: Gated Recurrent Unit cells
5. **Vanilla RNN**: Simple recurrent cells (ReLU/Tanh activations)

**Location**: `aten/src/ATen/native/Embedding.cpp`, `aten/src/ATen/native/RNN.cpp`

---

## 1. Embedding Layer

### 1.1 Algorithm Overview

**Embedding** is a lookup operation that maps integer indices to dense vectors:

```
Input: indices [*, seq_len] (integer tensor)
Weight: embedding_table [num_embeddings, embedding_dim]
Output: embeddings [*, seq_len, embedding_dim]

output[i, j] = embedding_table[indices[i, j]]
```

**Use Cases**:
- Word embeddings (word → vector)
- Token embeddings for transformers
- Categorical feature encoding
- Learnable position embeddings

### 1.2 PyTorch API

```python
import torch.nn as nn

# Create embedding layer
embedding = nn.Embedding(
    num_embeddings=10000,    # Vocabulary size
    embedding_dim=300,       # Embedding dimension
    padding_idx=0,           # Padding token index (optional)
    max_norm=None,           # Max L2 norm for embeddings
    norm_type=2.0,           # Norm type (2 = L2)
    scale_grad_by_freq=False # Scale gradients by token frequency
)

# Forward pass
indices = torch.tensor([[1, 2, 3], [4, 5, 0]])  # [batch, seq_len]
output = embedding(indices)  # [batch, seq_len, embedding_dim]
```

**Parameters**:
- `num_embeddings`: Size of lookup table (vocabulary size)
- `embedding_dim`: Dimension of each embedding vector
- `padding_idx`: If specified, this index doesn't contribute gradients
- `max_norm`: If specified, embeddings are renormalized to have max L2 norm
- `scale_grad_by_freq`: Gradient scaling based on word frequency

### 1.3 Implementation Details

**Forward Pass** (`aten/src/ATen/native/Embedding.cpp`):

```cpp
Tensor embedding_symint(
    const Tensor& weight,        // [num_embeddings, embedding_dim]
    const Tensor& indices,       // [*, seq_len]
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {

  TORCH_CHECK(weight.dim() == 2, "'weight' must be 2-D");

  // Simple case: 1D indices
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);  // Direct indexing
  }

  // General case: Reshape indices, lookup, reshape back
  auto size = indices.sym_sizes().vec();       // Original shape
  for (const auto& d : weight.sym_sizes().slice(1)) {
    size.push_back(d);                         // Add embedding_dim
  }

  // Flatten indices → lookup → reshape to output shape
  return weight.index_select(0, indices.reshape(-1)).view_symint(size);
}
```

**Key Insight**: Embedding is **just an indexed lookup** using `index_select`.

**Performance Optimization**:

```cpp
// CPU: Direct memory access with multi-threading
#pragma omp parallel for if (num_indices > 1000)
for (int64_t i = 0; i < num_indices; ++i) {
  int64_t idx = indices_data[i];
  if (idx != padding_idx) {
    memcpy(output_data + i * embedding_dim,
           weight_data + idx * embedding_dim,
           embedding_dim * sizeof(scalar_t));
  } else {
    memset(output_data + i * embedding_dim, 0,
           embedding_dim * sizeof(scalar_t));  // Zero out padding
  }
}
```

**CUDA**: Uses gather kernel for coalesced memory access

### 1.4 Backward Pass

**Dense Gradient** (default):

```cpp
Tensor embedding_dense_backward_cpu(
    const Tensor& grad_output,     // [*, seq_len, embedding_dim]
    const Tensor& indices,         // [*, seq_len]
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {

  // Initialize gradient table
  auto grad_weight = at::zeros({num_weights, embedding_dim}, grad_output.options());

  // Accumulate gradients for each index
  int64_t numel = indices.numel();
  auto indices_flat = indices.reshape(-1);
  auto grad_flat = grad_output.reshape({numel, embedding_dim});

  // Parallel accumulation
  for (int64_t i = 0; i < numel; ++i) {
    int64_t idx = indices_flat[i];
    if (idx != padding_idx) {
      double scale = 1.0;
      if (scale_grad_by_freq) {
        scale /= count_frequency(indices_flat, idx);
      }
      grad_weight[idx] += grad_flat[i] * scale;  // Accumulate
    }
  }

  return grad_weight;
}
```

**Sparse Gradient** (when `sparse=True`):

```cpp
Tensor embedding_sparse_backward(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {

  // Filter out padding indices
  if (padding_idx != -1) {
    indices = indices.index({indices != padding_idx});
    grad = grad.index({indices != padding_idx});
  }

  // Return sparse COO tensor
  auto index = indices.reshape({1, -1});
  auto values = grad.reshape({-1, embedding_dim});
  return at::sparse_coo_tensor(
    index, values, {num_weights, embedding_dim}
  );
}
```

**When to Use Sparse Gradients**:
- Large vocabulary (num_embeddings >> seq_len * batch_size)
- Most embeddings not updated each batch
- Using sparse optimizers (SparseAdam)

### 1.5 Embedding Renormalization

**Max Norm Constraint**: Ensures embedding vectors don't exceed max L2 norm

```cpp
Tensor& embedding_renorm_cpu_(
    Tensor& self,                // Embedding weight [num_embeddings, embedding_dim]
    const Tensor& indices,       // Indices used in forward pass
    double max_norm,
    double norm_type) {

  // Only renormalize embeddings that were accessed
  auto indices_unique = indices.unique();

  for (int64_t idx : indices_unique) {
    auto row = self[idx];                           // Get embedding vector
    auto norm = row.norm(norm_type).item<double>(); // Compute norm

    if (norm > max_norm) {
      double scale = max_norm / (norm + 1e-7);      // Scale factor
      row *= scale;                                 // In-place renormalization
    }
  }

  return self;
}
```

**Applied After**: Each optimizer step

---

## 2. EmbeddingBag

### 2.1 Algorithm Overview

**EmbeddingBag** combines embedding lookup with **automatic pooling**, useful for bag-of-words representations:

```
Input:
  - indices: [total_indices] (flattened)
  - offsets: [batch_size + 1] (bag boundaries)
  - weight: [num_embeddings, embedding_dim]

For each bag i:
  bag_indices = indices[offsets[i]:offsets[i+1]]
  bag_embeddings = weight[bag_indices]  # Lookup

  if mode == MODE_SUM:
    output[i] = sum(bag_embeddings)
  elif mode == MODE_MEAN:
    output[i] = mean(bag_embeddings)
  elif mode == MODE_MAX:
    output[i] = max(bag_embeddings, dim=0)

Output: [batch_size, embedding_dim]
```

**Modes**:
- `MODE_SUM` (0): Sum pooling
- `MODE_MEAN` (1): Mean pooling
- `MODE_MAX` (2): Max pooling

### 2.2 PyTorch API

```python
import torch.nn as nn

# EmbeddingBag with sum pooling
embedding_bag = nn.EmbeddingBag(
    num_embeddings=10000,
    embedding_dim=300,
    mode='sum',              # 'sum', 'mean', or 'max'
    sparse=False,
    scale_grad_by_freq=False,
    include_last_offset=False
)

# Input: Variable-length sequences (bags)
indices = torch.tensor([1, 2, 3, 4, 5, 6, 7])  # Flattened indices
offsets = torch.tensor([0, 3, 5, 7])           # Bag boundaries: [0:3], [3:5], [5:7]

output = embedding_bag(indices, offsets)       # [3, 300]
# Bag 0: embeddings[1] + embeddings[2] + embeddings[3]
# Bag 1: embeddings[4] + embeddings[5]
# Bag 2: embeddings[6] + embeddings[7]
```

**Per-Sample Weights** (optional):

```python
# Weighted sum: each index has associated weight
weights = torch.tensor([0.5, 1.0, 0.3, 0.8, 0.2, 1.0, 0.6])

output = embedding_bag(indices, offsets, per_sample_weights=weights)
# Bag 0: 0.5*emb[1] + 1.0*emb[2] + 0.3*emb[3]
```

### 2.3 Implementation (CPU)

```cpp
std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_cpu(
    const Tensor& weight,           // [num_embeddings, embedding_dim]
    const Tensor& indices,          // [total_indices]
    const Tensor& offsets,          // [batch_size] or [batch_size + 1]
    bool scale_grad_by_freq,
    int mode,                       // 0=sum, 1=mean, 2=max
    bool sparse,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset,
    int padding_idx) {

  int64_t num_bags = offsets.size(0);
  if (include_last_offset) num_bags -= 1;

  auto output = at::zeros({num_bags, weight.size(1)}, weight.options());

  // For each bag
  for (int64_t bag = 0; bag < num_bags; ++bag) {
    int64_t start = offsets[bag];
    int64_t end = include_last_offset ? offsets[bag + 1] :
                  (bag == num_bags - 1 ? indices.size(0) : offsets[bag + 1]);

    // Accumulate embeddings in bag
    for (int64_t idx = start; idx < end; ++idx) {
      int64_t emb_idx = indices[idx];
      if (emb_idx == padding_idx) continue;

      auto embedding = weight[emb_idx];

      if (per_sample_weights.has_value()) {
        embedding = embedding * per_sample_weights.value()[idx];
      }

      if (mode == MODE_SUM || mode == MODE_MEAN) {
        output[bag] += embedding;
      } else if (mode == MODE_MAX) {
        output[bag] = at::max(output[bag], embedding);
      }
    }

    // Mean pooling: divide by bag size
    if (mode == MODE_MEAN && (end - start) > 0) {
      output[bag] /= (end - start);
    }
  }

  // Return (output, offset2bag, bag_size, max_indices)
  return std::make_tuple(output, offset2bag, bag_size, max_indices);
}
```

**CUDA Optimization**: Uses warp-level reduction for sum/mean pooling

### 2.4 Backward Pass

**Gradient Redistribution**:

```cpp
// For sum mode:
//   grad_weight[indices[i]] += grad_output[bag_of(i)]

// For mean mode:
//   grad_weight[indices[i]] += grad_output[bag_of(i)] / bag_size[bag_of(i)]

// For max mode:
//   grad_weight[indices[i]] += grad_output[bag_of(i)]  (only if i == argmax)
```

**Implementation** (simplified):

```cpp
for (int64_t i = 0; i < indices.size(0); ++i) {
  int64_t bag = offset2bag[i];  // Which bag does index i belong to?
  int64_t emb_idx = indices[i];

  if (mode == MODE_SUM) {
    grad_weight[emb_idx] += grad_output[bag];
  } else if (mode == MODE_MEAN) {
    grad_weight[emb_idx] += grad_output[bag] / bag_sizes[bag];
  } else if (mode == MODE_MAX) {
    if (i == max_indices[bag]) {  // Only update max element
      grad_weight[emb_idx] += grad_output[bag];
    }
  }
}
```

---

## 3. LSTM (Long Short-Term Memory)

### 3.1 Algorithm Overview

**LSTM Cell** processes sequence one step at a time:

```
Input:
  x_t: input at time t [batch, input_size]
  h_{t-1}: hidden state [batch, hidden_size]
  c_{t-1}: cell state [batch, hidden_size]

Gates:
  i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # Input gate
  f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # Forget gate
  g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)     # Cell gate
  o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # Output gate

Update:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t  # New cell state
  h_t = o_t ⊙ tanh(c_t)             # New hidden state

Output: h_t, c_t
```

**Key Properties**:
- **Cell State** (c_t): Long-term memory
- **Hidden State** (h_t): Short-term memory / output
- **Gates**: Control information flow (forget old, learn new, output)

### 3.2 PyTorch API

```python
import torch.nn as nn

# Multi-layer bidirectional LSTM
lstm = nn.LSTM(
    input_size=300,      # Input dimension
    hidden_size=512,     # Hidden state dimension
    num_layers=2,        # Number of LSTM layers
    bias=True,           # Use bias terms
    batch_first=False,   # Input shape: [seq_len, batch, input_size]
    dropout=0.3,         # Dropout between layers (if num_layers > 1)
    bidirectional=True   # Bidirectional LSTM
)

# Forward pass
input = torch.randn(10, 32, 300)  # [seq_len, batch, input_size]
h0 = torch.randn(4, 32, 512)      # [num_layers*num_directions, batch, hidden]
c0 = torch.randn(4, 32, 512)      # [num_layers*num_directions, batch, hidden]

output, (hn, cn) = lstm(input, (h0, c0))
# output: [10, 32, 1024]  (hidden_size * num_directions)
# hn: [4, 32, 512]        (final hidden states for each layer/direction)
# cn: [4, 32, 512]        (final cell states)
```

### 3.3 Implementation: Fused LSTM Cell

**Single-Step LSTM Cell** (`_thnn_fused_lstm_cell`):

```cpp
std::tuple<Tensor, Tensor> lstm_cell(
    const Tensor& input,      // [batch, input_size]
    TensorList hx,            // [h, c] where h,c: [batch, hidden_size]
    const Tensor& w_ih,       // [4*hidden_size, input_size]
    const Tensor& w_hh,       // [4*hidden_size, hidden_size]
    const std::optional<Tensor>& b_ih,  // [4*hidden_size]
    const std::optional<Tensor>& b_hh   // [4*hidden_size]
) {
  auto h = hx[0];  // Previous hidden state
  auto c = hx[1];  // Previous cell state

  // Fused linear: compute all 4 gates at once
  // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
  auto gates = at::linear(input, w_ih, b_ih) + at::linear(h, w_hh, b_hh);

  // Split into 4 gates: [i, f, g, o]
  auto chunked_gates = gates.chunk(4, 1);

  auto i = chunked_gates[0].sigmoid();  // Input gate
  auto f = chunked_gates[1].sigmoid();  // Forget gate
  auto g = chunked_gates[2].tanh();     // Cell gate
  auto o = chunked_gates[3].sigmoid();  // Output gate

  // Update cell state
  auto cy = f * c + i * g;

  // Update hidden state
  auto hy = o * cy.tanh();

  return std::make_tuple(hy, cy);
}
```

**Key Optimization**: Fused matrix multiplication for all 4 gates reduces kernel launches

### 3.4 Multi-Layer LSTM

**Stacked Layers** with dropout:

```cpp
std::tuple<Tensor, Tensor, Tensor> lstm_input(
    const Tensor& input,      // [seq_len, batch, input_size]
    TensorList hx,            // Initial states
    TensorList params,        // [w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0, ...]
    bool has_biases,
    int num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {

  // Transpose if batch_first
  auto input_t = batch_first ? input.transpose(0, 1) : input;
  int seq_len = input_t.size(0);
  int batch_size = input_t.size(1);

  // Split initial states by layer and direction
  auto h_layers = hx[0].chunk(num_layers * (bidirectional ? 2 : 1), 0);
  auto c_layers = hx[1].chunk(num_layers * (bidirectional ? 2 : 1), 0);

  auto layer_input = input_t;

  // For each layer
  for (int layer = 0; layer < num_layers; ++layer) {
    // Forward direction
    auto [output_fwd, h_fwd, c_fwd] = lstm_layer_forward(
      layer_input, h_layers[layer*2], c_layers[layer*2], params_fwd
    );

    if (bidirectional) {
      // Backward direction (reverse sequence)
      auto [output_bwd, h_bwd, c_bwd] = lstm_layer_backward(
        layer_input, h_layers[layer*2+1], c_layers[layer*2+1], params_bwd
      );

      // Concatenate forward and backward outputs
      layer_input = at::cat({output_fwd, output_bwd}, 2);
    } else {
      layer_input = output_fwd;
    }

    // Apply dropout between layers (except last layer)
    if (layer < num_layers - 1 && dropout > 0) {
      layer_input = at::dropout(layer_input, dropout, train);
    }
  }

  return std::make_tuple(layer_input, h_final, c_final);
}
```

### 3.5 Packed Sequences

**Handling Variable-Length Sequences**:

PyTorch uses `PackedSequence` to efficiently process batches with different sequence lengths:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Padded sequences [seq_len, batch, input_size]
sequences = torch.randn(10, 3, 300)
lengths = torch.tensor([10, 7, 5])  # Actual lengths

# Pack sequences (removes padding)
packed = pack_padded_sequence(sequences, lengths, enforce_sorted=False)

# Process with LSTM
packed_output, (hn, cn) = lstm(packed)

# Unpack back to padded format
output, output_lengths = pad_packed_sequence(packed_output)
```

**Internal Representation**:

```
PackedSequence:
  data: [total_length, input_size]  # Concatenated sequences
  batch_sizes: [seq_len]             # Number of sequences at each timestep

Example:
  Seq1: [x1_1, x1_2, x1_3, x1_4]  (length 4)
  Seq2: [x2_1, x2_2, x2_3]        (length 3)
  Seq3: [x3_1, x3_2]              (length 2)

  data = [x1_1, x2_1, x3_1,  # Timestep 0: 3 sequences
          x1_2, x2_2, x3_2,  # Timestep 1: 3 sequences
          x1_3, x2_3,        # Timestep 2: 2 sequences
          x1_4]              # Timestep 3: 1 sequence

  batch_sizes = [3, 3, 2, 1]
```

**Benefit**: No wasted computation on padding tokens

### 3.6 cuDNN Acceleration

**GPU Optimization**: PyTorch uses cuDNN's optimized RNN kernels when available:

```cpp
bool use_cudnn_lstm(const Tensor& input, TensorList params) {
  return input.is_cuda() &&
         at::cudnn_is_acceptable(input) &&
         (input.dtype() == kFloat || input.dtype() == kHalf);
}

if (use_cudnn_lstm(input, params)) {
  return at::cudnn_rnn(input, params, ...);  // cuDNN optimized
} else {
  return lstm_native(input, params, ...);     // PyTorch native
}
```

**cuDNN Benefits**:
- Fused operations (3-5x faster)
- Optimized memory layout
- Tensor Core support (on Ampere+)

---

## 4. GRU (Gated Recurrent Unit)

### 4.1 Algorithm Overview

**GRU Cell**: Simplified alternative to LSTM (fewer gates, faster)

```
Input:
  x_t: [batch, input_size]
  h_{t-1}: [batch, hidden_size]

Gates:
  r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # Reset gate
  z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # Update gate

Candidate:
  n_t = tanh(W_in @ x_t + b_in + r_t ⊙ (W_hn @ h_{t-1} + b_hn))

Update:
  h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

Output: h_t
```

**vs LSTM**:
- No separate cell state (only hidden state)
- 2 gates instead of 4 (reset, update)
- Faster to compute, slightly less expressive

### 4.2 Implementation

```cpp
Tensor gru_cell(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& w_ih,   // [3*hidden_size, input_size]
    const Tensor& w_hh,   // [3*hidden_size, hidden_size]
    const std::optional<Tensor>& b_ih,
    const std::optional<Tensor>& b_hh) {

  // Input transformation
  auto gi = at::linear(input, w_ih, b_ih);
  auto gh = at::linear(hx, w_hh, b_hh);

  // Split into gates
  auto i_r = gi.slice(1, 0, hidden_size);           // Reset input
  auto i_z = gi.slice(1, hidden_size, 2*hidden_size);  // Update input
  auto i_n = gi.slice(1, 2*hidden_size, 3*hidden_size); // New input

  auto h_r = gh.slice(1, 0, hidden_size);
  auto h_z = gh.slice(1, hidden_size, 2*hidden_size);
  auto h_n = gh.slice(1, 2*hidden_size, 3*hidden_size);

  // Compute gates
  auto r = (i_r + h_r).sigmoid();  // Reset gate
  auto z = (i_z + h_z).sigmoid();  // Update gate

  // Candidate activation
  auto n = (i_n + r * h_n).tanh();

  // New hidden state
  auto hy = (1 - z) * n + z * hx;

  return hy;
}
```

---

## 5. Gradient Formulas

### 5.1 Embedding Backward

**Forward**: `output = weight[indices]`

**Backward**: Scatter gradients back to weight

```
grad_weight = zeros_like(weight)
for i in range(len(indices)):
  grad_weight[indices[i]] += grad_output[i]
```

### 5.2 LSTM Backward (BPTT)

**Complex due to recurrence**. Gradient flows backward through time:

```
At timestep t:
  dL/dh_t (from next layer or loss)
  dL/dc_t (from next timestep's forget gate)

Compute:
  dL/do_t = dL/dh_t ⊙ tanh(c_t)
  dL/dc_t += dL/dh_t ⊙ o_t ⊙ (1 - tanh²(c_t))

  dL/di_t = dL/dc_t ⊙ g_t ⊙ i_t ⊙ (1 - i_t)
  dL/df_t = dL/dc_t ⊙ c_{t-1} ⊙ f_t ⊙ (1 - f_t)
  dL/dg_t = dL/dc_t ⊙ i_t ⊙ (1 - g_t²)

Propagate to inputs:
  dL/dx_t = W_i^T @ dL/di_t + W_f^T @ dL/df_t + ...
  dL/dh_{t-1} = W_hi^T @ dL/di_t + W_hf^T @ dL/df_t + ...
  dL/dc_{t-1} = dL/dc_t ⊙ f_t
```

**Implementation**: PyTorch uses cuDNN's backward kernel or native BPTT

---

## 6. MLX Porting Recommendations

### 6.1 What to Port

**ADOPT**:

1. **Embedding Lookup**: Simple and essential
   ```python
   mlx.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)
   ```

2. **EmbeddingBag**: Useful for bag-of-words models
   ```python
   mlx.nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum')
   ```

3. **LSTM/GRU Cells**: Core RNN primitives
   ```python
   mlx.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, bidirectional=False)
   mlx.nn.GRU(input_size, hidden_size, ...)
   ```

4. **Fused Cell Implementation**: Single kernel for all gates
   - Reduce kernel launches
   - Better memory locality

**SIMPLIFY**:

1. **Single Backend**: Metal-only implementation
   - No cuDNN/MKL-DNN complexity
   - No CPU/GPU dispatch

2. **Fewer Options**: Start with core features
   - Skip: packed sequences initially
   - Skip: quantized RNNs
   - Focus: bidirectional, multi-layer

3. **Modern Architectures**: Prioritize what's used today
   - LSTM/GRU essential
   - Vanilla RNN (Tanh/ReLU) lower priority

### 6.2 What NOT to Port

**SKIP**:

1. **cuDNN Integration**: NVIDIA-specific
2. **Quantized RNNs**: Complex, low ROI initially
3. **MKL-DNN RNN**: Intel-specific
4. **Nested Tensors**: Complex feature

### 6.3 Recommended MLX Implementation

**Metal Shader for LSTM Cell**:

```metal
// mlx/backend/metal/kernels/lstm_cell.metal

kernel void lstm_cell_forward(
    device const float* input [[buffer(0)]],        // [batch, input_size]
    device const float* hx [[buffer(1)]],           // [batch, hidden_size]
    device const float* cx [[buffer(2)]],           // [batch, hidden_size]
    device const float* w_ih [[buffer(3)]],         // [4*hidden, input_size]
    device const float* w_hh [[buffer(4)]],         // [4*hidden, hidden_size]
    device const float* bias [[buffer(5)]],         // [4*hidden]
    device float* hy [[buffer(6)]],                 // [batch, hidden_size]
    device float* cy [[buffer(7)]],                 // [batch, hidden_size]
    constant int& batch_size [[buffer(8)]],
    constant int& input_size [[buffer(9)]],
    constant int& hidden_size [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]) {

  int batch_idx = gid.y;
  int hidden_idx = gid.x;

  if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;

  // Compute 4 gates (i, f, g, o)
  float gates[4] = {0, 0, 0, 0};

  // W_ih @ x
  for (int i = 0; i < input_size; ++i) {
    float x_val = input[batch_idx * input_size + i];
    for (int g = 0; g < 4; ++g) {
      gates[g] += w_ih[g * hidden_size * input_size + hidden_idx * input_size + i] * x_val;
    }
  }

  // W_hh @ h
  for (int i = 0; i < hidden_size; ++i) {
    float h_val = hx[batch_idx * hidden_size + i];
    for (int g = 0; g < 4; ++g) {
      gates[g] += w_hh[g * hidden_size * hidden_size + hidden_idx * hidden_size + i] * h_val;
    }
  }

  // Add bias
  for (int g = 0; g < 4; ++g) {
    gates[g] += bias[g * hidden_size + hidden_idx];
  }

  // Apply activations
  float i_t = 1.0f / (1.0f + exp(-gates[0]));  // Input gate (sigmoid)
  float f_t = 1.0f / (1.0f + exp(-gates[1]));  // Forget gate (sigmoid)
  float g_t = tanh(gates[2]);                   // Cell gate (tanh)
  float o_t = 1.0f / (1.0f + exp(-gates[3]));  // Output gate (sigmoid)

  // Update cell state
  float c_prev = cx[batch_idx * hidden_size + hidden_idx];
  float c_new = f_t * c_prev + i_t * g_t;
  cy[batch_idx * hidden_size + hidden_idx] = c_new;

  // Update hidden state
  float h_new = o_t * tanh(c_new);
  hy[batch_idx * hidden_size + hidden_idx] = h_new;
}
```

**C++ API**:

```cpp
// mlx/ops/rnn.cpp

std::pair<array, array> lstm_cell(
    const array& input,
    const std::pair<array, array>& hx,
    const array& w_ih,
    const array& w_hh,
    const std::optional<array>& bias) {

  // Validate inputs
  check_lstm_cell_inputs(input, hx, w_ih, w_hh);

  // Dispatch to Metal kernel
  return lstm_cell_metal(input, hx.first, hx.second, w_ih, w_hh, bias);
}
```

**Python Wrapper**:

```python
# mlx/nn/rnn.py

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bias=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Initialize weights for each layer
        self.weights = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.weights.append(self._init_layer_weights(layer_input_size))

    def __call__(self, x, hx=None):
        # x: [seq_len, batch, input_size]
        seq_len, batch_size, _ = x.shape

        # Initialize hidden states if not provided
        if hx is None:
            h = mx.zeros((self.num_layers, batch_size, self.hidden_size))
            c = mx.zeros((self.num_layers, batch_size, self.hidden_size))
        else:
            h, c = hx

        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self._lstm_cell(
                    x_t, h[layer], c[layer], self.weights[layer]
                )
                x_t = h[layer]
            outputs.append(h[-1])

        return mx.stack(outputs), (h, c)
```

---

## Summary

PyTorch's embedding and RNN implementations are **highly optimized** with backend-specific kernels:

**Key Implementations**:
1. **Embedding**: Index-select lookup with sparse/dense gradients
2. **EmbeddingBag**: Fused embedding + pooling
3. **LSTM**: Fused 4-gate cell with cuDNN acceleration
4. **GRU**: Fused 3-gate cell, faster than LSTM

**For MLX**:
- Port **embedding** and **EmbeddingBag** (simple, high value)
- Implement **fused LSTM/GRU cells** in Metal shaders
- Start with **bidirectional, multi-layer** support
- Skip packed sequences initially (add later)
- Focus on **modern architectures** (Transformers use embeddings heavily, RNNs less so)

Embeddings are **critical** for all NLP models. LSTM/GRU are less common today (Transformers dominate) but still used in some domains (speech, time-series).
