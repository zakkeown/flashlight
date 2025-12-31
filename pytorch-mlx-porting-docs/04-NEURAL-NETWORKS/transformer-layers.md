# Transformer Layers

## Overview

PyTorch's transformer implementation provides the building blocks for attention-based sequence models as described in "Attention Is All You Need" (Vaswani et al., 2017). These layers are critical for LLMs, machine translation, and modern NLP/vision architectures.

**Reference Files:**
- `torch/nn/modules/transformer.py` - Transformer, TransformerEncoder, TransformerDecoder
- `torch/nn/modules/activation.py` - MultiheadAttention

## Core Components

### Component Hierarchy

```
Transformer
├── TransformerEncoder
│   ├── TransformerEncoderLayer (×N)
│   │   ├── MultiheadAttention (self-attention)
│   │   ├── LayerNorm (×2)
│   │   ├── Linear (feedforward ×2)
│   │   └── Dropout (×2)
│   └── LayerNorm (final)
└── TransformerDecoder
    ├── TransformerDecoderLayer (×N)
    │   ├── MultiheadAttention (self-attention)
    │   ├── MultiheadAttention (cross-attention)
    │   ├── LayerNorm (×3)
    │   ├── Linear (feedforward ×2)
    │   └── Dropout (×3)
    └── LayerNorm (final)
```

---

## MultiheadAttention

The fundamental attention mechanism used throughout transformer architectures.

### Mathematical Formulation

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

### Class Definition

```python
class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,          # Total model dimension
        num_heads: int,          # Number of attention heads
        dropout: float = 0.0,    # Dropout on attention weights
        bias: bool = True,       # Add bias to projections
        add_bias_kv: bool = False,  # Bias for key/value at dim=0
        add_zero_attn: bool = False,  # Zero attention padding
        kdim: int = None,        # Key dimension (default: embed_dim)
        vdim: int = None,        # Value dimension (default: embed_dim)
        batch_first: bool = False,  # Input shape convention
        device=None,
        dtype=None
    )
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `embed_dim` | Total dimension of the model. Must be divisible by `num_heads` |
| `num_heads` | Number of parallel attention heads. Each head has dimension `embed_dim // num_heads` |
| `dropout` | Dropout probability on attention weights (0.0 = no dropout) |
| `bias` | Whether to add learnable bias to input/output projections |
| `kdim` | Dimension of keys (allows cross-attention with different dimensions) |
| `vdim` | Dimension of values (allows cross-attention with different dimensions) |
| `batch_first` | If True, input/output shape is (batch, seq, feature); otherwise (seq, batch, feature) |

### Internal Weights

When `kdim == vdim == embed_dim` (standard self-attention):
```python
self.in_proj_weight  # Shape: (3 * embed_dim, embed_dim) - Combined Q, K, V projection
self.in_proj_bias    # Shape: (3 * embed_dim,)
self.out_proj.weight # Shape: (embed_dim, embed_dim)
self.out_proj.bias   # Shape: (embed_dim,)
```

When dimensions differ (cross-attention):
```python
self.q_proj_weight  # Shape: (embed_dim, embed_dim)
self.k_proj_weight  # Shape: (embed_dim, kdim)
self.v_proj_weight  # Shape: (embed_dim, vdim)
```

### Forward Signature

```python
def forward(
    self,
    query: Tensor,              # (L, N, E) or (N, L, E) if batch_first
    key: Tensor,                # (S, N, E) or (N, S, E)
    value: Tensor,              # (S, N, E) or (N, S, E)
    key_padding_mask: Tensor = None,  # (N, S) - positions to ignore
    need_weights: bool = True,        # Return attention weights
    attn_mask: Tensor = None,         # (L, S) or (N*num_heads, L, S)
    average_attn_weights: bool = True,
    is_causal: bool = False           # Apply causal mask
) -> tuple[Tensor, Tensor | None]:
    # Returns: (attn_output, attn_weights)
```

### Mask Types

| Mask | Shape | Purpose |
|------|-------|---------|
| `key_padding_mask` | (N, S) | Ignore padding tokens in source |
| `attn_mask` | (L, S) or (N·H, L, S) | Prevent attention to certain positions |
| `is_causal` | bool | Auto-generate causal (triangular) mask |

**Mask semantics:**
- Boolean mask: `True` = position is **ignored** (not attended to)
- Float mask: Values are **added** to attention scores before softmax

### Causal Mask Generation

```python
def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generate upper-triangular mask filled with -inf."""
    return torch.triu(
        torch.full((sz, sz), float("-inf")),
        diagonal=1
    )
```

Result for `sz=4`:
```
[[  0, -inf, -inf, -inf],
 [  0,    0, -inf, -inf],
 [  0,    0,    0, -inf],
 [  0,    0,    0,    0]]
```

---

## TransformerEncoderLayer

A single encoder layer: self-attention + feedforward network.

### Architecture

Two variants controlled by `norm_first`:

**Post-LN (norm_first=False, original paper):**
```
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FFN(x))
```

**Pre-LN (norm_first=True, more stable training):**
```
x = x + SelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Class Definition

```python
class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,              # Model dimension
        nhead: int,                # Number of attention heads
        dim_feedforward: int = 2048,  # FFN hidden dimension
        dropout: float = 0.1,      # Dropout probability
        activation: str | Callable = F.relu,  # FFN activation
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,  # Pre-LN vs Post-LN
        bias: bool = True,
        device=None,
        dtype=None
    )
```

### Internal Structure

```python
self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, ...)
self.linear1 = Linear(d_model, dim_feedforward)
self.linear2 = Linear(dim_feedforward, d_model)
self.norm1 = LayerNorm(d_model)
self.norm2 = LayerNorm(d_model)
self.dropout = Dropout(dropout)
self.dropout1 = Dropout(dropout)
self.dropout2 = Dropout(dropout)
```

### Forward Pass

```python
def forward(
    self,
    src: Tensor,                    # Source sequence
    src_mask: Tensor = None,        # Attention mask
    src_key_padding_mask: Tensor = None,
    is_causal: bool = False
) -> Tensor:
```

---

## TransformerDecoderLayer

A single decoder layer: self-attention + cross-attention + feedforward.

### Architecture

**Post-LN:**
```
x = LayerNorm(x + SelfAttention(x))          # Masked self-attention
x = LayerNorm(x + CrossAttention(x, memory)) # Attend to encoder
x = LayerNorm(x + FFN(x))
```

**Pre-LN:**
```
x = x + SelfAttention(LayerNorm(x))
x = x + CrossAttention(LayerNorm(x), memory)
x = x + FFN(LayerNorm(x))
```

### Class Definition

```python
class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None
    )
```

### Forward Pass

```python
def forward(
    self,
    tgt: Tensor,                    # Target sequence
    memory: Tensor,                 # Encoder output
    tgt_mask: Tensor = None,        # Target attention mask (usually causal)
    memory_mask: Tensor = None,     # Cross-attention mask
    tgt_key_padding_mask: Tensor = None,
    memory_key_padding_mask: Tensor = None,
    tgt_is_causal: bool = False,
    memory_is_causal: bool = False
) -> Tensor:
```

---

## TransformerEncoder

Stack of N encoder layers.

```python
class TransformerEncoder(Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,  # Layer to clone
        num_layers: int,                          # Number of layers
        norm: Module = None,                      # Final LayerNorm
        enable_nested_tensor: bool = True,        # Optimization for padding
        mask_check: bool = True
    )
```

**Note:** The `encoder_layer` is deep-copied `num_layers` times. Layers share initial weights but diverge during training.

---

## TransformerDecoder

Stack of N decoder layers.

```python
class TransformerDecoder(Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Module = None
    )
```

---

## Transformer (Full Model)

Complete encoder-decoder transformer.

### Class Definition

```python
class Transformer(Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable = F.relu,
        custom_encoder: Module = None,  # Replace default encoder
        custom_decoder: Module = None,  # Replace default decoder
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None
    )
```

### Forward Pass

```python
def forward(
    self,
    src: Tensor,                  # Encoder input
    tgt: Tensor,                  # Decoder input
    src_mask: Tensor = None,
    tgt_mask: Tensor = None,
    memory_mask: Tensor = None,
    src_key_padding_mask: Tensor = None,
    tgt_key_padding_mask: Tensor = None,
    memory_key_padding_mask: Tensor = None,
    src_is_causal: bool = None,
    tgt_is_causal: bool = None,
    memory_is_causal: bool = False
) -> Tensor:
```

### Shape Conventions

| Tensor | batch_first=False | batch_first=True |
|--------|-------------------|------------------|
| src | (S, N, E) | (N, S, E) |
| tgt | (T, N, E) | (N, T, E) |
| output | (T, N, E) | (N, T, E) |

Where:
- S = source sequence length
- T = target sequence length
- N = batch size
- E = embedding dimension (d_model)

---

## Usage Examples

### Basic Transformer

```python
# Create transformer
transformer = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    batch_first=True
)

# Input tensors
src = torch.rand(32, 10, 512)  # (batch, src_len, d_model)
tgt = torch.rand(32, 20, 512)  # (batch, tgt_len, d_model)

# Generate causal mask for decoder
tgt_mask = transformer.generate_square_subsequent_mask(20)

# Forward pass
output = transformer(src, tgt, tgt_mask=tgt_mask)
# output shape: (32, 20, 512)
```

### Encoder-Only (BERT-style)

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=768,
    nhead=12,
    dim_feedforward=3072,
    batch_first=True,
    norm_first=True  # Pre-LN for stability
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

# Process input
x = torch.rand(16, 128, 768)  # (batch, seq_len, d_model)
output = encoder(x)
```

### Decoder-Only (GPT-style)

```python
decoder_layer = nn.TransformerDecoderLayer(
    d_model=768,
    nhead=12,
    dim_feedforward=3072,
    batch_first=True
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)

# For decoder-only, memory can be zeros or use self-attention only
x = torch.rand(16, 128, 768)
memory = torch.zeros(16, 1, 768)  # Dummy memory
causal_mask = nn.Transformer.generate_square_subsequent_mask(128)

output = decoder(x, memory, tgt_mask=causal_mask)
```

---

## Fast Path Optimizations

PyTorch implements optimized execution paths using `scaled_dot_product_attention` and FlashAttention when certain conditions are met:

### Requirements for Fast Path

1. `batch_first=True`
2. Input is batched (3D tensor)
3. No gradient computation (`torch.no_grad()` or `torch.inference_mode()`)
4. Model in eval mode (`.eval()`)
5. Activation is ReLU or GELU
6. `num_heads` is even
7. No `add_bias_kv` or `add_zero_attn`
8. `kdim == vdim == embed_dim`

When conditions are met, the forward pass uses:
- `torch._transformer_encoder_layer_fwd` for fused encoder
- Nested tensors for efficient padding handling

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `MultiheadAttention` | `mlx.nn.MultiHeadAttention` |
| `TransformerEncoderLayer` | Custom composition or `mlx.nn.TransformerEncoderLayer` |
| `scaled_dot_product_attention` | `mlx.core.fast.scaled_dot_product_attention` |

### Key Differences

1. **Mask Convention**: PyTorch uses `True` for ignored positions; verify MLX convention
2. **Weight Layout**: PyTorch uses combined `in_proj_weight` (3E, E); MLX may use separate Q/K/V
3. **Batch Dimension**: MLX typically uses batch-first by default
4. **Dropout**: MLX's functional dropout differs in API

### Porting Considerations

```python
# PyTorch weight conversion for MLX
# Combined QKV weight -> separate Q, K, V
q_weight = in_proj_weight[:embed_dim]
k_weight = in_proj_weight[embed_dim:2*embed_dim]
v_weight = in_proj_weight[2*embed_dim:]
```

---

## Gradient Formulas

### Attention Backward

For `Y = softmax(QK^T / √d) V`:

```
∂L/∂Q = (∂L/∂Y V^T) ⊙ softmax_grad(S) @ K / √d
∂L/∂K = (∂L/∂Y V^T)^T ⊙ softmax_grad(S) @ Q / √d
∂L/∂V = S^T @ ∂L/∂Y

where S = softmax(QK^T / √d)
```

The softmax gradient involves:
```
softmax_grad(x) = diag(s) - s s^T
```

### LayerNorm Backward

```
∂L/∂x = γ/σ (∂L/∂y - mean(∂L/∂y) - x̂ mean(∂L/∂y ⊙ x̂))
```

---

## Implementation Notes

### Weight Initialization

```python
# Xavier uniform for all weight matrices
for p in self.parameters():
    if p.dim() > 1:
        xavier_uniform_(p)
```

### Memory Efficiency

- Use `need_weights=False` in MultiheadAttention to avoid storing attention matrices
- Consider gradient checkpointing for long sequences
- Nested tensors reduce memory for variable-length batches

### Common Issues

1. **Shape mismatch**: Ensure `embed_dim % num_heads == 0`
2. **Mask broadcasting**: 2D masks broadcast across batch; 3D masks are per-sample
3. **Causal mask caching**: Generate once and reuse for fixed sequence lengths
