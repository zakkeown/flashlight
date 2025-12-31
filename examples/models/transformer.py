"""
Transformer Architecture

Implements Transformer blocks and models.
"""

import sys
sys.path.insert(0, '../..')

import mlx_compat
import mlx_compat.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)

    Shape:
        - Input: [batch, seq_len, embed_dim]
        - Output: [batch, seq_len, embed_dim]
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: Query tensor [batch, q_len, embed_dim]
            key: Key tensor [batch, k_len, embed_dim]
            value: Value tensor [batch, v_len, embed_dim]
            attn_mask: Optional attention mask [batch, q_len, k_len]

        Returns:
            Output tensor [batch, q_len, embed_dim]
        """
        batch_size, q_len, _ = query.shape
        _, k_len, _ = key.shape
        _, v_len, _ = value.shape

        # Project and reshape to [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, num_heads, q_len, head_dim]

        k = self.k_proj(key).reshape(batch_size, k_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # [batch, num_heads, k_len, head_dim]

        v = self.v_proj(value).reshape(batch_size, v_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # [batch, num_heads, v_len, head_dim]

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # [batch, num_heads, q_len, head_dim] @ [batch, num_heads, head_dim, k_len]
        # -> [batch, num_heads, q_len, k_len]
        attn_scores = mlx_compat.matmul(q, k.transpose(2, 3)) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Softmax over last dimension
        attn_weights = mlx_compat.softmax(attn_scores, dim=-1)

        # Apply dropout
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, num_heads, q_len, k_len] @ [batch, num_heads, v_len, head_dim]
        # -> [batch, num_heads, q_len, head_dim]
        attn_output = mlx_compat.matmul(attn_weights, v)

        # Reshape back: [batch, num_heads, q_len, head_dim] -> [batch, q_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    Args:
        d_model: Model dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input: [batch, seq_len, d_model]
        - Output: [batch, seq_len, d_model]
    """

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Source tensor [batch, seq_len, d_model]
            src_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Multi-head attention with residual connection
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward with residual connection
        src2 = self.linear2(self.dropout(mlx_compat.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Args:
        d_model: Model dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout probability (default: 0.1)

    Shape:
        - Input (tgt): [batch, tgt_len, d_model]
        - Input (memory): [batch, src_len, d_model]
        - Output: [batch, tgt_len, d_model]
    """

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: Target tensor [batch, tgt_len, d_model]
            memory: Memory (encoder output) [batch, src_len, d_model]
            tgt_mask: Optional target attention mask
            memory_mask: Optional memory attention mask

        Returns:
            Output tensor [batch, tgt_len, d_model]
        """
        # Self-attention with residual connection
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with residual connection
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward with residual connection
        tgt2 = self.linear2(self.dropout(mlx_compat.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        position = mlx_compat.arange(0, max_len).reshape(-1, 1)
        div_term = mlx_compat.exp(
            mlx_compat.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = mlx_compat.zeros(max_len, d_model)
        pe[:, 0::2] = mlx_compat.sin(position * div_term)
        pe[:, 1::2] = mlx_compat.cos(position * div_term)

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Simple Transformer model for sequence-to-sequence tasks.

    Args:
        d_model: Model dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout probability (default: 0.1)
        vocab_size: Vocabulary size (default: 10000)
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        vocab_size=10000
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Linear(vocab_size, d_model)
        self.tgt_embedding = nn.Linear(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source tensor [batch, src_len, vocab_size] (one-hot or embedded)
            tgt: Target tensor [batch, tgt_len, vocab_size]
            src_mask: Optional source mask
            tgt_mask: Optional target mask

        Returns:
            Output logits [batch, tgt_len, vocab_size]
        """
        # Embed and add positional encoding
        src = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Encoder
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        # Decoder
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, src_mask)

        # Project to vocabulary
        output = self.output_proj(output)

        return output


__all__ = [
    'MultiHeadAttention',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'PositionalEncoding',
    'SimpleTransformer'
]
