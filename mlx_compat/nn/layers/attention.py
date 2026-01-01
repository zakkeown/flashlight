"""
Attention Layers

Implements attention mechanisms including Multi-Head Attention.
"""

import math
from typing import Optional, Tuple

from ...tensor import Tensor
from ..module import Module
from ..parameter import Parameter
from .linear import Linear
from .dropout import Dropout


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Scaled Dot-Product Attention.

    Computes attention as: softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
               or (batch, seq_len, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim)
             or (batch, seq_len, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim)
               or (batch, seq_len, head_dim)
        attn_mask: Optional attention mask. Can be:
                   - Boolean mask where True indicates positions to mask
                   - Float mask added to attention scores
        dropout_p: Dropout probability (default: 0.0)
        is_causal: If True, applies causal mask (default: False)
        scale: Scale factor. If None, uses 1/sqrt(head_dim) (default: None)

    Returns:
        Attention output tensor with same shape as query.

    Example:
        >>> q = mlx_compat.randn(2, 8, 10, 64)  # (batch, heads, seq, head_dim)
        >>> k = mlx_compat.randn(2, 8, 10, 64)
        >>> v = mlx_compat.randn(2, 8, 10, 64)
        >>> out = scaled_dot_product_attention(q, k, v)
        >>> out.shape
        (2, 8, 10, 64)
    """
    import mlx.core as mx

    # Get dimensions
    q_data = query._mlx_array
    k_data = key._mlx_array
    v_data = value._mlx_array

    # Determine head_dim for scaling
    head_dim = q_data.shape[-1]

    # Compute scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Compute attention scores: Q @ K^T
    # Transpose last two dimensions of key
    k_t = mx.swapaxes(k_data, -2, -1)
    attn_scores = mx.matmul(q_data, k_t) * scale

    # Apply causal mask if requested
    if is_causal:
        seq_len = q_data.shape[-2]
        # Create lower triangular mask
        causal_mask = mx.triu(
            mx.full((seq_len, seq_len), float('-inf')),
            k=1
        )
        attn_scores = attn_scores + causal_mask

    # Apply attention mask if provided
    if attn_mask is not None:
        mask_data = attn_mask._mlx_array

        # Handle boolean masks
        if mask_data.dtype == mx.bool_:
            # True positions should be masked (set to -inf)
            mask_data = mx.where(mask_data, float('-inf'), 0.0)

        attn_scores = attn_scores + mask_data

    # Softmax over last dimension
    attn_weights = mx.softmax(attn_scores, axis=-1)

    # Apply dropout
    if dropout_p > 0.0:
        # Simple dropout implementation
        mask = mx.random.uniform(shape=attn_weights.shape) > dropout_p
        attn_weights = mx.where(mask, attn_weights / (1 - dropout_p), 0.0)

    # Apply attention to values
    output = mx.matmul(attn_weights, v_data)

    result = Tensor._from_mlx_array(output)

    # Preserve gradient tracking
    if query.requires_grad or key.requires_grad or value.requires_grad:
        result.requires_grad = True

    return result


class MultiheadAttention(Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different
    representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. embed_dim must be
                   divisible by num_heads.
        dropout: Dropout probability on attention weights (default: 0.0)
        bias: If True, adds bias to input/output projection layers (default: True)
        add_bias_kv: If True, adds bias to key and value sequences (default: False)
        add_zero_attn: If True, adds a new batch of zeros to key and value
                       sequences (default: False)
        kdim: Total number of features for keys. If None, kdim=embed_dim (default: None)
        vdim: Total number of features for values. If None, vdim=embed_dim (default: None)
        batch_first: If True, input/output tensors are (batch, seq, feature).
                     If False, (seq, batch, feature). (default: True)

    Shape:
        - Query: (L, N, E) or (N, L, E) depending on batch_first
        - Key: (S, N, E) or (N, S, E) depending on batch_first
        - Value: (S, N, E) or (N, S, E) depending on batch_first
        - Output: (L, N, E) or (N, L, E) depending on batch_first

        where L is target sequence length, S is source sequence length,
        N is batch size, E is embedding dimension.

    Example:
        >>> mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        >>> query = mlx_compat.randn(32, 10, 512)  # (batch, seq, embed)
        >>> key = mlx_compat.randn(32, 20, 512)
        >>> value = mlx_compat.randn(32, 20, 512)
        >>> output, attn_weights = mha(query, key, value)
        >>> output.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ):
        """Initialize MultiheadAttention."""
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Scaling factor
        self.scale = self.head_dim ** -0.5

        # Check if Q, K, V projections can be combined
        # PyTorch combines them when kdim == vdim == embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        import mlx.core as mx

        if self._qkv_same_embed_dim:
            # Combined projection weight: [3*embed_dim, embed_dim]
            # This matches PyTorch's in_proj_weight layout
            self.in_proj_weight = Parameter(
                Tensor._from_mlx_array(mx.zeros((3 * embed_dim, embed_dim)))
            )
            if bias:
                self.in_proj_bias = Parameter(
                    Tensor._from_mlx_array(mx.zeros((3 * embed_dim,)))
                )
            else:
                self.register_parameter('in_proj_bias', None)
        else:
            # Separate projections when dimensions differ
            self.in_proj_weight = None
            self.q_proj_weight = Parameter(
                Tensor._from_mlx_array(mx.zeros((embed_dim, embed_dim)))
            )
            self.k_proj_weight = Parameter(
                Tensor._from_mlx_array(mx.zeros((embed_dim, self.kdim)))
            )
            self.v_proj_weight = Parameter(
                Tensor._from_mlx_array(mx.zeros((embed_dim, self.vdim)))
            )
            if bias:
                self.q_proj_bias = Parameter(
                    Tensor._from_mlx_array(mx.zeros((embed_dim,)))
                )
                self.k_proj_bias = Parameter(
                    Tensor._from_mlx_array(mx.zeros((embed_dim,)))
                )
                self.v_proj_bias = Parameter(
                    Tensor._from_mlx_array(mx.zeros((embed_dim,)))
                )
            else:
                self.register_parameter('q_proj_bias', None)
                self.register_parameter('k_proj_bias', None)
                self.register_parameter('v_proj_bias', None)

        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        # Optional bias for key/value
        if add_bias_kv:
            from ... import zeros
            self.bias_k = Parameter(zeros(1, 1, embed_dim))
            self.bias_v = Parameter(zeros(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Dropout for attention weights
        if dropout > 0:
            self._dropout = Dropout(dropout)
        else:
            self._dropout = None

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        import mlx.core as mx

        if self._qkv_same_embed_dim:
            # Xavier uniform for combined in_proj_weight
            std = math.sqrt(2.0 / (self.embed_dim + self.embed_dim))
            weight_data = mx.random.uniform(
                low=-std * math.sqrt(3),
                high=std * math.sqrt(3),
                shape=self.in_proj_weight.shape
            )
            self.in_proj_weight.data = Tensor._from_mlx_array(weight_data)
            if self.in_proj_bias is not None:
                self.in_proj_bias.data = Tensor._from_mlx_array(
                    mx.zeros(self.in_proj_bias.shape)
                )
        else:
            # Initialize separate projections
            for weight in [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]:
                std = math.sqrt(2.0 / (weight.shape[0] + weight.shape[1]))
                weight_data = mx.random.uniform(
                    low=-std * math.sqrt(3),
                    high=std * math.sqrt(3),
                    shape=weight.shape
                )
                weight.data = Tensor._from_mlx_array(weight_data)

        # Initialize output projection
        std = math.sqrt(2.0 / (self.out_proj.in_features + self.out_proj.out_features))
        weight_data = mx.random.uniform(
            low=-std * math.sqrt(3),
            high=std * math.sqrt(3),
            shape=self.out_proj.weight.shape
        )
        self.out_proj.weight.data = Tensor._from_mlx_array(weight_data)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data = Tensor._from_mlx_array(
                mx.zeros(self.out_proj.bias.shape)
            )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query: Query embeddings of shape (N, L, E) if batch_first else (L, N, E)
            key: Key embeddings of shape (N, S, E) if batch_first else (S, N, E)
            value: Value embeddings of shape (N, S, E) if batch_first else (S, N, E)
            key_padding_mask: If specified, a mask of shape (N, S) indicating which
                             elements should be ignored for attention.
            need_weights: If True, return attention weights (default: True)
            attn_mask: 2D or 3D mask preventing attention to certain positions.
            average_attn_weights: If True, average attention weights across heads.
            is_causal: If True, applies causal masking (upper triangle masked).

        Returns:
            Tuple of:
            - attn_output: Attention output of same shape as query
            - attn_weights: Attention weights (if need_weights=True)
        """
        import mlx.core as mx

        # Handle batch_first
        if not self.batch_first:
            # Transpose (L, N, E) -> (N, L, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project query, key, value using in_proj_weight or separate weights
        if self._qkv_same_embed_dim:
            # Use combined in_proj_weight: [3*embed_dim, embed_dim]
            # Split into q, k, v weights
            w = self.in_proj_weight._mlx_array
            w_q = w[:self.embed_dim, :]
            w_k = w[self.embed_dim:2*self.embed_dim, :]
            w_v = w[2*self.embed_dim:, :]

            if self.in_proj_bias is not None:
                b = self.in_proj_bias._mlx_array
                b_q = b[:self.embed_dim]
                b_k = b[self.embed_dim:2*self.embed_dim]
                b_v = b[2*self.embed_dim:]
            else:
                b_q = b_k = b_v = None

            # Apply projections: output = input @ weight.T + bias
            q_data = mx.matmul(query._mlx_array, mx.swapaxes(w_q, 0, 1))
            k_data = mx.matmul(key._mlx_array, mx.swapaxes(w_k, 0, 1))
            v_data = mx.matmul(value._mlx_array, mx.swapaxes(w_v, 0, 1))

            if b_q is not None:
                q_data = q_data + b_q
                k_data = k_data + b_k
                v_data = v_data + b_v

            q = Tensor._from_mlx_array(q_data)
            k = Tensor._from_mlx_array(k_data)
            v = Tensor._from_mlx_array(v_data)
        else:
            # Use separate projection weights
            q_data = mx.matmul(query._mlx_array, mx.swapaxes(self.q_proj_weight._mlx_array, 0, 1))
            k_data = mx.matmul(key._mlx_array, mx.swapaxes(self.k_proj_weight._mlx_array, 0, 1))
            v_data = mx.matmul(value._mlx_array, mx.swapaxes(self.v_proj_weight._mlx_array, 0, 1))

            if hasattr(self, 'q_proj_bias') and self.q_proj_bias is not None:
                q_data = q_data + self.q_proj_bias._mlx_array
                k_data = k_data + self.k_proj_bias._mlx_array
                v_data = v_data + self.v_proj_bias._mlx_array

            q = Tensor._from_mlx_array(q_data)
            k = Tensor._from_mlx_array(k_data)
            v = Tensor._from_mlx_array(v_data)

        # Add bias_k and bias_v if present
        if self.bias_k is not None:
            k = k + self.bias_k.expand(batch_size, -1, -1)
            v = v + self.bias_v.expand(batch_size, -1, -1)
            src_len += 1

        # Add zero attention if requested
        if self.add_zero_attn:
            from ... import zeros
            zero_attn = zeros(batch_size, 1, self.embed_dim)
            k = k.cat([k, zero_attn], dim=1)
            v = v.cat([v, zero_attn], dim=1)
            src_len += 1

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, tgt_len, head_dim)

        k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, src_len, head_dim)

        v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch, num_heads, src_len, head_dim)

        # Compute attention scores
        q_data = q._mlx_array
        k_data = k._mlx_array
        v_data = v._mlx_array

        # Q @ K^T / sqrt(d_k)
        k_t = mx.swapaxes(k_data, -2, -1)
        attn_scores = mx.matmul(q_data, k_t) * self.scale

        # Apply attention mask
        if attn_mask is not None:
            mask_data = attn_mask._mlx_array
            # Handle 2D mask -> expand to 4D
            if mask_data.ndim == 2:
                mask_data = mask_data.reshape(1, 1, tgt_len, src_len)
            elif mask_data.ndim == 3:
                mask_data = mask_data.reshape(batch_size, 1, tgt_len, src_len)

            # Handle boolean masks
            if mask_data.dtype == mx.bool_:
                mask_data = mx.where(mask_data, float('-inf'), 0.0)

            attn_scores = attn_scores + mask_data

        # Apply causal mask
        if is_causal:
            causal_mask = mx.triu(
                mx.full((tgt_len, src_len), float('-inf')),
                k=1
            )
            attn_scores = attn_scores + causal_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch, src_len) -> (batch, 1, 1, src_len)
            kpm_data = key_padding_mask._mlx_array
            if kpm_data.dtype == mx.bool_:
                kpm_data = mx.where(kpm_data, float('-inf'), 0.0)
            kpm_data = kpm_data.reshape(batch_size, 1, 1, src_len)
            attn_scores = attn_scores + kpm_data

        # Softmax
        attn_weights = mx.softmax(attn_scores, axis=-1)

        # Dropout
        if self._dropout is not None and self.training:
            attn_weights_tensor = Tensor._from_mlx_array(attn_weights)
            attn_weights_tensor = self._dropout(attn_weights_tensor)
            attn_weights = attn_weights_tensor._mlx_array

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v_data)

        # Reshape back: (batch, num_heads, tgt_len, head_dim) -> (batch, tgt_len, embed_dim)
        attn_output = mx.swapaxes(attn_output, 1, 2)
        attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        # Output projection
        attn_output = Tensor._from_mlx_array(attn_output)
        attn_output = self.out_proj(attn_output)

        # Handle batch_first for output
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Return attention weights if requested
        if need_weights:
            attn_weights_out = Tensor._from_mlx_array(attn_weights)
            if average_attn_weights:
                # Average over heads
                attn_weights_out = attn_weights_out.mean(dim=1)
            return attn_output, attn_weights_out
        else:
            return attn_output, None

    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
            f'dropout={self.dropout}, batch_first={self.batch_first}'
        )


__all__ = [
    'MultiheadAttention',
    'scaled_dot_product_attention',
]
