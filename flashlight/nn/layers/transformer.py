"""
Transformer Layers

Implements Transformer encoder and decoder layers following PyTorch's API.
"""

import math
from typing import Callable, Optional, Union

from ...tensor import Tensor
from ..containers import ModuleList
from ..functional import relu
from ..module import Module
from .attention import MultiheadAttention
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm


class TransformerEncoderLayer(Module):
    """
    TransformerEncoderLayer is made up of self-attention and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, et al.

    Args:
        d_model: The number of expected features in the input (required).
        nhead: The number of heads in the multiheadattention models (required).
        dim_feedforward: The dimension of the feedforward network model (default: 2048).
        dropout: The dropout value (default: 0.1).
        activation: The activation function of intermediate layer, can be a string
                   ("relu" or "gelu") or a unary callable (default: "relu").
        layer_norm_eps: The eps value in layer normalization (default: 1e-5).
        batch_first: If True, then the input and output tensors are provided
                    as (batch, seq, feature). Default: True.
        norm_first: If True, layer norm is done prior to attention and feedforward
                   operations, respectively ("Pre-LN"). Otherwise, layer norm is
                   done after ("Post-LN"). Default: False.

    Shape:
        - src: (N, S, E) if batch_first else (S, N, E)
        - src_mask: (S, S) or (N * num_heads, S, S)
        - src_key_padding_mask: (N, S)
        - Output: Same shape as src

    Example:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = flashlight.randn(32, 10, 512)  # (batch, seq, feature)
        >>> out = encoder_layer(src)
        >>> out.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """Initialize TransformerEncoderLayer."""
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Feedforward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Activation function
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = self._relu
            elif activation == "gelu":
                self.activation = self._gelu
            else:
                raise ValueError(f"Unknown activation: {activation}")
        else:
            self.activation = activation

        self.norm_first = norm_first
        self.d_model = d_model
        self.nhead = nhead

    def _relu(self, x: Tensor) -> Tensor:
        """ReLU activation."""
        from ... import relu

        return relu(x)

    def _gelu(self, x: Tensor) -> Tensor:
        """GELU activation."""
        from ... import gelu

        return gelu(x)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: The sequence to the encoder layer (required).
            src_mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional).
            is_causal: If True, applies causal masking (optional).

        Returns:
            Encoded sequence.
        """
        if self.norm_first:
            # Pre-LN: norm before attention/feedforward
            src2 = self.norm1(src)
            src2, _ = self.self_attn(
                src2,
                src2,
                src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            src = src + self.dropout1(src2)

            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-LN: norm after attention/feedforward
            src2, _ = self.self_attn(
                src,
                src,
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"d_model={self.d_model}, nhead={self.nhead}, norm_first={self.norm_first}"


class TransformerDecoderLayer(Module):
    """
    TransformerDecoderLayer is made up of self-attention, cross-attention,
    and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: The number of expected features in the input (required).
        nhead: The number of heads in the multiheadattention models (required).
        dim_feedforward: The dimension of the feedforward network model (default: 2048).
        dropout: The dropout value (default: 0.1).
        activation: The activation function (default: "relu").
        layer_norm_eps: The eps value in layer normalization (default: 1e-5).
        batch_first: If True, (batch, seq, feature). Default: True.
        norm_first: If True, layer norm is done prior to attention and feedforward.

    Shape:
        - tgt: (N, T, E) if batch_first else (T, N, E)
        - memory: (N, S, E) if batch_first else (S, N, E)
        - Output: Same shape as tgt

    Example:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = flashlight.randn(32, 20, 512)  # encoder output
        >>> tgt = flashlight.randn(32, 10, 512)     # decoder input
        >>> out = decoder_layer(tgt, memory)
        >>> out.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """Initialize TransformerDecoderLayer."""
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        # Self-attention
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Cross-attention
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Feedforward network
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Activation function
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = self._relu
            elif activation == "gelu":
                self.activation = self._gelu
            else:
                raise ValueError(f"Unknown activation: {activation}")
        else:
            self.activation = activation

        self.norm_first = norm_first
        self.d_model = d_model
        self.nhead = nhead

    def _relu(self, x: Tensor) -> Tensor:
        """ReLU activation."""
        from ... import relu

        return relu(x)

    def _gelu(self, x: Tensor) -> Tensor:
        """GELU activation."""
        from ... import gelu

        return gelu(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """
        Pass the inputs through the decoder layer.

        Args:
            tgt: The sequence to the decoder layer (required).
            memory: The sequence from the encoder (required).
            tgt_mask: The mask for the tgt sequence (optional).
            memory_mask: The mask for the memory sequence (optional).
            tgt_key_padding_mask: The mask for the tgt keys per batch (optional).
            memory_key_padding_mask: The mask for the memory keys per batch (optional).
            tgt_is_causal: If True, applies causal masking to tgt (optional).
            memory_is_causal: If True, applies causal masking to memory (optional).

        Returns:
            Decoded sequence.
        """
        if self.norm_first:
            # Pre-LN
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(
                tgt2,
                tgt2,
                tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            tgt = tgt + self.dropout1(tgt2)

            tgt2 = self.norm2(tgt)
            tgt2, _ = self.multihead_attn(
                tgt2,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            tgt = tgt + self.dropout2(tgt2)

            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-LN
            tgt2, _ = self.self_attn(
                tgt,
                tgt,
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                is_causal=tgt_is_causal,
            )
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            tgt2, _ = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                is_causal=memory_is_causal,
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"d_model={self.d_model}, nhead={self.nhead}, norm_first={self.norm_first}"


class TransformerEncoder(Module):
    """
    TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: An instance of TransformerEncoderLayer (required).
        num_layers: The number of sub-encoder-layers (required).
        norm: The layer normalization component (optional).
        enable_nested_tensor: Enable nested tensor optimization (not yet supported).
        mask_check: Enable mask checking (not yet supported).

    Shape:
        - src: (N, S, E) if batch_first else (S, N, E)
        - Output: Same shape as src

    Example:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = flashlight.randn(32, 10, 512)
        >>> out = encoder(src)
        >>> out.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ):
        """Initialize TransformerEncoder."""
        super().__init__()

        # Create copies of the encoder layer
        self.layers = ModuleList([self._clone_layer(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def _clone_layer(self, layer: TransformerEncoderLayer) -> TransformerEncoderLayer:
        """Create a new layer with the same configuration."""
        return TransformerEncoderLayer(
            d_model=layer.d_model,
            nhead=layer.nhead,
            dim_feedforward=layer.linear1.out_features,
            dropout=layer.dropout.p if hasattr(layer.dropout, "p") else 0.1,
            activation=layer.activation,
            layer_norm_eps=layer.norm1.eps,
            batch_first=layer.self_attn.batch_first,
            norm_first=layer.norm_first,
        )

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Pass the input through all encoder layers.

        Args:
            src: The sequence to the encoder (required).
            mask: The mask for the src sequence (optional).
            src_key_padding_mask: The mask for the src keys per batch (optional).
            is_causal: If True, applies causal masking (optional).

        Returns:
            Encoded sequence.
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    """
    TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: An instance of TransformerDecoderLayer (required).
        num_layers: The number of sub-decoder-layers (required).
        norm: The layer normalization component (optional).

    Shape:
        - tgt: (N, T, E) if batch_first else (T, N, E)
        - memory: (N, S, E) if batch_first else (S, N, E)
        - Output: Same shape as tgt

    Example:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = flashlight.randn(32, 20, 512)  # encoder output
        >>> tgt = flashlight.randn(32, 10, 512)
        >>> out = decoder(tgt, memory)
        >>> out.shape
        (32, 10, 512)
    """

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        """Initialize TransformerDecoder."""
        super().__init__()

        # Create copies of the decoder layer
        self.layers = ModuleList([self._clone_layer(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def _clone_layer(self, layer: TransformerDecoderLayer) -> TransformerDecoderLayer:
        """Create a new layer with the same configuration."""
        return TransformerDecoderLayer(
            d_model=layer.d_model,
            nhead=layer.nhead,
            dim_feedforward=layer.linear1.out_features,
            dropout=layer.dropout.p if hasattr(layer.dropout, "p") else 0.1,
            activation=layer.activation,
            layer_norm_eps=layer.norm1.eps,
            batch_first=layer.self_attn.batch_first,
            norm_first=layer.norm_first,
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """
        Pass the inputs through all decoder layers.

        Args:
            tgt: The sequence to the decoder (required).
            memory: The sequence from the encoder (required).
            tgt_mask: The mask for the tgt sequence (optional).
            memory_mask: The mask for the memory sequence (optional).
            tgt_key_padding_mask: The mask for the tgt keys per batch (optional).
            memory_key_padding_mask: The mask for the memory keys per batch (optional).
            tgt_is_causal: If True, applies causal masking to tgt (optional).
            memory_is_causal: If True, applies causal masking to memory (optional).

        Returns:
            Decoded sequence.
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(Module):
    """
    A transformer model.

    User is able to modify the attributes as needed. The architecture is based on
    the paper "Attention Is All You Need".

    Args:
        d_model: The number of expected features in the encoder/decoder inputs (default: 512).
        nhead: The number of heads in the multiheadattention models (default: 8).
        num_encoder_layers: The number of encoder layers (default: 6).
        num_decoder_layers: The number of decoder layers (default: 6).
        dim_feedforward: The dimension of the feedforward network (default: 2048).
        dropout: The dropout value (default: 0.1).
        activation: The activation function (default: "relu").
        custom_encoder: Custom encoder (optional).
        custom_decoder: Custom decoder (optional).
        layer_norm_eps: The eps value in layer normalization (default: 1e-5).
        batch_first: If True, (batch, seq, feature). Default: True.
        norm_first: If True, layer norm before attention/feedforward. Default: False.

    Example:
        >>> transformer = nn.Transformer(d_model=512, nhead=8)
        >>> src = flashlight.randn(32, 10, 512)
        >>> tgt = flashlight.randn(32, 20, 512)
        >>> out = transformer(src, tgt)
        >>> out.shape
        (32, 20, 512)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = relu,
        custom_encoder: Optional[Module] = None,
        custom_decoder: Optional[Module] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """Initialize Transformer."""
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
            )
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.encoder = TransformerEncoder(
                encoder_layer,
                num_encoder_layers,
                encoder_norm,
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
            )
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_decoder_layers,
                decoder_norm,
            )

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        for p in self.parameters():
            if p.ndim > 1:
                import mlx.core as mx

                std = math.sqrt(2.0 / sum(p.shape))
                data = mx.random.uniform(
                    low=-std * math.sqrt(3), high=std * math.sqrt(3), shape=p.shape
                )
                p.data = Tensor._from_mlx_array(data)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """
        Take in and process masked source/target sequences.

        Args:
            src: The sequence to the encoder (required).
            tgt: The sequence to the decoder (required).
            src_mask: The additive mask for the src sequence (optional).
            tgt_mask: The additive mask for the tgt sequence (optional).
            memory_mask: The additive mask for the encoder output (optional).
            src_key_padding_mask: The ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: The ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: The ByteTensor mask for memory keys per batch (optional).
            src_is_causal: If True, applies causal masking to src (optional).
            tgt_is_causal: If True, applies causal masking to tgt (optional).
            memory_is_causal: If True, applies causal masking to memory (optional).

        Returns:
            Transformer output.
        """
        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )

        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """
        Generate a square causal mask for self-attention.

        The masked positions are filled with float('-inf').
        Unmasked positions are filled with 0.0.

        Args:
            sz: Size of the mask (sequence length).

        Returns:
            Causal mask of shape (sz, sz).

        Example:
            >>> mask = nn.Transformer.generate_square_subsequent_mask(5)
            >>> mask
            tensor([[0., -inf, -inf, -inf, -inf],
                    [0., 0., -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., 0.]])
        """
        import mlx.core as mx

        mask = mx.triu(mx.full((sz, sz), float("-inf")), k=1)
        return Tensor._from_mlx_array(mask)


__all__ = [
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
]
