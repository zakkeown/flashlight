"""
Attention Layer Parity Tests

Tests numerical parity between flashlight attention layers and PyTorch.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import flashlight
import flashlight.nn as nn


def copy_mha_weights(mlx_mha, torch_mha):
    """Copy weights from PyTorch MultiheadAttention to flashlight."""
    # Copy in_proj_weight and in_proj_bias
    if torch_mha.in_proj_weight is not None:
        mlx_mha.in_proj_weight = nn.Parameter(
            flashlight.tensor(torch_mha.in_proj_weight.detach().numpy())
        )
    if torch_mha.in_proj_bias is not None:
        mlx_mha.in_proj_bias = nn.Parameter(
            flashlight.tensor(torch_mha.in_proj_bias.detach().numpy())
        )

    # Copy out_proj weights
    mlx_mha.out_proj.weight = nn.Parameter(
        flashlight.tensor(torch_mha.out_proj.weight.detach().numpy())
    )
    if torch_mha.out_proj.bias is not None:
        mlx_mha.out_proj.bias = nn.Parameter(
            flashlight.tensor(torch_mha.out_proj.bias.detach().numpy())
        )


class TestMultiheadAttentionParity:
    """Test MultiheadAttention parity with PyTorch."""

    @pytest.mark.parity
    def test_mha_basic_parity(self):
        """Test basic MultiheadAttention matches PyTorch."""
        batch, seq_len, embed_dim = 2, 10, 64
        num_heads = 8

        torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        mlx_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        copy_mha_weights(mlx_mha, torch_mha)

        np.random.seed(42)
        q_np = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
        k_np = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)
        v_np = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)

        torch_out, torch_attn = torch_mha(
            torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np)
        )
        mlx_out, mlx_attn = mlx_mha(
            flashlight.tensor(q_np), flashlight.tensor(k_np), flashlight.tensor(v_np)
        )

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-4, f"MHA output mismatch: {max_diff}"

    @pytest.mark.parity
    def test_mha_self_attention_parity(self):
        """Test self-attention (q=k=v) matches PyTorch."""
        batch, seq_len, embed_dim = 4, 16, 128
        num_heads = 8

        torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        mlx_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        copy_mha_weights(mlx_mha, torch_mha)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)

        torch_out, _ = torch_mha(torch.tensor(x_np), torch.tensor(x_np), torch.tensor(x_np))
        mlx_out, _ = mlx_mha(
            flashlight.tensor(x_np), flashlight.tensor(x_np), flashlight.tensor(x_np)
        )

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-4, f"MHA self-attention mismatch: {max_diff}"

    @pytest.mark.parity
    def test_mha_different_seq_lengths(self):
        """Test cross-attention with different sequence lengths."""
        batch, q_len, kv_len, embed_dim = 2, 10, 20, 64
        num_heads = 4

        torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        mlx_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        copy_mha_weights(mlx_mha, torch_mha)

        np.random.seed(42)
        q_np = np.random.randn(batch, q_len, embed_dim).astype(np.float32)
        k_np = np.random.randn(batch, kv_len, embed_dim).astype(np.float32)
        v_np = np.random.randn(batch, kv_len, embed_dim).astype(np.float32)

        torch_out, _ = torch_mha(torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np))
        mlx_out, _ = mlx_mha(
            flashlight.tensor(q_np), flashlight.tensor(k_np), flashlight.tensor(v_np)
        )

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-4, f"MHA cross-attention mismatch: {max_diff}"

    @pytest.mark.parity
    def test_mha_no_bias_parity(self):
        """Test MultiheadAttention without bias."""
        batch, seq_len, embed_dim = 2, 10, 64
        num_heads = 8

        torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
        mlx_mha = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
        copy_mha_weights(mlx_mha, torch_mha)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq_len, embed_dim).astype(np.float32)

        torch_out, _ = torch_mha(torch.tensor(x_np), torch.tensor(x_np), torch.tensor(x_np))
        mlx_out, _ = mlx_mha(
            flashlight.tensor(x_np), flashlight.tensor(x_np), flashlight.tensor(x_np)
        )

        max_diff = np.max(np.abs(torch_out.detach().numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-4, f"MHA no-bias mismatch: {max_diff}"


class TestScaledDotProductAttentionParity:
    """Test scaled_dot_product_attention parity."""

    @pytest.mark.parity
    def test_sdpa_basic_parity(self):
        """Test scaled_dot_product_attention matches PyTorch."""
        from flashlight.nn.layers.attention import scaled_dot_product_attention

        batch, heads, seq_len, head_dim = 2, 8, 16, 64

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        torch_out = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np)
        )
        mlx_out = scaled_dot_product_attention(
            flashlight.tensor(q_np), flashlight.tensor(k_np), flashlight.tensor(v_np)
        )

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"SDPA mismatch: {max_diff}"

    @pytest.mark.parity
    def test_sdpa_with_scale_parity(self):
        """Test scaled_dot_product_attention with custom scale."""
        from flashlight.nn.layers.attention import scaled_dot_product_attention

        batch, heads, seq_len, head_dim = 2, 4, 8, 32
        scale = 0.1

        np.random.seed(42)
        q_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)
        v_np = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float32)

        torch_out = torch.nn.functional.scaled_dot_product_attention(
            torch.tensor(q_np), torch.tensor(k_np), torch.tensor(v_np), scale=scale
        )
        mlx_out = scaled_dot_product_attention(
            flashlight.tensor(q_np), flashlight.tensor(k_np), flashlight.tensor(v_np), scale=scale
        )

        max_diff = np.max(np.abs(torch_out.numpy() - np.array(mlx_out.tolist())))
        assert max_diff < 1e-5, f"SDPA with scale mismatch: {max_diff}"


class TestTransformerLayerParity:
    """Test Transformer layer parity with PyTorch."""

    @pytest.mark.parity
    def test_transformer_encoder_layer_shape(self):
        """Test TransformerEncoderLayer output shape matches PyTorch."""
        batch, seq_len, d_model = 2, 10, 64
        nhead, dim_feedforward = 4, 256

        torch_layer = torch.nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        mlx_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)

        np.random.seed(42)
        x_np = np.random.randn(batch, seq_len, d_model).astype(np.float32)

        torch_out = torch_layer(torch.tensor(x_np))
        mlx_out = mlx_layer(flashlight.tensor(x_np))

        assert torch_out.shape == tuple(mlx_out.shape), "Shape mismatch"

    @pytest.mark.parity
    def test_transformer_decoder_layer_shape(self):
        """Test TransformerDecoderLayer output shape matches PyTorch."""
        batch, tgt_len, mem_len, d_model = 2, 10, 20, 64
        nhead, dim_feedforward = 4, 256

        torch_layer = torch.nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        mlx_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)

        np.random.seed(42)
        tgt_np = np.random.randn(batch, tgt_len, d_model).astype(np.float32)
        mem_np = np.random.randn(batch, mem_len, d_model).astype(np.float32)

        torch_out = torch_layer(torch.tensor(tgt_np), torch.tensor(mem_np))
        mlx_out = mlx_layer(flashlight.tensor(tgt_np), flashlight.tensor(mem_np))

        assert torch_out.shape == tuple(mlx_out.shape), "Shape mismatch"
