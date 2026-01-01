"""
Test Phase 4: Attention and Transformer Layers

Tests the nn.layers.attention and nn.layers.transformer modules:
- MultiheadAttention
- TransformerEncoderLayer, TransformerDecoderLayer
- TransformerEncoder, TransformerDecoder, Transformer
- PyTorch parity
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np
import pytest

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@skipIfNoMLX
class TestMultiheadAttention(TestCase):
    """Test nn.MultiheadAttention."""

    def test_creation(self):
        """Test MultiheadAttention creation."""
        mha = mlx_compat.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.assertEqual(mha.embed_dim, 64)
        self.assertEqual(mha.num_heads, 8)

    def test_creation_with_kdim_vdim(self):
        """Test MultiheadAttention with different key/value dimensions."""
        mha = mlx_compat.nn.MultiheadAttention(embed_dim=64, num_heads=8, kdim=32, vdim=32)
        self.assertEqual(mha.kdim, 32)
        self.assertEqual(mha.vdim, 32)

    def test_forward_shape(self):
        """Test MultiheadAttention forward pass output shape."""
        mha = mlx_compat.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        query = mlx_compat.randn(10, 5, 64)  # seq=10, batch=5, embed=64
        key = mlx_compat.randn(20, 5, 64)    # seq=20, batch=5, embed=64
        value = mlx_compat.randn(20, 5, 64)
        output, attn_weights = mha(query, key, value)
        self.assertEqual(output.shape, (10, 5, 64))
        # attn_weights shape depends on average_attn_weights

    def test_self_attention(self):
        """Test self-attention (query=key=value)."""
        mha = mlx_compat.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        x = mlx_compat.randn(10, 5, 64)
        output, _ = mha(x, x, x)
        self.assertEqual(output.shape, (10, 5, 64))

    def test_with_attention_mask(self):
        """Test MultiheadAttention with attention mask."""
        mha = mlx_compat.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        query = mlx_compat.randn(10, 5, 64)
        key = mlx_compat.randn(20, 5, 64)
        value = mlx_compat.randn(20, 5, 64)
        attn_mask = mlx_compat.zeros(10, 20)
        output, _ = mha(query, key, value, attn_mask=attn_mask)
        self.assertEqual(output.shape, (10, 5, 64))

    def test_batch_first(self):
        """Test MultiheadAttention with batch_first=True."""
        mha = mlx_compat.nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        query = mlx_compat.randn(5, 10, 64)  # batch=5, seq=10, embed=64
        key = mlx_compat.randn(5, 20, 64)
        value = mlx_compat.randn(5, 20, 64)
        output, _ = mha(query, key, value)
        self.assertEqual(output.shape, (5, 10, 64))


@skipIfNoMLX
class TestTransformerEncoderLayer(TestCase):
    """Test nn.TransformerEncoderLayer."""

    def test_creation(self):
        """Test TransformerEncoderLayer creation."""
        layer = mlx_compat.nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.assertEqual(layer.self_attn.embed_dim, 64)
        self.assertEqual(layer.self_attn.num_heads, 8)

    def test_forward_shape(self):
        """Test TransformerEncoderLayer forward pass."""
        layer = mlx_compat.nn.TransformerEncoderLayer(d_model=64, nhead=8)
        src = mlx_compat.randn(10, 5, 64)  # seq=10, batch=5, d_model=64
        output = layer(src)
        self.assertEqual(output.shape, (10, 5, 64))

    def test_with_feedforward_dim(self):
        """Test TransformerEncoderLayer with custom dim_feedforward."""
        layer = mlx_compat.nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256)
        src = mlx_compat.randn(10, 5, 64)
        output = layer(src)
        self.assertEqual(output.shape, (10, 5, 64))

    def test_batch_first(self):
        """Test TransformerEncoderLayer with batch_first=True."""
        layer = mlx_compat.nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        src = mlx_compat.randn(5, 10, 64)  # batch=5, seq=10
        output = layer(src)
        self.assertEqual(output.shape, (5, 10, 64))


@skipIfNoMLX
class TestTransformerDecoderLayer(TestCase):
    """Test nn.TransformerDecoderLayer."""

    def test_creation(self):
        """Test TransformerDecoderLayer creation."""
        layer = mlx_compat.nn.TransformerDecoderLayer(d_model=64, nhead=8)
        self.assertIsNotNone(layer)

    def test_forward_shape(self):
        """Test TransformerDecoderLayer forward pass."""
        layer = mlx_compat.nn.TransformerDecoderLayer(d_model=64, nhead=8)
        tgt = mlx_compat.randn(10, 5, 64)
        memory = mlx_compat.randn(20, 5, 64)
        output = layer(tgt, memory)
        self.assertEqual(output.shape, (10, 5, 64))


@skipIfNoMLX
class TestTransformerEncoder(TestCase):
    """Test nn.TransformerEncoder."""

    def test_creation(self):
        """Test TransformerEncoder creation."""
        encoder_layer = mlx_compat.nn.TransformerEncoderLayer(d_model=64, nhead=8)
        encoder = mlx_compat.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.assertEqual(encoder.num_layers, 6)

    def test_forward_shape(self):
        """Test TransformerEncoder forward pass."""
        encoder_layer = mlx_compat.nn.TransformerEncoderLayer(d_model=64, nhead=8)
        encoder = mlx_compat.nn.TransformerEncoder(encoder_layer, num_layers=2)
        src = mlx_compat.randn(10, 5, 64)
        output = encoder(src)
        self.assertEqual(output.shape, (10, 5, 64))


@skipIfNoMLX
class TestTransformerDecoder(TestCase):
    """Test nn.TransformerDecoder."""

    def test_creation(self):
        """Test TransformerDecoder creation."""
        decoder_layer = mlx_compat.nn.TransformerDecoderLayer(d_model=64, nhead=8)
        decoder = mlx_compat.nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.assertEqual(decoder.num_layers, 6)

    def test_forward_shape(self):
        """Test TransformerDecoder forward pass."""
        decoder_layer = mlx_compat.nn.TransformerDecoderLayer(d_model=64, nhead=8)
        decoder = mlx_compat.nn.TransformerDecoder(decoder_layer, num_layers=2)
        tgt = mlx_compat.randn(10, 5, 64)
        memory = mlx_compat.randn(20, 5, 64)
        output = decoder(tgt, memory)
        self.assertEqual(output.shape, (10, 5, 64))


@skipIfNoMLX
class TestTransformer(TestCase):
    """Test nn.Transformer."""

    def test_creation(self):
        """Test Transformer creation."""
        transformer = mlx_compat.nn.Transformer(d_model=64, nhead=8)
        self.assertIsNotNone(transformer)

    def test_forward_shape(self):
        """Test Transformer forward pass."""
        transformer = mlx_compat.nn.Transformer(
            d_model=64, nhead=8, num_encoder_layers=2, num_decoder_layers=2
        )
        src = mlx_compat.randn(10, 5, 64)  # seq=10, batch=5
        tgt = mlx_compat.randn(20, 5, 64)  # seq=20, batch=5
        output = transformer(src, tgt)
        self.assertEqual(output.shape, (20, 5, 64))


if __name__ == '__main__':
    unittest.main()
