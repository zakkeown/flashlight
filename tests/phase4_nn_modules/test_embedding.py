"""
Test Phase 4: Embedding Layers

Tests the nn.layers.embedding module:
- Embedding
- EmbeddingBag
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
class TestEmbedding(TestCase):
    """Test nn.Embedding."""

    def test_creation(self):
        """Test Embedding creation."""
        emb = mlx_compat.nn.Embedding(num_embeddings=100, embedding_dim=64)
        self.assertEqual(emb.num_embeddings, 100)
        self.assertEqual(emb.embedding_dim, 64)

    def test_forward_shape(self):
        """Test Embedding forward pass output shape."""
        emb = mlx_compat.nn.Embedding(num_embeddings=100, embedding_dim=64)
        indices = mlx_compat.tensor([1, 5, 10, 20], dtype=mlx_compat.int32)
        output = emb(indices)
        self.assertEqual(output.shape, (4, 64))

    def test_forward_2d_indices(self):
        """Test Embedding with 2D indices."""
        emb = mlx_compat.nn.Embedding(num_embeddings=100, embedding_dim=64)
        indices = mlx_compat.tensor([[1, 2, 3], [4, 5, 6]], dtype=mlx_compat.int32)
        output = emb(indices)
        self.assertEqual(output.shape, (2, 3, 64))

    def test_with_padding_idx(self):
        """Test Embedding with padding_idx."""
        emb = mlx_compat.nn.Embedding(num_embeddings=100, embedding_dim=64, padding_idx=0)
        # The embedding at index 0 should be zeros
        indices = mlx_compat.tensor([0], dtype=mlx_compat.int32)
        output = emb(indices)
        np.testing.assert_array_almost_equal(output.numpy()[0], 0)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        num_emb, emb_dim = 100, 64

        # Create PyTorch embedding
        emb_torch = torch.nn.Embedding(num_emb, emb_dim)

        # Create MLX embedding with same weights
        emb_mlx = mlx_compat.nn.Embedding(num_emb, emb_dim)
        emb_mlx.weight._mlx_array = mlx_compat.tensor(
            emb_torch.weight.detach().numpy()
        )._mlx_array

        # Test
        indices = [1, 5, 10, 20, 50]
        out_torch = emb_torch(torch.tensor(indices, dtype=torch.long))
        out_mlx = emb_mlx(mlx_compat.tensor(indices, dtype=mlx_compat.int32))

        np.testing.assert_allclose(
            out_torch.detach().numpy(),
            out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@skipIfNoMLX
class TestEmbeddingBag(TestCase):
    """Test nn.EmbeddingBag."""

    def test_creation(self):
        """Test EmbeddingBag creation."""
        emb = mlx_compat.nn.EmbeddingBag(num_embeddings=100, embedding_dim=64)
        self.assertEqual(emb.num_embeddings, 100)
        self.assertEqual(emb.embedding_dim, 64)

    def test_forward_with_offsets(self):
        """Test EmbeddingBag forward pass with offsets."""
        emb = mlx_compat.nn.EmbeddingBag(num_embeddings=100, embedding_dim=64, mode='mean')
        # Two bags: [1, 2, 3] and [4, 5]
        indices = mlx_compat.tensor([1, 2, 3, 4, 5], dtype=mlx_compat.int32)
        offsets = mlx_compat.tensor([0, 3], dtype=mlx_compat.int32)
        output = emb(indices, offsets=offsets)
        self.assertEqual(output.shape, (2, 64))

    def test_mode_sum(self):
        """Test EmbeddingBag with mode='sum'."""
        emb = mlx_compat.nn.EmbeddingBag(num_embeddings=100, embedding_dim=64, mode='sum')
        indices = mlx_compat.tensor([1, 2, 3], dtype=mlx_compat.int32)
        offsets = mlx_compat.tensor([0], dtype=mlx_compat.int32)
        output = emb(indices, offsets=offsets)
        self.assertEqual(output.shape, (1, 64))

    def test_mode_max(self):
        """Test EmbeddingBag with mode='max'."""
        emb = mlx_compat.nn.EmbeddingBag(num_embeddings=100, embedding_dim=64, mode='max')
        indices = mlx_compat.tensor([1, 2, 3], dtype=mlx_compat.int32)
        offsets = mlx_compat.tensor([0], dtype=mlx_compat.int32)
        output = emb(indices, offsets=offsets)
        self.assertEqual(output.shape, (1, 64))


if __name__ == '__main__':
    unittest.main()
