"""
Test Phase 4: Distance Layers

Tests the distance modules:
- CosineSimilarity
- PairwiseDistance
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
class TestCosineSimilarity(TestCase):
    """Test nn.CosineSimilarity."""

    def test_creation_default(self):
        """Test CosineSimilarity creation with default parameters."""
        cosine = mlx_compat.nn.CosineSimilarity()
        self.assertEqual(cosine.dim, 1)
        self.assertEqual(cosine.eps, 1e-8)

    def test_creation_custom(self):
        """Test CosineSimilarity creation with custom parameters."""
        cosine = mlx_compat.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.assertEqual(cosine.dim, 0)
        self.assertEqual(cosine.eps, 1e-6)

    def test_forward_shape(self):
        """Test CosineSimilarity forward pass output shape."""
        cosine = mlx_compat.nn.CosineSimilarity(dim=1)
        x1 = mlx_compat.randn(4, 10)
        x2 = mlx_compat.randn(4, 10)
        output = cosine(x1, x2)
        self.assertEqual(output.shape, (4,))

    def test_forward_3d(self):
        """Test CosineSimilarity with 3D input."""
        cosine = mlx_compat.nn.CosineSimilarity(dim=2)
        x1 = mlx_compat.randn(4, 5, 10)
        x2 = mlx_compat.randn(4, 5, 10)
        output = cosine(x1, x2)
        self.assertEqual(output.shape, (4, 5))

    def test_identical_vectors(self):
        """Test CosineSimilarity with identical vectors."""
        cosine = mlx_compat.nn.CosineSimilarity(dim=1)
        x = mlx_compat.randn(4, 10)
        output = cosine(x, x)
        # Cosine similarity of a vector with itself should be 1
        np.testing.assert_allclose(output.numpy(), np.ones(4), rtol=1e-5, atol=1e-6)

    def test_orthogonal_vectors(self):
        """Test CosineSimilarity with orthogonal vectors."""
        cosine = mlx_compat.nn.CosineSimilarity(dim=1)
        x1 = mlx_compat.tensor([[1.0, 0.0], [0.0, 1.0]])
        x2 = mlx_compat.tensor([[0.0, 1.0], [1.0, 0.0]])
        output = cosine(x1, x2)
        # Orthogonal vectors have cosine similarity of 0
        np.testing.assert_allclose(output.numpy(), np.zeros(2), rtol=1e-5, atol=1e-6)

    def test_opposite_vectors(self):
        """Test CosineSimilarity with opposite vectors."""
        cosine = mlx_compat.nn.CosineSimilarity(dim=1)
        x1 = mlx_compat.tensor([[1.0, 2.0, 3.0]])
        x2 = mlx_compat.tensor([[-1.0, -2.0, -3.0]])
        output = cosine(x1, x2)
        # Opposite vectors have cosine similarity of -1
        np.testing.assert_allclose(output.numpy(), np.array([-1.0]), rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        x1_np = np.random.randn(4, 10).astype(np.float32)
        x2_np = np.random.randn(4, 10).astype(np.float32)

        cosine_torch = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        cosine_mlx = mlx_compat.nn.CosineSimilarity(dim=1, eps=1e-8)

        out_torch = cosine_torch(torch.tensor(x1_np), torch.tensor(x2_np))
        out_mlx = cosine_mlx(mlx_compat.tensor(x1_np), mlx_compat.tensor(x2_np))

        np.testing.assert_allclose(
            out_torch.numpy(), out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


@skipIfNoMLX
class TestPairwiseDistance(TestCase):
    """Test nn.PairwiseDistance."""

    def test_creation_default(self):
        """Test PairwiseDistance creation with default parameters."""
        dist = mlx_compat.nn.PairwiseDistance()
        self.assertEqual(dist.p, 2.0)
        self.assertEqual(dist.eps, 1e-6)

    def test_creation_custom(self):
        """Test PairwiseDistance creation with custom parameters."""
        dist = mlx_compat.nn.PairwiseDistance(p=1.0, eps=1e-8)
        self.assertEqual(dist.p, 1.0)
        self.assertEqual(dist.eps, 1e-8)

    def test_forward_shape(self):
        """Test PairwiseDistance forward pass output shape."""
        dist = mlx_compat.nn.PairwiseDistance(p=2.0)
        x1 = mlx_compat.randn(4, 10)
        x2 = mlx_compat.randn(4, 10)
        output = dist(x1, x2)
        self.assertEqual(output.shape, (4,))

    def test_forward_3d(self):
        """Test PairwiseDistance with 3D input."""
        dist = mlx_compat.nn.PairwiseDistance(p=2.0)
        x1 = mlx_compat.randn(4, 5, 10)
        x2 = mlx_compat.randn(4, 5, 10)
        output = dist(x1, x2)
        # Last dim is reduced
        self.assertEqual(output.shape, (4, 5))

    def test_same_vectors_zero_distance(self):
        """Test PairwiseDistance with identical vectors gives near-zero."""
        dist = mlx_compat.nn.PairwiseDistance(p=2.0)
        x = mlx_compat.randn(4, 10)
        output = dist(x, x)
        # Due to eps parameter, output won't be exactly zero
        np.testing.assert_allclose(output.numpy(), np.zeros(4), rtol=1e-2, atol=1e-2)

    def test_l1_distance(self):
        """Test PairwiseDistance with p=1 (Manhattan distance)."""
        dist = mlx_compat.nn.PairwiseDistance(p=1.0)
        x1 = mlx_compat.tensor([[1.0, 2.0, 3.0]])
        x2 = mlx_compat.tensor([[4.0, 5.0, 6.0]])
        output = dist(x1, x2)
        # L1: |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        np.testing.assert_allclose(output.numpy(), np.array([9.0]), rtol=1e-5, atol=1e-6)

    def test_l2_distance(self):
        """Test PairwiseDistance with p=2 (Euclidean distance)."""
        dist = mlx_compat.nn.PairwiseDistance(p=2.0)
        x1 = mlx_compat.tensor([[0.0, 0.0, 0.0]])
        x2 = mlx_compat.tensor([[3.0, 4.0, 0.0]])
        output = dist(x1, x2)
        # L2: sqrt(3^2 + 4^2) = 5
        np.testing.assert_allclose(output.numpy(), np.array([5.0]), rtol=1e-5, atol=1e-6)

    def test_keepdim(self):
        """Test PairwiseDistance with keepdim=True."""
        dist = mlx_compat.nn.PairwiseDistance(p=2.0, keepdim=True)
        x1 = mlx_compat.randn(4, 10)
        x2 = mlx_compat.randn(4, 10)
        output = dist(x1, x2)
        self.assertEqual(output.shape, (4, 1))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_with_pytorch(self):
        """Test numerical parity with PyTorch."""
        np.random.seed(42)
        x1_np = np.random.randn(4, 10).astype(np.float32)
        x2_np = np.random.randn(4, 10).astype(np.float32)

        dist_torch = torch.nn.PairwiseDistance(p=2.0, eps=1e-6)
        dist_mlx = mlx_compat.nn.PairwiseDistance(p=2.0, eps=1e-6)

        out_torch = dist_torch(torch.tensor(x1_np), torch.tensor(x2_np))
        out_mlx = dist_mlx(mlx_compat.tensor(x1_np), mlx_compat.tensor(x2_np))

        np.testing.assert_allclose(
            out_torch.numpy(), out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    @pytest.mark.parity
    def test_parity_l1_with_pytorch(self):
        """Test L1 distance parity with PyTorch."""
        np.random.seed(42)
        x1_np = np.random.randn(4, 10).astype(np.float32)
        x2_np = np.random.randn(4, 10).astype(np.float32)

        dist_torch = torch.nn.PairwiseDistance(p=1.0, eps=1e-6)
        dist_mlx = mlx_compat.nn.PairwiseDistance(p=1.0, eps=1e-6)

        out_torch = dist_torch(torch.tensor(x1_np), torch.tensor(x2_np))
        out_mlx = dist_mlx(mlx_compat.tensor(x1_np), mlx_compat.tensor(x2_np))

        np.testing.assert_allclose(
            out_torch.numpy(), out_mlx.numpy(),
            rtol=1e-5, atol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
