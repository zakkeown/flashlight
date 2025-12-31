"""
Test distance functions (cosine_similarity, pairwise_distance, cdist, etc.).
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    import mlx_compat.nn.functional as F
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as torch_F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@skipIfNoMLX
class TestCosineSimilarity(TestCase):
    """Test cosine_similarity function."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        x = mlx_compat.tensor([[1., 2., 3.]])
        sim = F.cosine_similarity(x, x, dim=1)
        # Identical vectors should have similarity ~1
        self.assertAlmostEqual(sim.numpy().item(), 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        x1 = mlx_compat.tensor([[1., 0.]])
        x2 = mlx_compat.tensor([[0., 1.]])
        sim = F.cosine_similarity(x1, x2, dim=1)
        # Orthogonal vectors should have similarity ~0
        self.assertAlmostEqual(sim.numpy().item(), 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        x1 = mlx_compat.tensor([[1., 2., 3.]])
        x2 = mlx_compat.tensor([[-1., -2., -3.]])
        sim = F.cosine_similarity(x1, x2, dim=1)
        # Opposite vectors should have similarity ~-1
        self.assertAlmostEqual(sim.numpy().item(), -1.0, places=5)

    def test_cosine_similarity_batch(self):
        """Test cosine similarity with batch."""
        x1 = mlx_compat.tensor([[1., 2., 3.], [4., 5., 6.]])
        x2 = mlx_compat.tensor([[1., 2., 3.], [-1., -2., -3.]])
        sim = F.cosine_similarity(x1, x2, dim=1)
        self.assertEqual(sim.shape, (2,))
        # First pair: identical, second pair: opposite of scaled version
        self.assertAlmostEqual(sim.numpy()[0], 1.0, places=5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_cosine_similarity_parity(self):
        """Test parity with PyTorch."""
        np.random.seed(42)
        x1_data = np.random.randn(10, 128).astype(np.float32)
        x2_data = np.random.randn(10, 128).astype(np.float32)

        mlx_x1 = mlx_compat.tensor(x1_data)
        mlx_x2 = mlx_compat.tensor(x2_data)
        torch_x1 = torch.tensor(x1_data)
        torch_x2 = torch.tensor(x2_data)

        mlx_result = F.cosine_similarity(mlx_x1, mlx_x2)
        torch_result = torch_F.cosine_similarity(torch_x1, torch_x2)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestPairwiseDistance(TestCase):
    """Test pairwise_distance function."""

    def test_pairwise_distance_same(self):
        """Test distance of same vectors."""
        x = mlx_compat.tensor([[1., 2., 3.]])
        dist = F.pairwise_distance(x, x)
        # Same vectors should have distance close to 0 (eps adds small offset for numerical stability)
        self.assertLess(dist.numpy().item(), 0.01)

    def test_pairwise_distance_euclidean(self):
        """Test Euclidean distance."""
        x1 = mlx_compat.tensor([[0., 0.]])
        x2 = mlx_compat.tensor([[3., 4.]])
        dist = F.pairwise_distance(x1, x2, p=2.0)
        # Distance should be 5 (3-4-5 triangle)
        self.assertAlmostEqual(dist.numpy().item(), 5.0, places=4)

    def test_pairwise_distance_manhattan(self):
        """Test Manhattan distance."""
        x1 = mlx_compat.tensor([[0., 0.]])
        x2 = mlx_compat.tensor([[3., 4.]])
        dist = F.pairwise_distance(x1, x2, p=1.0)
        # Manhattan distance should be 7
        self.assertAlmostEqual(dist.numpy().item(), 7.0, places=4)

    def test_pairwise_distance_batch(self):
        """Test with batch."""
        x1 = mlx_compat.tensor([[0., 0.], [0., 0.]])
        x2 = mlx_compat.tensor([[3., 4.], [5., 12.]])
        dist = F.pairwise_distance(x1, x2)
        expected = np.array([5., 13.])
        np.testing.assert_allclose(dist.numpy(), expected, rtol=1e-4)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_pairwise_distance_parity(self):
        """Test parity with PyTorch."""
        np.random.seed(42)
        x1_data = np.random.randn(10, 128).astype(np.float32)
        x2_data = np.random.randn(10, 128).astype(np.float32)

        mlx_x1 = mlx_compat.tensor(x1_data)
        mlx_x2 = mlx_compat.tensor(x2_data)
        torch_x1 = torch.tensor(x1_data)
        torch_x2 = torch.tensor(x2_data)

        mlx_result = F.pairwise_distance(mlx_x1, mlx_x2)
        torch_result = torch_F.pairwise_distance(torch_x1, torch_x2)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestCdist(TestCase):
    """Test cdist function."""

    def test_cdist_basic(self):
        """Test basic cdist."""
        x1 = mlx_compat.tensor([[0., 0.], [1., 1.]])  # 2 vectors
        x2 = mlx_compat.tensor([[0., 0.], [1., 0.], [0., 1.]])  # 3 vectors
        dist = F.cdist(x1, x2)
        # Shape should be (2, 3)
        self.assertEqual(dist.shape, (2, 3))
        # Distance from [0,0] to [0,0] should be 0
        self.assertAlmostEqual(dist.numpy()[0, 0], 0.0, places=4)
        # Distance from [0,0] to [1,0] should be 1
        self.assertAlmostEqual(dist.numpy()[0, 1], 1.0, places=4)

    def test_cdist_self(self):
        """Test cdist with same input."""
        x = mlx_compat.tensor([[0., 0.], [3., 4.], [1., 0.]])
        dist = F.cdist(x, x)
        # Diagonal should be zeros
        self.assertAlmostEqual(dist.numpy()[0, 0], 0.0, places=4)
        self.assertAlmostEqual(dist.numpy()[1, 1], 0.0, places=4)
        self.assertAlmostEqual(dist.numpy()[2, 2], 0.0, places=4)
        # Distance should be symmetric
        np.testing.assert_allclose(dist.numpy(), dist.numpy().T, rtol=1e-5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_cdist_parity(self):
        """Test parity with PyTorch."""
        np.random.seed(42)
        x1_data = np.random.randn(10, 128).astype(np.float32)
        x2_data = np.random.randn(20, 128).astype(np.float32)

        mlx_x1 = mlx_compat.tensor(x1_data)
        mlx_x2 = mlx_compat.tensor(x2_data)
        torch_x1 = torch.tensor(x1_data)
        torch_x2 = torch.tensor(x2_data)

        mlx_result = F.cdist(mlx_x1, mlx_x2)
        torch_result = torch.cdist(torch_x1, torch_x2)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestPdist(TestCase):
    """Test pdist function."""

    def test_pdist_basic(self):
        """Test basic pdist."""
        # 3 vectors of dimension 2
        x = mlx_compat.tensor([[0., 0.], [3., 4.], [0., 1.]])
        dist = F.pdist(x)
        # Should return (3*(3-1)/2) = 3 distances
        self.assertEqual(dist.shape[0], 3)
        # Distance from [0,0] to [3,4] = 5
        self.assertAlmostEqual(dist.numpy()[0], 5.0, places=4)
        # Distance from [0,0] to [0,1] = 1
        self.assertAlmostEqual(dist.numpy()[1], 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
