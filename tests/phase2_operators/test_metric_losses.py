"""
Test metric learning loss functions (triplet_margin_loss, margin_ranking_loss, etc.).
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
class TestTripletMarginLoss(TestCase):
    """Test triplet_margin_loss function."""

    def test_triplet_loss_zero(self):
        """Test triplet loss when anchor=positive and negative is far."""
        anchor = mlx_compat.tensor([[0., 0.]])
        positive = mlx_compat.tensor([[0., 0.]])  # Same as anchor
        negative = mlx_compat.tensor([[10., 10.]])  # Far away

        loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
        # d(a,p) = 0, d(a,n) >> margin, so loss should be 0
        self.assertAlmostEqual(loss.numpy().item(), 0.0, places=4)

    def test_triplet_loss_positive(self):
        """Test triplet loss when positive is far and negative is close."""
        anchor = mlx_compat.tensor([[0., 0.]])
        positive = mlx_compat.tensor([[10., 0.]])  # Far from anchor
        negative = mlx_compat.tensor([[1., 0.]])   # Close to anchor

        loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
        # d(a,p) = 10, d(a,n) = 1, loss = max(10 - 1 + 1, 0) = 10
        self.assertGreater(loss.numpy().item(), 0.0)

    def test_triplet_loss_batch(self):
        """Test triplet loss with batch."""
        np.random.seed(42)
        anchor = mlx_compat.tensor(np.random.randn(10, 64).astype(np.float32))
        positive = mlx_compat.tensor(np.random.randn(10, 64).astype(np.float32))
        negative = mlx_compat.tensor(np.random.randn(10, 64).astype(np.float32))

        loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
        # Should return a scalar
        self.assertEqual(loss.shape, ())

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_triplet_loss_parity(self):
        """Test parity with PyTorch."""
        np.random.seed(42)
        anchor_data = np.random.randn(10, 64).astype(np.float32)
        positive_data = np.random.randn(10, 64).astype(np.float32)
        negative_data = np.random.randn(10, 64).astype(np.float32)

        mlx_anchor = mlx_compat.tensor(anchor_data)
        mlx_positive = mlx_compat.tensor(positive_data)
        mlx_negative = mlx_compat.tensor(negative_data)

        torch_anchor = torch.tensor(anchor_data)
        torch_positive = torch.tensor(positive_data)
        torch_negative = torch.tensor(negative_data)

        mlx_result = F.triplet_margin_loss(mlx_anchor, mlx_positive, mlx_negative, margin=1.0)
        torch_result = torch_F.triplet_margin_loss(torch_anchor, torch_positive, torch_negative, margin=1.0)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-5
        )


@skipIfNoMLX
class TestMarginRankingLoss(TestCase):
    """Test margin_ranking_loss function."""

    def test_margin_ranking_correct_order(self):
        """Test when x1 > x2 and target=1."""
        x1 = mlx_compat.tensor([2.0])
        x2 = mlx_compat.tensor([1.0])
        target = mlx_compat.tensor([1.0])

        loss = F.margin_ranking_loss(x1, x2, target, margin=0.5)
        # loss = max(0, -1 * (2 - 1) + 0.5) = max(0, -0.5) = 0
        self.assertAlmostEqual(loss.numpy().item(), 0.0, places=4)

    def test_margin_ranking_wrong_order(self):
        """Test when x1 < x2 but target=1."""
        x1 = mlx_compat.tensor([1.0])
        x2 = mlx_compat.tensor([2.0])
        target = mlx_compat.tensor([1.0])

        loss = F.margin_ranking_loss(x1, x2, target, margin=0.5)
        # loss = max(0, -1 * (1 - 2) + 0.5) = max(0, 1.5) = 1.5
        self.assertAlmostEqual(loss.numpy().item(), 1.5, places=4)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_margin_ranking_parity(self):
        """Test parity with PyTorch."""
        np.random.seed(42)
        x1_data = np.random.randn(10).astype(np.float32)
        x2_data = np.random.randn(10).astype(np.float32)
        target_data = np.sign(np.random.randn(10)).astype(np.float32)
        target_data[target_data == 0] = 1  # Ensure no zeros

        mlx_x1 = mlx_compat.tensor(x1_data)
        mlx_x2 = mlx_compat.tensor(x2_data)
        mlx_target = mlx_compat.tensor(target_data)

        torch_x1 = torch.tensor(x1_data)
        torch_x2 = torch.tensor(x2_data)
        torch_target = torch.tensor(target_data)

        mlx_result = F.margin_ranking_loss(mlx_x1, mlx_x2, mlx_target, margin=0.5)
        torch_result = torch_F.margin_ranking_loss(torch_x1, torch_x2, torch_target, margin=0.5)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4
        )


@skipIfNoMLX
class TestHingeEmbeddingLoss(TestCase):
    """Test hinge_embedding_loss function."""

    def test_hinge_positive(self):
        """Test with positive target (y=1)."""
        x = mlx_compat.tensor([2.0])
        target = mlx_compat.tensor([1.0])

        loss = F.hinge_embedding_loss(x, target, margin=1.0)
        # For y=1: loss = x = 2.0
        self.assertAlmostEqual(loss.numpy().item(), 2.0, places=4)

    def test_hinge_negative_close(self):
        """Test with negative target (y=-1) and x close to margin."""
        x = mlx_compat.tensor([0.5])
        target = mlx_compat.tensor([-1.0])

        loss = F.hinge_embedding_loss(x, target, margin=1.0)
        # For y=-1: loss = max(0, margin - x) = max(0, 1 - 0.5) = 0.5
        self.assertAlmostEqual(loss.numpy().item(), 0.5, places=4)

    def test_hinge_negative_far(self):
        """Test with negative target (y=-1) and x > margin."""
        x = mlx_compat.tensor([2.0])
        target = mlx_compat.tensor([-1.0])

        loss = F.hinge_embedding_loss(x, target, margin=1.0)
        # For y=-1: loss = max(0, 1 - 2) = 0
        self.assertAlmostEqual(loss.numpy().item(), 0.0, places=4)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_hinge_parity(self):
        """Test parity with PyTorch."""
        np.random.seed(42)
        x_data = np.random.randn(10).astype(np.float32)
        target_data = np.sign(np.random.randn(10)).astype(np.float32)
        target_data[target_data == 0] = 1

        mlx_x = mlx_compat.tensor(x_data)
        mlx_target = mlx_compat.tensor(target_data)

        torch_x = torch.tensor(x_data)
        torch_target = torch.tensor(target_data)

        mlx_result = F.hinge_embedding_loss(mlx_x, mlx_target, margin=1.0)
        torch_result = torch_F.hinge_embedding_loss(torch_x, torch_target, margin=1.0)

        np.testing.assert_allclose(
            mlx_result.numpy(), torch_result.numpy(), rtol=1e-4, atol=1e-4
        )


if __name__ == '__main__':
    unittest.main()
