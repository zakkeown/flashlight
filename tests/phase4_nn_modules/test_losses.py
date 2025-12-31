"""
Test Phase 4: Loss Functions

Tests the nn.losses module:
- MSELoss, L1Loss
- CrossEntropyLoss, NLLLoss
- BCELoss, BCEWithLogitsLoss
- SmoothL1Loss, HuberLoss
- KLDivLoss, MarginRankingLoss
- HingeEmbeddingLoss, CosineEmbeddingLoss
- SoftMarginLoss, TripletMarginLoss, PoissonNLLLoss
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestMSELoss(TestCase):
    """Test nn.MSELoss."""

    def test_mse_loss_creation(self):
        """Test MSELoss creation with default parameters."""
        criterion = mlx_compat.nn.MSELoss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_mse_loss_forward_mean(self):
        """Test MSELoss with reduction='mean'."""
        criterion = mlx_compat.nn.MSELoss(reduction='mean')
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: mean((0^2 + 1^2 + 2^2 + 3^2)) = mean(0 + 1 + 4 + 9) = 3.5
        self.assertEqual(loss.shape, ())
        np.testing.assert_allclose(loss.numpy(), 3.5, rtol=1e-5)

    def test_mse_loss_forward_sum(self):
        """Test MSELoss with reduction='sum'."""
        criterion = mlx_compat.nn.MSELoss(reduction='sum')
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: sum((0^2 + 1^2 + 2^2 + 3^2)) = 14
        np.testing.assert_allclose(loss.numpy(), 14.0, rtol=1e-5)

    def test_mse_loss_forward_none(self):
        """Test MSELoss with reduction='none'."""
        criterion = mlx_compat.nn.MSELoss(reduction='none')
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        expected = np.array([[0.0, 1.0], [4.0, 9.0]])
        self.assertEqual(loss.shape, (2, 2))
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestL1Loss(TestCase):
    """Test nn.L1Loss."""

    def test_l1_loss_creation(self):
        """Test L1Loss creation with default parameters."""
        criterion = mlx_compat.nn.L1Loss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_l1_loss_forward_mean(self):
        """Test L1Loss with reduction='mean'."""
        criterion = mlx_compat.nn.L1Loss(reduction='mean')
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: mean(|0| + |1| + |2| + |3|) = mean(0 + 1 + 2 + 3) = 1.5
        np.testing.assert_allclose(loss.numpy(), 1.5, rtol=1e-5)

    def test_l1_loss_forward_sum(self):
        """Test L1Loss with reduction='sum'."""
        criterion = mlx_compat.nn.L1Loss(reduction='sum')
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: sum(|0| + |1| + |2| + |3|) = 6
        np.testing.assert_allclose(loss.numpy(), 6.0, rtol=1e-5)

    def test_l1_loss_forward_none(self):
        """Test L1Loss with reduction='none'."""
        criterion = mlx_compat.nn.L1Loss(reduction='none')
        x = mlx_compat.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = mlx_compat.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        expected = np.array([[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(loss.shape, (2, 2))
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestCrossEntropyLoss(TestCase):
    """Test nn.CrossEntropyLoss."""

    def test_cross_entropy_creation(self):
        """Test CrossEntropyLoss creation with default parameters."""
        criterion = mlx_compat.nn.CrossEntropyLoss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_cross_entropy_forward_basic(self):
        """Test CrossEntropyLoss forward pass."""
        criterion = mlx_compat.nn.CrossEntropyLoss()
        # Simple case: logits where correct class has highest value
        logits = mlx_compat.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        targets = mlx_compat.tensor([0, 1], dtype=mlx_compat.int32)
        loss = criterion(logits, targets)
        # Loss should be small since predictions are correct
        self.assertEqual(loss.shape, ())
        self.assertLess(float(loss.numpy()), 1.0)

    def test_cross_entropy_forward_none(self):
        """Test CrossEntropyLoss with reduction='none'."""
        criterion = mlx_compat.nn.CrossEntropyLoss(reduction='none')
        logits = mlx_compat.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        targets = mlx_compat.tensor([0, 1], dtype=mlx_compat.int32)
        loss = criterion(logits, targets)
        self.assertEqual(loss.shape, (2,))


@skipIfNoMLX
class TestNLLLoss(TestCase):
    """Test nn.NLLLoss."""

    def test_nll_loss_creation(self):
        """Test NLLLoss creation with default parameters."""
        criterion = mlx_compat.nn.NLLLoss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_nll_loss_forward(self):
        """Test NLLLoss forward pass."""
        criterion = mlx_compat.nn.NLLLoss()
        # Log probabilities
        log_probs = mlx_compat.tensor([[-0.1, -2.0, -3.0], [-2.0, -0.1, -3.0]])
        targets = mlx_compat.tensor([0, 1], dtype=mlx_compat.int32)
        loss = criterion(log_probs, targets)
        # Expected: -mean(-0.1, -0.1) = 0.1
        self.assertEqual(loss.shape, ())
        np.testing.assert_allclose(loss.numpy(), 0.1, rtol=1e-4)

    def test_nll_loss_forward_none(self):
        """Test NLLLoss with reduction='none'."""
        criterion = mlx_compat.nn.NLLLoss(reduction='none')
        log_probs = mlx_compat.tensor([[-0.5, -2.0], [-2.0, -0.5]])
        targets = mlx_compat.tensor([0, 1], dtype=mlx_compat.int32)
        loss = criterion(log_probs, targets)
        self.assertEqual(loss.shape, (2,))


@skipIfNoMLX
class TestBCELoss(TestCase):
    """Test nn.BCELoss."""

    def test_bce_loss_creation(self):
        """Test BCELoss creation with default parameters."""
        criterion = mlx_compat.nn.BCELoss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_bce_loss_parameters(self):
        """Test BCELoss parameters."""
        criterion = mlx_compat.nn.BCELoss(reduction='sum')
        self.assertEqual(criterion.reduction, 'sum')

    def test_bce_loss_none_reduction(self):
        """Test BCELoss with reduction='none'."""
        criterion = mlx_compat.nn.BCELoss(reduction='none')
        self.assertEqual(criterion.reduction, 'none')


@skipIfNoMLX
class TestBCEWithLogitsLoss(TestCase):
    """Test nn.BCEWithLogitsLoss."""

    def test_bce_with_logits_creation(self):
        """Test BCEWithLogitsLoss creation with default parameters."""
        criterion = mlx_compat.nn.BCEWithLogitsLoss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_bce_with_logits_forward_mean(self):
        """Test BCEWithLogitsLoss forward pass."""
        criterion = mlx_compat.nn.BCEWithLogitsLoss(reduction='mean')
        # Positive logits for positive targets
        logits = mlx_compat.tensor([2.0, -2.0, 1.5])
        targets = mlx_compat.tensor([1.0, 0.0, 1.0])
        loss = criterion(logits, targets)
        self.assertEqual(loss.shape, ())
        self.assertLess(float(loss.numpy()), 0.5)

    def test_bce_with_logits_forward_none(self):
        """Test BCEWithLogitsLoss with reduction='none'."""
        criterion = mlx_compat.nn.BCEWithLogitsLoss(reduction='none')
        logits = mlx_compat.tensor([2.0, -2.0])
        targets = mlx_compat.tensor([1.0, 0.0])
        loss = criterion(logits, targets)
        self.assertEqual(loss.shape, (2,))


@skipIfNoMLX
class TestSmoothL1Loss(TestCase):
    """Test nn.SmoothL1Loss."""

    def test_smooth_l1_creation(self):
        """Test SmoothL1Loss creation with default parameters."""
        criterion = mlx_compat.nn.SmoothL1Loss()
        self.assertEqual(criterion.reduction, 'mean')
        self.assertEqual(criterion.beta, 1.0)

    def test_smooth_l1_forward_mean(self):
        """Test SmoothL1Loss forward pass."""
        criterion = mlx_compat.nn.SmoothL1Loss(reduction='mean')
        x = mlx_compat.tensor([0.5, 2.0])
        y = mlx_compat.tensor([0.0, 0.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())

    def test_smooth_l1_beta_parameter(self):
        """Test SmoothL1Loss with different beta values."""
        criterion = mlx_compat.nn.SmoothL1Loss(beta=0.5)
        self.assertEqual(criterion.beta, 0.5)


@skipIfNoMLX
class TestHuberLoss(TestCase):
    """Test nn.HuberLoss."""

    def test_huber_creation(self):
        """Test HuberLoss creation with default parameters."""
        criterion = mlx_compat.nn.HuberLoss()
        self.assertEqual(criterion.reduction, 'mean')
        self.assertEqual(criterion.delta, 1.0)

    def test_huber_forward_mean(self):
        """Test HuberLoss forward pass."""
        criterion = mlx_compat.nn.HuberLoss(reduction='mean')
        x = mlx_compat.tensor([0.5, 2.0])
        y = mlx_compat.tensor([0.0, 0.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())

    def test_huber_delta_parameter(self):
        """Test HuberLoss with different delta values."""
        criterion = mlx_compat.nn.HuberLoss(delta=0.5)
        self.assertEqual(criterion.delta, 0.5)


@skipIfNoMLX
class TestKLDivLoss(TestCase):
    """Test nn.KLDivLoss."""

    def test_kl_div_creation(self):
        """Test KLDivLoss creation with default parameters."""
        criterion = mlx_compat.nn.KLDivLoss()
        self.assertEqual(criterion.reduction, 'mean')
        self.assertFalse(criterion.log_target)

    def test_kl_div_forward(self):
        """Test KLDivLoss forward pass."""
        criterion = mlx_compat.nn.KLDivLoss(reduction='mean')
        # Input is log-probabilities, target is probabilities
        log_q = mlx_compat.tensor([[-1.0, -0.5, -0.3]])
        p = mlx_compat.tensor([[0.2, 0.3, 0.5]])
        loss = criterion(log_q, p)
        self.assertEqual(loss.shape, ())

    def test_kl_div_log_target(self):
        """Test KLDivLoss with log_target=True."""
        criterion = mlx_compat.nn.KLDivLoss(log_target=True)
        self.assertTrue(criterion.log_target)


@skipIfNoMLX
class TestMarginRankingLoss(TestCase):
    """Test nn.MarginRankingLoss."""

    def test_margin_ranking_creation(self):
        """Test MarginRankingLoss creation with default parameters."""
        criterion = mlx_compat.nn.MarginRankingLoss()
        self.assertEqual(criterion.margin, 0.0)
        self.assertEqual(criterion.reduction, 'mean')

    def test_margin_ranking_forward(self):
        """Test MarginRankingLoss forward pass."""
        criterion = mlx_compat.nn.MarginRankingLoss(margin=1.0)
        x1 = mlx_compat.tensor([1.0, 2.0, 3.0])
        x2 = mlx_compat.tensor([0.5, 1.5, 2.5])
        y = mlx_compat.tensor([1.0, 1.0, 1.0])  # x1 should be ranked higher
        loss = criterion(x1, x2, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestHingeEmbeddingLoss(TestCase):
    """Test nn.HingeEmbeddingLoss."""

    def test_hinge_embedding_creation(self):
        """Test HingeEmbeddingLoss creation with default parameters."""
        criterion = mlx_compat.nn.HingeEmbeddingLoss()
        self.assertEqual(criterion.margin, 1.0)

    def test_hinge_embedding_forward(self):
        """Test HingeEmbeddingLoss forward pass."""
        criterion = mlx_compat.nn.HingeEmbeddingLoss(margin=1.0)
        x = mlx_compat.tensor([0.5, -0.5, 0.2])
        y = mlx_compat.tensor([1.0, -1.0, 1.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestCosineEmbeddingLoss(TestCase):
    """Test nn.CosineEmbeddingLoss."""

    def test_cosine_embedding_creation(self):
        """Test CosineEmbeddingLoss creation with default parameters."""
        criterion = mlx_compat.nn.CosineEmbeddingLoss()
        self.assertEqual(criterion.margin, 0.0)

    def test_cosine_embedding_forward(self):
        """Test CosineEmbeddingLoss forward pass."""
        criterion = mlx_compat.nn.CosineEmbeddingLoss()
        x1 = mlx_compat.tensor([[1.0, 0.0], [0.0, 1.0]])
        x2 = mlx_compat.tensor([[1.0, 0.0], [1.0, 0.0]])
        y = mlx_compat.tensor([1.0, -1.0])
        loss = criterion(x1, x2, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestSoftMarginLoss(TestCase):
    """Test nn.SoftMarginLoss."""

    def test_soft_margin_creation(self):
        """Test SoftMarginLoss creation with default parameters."""
        criterion = mlx_compat.nn.SoftMarginLoss()
        self.assertEqual(criterion.reduction, 'mean')

    def test_soft_margin_forward(self):
        """Test SoftMarginLoss forward pass."""
        criterion = mlx_compat.nn.SoftMarginLoss()
        x = mlx_compat.tensor([0.5, -0.5, 1.0])
        y = mlx_compat.tensor([1.0, -1.0, 1.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestTripletMarginLoss(TestCase):
    """Test nn.TripletMarginLoss."""

    def test_triplet_margin_creation(self):
        """Test TripletMarginLoss creation with default parameters."""
        criterion = mlx_compat.nn.TripletMarginLoss()
        self.assertEqual(criterion.margin, 1.0)
        self.assertEqual(criterion.p, 2.0)

    def test_triplet_margin_forward(self):
        """Test TripletMarginLoss forward pass."""
        criterion = mlx_compat.nn.TripletMarginLoss(margin=1.0)
        anchor = mlx_compat.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive = mlx_compat.tensor([[0.9, 0.1], [0.1, 0.9]])
        negative = mlx_compat.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = criterion(anchor, positive, negative)
        self.assertEqual(loss.shape, ())

    def test_triplet_margin_swap(self):
        """Test TripletMarginLoss with swap=True."""
        criterion = mlx_compat.nn.TripletMarginLoss(swap=True)
        self.assertTrue(criterion.swap)


@skipIfNoMLX
class TestPoissonNLLLoss(TestCase):
    """Test nn.PoissonNLLLoss."""

    def test_poisson_nll_creation(self):
        """Test PoissonNLLLoss creation with default parameters."""
        criterion = mlx_compat.nn.PoissonNLLLoss()
        self.assertTrue(criterion.log_input)
        self.assertFalse(criterion.full)

    def test_poisson_nll_forward_log_input(self):
        """Test PoissonNLLLoss with log_input=True."""
        criterion = mlx_compat.nn.PoissonNLLLoss(log_input=True)
        log_input = mlx_compat.tensor([0.0, 1.0, 2.0])
        target = mlx_compat.tensor([1.0, 2.0, 3.0])
        loss = criterion(log_input, target)
        self.assertEqual(loss.shape, ())

    def test_poisson_nll_forward_no_log_input(self):
        """Test PoissonNLLLoss with log_input=False."""
        criterion = mlx_compat.nn.PoissonNLLLoss(log_input=False)
        input_vals = mlx_compat.tensor([1.0, 2.0, 3.0])
        target = mlx_compat.tensor([1.0, 2.0, 3.0])
        loss = criterion(input_vals, target)
        self.assertEqual(loss.shape, ())

    def test_poisson_nll_full(self):
        """Test PoissonNLLLoss with full=True."""
        criterion = mlx_compat.nn.PoissonNLLLoss(full=True)
        self.assertTrue(criterion.full)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
