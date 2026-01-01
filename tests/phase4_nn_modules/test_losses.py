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

sys.path.insert(0, "../..")

import unittest

import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestMSELoss(TestCase):
    """Test nn.MSELoss."""

    def test_mse_loss_creation(self):
        """Test MSELoss creation with default parameters."""
        criterion = flashlight.nn.MSELoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_mse_loss_forward_mean(self):
        """Test MSELoss with reduction='mean'."""
        criterion = flashlight.nn.MSELoss(reduction="mean")
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: mean((0^2 + 1^2 + 2^2 + 3^2)) = mean(0 + 1 + 4 + 9) = 3.5
        self.assertEqual(loss.shape, ())
        np.testing.assert_allclose(loss.numpy(), 3.5, rtol=1e-5)

    def test_mse_loss_forward_sum(self):
        """Test MSELoss with reduction='sum'."""
        criterion = flashlight.nn.MSELoss(reduction="sum")
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: sum((0^2 + 1^2 + 2^2 + 3^2)) = 14
        np.testing.assert_allclose(loss.numpy(), 14.0, rtol=1e-5)

    def test_mse_loss_forward_none(self):
        """Test MSELoss with reduction='none'."""
        criterion = flashlight.nn.MSELoss(reduction="none")
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        expected = np.array([[0.0, 1.0], [4.0, 9.0]])
        self.assertEqual(loss.shape, (2, 2))
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestL1Loss(TestCase):
    """Test nn.L1Loss."""

    def test_l1_loss_creation(self):
        """Test L1Loss creation with default parameters."""
        criterion = flashlight.nn.L1Loss()
        self.assertEqual(criterion.reduction, "mean")

    def test_l1_loss_forward_mean(self):
        """Test L1Loss with reduction='mean'."""
        criterion = flashlight.nn.L1Loss(reduction="mean")
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: mean(|0| + |1| + |2| + |3|) = mean(0 + 1 + 2 + 3) = 1.5
        np.testing.assert_allclose(loss.numpy(), 1.5, rtol=1e-5)

    def test_l1_loss_forward_sum(self):
        """Test L1Loss with reduction='sum'."""
        criterion = flashlight.nn.L1Loss(reduction="sum")
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        # Expected: sum(|0| + |1| + |2| + |3|) = 6
        np.testing.assert_allclose(loss.numpy(), 6.0, rtol=1e-5)

    def test_l1_loss_forward_none(self):
        """Test L1Loss with reduction='none'."""
        criterion = flashlight.nn.L1Loss(reduction="none")
        x = flashlight.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = flashlight.tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = criterion(x, y)
        expected = np.array([[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(loss.shape, (2, 2))
        np.testing.assert_allclose(loss.numpy(), expected, rtol=1e-5)


@skipIfNoMLX
class TestCrossEntropyLoss(TestCase):
    """Test nn.CrossEntropyLoss."""

    def test_cross_entropy_creation(self):
        """Test CrossEntropyLoss creation with default parameters."""
        criterion = flashlight.nn.CrossEntropyLoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_cross_entropy_forward_basic(self):
        """Test CrossEntropyLoss forward pass."""
        criterion = flashlight.nn.CrossEntropyLoss()
        # Simple case: logits where correct class has highest value
        logits = flashlight.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        targets = flashlight.tensor([0, 1], dtype=flashlight.int32)
        loss = criterion(logits, targets)
        # Loss should be small since predictions are correct
        self.assertEqual(loss.shape, ())
        self.assertLess(float(loss.numpy()), 1.0)

    def test_cross_entropy_forward_none(self):
        """Test CrossEntropyLoss with reduction='none'."""
        criterion = flashlight.nn.CrossEntropyLoss(reduction="none")
        logits = flashlight.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        targets = flashlight.tensor([0, 1], dtype=flashlight.int32)
        loss = criterion(logits, targets)
        self.assertEqual(loss.shape, (2,))


@skipIfNoMLX
class TestNLLLoss(TestCase):
    """Test nn.NLLLoss."""

    def test_nll_loss_creation(self):
        """Test NLLLoss creation with default parameters."""
        criterion = flashlight.nn.NLLLoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_nll_loss_forward(self):
        """Test NLLLoss forward pass."""
        criterion = flashlight.nn.NLLLoss()
        # Log probabilities
        log_probs = flashlight.tensor([[-0.1, -2.0, -3.0], [-2.0, -0.1, -3.0]])
        targets = flashlight.tensor([0, 1], dtype=flashlight.int32)
        loss = criterion(log_probs, targets)
        # Expected: -mean(-0.1, -0.1) = 0.1
        self.assertEqual(loss.shape, ())
        np.testing.assert_allclose(loss.numpy(), 0.1, rtol=1e-4)

    def test_nll_loss_forward_none(self):
        """Test NLLLoss with reduction='none'."""
        criterion = flashlight.nn.NLLLoss(reduction="none")
        log_probs = flashlight.tensor([[-0.5, -2.0], [-2.0, -0.5]])
        targets = flashlight.tensor([0, 1], dtype=flashlight.int32)
        loss = criterion(log_probs, targets)
        self.assertEqual(loss.shape, (2,))


@skipIfNoMLX
class TestBCELoss(TestCase):
    """Test nn.BCELoss."""

    def test_bce_loss_creation(self):
        """Test BCELoss creation with default parameters."""
        criterion = flashlight.nn.BCELoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_bce_loss_parameters(self):
        """Test BCELoss parameters."""
        criterion = flashlight.nn.BCELoss(reduction="sum")
        self.assertEqual(criterion.reduction, "sum")

    def test_bce_loss_none_reduction(self):
        """Test BCELoss with reduction='none'."""
        criterion = flashlight.nn.BCELoss(reduction="none")
        self.assertEqual(criterion.reduction, "none")


@skipIfNoMLX
class TestBCEWithLogitsLoss(TestCase):
    """Test nn.BCEWithLogitsLoss."""

    def test_bce_with_logits_creation(self):
        """Test BCEWithLogitsLoss creation with default parameters."""
        criterion = flashlight.nn.BCEWithLogitsLoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_bce_with_logits_forward_mean(self):
        """Test BCEWithLogitsLoss forward pass."""
        criterion = flashlight.nn.BCEWithLogitsLoss(reduction="mean")
        # Positive logits for positive targets
        logits = flashlight.tensor([2.0, -2.0, 1.5])
        targets = flashlight.tensor([1.0, 0.0, 1.0])
        loss = criterion(logits, targets)
        self.assertEqual(loss.shape, ())
        self.assertLess(float(loss.numpy()), 0.5)

    def test_bce_with_logits_forward_none(self):
        """Test BCEWithLogitsLoss with reduction='none'."""
        criterion = flashlight.nn.BCEWithLogitsLoss(reduction="none")
        logits = flashlight.tensor([2.0, -2.0])
        targets = flashlight.tensor([1.0, 0.0])
        loss = criterion(logits, targets)
        self.assertEqual(loss.shape, (2,))


@skipIfNoMLX
class TestSmoothL1Loss(TestCase):
    """Test nn.SmoothL1Loss."""

    def test_smooth_l1_creation(self):
        """Test SmoothL1Loss creation with default parameters."""
        criterion = flashlight.nn.SmoothL1Loss()
        self.assertEqual(criterion.reduction, "mean")
        self.assertEqual(criterion.beta, 1.0)

    def test_smooth_l1_forward_mean(self):
        """Test SmoothL1Loss forward pass."""
        criterion = flashlight.nn.SmoothL1Loss(reduction="mean")
        x = flashlight.tensor([0.5, 2.0])
        y = flashlight.tensor([0.0, 0.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())

    def test_smooth_l1_beta_parameter(self):
        """Test SmoothL1Loss with different beta values."""
        criterion = flashlight.nn.SmoothL1Loss(beta=0.5)
        self.assertEqual(criterion.beta, 0.5)


@skipIfNoMLX
class TestHuberLoss(TestCase):
    """Test nn.HuberLoss."""

    def test_huber_creation(self):
        """Test HuberLoss creation with default parameters."""
        criterion = flashlight.nn.HuberLoss()
        self.assertEqual(criterion.reduction, "mean")
        self.assertEqual(criterion.delta, 1.0)

    def test_huber_forward_mean(self):
        """Test HuberLoss forward pass."""
        criterion = flashlight.nn.HuberLoss(reduction="mean")
        x = flashlight.tensor([0.5, 2.0])
        y = flashlight.tensor([0.0, 0.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())

    def test_huber_delta_parameter(self):
        """Test HuberLoss with different delta values."""
        criterion = flashlight.nn.HuberLoss(delta=0.5)
        self.assertEqual(criterion.delta, 0.5)


@skipIfNoMLX
class TestKLDivLoss(TestCase):
    """Test nn.KLDivLoss."""

    def test_kl_div_creation(self):
        """Test KLDivLoss creation with default parameters."""
        criterion = flashlight.nn.KLDivLoss()
        self.assertEqual(criterion.reduction, "mean")
        self.assertFalse(criterion.log_target)

    def test_kl_div_forward(self):
        """Test KLDivLoss forward pass."""
        criterion = flashlight.nn.KLDivLoss(reduction="mean")
        # Input is log-probabilities, target is probabilities
        log_q = flashlight.tensor([[-1.0, -0.5, -0.3]])
        p = flashlight.tensor([[0.2, 0.3, 0.5]])
        loss = criterion(log_q, p)
        self.assertEqual(loss.shape, ())

    def test_kl_div_log_target(self):
        """Test KLDivLoss with log_target=True."""
        criterion = flashlight.nn.KLDivLoss(log_target=True)
        self.assertTrue(criterion.log_target)


@skipIfNoMLX
class TestMarginRankingLoss(TestCase):
    """Test nn.MarginRankingLoss."""

    def test_margin_ranking_creation(self):
        """Test MarginRankingLoss creation with default parameters."""
        criterion = flashlight.nn.MarginRankingLoss()
        self.assertEqual(criterion.margin, 0.0)
        self.assertEqual(criterion.reduction, "mean")

    def test_margin_ranking_forward(self):
        """Test MarginRankingLoss forward pass."""
        criterion = flashlight.nn.MarginRankingLoss(margin=1.0)
        x1 = flashlight.tensor([1.0, 2.0, 3.0])
        x2 = flashlight.tensor([0.5, 1.5, 2.5])
        y = flashlight.tensor([1.0, 1.0, 1.0])  # x1 should be ranked higher
        loss = criterion(x1, x2, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestHingeEmbeddingLoss(TestCase):
    """Test nn.HingeEmbeddingLoss."""

    def test_hinge_embedding_creation(self):
        """Test HingeEmbeddingLoss creation with default parameters."""
        criterion = flashlight.nn.HingeEmbeddingLoss()
        self.assertEqual(criterion.margin, 1.0)

    def test_hinge_embedding_forward(self):
        """Test HingeEmbeddingLoss forward pass."""
        criterion = flashlight.nn.HingeEmbeddingLoss(margin=1.0)
        x = flashlight.tensor([0.5, -0.5, 0.2])
        y = flashlight.tensor([1.0, -1.0, 1.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestCosineEmbeddingLoss(TestCase):
    """Test nn.CosineEmbeddingLoss."""

    def test_cosine_embedding_creation(self):
        """Test CosineEmbeddingLoss creation with default parameters."""
        criterion = flashlight.nn.CosineEmbeddingLoss()
        self.assertEqual(criterion.margin, 0.0)

    def test_cosine_embedding_forward(self):
        """Test CosineEmbeddingLoss forward pass."""
        criterion = flashlight.nn.CosineEmbeddingLoss()
        x1 = flashlight.tensor([[1.0, 0.0], [0.0, 1.0]])
        x2 = flashlight.tensor([[1.0, 0.0], [1.0, 0.0]])
        y = flashlight.tensor([1.0, -1.0])
        loss = criterion(x1, x2, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestSoftMarginLoss(TestCase):
    """Test nn.SoftMarginLoss."""

    def test_soft_margin_creation(self):
        """Test SoftMarginLoss creation with default parameters."""
        criterion = flashlight.nn.SoftMarginLoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_soft_margin_forward(self):
        """Test SoftMarginLoss forward pass."""
        criterion = flashlight.nn.SoftMarginLoss()
        x = flashlight.tensor([0.5, -0.5, 1.0])
        y = flashlight.tensor([1.0, -1.0, 1.0])
        loss = criterion(x, y)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestTripletMarginLoss(TestCase):
    """Test nn.TripletMarginLoss."""

    def test_triplet_margin_creation(self):
        """Test TripletMarginLoss creation with default parameters."""
        criterion = flashlight.nn.TripletMarginLoss()
        self.assertEqual(criterion.margin, 1.0)
        self.assertEqual(criterion.p, 2.0)

    def test_triplet_margin_forward(self):
        """Test TripletMarginLoss forward pass."""
        criterion = flashlight.nn.TripletMarginLoss(margin=1.0)
        anchor = flashlight.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive = flashlight.tensor([[0.9, 0.1], [0.1, 0.9]])
        negative = flashlight.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = criterion(anchor, positive, negative)
        self.assertEqual(loss.shape, ())

    def test_triplet_margin_swap(self):
        """Test TripletMarginLoss with swap=True."""
        criterion = flashlight.nn.TripletMarginLoss(swap=True)
        self.assertTrue(criterion.swap)


@skipIfNoMLX
class TestPoissonNLLLoss(TestCase):
    """Test nn.PoissonNLLLoss."""

    def test_poisson_nll_creation(self):
        """Test PoissonNLLLoss creation with default parameters."""
        criterion = flashlight.nn.PoissonNLLLoss()
        self.assertTrue(criterion.log_input)
        self.assertFalse(criterion.full)

    def test_poisson_nll_forward_log_input(self):
        """Test PoissonNLLLoss with log_input=True."""
        criterion = flashlight.nn.PoissonNLLLoss(log_input=True)
        log_input = flashlight.tensor([0.0, 1.0, 2.0])
        target = flashlight.tensor([1.0, 2.0, 3.0])
        loss = criterion(log_input, target)
        self.assertEqual(loss.shape, ())

    def test_poisson_nll_forward_no_log_input(self):
        """Test PoissonNLLLoss with log_input=False."""
        criterion = flashlight.nn.PoissonNLLLoss(log_input=False)
        input_vals = flashlight.tensor([1.0, 2.0, 3.0])
        target = flashlight.tensor([1.0, 2.0, 3.0])
        loss = criterion(input_vals, target)
        self.assertEqual(loss.shape, ())

    def test_poisson_nll_full(self):
        """Test PoissonNLLLoss with full=True."""
        criterion = flashlight.nn.PoissonNLLLoss(full=True)
        self.assertTrue(criterion.full)


@skipIfNoMLX
class TestMultiMarginLoss(TestCase):
    """Test nn.MultiMarginLoss."""

    def test_creation(self):
        """Test MultiMarginLoss creation."""
        criterion = flashlight.nn.MultiMarginLoss()
        self.assertEqual(criterion.p, 1)
        self.assertEqual(criterion.margin, 1.0)

    def test_creation_with_p(self):
        """Test MultiMarginLoss with p=2."""
        criterion = flashlight.nn.MultiMarginLoss(p=2)
        self.assertEqual(criterion.p, 2)

    def test_forward_shape(self):
        """Test MultiMarginLoss forward pass."""
        criterion = flashlight.nn.MultiMarginLoss()
        x = flashlight.randn(4, 10)
        target = flashlight.tensor([0, 1, 2, 3], dtype=flashlight.int32)
        loss = criterion(x, target)
        self.assertEqual(loss.shape, ())

    def test_forward_none(self):
        """Test MultiMarginLoss with reduction='none'."""
        criterion = flashlight.nn.MultiMarginLoss(reduction="none")
        x = flashlight.randn(4, 10)
        target = flashlight.tensor([0, 1, 2, 3], dtype=flashlight.int32)
        loss = criterion(x, target)
        self.assertEqual(loss.shape, (4,))


@skipIfNoMLX
class TestMultiLabelMarginLoss(TestCase):
    """Test nn.MultiLabelMarginLoss."""

    def test_creation(self):
        """Test MultiLabelMarginLoss creation."""
        criterion = flashlight.nn.MultiLabelMarginLoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_forward_shape(self):
        """Test MultiLabelMarginLoss forward pass."""
        criterion = flashlight.nn.MultiLabelMarginLoss()
        x = flashlight.randn(4, 10)
        target = flashlight.tensor(
            [
                [0, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [2, 3, 4, -1, -1, -1, -1, -1, -1, -1],
                [5, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [6, 7, 8, 9, -1, -1, -1, -1, -1, -1],
            ],
            dtype=flashlight.int32,
        )
        loss = criterion(x, target)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestMultiLabelSoftMarginLoss(TestCase):
    """Test nn.MultiLabelSoftMarginLoss."""

    def test_creation(self):
        """Test MultiLabelSoftMarginLoss creation."""
        criterion = flashlight.nn.MultiLabelSoftMarginLoss()
        self.assertEqual(criterion.reduction, "mean")

    def test_forward_shape(self):
        """Test MultiLabelSoftMarginLoss forward pass."""
        criterion = flashlight.nn.MultiLabelSoftMarginLoss()
        x = flashlight.randn(4, 10)
        target = flashlight.zeros(4, 10)
        loss = criterion(x, target)
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestTripletMarginWithDistanceLoss(TestCase):
    """Test nn.TripletMarginWithDistanceLoss."""

    def test_creation(self):
        """Test TripletMarginWithDistanceLoss creation."""
        criterion = flashlight.nn.TripletMarginWithDistanceLoss()
        self.assertEqual(criterion.margin, 1.0)

    def test_forward_shape(self):
        """Test TripletMarginWithDistanceLoss forward pass."""
        criterion = flashlight.nn.TripletMarginWithDistanceLoss()
        anchor = flashlight.randn(4, 10)
        positive = flashlight.randn(4, 10)
        negative = flashlight.randn(4, 10)
        loss = criterion(anchor, positive, negative)
        self.assertEqual(loss.shape, ())

    def test_custom_margin(self):
        """Test TripletMarginWithDistanceLoss with custom margin."""
        criterion = flashlight.nn.TripletMarginWithDistanceLoss(margin=2.0)
        self.assertEqual(criterion.margin, 2.0)


@skipIfNoMLX
class TestGaussianNLLLoss(TestCase):
    """Test nn.GaussianNLLLoss."""

    def test_creation(self):
        """Test GaussianNLLLoss creation."""
        criterion = flashlight.nn.GaussianNLLLoss()
        self.assertIsNotNone(criterion)

    def test_forward_shape(self):
        """Test GaussianNLLLoss forward pass."""
        criterion = flashlight.nn.GaussianNLLLoss()
        pred = flashlight.randn(4, 10)
        target = flashlight.randn(4, 10)
        var = flashlight.ones(4, 10)
        loss = criterion(pred, target, var)
        self.assertEqual(loss.shape, ())

    def test_forward_none(self):
        """Test GaussianNLLLoss with reduction='none'."""
        criterion = flashlight.nn.GaussianNLLLoss(reduction="none")
        pred = flashlight.randn(4, 10)
        target = flashlight.randn(4, 10)
        var = flashlight.ones(4, 10)
        loss = criterion(pred, target, var)
        self.assertEqual(loss.shape, (4, 10))


@skipIfNoMLX
class TestCTCLoss(TestCase):
    """Test nn.CTCLoss."""

    def test_creation(self):
        """Test CTCLoss creation."""
        criterion = flashlight.nn.CTCLoss()
        self.assertIsNotNone(criterion)

    def test_creation_with_blank(self):
        """Test CTCLoss creation with custom blank."""
        criterion = flashlight.nn.CTCLoss(blank=10)
        self.assertEqual(criterion.blank, 10)


@skipIfNoMLX
class TestAdaptiveLogSoftmaxWithLoss(TestCase):
    """Test nn.AdaptiveLogSoftmaxWithLoss."""

    def test_creation(self):
        """Test AdaptiveLogSoftmaxWithLoss creation."""
        criterion = flashlight.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=64, n_classes=1000, cutoffs=[100, 500]
        )
        self.assertEqual(criterion.in_features, 64)
        self.assertEqual(criterion.n_classes, 1000)

    def test_creation_with_div_value(self):
        """Test AdaptiveLogSoftmaxWithLoss with custom div_value."""
        criterion = flashlight.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=64, n_classes=100, cutoffs=[20, 50], div_value=2.0
        )
        self.assertEqual(criterion.div_value, 2.0)

    def test_forward(self):
        """Test AdaptiveLogSoftmaxWithLoss forward pass."""
        criterion = flashlight.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=64, n_classes=100, cutoffs=[20, 50]
        )
        x = flashlight.randn(4, 64)
        target = flashlight.tensor([0, 10, 30, 70], dtype=flashlight.int32)
        result = criterion(x, target)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
