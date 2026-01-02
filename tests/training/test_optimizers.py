"""
Test Phase 5: Optimizers

Tests optimizer functionality:
- SGD, Adam, AdamW optimizers
- Parameter updates
- Momentum and weight decay
- Learning rate schedulers
"""

import sys

sys.path.insert(0, "../..")

import unittest

import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    import flashlight.nn as nn
    import flashlight.optim as optim

    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestSGD(TestCase):
    """Test SGD optimizer."""

    def test_sgd_step(self):
        """Test basic SGD step."""
        # Create a simple parameter
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True))

        # Create optimizer
        optimizer = optim.SGD([param], lr=0.1)

        # Simulate gradient
        param.grad = flashlight.tensor([1.0, 1.0, 1.0])

        # Step
        optimizer.step()

        # Check parameter was updated: param = param - lr * grad
        expected = np.array([0.9, 1.9, 2.9])
        np.testing.assert_array_almost_equal(param.numpy(), expected, decimal=5)

    def test_sgd_zero_grad(self):
        """Test zero_grad."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = optim.SGD([param], lr=0.1)

        # Set gradient
        param.grad = flashlight.randn(3, 3)
        self.assertIsNotNone(param.grad)

        # Zero gradient
        optimizer.zero_grad()
        self.assertIsNone(param.grad)

    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0], requires_grad=True))
        optimizer = optim.SGD([param], lr=0.1, momentum=0.9)

        # First step
        param.grad = flashlight.tensor([1.0])
        optimizer.step()

        # param should be: 1.0 - 0.1 * 1.0 = 0.9
        self.assertAlmostEqual(float(param.numpy()[0]), 0.9, places=5)

        # Second step (momentum should accumulate)
        optimizer.zero_grad()
        param.grad = flashlight.tensor([1.0])
        optimizer.step()

        # momentum buffer: 0.9 * 1.0 + 1.0 = 1.9
        # param: 0.9 - 0.1 * 1.9 = 0.71
        self.assertAlmostEqual(float(param.numpy()[0]), 0.71, places=5)

    def test_sgd_weight_decay(self):
        """Test SGD with weight decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0], requires_grad=True))
        optimizer = optim.SGD([param], lr=0.1, weight_decay=0.01)

        param.grad = flashlight.tensor([0.0])  # No gradient
        optimizer.step()

        # With weight decay: grad = 0.0 + 0.01 * 1.0 = 0.01
        # param = 1.0 - 0.1 * 0.01 = 0.999
        self.assertAlmostEqual(float(param.numpy()[0]), 0.999, places=5)


@skipIfNoMLX
class TestAdam(TestCase):
    """Test Adam optimizer."""

    def test_adam_step(self):
        """Test basic Adam step."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0], requires_grad=True))
        optimizer = optim.Adam([param], lr=0.1)

        # First step
        param.grad = flashlight.tensor([1.0])
        optimizer.step()

        # Adam uses bias correction, so first step should make a noticeable change
        self.assertNotAlmostEqual(float(param.numpy()[0]), 1.0)
        self.assertLess(float(param.numpy()[0]), 1.0)  # Should decrease

    def test_adam_convergence(self):
        """Test that Adam can optimize a simple function."""
        # Minimize f(x) = (x - 5)^2
        param = flashlight.nn.Parameter(flashlight.tensor([0.0], requires_grad=True))
        optimizer = optim.Adam([param], lr=0.1)

        for _ in range(100):
            optimizer.zero_grad()
            # Gradient of (x - 5)^2 is 2(x - 5)
            param.grad = 2 * (param - 5.0)
            optimizer.step()

        # Should be close to 5.0
        self.assertAlmostEqual(float(param.numpy()[0]), 5.0, places=1)

    def test_adam_weight_decay(self):
        """Test Adam with weight decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([10.0], requires_grad=True))
        optimizer = optim.Adam([param], lr=0.01, weight_decay=0.1)

        # Multiple steps with zero gradient - weight decay should shrink parameter
        for _ in range(10):
            optimizer.zero_grad()
            param.grad = flashlight.tensor([0.0])
            optimizer.step()

        # Parameter should have decreased due to weight decay
        self.assertLess(float(param.numpy()[0]), 10.0)


@skipIfNoMLX
class TestAdamW(TestCase):
    """Test AdamW optimizer."""

    def test_adamw_step(self):
        """Test basic AdamW step."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0], requires_grad=True))
        optimizer = optim.AdamW([param], lr=0.1)

        param.grad = flashlight.tensor([1.0])
        optimizer.step()

        # Should update parameter
        self.assertNotAlmostEqual(float(param.numpy()[0]), 1.0)

    def test_adamw_weight_decay(self):
        """Test AdamW decoupled weight decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([10.0], requires_grad=True))
        optimizer = optim.AdamW([param], lr=0.01, weight_decay=0.1)

        # Multiple steps - weight decay should work differently than Adam
        for _ in range(10):
            optimizer.zero_grad()
            param.grad = flashlight.tensor([0.0])
            optimizer.step()

        # Parameter should have decreased
        self.assertLess(float(param.numpy()[0]), 10.0)


@skipIfNoMLX
class TestWithModel(TestCase):
    """Test optimizers with actual models."""

    def test_optimizer_with_linear_layer(self):
        """Test optimizer updating Linear layer parameters."""
        # Create a simple model
        linear = nn.Linear(10, 5)
        optimizer = optim.SGD(linear.parameters(), lr=0.01)

        # Get initial weight
        initial_weight = linear.weight.numpy().copy()

        # Forward and backward
        x = flashlight.randn(3, 10)
        y = linear(x)
        loss = flashlight.sum(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weight should have changed
        new_weight = linear.weight.numpy()
        self.assertFalse(np.allclose(initial_weight, new_weight))

    def test_full_training_step(self):
        """Test complete training step with model, loss, and optimizer."""
        # Create model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))

        # Create optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training data
        x = flashlight.randn(5, 10, requires_grad=False)
        y_true = flashlight.randn(5, 2, requires_grad=False)

        # Training step
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        # Loss should be a scalar
        self.assertEqual(loss.shape, ())


@skipIfNoMLX
class TestLRSchedulers(TestCase):
    """Test learning rate schedulers."""

    def test_step_lr(self):
        """Test StepLR scheduler."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = optim.SGD([param], lr=0.1)
        scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.1)

        # Initial LR
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

        # Step 1: no change
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

        # Step 2: decay
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01, places=5)

    def test_exponential_lr(self):
        """Test ExponentialLR scheduler."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = optim.SGD([param], lr=1.0)
        scheduler = optim.ExponentialLR(optimizer, gamma=0.9)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.9, places=5)

        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.81, places=5)

    def test_cosine_annealing_lr(self):
        """Test CosineAnnealingLR scheduler."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = optim.SGD([param], lr=1.0)
        scheduler = optim.CosineAnnealingLR(optimizer, T_max=10)

        # At T=0, lr = initial
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

        # At T=T_max/2, lr should be around eta_min
        for _ in range(5):
            scheduler.step()

        # LR should have decreased
        self.assertLess(optimizer.param_groups[0]["lr"], 1.0)

    def test_reduce_lr_on_plateau(self):
        """Test ReduceLROnPlateau scheduler."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = optim.SGD([param], lr=1.0)
        scheduler = optim.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

        # Initial LR
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)

        # Metrics not improving
        scheduler.step(1.0)  # epoch 1
        scheduler.step(1.0)  # epoch 2
        scheduler.step(1.0)  # epoch 3
        scheduler.step(1.0)  # epoch 4 - should reduce after patience

        # LR should have been reduced
        self.assertLess(optimizer.param_groups[0]["lr"], 1.0)


if __name__ == "__main__":
    from tests.common_utils import run_tests

    run_tests()
