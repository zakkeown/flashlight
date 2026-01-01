"""
Test Phase 5: Additional Optimizers

Tests additional optimizer implementations:
- RMSprop
- Adadelta
- Adagrad
"""

import sys
sys.path.insert(0, '../..')

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestRMSprop(TestCase):
    """Test RMSprop optimizer."""

    def test_rmsprop_creation(self):
        """Test RMSprop creation with default parameters."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = flashlight.optim.RMSprop([param])
        self.assertIsNotNone(optimizer)

    def test_rmsprop_creation_with_lr(self):
        """Test RMSprop creation with custom learning rate."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = flashlight.optim.RMSprop([param], lr=0.001)
        self.assertEqual(optimizer.defaults['lr'], 0.001)

    def test_rmsprop_step(self):
        """Test RMSprop step updates parameters."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.RMSprop([param], lr=0.1)

        initial_values = param.numpy().copy()
        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()

        # Parameters should have changed
        self.assertFalse(np.allclose(param.numpy(), initial_values))

    def test_rmsprop_with_momentum(self):
        """Test RMSprop with momentum."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.RMSprop([param], lr=0.1, momentum=0.9)

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()
        # Just verify it runs without error

    def test_rmsprop_centered(self):
        """Test RMSprop with centered=True."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.RMSprop([param], lr=0.1, centered=True)

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()
        # Just verify it runs without error

    def test_rmsprop_weight_decay(self):
        """Test RMSprop with weight decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.RMSprop([param], lr=0.1, weight_decay=0.01)

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()
        # Just verify it runs without error

    def test_rmsprop_zero_grad(self):
        """Test RMSprop zero_grad clears gradients."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.RMSprop([param])

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        self.assertIsNotNone(param.grad)

        optimizer.zero_grad()
        self.assertIsNone(param.grad)


@skipIfNoMLX
class TestAdadelta(TestCase):
    """Test Adadelta optimizer."""

    def test_adadelta_creation(self):
        """Test Adadelta creation with default parameters."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = flashlight.optim.Adadelta([param])
        self.assertIsNotNone(optimizer)

    def test_adadelta_creation_with_params(self):
        """Test Adadelta creation with custom parameters."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = flashlight.optim.Adadelta([param], lr=1.0, rho=0.95, eps=1e-8)
        self.assertEqual(optimizer.defaults['rho'], 0.95)

    def test_adadelta_step(self):
        """Test Adadelta step updates parameters."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adadelta([param], lr=1.0)

        initial_values = param.numpy().copy()
        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()

        # Parameters should have changed
        self.assertFalse(np.allclose(param.numpy(), initial_values))

    def test_adadelta_weight_decay(self):
        """Test Adadelta with weight decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adadelta([param], weight_decay=0.01)

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()
        # Just verify it runs without error

    def test_adadelta_zero_grad(self):
        """Test Adadelta zero_grad clears gradients."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adadelta([param])

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.zero_grad()
        self.assertIsNone(param.grad)


@skipIfNoMLX
class TestAdagrad(TestCase):
    """Test Adagrad optimizer."""

    def test_adagrad_creation(self):
        """Test Adagrad creation with default parameters."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = flashlight.optim.Adagrad([param])
        self.assertIsNotNone(optimizer)

    def test_adagrad_creation_with_lr(self):
        """Test Adagrad creation with custom learning rate."""
        param = flashlight.nn.Parameter(flashlight.randn(3, 3))
        optimizer = flashlight.optim.Adagrad([param], lr=0.1)
        self.assertEqual(optimizer.defaults['lr'], 0.1)

    def test_adagrad_step(self):
        """Test Adagrad step updates parameters."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adagrad([param], lr=0.1)

        initial_values = param.numpy().copy()
        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()

        # Parameters should have changed
        self.assertFalse(np.allclose(param.numpy(), initial_values))

    def test_adagrad_lr_decay(self):
        """Test Adagrad with learning rate decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adagrad([param], lr=0.1, lr_decay=0.01)

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()
        # Just verify it runs without error

    def test_adagrad_weight_decay(self):
        """Test Adagrad with weight decay."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adagrad([param], weight_decay=0.01)

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.step()
        # Just verify it runs without error

    def test_adagrad_zero_grad(self):
        """Test Adagrad zero_grad clears gradients."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adagrad([param])

        param.grad = flashlight.tensor([1.0, 1.0, 1.0])
        optimizer.zero_grad()
        self.assertIsNone(param.grad)

    def test_adagrad_multiple_steps(self):
        """Test Adagrad accumulates squared gradients over multiple steps."""
        param = flashlight.nn.Parameter(flashlight.tensor([1.0, 2.0, 3.0]))
        optimizer = flashlight.optim.Adagrad([param], lr=0.1)

        values = []
        for _ in range(3):
            param.grad = flashlight.tensor([1.0, 1.0, 1.0])
            optimizer.step()
            values.append(param.numpy().copy())

        # Each step should produce a smaller update due to accumulated sum
        # Verify the sequence is changing (not stalled)
        self.assertFalse(np.allclose(values[0], values[1]))
        self.assertFalse(np.allclose(values[1], values[2]))


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
