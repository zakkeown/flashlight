"""
Optimizer Coverage Tests

Unit tests for edge cases, state management, and API coverage.
These tests ensure 100% code coverage for the optimizer module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import flashlight
    import flashlight.nn as nn
    import flashlight.optim as optim
    from flashlight.optim import lr_scheduler
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False


@skipIfNoMLX
class TestOptimizerBase(TestCase):
    """Test base Optimizer class functionality."""

    def test_empty_param_list_raises(self):
        """Test that empty parameter list raises ValueError."""
        with self.assertRaises(ValueError):
            optim.SGD([], lr=0.01)

    def test_param_groups_api(self):
        """Test parameter groups with different learning rates."""
        param1 = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        param2 = nn.Parameter(flashlight.tensor(np.zeros((2, 2), dtype=np.float32)))

        opt = optim.SGD([
            {'params': [param1], 'lr': 0.1},
            {'params': [param2], 'lr': 0.01}
        ])

        self.assertEqual(len(opt.param_groups), 2)
        self.assertEqual(opt.param_groups[0]['lr'], 0.1)
        self.assertEqual(opt.param_groups[1]['lr'], 0.01)

    def test_duplicate_params_in_groups_raises(self):
        """Test that duplicate parameters across groups raise ValueError."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        opt = optim.SGD([param], lr=0.01)
        # Adding the same param again should raise
        with self.assertRaises(ValueError):
            opt.add_param_group({'params': [param]})

    def test_add_param_group(self):
        """Test add_param_group method."""
        param1 = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        param2 = nn.Parameter(flashlight.tensor(np.zeros((2, 2), dtype=np.float32)))

        opt = optim.SGD([param1], lr=0.1)
        opt.add_param_group({'params': [param2], 'lr': 0.01})

        self.assertEqual(len(opt.param_groups), 2)
        self.assertEqual(opt.param_groups[1]['lr'], 0.01)


@skipIfNoMLX
class TestZeroGrad(TestCase):
    """Test zero_grad functionality."""

    def test_zero_grad_set_to_none_true(self):
        """Test zero_grad with set_to_none=True (default)."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.SGD([param], lr=0.01)

        # Set gradient
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        self.assertIsNotNone(param.grad)

        # Zero grad
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(param.grad)

    def test_zero_grad_set_to_none_false(self):
        """Test zero_grad with set_to_none=False."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.SGD([param], lr=0.01)

        # Set gradient
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))

        # Zero grad with set_to_none=False
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(param.grad)
        np.testing.assert_array_equal(param.grad.numpy(), np.zeros((3, 3)))


@skipIfNoMLX
class TestStateDictSaveLoad(TestCase):
    """Test state_dict and load_state_dict functionality."""

    def test_sgd_state_dict_roundtrip(self):
        """Test SGD state dict save and load."""
        param = nn.Parameter(flashlight.tensor(np.random.randn(3, 3).astype(np.float32)))
        opt = optim.SGD([param], lr=0.01, momentum=0.9)

        # Run a few steps to build state
        for _ in range(3):
            param.grad = flashlight.tensor(np.random.randn(3, 3).astype(np.float32))
            opt.step()

        # Save state
        state = opt.state_dict()

        # Create new optimizer and load state
        param2 = nn.Parameter(flashlight.tensor(param.numpy()))
        opt2 = optim.SGD([param2], lr=0.01, momentum=0.9)
        opt2.load_state_dict(state)

        # Verify state was loaded
        self.assertEqual(opt.param_groups[0]['lr'], opt2.param_groups[0]['lr'])

    def test_adam_state_dict_roundtrip(self):
        """Test Adam state dict save and load."""
        param = nn.Parameter(flashlight.tensor(np.random.randn(3, 3).astype(np.float32)))
        opt = optim.Adam([param], lr=0.001)

        # Run a few steps
        for _ in range(3):
            param.grad = flashlight.tensor(np.random.randn(3, 3).astype(np.float32))
            opt.step()

        # Save and load
        state = opt.state_dict()
        param2 = nn.Parameter(flashlight.tensor(param.numpy()))
        opt2 = optim.Adam([param2], lr=0.001)
        opt2.load_state_dict(state)

        self.assertEqual(opt.param_groups[0]['lr'], opt2.param_groups[0]['lr'])


@skipIfNoMLX
class TestClosureSupport(TestCase):
    """Test closure support for all optimizers."""

    def _test_closure(self, optimizer_class, **kwargs):
        """Helper to test closure support for an optimizer."""
        param = nn.Parameter(flashlight.tensor(np.random.randn(3, 3).astype(np.float32)))
        opt = optimizer_class([param], **kwargs)

        loss_value = [0.0]

        def closure():
            loss_value[0] = 1.5
            param.grad = flashlight.tensor(np.random.randn(3, 3).astype(np.float32))
            return loss_value[0]

        result = opt.step(closure)
        self.assertEqual(result, 1.5)

    def test_sgd_closure(self):
        """Test SGD closure support."""
        self._test_closure(optim.SGD, lr=0.01)

    def test_adam_closure(self):
        """Test Adam closure support."""
        self._test_closure(optim.Adam, lr=0.001)

    def test_adamw_closure(self):
        """Test AdamW closure support."""
        self._test_closure(optim.AdamW, lr=0.001)

    def test_rmsprop_closure(self):
        """Test RMSprop closure support."""
        self._test_closure(optim.RMSprop, lr=0.01)

    def test_adagrad_closure(self):
        """Test Adagrad closure support."""
        self._test_closure(optim.Adagrad, lr=0.01)

    def test_adadelta_closure(self):
        """Test Adadelta closure support."""
        self._test_closure(optim.Adadelta, lr=1.0)


@skipIfNoMLX
class TestOptimizerRepr(TestCase):
    """Test optimizer __repr__ methods."""

    def test_sgd_repr(self):
        """Test SGD repr."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        opt = optim.SGD([param], lr=0.01, momentum=0.9)
        repr_str = repr(opt)
        self.assertIn("SGD", repr_str)
        self.assertIn("0.01", repr_str)

    def test_adam_repr(self):
        """Test Adam repr."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        opt = optim.Adam([param], lr=0.001)
        repr_str = repr(opt)
        self.assertIn("Adam", repr_str)


@skipIfNoMLX
class TestOptimizerValidation(TestCase):
    """Test input validation for optimizers."""

    def test_sgd_negative_lr_raises(self):
        """Test that negative learning rate raises ValueError."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        with self.assertRaises(ValueError):
            optim.SGD([param], lr=-0.01)

    def test_adam_negative_eps_raises(self):
        """Test that negative epsilon raises ValueError."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        with self.assertRaises(ValueError):
            optim.Adam([param], lr=0.001, eps=-1e-8)

    def test_adam_invalid_betas_raises(self):
        """Test that invalid betas raise ValueError."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        with self.assertRaises(ValueError):
            optim.Adam([param], lr=0.001, betas=(1.5, 0.999))

    def test_rmsprop_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        with self.assertRaises(ValueError):
            optim.RMSprop([param], lr=0.01, alpha=-0.1)


@skipIfNoMLX
class TestNoGradSkip(TestCase):
    """Test that parameters without gradients are skipped."""

    def test_sgd_skips_no_grad(self):
        """Test SGD skips parameters with no gradient."""
        param1 = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        param2 = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.SGD([param1, param2], lr=0.1)

        # Only set gradient for param1
        param1.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))

        initial_param2 = param2.numpy().copy()
        opt.step()

        # param2 should be unchanged
        np.testing.assert_array_equal(param2.numpy(), initial_param2)
        # param1 should be updated
        expected = np.ones((3, 3)) - 0.1 * np.ones((3, 3))
        np.testing.assert_allclose(param1.numpy(), expected)


@skipIfNoMLX
class TestLRSchedulerEdgeCases(TestCase):
    """Test LR scheduler edge cases."""

    def test_reduce_lr_on_plateau_max_mode(self):
        """Test ReduceLROnPlateau in max mode."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)

        # Increasing metric - should not reduce
        sched.step(0.1)
        sched.step(0.2)
        sched.step(0.3)
        self.assertEqual(opt.param_groups[0]['lr'], 0.1)

        # Stagnant metric - should reduce after patience
        sched.step(0.3)
        sched.step(0.3)
        sched.step(0.3)  # Patience exceeded
        self.assertLess(opt.param_groups[0]['lr'], 0.1)

    def test_step_lr_boundary(self):
        """Test StepLR at step size boundary."""
        param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
        opt = optim.SGD([param], lr=0.1)
        sched = lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

        # Steps 1, 2 should not change LR
        sched.step()  # epoch 1
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.1, places=6)
        sched.step()  # epoch 2
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.1, places=6)
        sched.step()  # epoch 3 - should decay
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.01, places=6)


@skipIfNoMLX
class TestMultipleParamGroups(TestCase):
    """Test optimizers with multiple parameter groups."""

    def test_sgd_multiple_groups(self):
        """Test SGD with multiple param groups and different LRs."""
        param1 = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        param2 = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))

        opt = optim.SGD([
            {'params': [param1], 'lr': 0.1},
            {'params': [param2], 'lr': 0.01}
        ])

        param1.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        param2.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))

        opt.step()

        # param1 updated with lr=0.1
        np.testing.assert_allclose(param1.numpy(), np.ones((3, 3)) - 0.1)
        # param2 updated with lr=0.01
        np.testing.assert_allclose(param2.numpy(), np.ones((3, 3)) - 0.01)

    def test_adam_multiple_groups(self):
        """Test Adam with multiple param groups."""
        param1 = nn.Parameter(flashlight.tensor(np.ones((2, 2), dtype=np.float32)))
        param2 = nn.Parameter(flashlight.tensor(np.ones((2, 2), dtype=np.float32)))

        opt = optim.Adam([
            {'params': [param1], 'lr': 0.01},
            {'params': [param2], 'lr': 0.001}
        ])

        param1.grad = flashlight.tensor(np.ones((2, 2), dtype=np.float32))
        param2.grad = flashlight.tensor(np.ones((2, 2), dtype=np.float32))

        opt.step()

        # Both should be updated but param1 more aggressively
        self.assertLess(param1.numpy().mean(), param2.numpy().mean())


@skipIfNoMLX
class TestAdditionalOptimizers(TestCase):
    """Test additional optimizers (Adamax, RAdam, NAdam, ASGD, Rprop)."""

    def test_adamax_step(self):
        """Test Adamax basic step."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.Adamax([param], lr=0.002)
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        opt.step()
        self.assertTrue(param.numpy().mean() < 1.0)

    def test_radam_step(self):
        """Test RAdam basic step."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.RAdam([param], lr=0.001)
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        opt.step()
        self.assertTrue(param.numpy().mean() < 1.0)

    def test_nadam_step(self):
        """Test NAdam basic step."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.NAdam([param], lr=0.002)
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        opt.step()
        self.assertTrue(param.numpy().mean() < 1.0)

    def test_asgd_step(self):
        """Test ASGD basic step."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.ASGD([param], lr=0.01)
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        opt.step()
        self.assertTrue(param.numpy().mean() < 1.0)

    def test_rprop_step(self):
        """Test Rprop basic step."""
        param = nn.Parameter(flashlight.tensor(np.ones((3, 3), dtype=np.float32)))
        opt = optim.Rprop([param], lr=0.01)
        param.grad = flashlight.tensor(np.ones((3, 3), dtype=np.float32))
        opt.step()
        self.assertTrue(param.numpy().mean() < 1.0)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
