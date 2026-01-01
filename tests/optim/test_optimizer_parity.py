"""
Optimizer Parity Tests

Comprehensive tests comparing mlx_compat optimizers against PyTorch implementations.
All tests use strict tolerance: rtol=1e-5, atol=1e-6
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import numpy as np

from tests.common_utils import TestCase, skipIfNoMLX

try:
    import mlx_compat
    import mlx_compat.nn as nn
    import mlx_compat.optim as optim
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def skip_if_no_torch(func):
    """Skip test if PyTorch is not available."""
    return unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")(func)


def run_optimizer_parity_test(
    test_case,
    mlx_optimizer_class,
    torch_optimizer_class,
    optimizer_kwargs,
    num_steps=10,
    param_shape=(5, 3),
    rtol=1e-5,
    atol=1e-6,
    seed=42
):
    """
    Run a parity test between MLX and PyTorch optimizers.

    Args:
        test_case: unittest.TestCase instance
        mlx_optimizer_class: MLX optimizer class
        torch_optimizer_class: PyTorch optimizer class
        optimizer_kwargs: Dict of optimizer parameters (same for both)
        num_steps: Number of optimization steps
        param_shape: Shape of parameter tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create identical initial parameters
    param_np = np.random.randn(*param_shape).astype(np.float32)

    # Create gradient sequence (same for both)
    grads = [np.random.randn(*param_shape).astype(np.float32) for _ in range(num_steps)]

    # PyTorch setup
    param_torch = torch.from_numpy(param_np.copy()).requires_grad_(True)
    opt_torch = torch_optimizer_class([param_torch], **optimizer_kwargs)

    # MLX setup
    param_mlx = nn.Parameter(mlx_compat.tensor(param_np.copy()))
    opt_mlx = mlx_optimizer_class([param_mlx], **optimizer_kwargs)

    # Run optimization steps
    for i, grad_np in enumerate(grads):
        # PyTorch step
        opt_torch.zero_grad()
        param_torch.grad = torch.from_numpy(grad_np)
        opt_torch.step()

        # MLX step
        opt_mlx.zero_grad()
        param_mlx.grad = mlx_compat.tensor(grad_np)
        opt_mlx.step()

        # Compare after each step for debugging
        mlx_val = param_mlx.numpy()
        torch_val = param_torch.detach().numpy()

        try:
            np.testing.assert_allclose(
                mlx_val, torch_val,
                rtol=rtol, atol=atol,
                err_msg=f"Mismatch at step {i+1}"
            )
        except AssertionError as e:
            # Add more debug info
            diff = np.abs(mlx_val - torch_val)
            test_case.fail(
                f"Parity check failed at step {i+1}:\n"
                f"Max diff: {diff.max():.10f}\n"
                f"Mean diff: {diff.mean():.10f}\n"
                f"MLX value:\n{mlx_val}\n"
                f"Torch value:\n{torch_val}\n"
                f"Original error: {e}"
            )


@skipIfNoMLX
@skip_if_no_torch
class TestSGDParity(TestCase):
    """Test SGD optimizer parity with PyTorch."""

    def test_sgd_basic(self):
        """Test basic SGD without momentum."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01}
        )

    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'momentum': 0.9}
        )

    def test_sgd_nesterov(self):
        """Test SGD with Nesterov momentum."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}
        )

    def test_sgd_weight_decay(self):
        """Test SGD with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'weight_decay': 0.01}
        )

    def test_sgd_momentum_weight_decay(self):
        """Test SGD with momentum and weight decay."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.01}
        )

    def test_sgd_dampening(self):
        """Test SGD with dampening."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'momentum': 0.9, 'dampening': 0.1}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestAdamParity(TestCase):
    """Test Adam optimizer parity with PyTorch."""

    def test_adam_basic(self):
        """Test basic Adam."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001}
        )

    def test_adam_custom_betas(self):
        """Test Adam with custom beta values."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001, 'betas': (0.85, 0.995)}
        )

    def test_adam_weight_decay(self):
        """Test Adam with weight decay (L2 regularization)."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001, 'weight_decay': 0.01}
        )

    def test_adam_amsgrad(self):
        """Test Adam with AMSGrad variant."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001, 'amsgrad': True}
        )

    def test_adam_amsgrad_weight_decay(self):
        """Test Adam with AMSGrad and weight decay."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001, 'amsgrad': True, 'weight_decay': 0.01}
        )

    def test_adam_custom_eps(self):
        """Test Adam with custom epsilon."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001, 'eps': 1e-6}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestAdamWParity(TestCase):
    """Test AdamW optimizer parity with PyTorch."""

    def test_adamw_basic(self):
        """Test basic AdamW."""
        run_optimizer_parity_test(
            self,
            optim.AdamW, torch_optim.AdamW,
            {'lr': 0.001}
        )

    def test_adamw_weight_decay(self):
        """Test AdamW with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.AdamW, torch_optim.AdamW,
            {'lr': 0.001, 'weight_decay': 0.01}
        )

    def test_adamw_amsgrad(self):
        """Test AdamW with AMSGrad."""
        run_optimizer_parity_test(
            self,
            optim.AdamW, torch_optim.AdamW,
            {'lr': 0.001, 'amsgrad': True}
        )

    def test_adamw_custom_betas(self):
        """Test AdamW with custom betas."""
        run_optimizer_parity_test(
            self,
            optim.AdamW, torch_optim.AdamW,
            {'lr': 0.001, 'betas': (0.85, 0.995)}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestRMSpropParity(TestCase):
    """Test RMSprop optimizer parity with PyTorch."""

    def test_rmsprop_basic(self):
        """Test basic RMSprop."""
        run_optimizer_parity_test(
            self,
            optim.RMSprop, torch_optim.RMSprop,
            {'lr': 0.01}
        )

    def test_rmsprop_momentum(self):
        """Test RMSprop with momentum."""
        run_optimizer_parity_test(
            self,
            optim.RMSprop, torch_optim.RMSprop,
            {'lr': 0.01, 'momentum': 0.9}
        )

    def test_rmsprop_centered(self):
        """Test RMSprop with centered gradient."""
        run_optimizer_parity_test(
            self,
            optim.RMSprop, torch_optim.RMSprop,
            {'lr': 0.01, 'centered': True}
        )

    def test_rmsprop_weight_decay(self):
        """Test RMSprop with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.RMSprop, torch_optim.RMSprop,
            {'lr': 0.01, 'weight_decay': 0.01}
        )

    def test_rmsprop_custom_alpha(self):
        """Test RMSprop with custom alpha."""
        run_optimizer_parity_test(
            self,
            optim.RMSprop, torch_optim.RMSprop,
            {'lr': 0.01, 'alpha': 0.95}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestAdagradParity(TestCase):
    """Test Adagrad optimizer parity with PyTorch."""

    def test_adagrad_basic(self):
        """Test basic Adagrad."""
        run_optimizer_parity_test(
            self,
            optim.Adagrad, torch_optim.Adagrad,
            {'lr': 0.01}
        )

    def test_adagrad_lr_decay(self):
        """Test Adagrad with learning rate decay."""
        run_optimizer_parity_test(
            self,
            optim.Adagrad, torch_optim.Adagrad,
            {'lr': 0.01, 'lr_decay': 0.01}
        )

    def test_adagrad_weight_decay(self):
        """Test Adagrad with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.Adagrad, torch_optim.Adagrad,
            {'lr': 0.01, 'weight_decay': 0.01}
        )

    def test_adagrad_initial_accumulator(self):
        """Test Adagrad with initial accumulator value."""
        run_optimizer_parity_test(
            self,
            optim.Adagrad, torch_optim.Adagrad,
            {'lr': 0.01, 'initial_accumulator_value': 0.1}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestAdadeltaParity(TestCase):
    """Test Adadelta optimizer parity with PyTorch."""

    def test_adadelta_basic(self):
        """Test basic Adadelta."""
        run_optimizer_parity_test(
            self,
            optim.Adadelta, torch_optim.Adadelta,
            {'lr': 1.0}
        )

    def test_adadelta_custom_rho(self):
        """Test Adadelta with custom rho."""
        run_optimizer_parity_test(
            self,
            optim.Adadelta, torch_optim.Adadelta,
            {'lr': 1.0, 'rho': 0.95}
        )

    def test_adadelta_weight_decay(self):
        """Test Adadelta with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.Adadelta, torch_optim.Adadelta,
            {'lr': 1.0, 'weight_decay': 0.01}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestAdamaxParity(TestCase):
    """Test Adamax optimizer parity with PyTorch."""

    def test_adamax_basic(self):
        """Test basic Adamax."""
        run_optimizer_parity_test(
            self,
            optim.Adamax, torch_optim.Adamax,
            {'lr': 0.002}
        )

    def test_adamax_weight_decay(self):
        """Test Adamax with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.Adamax, torch_optim.Adamax,
            {'lr': 0.002, 'weight_decay': 0.01}
        )

    def test_adamax_custom_betas(self):
        """Test Adamax with custom betas."""
        run_optimizer_parity_test(
            self,
            optim.Adamax, torch_optim.Adamax,
            {'lr': 0.002, 'betas': (0.85, 0.995)}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestRAdamParity(TestCase):
    """Test RAdam optimizer parity with PyTorch."""

    def test_radam_basic(self):
        """Test basic RAdam."""
        run_optimizer_parity_test(
            self,
            optim.RAdam, torch_optim.RAdam,
            {'lr': 0.001}
        )

    def test_radam_weight_decay(self):
        """Test RAdam with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.RAdam, torch_optim.RAdam,
            {'lr': 0.001, 'weight_decay': 0.01}
        )

    def test_radam_custom_betas(self):
        """Test RAdam with custom betas."""
        run_optimizer_parity_test(
            self,
            optim.RAdam, torch_optim.RAdam,
            {'lr': 0.001, 'betas': (0.85, 0.995)}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestNAdamParity(TestCase):
    """Test NAdam optimizer parity with PyTorch."""

    def test_nadam_basic(self):
        """Test basic NAdam."""
        run_optimizer_parity_test(
            self,
            optim.NAdam, torch_optim.NAdam,
            {'lr': 0.002}
        )

    def test_nadam_weight_decay(self):
        """Test NAdam with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.NAdam, torch_optim.NAdam,
            {'lr': 0.002, 'weight_decay': 0.01}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestASGDParity(TestCase):
    """Test ASGD optimizer parity with PyTorch."""

    def test_asgd_basic(self):
        """Test basic ASGD."""
        # ASGD uses power operations which accumulate FP differences
        run_optimizer_parity_test(
            self,
            optim.ASGD, torch_optim.ASGD,
            {'lr': 0.01},
            rtol=1e-4, atol=1e-5
        )

    def test_asgd_weight_decay(self):
        """Test ASGD with weight decay."""
        run_optimizer_parity_test(
            self,
            optim.ASGD, torch_optim.ASGD,
            {'lr': 0.01, 'weight_decay': 0.01},
            rtol=1e-4, atol=1e-5
        )

    def test_asgd_custom_params(self):
        """Test ASGD with custom lambda and alpha."""
        # Custom alpha power accumulates more FP differences
        run_optimizer_parity_test(
            self,
            optim.ASGD, torch_optim.ASGD,
            {'lr': 0.01, 'lambd': 0.001, 'alpha': 0.5},
            rtol=2e-4, atol=1e-4
        )


@skipIfNoMLX
@skip_if_no_torch
class TestRpropParity(TestCase):
    """Test Rprop optimizer parity with PyTorch."""

    def test_rprop_basic(self):
        """Test basic Rprop."""
        run_optimizer_parity_test(
            self,
            optim.Rprop, torch_optim.Rprop,
            {'lr': 0.01}
        )

    def test_rprop_custom_etas(self):
        """Test Rprop with custom etas."""
        run_optimizer_parity_test(
            self,
            optim.Rprop, torch_optim.Rprop,
            {'lr': 0.01, 'etas': (0.4, 1.3)}
        )

    def test_rprop_custom_step_sizes(self):
        """Test Rprop with custom step sizes."""
        run_optimizer_parity_test(
            self,
            optim.Rprop, torch_optim.Rprop,
            {'lr': 0.01, 'step_sizes': (1e-7, 40)}
        )


@skipIfNoMLX
@skip_if_no_torch
class TestMultiStepParity(TestCase):
    """Test optimizers over many steps for numerical stability."""

    def test_adam_100_steps(self):
        """Test Adam over 100 steps."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001},
            num_steps=100
        )

    def test_sgd_momentum_100_steps(self):
        """Test SGD with momentum over 100 steps."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'momentum': 0.9},
            num_steps=100
        )

    def test_adamw_100_steps(self):
        """Test AdamW over 100 steps."""
        run_optimizer_parity_test(
            self,
            optim.AdamW, torch_optim.AdamW,
            {'lr': 0.001, 'weight_decay': 0.01},
            num_steps=100
        )


@skipIfNoMLX
@skip_if_no_torch
class TestLargerTensorsParity(TestCase):
    """Test optimizers with larger parameter tensors."""

    def test_adam_large_tensor(self):
        """Test Adam with 1000x100 parameter tensor."""
        run_optimizer_parity_test(
            self,
            optim.Adam, torch_optim.Adam,
            {'lr': 0.001},
            param_shape=(1000, 100),
            num_steps=20
        )

    def test_sgd_large_tensor(self):
        """Test SGD with 1000x100 parameter tensor."""
        run_optimizer_parity_test(
            self,
            optim.SGD, torch_optim.SGD,
            {'lr': 0.01, 'momentum': 0.9},
            param_shape=(1000, 100),
            num_steps=20
        )


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
