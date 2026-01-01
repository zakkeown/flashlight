"""
Learning Rate Scheduler Parity Tests

Comprehensive tests comparing flashlight LR schedulers against PyTorch implementations.
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
    from flashlight.optim import lr_scheduler as mlx_lr_scheduler
    MLX_COMPAT_AVAILABLE = True
except ImportError:
    MLX_COMPAT_AVAILABLE = False

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
    from torch.optim import lr_scheduler as torch_lr_scheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def skip_if_no_torch(func):
    """Skip test if PyTorch is not available."""
    return unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")(func)


def compare_lr_schedules(test_case, mlx_scheduler, torch_scheduler, num_epochs, rtol=1e-6, atol=1e-8):
    """
    Compare learning rate schedules between MLX and PyTorch.

    Args:
        test_case: unittest.TestCase instance
        mlx_scheduler: MLX scheduler instance
        torch_scheduler: PyTorch scheduler instance
        num_epochs: Number of epochs to step
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    for epoch in range(num_epochs):
        mlx_lrs = [group['lr'] for group in mlx_scheduler.optimizer.param_groups]
        torch_lrs = [group['lr'] for group in torch_scheduler.optimizer.param_groups]

        for i, (mlx_lr, torch_lr) in enumerate(zip(mlx_lrs, torch_lrs)):
            try:
                np.testing.assert_allclose(
                    mlx_lr, torch_lr,
                    rtol=rtol, atol=atol,
                    err_msg=f"LR mismatch at epoch {epoch}, param_group {i}"
                )
            except AssertionError as e:
                test_case.fail(
                    f"LR parity failed at epoch {epoch}:\n"
                    f"MLX LR: {mlx_lr}\n"
                    f"Torch LR: {torch_lr}\n"
                    f"Diff: {abs(mlx_lr - torch_lr)}\n"
                    f"Original error: {e}"
                )

        mlx_scheduler.step()
        torch_scheduler.step()


def create_optimizers(lr=0.1):
    """Create matching MLX and PyTorch optimizers for testing."""
    # MLX setup
    mlx_param = nn.Parameter(flashlight.tensor(np.zeros((3, 3), dtype=np.float32)))
    mlx_opt = optim.SGD([mlx_param], lr=lr)

    # PyTorch setup
    torch_param = torch.zeros(3, 3, requires_grad=True)
    torch_opt = torch_optim.SGD([torch_param], lr=lr)

    return mlx_opt, torch_opt


@skipIfNoMLX
@skip_if_no_torch
class TestStepLRParity(TestCase):
    """Test StepLR scheduler parity with PyTorch."""

    def test_step_lr_basic(self):
        """Test basic StepLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.StepLR(mlx_opt, step_size=10, gamma=0.1)
        torch_sched = torch_lr_scheduler.StepLR(torch_opt, step_size=10, gamma=0.1)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50)

    def test_step_lr_custom_gamma(self):
        """Test StepLR with custom gamma."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.StepLR(mlx_opt, step_size=5, gamma=0.5)
        torch_sched = torch_lr_scheduler.StepLR(torch_opt, step_size=5, gamma=0.5)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=30)


@skipIfNoMLX
@skip_if_no_torch
class TestMultiStepLRParity(TestCase):
    """Test MultiStepLR scheduler parity with PyTorch."""

    def test_multi_step_lr_basic(self):
        """Test basic MultiStepLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.MultiStepLR(mlx_opt, milestones=[10, 20, 30], gamma=0.1)
        torch_sched = torch_lr_scheduler.MultiStepLR(torch_opt, milestones=[10, 20, 30], gamma=0.1)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50)


@skipIfNoMLX
@skip_if_no_torch
class TestExponentialLRParity(TestCase):
    """Test ExponentialLR scheduler parity with PyTorch."""

    def test_exponential_lr_basic(self):
        """Test basic ExponentialLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.ExponentialLR(mlx_opt, gamma=0.9)
        torch_sched = torch_lr_scheduler.ExponentialLR(torch_opt, gamma=0.9)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50)

    def test_exponential_lr_high_decay(self):
        """Test ExponentialLR with high decay rate."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.ExponentialLR(mlx_opt, gamma=0.5)
        torch_sched = torch_lr_scheduler.ExponentialLR(torch_opt, gamma=0.5)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=20)


@skipIfNoMLX
@skip_if_no_torch
class TestCosineAnnealingLRParity(TestCase):
    """Test CosineAnnealingLR scheduler parity with PyTorch."""

    def test_cosine_annealing_lr_basic(self):
        """Test basic CosineAnnealingLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.CosineAnnealingLR(mlx_opt, T_max=50)
        torch_sched = torch_lr_scheduler.CosineAnnealingLR(torch_opt, T_max=50)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50)

    def test_cosine_annealing_lr_with_eta_min(self):
        """Test CosineAnnealingLR with eta_min."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.CosineAnnealingLR(mlx_opt, T_max=30, eta_min=0.01)
        torch_sched = torch_lr_scheduler.CosineAnnealingLR(torch_opt, T_max=30, eta_min=0.01)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=30)


@skipIfNoMLX
@skip_if_no_torch
class TestLinearLRParity(TestCase):
    """Test LinearLR scheduler parity with PyTorch."""

    def test_linear_lr_basic(self):
        """Test basic LinearLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.LinearLR(mlx_opt, start_factor=0.5, total_iters=10)
        torch_sched = torch_lr_scheduler.LinearLR(torch_opt, start_factor=0.5, total_iters=10)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=20)

    def test_linear_lr_custom_factors(self):
        """Test LinearLR with custom start and end factors."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.LinearLR(mlx_opt, start_factor=0.1, end_factor=1.0, total_iters=20)
        torch_sched = torch_lr_scheduler.LinearLR(torch_opt, start_factor=0.1, end_factor=1.0, total_iters=20)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=30)


@skipIfNoMLX
@skip_if_no_torch
class TestConstantLRParity(TestCase):
    """Test ConstantLR scheduler parity with PyTorch."""

    def test_constant_lr_basic(self):
        """Test basic ConstantLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.ConstantLR(mlx_opt, factor=0.5, total_iters=10)
        torch_sched = torch_lr_scheduler.ConstantLR(torch_opt, factor=0.5, total_iters=10)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=20)


@skipIfNoMLX
@skip_if_no_torch
class TestPolynomialLRParity(TestCase):
    """Test PolynomialLR scheduler parity with PyTorch."""

    def test_polynomial_lr_linear(self):
        """Test PolynomialLR with linear decay (power=1)."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.PolynomialLR(mlx_opt, total_iters=20, power=1.0)
        torch_sched = torch_lr_scheduler.PolynomialLR(torch_opt, total_iters=20, power=1.0)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=25)

    def test_polynomial_lr_quadratic(self):
        """Test PolynomialLR with quadratic decay (power=2)."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.PolynomialLR(mlx_opt, total_iters=20, power=2.0)
        torch_sched = torch_lr_scheduler.PolynomialLR(torch_opt, total_iters=20, power=2.0)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=25)


@skipIfNoMLX
@skip_if_no_torch
class TestLambdaLRParity(TestCase):
    """Test LambdaLR scheduler parity with PyTorch."""

    def test_lambda_lr_basic(self):
        """Test basic LambdaLR with simple decay."""
        lr_lambda = lambda epoch: 0.95 ** epoch
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.LambdaLR(mlx_opt, lr_lambda=lr_lambda)
        torch_sched = torch_lr_scheduler.LambdaLR(torch_opt, lr_lambda=lr_lambda)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=30)


@skipIfNoMLX
@skip_if_no_torch
class TestMultiplicativeLRParity(TestCase):
    """Test MultiplicativeLR scheduler parity with PyTorch."""

    def test_multiplicative_lr_basic(self):
        """Test basic MultiplicativeLR."""
        lr_lambda = lambda epoch: 0.95
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.MultiplicativeLR(mlx_opt, lr_lambda=lr_lambda)
        torch_sched = torch_lr_scheduler.MultiplicativeLR(torch_opt, lr_lambda=lr_lambda)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=30)


@skipIfNoMLX
@skip_if_no_torch
class TestCosineAnnealingWarmRestartsParity(TestCase):
    """Test CosineAnnealingWarmRestarts scheduler parity with PyTorch."""

    def test_warm_restarts_basic(self):
        """Test basic CosineAnnealingWarmRestarts."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.CosineAnnealingWarmRestarts(mlx_opt, T_0=10)
        torch_sched = torch_lr_scheduler.CosineAnnealingWarmRestarts(torch_opt, T_0=10)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50)

    def test_warm_restarts_t_mult(self):
        """Test CosineAnnealingWarmRestarts with T_mult."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.CosineAnnealingWarmRestarts(mlx_opt, T_0=10, T_mult=2)
        torch_sched = torch_lr_scheduler.CosineAnnealingWarmRestarts(torch_opt, T_0=10, T_mult=2)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50)


@skipIfNoMLX
@skip_if_no_torch
class TestCyclicLRParity(TestCase):
    """Test CyclicLR scheduler parity with PyTorch."""

    def test_cyclic_lr_triangular(self):
        """Test CyclicLR with triangular mode."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.CyclicLR(
            mlx_opt, base_lr=0.001, max_lr=0.1, step_size_up=10, mode='triangular'
        )
        torch_sched = torch_lr_scheduler.CyclicLR(
            torch_opt, base_lr=0.001, max_lr=0.1, step_size_up=10, mode='triangular'
        )
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=50, rtol=1e-5, atol=1e-7)


@skipIfNoMLX
@skip_if_no_torch
class TestOneCycleLRParity(TestCase):
    """Test OneCycleLR scheduler parity with PyTorch."""

    def test_one_cycle_lr_basic(self):
        """Test basic OneCycleLR."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.OneCycleLR(mlx_opt, max_lr=0.1, total_steps=100)
        torch_sched = torch_lr_scheduler.OneCycleLR(torch_opt, max_lr=0.1, total_steps=100)
        compare_lr_schedules(self, mlx_sched, torch_sched, num_epochs=100, rtol=1e-5, atol=1e-7)


@skipIfNoMLX
@skip_if_no_torch
class TestReduceLROnPlateauParity(TestCase):
    """Test ReduceLROnPlateau scheduler parity with PyTorch."""

    def test_reduce_on_plateau_basic(self):
        """Test ReduceLROnPlateau with decreasing metric."""
        mlx_opt, torch_opt = create_optimizers(lr=0.1)
        mlx_sched = mlx_lr_scheduler.ReduceLROnPlateau(mlx_opt, mode='min', factor=0.1, patience=5)
        torch_sched = torch_lr_scheduler.ReduceLROnPlateau(torch_opt, mode='min', factor=0.1, patience=5)

        # Simulate training with gradually decreasing loss, then plateau
        metrics = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3]
        for metric in metrics:
            mlx_sched.step(metric)
            torch_sched.step(metric)

            mlx_lr = mlx_opt.param_groups[0]['lr']
            torch_lr = torch_opt.param_groups[0]['lr']
            np.testing.assert_allclose(mlx_lr, torch_lr, rtol=1e-6, atol=1e-8)


if __name__ == '__main__':
    from tests.common_utils import run_tests
    run_tests()
