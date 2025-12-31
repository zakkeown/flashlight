"""
Learning Rate Schedulers

Implements various learning rate scheduling strategies.
"""

import math
from typing import List, Optional
from .optimizer import Optimizer


class _LRScheduler:
    """
    Base class for learning rate schedulers.

    Args:
        optimizer: Wrapped optimizer
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # Store initial learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> List[float]:
        """Compute learning rate for each parameter group."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None):
        """Update learning rates."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class StepLR(_LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs.

    Args:
        optimizer: Wrapped optimizer
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     train(...)
        ...     validate(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class ExponentialLR(_LRScheduler):
    """
    Decays the learning rate by gamma every epoch.

    Args:
        optimizer: Wrapped optimizer
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.9)
        >>> for epoch in range(100):
        ...     train(...)
        ...     validate(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(_LRScheduler):
    """
    Set learning rate using cosine annealing schedule.

    The learning rate is annealed from the initial lr to eta_min using a cosine curve.

    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
        >>> for epoch in range(100):
        ...     train(...)
        ...     validate(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
            (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
            (group['lr'] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.

    Args:
        optimizer: Wrapped optimizer
        mode: One of 'min' or 'max'. In 'min' mode, lr will be reduced when
              the metric has stopped decreasing; in 'max' mode it will be reduced
              when the metric has stopped increasing (default: 'min')
        factor: Factor by which the learning rate will be reduced (default: 0.1)
        patience: Number of epochs with no improvement after which learning rate
                  will be reduced (default: 10)
        threshold: Threshold for measuring the new optimum (default: 1e-4)
        threshold_mode: One of 'rel' or 'abs' (default: 'rel')
        cooldown: Number of epochs to wait before resuming normal operation
                  after lr has been reduced (default: 0)
        min_lr: Lower bound on the learning rate (default: 0)
        eps: Minimal decay applied to lr (default: 1e-8)

    Example:
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
        >>> for epoch in range(100):
        ...     train(...)
        ...     val_loss = validate(...)
        ...     scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8
    ):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Reset num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: float):
        """Update learning rate based on metric value."""
        current = float(metrics)
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = float('-inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class LinearLR(_LRScheduler):
    """
    Decays the learning rate linearly.

    Args:
        optimizer: Wrapped optimizer
        start_factor: The number we multiply learning rate in the first epoch (default: 1./3.)
        end_factor: The number we multiply learning rate at the end of linear changing (default: 1.0)
        total_iters: The number of iterations that multiplicative factor reaches to 1 (default: 5)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
        >>> for epoch in range(10):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1
    ):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(f'Starting multiplicative factor should be between 0 and 1, got {start_factor}')
        if end_factor > 1.0 or end_factor <= 0:
            raise ValueError(f'Ending multiplicative factor should be between 0 and 1, got {end_factor}')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]

        if self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [
            base_lr * (self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters)
            for base_lr in self.base_lrs
        ]


# Alias for base class (PyTorch compatibility)
LRScheduler = _LRScheduler


class LambdaLR(_LRScheduler):
    """
    Sets the learning rate using a user-defined lambda function.

    Args:
        optimizer: Wrapped optimizer
        lr_lambda: A function or list of functions that computes a multiplicative
                   factor given an integer epoch
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> lambda1 = lambda epoch: epoch // 30
        >>> scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    """

    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [
            base_lr * lmbda(self.last_epoch)
            for base_lr, lmbda in zip(self.base_lrs, self.lr_lambdas)
        ]


class MultiplicativeLR(_LRScheduler):
    """
    Multiply the learning rate by a factor given by a lambda function.

    Args:
        optimizer: Wrapped optimizer
        lr_lambda: A function or list of functions that computes a multiplicative
                   factor given an integer epoch
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> lambda1 = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda1)
    """

    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        return [
            group['lr'] * lmbda(self.last_epoch)
            for group, lmbda in zip(self.optimizer.param_groups, self.lr_lambdas)
        ]


class MultiStepLR(_LRScheduler):
    """
    Decays the learning rate by gamma once the epoch reaches one of the milestones.

    Args:
        optimizer: Wrapped optimizer
        milestones: List of epoch indices. Must be increasing.
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    """

    def __init__(self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class ConstantLR(_LRScheduler):
    """
    Decays the learning rate by a constant factor until epoch reaches total_iters.

    Args:
        optimizer: Wrapped optimizer
        factor: The number we multiply learning rate until total_iters (default: 1./3.)
        total_iters: The number of iterations that the constant factor is applied (default: 5)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=4)
    """

    def __init__(self, optimizer: Optimizer, factor: float = 1.0 / 3, total_iters: int = 5, last_epoch: int = -1):
        if factor > 1.0 or factor < 0:
            raise ValueError(f'Constant multiplicative factor should be between 0 and 1, got {factor}')
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        if self.last_epoch >= self.total_iters:
            return self.base_lrs
        return [group['lr'] for group in self.optimizer.param_groups]


class PolynomialLR(_LRScheduler):
    """
    Decays the learning rate using a polynomial function.

    Args:
        optimizer: Wrapped optimizer
        total_iters: The number of steps that the scheduler decays the LR (default: 5)
        power: The power of the polynomial (default: 1.0)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = PolynomialLR(optimizer, total_iters=5, power=1.0)
    """

    def __init__(self, optimizer: Optimizer, total_iters: int = 5, power: float = 1.0, last_epoch: int = -1):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        decay = (1 - self.last_epoch / self.total_iters) / (1 - (self.last_epoch - 1) / self.total_iters)
        return [group['lr'] * (decay ** self.power) for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Set learning rate using cosine annealing with warm restarts.

    Args:
        optimizer: Wrapped optimizer
        T_0: Number of iterations for the first restart
        T_mult: A factor increases T_i after each restart (default: 1)
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    """

    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, got {T_mult}")
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class SequentialLR(_LRScheduler):
    """
    Chains a list of schedulers and switches between them at specified milestones.

    Args:
        optimizer: Wrapped optimizer
        schedulers: List of schedulers to chain
        milestones: List of epoch indices when to switch to next scheduler
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[2])
    """

    def __init__(self, optimizer: Optimizer, schedulers: List, milestones: List[int], last_epoch: int = -1):
        if len(schedulers) < 1:
            raise ValueError("SequentialLR requires at least one scheduler")
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(f"Expected {len(schedulers) - 1} milestones, got {len(milestones)}")

        self._schedulers = schedulers
        self._milestones = milestones
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # Don't call super().__init__ since schedulers already initialized

    def get_lr(self) -> List[float]:
        # Find which scheduler is active
        idx = 0
        for i, m in enumerate(self._milestones):
            if self.last_epoch >= m:
                idx = i + 1
        return self._schedulers[idx].get_lr()

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        # Find active scheduler and step it
        idx = 0
        for i, m in enumerate(self._milestones):
            if self.last_epoch >= m:
                idx = i + 1

        self._schedulers[idx].step()


class ChainedScheduler:
    """
    Chains a list of learning rate schedulers.

    Takes a list of schedulers and calls step() on all of them.

    Args:
        schedulers: List of schedulers to chain
        optimizer: Wrapped optimizer (optional, for compatibility)

    Example:
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
    """

    def __init__(self, schedulers: List, optimizer: Optional[Optimizer] = None):
        if len(schedulers) < 1:
            raise ValueError("ChainedScheduler requires at least one scheduler")
        self._schedulers = schedulers
        # optimizer param is for PyTorch compatibility, but not used
        # since each scheduler already has its own optimizer reference

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()


class CyclicLR(_LRScheduler):
    """
    Sets the learning rate cyclically between min and max values.

    Args:
        optimizer: Wrapped optimizer
        base_lr: Initial learning rate (lower boundary in cycle)
        max_lr: Upper learning rate boundary in cycle
        step_size_up: Number of iterations in the increasing half of a cycle (default: 2000)
        step_size_down: Number of iterations in the decreasing half of a cycle.
                        If None, it's set to step_size_up (default: None)
        mode: One of 'triangular', 'triangular2', or 'exp_range' (default: 'triangular')
        gamma: Constant for 'exp_range' mode (default: 1.0)
        scale_fn: Custom scaling function (default: None)
        scale_mode: 'cycle' or 'iterations' (default: 'cycle')
        cycle_momentum: If True, momentum is cycled inversely to learning rate (default: True)
        base_momentum: Lower momentum boundary in cycle (default: 0.8)
        max_momentum: Upper momentum boundary in cycle (default: 0.9)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn=None,
        scale_mode: str = 'cycle',
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum

        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = 'cycle'
            elif mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle

        if x <= self.step_size_up / self.total_size:
            scale = x * self.total_size / self.step_size_up
        else:
            scale = (self.total_size - x * self.total_size) / self.step_size_down

        if self.scale_mode == 'cycle':
            scale_factor = self.scale_fn(cycle)
        else:
            scale_factor = self.scale_fn(self.last_epoch)

        return [
            self.base_lr + (self.max_lr - self.base_lr) * scale * scale_factor
            for _ in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """
    Sets the learning rate according to the 1cycle learning rate policy.

    Args:
        optimizer: Wrapped optimizer
        max_lr: Upper learning rate boundary in the cycle
        total_steps: Total number of steps in the cycle. If None, inferred from epochs and steps_per_epoch
        epochs: Number of epochs to train (default: None)
        steps_per_epoch: Number of steps per epoch (default: None)
        pct_start: Percentage of cycle spent increasing LR (default: 0.3)
        anneal_strategy: 'cos' or 'linear' (default: 'cos')
        cycle_momentum: If True, momentum is cycled inversely (default: True)
        base_momentum: Lower momentum boundary (default: 0.85)
        max_momentum: Upper momentum boundary (default: 0.95)
        div_factor: Initial lr = max_lr / div_factor (default: 25)
        final_div_factor: Final lr = initial_lr / final_div_factor (default: 1e4)
        three_phase: If True, use three-phase schedule (default: False)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1
    ):
        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("Must specify total_steps or (epochs and steps_per_epoch)")
            total_steps = epochs * steps_per_epoch

        self.total_steps = total_steps
        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.three_phase = three_phase

        # Build schedule phases (PyTorch-compatible)
        if three_phase:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * total_steps) - 1,
                    "start_lr": self.initial_lr,
                    "end_lr": self.max_lr,
                },
                {
                    "end_step": float(2 * pct_start * total_steps) - 2,
                    "start_lr": self.max_lr,
                    "end_lr": self.initial_lr,
                },
                {
                    "end_step": total_steps - 1,
                    "start_lr": self.initial_lr,
                    "end_lr": self.min_lr,
                },
            ]
        else:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * total_steps) - 1,
                    "start_lr": self.initial_lr,
                    "end_lr": self.max_lr,
                },
                {
                    "end_step": total_steps - 1,
                    "start_lr": self.max_lr,
                    "end_lr": self.min_lr,
                },
            ]

        super().__init__(optimizer, last_epoch)

    def _anneal_func(self, start, end, pct):
        """Compute annealed value between start and end at percentage pct."""
        if self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * pct)) / 2
        else:  # linear
            return start + (end - start) * pct

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step > self.total_steps:
            raise ValueError(
                f"Tried to step {step} times. The specified number of total steps is {self.total_steps}"
            )

        # Find which phase we're in and compute LR
        start_step = 0.0
        for i, phase in enumerate(self._schedule_phases):
            end_step = phase["end_step"]
            if step <= end_step or i == len(self._schedule_phases) - 1:
                pct = (step - start_step) / (end_step - start_step) if end_step > start_step else 0.0
                lr = self._anneal_func(phase["start_lr"], phase["end_lr"], pct)
                break
            start_step = phase["end_step"]

        return [lr for _ in self.base_lrs]


__all__ = [
    # Base class
    'LRScheduler',
    # Standard schedulers
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'ReduceLROnPlateau',
    'LinearLR',
    'ConstantLR',
    'PolynomialLR',
    # Lambda-based schedulers
    'LambdaLR',
    'MultiplicativeLR',
    # Composite schedulers
    'SequentialLR',
    'ChainedScheduler',
    # Cyclic schedulers
    'CyclicLR',
    'OneCycleLR',
]
