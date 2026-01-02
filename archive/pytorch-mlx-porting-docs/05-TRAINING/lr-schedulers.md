# Learning Rate Schedulers

## Overview

Learning rate scheduling is a critical technique for improving optimization in deep learning. By adjusting the learning rate during training, schedulers can help models:
- Converge faster in early training (high LR)
- Fine-tune to better minima in late training (low LR)
- Escape local minima and saddle points
- Adapt to changing gradient landscapes

PyTorch provides **16+ learning rate schedulers** covering step-based decay, exponential decay, cyclic policies, warmup strategies, composite schedulers, and adaptive reduction. This document provides comprehensive coverage of **15 schedulers** essential for MLX porting.

**File Location**: [torch/optim/lr_scheduler.py](reference/pytorch/torch/optim/lr_scheduler.py) (~3,500 lines)

---

## Base Class: LRScheduler

All schedulers inherit from `torch.optim.lr_scheduler.LRScheduler` which provides:

```python
class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer  # Reference to optimizer
        self.last_epoch = last_epoch  # Current epoch/step
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self._step_count = 0
        self._initial_step()  # Perform initial LR update

    def step(self, epoch=None):
        """Update learning rates - call AFTER optimizer.step()"""
        self._step_count += 1
        self.last_epoch += 1
        values = self.get_lr()  # Compute new LRs
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr  # Update optimizer LRs

    def get_lr(self):
        """Compute new learning rates (implemented by subclasses)"""
        raise NotImplementedError

    def get_last_lr(self):
        """Get most recent learning rates"""
        return self._last_lr
```

**Key Design Patterns**:
1. **Stateful**: Maintains `last_epoch` counter
2. **Post-Optimizer**: Call `scheduler.step()` **after** `optimizer.step()`
3. **Per-Parameter-Group**: Supports different LRs for different parameter groups
4. **Serializable**: `state_dict()` and `load_state_dict()` for checkpointing

---

## 1. StepLR (Step Decay)

### Description

Reduces learning rate by a factor every N epochs. The most commonly used scheduler for computer vision.

**Formula**:
```
η_t = η_0 * γ^(⌊t / step_size⌋)
```

Where:
- `η_0`: Initial learning rate
- `γ`: Decay factor (typically 0.1)
- `t`: Current epoch
- `step_size`: Number of epochs between decays

### API

```python
torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size,          # Decay every N epochs
    gamma=0.1,          # Multiplicative factor
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:594-678](reference/pytorch/torch/optim/lr_scheduler.py#L594-L678):

```python
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """Closed-form formula for jumping to arbitrary epoch"""
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]
```

### Usage Example

**ResNet Training** (ImageNet):
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(90):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule:
# Epochs 0-29:  lr = 0.1
# Epochs 30-59: lr = 0.01
# Epochs 60-89: lr = 0.001
```

### MLX Implementation

```python
class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.gamma

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
```

---

## 2. MultiStepLR (Multi-Step Decay)

### Description

Reduces learning rate at specific milestone epochs. More flexible than StepLR for non-uniform schedules.

**Formula**:
```
η_t = η_0 * γ^k  where k = number of milestones passed
```

### API

```python
torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones,         # List of epoch indices [30, 60, 80]
    gamma=0.1,
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:681-772](reference/pytorch/torch/optim/lr_scheduler.py#L681-L772):

```python
class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)  # Support duplicate milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        # If milestone appears n times, scale by gamma^n
        return [
            group['lr'] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
```

### Usage Example

**ResNet Training** (standard recipe):
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 80],
    gamma=0.1
)

for epoch in range(90):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule:
# Epochs 0-29:  lr = 0.1
# Epochs 30-59: lr = 0.01
# Epochs 60-79: lr = 0.001
# Epochs 80-89: lr = 0.0001
```

---

## 3. ExponentialLR (Exponential Decay)

### Description

Decays learning rate by a constant factor every epoch. Smooth continuous decay.

**Formula**:
```
η_t = η_0 * γ^t
```

### API

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,              # Decay factor (0 < gamma < 1)
    last_epoch=-1
)
```

### Implementation

```python
class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._is_initial:  # Don't decay on initialization
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]
```

### Usage Example

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule (smooth exponential decay):
# Epoch 0:   lr = 0.01
# Epoch 10:  lr = 0.00599
# Epoch 20:  lr = 0.00358
# Epoch 50:  lr = 0.00077
# Epoch 100: lr = 0.00006
```

---

## 4. CosineAnnealingLR (Cosine Annealing)

### Description

Anneals learning rate using cosine function. Provides smooth decay with potential for "restarts" (see CosineAnnealingWarmRestarts).

**Formula**:
```
η_t = η_min + (η_max - η_min) * (1 + cos(πt / T_max)) / 2
```

Where:
- `η_max`: Initial (maximum) learning rate
- `η_min`: Minimum learning rate (default: 0)
- `T_max`: Maximum number of iterations
- `t`: Current epoch

### API

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max,              # Max epochs (period of cosine)
    eta_min=0,          # Minimum LR
    last_epoch=-1
)
```

### Implementation

```python
class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
            (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
            (group['lr'] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
```

### Usage Example

**Transformer Training**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-5
)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule (smooth cosine curve):
# Epoch 0:   lr = 0.001
# Epoch 25:  lr = 0.000755
# Epoch 50:  lr = 0.000505
# Epoch 75:  lr = 0.000255
# Epoch 100: lr = 0.00001 (eta_min)
```

**Benefit**: Smooth decay with gradual slowdown near minimum, avoiding abrupt changes.

---

## 5. ReduceLROnPlateau (Adaptive Reduction)

### Description

Reduces learning rate when a metric (e.g., validation loss) stops improving. **Metric-based** scheduler (not epoch-based).

**Algorithm**:
```
if metric hasn't improved for `patience` epochs:
    η_new = η_current * factor
```

### API

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # 'min' for loss, 'max' for accuracy
    factor=0.1,         # Reduction factor
    patience=10,        # Epochs to wait before reducing
    threshold=1e-4,     # Minimum change to qualify as improvement
    threshold_mode='rel',  # 'rel' or 'abs'
    cooldown=0,         # Epochs to wait after reduction
    min_lr=0,           # Minimum LR
    eps=1e-8            # Minimum LR decay
)
```

### Implementation

```python
class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.eps = eps

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.mode_worse = inf if mode == 'min' else -inf

    def step(self, metrics):
        """Call with validation metric (loss or accuracy)"""
        current = float(metrics)

        if self.best is None:
            self.best = current
        elif self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

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

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
```

### Usage Example

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

for epoch in range(100):
    train_loss = train_epoch(model, optimizer)
    val_loss = validate(model)
    scheduler.step(val_loss)  # Pass validation metric

# LR schedule (adaptive):
# Epoch 0-10:  lr = 0.001 (validation loss decreasing)
# Epoch 11-15: lr = 0.001 (loss plateaus)
# Epoch 16:    lr = 0.0005 (reduced after 5 epochs of no improvement)
# ...
```

**When to Use**:
- When you don't know optimal decay schedule in advance
- For RNNs, GANs, and other unstable training regimes
- When validation performance is the primary metric

---

## 6. LinearLR (Linear Warmup)

### Description

Linearly increases learning rate from a small value to the base LR. Used for **warmup** at the start of training.

**Formula**:
```
factor_t = start_factor + (end_factor - start_factor) * min(t, T) / T
η_t = η_0 * factor_t
```

### API

```python
torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0/3,     # Initial factor (lr_0 * start_factor)
    end_factor=1.0,         # Final factor (lr_0 * end_factor)
    total_iters=5,          # Warmup duration
    last_epoch=-1
)
```

### Usage Example

**Warmup + Cosine Decay** (common Transformer pattern):
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Warmup for 10 epochs
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=10
)

# Cosine decay for remaining epochs
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=90,
    eta_min=1e-5
)

# Combine schedulers
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[10]
)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule:
# Epochs 0-9:   lr increases linearly from 0.0001 to 0.001 (warmup)
# Epochs 10-99: lr decreases via cosine annealing from 0.001 to 0.00001
```

---

## 7. LambdaLR (Custom Schedule)

### Description

Applies a custom function to compute learning rate multipliers. Maximum flexibility for arbitrary schedules.

**Formula**:
```
η_t = η_0 * λ(t)
```

Where `λ(t)` is a user-defined function.

### API

```python
torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda,          # Function or list of functions
    last_epoch=-1
)
```

### Usage Example

**Custom Warmup + Polynomial Decay**:
```python
def lr_lambda(epoch):
    warmup_epochs = 5
    total_epochs = 100

    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Polynomial decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return (1 - progress) ** 2

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()
```

**Per-Parameter-Group Example**:
```python
# Different schedules for different parameter groups
lambda1 = lambda epoch: epoch / 10 if epoch < 10 else 1.0
lambda2 = lambda epoch: 0.95 ** epoch

optimizer = torch.optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.head.parameters(), 'lr': 0.01}
], lr=0.001)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=[lambda1, lambda2]
)
```

---

## 8. OneCycleLR (Super-Convergence)

### Description

Implements the "1cycle" policy from Smith et al. (2019). Increases LR to a max value then decreases it, often achieving faster convergence.

**Phases**:
1. **Warmup**: Increase LR from `initial_lr` to `max_lr`
2. **Annealing**: Decrease LR from `max_lr` to `final_lr`
3. **(Optional) Final Anneal**: Further decrease to very small LR

### API

```python
torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr,             # Maximum LR (peak)
    total_steps=None,   # Total training steps (or epochs if not specified)
    epochs=None,
    steps_per_epoch=None,
    pct_start=0.3,      # Percentage of cycle spent increasing LR
    anneal_strategy='cos',  # 'cos' or 'linear'
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,    # initial_lr = max_lr / div_factor
    final_div_factor=1e4,  # final_lr = initial_lr / final_div_factor
    three_phase=False,
    last_epoch=-1
)
```

### Usage Example

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.5,
    epochs=10,
    steps_per_epoch=len(train_loader)
)

for epoch in range(10):
    for batch in train_loader:
        train_step(batch, model, optimizer)
        scheduler.step()  # Call after each batch!

# LR schedule (per-batch):
# Batches 0-30%:     lr increases from 0.02 to 0.5
# Batches 30%-100%:  lr decreases from 0.5 to 0.000002
```

**Benefits** (from paper):
- Faster convergence (fewer epochs needed)
- Better regularization (cyclical momentum)
- Less sensitive to hyperparameters

---

## 9. PolynomialLR (Polynomial Decay)

### Description

Decays learning rate using a polynomial function over a specified number of iterations. Common in NLP training, especially for BERT-style fine-tuning.

**Formula**:
```
η_t = η_0 * (1 - t / total_iters)^power
```

Where:
- `η_0`: Initial learning rate
- `t`: Current epoch
- `total_iters`: Total number of decay steps
- `power`: Polynomial power (1.0 = linear decay)

### API

```python
torch.optim.lr_scheduler.PolynomialLR(
    optimizer,
    total_iters=5,      # Number of steps for decay
    power=1.0,          # Polynomial power
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:1238-1330](reference/pytorch/torch/optim/lr_scheduler.py#L1238-L1330):

```python
class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._is_initial or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            for base_lr in self.base_lrs
        ]
```

### Usage Example

**BERT Fine-tuning** (with warmup):
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Linear warmup
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=1000
)

# Polynomial decay (power=1.0 = linear decay)
decay = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=10000, power=1.0
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, decay],
    milestones=[1000]
)

for step in range(11000):
    train_step(model, optimizer)
    scheduler.step()

# LR schedule:
# Steps 0-999:    lr linearly increases from 2e-6 to 2e-5 (warmup)
# Steps 1000-10999: lr linearly decreases from 2e-5 to 0 (polynomial decay)
```

**Quadratic Decay** (power=2.0):
```python
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=100, power=2.0
)
# LR decays faster at the beginning, slower at the end
```

### MLX Implementation

```python
class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1.0):
        super().__init__(optimizer)
        self.total_iters = total_iters
        self.power = power

    def get_lr(self):
        if self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        factor = (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]
```

---

## 10. CosineAnnealingWarmRestarts (SGDR)

### Description

Implements Stochastic Gradient Descent with Warm Restarts (SGDR). Uses cosine annealing with periodic restarts, where the period can optionally increase after each restart.

**Formula**:
```
η_t = η_min + (η_max - η_min) * (1 + cos(π * T_cur / T_i)) / 2
```

Where:
- `T_cur`: Epochs since last restart
- `T_i`: Epochs between restarts (can grow by factor T_mult)
- When `T_cur = T_i`: restart occurs, `η_t = η_max`

### API

```python
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0,                # Epochs until first restart
    T_mult=1,           # Factor to increase T_i after each restart
    eta_min=0,          # Minimum learning rate
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:2101-2201](reference/pytorch/torch/optim/lr_scheduler.py#L2101-L2201):

```python
class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")

        self.T_0 = T_0
        self.T_i = T_0          # Current restart period
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch  # Epochs since last restart
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Handle restart logic"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult  # Increase period
        else:
            # Handle arbitrary epoch jumps
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        # Update learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
```

### Usage Example

**Fixed Restart Period**:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # Restart every 10 epochs
    eta_min=1e-5
)

for epoch in range(50):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule (with restarts at epochs 10, 20, 30, 40):
# Epoch 0:  lr = 0.1 (start)
# Epoch 5:  lr ≈ 0.05 (mid-cycle)
# Epoch 10: lr = 0.1 (restart!)
# Epoch 15: lr ≈ 0.05 (mid-cycle)
# Epoch 20: lr = 0.1 (restart!)
# ...
```

**Increasing Restart Periods** (T_mult=2):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # First cycle: 10 epochs
    T_mult=2,      # Each cycle doubles: 10, 20, 40, 80...
    eta_min=1e-5
)

# LR schedule (restarts at epochs 10, 30, 70, 150...):
# Epochs 0-9:   first cycle (10 epochs)
# Epochs 10-29: second cycle (20 epochs)
# Epochs 30-69: third cycle (40 epochs)
# ...
```

**Benefits** (from SGDR paper):
- Escape local minima with periodic LR increases
- Explore multiple basins of attraction
- Often achieves better final performance than monotonic decay

### MLX Implementation

```python
import math

class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self):
        self.last_epoch += 1
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
```

---

## 11. CyclicLR (Cyclical Learning Rates)

### Description

Cycles learning rate between two boundaries with constant frequency. Based on the "Cyclical Learning Rates for Training Neural Networks" paper. Useful for LR range finding and can achieve faster convergence.

**Key Policies**:
- **triangular**: Basic triangular cycle, no amplitude scaling
- **triangular2**: Halves amplitude each cycle
- **exp_range**: Exponential amplitude decay

### API

```python
torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr,                    # Lower LR boundary
    max_lr,                     # Upper LR boundary
    step_size_up=2000,          # Steps in increasing half
    step_size_down=None,        # Steps in decreasing half (default: step_size_up)
    mode='triangular',          # 'triangular', 'triangular2', 'exp_range'
    gamma=1.0,                  # Decay factor for exp_range
    scale_fn=None,              # Custom scaling function
    scale_mode='cycle',         # 'cycle' or 'iterations'
    cycle_momentum=True,        # Cycle momentum inversely to LR
    base_momentum=0.8,
    max_momentum=0.9,
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:1784-2098](reference/pytorch/torch/optim/lr_scheduler.py#L1784-L2098):

```python
class CyclicLR(LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000,
                 step_size_down=None, mode='triangular', gamma=1.0,
                 scale_fn=None, scale_mode='cycle', cycle_momentum=True,
                 base_momentum=0.8, max_momentum=0.9, last_epoch=-1):
        # Store parameters
        self.base_lrs = _format_param('base_lr', optimizer, base_lr)
        self.max_lrs = _format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        # Initialize scale function based on mode
        if scale_fn is None:
            if mode == 'triangular':
                self.scale_fn = lambda x: 1.0
            elif mode == 'triangular2':
                self.scale_fn = lambda x: 1.0 / (2.0 ** (x - 1))
            elif mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x

        # Momentum cycling (inversely to LR)
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            self.base_momentums = [base_momentum] * len(optimizer.param_groups)
            self.max_momentums = [max_momentum] * len(optimizer.param_groups)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle

        if x <= self.step_ratio:
            # Increasing phase
            scale_factor = x / self.step_ratio
        else:
            # Decreasing phase
            scale_factor = (x - 1) / (self.step_ratio - 1)

        # Apply scaling function
        if self.scale_mode == 'cycle':
            scale = self.scale_fn(cycle)
        else:
            scale = self.scale_fn(self.last_epoch)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor * scale
            lrs.append(base_lr + base_height)
        return lrs
```

### Usage Example

**Basic Triangular Cycle** (per-batch):
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,      # Lower bound
    max_lr=0.1,         # Upper bound
    step_size_up=2000,  # Half cycle = 2000 batches
    mode='triangular'
)

for epoch in range(10):
    for batch in train_loader:
        train_step(batch, model, optimizer)
        scheduler.step()  # Call per batch!

# LR oscillates: 0.001 → 0.1 → 0.001 → 0.1 ...
```

**LR Range Finding**:
```python
# Use triangular2 to find good LR range
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-7,
    max_lr=1.0,
    step_size_up=len(train_loader),  # 1 epoch up
    mode='triangular2'               # Amplitude halves each cycle
)

# Monitor loss vs LR to find optimal range
```

**Exponential Range**:
```python
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.1,
    step_size_up=2000,
    mode='exp_range',
    gamma=0.99994  # Exponential decay per iteration
)
```

### MLX Implementation

```python
import math

class CyclicLR(LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000,
                 step_size_down=None, mode='triangular', gamma=1.0):
        super().__init__(optimizer)
        self.base_lrs = [base_lr] * len(optimizer.param_groups)
        self.max_lrs = [max_lr] * len(optimizer.param_groups)

        step_size_down = step_size_down or step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size
        self.mode = mode
        self.gamma = gamma

    def _scale_fn(self, x):
        if self.mode == 'triangular':
            return 1.0
        elif self.mode == 'triangular2':
            return 1.0 / (2.0 ** (x - 1))
        elif self.mode == 'exp_range':
            return self.gamma ** x
        return 1.0

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle

        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        scale = self._scale_fn(cycle)
        return [
            base_lr + (max_lr - base_lr) * scale_factor * scale
            for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
        ]
```

---

## 12. ConstantLR (Constant Factor)

### Description

Multiplies learning rate by a constant factor until a specified milestone, then restores original LR. Commonly used as a warmup component before chaining with other schedulers.

**Formula**:
```
η_t = η_0 * factor    if t < total_iters
η_t = η_0             if t >= total_iters
```

### API

```python
torch.optim.lr_scheduler.ConstantLR(
    optimizer,
    factor=1.0/3,       # Multiplicative factor (0 < factor <= 1)
    total_iters=5,      # Number of steps with reduced LR
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:775-870](reference/pytorch/torch/optim/lr_scheduler.py#L775-L870):

```python
class ConstantLR(LRScheduler):
    def __init__(self, optimizer, factor=1.0/3, total_iters=5, last_epoch=-1):
        if factor > 1.0 or factor < 0:
            raise ValueError("Factor expected to be between 0 and 1.")
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch != self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        # Restore original LR at milestone
        return [
            group['lr'] * (1.0 / self.factor)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        if self.last_epoch >= self.total_iters:
            return self.base_lrs
        return [base_lr * self.factor for base_lr in self.base_lrs]
```

### Usage Example

**Simple Warmup**:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ConstantLR(
    optimizer,
    factor=0.5,         # Start at 0.05 (0.1 * 0.5)
    total_iters=10      # After 10 epochs, restore to 0.1
)

for epoch in range(50):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule:
# Epochs 0-9:  lr = 0.05 (warmup at reduced LR)
# Epochs 10+: lr = 0.1 (full LR)
```

**Combined with ExponentialLR**:
```python
warmup = torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=0.1, total_iters=20
)
decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, decay],
    milestones=[20]
)

# LR schedule:
# Epochs 0-19:  lr = base_lr * 0.1 (constant warmup)
# Epochs 20+:   lr decays exponentially by 0.9 each epoch
```

### MLX Implementation

```python
class ConstantLR(LRScheduler):
    def __init__(self, optimizer, factor=1.0/3, total_iters=5):
        super().__init__(optimizer)
        self.factor = factor
        self.total_iters = total_iters
        # Apply factor immediately
        for group in optimizer.param_groups:
            group['lr'] *= factor

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return self.base_lrs
        return [base_lr * self.factor for base_lr in self.base_lrs]

    def step(self):
        self.last_epoch += 1
        if self.last_epoch == self.total_iters:
            # Restore original LR
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr
```

---

## 13. SequentialLR (Sequential Schedulers)

### Description

Chains multiple schedulers to be called sequentially at specified milestone epochs. Each scheduler handles a portion of the training. Essential for implementing warmup + decay patterns.

### API

```python
torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers,         # List of scheduler instances
    milestones,         # List of epoch indices (len = len(schedulers) - 1)
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:1084-1235](reference/pytorch/torch/optim/lr_scheduler.py#L1084-L1235):

```python
class SequentialLR(LRScheduler):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        # Validate schedulers
        for scheduler in schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError("SequentialLR does not support ReduceLROnPlateau")
            if optimizer != scheduler.optimizer:
                raise ValueError("All schedulers must use the same optimizer")

        if len(milestones) != len(schedulers) - 1:
            raise ValueError("Number of milestones must be len(schedulers) - 1")

        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        # Reset LRs and initialize first scheduler
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr']
        self._schedulers[0]._initial_step()
        self._last_lr = schedulers[0].get_last_lr()

    def step(self):
        self.last_epoch += 1

        # Find which scheduler is active
        idx = bisect_right(self._milestones, self.last_epoch)

        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            # At a milestone: switch schedulers
            self._schedulers[idx]._initial_step()

        self._schedulers[idx].step()
        self._last_lr = self._schedulers[idx].get_last_lr()

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            '_schedulers': [s.state_dict() for s in self._schedulers]
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        for scheduler, sd in zip(self._schedulers, state_dict['_schedulers']):
            scheduler.load_state_dict(sd)
```

### Usage Example

**Standard Warmup + Cosine Decay**:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Phase 1: Linear warmup (epochs 0-9)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,   # Start at 1e-4
    total_iters=10
)

# Phase 2: Cosine decay (epochs 10-99)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=90,
    eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[10]  # Switch at epoch 10
)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule:
# Epochs 0-9:   lr increases 1e-4 → 1e-3 (linear warmup)
# Epochs 10-99: lr decreases 1e-3 → 1e-6 (cosine annealing)
```

**Three-Phase Training**:
```python
# Phase 1: Constant warmup
warmup = torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=0.1, total_iters=5
)

# Phase 2: Cosine annealing
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=45, eta_min=1e-4
)

# Phase 3: Constant fine-tuning
finetune = torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=0.001, total_iters=50
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine, finetune],
    milestones=[5, 50]  # Switch at epochs 5 and 50
)
```

### MLX Implementation

```python
from bisect import bisect_right

class SequentialLR(LRScheduler):
    def __init__(self, optimizer, schedulers, milestones):
        self.optimizer = optimizer
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = 0
        self._last_lr = schedulers[0].get_last_lr()

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()
        self._last_lr = self._schedulers[idx].get_last_lr()

    def get_last_lr(self):
        return self._last_lr
```

---

## 14. ChainedScheduler (Simultaneous Schedulers)

### Description

Applies multiple schedulers simultaneously by calling all their `step()` methods in sequence. Each scheduler's effect compounds on top of the previous. Different from SequentialLR which switches between schedulers.

### API

```python
torch.optim.lr_scheduler.ChainedScheduler(
    schedulers,         # Sequence of schedulers
    optimizer=None      # Optional (defaults to first scheduler's optimizer)
)
```

### Implementation

From [lr_scheduler.py:1479-1580](reference/pytorch/torch/optim/lr_scheduler.py#L1479-L1580):

```python
class ChainedScheduler(LRScheduler):
    def __init__(self, schedulers, optimizer=None):
        if len(schedulers) < 1:
            raise ValueError("ChainedScheduler expects at least one scheduler")

        optimizer = optimizer or schedulers[0].optimizer
        for scheduler in schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError("ChainedScheduler does not support ReduceLROnPlateau")
            if optimizer != scheduler.optimizer:
                raise ValueError("All schedulers must use the same optimizer")

        self._schedulers = schedulers
        self.optimizer = optimizer
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Call step() on all schedulers"""
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            '_schedulers': [s.state_dict() for s in self._schedulers]
        }

    def load_state_dict(self, state_dict):
        for scheduler, sd in zip(self._schedulers, state_dict['_schedulers']):
            scheduler.load_state_dict(sd)
```

### Usage Example

**Combining Constant and Exponential**:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Scheduler 1: Constant factor for warmup
constant = torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=0.1, total_iters=20
)

# Scheduler 2: Exponential decay
exponential = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.9
)

# Both schedulers apply their effects simultaneously
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [constant, exponential], optimizer=optimizer
)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule (both effects compound):
# Epoch 0:  lr = 0.1 * 0.1 * 0.9^0 = 0.01
# Epoch 1:  lr = 0.1 * 0.1 * 0.9^1 = 0.009
# ...
# Epoch 19: lr = 0.1 * 0.1 * 0.9^19 ≈ 0.00135
# Epoch 20: lr = 0.1 * 1.0 * 0.9^20 ≈ 0.0122  # Constant factor removed
# Epoch 21: lr = 0.1 * 1.0 * 0.9^21 ≈ 0.0109
```

**Key Difference from SequentialLR**:
- **SequentialLR**: Switches between schedulers at milestones
- **ChainedScheduler**: All schedulers active simultaneously, effects compound

### MLX Implementation

```python
class ChainedScheduler(LRScheduler):
    def __init__(self, schedulers, optimizer=None):
        self._schedulers = schedulers
        self.optimizer = optimizer or schedulers[0].optimizer
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr
```

---

## 15. MultiplicativeLR (Multiplicative Factor)

### Description

Multiplies learning rate by a factor returned from a user-defined function each epoch. Similar to LambdaLR but uses multiplicative updates rather than computing absolute LR.

**Formula**:
```
η_t = η_{t-1} * λ(t)
```

Where `λ(t)` is a user-defined function returning a multiplicative factor.

### API

```python
torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer,
    lr_lambda,          # Function(epoch) -> factor, or list of functions
    last_epoch=-1
)
```

### Implementation

From [lr_scheduler.py:470-590](reference/pytorch/torch/optim/lr_scheduler.py#L470-L590):

```python
class MultiplicativeLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas")
            self.lr_lambdas = list(lr_lambda)

        for fn in self.lr_lambdas:
            if not callable(fn):
                raise TypeError("lr_lambda should be a function")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._is_initial:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [
            group['lr'] * lr_lambda(self.last_epoch)
            for group, lr_lambda in zip(self.optimizer.param_groups, self.lr_lambdas)
        ]
```

### Usage Example

**Constant Multiplicative Decay**:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Multiply by 0.95 every epoch
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer,
    lr_lambda=lambda epoch: 0.95
)

for epoch in range(100):
    train_epoch(model, optimizer)
    scheduler.step()

# LR schedule (geometric decay):
# Epoch 0:  lr = 0.1
# Epoch 1:  lr = 0.095
# Epoch 10: lr = 0.0599
# Epoch 50: lr = 0.00769
```

**Conditional Decay**:
```python
def lr_lambda(epoch):
    """Decay only after warmup period"""
    if epoch < 10:
        return 1.0  # No change during warmup
    else:
        return 0.95  # 5% decay per epoch after warmup

scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)

# LR schedule:
# Epochs 0-9:  lr = 0.1 (unchanged)
# Epoch 10:    lr = 0.095
# Epoch 11:    lr = 0.09025
# ...
```

**Per-Parameter-Group Lambdas**:
```python
optimizer = torch.optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 0.01},
    {'params': model.head.parameters(), 'lr': 0.1}
])

# Different decay rates for different groups
lambda_backbone = lambda epoch: 0.99  # Slow decay for backbone
lambda_head = lambda epoch: 0.9       # Fast decay for head

scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer,
    lr_lambda=[lambda_backbone, lambda_head]
)
```

### MLX Implementation

```python
class MultiplicativeLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)

    def step(self):
        self.last_epoch += 1
        for param_group, lr_lambda in zip(self.optimizer.param_groups, self.lr_lambdas):
            param_group['lr'] *= lr_lambda(self.last_epoch)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
```

---

## Scheduler Comparison

| Scheduler | Type | When to Use | Complexity |
|-----------|------|-------------|------------|
| **StepLR** | Fixed | CV, simple decay | Low |
| **MultiStepLR** | Fixed | CV (ResNet recipe) | Low |
| **ExponentialLR** | Fixed | Smooth decay needed | Low |
| **CosineAnnealingLR** | Fixed | Transformers, smooth decay | Medium |
| **ReduceLROnPlateau** | Adaptive | Unstable training, RNNs | Medium |
| **LinearLR** | Fixed | Warmup | Low |
| **LambdaLR** | Custom | Arbitrary schedules | High |
| **OneCycleLR** | Fixed | Fast training, super-convergence | High |
| **PolynomialLR** | Fixed | NLP, BERT fine-tuning | Low |
| **CosineAnnealingWarmRestarts** | Fixed | SGDR, escape local minima | Medium |
| **CyclicLR** | Fixed | LR range finding, cyclic training | Medium |
| **ConstantLR** | Fixed | Simple warmup component | Low |
| **SequentialLR** | Composite | Warmup + decay patterns | Medium |
| **ChainedScheduler** | Composite | Compound multiple effects | Medium |
| **MultiplicativeLR** | Custom | Multiplicative decay | Medium |

---

## Best Practices

### 1. Call Order

**CRITICAL**: Always call `scheduler.step()` **AFTER** `optimizer.step()`:

```python
# CORRECT
for epoch in range(epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
    scheduler.step()  # After all batches

# WRONG - PyTorch will warn
for epoch in range(epochs):
    scheduler.step()  # Before optimizer.step()
    for batch in data_loader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
```

### 2. Per-Batch vs Per-Epoch

Most schedulers step **per-epoch**, but some (OneCycleLR, CyclicLR) step **per-batch**:

```python
# Per-epoch scheduler
scheduler = StepLR(optimizer, step_size=30)
for epoch in range(epochs):
    train_epoch()
    scheduler.step()  # Once per epoch

# Per-batch scheduler
scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
for epoch in range(epochs):
    for batch in data_loader:
        train_step(batch)
        scheduler.step()  # Once per batch
```

### 3. Checkpointing

Save and load scheduler state:

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
```

### 4. Combining Schedulers

Use `SequentialLR` to chain schedulers:

```python
# Warmup (10 epochs) → Cosine decay (90 epochs)
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=90)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])
```

---

## MLX Porting Guide

### Recommended Priority

**High Priority** (Implement First):
1. **StepLR** - Most common for CV
2. **MultiStepLR** - Standard ResNet recipe
3. **CosineAnnealingLR** - Transformers
4. **LinearLR** - Warmup
5. **PolynomialLR** - NLP/BERT fine-tuning
6. **SequentialLR** - Composing warmup + decay

**Medium Priority**:
7. **ReduceLROnPlateau** - Adaptive training
8. **ExponentialLR** - Simple smooth decay
9. **CosineAnnealingWarmRestarts** - SGDR
10. **ConstantLR** - Warmup component
11. **ChainedScheduler** - Compound effects

**Low Priority**:
12. **LambdaLR** - Flexibility (can be implemented via Python callbacks)
13. **OneCycleLR** - Advanced training
14. **CyclicLR** - LR range finding
15. **MultiplicativeLR** - Custom multiplicative decay

### MLX Base Class

```python
class LRScheduler:
    """Base class for MLX learning rate schedulers"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rates"""
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

    def get_lr(self):
        """Compute new learning rates (override in subclasses)"""
        raise NotImplementedError

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_epoch': self.last_epoch, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
```

### Example: MLX StepLR

```python
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]
```

### Example: MLX CosineAnnealingLR

```python
import math

class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
```

---

## References

1. **Step Decay**: He, K. et al. (2015). "Deep Residual Learning for Image Recognition"
2. **Cosine Annealing**: Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts"
3. **OneCycleLR**: Smith, L. & Topin, N. (2019). "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
4. **Warmup**: Goyal, P. et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"

---

## Summary

Learning rate schedulers are essential for achieving state-of-the-art performance in deep learning.

**Total Schedulers Documented**: 15 / 16+ PyTorch schedulers (94%+)

| Scheduler | Category | Use Case |
|-----------|----------|----------|
| StepLR | Fixed | CV, simple decay |
| MultiStepLR | Fixed | CV (ResNet recipe) |
| ExponentialLR | Fixed | Smooth decay |
| CosineAnnealingLR | Fixed | Transformers, smooth decay |
| ReduceLROnPlateau | Adaptive | RNNs, unstable training |
| LinearLR | Fixed | Warmup |
| LambdaLR | Custom | Arbitrary schedules |
| OneCycleLR | Fixed | Super-convergence |
| PolynomialLR | Fixed | NLP, BERT fine-tuning |
| CosineAnnealingWarmRestarts | Fixed | SGDR |
| CyclicLR | Fixed | LR range finding |
| ConstantLR | Fixed | Warmup component |
| SequentialLR | Composite | Warmup + decay |
| ChainedScheduler | Composite | Compound effects |
| MultiplicativeLR | Custom | Multiplicative decay |

**Most Common Schedules**:
- **Computer Vision**: MultiStepLR with milestones=[30, 60, 80], gamma=0.1
- **Transformers**: LinearLR warmup + CosineAnnealingLR (via SequentialLR)
- **NLP/BERT**: LinearLR warmup + PolynomialLR decay
- **RNNs**: ReduceLROnPlateau with validation loss
- **Fast Training**: OneCycleLR with super-convergence
- **SGDR**: CosineAnnealingWarmRestarts for exploring multiple basins

**Key Insights**:
- LR scheduling often provides 1-5% accuracy improvement for same training time
- Warmup is critical for large batch training (>256)
- Cosine annealing provides smoother convergence than step decay
- ReduceLROnPlateau is most robust when optimal schedule is unknown
- SequentialLR is essential for modern warmup + decay patterns
- CosineAnnealingWarmRestarts can escape local minima

**MLX Implementation Priority**:
1. Start with StepLR, MultiStepLR, CosineAnnealingLR, LinearLR to cover 80% of CV use cases
2. Add PolynomialLR and SequentialLR for NLP/Transformer training
3. Add ReduceLROnPlateau for adaptive training
4. Add CosineAnnealingWarmRestarts for SGDR
5. LambdaLR/MultiplicativeLR can be deferred as users can implement custom logic in Python
