# Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA)

## Overview

Weight averaging techniques improve model generalization by maintaining averaged copies of model parameters during training. PyTorch provides these utilities in `torch.optim.swa_utils`.

**Key Concepts**:
- **SWA**: Averages weights over training trajectory (equal weights)
- **EMA**: Exponentially weighted moving average (recent weights emphasized)
- **Polyak Averaging**: General term for iterative weight averaging

**Benefits**:
- Wider optima (better generalization)
- Reduced sensitivity to learning rate
- Smoother convergence
- Often improves test accuracy 1-2%

---

## API Components

```python
from torch.optim.swa_utils import (
    AveragedModel,        # Maintains averaged weights
    SWALR,                # Learning rate scheduler for SWA
    update_bn,            # Update BatchNorm statistics
    get_ema_multi_avg_fn, # EMA averaging function
    get_swa_multi_avg_fn, # SWA averaging function
    get_ema_avg_fn,       # Single-param EMA function
    get_swa_avg_fn,       # Single-param SWA function
)
```

**Source**: `torch/optim/swa_utils.py`

---

## AveragedModel

The `AveragedModel` class maintains a copy of a model with averaged parameters.

### Signature

```python
class AveragedModel(Module):
    def __init__(
        self,
        model: Module,
        device: int | torch.device | None = None,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[list, list, Tensor | int], None] | None = None,
        use_buffers: bool = False,
    )
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | Model to average |
| `device` | device | Device for averaged model |
| `avg_fn` | Callable | Custom averaging function (single param) |
| `multi_avg_fn` | Callable | Vectorized averaging function |
| `use_buffers` | bool | Also average buffers (not just parameters) |

### Basic SWA Usage

```python
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# Create model and optimizer
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 10)
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Create SWA model
swa_model = AveragedModel(model)

# Training loop
swa_start = 160
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(300):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    if epoch >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

# Update BatchNorm statistics for SWA model
update_bn(train_loader, swa_model)

# Use swa_model for inference
with torch.no_grad():
    predictions = swa_model(test_data)
```

### EMA Usage

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# EMA model with decay=0.999
ema_model = AveragedModel(
    model,
    multi_avg_fn=get_ema_multi_avg_fn(decay=0.999),
    use_buffers=True  # Also average buffers
)

# Training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        # Update EMA after each batch
        ema_model.update_parameters(model)

# Use ema_model for inference
ema_model.eval()
predictions = ema_model(test_data)
```

### update_parameters Method

```python
def update_parameters(self, model: Module) -> None:
    """Update averaged model parameters.

    On first call (n_averaged=0): Copy parameters from model
    On subsequent calls: Apply averaging function
    """
```

**Source**: `torch/optim/swa_utils.py:251-325`

---

## Averaging Functions

### SWA Averaging

Equal-weighted running average:
```
param_avg = param_avg + (param_model - param_avg) / (n + 1)
```

```python
def get_swa_avg_fn():
    """Single parameter SWA update."""
    @torch.no_grad()
    def swa_update(averaged_param, current_param, num_averaged):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)
    return swa_update

def get_swa_multi_avg_fn():
    """Vectorized SWA update (more efficient)."""
    @torch.no_grad()
    def swa_update(averaged_param_list, current_param_list, num_averaged):
        torch._foreach_lerp_(
            averaged_param_list,
            current_param_list,
            1 / (num_averaged + 1)
        )
    return swa_update
```

### EMA Averaging

Exponentially weighted average:
```
param_ema = decay * param_ema + (1 - decay) * param_model
```

```python
def get_ema_avg_fn(decay=0.999):
    """Single parameter EMA update."""
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param
    return ema_update

def get_ema_multi_avg_fn(decay=0.999):
    """Vectorized EMA update (more efficient)."""
    @torch.no_grad()
    def ema_update(ema_param_list, current_param_list, _):
        torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
    return ema_update
```

### Custom Averaging Function

```python
def get_custom_avg_fn(warmup_steps=1000, target_decay=0.9999):
    """Custom averaging with warmup."""
    @torch.no_grad()
    def custom_update(ema_param, current_param, num_averaged):
        # Linear warmup for decay
        decay = min(target_decay, (1 + num_averaged) / (10 + num_averaged))
        return decay * ema_param + (1 - decay) * current_param
    return custom_update

# Use with AveragedModel
ema_model = AveragedModel(model, avg_fn=get_custom_avg_fn())
```

---

## SWALR Scheduler

The `SWALR` scheduler anneals learning rate to a fixed SWA learning rate.

### Signature

```python
class SWALR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs: int = 10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch: int = -1,
    )
```

**Source**: `torch/optim/swa_utils.py:387-550`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `optimizer` | Optimizer | Wrapped optimizer |
| `swa_lr` | float | Target SWA learning rate |
| `anneal_epochs` | int | Epochs to anneal to swa_lr |
| `anneal_strategy` | str | "cos" or "linear" |

### Annealing Strategies

**Cosine Annealing**:
```python
def _cosine_anneal(t):
    return (1 - math.cos(math.pi * t)) / 2
```

**Linear Annealing**:
```python
def _linear_anneal(t):
    return t
```

### Example

```python
from torch.optim.swa_utils import SWALR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Anneal to swa_lr=0.05 over 10 epochs
swa_scheduler = SWALR(
    optimizer,
    swa_lr=0.05,
    anneal_epochs=10,
    anneal_strategy="cos"
)

# During SWA phase
for epoch in range(swa_start, total_epochs):
    train(...)
    swa_model.update_parameters(model)
    swa_scheduler.step()
```

---

## update_bn

Update BatchNorm running statistics for the averaged model.

### Signature

```python
@torch.no_grad()
def update_bn(
    loader: Iterable[Any],
    model: Module,
    device: int | torch.device | None = None,
) -> None
```

**Source**: `torch/optim/swa_utils.py:328-384`

### Why Update BN?

Averaged weights produce different activations than the original model. BatchNorm statistics (running_mean, running_var) need to be recalculated for accurate inference.

### Example

```python
from torch.optim.swa_utils import update_bn

# After training with SWA
update_bn(train_loader, swa_model)

# Or with specific device
update_bn(train_loader, swa_model, device='cuda:0')
```

### Alternative: use_buffers=True

```python
# Average buffers during training (including BN stats)
swa_model = AveragedModel(model, use_buffers=True)

# No need to call update_bn
# But results may differ from post-hoc update
```

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

def train_with_swa(
    model,
    train_loader,
    val_loader,
    epochs=300,
    swa_start=160,
    swa_lr=0.05,
    base_lr=0.1
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Pre-SWA scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=swa_start
    )

    # SWA scheduler
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_epochs=5,
        anneal_strategy='cos'
    )

    # SWA model
    swa_model = AveragedModel(model)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Scheduler step
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation
        if epoch % 10 == 0:
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: Val Acc = {val_acc:.2%}")

    # Update BN for SWA model
    update_bn(train_loader, swa_model, device=device)

    # Final evaluation
    swa_val_acc = evaluate(swa_model, val_loader, device)
    print(f"SWA Val Acc = {swa_val_acc:.2%}")

    return swa_model

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
```

---

## EMA for Generative Models

EMA is commonly used in diffusion models and GANs:

```python
class DiffusionTrainer:
    def __init__(self, model, ema_decay=0.9999):
        self.model = model
        self.ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
            use_buffers=True
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step(self, x, noise):
        self.optimizer.zero_grad()
        loss = self.compute_loss(x, noise)
        loss.backward()
        self.optimizer.step()

        # Update EMA after each step
        self.ema_model.update_parameters(self.model)

    @torch.no_grad()
    def sample(self, shape):
        # Use EMA model for sampling (more stable)
        self.ema_model.eval()
        return self.ema_model.denoise(torch.randn(shape))
```

---

## MLX Implementation

MLX doesn't have built-in SWA/EMA utilities. Here's an implementation:

### EMA Model

```python
import mlx.core as mx
import mlx.nn as nn

class EMAModel:
    """Exponential Moving Average model for MLX."""

    def __init__(self, model: nn.Module, decay: float = 0.999, use_buffers: bool = True):
        self.decay = decay
        self.use_buffers = use_buffers
        self.n_averaged = 0

        # Create deep copy of model
        self.model = self._copy_model(model)

    def _copy_model(self, model):
        """Create a copy of model with same parameters."""
        import copy
        return copy.deepcopy(model)

    def update_parameters(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        ema_params = dict(self.model.parameters())
        model_params = dict(model.parameters())

        if self.n_averaged == 0:
            # First update: copy parameters
            for name in ema_params:
                ema_params[name] = model_params[name]
        else:
            # EMA update
            for name in ema_params:
                ema_params[name] = (
                    self.decay * ema_params[name] +
                    (1 - self.decay) * model_params[name]
                )

        self.model.update(ema_params)
        self.n_averaged += 1

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class SWAModel:
    """Stochastic Weight Average model for MLX."""

    def __init__(self, model: nn.Module, use_buffers: bool = False):
        self.use_buffers = use_buffers
        self.n_averaged = 0
        self.model = self._copy_model(model)

    def _copy_model(self, model):
        import copy
        return copy.deepcopy(model)

    def update_parameters(self, model: nn.Module) -> None:
        """Update SWA parameters with equal weighting."""
        swa_params = dict(self.model.parameters())
        model_params = dict(model.parameters())

        if self.n_averaged == 0:
            for name in swa_params:
                swa_params[name] = model_params[name]
        else:
            # SWA update: running average
            for name in swa_params:
                diff = model_params[name] - swa_params[name]
                swa_params[name] = swa_params[name] + diff / (self.n_averaged + 1)

        self.model.update(swa_params)
        self.n_averaged += 1

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
```

### SWALR Scheduler for MLX

```python
import math

class SWALR:
    """SWA Learning Rate scheduler for MLX."""

    def __init__(
        self,
        optimizer,  # MLX optimizer
        swa_lr: float,
        anneal_epochs: int = 10,
        anneal_strategy: str = "cos"
    ):
        self.optimizer = optimizer
        self.swa_lr = swa_lr
        self.anneal_epochs = anneal_epochs
        self.anneal_strategy = anneal_strategy
        self.initial_lr = optimizer.learning_rate
        self._step = 0

    def _anneal_func(self, t: float) -> float:
        if self.anneal_strategy == "cos":
            return (1 - math.cos(math.pi * t)) / 2
        else:  # linear
            return t

    def step(self):
        """Update learning rate."""
        self._step += 1
        t = max(0, min(1, self._step / max(1, self.anneal_epochs)))
        alpha = self._anneal_func(t)

        new_lr = self.swa_lr * alpha + self.initial_lr * (1 - alpha)
        self.optimizer.learning_rate = new_lr

    def get_last_lr(self):
        return self.optimizer.learning_rate
```

### Usage Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Model and optimizer
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
optimizer = optim.SGD(learning_rate=0.1)

# EMA model
ema_model = EMAModel(model, decay=0.999)

# Training loop
for epoch in range(100):
    for x, y in train_loader:
        loss, grads = loss_fn_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Update EMA
        ema_model.update_parameters(model)

# Use EMA model for inference
predictions = ema_model(test_data)
mx.eval(predictions)
```

---

## Comparison

| Aspect | SWA | EMA |
|--------|-----|-----|
| Weighting | Equal | Exponential decay |
| Update frequency | Per epoch (typically) | Per step (typically) |
| Memory | 2x parameters | 2x parameters |
| Typical use | Classification | Generative models |
| Convergence | End of training | Throughout training |

---

## Implementation Files

- `torch/optim/swa_utils.py` - All SWA/EMA utilities

**Key Classes**:
- `AveragedModel`: Lines 122-325
- `SWALR`: Lines 387-550
- `update_bn`: Lines 328-384

**Averaging Functions**:
- `get_ema_multi_avg_fn`: Lines 37-58
- `get_swa_multi_avg_fn`: Lines 61-92
- `get_ema_avg_fn`: Lines 95-107
- `get_swa_avg_fn`: Lines 110-119

---

## References

**Papers**:
- "Averaging Weights Leads to Wider Optima and Better Generalization" (Izmailov et al., 2018)
- Polyak averaging for improved convergence
- EMA in diffusion models (Song et al., 2020)

**Use Cases**:
- Image classification (SWA)
- Diffusion models (EMA)
- Semi-supervised learning
- Domain adaptation
