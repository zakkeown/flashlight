# PyTorch Optimizer Base Class

## Purpose

This document analyzes PyTorch's `Optimizer` base class, the foundation for all optimization algorithms (SGD, Adam, RMSprop, etc.). Understanding the optimizer architecture is critical for MLX porting because:

1. It defines the interface for parameter updates during training
2. It establishes parameter grouping and learning rate management
3. It provides state management for momentum and adaptive methods
4. It integrates with autograd for gradient-based optimization

## Architecture Overview

### Optimizer: The Base Class

Every optimizer in PyTorch inherits from `torch.optim.Optimizer`.

**Source**: [torch/optim/optimizer.py](../reference/pytorch/torch/optim/optimizer.py)

```
┌──────────────────────────────────────────────────────────┐
│                     Optimizer (Base)                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  State (OrderedDict)                               │ │
│  │  • param_id → state_dict                           │ │
│  │  • momentum_buffer, exp_avg, exp_avg_sq, etc.     │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Parameter Groups (list of dicts)                  │ │
│  │  [{'params': [p1, p2], 'lr': 0.01},               │ │
│  │   {'params': [p3, p4], 'lr': 0.001}]              │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Core Methods                                      │ │
│  │  • __init__(params, defaults)                      │ │
│  │  • step(closure=None): Apply updates              │ │
│  │  • zero_grad(set_to_none=False): Clear gradients  │ │
│  │  • state_dict(): Serialize state                   │ │
│  │  • load_state_dict(state_dict): Deserialize       │ │
│  │  • add_param_group(param_group): Add params       │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                        ▲
          ┌─────────────┼─────────────┐
          │             │             │
     ┌────────┐   ┌─────────┐   ┌─────────┐
     │  SGD   │   │  Adam   │   │ RMSprop │
     └────────┘   └─────────┘   └─────────┘
```

## Core Components

### Component 1: Initialization and Parameter Groups

#### __init__ Method

**Source**: [torch/optim/optimizer.py:350-450](../reference/pytorch/torch/optim/optimizer.py)

```python
class Optimizer:
    r"""Base class for all optimizers.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        defaults: dict containing default values of optimization options
    """

    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None:
        self.defaults = defaults
        self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []

        # Parse parameter groups
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if not isinstance(param_groups[0], dict):
            # Single group of parameters
            param_groups = [{'params': param_groups}]

        # Process each parameter group
        for param_group in param_groups:
            self.add_param_group(cast(dict[str, Any], param_group))

        # Hooks for step
        self._optimizer_step_pre_hooks: dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: dict[int, Callable] = OrderedDict()
        self._optimizer_state_dict_pre_hooks: dict[int, Callable] = OrderedDict()
        self._optimizer_state_dict_post_hooks: dict[int, Callable] = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks: dict[int, Callable] = OrderedDict()
        self._optimizer_load_state_dict_post_hooks: dict[int, Callable] = OrderedDict()
```

#### Parameter Groups

Parameter groups allow different hyperparameters for different subsets of parameters:

```python
# Example: Different learning rates for encoder and decoder
model = MyModel()
optimizer = torch.optim.SGD([
    {'params': model.encoder.parameters(), 'lr': 0.01},
    {'params': model.decoder.parameters(), 'lr': 0.001}
])
```

**add_param_group Method**:

```python
def add_param_group(self, param_group: dict[str, Any]) -> None:
    r"""Add a param group to the :class:`Optimizer` s `param_groups`.

    This can be useful when fine tuning a pre-trained network as frozen layers can be made
    trainable and added to the :class:`Optimizer` as training progresses.

    Args:
        param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
    """
    if not isinstance(param_group, dict):
        raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

    # Extract parameters
    params = param_group['params']
    if isinstance(params, torch.Tensor):
        param_group['params'] = [params]
    elif isinstance(params, set):
        raise TypeError('optimizer parameters need to be organized in ordered collections')
    else:
        param_group['params'] = list(params)

    # Check for duplicate parameters
    param_set: set[torch.Tensor] = set()
    for group in self.param_groups:
        param_set.update(set(group['params']))

    if not param_set.isdisjoint(set(param_group['params'])):
        raise ValueError("some parameters appear in more than one parameter group")

    # Apply defaults
    for key, value in self.defaults.items():
        param_group.setdefault(key, value)

    # Validate parameters
    for param in param_group['params']:
        if not isinstance(param, torch.Tensor):
            raise TypeError(f"optimizer can only optimize Tensors, but one of the params is {type(param)}")
        if not param.is_leaf:
            raise ValueError("can't optimize a non-leaf Tensor")

    self.param_groups.append(param_group)
```

**Usage Pattern**:
```python
# Initially train only classifier
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

# After N epochs, unfreeze feature extractor
for param in model.features.parameters():
    param.requires_grad = True

# Add feature extractor with lower learning rate
optimizer.add_param_group({
    'params': model.features.parameters(),
    'lr': 0.001
})
```

### Component 2: State Management

#### State Dictionary

Optimizers maintain state for each parameter (e.g., momentum buffers, adaptive learning rates):

```python
class Optimizer:
    def __init__(self, params, defaults):
        # ...
        self.state: defaultdict[torch.Tensor, Any] = defaultdict(dict)
```

**State Examples**:
- **SGD with momentum**: `momentum_buffer`
- **Adam**: `exp_avg` (first moment), `exp_avg_sq` (second moment), `step`
- **RMSprop**: `square_avg`, `step`

**Accessing State**:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for a few steps
for epoch in range(10):
    optimizer.zero_grad()
    loss = model(data).sum()
    loss.backward()
    optimizer.step()

# Inspect optimizer state
for param in model.parameters():
    if param in optimizer.state:
        state = optimizer.state[param]
        print(f"Step: {state.get('step', 0)}")
        print(f"Exp avg: {state.get('exp_avg', 'N/A')}")
        print(f"Exp avg sq: {state.get('exp_avg_sq', 'N/A')}")
```

### Component 3: zero_grad() Method

#### Gradient Clearing

**Source**: [torch/optim/optimizer.py:580-620](../reference/pytorch/torch/optim/optimizer.py)

```python
def zero_grad(self, set_to_none: bool = True) -> None:
    r"""Reset the gradients of all optimized :class:`torch.Tensor` s.

    Args:
        set_to_none (bool): instead of setting to zero, set the grads to None.
            This will in general have lower memory footprint, and can modestly improve performance.
            However, it changes certain behaviors. For example:
            1. When the user tries to access a gradient and perform manual ops on it,
            a None attribute or a Tensor full of 0s will behave differently.
            2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass,
            ``.grad``s are guaranteed to be None for params that did not receive a gradient.
            3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
            (in one case it does the step with a gradient of 0 and in the other it skips
            the step altogether).
    """
    if not hasattr(self, "_zero_grad_profile_name"):
        self._patch_step_function()

    # Call global hooks if any
    for hook in _global_optimizer_pre_hooks.values():
        result = hook(self, [], {})
        if result is not None:
            raise RuntimeError(
                "global optimizer pre-hooks can only return None, but got "
                f"{type(result)}"
            )

    # Zero gradients for all parameters
    foreach = self.defaults.get('foreach', False) or self.defaults.get('fused', False)

    if foreach:
        # Fast vectorized zero_grad
        per_device_and_dtype_grads: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)

                    # Group by device and dtype for vectorized operation
                    key = (p.grad.device, p.grad.dtype)
                    per_device_and_dtype_grads.setdefault(key, []).append(p.grad)

        # Vectorized zero for each (device, dtype) group
        for grads in per_device_and_dtype_grads.values():
            if set_to_none:
                for grad in grads:
                    grad.zero_()
            else:
                torch._foreach_zero_(grads)

        # Set gradients to None if requested
        if set_to_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
    else:
        # Slow single-tensor zero_grad
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
```

**Performance Comparison**:
```python
# set_to_none=True (faster, lower memory)
optimizer.zero_grad(set_to_none=True)

# set_to_none=False (default, safer)
optimizer.zero_grad(set_to_none=False)
```

**Why `set_to_none=True` is faster**:
1. Avoids allocating zeros
2. Reduces memory footprint
3. Allows optimizer to skip params with no gradient

### Component 4: step() Method

#### Parameter Update

The `step()` method is **overridden by each optimizer** to implement the specific update rule.

**Base Class Template**:
```python
@_use_grad_for_differentiable
def step(self, closure: Callable[[], float] | None = None) -> float | None:
    r"""Performs a single optimization step (parameter update).

    Args:
        closure (Callable): A closure that reevaluates the model and returns the loss.
            Optional for most optimizers.

    .. note::
        Unless otherwise specified, this function should not modify the
        ``.grad`` field of the parameters.
    """
    raise NotImplementedError("Subclass must implement step()")
```

**SGD Implementation Example**:
```python
@_use_grad_for_differentiable
def step(self, closure=None):
    """Performs a single optimization step."""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []

        # Collect parameters with gradients
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        # Perform SGD update
        sgd(
            params_with_grad,
            d_p_list,
            momentum_buffer_list,
            weight_decay=group['weight_decay'],
            momentum=group['momentum'],
            lr=group['lr'],
            dampening=group['dampening'],
            nesterov=group['nesterov'],
            maximize=group['maximize'],
            has_sparse_grad=False,
            foreach=group['foreach'],
        )

        # Update momentum buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state['momentum_buffer'] = momentum_buffer

    return loss
```

**Closure Usage** (for line search optimizers like LBFGS):
```python
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

### Component 5: State Dict Serialization

#### save_state_dict() and load_state_dict()

**state_dict Method**:
```python
def state_dict(self) -> StateDict:
    r"""Returns the state of the optimizer as a :class:`dict`.

    It contains two entries:

    * ``state``: a Dict holding current optimization state. Its content
        differs between optimizer classes, but some common keys are the
        buffers used for momentum and adaptive learning rates.
    * ``param_groups``: a list containing all parameter groups where each
        parameter group is a Dict. Each parameter group contains metadata
        specific to the optimization, such as learning rate and momentum.
    """
    # Call pre-hooks
    for hook in chain(
        self._optimizer_state_dict_pre_hooks.values(),
        _global_optimizer_state_dict_pre_hooks.values()
    ):
        hook(self)

    # Pack state
    packed_state = {
        id(k) if isinstance(k, torch.Tensor) else k: v
        for k, v in self.state.items()
    }

    # Pack param_groups
    param_groups = [
        {
            **{k: v for k, v in group.items() if k != 'params'},
            'params': [id(p) if isinstance(p, torch.Tensor) else p for p in group['params']]
        }
        for group in self.param_groups
    ]

    # Call post-hooks
    for hook in chain(
        self._optimizer_state_dict_post_hooks.values(),
        _global_optimizer_state_dict_post_hooks.values()
    ):
        hook(self)

    return {
        'state': packed_state,
        'param_groups': param_groups,
    }
```

**load_state_dict Method**:
```python
def load_state_dict(self, state_dict: StateDict) -> None:
    r"""Loads the optimizer state.

    Args:
        state_dict (dict): optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
    """
    # Call pre-hooks
    for hook in chain(
        self._optimizer_load_state_dict_pre_hooks.values(),
        _global_optimizer_load_state_dict_pre_hooks.values()
    ):
        hook(self, state_dict)

    # Deepcopy to avoid modifying input
    state_dict = deepcopy(state_dict)

    # Validate state_dict
    groups = self.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of parameter groups")

    # Map old parameter IDs to new parameters
    param_lens = [len(g['params']) for g in groups]
    saved_lens = [len(g['params']) for g in saved_groups]
    if param_lens != saved_lens:
        raise ValueError(
            "loaded state dict contains a parameter group that doesn't match the size "
            f"of optimizer's group: {param_lens} vs {saved_lens}"
        )

    # Update param_groups
    id_map = {}
    for group, saved_group in zip(groups, saved_groups):
        # Update hyperparameters
        for key, value in saved_group.items():
            if key != 'params':
                group[key] = value

        # Map parameter IDs
        for p, saved_id in zip(group['params'], saved_group['params']):
            id_map[saved_id] = id(p)

    # Load state
    def cast(param, value):
        """Cast state value to param's dtype and device."""
        if isinstance(value, torch.Tensor):
            if param.is_floating_point():
                value = value.to(param.dtype)
            value = value.to(param.device)
        return value

    # Update state
    self.state = defaultdict(dict)
    for param_id, state in state_dict['state'].items():
        if param_id not in id_map:
            continue

        param = next(p for p in chain(*[g['params'] for g in groups]) if id(p) == id_map[param_id])
        self.state[param] = {
            key: cast(param, value) for key, value in state.items()
        }

    # Call post-hooks
    for hook in chain(
        self._optimizer_load_state_dict_post_hooks.values(),
        _global_optimizer_load_state_dict_post_hooks.values()
    ):
        hook(self, state_dict)
```

**Usage: Checkpointing**:
```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Component 6: Learning Rate Scheduling Integration

#### Accessing and Modifying Learning Rates

```python
# Get current learning rate
for param_group in optimizer.param_groups:
    print(f"Learning rate: {param_group['lr']}")

# Manually set learning rate
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.001

# Using a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()  # Update learning rate
```

**Learning Rate Scheduler Example**:
```python
class StepLR:
    """Decays learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rates."""
        self.last_epoch += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * (self.gamma ** (self.last_epoch // self.step_size))
```

### Component 7: Gradient Clipping Integration

#### Gradient Norm Clipping

```python
# Clip gradients before optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Implementation**:
```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """Clips gradient norm of an iterable of parameters."""
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]),
            norm_type
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm
```

## MLX Optimizer Patterns

### MLX Optimizer Approach

MLX uses a functional approach instead of object-oriented optimizers:

```python
import mlx.core as mx
import mlx.optimizers as optim

# Create optimizer
optimizer = optim.Adam(learning_rate=0.001)

# Training loop
def loss_fn(model, x, y):
    return mx.mean((model(x) - y) ** 2)

# Get gradients using value_and_grad
loss_and_grad_fn = mx.value_and_grad(loss_fn)

for epoch in range(num_epochs):
    # Forward and backward
    loss, grads = loss_and_grad_fn(model, x_batch, y_batch)

    # Update parameters
    optimizer.update(model, grads)

    # Evaluate to execute graph
    mx.eval(model.parameters())
```

**Comparison**:

| Aspect | PyTorch Optimizer | MLX Optimizer |
|--------|------------------|---------------|
| **Style** | Object-oriented (class-based) | Functional (update function) |
| **State** | `self.state` dict per parameter | Internal state in optimizer |
| **Gradients** | Stored in `param.grad` | Passed as argument to `update()` |
| **Parameter Groups** | Explicit groups with different LRs | Single learning rate (can create multiple optimizers) |
| **zero_grad()** | Required before backward | Not needed (grads passed directly) |
| **Step** | `optimizer.step()` | `optimizer.update(model, grads)` |

### Migration Pattern: PyTorch → MLX

**PyTorch Training Loop**:
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

**MLX Equivalent**:
```python
model = MyMLXModel()
optimizer = optim.Adam(learning_rate=0.001)

def loss_fn(model, x, y):
    output = model(x)
    return mx.mean((output - y) ** 2)

loss_and_grad_fn = mx.value_and_grad(loss_fn)

for epoch in range(num_epochs):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
```

## Critical File References

### PyTorch Optimizer Core
- [torch/optim/optimizer.py](../reference/pytorch/torch/optim/optimizer.py) - Base Optimizer class
- [torch/optim/sgd.py](../reference/pytorch/torch/optim/sgd.py) - SGD implementation
- [torch/optim/adam.py](../reference/pytorch/torch/optim/adam.py) - Adam implementation
- [torch/optim/lr_scheduler.py](../reference/pytorch/torch/optim/lr_scheduler.py) - Learning rate schedulers

### Utilities
- [torch/nn/utils/clip_grad.py](../reference/pytorch/torch/nn/utils/clip_grad.py) - Gradient clipping
- [torch/utils/_foreach_utils.py](../reference/pytorch/torch/utils/_foreach_utils.py) - Vectorized operations

## Next Steps

To continue understanding PyTorch → MLX porting:

1. **Read [05-TRAINING/optimizers/adam.md](optimizers/adam.md)** - Concrete optimizer implementation
2. **Read [04-NEURAL-NETWORKS/module-system.md](../04-NEURAL-NETWORKS/module-system.md)** - How modules interact with optimizers
3. **Read [03-AUTOGRAD/autograd-overview.md](../03-AUTOGRAD/autograd-overview.md)** - How gradients are computed
4. **Study MLX optimizers** - See MLX's functional approach

## Summary

PyTorch's `Optimizer` base class provides a robust, extensible framework for optimization algorithms. Key takeaways for MLX porting:

### PyTorch Optimizer Strengths
- ✅ Parameter groups for different learning rates
- ✅ Comprehensive state management
- ✅ Efficient vectorized operations (foreach/fused)
- ✅ Serialization for checkpointing
- ✅ Hook system for custom behavior

### PyTorch Optimizer Complexity (MLX Simplifications)
- ❌ Object-oriented (MLX uses functional)
- ❌ Explicit `zero_grad()` required (MLX passes grads directly)
- ❌ Manual state dict handling (MLX hides internally)

### Porting Strategy
1. **Core Pattern**: MLX's functional optimizers are simpler than PyTorch's OOP approach
2. **State Management**: MLX handles state internally, no explicit state dict needed
3. **Parameter Groups**: Create multiple MLX optimizers instead of parameter groups
4. **Gradient Handling**: MLX's `value_and_grad` replaces `backward()` + `zero_grad()`
5. **Checkpointing**: Use MLX's `model.save_weights()` instead of optimizer state dict

The optimizer system is critical for training, but MLX's functional approach provides a cleaner, more composable alternative to PyTorch's object-oriented design.
