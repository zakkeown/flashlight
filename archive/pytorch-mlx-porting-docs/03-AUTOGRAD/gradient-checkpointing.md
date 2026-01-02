# Gradient Checkpointing (Activation Checkpointing)

## Overview

Gradient checkpointing (also called activation checkpointing) is a technique that trades compute for memory during backpropagation. Instead of storing all intermediate activations from the forward pass, checkpointed regions discard activations and recompute them during the backward pass. This significantly reduces memory usage at the cost of additional computation.

**Reference File:** `torch/utils/checkpoint.py`

## Core Concept

### Standard Backpropagation Memory

In normal training:
1. Forward pass computes and stores all intermediate activations
2. Backward pass uses stored activations to compute gradients
3. Memory scales with model depth (O(n) for n layers)

### Checkpointed Backpropagation

With checkpointing:
1. Forward pass computes but discards intermediate activations
2. Only checkpoint boundaries store activations
3. Backward pass recomputes activations as needed
4. Memory scales with O(√n) when properly segmented

```
Standard:    [input] → [a1] → [a2] → [a3] → [a4] → [output]
                ↓        ↓       ↓       ↓       ↓
             (stored) (stored) (stored) (stored) (stored)

Checkpointed: [input] → [a1] → [a2] → [a3] → [a4] → [output]
                ↓                               ↓
             (stored)                        (stored)
                     ↑_____recompute_____↑
```

---

## API Reference

### checkpoint()

The primary function for activation checkpointing.

```python
def checkpoint(
    function: Callable,             # Function to checkpoint
    *args,                          # Inputs to the function
    use_reentrant: bool = None,     # MUST be specified (recommended: False)
    preserve_rng_state: bool = True,  # Preserve RNG state
    context_fn: Callable = noop_context_fn,  # Custom context managers
    determinism_check: str = "default",  # Check recomputation matches
    debug: bool = False,            # Print debug info
    early_stop: bool = True,        # Stop recomputation early
    **kwargs                        # Additional function args
) -> Any
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `function` | The forward function to checkpoint |
| `*args` | Positional arguments to the function |
| `use_reentrant` | **Required.** Use `False` (recommended) or `True` |
| `preserve_rng_state` | Save/restore RNG state for reproducibility |
| `context_fn` | Tuple of context managers for forward/recompute |
| `determinism_check` | `"default"` or `"none"` to check recomputation |
| `debug` | Print operator trace for debugging |
| `early_stop` | Stop recomputing when all needed activations computed |

### use_reentrant Modes

**`use_reentrant=False` (Recommended):**
- Modern implementation
- Supports `torch.autograd.grad`
- Supports keyword arguments
- Allows nested structures
- Early stopping optimization
- Works with detached tensors

**`use_reentrant=True` (Legacy):**
- Original implementation
- Runs forward under `torch.no_grad()`
- Requires at least one input/output with `requires_grad=True`
- No support for `inputs` argument in `.backward()`

---

## Basic Usage

### Simple Example

```python
import torch
from torch.utils.checkpoint import checkpoint

def expensive_block(x):
    """Block that we want to checkpoint."""
    x = x @ weights1
    x = torch.relu(x)
    x = x @ weights2
    x = torch.relu(x)
    return x

# Without checkpointing
output = expensive_block(input_tensor)

# With checkpointing
output = checkpoint(expensive_block, input_tensor, use_reentrant=False)
```

### In a Module

```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return checkpoint(self.layers, x, use_reentrant=False)


class TransformerWithCheckpointing(nn.Module):
    def __init__(self, num_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([
            CheckpointedBlock(dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### Selective Checkpointing

```python
class SelectiveCheckpointModel(nn.Module):
    def __init__(self, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # Checkpoint heavy encoder, not decoder
        if self.use_checkpoint:
            encoded = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            encoded = self.encoder(x)

        # Decoder runs normally
        return self.decoder(encoded)
```

---

## checkpoint_sequential()

Convenience function for checkpointing sequential models.

### Function Signature

```python
def checkpoint_sequential(
    functions: nn.Sequential | list,  # Sequential modules
    segments: int,                     # Number of checkpoint segments
    input: Tensor,                     # Input tensor
    use_reentrant: bool = None,        # Required
    preserve_rng_state: bool = True
) -> Tensor
```

### Usage

```python
from torch.utils.checkpoint import checkpoint_sequential

model = nn.Sequential(
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)

# Divide into 4 segments, checkpoint each
# Only segment boundaries store activations
output = checkpoint_sequential(
    model,
    segments=4,
    input=x,
    use_reentrant=False
)
```

### Optimal Segment Count

For a model with n layers:
- Optimal segments ≈ √n for O(√n) memory
- More segments = less memory, more recomputation
- Fewer segments = more memory, less recomputation

```python
import math

num_layers = 48
optimal_segments = int(math.sqrt(num_layers))  # ≈ 7
```

---

## RNG State Handling

Checkpointing must preserve random state for deterministic recomputation.

### How It Works

```python
# Forward pass
ctx.fwd_cpu_state = torch.get_rng_state()
ctx.fwd_device_states = get_device_states(*args)

# ... run forward ...

# Backward pass (recomputation)
with torch.random.fork_rng(devices=rng_devices):
    torch.set_rng_state(ctx.fwd_cpu_state)
    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
    # Recompute with same random state
    outputs = function(*inputs)
```

### Disabling RNG Preservation

If no randomness in checkpointed region:

```python
output = checkpoint(
    deterministic_function,
    input,
    use_reentrant=False,
    preserve_rng_state=False  # Skip RNG stashing
)
```

---

## Context Functions

Custom context managers for forward and recomputation phases.

### Example: Mixed Precision

```python
from torch.amp import autocast

def get_amp_contexts():
    """Different precision for forward vs recompute."""
    forward_ctx = autocast(device_type='cuda', dtype=torch.float16)
    recompute_ctx = autocast(device_type='cuda', dtype=torch.float32)
    return forward_ctx, recompute_ctx

output = checkpoint(
    function,
    input,
    use_reentrant=False,
    context_fn=get_amp_contexts
)
```

### Example: Profiling

```python
def get_profiling_contexts():
    """Profile recomputation separately."""
    forward_ctx = contextlib.nullcontext()
    recompute_ctx = torch.profiler.record_function("checkpoint_recompute")
    return forward_ctx, recompute_ctx
```

---

## Memory-Compute Tradeoff

### Theoretical Analysis

For a model with:
- `L` layers
- `A` activation memory per layer
- `C` compute time per layer

| Strategy | Memory | Compute |
|----------|--------|---------|
| No checkpointing | O(L × A) | O(L × C) |
| All layers checkpointed | O(A) | O(2 × L × C) |
| √L segments | O(√L × A) | O(1.5 × L × C) |

### Practical Guidelines

**When to use checkpointing:**
- Training very deep models (transformers, ResNets)
- Limited GPU memory
- Large batch sizes needed
- Activation memory dominates

**When NOT to use:**
- Inference (no gradients needed)
- Very short/shallow models
- Already memory-limited on parameters

---

## Common Patterns

### Transformer Block Checkpointing

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, use_checkpoint=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
```

### Every-N-Layers Checkpointing

```python
class EfficientTransformer(nn.Module):
    def __init__(self, num_layers, dim, checkpoint_every=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_layers)
        ])
        self.checkpoint_every = checkpoint_every

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.training and i % self.checkpoint_every == 0:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
```

### Selective Checkpointing by Memory

```python
def should_checkpoint(layer_idx, total_layers, memory_threshold=0.8):
    """Checkpoint based on GPU memory usage."""
    if not torch.cuda.is_available():
        return False

    memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

    # Checkpoint if memory usage high, or every N layers
    return memory_used > memory_threshold or layer_idx % 4 == 0
```

---

## Debugging

### Determinism Check

```python
# Check if recomputation matches forward
output = checkpoint(
    function,
    input,
    use_reentrant=False,
    determinism_check="default"  # Checks shapes, dtypes, devices
)
```

### Debug Mode

```python
# Print operator trace
with set_checkpoint_debug_enabled(True):
    output = checkpoint(
        function,
        input,
        use_reentrant=False,
        debug=True
    )
```

### Common Issues

**Issue: Gradients are None**
- Ensure inputs have `requires_grad=True`
- Check function returns differentiable tensors

**Issue: Non-deterministic recomputation**
- Check for global state changes
- Ensure RNG state preserved
- Avoid caching in the function

**Issue: Memory not reduced**
- Verify function is actually being checkpointed
- Check if eager garbage collection needed: `torch.cuda.empty_cache()`

---

## MLX Mapping

### MLX Approach

MLX uses lazy evaluation and compilation rather than explicit checkpointing:

```python
import mlx.core as mx

# MLX's lazy evaluation naturally manages memory
# Explicit checkpointing less common

# For memory-conscious training, use:
# 1. mx.compile() for graph optimization
# 2. Smaller batch sizes
# 3. Gradient accumulation

@mx.compile
def train_step(model, x, y):
    def loss_fn(params):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
    return loss, grads
```

### Conceptual Equivalent

If implementing checkpointing in MLX:

```python
def mlx_checkpoint(fn, *args):
    """Conceptual MLX checkpointing pattern."""
    # Forward: compute but don't keep intermediate graph
    with mx.stop_gradient():  # Conceptual - MLX may differ
        output = fn(*args)

    # For backward: recompute
    # This would be handled by custom vjp implementation
    return output
```

### Key Differences

| PyTorch | MLX |
|---------|-----|
| Explicit `checkpoint()` | Lazy evaluation handles memory |
| Eager execution by default | Lazy by default |
| Manual memory management | Automatic with `mx.eval()` |
| RNG state stashing | Handled by key-based RNG |

---

## Best Practices

1. **Always use `use_reentrant=False`** - Modern, more compatible

2. **Checkpoint at natural boundaries** - Transformer blocks, residual blocks

3. **Test memory savings** - Profile before/after to verify

4. **Balance segments** - Too many = excessive recompute, too few = little memory savings

5. **Disable for inference** - Only needed during training

6. **Preserve RNG state** - Unless function is deterministic

7. **Use with gradient accumulation** - Combines well for large effective batch sizes

```python
# Combined pattern: checkpointing + gradient accumulation
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with torch.autocast(device_type='cuda'):
        output = checkpoint(model, batch, use_reentrant=False)
        loss = criterion(output, target) / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Summary

| Function | Purpose |
|----------|---------|
| `checkpoint(fn, *args)` | Checkpoint single function call |
| `checkpoint_sequential(seq, n, x)` | Checkpoint Sequential in n segments |
| `set_checkpoint_debug_enabled(bool)` | Global debug toggle |
| `set_checkpoint_early_stop(bool)` | Control early stopping |

Memory savings scale with:
- Model depth (more layers = more savings)
- Activation size (larger features = more savings)
- Segment strategy (√n segments optimal)
