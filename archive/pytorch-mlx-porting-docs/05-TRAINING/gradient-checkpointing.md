# Gradient Checkpointing (torch.utils.checkpoint)

## Purpose

Gradient checkpointing (also called activation checkpointing or rematerialization) is a memory optimization technique that trades compute for memory during training. Instead of storing all intermediate activations for backward pass, it recomputes them on-demand during backpropagation.

**Key Benefits:**
- Reduces memory usage significantly (often 2-10x)
- Enables training of larger models
- Enables larger batch sizes
- Critical for training LLMs and large vision models

**Reference Files:**
- `torch/utils/checkpoint.py` - Main implementation (~1,700 lines)

---

## Overview

### The Memory Problem

During forward pass, PyTorch stores activations for gradient computation:

```python
# Without checkpointing:
# Memory = O(layers × batch_size × hidden_dim)

def forward(x):
    a = layer1(x)      # Store a for backward
    b = layer2(a)      # Store b for backward
    c = layer3(b)      # Store c for backward
    d = layer4(c)      # Store d for backward
    return d

# For a 100-layer model with large activations,
# memory can be prohibitive
```

### Checkpointing Solution

```python
# With checkpointing:
# Memory = O(sqrt(layers) × batch_size × hidden_dim)

from torch.utils.checkpoint import checkpoint

def forward(x):
    # Only store x, recompute a, b during backward
    ab = checkpoint(lambda x: layer2(layer1(x)), x, use_reentrant=False)

    # Only store ab, recompute c, d during backward
    cd = checkpoint(lambda x: layer4(layer3(x)), ab, use_reentrant=False)

    return cd
```

### Memory vs Compute Tradeoff

| Approach | Memory | Compute |
|----------|--------|---------|
| No checkpointing | O(n) activations | 1× forward, 1× backward |
| Full checkpointing | O(1) activations | 2× forward, 1× backward |
| Segmented checkpointing | O(√n) activations | ~1.5× forward, 1× backward |

---

## Core API

### checkpoint (Main Function)

**Purpose**: Checkpoint a model or part of the model by recomputing during backward.

**Signature**:
```python
torch.utils.checkpoint.checkpoint(
    function: Callable,
    *args,
    use_reentrant: bool,
    preserve_rng_state: bool = True,
    context_fn: Callable = noop_context_fn,
    determinism_check: str = "default",
    debug: bool = False,
    early_stop: bool = True,
    **kwargs
) -> Any
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `function` | Callable | Forward function to checkpoint |
| `*args` | Any | Positional arguments to function |
| `use_reentrant` | bool | Reentrant vs non-reentrant mode (required) |
| `preserve_rng_state` | bool | Save/restore RNG state |
| `context_fn` | Callable | Custom context managers for forward/recompute |
| `determinism_check` | str | Check recomputation matches forward |
| `debug` | bool | Print debug trace information |
| `early_stop` | bool | Stop recomputation early when possible |
| `**kwargs` | Any | Keyword arguments (non-reentrant only) |

**Basic Usage**:
```python
import torch
from torch.utils.checkpoint import checkpoint

def transformer_block(x, attention, ffn):
    x = x + attention(x)
    x = x + ffn(x)
    return x

# Checkpoint each transformer block
x = input_tensor
for i, block in enumerate(transformer_blocks):
    x = checkpoint(
        transformer_block,
        x, block.attention, block.ffn,
        use_reentrant=False
    )
```

---

### Reentrant vs Non-Reentrant Mode

**Recommended: `use_reentrant=False`**

| Feature | Reentrant (`True`) | Non-Reentrant (`False`) |
|---------|-------------------|------------------------|
| Backward API | Only `backward()` | `backward()` and `grad()` |
| Keyword args | Not supported | Supported |
| Nested structures | Not tracked | Fully tracked |
| Detached tensors | Not supported | Supported |
| Early stop | No | Yes |
| Debug mode | No | Yes |
| Context managers | No | Yes |
| requires_grad | At least one I/O | No requirement |

**Non-Reentrant Example**:
```python
# Recommended approach
output = checkpoint(
    model.forward,
    input_tensor,
    use_reentrant=False,
    preserve_rng_state=True,
)

# Supports keyword arguments
output = checkpoint(
    model.forward,
    input_tensor,
    use_reentrant=False,
    attention_mask=mask,
    position_ids=positions,
)
```

**Reentrant Example** (legacy):
```python
# Legacy approach - avoid if possible
output = checkpoint(
    model.forward,
    input_tensor,
    use_reentrant=True,
)
```

---

### checkpoint_sequential (For Sequential Models)

**Purpose**: Checkpoint a sequential model by dividing it into segments.

**Signature**:
```python
torch.utils.checkpoint.checkpoint_sequential(
    functions: nn.Sequential | List[Callable],
    segments: int,
    input: Tensor,
    use_reentrant: bool,
    preserve_rng_state: bool = True,
) -> Tensor
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `functions` | Sequential/List | List of modules to run sequentially |
| `segments` | int | Number of checkpoint segments |
| `input` | Tensor | Input tensor |
| `use_reentrant` | bool | Reentrant mode |

**Usage Example**:
```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

# Create a deep sequential model
model = nn.Sequential(
    *[nn.Linear(512, 512) for _ in range(100)]
)

# Divide into 10 checkpointed segments
# Each segment recomputes 10 layers during backward
output = checkpoint_sequential(
    model,
    segments=10,
    input=x,
    use_reentrant=False,
)
```

**How It Works**:
```
Input → [Segment 1: 10 layers] → [Segment 2: 10 layers] → ... → Output
         ↑ Checkpoint            ↑ Checkpoint
         (recompute on backward) (recompute on backward)
```

---

## Advanced Features

### Custom Context Managers

Control the execution context during forward and recomputation:

```python
def my_context_fn():
    """Return (forward_context, recompute_context)"""
    return (
        torch.cuda.amp.autocast(dtype=torch.float16),
        torch.cuda.amp.autocast(dtype=torch.float16),
    )

output = checkpoint(
    model.forward,
    x,
    use_reentrant=False,
    context_fn=my_context_fn,
)
```

### Determinism Checking

Verify that recomputation produces identical results:

```python
output = checkpoint(
    model.forward,
    x,
    use_reentrant=False,
    determinism_check="default",  # Check shapes, dtypes, devices
    # determinism_check="none",   # Disable checking
)
```

### Debug Mode

Print detailed traces for debugging:

```python
output = checkpoint(
    model.forward,
    x,
    use_reentrant=False,
    debug=True,  # Prints operator traces
)
```

### Early Stop Control

Control whether recomputation stops early:

```python
from torch.utils.checkpoint import set_checkpoint_early_stop

# Disable early stop globally
with set_checkpoint_early_stop(False):
    output = checkpoint(fn, x, use_reentrant=False)

# Or per-call
output = checkpoint(
    fn, x,
    use_reentrant=False,
    early_stop=False,
)
```

---

## How It Works Internally

### Forward Pass (with checkpointing)

```python
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # 1. Save metadata (not activations)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # 2. Save RNG state for reproducibility
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # 3. Save only input tensors
        ctx.save_for_backward(*[a for a in args if torch.is_tensor(a)])

        # 4. Run forward WITHOUT gradient tracking
        with torch.no_grad():
            outputs = run_function(*args)

        return outputs
```

### Backward Pass (with recomputation)

```python
    @staticmethod
    def backward(ctx, *grad_outputs):
        # 1. Restore inputs
        inputs = ctx.saved_tensors

        # 2. Restore RNG state
        with torch.random.fork_rng(devices=ctx.fwd_devices):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                set_device_states(ctx.fwd_devices, ctx.fwd_device_states)

            # 3. RECOMPUTE forward pass (with gradients)
            with torch.enable_grad():
                outputs = ctx.run_function(*detach_variable(inputs))

        # 4. Compute gradients on recomputed graph
        torch.autograd.backward(outputs, grad_outputs)

        # 5. Return gradients
        return tuple(inp.grad for inp in inputs)
```

### Memory During Forward

```
Without Checkpointing:
┌─────────────────────────────────────────────────────────────┐
│ Input │ Act1 │ Act2 │ Act3 │ ... │ ActN │ Output            │
└─────────────────────────────────────────────────────────────┘
Memory: O(N) activations

With Checkpointing (segments):
┌───────────────────────────────────────────────┐
│ Input │ Seg1_out │ Seg2_out │ ... │ Output    │
└───────────────────────────────────────────────┘
Memory: O(√N) activations (segment boundaries only)
```

### Memory During Backward

```
Step 1: Recompute Segment K
┌───────────────────────────────────────────────────┐
│ SegK_in │ SegK_Act1 │ SegK_Act2 │ ... │ SegK_out │
└───────────────────────────────────────────────────┘

Step 2: Compute gradients for Segment K, then free

Step 3: Recompute Segment K-1, repeat...
```

---

## Nested Checkpointing

Checkpoints can be nested with well-defined semantics:

```python
def outer_fn(x):
    y = layer1(x)

    # Inner checkpoint
    z = checkpoint(inner_fn, y, use_reentrant=False)

    return layer2(z)

def inner_fn(y):
    return expensive_op(y)

# Outer checkpoint
output = checkpoint(outer_fn, input, use_reentrant=False)
```

**Rules for Nested Checkpointing**:
1. Saved tensors are managed by innermost checkpoint only
2. Inputs to inner checkpoints are saved by parent checkpoint
3. To recompute any saved tensor, all wrapping checkpoints must recompute

---

## Common Patterns

### Checkpointing Transformer Layers

```python
class CheckpointedTransformer(nn.Module):
    def __init__(self, layers, checkpoint_every=1):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_every = checkpoint_every

    def forward(self, x, attention_mask=None):
        for i, layer in enumerate(self.layers):
            if self.training and i % self.checkpoint_every == 0:
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                    attention_mask=attention_mask,
                )
            else:
                x = layer(x, attention_mask=attention_mask)
        return x
```

### Selective Checkpointing

Checkpoint only expensive layers:

```python
def forward(self, x):
    # Cheap layers - don't checkpoint
    x = self.embedding(x)
    x = self.positional_encoding(x)

    # Expensive layers - checkpoint
    for layer in self.transformer_layers:
        x = checkpoint(layer, x, use_reentrant=False)

    # Final layers - don't checkpoint (need grads immediately)
    return self.output_projection(x)
```

### With Mixed Precision Training

```python
from torch.cuda.amp import autocast

def amp_context_fn():
    return autocast(), autocast()

# Checkpointing works with AMP
with autocast():
    output = checkpoint(
        model.forward,
        x,
        use_reentrant=False,
        context_fn=amp_context_fn,
    )
```

### Gradient Accumulation

```python
# Checkpointing is orthogonal to gradient accumulation
accumulation_steps = 4

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    # Checkpointed forward
    output = checkpoint(model, batch, use_reentrant=False)
    loss = criterion(output, target) / accumulation_steps

    # Backward (triggers recomputation)
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Memory Savings Analysis

### Theoretical Savings

For a model with N layers:
- Without checkpointing: O(N) activation memory
- With checkpointing every layer: O(1) activation memory, 2× compute
- With √N segments: O(√N) activation memory, ~1.5× compute

### Practical Example

```python
# 24-layer transformer, hidden_dim=4096, batch_size=32, seq_len=2048

# Without checkpointing (estimated):
# 24 layers × 32 batch × 2048 seq × 4096 dim × 4 bytes = ~24 GB activations

# With checkpointing every 4 layers (6 segments):
# 6 boundaries × 32 × 2048 × 4096 × 4 = ~6 GB activations
# Compute overhead: ~33% (recompute 4 layers per segment)

# With checkpointing every layer:
# 1 × 32 × 2048 × 4096 × 4 = ~1 GB activations
# Compute overhead: ~100% (recompute every layer)
```

---

## RNG State Preservation

Checkpointing preserves RNG state for reproducibility:

```python
# Operations with randomness (dropout, etc.) will produce
# identical results during recomputation

def forward_with_dropout(x):
    x = self.layer1(x)
    x = F.dropout(x, p=0.1, training=self.training)  # Uses RNG
    x = self.layer2(x)
    return x

# RNG state saved during forward, restored during recompute
output = checkpoint(
    forward_with_dropout,
    x,
    use_reentrant=False,
    preserve_rng_state=True,  # Default
)
```

---

## MLX Porting Considerations

### MLX's Lazy Evaluation

MLX uses lazy evaluation which may provide automatic optimization:

```python
import mlx.core as mx

# MLX's lazy evaluation may reduce memory pressure automatically
# by not materializing intermediate results until needed

def forward(x):
    a = layer1(x)  # Not computed yet
    b = layer2(a)  # Not computed yet
    c = layer3(b)  # Not computed yet
    return c       # Computed when evaluated

# mx.eval() or accessing values triggers computation
result = forward(x)
mx.eval(result)
```

### Manual Checkpointing for MLX

If explicit checkpointing is needed:

```python
import mlx.core as mx

def checkpoint_mlx(fn, *args):
    """Simple checkpointing for MLX"""

    # Forward: compute and return output
    output = fn(*args)

    # Store function and inputs for recomputation
    # (MLX doesn't have autograd.Function, use different approach)

    return output

# Alternative: Use mx.checkpoint if available
# Check MLX documentation for native support
```

### Key Differences from PyTorch

1. **Lazy Evaluation**: MLX's lazy eval may reduce need for explicit checkpointing
2. **Unified Memory**: Apple Silicon's unified memory reduces CPU-GPU transfer overhead
3. **No RNG State API**: May need different approach for RNG preservation
4. **Autograd Differences**: MLX's grad system differs from PyTorch's

### Potential MLX Implementation

```python
import mlx.core as mx
import mlx.nn as nn

class CheckpointedLayer(nn.Module):
    """Checkpointing wrapper for MLX"""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def __call__(self, x):
        if self.training:
            # During training, don't store intermediate activations
            # Rely on MLX's value_and_grad for recomputation
            return self.layer(x)
        else:
            return self.layer(x)

def train_step_with_checkpoint(model, x, y, loss_fn):
    """Training step that leverages MLX's lazy evaluation"""

    def loss_and_forward(params, x, y):
        model.update(params)
        pred = model(x)
        return loss_fn(pred, y)

    # MLX's value_and_grad recomputes as needed
    loss, grads = mx.value_and_grad(loss_and_forward)(model.parameters(), x, y)

    return loss, grads
```

---

## Common Issues and Solutions

### Issue: Gradients are None

```python
# Problem: No input has requires_grad=True
x = torch.randn(10, 10)  # requires_grad=False by default

# Solution
x = torch.randn(10, 10, requires_grad=True)
output = checkpoint(fn, x, use_reentrant=False)
```

### Issue: In-place Operations

```python
# Problem: In-place ops can cause issues
def bad_fn(x):
    x.relu_()  # In-place!
    return x

# Solution: Use out-of-place operations
def good_fn(x):
    return x.relu()  # Returns new tensor
```

### Issue: Non-deterministic Recomputation

```python
# Problem: Global state changes between forward and backward
global_counter = 0

def bad_fn(x):
    global global_counter
    global_counter += 1
    return x * global_counter  # Different result during recompute!

# Solution: Make function pure
def good_fn(x, counter):
    return x * counter
```

---

## Summary

### When to Use Checkpointing

| Scenario | Recommendation |
|----------|----------------|
| GPU OOM during training | Use checkpointing |
| Want larger batch size | Use checkpointing |
| Training large models (LLMs) | Essential |
| Inference only | Not needed |
| Training small models | Usually not needed |

### API Quick Reference

| Function | Purpose |
|----------|---------|
| `checkpoint(fn, *args, use_reentrant=False)` | Checkpoint a function |
| `checkpoint_sequential(model, segments, input)` | Checkpoint sequential model |
| `set_checkpoint_debug_enabled(True)` | Enable debug output |
| `set_checkpoint_early_stop(False)` | Disable early stop |

### Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `use_reentrant` | `False` | Required, use non-reentrant |
| `preserve_rng_state` | `True` (default) | For reproducibility |
| `early_stop` | `True` (default) | Better performance |
| `determinism_check` | `"default"` | Catch bugs |
