# Best Practices

This guide covers best practices for getting the most out of Flashlight, including performance optimization, MLX-specific considerations, debugging tips, and PyTorch compatibility strategies.

## Performance Optimization

### Batch Size Selection

Larger batch sizes generally improve throughput on Apple Silicon:

```python
# Start with larger batches and tune down if needed
batch_sizes_to_try = [256, 128, 64, 32]

for batch_size in batch_sizes_to_try:
    try:
        train_with_batch_size(batch_size)
        break
    except MemoryError:
        continue
```

**Guidelines:**
- Start with batch size 128-256 for image classification
- Use gradient accumulation if memory-limited
- Larger batches benefit from MLX's Metal optimization

### Avoiding Unnecessary Operations

```python
# Avoid: Creating intermediate tensors
x = flashlight.tensor([1, 2, 3])
y = x + 1
z = y * 2
w = z - 1

# Prefer: Chain operations
x = flashlight.tensor([1, 2, 3])
w = (x + 1) * 2 - 1
```

### Efficient Data Loading

```python
# Pre-convert data to tensors once
train_x = flashlight.tensor(train_data)
train_y = flashlight.tensor(train_labels)

# Avoid converting in the loop
for i in range(0, len(train_x), batch_size):
    batch_x = train_x[i:i+batch_size]  # Slicing is efficient
    batch_y = train_y[i:i+batch_size]
    ...
```

### Contiguous Memory

```python
# After transpose/permute, make contiguous for efficient operations
x = x.permute(0, 2, 1).contiguous()

# Check if contiguous
if not x.is_contiguous():
    x = x.contiguous()
```

## MLX-Specific Considerations

### Unified Memory

MLX uses Apple's unified memory architecture, eliminating explicit CPU-GPU transfers:

```python
# No need for .cuda() or .to('mps')
model = MyModel()  # Automatically uses unified memory
x = flashlight.randn(32, 3, 224, 224)
output = model(x)  # No transfer needed
```

**Benefits:**
- No data transfer overhead
- Larger effective memory (shared with system RAM)
- Simpler code

### Layout Handling

MLX prefers NHWC layout while PyTorch uses NCHW. Flashlight handles this automatically:

```python
# Your code uses NCHW (PyTorch convention)
x = flashlight.randn(32, 3, 224, 224)  # N, C, H, W
conv = nn.Conv2d(3, 64, kernel_size=3)
output = conv(x)  # Works correctly

# Flashlight internally converts to NHWC for MLX
# You don't need to worry about this
```

**Tip:** If doing manual layout changes, be aware of the conversion:
```python
# If you need explicit NHWC
x_nhwc = x.permute(0, 2, 3, 1)  # N, H, W, C
```

### Immutability Patterns

MLX arrays are immutable. Flashlight simulates in-place operations:

```python
# These work but create internal copies
x.add_(y)      # x = x + y
x.mul_(2)      # x = x * 2
x.zero_()      # x = zeros_like(x)

# For performance-critical code, prefer functional style
x = x + y      # Explicit allocation
x = x * 2
```

### Lazy Evaluation

MLX uses lazy evaluation. Operations are only executed when needed:

```python
# These operations are lazy
x = flashlight.randn(1000, 1000)
y = x @ x.T
z = y.sum()

# Evaluation happens here
result = z.item()  # Forces computation
```

**Tip:** For benchmarking, ensure computation is complete:
```python
import mlx.core as mx

start = time.time()
output = model(x)
mx.eval(output._mlx_array)  # Force evaluation
end = time.time()
```

## Numerical Precision

### Float32 vs Float16

```python
# Default: float32
x = flashlight.randn(3, 4)  # float32

# Half precision for memory savings
x_half = flashlight.randn(3, 4, dtype=flashlight.float16)

# Mixed precision pattern
model.half()  # Convert model to float16
x = x.half()  # Convert input
output = model(x)
```

### Handling Float64 Code

MLX doesn't support float64. Convert PyTorch code:

```python
# PyTorch (may use float64)
x = torch.tensor([1.0, 2.0], dtype=torch.float64)

# Flashlight (use float32)
x = flashlight.tensor([1.0, 2.0], dtype=flashlight.float32)
```

### Numerical Stability

```python
# Use log-sum-exp for stability
def log_softmax(x):
    max_x = x.max(dim=-1, keepdim=True).values
    return x - max_x - flashlight.log(flashlight.exp(x - max_x).sum(dim=-1, keepdim=True))

# Use built-in functions when available
output = flashlight.nn.functional.log_softmax(x, dim=-1)
```

## Debugging

### Shape Debugging

```python
def debug_forward(self, x):
    print(f"Input: {x.shape}")
    x = self.conv1(x)
    print(f"After conv1: {x.shape}")
    x = self.pool(x)
    print(f"After pool: {x.shape}")
    return x
```

### Gradient Debugging

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name}:")
            print(f"  shape: {grad.shape}")
            print(f"  min: {grad.min().item():.6f}")
            print(f"  max: {grad.max().item():.6f}")
            print(f"  mean: {grad.mean().item():.6f}")
            print(f"  norm: {grad.norm().item():.6f}")
            if flashlight.isnan(grad).any():
                print("  WARNING: Contains NaN!")
            if flashlight.isinf(grad).any():
                print("  WARNING: Contains Inf!")
```

### NaN Detection

```python
def check_for_nan(tensor, name="tensor"):
    if flashlight.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if flashlight.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

# In training loop
output = model(x)
check_for_nan(output, "model output")
loss = criterion(output, y)
check_for_nan(loss, "loss")
```

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| Shape mismatch | Wrong tensor dimensions | Check input shapes, use `.shape` |
| dtype mismatch | Mixed float32/float16 | Ensure consistent dtypes |
| Gradient is None | Parameter not used in forward | Check computation graph |
| Memory error | Batch too large | Reduce batch size |

## Code Organization

### Model Definition

```python
# Separate model definition from training code
# models/cnn.py
import flashlight.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.features = self._make_features()
        self.classifier = self._make_classifier(num_classes, dropout)

    def _make_features(self):
        return nn.Sequential(...)

    def _make_classifier(self, num_classes, dropout):
        return nn.Sequential(...)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Configuration Management

```python
# Use dataclasses or dictionaries for config
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 0.01
    weight_decay: float = 1e-4

config = TrainingConfig()
optimizer = flashlight.optim.Adam(model.parameters(), lr=config.lr)
```

### Reproducibility

```python
def set_seed(seed):
    flashlight.manual_seed(seed)
    # numpy if used
    import numpy as np
    np.random.seed(seed)

# Set seed before training
set_seed(42)
```

## PyTorch Compatibility Tips

### What Transfers Directly

These patterns work identically in Flashlight:

```python
# Module definition
class MyModel(nn.Module):
    ...

# Training loop
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Model serialization
flashlight.save(model.state_dict(), 'model.pth')
model.load_state_dict(flashlight.load('model.pth'))
```

### Common Modifications

```python
# PyTorch                          # Flashlight
# ---------                        # ----------
model.cuda()                       # (not needed)
x = x.to('cuda')                   # (not needed)
torch.cuda.synchronize()           # mx.eval(...)
torch.float64                      # flashlight.float32
torch.autocast('cuda')             # (use dtype directly)
```

### Conditional Code

For code that needs to work with both:

```python
try:
    import flashlight as torch_compat
    USING_FLASHLIGHT = True
except ImportError:
    import torch as torch_compat
    USING_FLASHLIGHT = False

# Use torch_compat for compatible operations
x = torch_compat.randn(3, 4)

if not USING_FLASHLIGHT:
    x = x.cuda()  # Only for PyTorch
```

### Testing Parity

```python
def test_layer_parity():
    import torch
    import flashlight

    # Create same input
    np_input = np.random.randn(32, 64).astype(np.float32)
    torch_input = torch.tensor(np_input)
    flash_input = flashlight.tensor(np_input)

    # Create layers with same weights
    torch_layer = torch.nn.Linear(64, 32)
    flash_layer = flashlight.nn.Linear(64, 32)

    # Copy weights
    flash_layer.weight.data = flashlight.tensor(
        torch_layer.weight.detach().numpy()
    )
    flash_layer.bias.data = flashlight.tensor(
        torch_layer.bias.detach().numpy()
    )

    # Compare outputs
    torch_out = torch_layer(torch_input).detach().numpy()
    flash_out = flash_layer(flash_input).numpy()

    np.testing.assert_allclose(torch_out, flash_out, rtol=1e-5, atol=1e-6)
```

## Summary Checklist

### Before Training
- [ ] Set random seed for reproducibility
- [ ] Verify model architecture with a test forward pass
- [ ] Check input data shapes and dtypes
- [ ] Start with a small subset to verify pipeline

### During Training
- [ ] Monitor loss for NaN/Inf
- [ ] Check gradient norms periodically
- [ ] Save checkpoints regularly
- [ ] Log learning rate if using schedulers

### Performance
- [ ] Use appropriate batch size
- [ ] Avoid unnecessary tensor copies
- [ ] Keep data as tensors (avoid repeated conversions)
- [ ] Use float16 if memory-limited

### Debugging
- [ ] Print shapes at each layer for shape issues
- [ ] Check gradients for vanishing/exploding
- [ ] Verify weights are being updated
- [ ] Compare against PyTorch for parity issues
