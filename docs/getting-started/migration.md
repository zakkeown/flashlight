# Migration from PyTorch

Flashlight is designed to be a drop-in replacement for PyTorch on Apple Silicon. This guide covers the key differences and migration steps.

## Import Changes

Replace PyTorch imports with Flashlight equivalents:

```python
# Before (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim

# After (Flashlight)
import flashlight as fl
import flashlight.nn as nn
import flashlight.optim as optim
```

## Key Differences

### 1. Device Management

Flashlight uses MLX's unified memory model. Device placement is simplified:

```python
# PyTorch
x = torch.randn(3, 3).to("mps")

# Flashlight (no explicit device needed)
x = fl.randn(3, 3)  # Automatically uses Apple Silicon
```

### 2. Data Types

MLX doesn't support float64. Use float32 instead:

```python
# PyTorch
x = torch.randn(3, 3, dtype=torch.float64)

# Flashlight
x = fl.randn(3, 3, dtype=fl.float32)  # float64 not available
```

### 3. In-Place Operations

MLX arrays are immutable. In-place operations create new tensors:

```python
# These work the same way in both, but Flashlight
# creates a new tensor internally
x.add_(1)
x.zero_()
```

### 4. Layout Conventions

Flashlight handles NCHW (PyTorch) to NHWC (MLX) conversion automatically for spatial operations like convolutions.

## API Compatibility

Most PyTorch APIs work directly:

| PyTorch | Flashlight | Notes |
|---------|------------|-------|
| `torch.Tensor` | `fl.Tensor` | Full compatibility |
| `nn.Module` | `nn.Module` | Same API |
| `optim.Adam` | `optim.Adam` | Same API |
| `autograd` | `autograd` | Tape-based |

## Example Migration

### PyTorch Code

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to("mps")

x = torch.randn(32, 784, device="mps")
output = model(x)
```

### Flashlight Code

```python
import flashlight as fl
import flashlight.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = fl.randn(32, 784)
output = model(x)
```

## What's Not Supported

- **float64**: Use float32 or float16
- **Distributed training**: Single GPU only
- **CUDA operations**: Apple Silicon only
- **Some specialized ops**: Check API reference for coverage
