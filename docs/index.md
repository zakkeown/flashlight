# Flashlight

**PyTorch-compatible API layer for Apple's MLX framework**

Flashlight enables PyTorch code to run on Apple Silicon with minimal modifications through a complete bottom-up implementation.

## Features

- **PyTorch API Compatibility**: Drop-in replacement for common PyTorch operations
- **Apple Silicon Optimized**: Built on MLX for native Metal acceleration
- **Complete Implementation**: Tensor operations, autograd, neural network layers, and optimizers
- **Numerical Parity**: Tested against PyTorch for accuracy

## Quick Start

```python
import flashlight as fl
import flashlight.nn as nn

# Create tensors (PyTorch-compatible API)
x = fl.randn(32, 784)

# Build neural networks
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Forward pass with autograd
output = model(x)
```

## Installation

```bash
pip install flashlight
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- MLX 0.20.0+

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Migration from PyTorch](getting-started/migration.md)
- [API Reference](reference/)

## Status

Flashlight is in active development. The following components are implemented:

| Component | Status |
|-----------|--------|
| Tensor Core | ✅ Complete |
| Operators (50+) | ✅ Complete |
| Autograd | ✅ Complete |
| NN Modules | ✅ Complete |
| Optimizers | ✅ Complete |
| Distributions | ✅ Complete |
| Signal Processing | ✅ Complete |
