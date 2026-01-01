# Installation

## Requirements

- **macOS** with Apple Silicon (M1, M2, M3, or newer)
- **Python 3.9** or higher
- **MLX 0.20.0** or higher

!!! warning "Apple Silicon Only"
    Flashlight requires Apple Silicon for MLX support. Intel Macs are not supported.

## Install from PyPI

```bash
pip install flashlight
```

## Install from Source

For development or the latest features:

```bash
git clone https://github.com/flashlight/flashlight.git
cd flashlight
pip install -e ".[dev]"
```

## Verify Installation

```python
import flashlight as fl

# Check version
print(f"Flashlight version: {fl.__version__}")

# Create a simple tensor
x = fl.randn(3, 3)
print(x)
```

## Optional: PyTorch for Parity Testing

If you want to run parity tests against PyTorch:

```bash
pip install torch
```

This is only needed for development and testing, not for using Flashlight.
