# Flashlight: PyTorch-Compatible API for Apple MLX

[![CI](https://github.com/yourusername/flashlight/workflows/CI/badge.svg)](https://github.com/yourusername/flashlight/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-compatible API layer for Apple's MLX framework, enabling PyTorch code to run on Apple Silicon with minimal modifications.

## Features

- **Tensor Core**: Full tensor operations with PyTorch-compatible API
- **50+ Operators**: Arithmetic, activations, reductions, convolutions, and more
- **Automatic Differentiation**: Tape-based autograd system
- **Neural Network Modules**: 15+ layer types (Linear, Conv2d, LSTM, BatchNorm, etc.)
- **Optimizers**: SGD, Adam, AdamW with learning rate schedulers
- **Distributions**: Statistical probability distributions
- **Signal Processing**: FFT and window functions

## Project Goals

- **PyTorch Compatibility**: ~90% API compatibility for common ML workflows
- **Numerical Parity**: <1e-5 error on forward pass, <1e-4 on gradients
- **Performance**: Within 20% of PyTorch MPS backend performance
- **Test Coverage**: >90% code coverage with comprehensive testing

## Quick Start

```python
import flashlight

# Create tensors (PyTorch-like API)
x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

# Automatic differentiation
loss = y.sum()
loss.backward()
print(x.grad)  # [2.0, 2.0, 2.0]

# Neural networks
model = flashlight.nn.Sequential(
    flashlight.nn.Linear(784, 256),
    flashlight.nn.ReLU(),
    flashlight.nn.Linear(256, 10),
)

# Training
optimizer = flashlight.optim.Adam(model.parameters(), lr=1e-3)
criterion = flashlight.nn.CrossEntropyLoss()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
```

## Installation

### Prerequisites
- Python 3.9+
- macOS (Apple Silicon recommended for MLX)
- MLX 0.20.0+

### Install from source
```bash
git clone https://github.com/yourusername/flashlight.git
cd flashlight
pip install -r requirements.txt
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Architecture

Flashlight follows a layered, bottom-up design:

```
┌─────────────────────────────────────┐
│  Python API (torch-like interface)  │
├─────────────────────────────────────┤
│      Autograd (tape-based)          │
├─────────────────────────────────────┤
│    Operators (50+ operations)       │
├─────────────────────────────────────┤
│     Tensor Core (wrappers)          │
├─────────────────────────────────────┤
│         MLX Backend                 │
└─────────────────────────────────────┘
```

### Key Design Decisions

1. **Layout Conversion**: Handle NCHW (PyTorch) ↔ NHWC (MLX) transparently
2. **Immutability**: Implement in-place operations as copy + functional op
3. **Unified Memory**: Device management is compatibility shim (MLX uses unified memory)
4. **Tape-based Autograd**: Build PyTorch-style tape system on top of MLX transforms

## Testing

We use **Test-Driven Development (TDD)** with comprehensive numerical parity testing:

```bash
# Run all tests
pytest tests/

# Run specific component tests
pytest tests/tensor_core/
pytest tests/operators/
pytest tests/autograd/
pytest tests/nn_modules/

# Run with coverage
pytest tests/ --cov=flashlight --cov-report=html

# Run integration tests
pytest tests/integration/ -m integration

# Run parity tests (compare against PyTorch)
pytest tests/ -m parity
```

### Test Organization
```
tests/
├── common_utils.py       # Shared utilities
├── tensor_core/          # Tensor operations
├── operators/            # Mathematical operators
├── autograd/             # Automatic differentiation
├── nn_modules/           # Neural network layers
├── training/             # Optimizers and schedulers
├── distributions/        # Probability distributions
├── signal/               # Signal processing
└── integration/          # End-to-end tests
```

## Examples

```
examples/
├── mnist_mlp.py          # MLP training on MNIST
├── lenet_cifar10.py      # CNN training on CIFAR-10
├── resnet.py             # ResNet model
└── transformer.py        # Transformer model
```

## Development

### Code Style
- **Formatter**: Black (line length: 100)
- **Import sorting**: isort
- **Linter**: flake8

```bash
# Format code
black flashlight tests
isort flashlight tests

# Lint
flake8 flashlight tests
```

### Contributing
1. Follow TDD: Write tests first
2. Ensure numerical parity with PyTorch
3. Maintain >90% code coverage
4. Follow existing code style

## Known Limitations

### MLX Constraints
- **No float64**: MLX only supports float16/float32/bfloat16
- **Immutability**: In-place operations require workarounds
- **Apple Silicon only**: MLX requires macOS with Metal
- **Single GPU**: No distributed training support

### Layout Differences
- PyTorch uses NCHW (channels-first) for convolutions
- MLX uses NHWC (channels-last) for Metal optimization
- Automatic transpose operations may impact performance

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch Team**: For the reference implementation and design patterns
- **Apple MLX Team**: For the high-performance ML framework

## Contact

For questions, issues, or contributions:
- GitHub Issues: [https://github.com/yourusername/flashlight/issues](https://github.com/yourusername/flashlight/issues)
