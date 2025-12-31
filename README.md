# MLX Compat: PyTorch-Compatible API for Apple MLX

[![CI](https://github.com/yourusername/flashlight/workflows/CI/badge.svg)](https://github.com/yourusername/flashlight/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-compatible API layer for Apple's MLX framework, enabling PyTorch code to run on Apple Silicon with minimal modifications.

## 🎯 Project Goals

- **PyTorch Compatibility**: ~90% API compatibility for common ML workflows
- **Numerical Parity**: <1e-5 error on forward pass, <1e-4 on gradients
- **Performance**: Within 20% of PyTorch MPS backend performance
- **Test Coverage**: >90% code coverage with comprehensive testing

## 🚀 Quick Start

```python
import mlx_compat

# Create tensors (PyTorch-like API)
x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2 + 1

# Automatic differentiation
loss = y.sum()
loss.backward()
print(x.grad)  # [2.0, 2.0, 2.0]

# Neural networks
model = mlx_compat.nn.Sequential(
    mlx_compat.nn.Linear(784, 256),
    mlx_compat.nn.ReLU(),
    mlx_compat.nn.Linear(256, 10),
)

# Training
optimizer = mlx_compat.optim.Adam(model.parameters(), lr=1e-3)
criterion = mlx_compat.nn.CrossEntropyLoss()

for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
```

## 📦 Installation

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

## 🗺️ Implementation Roadmap

This project follows a **6-phase bottom-up implementation** strategy:

### ✅ Phase 0: Project Scaffolding
- [x] Package structure
- [x] Testing infrastructure
- [x] CI/CD pipeline
- [x] Documentation

### ✅ Phase 1: Tensor Core
- [x] Tensor wrapper class
- [x] Dtype system mapping
- [x] Device management
- [x] View semantics
- [x] 20+ tensor creation functions

### ✅ Phase 2: Operators
- [x] 50+ Tier 1 operators
  - Arithmetic (12): add, sub, mul, div, matmul, etc.
  - Activations (7): relu, gelu, sigmoid, tanh, softmax, etc.
  - Reductions (8): sum, mean, max, min, argmax, etc.
  - Shape (8): cat, stack, split, reshape, etc.
  - Indexing (5): gather, scatter, where, etc.
  - Convolution (3): conv1d, conv2d, conv3d
  - Other (7): embedding, dropout, layer_norm, etc.

### ✅ Phase 3: Autograd
- [x] Tape-based computational graph
- [x] Backward pass engine
- [x] Gradient formulas for all operators
- [x] Custom autograd functions
- [x] Numerical gradient checking

### ✅ Phase 4: Neural Networks
- [x] nn.Module base class
- [x] 15+ layer types
  - Linear, Conv2d, BatchNorm, LayerNorm
  - ReLU, GELU, Dropout, MaxPool2d
  - RNN, LSTM, GRU
- [x] Loss functions
- [x] Model serialization

### ✅ Phase 5: Training Infrastructure
- [x] Optimizers: SGD, Adam, AdamW
- [x] Learning rate schedulers
- [x] Gradient clipping
- [x] Checkpointing

### ✅ Phase 6: Validation
- [x] Reference models (ResNet, Transformer)
- [x] MNIST/CIFAR-10 training
- [x] Numerical parity validation
- [x] Performance benchmarking

## 🏗️ Architecture

### Bottom-Up Design

```
┌─────────────────────────────────────┐
│  Python API (torch-like interface) │  Phase 4-5
├─────────────────────────────────────┤
│      Autograd (tape-based)          │  Phase 3
├─────────────────────────────────────┤
│    Operators (50 Tier 1 ops)        │  Phase 2
├─────────────────────────────────────┤
│     Tensor Core (wrappers)          │  Phase 1
├─────────────────────────────────────┤
│         MLX Backend                 │  External
└─────────────────────────────────────┘
```

### Key Design Decisions

1. **Layout Conversion**: Handle NCHW (PyTorch) ↔ NHWC (MLX) transparently
2. **Immutability**: Implement in-place operations as copy + functional op
3. **Unified Memory**: Device management is compatibility shim (MLX uses unified memory)
4. **Tape-based Autograd**: Build PyTorch-style tape system on top of MLX transforms

## 🧪 Testing

We use **Test-Driven Development (TDD)** with comprehensive numerical parity testing:

```bash
# Run all tests
pytest tests/

# Run specific phase tests
pytest tests/phase1_tensor_core/
pytest tests/phase2_operators/

# Run with coverage
pytest tests/ --cov=mlx_compat --cov-report=html

# Run integration tests
pytest tests/integration/ -m integration

# Run parity tests (compare against PyTorch)
pytest tests/ -m parity
```

### Test Organization
```
tests/
├── common_utils.py              # Shared utilities
├── phase1_tensor_core/          # 15+ tests
├── phase2_operators/            # 50+ tests
├── phase3_autograd/             # 20+ tests
├── phase4_nn_modules/           # 30+ tests
├── phase5_training/             # 10+ tests
└── integration/                 # End-to-end tests
```

## 📚 Documentation

Comprehensive documentation is available in the `pytorch-mlx-porting-docs/` directory:

- **[00-OVERVIEW.md](pytorch-mlx-porting-docs/00-OVERVIEW.md)**: PyTorch architecture overview
- **[08-PORTING-GUIDE/implementation-roadmap.md](pytorch-mlx-porting-docs/08-PORTING-GUIDE/implementation-roadmap.md)**: 12-week implementation plan
- **[08-PORTING-GUIDE/mlx-mapping.md](pytorch-mlx-porting-docs/08-PORTING-GUIDE/mlx-mapping.md)**: PyTorch → MLX API mappings

### Documentation Structure
- `01-FOUNDATIONS/`: Tensor core, dispatch, type system (6 files)
- `02-OPERATORS/`: Operator reference (19 files)
- `03-AUTOGRAD/`: Automatic differentiation (8 files)
- `04-NEURAL-NETWORKS/`: Layers and modules (22 files)
- `05-TRAINING/`: Optimizers and schedulers (8 files)
- `08-PORTING-GUIDE/`: Implementation guides (3 files)

### Reference Implementation
The full PyTorch source code is available at `reference/pytorch/` for reference.

## 🎓 Examples

```
examples/
├── mnist_mlp.py          # MLP training on MNIST
├── lenet_cifar10.py      # CNN training on CIFAR-10
├── resnet.py             # ResNet model
└── transformer.py        # Transformer model
```

## 🔧 Development

### Code Style
- **Formatter**: Black (line length: 100)
- **Import sorting**: isort
- **Linter**: flake8

```bash
# Format code
black mlx_compat tests
isort mlx_compat tests

# Lint
flake8 mlx_compat tests
```

### Contributing
1. Follow TDD: Write tests first
2. Ensure numerical parity with PyTorch
3. Maintain >90% code coverage
4. Follow existing code style
5. Reference documentation and PyTorch implementation

## 📊 Success Metrics

### Phase Completion Criteria
| Phase | Success Criteria |
|-------|------------------|
| Phase 0 | ✅ Package structure, tests run |
| Phase 1 | All tensor operations work, 100% coverage |
| Phase 2 | 50 Tier 1 operators pass numerical parity |
| Phase 3 | Autograd backward works, gradient checking passes |
| Phase 4 | 15+ layers, model composition works |
| Phase 5 | Full training loop runs, checkpointing works |
| Phase 6 | 3+ models train, benchmarks complete |

### Overall Success
- **API Coverage**: 90%+ of common PyTorch workflows
- **Numerical Parity**: <1e-5 error (forward), <1e-4 error (gradients)
- **Performance**: Within 20% of PyTorch MPS
- **Test Coverage**: >90%

## ⚠️ Known Limitations

### MLX Constraints
- **No float64**: MLX only supports float16/float32/bfloat16
- **Immutability**: In-place operations require workarounds
- **Apple Silicon only**: MLX requires macOS with Metal
- **Single GPU**: No distributed training support

### Layout Differences
- PyTorch uses NCHW (channels-first) for convolutions
- MLX uses NHWC (channels-last) for Metal optimization
- Automatic transpose operations may impact performance

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the reference implementation and design patterns
- **Apple MLX Team**: For the high-performance ML framework
- **Documentation**: Based on extensive PyTorch architecture analysis

## 📞 Contact

For questions, issues, or contributions:
- GitHub Issues: [https://github.com/yourusername/flashlight/issues](https://github.com/yourusername/flashlight/issues)
- Documentation: See `pytorch-mlx-porting-docs/`

---

**Status**: Phase 6 Complete ✅ | Production Ready
