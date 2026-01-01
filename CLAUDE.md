# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MLX Compat** is a PyTorch-compatible API layer for Apple's MLX framework. It enables PyTorch code to run on Apple Silicon with minimal modifications through a complete bottom-up implementation.

**Package name**: `flashlight` (import as `flashlight`)
**Status**: Phase 6 Complete - Full training pipeline verified

## Build/Test/Lint Commands

```bash
# Install (use pip3/python3, not pip/python)
pip3 install -r requirements.txt
pip3 install -e ".[dev]"           # Development mode with all deps

# Testing
pytest tests/                      # Run all tests
pytest tests/phase1_tensor_core/   # Specific phase
pytest tests/ -m parity            # PyTorch parity tests only
pytest tests/ -m integration       # Integration tests
pytest tests/ -k "test_name"       # Single test by name

# Linting/Formatting
black flashlight tests
isort flashlight tests
flake8 flashlight tests

# Parity check tool
python3 -m parity_check --format console
```

## Architecture

Layered design:

```
Python API (nn.Module, optim, training)
Autograd (tape-based automatic differentiation)
Operators (50+ ops: arithmetic, activations, reductions, etc.)
Tensor Core (wrapper around MLX arrays)
-----------
MLX Backend (external, Apple's Metal-based ML framework)
```

### Key Source Files

- `flashlight/tensor.py` - Core Tensor class wrapping MLX arrays
- `flashlight/ops/` - All operators (arithmetic, activations, reductions, shape, indexing, conv, pooling)
- `flashlight/autograd/engine.py` - Backward pass engine
- `flashlight/nn/module.py` - Base Module class
- `flashlight/nn/layers/` - Layer implementations (Linear, Conv2d, BatchNorm, etc.)
- `flashlight/optim/` - Optimizers (SGD, Adam, AdamW) and LR schedulers

### Critical Design Patterns

1. **Layout Conversion**: Automatic NCHW (PyTorch) <-> NHWC (MLX) for spatial ops
2. **Conv2d Weight Format**: Transpose from [out, in, kH, kW] to [out, kH, kW, in] for MLX
3. **Immutability**: In-place ops implemented as copy + functional op (MLX is immutable)
4. **Unified Memory**: Device management is a compatibility shim (MLX uses unified memory)

### MLX Constraints

- No float64 support (only float16/float32/bfloat16)
- No true in-place operations
- Apple Silicon only (macOS with Metal)
- Single GPU (no distributed training)

## Testing

- Parity tests compare against PyTorch (markers: `@pytest.mark.parity`)
- Numerical tolerance: <1e-5 (forward), <1e-4 (gradients)
- Test files organized by phase in `tests/phase*/`
