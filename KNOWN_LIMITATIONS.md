# Known Limitations: PyTorch vs Flashlight

This document describes fundamental differences between Flashlight and PyTorch that cannot be resolved due to underlying MLX framework constraints.

## Fundamental Limitations

These limitations stem from MLX's architecture and Apple Silicon hardware constraints. They cannot be fixed with code changes.

### 1. No Float64 (Double Precision) Support

**Impact**: Critical for scientific computing requiring high precision.

MLX does not support float64 on Apple Silicon. All double-precision operations are automatically downcast to float32.

```python
# PyTorch
x = torch.tensor([1.0], dtype=torch.float64)  # 15 significant digits

# Flashlight
x = fl.tensor([1.0], dtype=fl.float64)  # Silently uses float32 (~7 significant digits)
```

**Workaround**: None. Use PyTorch on CPU for computations requiring double precision.

### 2. No True In-Place Operations

**Impact**: Memory optimization patterns don't work as expected.

MLX arrays are immutable. Operations like `tensor.add_(value)` create new arrays internally rather than modifying in place.

```python
# PyTorch - modifies x in place
x = torch.tensor([1.0])
y = x
x.add_(1)
print(y)  # tensor([2.0]) - y also changed

# Flashlight - creates new array
x = fl.tensor([1.0])
y = x
x.add_(1)
print(y)  # tensor([1.0]) - y unchanged
```

**Workaround**: Avoid relying on in-place aliasing semantics. Reassign explicitly: `x = x + 1`.

### 3. Views Are Copies (No Shared Storage)

**Impact**: Memory-sharing patterns don't work; views consume additional memory.

MLX doesn't support memory views. Operations like `view()`, `reshape()`, `transpose()` create new arrays.

```python
# PyTorch - shares memory
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = x.view(2, 2)
y[0, 0] = 99
print(x[0])  # tensor(99.0) - x changed

# Flashlight - independent copy
x = fl.tensor([1.0, 2.0, 3.0, 4.0])
y = x.view(2, 2)
y[0, 0] = 99
print(x[0])  # tensor(1.0) - x unchanged
```

**Workaround**: None. Be aware of increased memory usage with view operations.

### 4. No Sparse Tensor Support

**Impact**: Sparse embeddings and sparse gradients are simulated as dense tensors.

```python
# PyTorch
embedding = nn.Embedding(10000, 512, sparse=True)
# Gradients are truly sparse

# Flashlight
embedding = fl.nn.Embedding(10000, 512, sparse=True)
# Gradients are dense (sparse=True is accepted but has no effect)
```

**Workaround**: None. Large vocabularies will use more memory for gradients.

### 5. Limited Complex Number Support

**Impact**: Complex64 support depends on MLX version; complex128 not available.

```python
# complex64 may work if MLX supports it
# complex128 (double precision complex) never available
```

### 6. RNG State Format Incompatibility

**Impact**: Cannot share random state between PyTorch and Flashlight.

Flashlight uses a 44-byte MPS-compatible state format with Philox algorithm. PyTorch CPU uses 800-byte state with different internal representation.

```python
# Cannot do:
state = torch.get_rng_state()
fl.set_rng_state(state)  # Won't work - different format
```

**Workaround**: Use seeds for reproducibility instead of state transfer.

### 7. NCHW↔NHWC Layout Conversion Overhead

**Impact**: Slight numerical differences due to transposes.

MLX internally uses NHWC layout while PyTorch uses NCHW. Automatic conversion adds computational overhead and may introduce small floating-point differences.

### 8. SVD and Some Linalg Operations Require CPU Stream

**Impact**: Must explicitly request CPU stream for certain linalg operations.

```python
# MLX SVD requires explicit CPU stream
# This is handled internally but may affect performance
```

## Documented Design Differences

These are intentional implementation choices that match PyTorch behavior:

### 1. Median Returns Lower Middle Value

For even-length arrays, both PyTorch and Flashlight return the lower of the two middle values (not their average like NumPy).

### 2. Quantile Uses Linear Interpolation

Default interpolation method is 'linear', matching PyTorch's default.

### 3. Mode Index Threshold at n=24

The mode operation uses different algorithms for small (n<24) vs large arrays, matching PyTorch's internal implementation.

## Tolerance Relaxations

Some operations require looser numerical tolerances due to accumulated floating-point differences:

| Operation | rtol | atol | Reason |
|-----------|------|------|--------|
| conv2d, conv3d | 1e-4 | 1e-5 | Many accumulated multiply-adds |
| conv_transpose2d/3d | 1e-4 | 1e-5 | FP error accumulation |
| rnn_tanh | 1e-4 | 1e-5 | Multi-layer accumulation |
| rnn_relu | 1e-2 | 1e-3 | Sharper gradients |
| multi_head_attention | 1e-4 | 1e-4 | Many matmuls |

## Checking for Limitations at Runtime

```python
import flashlight as fl

# Check dtype support
print(fl.float64)  # Will show it maps to float32

# Check if complex is available
try:
    x = fl.tensor([1+2j])
    print("Complex64 supported")
except:
    print("Complex64 not supported")
```

## Reporting Issues

If you encounter a divergence not listed here, please report it at:
https://github.com/anthropics/flashlight/issues
