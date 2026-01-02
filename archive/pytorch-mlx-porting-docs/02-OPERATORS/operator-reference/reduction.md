# Reduction Operators

## Purpose

Reduction operators aggregate tensor values along specified dimensions, computing summary statistics and finding extrema. This document covers Tier 1 reduction operators essential for ML workloads.

**Tier 1 Reduction Operators** (8 total):
- `sum` - Sum of elements
- `mean` - Average of elements
- `max` - Maximum value (with optional indices)
- `min` - Minimum value (with optional indices)
- `argmax` - Index of maximum value
- `argmin` - Index of minimum value
- `prod` - Product of elements
- `var` - Variance

## Common Properties

**Tags**: `[core, reduction]` or `[reduction]`

**Dimension Handling**:
- **No dim specified**: Reduce over all dimensions → scalar
- **dim specified**: Reduce along that dimension
- **keepdim=True**: Preserve reduced dimensions as size 1

**Type Promotion**: Often promotes to wider type (int32 → int64 for sum)

**NaN Handling**: Most propagate NaN (use `nansum`, `nanmean`, etc. for NaN-ignoring versions)

**Performance**: Typically memory-bound, critical for gradients

## Operator Details

### sum (Summation)

**Purpose**: Sum tensor elements along specified dimensions

**Signature**:
```python
sum(Tensor self, *, ScalarType? dtype=None) -> Tensor  # All dims
sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
```

**Formula**:
- All dims: `out = Σ_i self[i]`
- Along dim: `out[...] = Σ_i self[..., i, ...]`

**YAML Definition** (`native_functions.yaml:5936-5968`):
```yaml
- func: sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: sum
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: sum_coo
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sum_csr
  autogen: sum.out
  tags: reduction

- func: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  structured_delegate: sum.IntList_out
  device_check: NoCheck
  variants: function, method
  dispatch:
    NestedTensorCPU: NestedTensor_sum_dim_CPU
    SparseCPU, SparseCUDA, SparseMPS: sum_sparse_coo
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sum_sparse_compressed
  tags: [core, reduction]
```

**CPU Implementation** (`native/cpu/ReduceOpsKernel.cpp`):
```cpp
void sum_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "sum_cpu", [&]() {
    binary_kernel_reduce(
        iter,
        SumOps<scalar_t>{},
        scalar_t(0));  // Initial value
  });
}

// SumOps functor
template <typename scalar_t>
struct SumOps {
  inline scalar_t reduce(scalar_t acc, scalar_t value, int64_t /*idx*/) const {
    return acc + value;
  }

  inline scalar_t combine(scalar_t a, scalar_t b) const {
    return a + b;
  }

  inline scalar_t project(scalar_t acc) const {
    return acc;
  }
};
```

**MPS Implementation** (`mps/operations/ReduceOps.mm`):
```objective-c
Tensor& sum_out_mps(const Tensor& self, IntArrayRef dim, bool keepdim,
                    std::optional<ScalarType> dtype, Tensor& output) {
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSGraph* graph = make_mps_graph();

    auto inputTensor = mpsGraphRankedPlaceHolder(graph, self);

    // Convert dim to NSArray
    NSMutableArray* axes = [[NSMutableArray alloc] init];
    for (int64_t d : dim) {
      [axes addObject:@(d)];
    }

    // Use MPSGraph reduction operation
    MPSGraphTensor* outputTensor = [graph reductionSumWithTensor:inputTensor
                                                            axes:axes
                                                            name:nil];

    if (keepdim) {
      // Expand dims back to original rank
      outputTensor = [graph expandDimsOfTensor:outputTensor
                                          axes:axes
                                          name:nil];
    }

    runMPSGraph(stream, graph, inputTensor, outputTensor, output);
  }
  return output;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def sum(x, dim=None, keepdim=False, dtype=None):
    """Sum reduction"""
    if dim is None:
        result = mx.sum(x)
    else:
        result = mx.sum(x, axis=dim, keepdims=keepdim)

    if dtype is not None:
        result = result.astype(dtype)

    return result
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: grad.expand_as(self)  # Broadcast gradient back
```

**Gradient Explanation**:
Forward reduces dimensions, backward broadcasts gradient:
```python
# Forward: (3, 4, 5) --sum(dim=1)--> (3, 5)
# Backward: (3, 5) --expand--> (3, 4, 5)
# Each element gets same gradient
```

**Usage Examples**:
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Sum all elements
torch.sum(x)  # tensor(21.)

# Sum along dimension 0 (columns)
torch.sum(x, dim=0)  # tensor([5., 7., 9.])

# Sum along dimension 1 (rows)
torch.sum(x, dim=1)  # tensor([6., 15.])

# Keep dimensions
torch.sum(x, dim=1, keepdim=True)  # tensor([[6.], [15.]])

# Multiple dimensions
x = torch.randn(2, 3, 4)
torch.sum(x, dim=[0, 2])  # Sum over dims 0 and 2, result shape: (3,)
```

**Type Promotion**:
```python
# Integer sum promotes to int64 to avoid overflow
x = torch.tensor([1, 2, 3], dtype=torch.int32)
torch.sum(x).dtype  # torch.int64

# Float types preserve dtype
x = torch.randn(10, dtype=torch.float32)
torch.sum(x).dtype  # torch.float32
```

---

### mean (Average)

**Purpose**: Compute arithmetic mean along dimensions

**Signature**:
```python
mean(Tensor self, *, ScalarType? dtype=None) -> Tensor  # All dims
mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
```

**Formula**:
```
mean = sum(x) / count(x)
```

**YAML Definition** (`native_functions.yaml:4007-4036`):
```yaml
- func: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: mean
  tags: [core, reduction]

- func: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  structured_delegate: mean.out
  device_check: NoCheck
  variants: function, method
  dispatch:
    QuantizedCPU: mean_quantized_cpu
  tags: [core, reduction]
```

**CPU Implementation**:
```cpp
Tensor mean_cpu(const Tensor& self, std::optional<ScalarType> dtype) {
  ScalarType dtype_to_sum = dtype.has_value() ? dtype.value() : self.scalar_type();

  Tensor result = at::sum(self, dtype_to_sum);
  return result.div_(self.numel());  // Divide by count
}

Tensor mean_dim(const Tensor& self, IntArrayRef dim, bool keepdim,
                std::optional<ScalarType> dtype) {
  Tensor result = at::sum(self, dim, keepdim, dtype);
  int64_t count = self.numel() / result.numel();
  return result.div_(count);
}
```

**MLX Equivalent**:
```python
def mean(x, dim=None, keepdim=False, dtype=None):
    """Mean reduction"""
    if dim is None:
        result = mx.mean(x)
    else:
        result = mx.mean(x, axis=dim, keepdims=keepdim)

    if dtype is not None:
        result = result.astype(dtype)

    return result
```

**Gradient**:
```yaml
- name: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: grad.expand_as(self) / self.numel() * result.numel()
```

**Gradient Explanation**:
```python
# Each input contributes 1/n to the mean
# So gradient of mean is grad / n for each element
```

**Usage Examples**:
```python
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

torch.mean(x)  # tensor(3.5)

torch.mean(x, dim=0)  # tensor([2.5, 3.5, 4.5])
torch.mean(x, dim=1)  # tensor([2.0, 5.0])

# Mean with keepdim
torch.mean(x, dim=1, keepdim=True)  # tensor([[2.0], [5.0]])
```

**Integer Inputs**:
```python
# Mean of integers returns float
x = torch.tensor([1, 2, 3])
torch.mean(x)  # Error! Use float tensor

# Must convert first
torch.mean(x.float())  # tensor(2.0)
```

---

### max (Maximum Value)

**Purpose**: Find maximum value, optionally with indices

**Signature**:
```python
max(Tensor self) -> Tensor  # Maximum over all elements
max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
```

**Returns**:
- Unary max: Single maximum value
- Max along dim: (max_values, indices)

**YAML Definition** (`native_functions.yaml:3905-3921 and 10247-10254`):
```yaml
- func: max(Tensor self) -> Tensor
  device_check: NoCheck
  variants: method, function
  dispatch:
    CPU, CUDA: max
    MPS: max_mps
    QuantizedCPU: max_quantized_cpu
  tags: [reduction]

- func: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  device_check: NoCheck
  structured_delegate: max.dim_max
  variants: function, method
  dispatch:
    QuantizedCPU, QuantizedCUDA: qmax
  tags: [core, reduction]
```

**CPU Implementation**:
```cpp
void max_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "max_cpu", [&]() {
    binary_kernel_reduce_vec(
        iter,
        MaxOps<scalar_t>{},
        std::numeric_limits<scalar_t>::lowest());  // Initial value
  });
}

template <typename scalar_t>
struct MaxOps {
  inline scalar_t reduce(scalar_t acc, scalar_t value, int64_t /*idx*/) const {
    return std::max(acc, value);
  }

  inline scalar_t combine(scalar_t a, scalar_t b) const {
    return std::max(a, b);
  }
};
```

**MLX Equivalent**:
```python
def max(x, dim=None, keepdim=False):
    """Max reduction"""
    if dim is None:
        return mx.max(x)
    else:
        values = mx.max(x, axis=dim, keepdims=keepdim)
        indices = mx.argmax(x, axis=dim, keepdims=keepdim)
        return values, indices
```

**Gradient**:
```yaml
- name: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  self: value_selecting_reduction_backward(grad, dim, indices, self.sizes(), keepdim)
```

**Gradient Explanation**:
```python
# Gradient flows only to the maximum element(s)
# Use indices to scatter gradient back
```

**Usage Examples**:
```python
x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

# Maximum over all elements
torch.max(x)  # tensor(6.)

# Maximum along dimension (returns values and indices)
values, indices = torch.max(x, dim=0)
# values:  tensor([4., 5., 6.])
# indices: tensor([1, 0, 1])  # Row indices

values, indices = torch.max(x, dim=1)
# values:  tensor([5., 6.])
# indices: tensor([1, 2])  # Column indices

# Keepdim
values, indices = torch.max(x, dim=1, keepdim=True)
# values:  tensor([[5.], [6.]])
# indices: tensor([[1], [2]])
```

**NaN Handling**:
```python
x = torch.tensor([1.0, float('nan'), 3.0])
torch.max(x)  # tensor(nan)  # NaN propagates
```

---

### min (Minimum Value)

**Purpose**: Find minimum value, optionally with indices

**Signature**:
```python
min(Tensor self) -> Tensor
min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
```

**YAML Definition** (`native_functions.yaml:4102-4118 and 10217-10231`):
```yaml
- func: min(Tensor self) -> Tensor
  device_check: NoCheck
  variants: method, function
  dispatch:
    CPU, CUDA: min
    MPS: min_mps
    QuantizedCPU: min_quantized_cpu
  tags: [reduction]

- func: min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  device_check: NoCheck
  structured_delegate: min.dim_min
  variants: function, method
  dispatch:
    QuantizedCPU, QuantizedCUDA: qmin
  tags: [core, reduction]
```

**MLX Equivalent**:
```python
def min(x, dim=None, keepdim=False):
    """Min reduction"""
    if dim is None:
        return mx.min(x)
    else:
        values = mx.min(x, axis=dim, keepdims=keepdim)
        indices = mx.argmin(x, axis=dim, keepdims=keepdim)
        return values, indices
```

**Usage Examples**:
```python
x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

torch.min(x)  # tensor(1.)

values, indices = torch.min(x, dim=0)
# values:  tensor([1., 2., 3.])
# indices: tensor([0, 1, 0])

values, indices = torch.min(x, dim=1)
# values:  tensor([1., 2.])
# indices: tensor([0, 1])
```

---

### argmax (Index of Maximum)

**Purpose**: Find index of maximum value

**Signature**:
```python
argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
```

**Returns**:
- No dim: Flattened index of global maximum
- With dim: Indices of maximum along dimension

**YAML Definition** (`native_functions.yaml:835-846`):
```yaml
- func: argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  structured_delegate: argmax.out
  device_check: NoCheck
  variants: function, method
  tags: [core, reduction]
```

**CPU Implementation**:
```cpp
void argmax_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "argmax_cpu", [&]() {
    binary_kernel_reduce(
        iter,
        ArgMaxOps<scalar_t>{},
        std::pair<scalar_t, int64_t>(
            std::numeric_limits<scalar_t>::lowest(), 0));
  });
}

template <typename scalar_t>
struct ArgMaxOps {
  using acc_t = std::pair<scalar_t, int64_t>;  // (value, index)

  inline acc_t reduce(acc_t acc, scalar_t value, int64_t idx) const {
    if (value > acc.first) {
      return {value, idx};
    }
    return acc;
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return (a.first > b.first) ? a : b;
  }

  inline int64_t project(acc_t acc) const {
    return acc.second;  // Return index
  }
};
```

**MLX Equivalent**:
```python
def argmax(x, dim=None, keepdim=False):
    """Argmax reduction"""
    if dim is None:
        # Flatten and find global argmax
        return mx.argmax(x.reshape(-1))
    else:
        return mx.argmax(x, axis=dim, keepdims=keepdim)
```

**Gradient**: Not differentiable (returns indices, not values)

**Usage Examples**:
```python
x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

# Global argmax (flattened index)
torch.argmax(x)  # tensor(5)  # Index in flattened array

# Argmax along dimension
torch.argmax(x, dim=0)  # tensor([1, 0, 1])  # Row with max in each column
torch.argmax(x, dim=1)  # tensor([1, 2])     # Column with max in each row

# Keepdim
torch.argmax(x, dim=1, keepdim=True)  # tensor([[1], [2]])
```

**Tie-breaking**: Returns first index when multiple maxima exist
```python
x = torch.tensor([3.0, 5.0, 5.0, 2.0])
torch.argmax(x)  # tensor(1)  # First occurrence of max
```

---

### argmin (Index of Minimum)

**Purpose**: Find index of minimum value

**Signature**:
```python
argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
```

**YAML Definition** (`native_functions.yaml:848-859`):
```yaml
- func: argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  structured_delegate: argmin.out
  device_check: NoCheck
  variants: function, method
  tags: [core, reduction]
```

**MLX Equivalent**:
```python
def argmin(x, dim=None, keepdim=False):
    """Argmin reduction"""
    if dim is None:
        return mx.argmin(x.reshape(-1))
    else:
        return mx.argmin(x, axis=dim, keepdims=keepdim)
```

**Usage Examples**:
```python
x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

torch.argmin(x)  # tensor(0)  # Global minimum

torch.argmin(x, dim=0)  # tensor([0, 1, 0])
torch.argmin(x, dim=1)  # tensor([0, 1])
```

---

### prod (Product)

**Purpose**: Compute product of tensor elements

**Signature**:
```python
prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
```

**Formula**:
```
prod = x[0] * x[1] * ... * x[n]
```

**YAML Definition** (`native_functions.yaml:6136-6157`):
```yaml
- func: prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CPU, CUDA: prod
    MPS: prod_mps
  autogen: prod.out
  tags: [core, reduction]

- func: prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  structured_delegate: prod.int_out
  device_check: NoCheck
  variants: function, method
  tags: [core, reduction]
```

**CPU Implementation**:
```cpp
void prod_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "prod_cpu", [&]() {
    binary_kernel_reduce(
        iter,
        ProdOps<scalar_t>{},
        scalar_t(1));  // Initial value
  });
}

template <typename scalar_t>
struct ProdOps {
  inline scalar_t reduce(scalar_t acc, scalar_t value, int64_t /*idx*/) const {
    return acc * value;
  }

  inline scalar_t combine(scalar_t a, scalar_t b) const {
    return a * b;
  }
};
```

**MLX Equivalent**:
```python
def prod(x, dim=None, keepdim=False, dtype=None):
    """Product reduction"""
    if dim is None:
        result = mx.prod(x)
    else:
        result = mx.prod(x, axis=dim, keepdims=keepdim)

    if dtype is not None:
        result = result.astype(dtype)

    return result
```

**Gradient**:
```yaml
- name: prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  self: prod_backward(grad, self, result, dim)
```

**Gradient Formula**:
```
∂(∏x_i)/∂x_j = (∏x_i) / x_j
```

**Usage Examples**:
```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
torch.prod(x)  # tensor(24.)  # 1*2*3*4

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
torch.prod(x, dim=0)  # tensor([3., 8.])   # [1*3, 2*4]
torch.prod(x, dim=1)  # tensor([2., 12.])  # [1*2, 3*4]
```

**Overflow Warning**:
```python
# Product grows quickly, can overflow
x = torch.tensor([100.0] * 100)
torch.prod(x)  # inf (overflow)
```

---

### var (Variance)

**Purpose**: Compute variance (measure of spread)

**Signature**:
```python
var(Tensor self, bool unbiased=True) -> Tensor
var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
```

**Formula**:

**Unbiased (Bessel's correction)**:
```
var = Σ(x_i - mean)² / (n - 1)
```

**Biased**:
```
var = Σ(x_i - mean)² / n
```

**correction parameter**: General form, `var = Σ(x_i - mean)² / (n - correction)`

**YAML Definition** (`native_functions.yaml:6623-6643`):
```yaml
- func: var(Tensor self, bool unbiased=True) -> Tensor
  device_check: NoCheck
  variants: function, method
  cpp_no_default_args: ["unbiased"]
  tags: reduction

- func: var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
  device_check: NoCheck
  variants: function, method
  tags: [core, reduction]
  cpp_no_default_args: ["unbiased"]

- func: var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CPU, CUDA: var
    MPS: var_mps
    MTIA: var_mtia
  tags: [core, reduction]
```

**CPU Implementation**:
```cpp
Tensor var_cpu(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  // Two-pass algorithm:
  // 1. Compute mean
  Tensor mean = at::mean(self, dim, /*keepdim=*/true);

  // 2. Compute variance
  Tensor centered = self - mean;
  Tensor sq = centered.pow(2);
  Tensor sum_sq = at::sum(sq, dim, keepdim);

  int64_t n = self.numel() / sum_sq.numel();
  int64_t correction = unbiased ? 1 : 0;

  return sum_sq.div_(n - correction);
}
```

**MLX Equivalent**:
```python
def var(x, dim=None, unbiased=True, keepdim=False):
    """Variance reduction"""
    if dim is None:
        result = mx.var(x, ddof=1 if unbiased else 0)
    else:
        result = mx.var(x, axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)

    return result
```

**Gradient**:
```
∂var/∂x_i = 2(x_i - mean) / (n - correction)
```

**Usage Examples**:
```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Unbiased variance (default, n-1 denominator)
torch.var(x)  # tensor(2.5)

# Biased variance (n denominator)
torch.var(x, unbiased=False)  # tensor(2.0)

# Manual check
mean = x.mean()
((x - mean) ** 2).sum() / 4  # tensor(2.5)  # n-1 = 4

# Variance along dimension
x = torch.randn(3, 4, 5)
torch.var(x, dim=1)  # Shape: (3, 5)
torch.var(x, dim=[0, 2])  # Shape: (4,)
```

**Numerical Stability**:
PyTorch uses **two-pass algorithm** (compute mean first, then variance) for numerical stability. Single-pass algorithms can suffer from catastrophic cancellation.

---

## Common Reduction Patterns

### keepdim Behavior

**keepdim=False** (default):
```python
x = torch.randn(2, 3, 4)
torch.sum(x, dim=1).shape  # (2, 4)  # Dimension 1 removed
```

**keepdim=True**:
```python
torch.sum(x, dim=1, keepdim=True).shape  # (2, 1, 4)  # Dimension 1 preserved as 1
```

**Why keepdim?** Enables broadcasting without reshaping:
```python
x = torch.randn(2, 3, 4)
mean = torch.mean(x, dim=1, keepdim=True)  # (2, 1, 4)
centered = x - mean  # Broadcasts correctly
```

### Multiple Dimensions

```python
x = torch.randn(2, 3, 4, 5)

# Reduce over multiple dimensions
torch.sum(x, dim=[1, 3])  # Shape: (2, 4)

# Reduce all but one dimension
torch.sum(x, dim=[0, 1, 2])  # Shape: (5,)
```

### NaN-Aware Variants

```python
x = torch.tensor([1.0, float('nan'), 3.0])

torch.sum(x)     # tensor(nan)  # NaN propagates
torch.nansum(x)  # tensor(4.0)  # NaN ignored

torch.mean(x)    # tensor(nan)
torch.nanmean(x) # tensor(2.0)
```

---

## Gradient Flow in Reductions

**General Pattern**: Gradients broadcast back to input shape

**Sum**:
```python
x = torch.randn(3, 4, requires_grad=True)
y = x.sum(dim=1)  # Shape: (3,)
y.backward(torch.ones(3))
# x.grad has same shape as x: (3, 4)
# Each element in row gets same gradient
```

**Mean**:
```python
x = torch.randn(3, 4, requires_grad=True)
y = x.mean(dim=1)  # Shape: (3,)
y.backward(torch.ones(3))
# x.grad: Each element = 1/4 (divided by dimension size)
```

**Max**:
```python
x = torch.randn(3, 4, requires_grad=True)
y, idx = x.max(dim=1)  # Shape: (3,)
y.backward(torch.ones(3))
# x.grad: Only max elements get gradient, others get 0
```

---

## Implementation Files

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:5936-5985` (sum)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:4007-4052` (mean)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:3905-3941` (max)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:4102-4139` (min)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:835-859` (argmax/argmin)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:6136-6166` (prod)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:6623-6695` (var)

**CPU Kernels**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/SumKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/ReduceOps.cpp`

**MPS Kernels** (Metal):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/ReduceOps.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/SummaryOps.mm`

**Gradients**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/tools/autograd/derivatives.yaml`

---

## MLX Porting Summary

**Direct Mappings**:
```python
# PyTorch → MLX
torch.sum     → mx.sum
torch.mean    → mx.mean
torch.max     → mx.max (values only) + mx.argmax (indices)
torch.min     → mx.min (values only) + mx.argmin (indices)
torch.argmax  → mx.argmax
torch.argmin  → mx.argmin
torch.prod    → mx.prod
torch.var     → mx.var (with ddof parameter)
```

**Considerations**:

1. **keepdim Parameter**: MLX uses `keepdims` (plural), PyTorch uses `keepdim` (singular)

2. **Multiple Return Values**: PyTorch's `max.dim` returns `(values, indices)`, MLX requires separate calls

3. **Variance Correction**:
   - PyTorch: `unbiased=True` (default)
   - MLX: `ddof=1` (degrees of freedom)

4. **Axis vs Dim**: MLX uses `axis`, PyTorch uses `dim`

**Example Compatibility Layer**:
```python
import mlx.core as mx

class torch_reductions:
    @staticmethod
    def sum(x, dim=None, keepdim=False):
        if dim is None:
            return mx.sum(x)
        return mx.sum(x, axis=dim, keepdims=keepdim)

    @staticmethod
    def mean(x, dim=None, keepdim=False):
        if dim is None:
            return mx.mean(x)
        return mx.mean(x, axis=dim, keepdims=keepdim)

    @staticmethod
    def max(x, dim=None, keepdim=False):
        if dim is None:
            return mx.max(x)
        values = mx.max(x, axis=dim, keepdims=keepdim)
        indices = mx.argmax(x, axis=dim, keepdims=keepdim)
        return values, indices

    @staticmethod
    def var(x, dim=None, unbiased=True, keepdim=False):
        ddof = 1 if unbiased else 0
        if dim is None:
            return mx.var(x, ddof=ddof)
        return mx.var(x, axis=dim, keepdims=keepdim, ddof=ddof)
```

Reduction operators are fundamental for computing statistics and gradients, and have straightforward mappings to MLX with minor API differences.
