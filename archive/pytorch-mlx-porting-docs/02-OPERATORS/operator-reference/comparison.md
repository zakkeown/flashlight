# Comparison Operators Reference

**Purpose**: Element-wise comparison operations that return boolean tensors

**File**: `pytorch/aten/src/ATen/native/native_functions.yaml` (comparison operators)

**Related Files**:
- CPU Implementation: `pytorch/aten/src/ATen/native/CompareKernel.cpp`
- MPS Implementation: `pytorch/aten/src/ATen/native/mps/operations/Compare.mm`
- CUDA Implementation: `pytorch/aten/src/ATen/native/cuda/CompareKernels.cu`

---

## Overview

Comparison operators perform element-wise comparisons between tensors (or tensor and scalar) and return boolean tensors indicating where the condition is true. These are fundamental for:
- Conditional logic and masking
- Filtering and selection
- Validation and assertions
- Loss function implementation

**Key Characteristics**:
- **Element-wise**: Operate independently on each element
- **Broadcasting**: Automatically broadcast mismatched shapes
- **Boolean output**: Return `torch.bool` tensors (True/False)
- **Non-differentiable**: No gradients (discrete operations)
- **Pointwise tag**: Eligible for kernel fusion optimizations

---

## Week 3-4 Day 1 Operators - Basic Comparisons

### eq (Equal)

**Purpose**: Element-wise equality comparison (==)

**Signature**:
- `eq.Scalar(Tensor self, Scalar other) -> Tensor`
- `eq.Tensor(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9184-9210):
```yaml
- func: eq.Scalar(Tensor self, Scalar other) -> Tensor
  structured_delegate: eq.Scalar_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: eq_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: eq_scalar_nested
  tags: [core, pointwise]

- func: eq.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: eq.Tensor_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: eq_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: eq_tensor_nested
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] == other[i])
# Broadcasts shapes if needed
result = torch.empty(broadcast_shape(self, other), dtype=torch.bool)
for i in range(result.numel()):
    result[i] = (self[i] == other[i])
return result
```

**CPU Implementation** (native/CompareKernel.cpp):
```cpp
void eq_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "eq_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
      return a == b;
    });
  });
}
```

**MPS Implementation** (native/mps/operations/Compare.mm):
```objc
Tensor eq_tensor_mps(const Tensor& self, const Tensor& other) {
  MPSGraph* mpsGraph = make_mps_graph();

  auto selfTensor = getMPSGraphTensor(self);
  auto otherTensor = getMPSGraphTensor(other);

  MPSGraphTensor* resultTensor = [mpsGraph equalWithPrimaryTensor:selfTensor
                                                   secondaryTensor:otherTensor
                                                              name:@"eq"];

  return createBoolTensorFromMPSGraph(resultTensor);
}
```

**Backward Pass**:
```
Not differentiable - comparison operations have no gradients
```

**MLX Equivalent**:
```python
import mlx.core as mx

def eq(x, y):
    """
    PyTorch: torch.eq(x, y) or x == y
    MLX: mx.equal(x, y) or x == y
    """
    return mx.equal(x, y)
    # Or use operator: x == y

# Example 1: Tensor comparison
x = mx.array([1, 2, 3, 4])
y = mx.array([1, 0, 3, 0])
result = mx.equal(x, y)  # [True, False, True, False]

# Example 2: Scalar comparison
x = mx.array([1.0, 2.0, 3.0])
result = x == 2.0  # [False, True, False]

# Example 3: Broadcasting
x = mx.array([[1, 2], [3, 4]])  # [2, 2]
y = mx.array([1, 2])  # [2]
result = x == y  # [[True, True], [False, False]]
```

**Common Patterns**:
```python
# Pattern 1: Create boolean mask
data = torch.tensor([1, 2, 3, 4, 5])
mask = data == 3  # [False, False, True, False, False]
selected = data[mask]  # tensor([3])

# Pattern 2: Count matches
predictions = torch.tensor([0, 1, 1, 0, 1])
labels = torch.tensor([0, 1, 0, 0, 1])
correct = (predictions == labels).sum()  # 4

# Pattern 3: Find indices
x = torch.tensor([10, 20, 30, 20, 40])
indices = torch.where(x == 20)  # (tensor([1, 3]),)

# Pattern 4: Validation
result = model(x)
expected = torch.zeros_like(result)
assert torch.eq(result, expected).all(), "Result should be zero"
```

**Edge Cases**:
- **NaN handling**: `NaN == NaN` is False (IEEE 754 standard)
- **Floating point**: May have precision issues (`torch.isclose` preferred)
- **Different dtypes**: Promotes to common dtype before comparison
- **Broadcasting**: Expands dimensions automatically

**Performance Notes**:
- Highly optimized (vectorized on CPU, parallel on GPU)
- Element-wise operation (O(n) complexity)
- Tagged `pointwise` for kernel fusion
- Boolean output is memory efficient (1 byte per element)

**MLX Porting Considerations**:
- PyTorch `torch.eq` → MLX `mx.equal`
- Both support `==` operator overload
- MLX is lazy: comparison deferred until result accessed

---

### ne (Not Equal)

**Purpose**: Element-wise inequality comparison (!=)

**Signature**:
- `ne.Scalar(Tensor self, Scalar other) -> Tensor`
- `ne.Tensor(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9121-9147):
```yaml
- func: ne.Scalar(Tensor self, Scalar other) -> Tensor
  structured_delegate: ne.Scalar_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: ne_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: ne_scalar_nested
  tags: [core, pointwise]

- func: ne.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: ne.Tensor_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: ne_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: ne_tensor_nested
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] != other[i])
result[i] = !(self[i] == other[i])  # Logical NOT of equality
```

**CPU Implementation**:
```cpp
void ne_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "ne_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
      return a != b;
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def ne(x, y):
    """
    PyTorch: torch.ne(x, y) or x != y
    MLX: mx.not_equal(x, y) or x != y
    """
    return mx.not_equal(x, y)

# Example
x = mx.array([1, 2, 3])
y = mx.array([1, 0, 3])
result = x != y  # [False, True, False]
```

**Common Patterns**:
```python
# Pattern 1: Filter non-zero elements
data = torch.tensor([0, 1, 0, 2, 0, 3])
non_zero = data[data != 0]  # tensor([1, 2, 3])

# Pattern 2: Find mismatches
pred = torch.tensor([0, 1, 1, 0])
true_labels = torch.tensor([0, 1, 0, 0])
errors = (pred != true_labels).sum()  # 1

# Pattern 3: Exclude specific value
x = torch.randn(100)
filtered = x[x != 0.0]  # Remove zeros

# Pattern 4: Data validation
assert (labels != -1).all(), "Labels contain invalid -1"
```

**Edge Cases**:
- **NaN**: `NaN != NaN` is True (opposite of `==`)
- **Inverse of eq**: `ne(a, b) == ~eq(a, b)` (logical NOT)

**MLX Porting Considerations**:
- PyTorch `torch.ne` → MLX `mx.not_equal`
- Both support `!=` operator

---

### lt (Less Than)

**Purpose**: Element-wise less than comparison (<)

**Signature**:
- `lt.Scalar(Tensor self, Scalar other) -> Tensor`
- `lt.Tensor(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9413-9439):
```yaml
- func: lt.Scalar(Tensor self, Scalar other) -> Tensor
  structured_delegate: lt.Scalar_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: lt_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: lt_scalar_nested
  tags: [core, pointwise]

- func: lt.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: lt.Tensor_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: lt_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: lt_tensor_nested
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] < other[i])
result[i] = self[i] < other[i]
```

**CPU Implementation**:
```cpp
void lt_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
      return a < b;
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def lt(x, y):
    """
    PyTorch: torch.lt(x, y) or x < y
    MLX: mx.less(x, y) or x < y
    """
    return mx.less(x, y)

# Example
x = mx.array([1, 2, 3, 4])
y = mx.array([2, 2, 2, 2])
result = x < y  # [True, False, False, False]
```

**Common Patterns**:
```python
# Pattern 1: Threshold filtering
activations = torch.randn(100)
low_activations = activations[activations < 0.0]  # Negative values

# Pattern 2: Clipping mask
x = torch.randn(1000)
clip_mask = x < -1.0
x[clip_mask] = -1.0  # Clip to minimum

# Pattern 3: Confidence thresholding
confidence = torch.sigmoid(logits)
low_conf_mask = confidence < 0.5
predictions[low_conf_mask] = 0  # Set low confidence to 0

# Pattern 4: Early stopping check
val_loss = torch.tensor(0.15)
if val_loss < best_loss:
    best_loss = val_loss
    save_checkpoint()
```

**Edge Cases**:
- **NaN**: Any comparison with NaN is False
- **Complex numbers**: Not supported (no natural ordering)
- **Boolean**: False < True (0 < 1)

**MLX Porting Considerations**:
- PyTorch `torch.lt` → MLX `mx.less`
- Both support `<` operator

---

### le (Less Than or Equal)

**Purpose**: Element-wise less than or equal comparison (<=)

**Signature**:
- `le.Scalar(Tensor self, Scalar other) -> Tensor`
- `le.Tensor(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9286-9312):
```yaml
- func: le.Scalar(Tensor self, Scalar other) -> Tensor
  structured_delegate: le.Scalar_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: le_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: le_scalar_nested
  tags: [core, pointwise]

- func: le.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: le.Tensor_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: le_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: le_tensor_nested
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] <= other[i])
result[i] = (self[i] < other[i]) | (self[i] == other[i])
```

**CPU Implementation**:
```cpp
void le_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
      return a <= b;
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def le(x, y):
    """
    PyTorch: torch.le(x, y) or x <= y
    MLX: mx.less_equal(x, y) or x <= y
    """
    return mx.less_equal(x, y)

# Example
x = mx.array([1, 2, 3, 4])
y = mx.array([2, 2, 2, 2])
result = x <= y  # [True, True, False, False]
```

**Common Patterns**:
```python
# Pattern 1: Range validation
data = torch.randn(1000)
in_range = (data >= -1.0) & (data <= 1.0)
clipped = data[in_range]

# Pattern 2: Percentile filtering
sorted_vals = torch.sort(data)[0]
threshold = sorted_vals[int(len(sorted_vals) * 0.95)]
bottom_95 = data[data <= threshold]

# Pattern 3: Budget constraint
costs = torch.tensor([10.0, 15.0, 25.0, 30.0])
budget = 20.0
affordable = costs[costs <= budget]  # [10, 15]

# Pattern 4: Saturation check
output = model(x)
saturated = (output <= 0.0) | (output >= 1.0)
if saturated.any():
    warnings.warn("Output saturated")
```

**MLX Porting Considerations**:
- PyTorch `torch.le` → MLX `mx.less_equal`
- Both support `<=` operator

---

### gt (Greater Than)

**Purpose**: Element-wise greater than comparison (>)

**Signature**:
- `gt.Scalar(Tensor self, Scalar other) -> Tensor`
- `gt.Tensor(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9349-9376):
```yaml
- func: gt.Scalar(Tensor self, Scalar other) -> Tensor
  structured_delegate: gt.Scalar_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: gt_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: gt_scalar_nested
  tags: [core, pointwise]

- func: gt.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: gt.Tensor_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: gt_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: gt_tensor_nested
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] > other[i])
result[i] = self[i] > other[i]
```

**CPU Implementation**:
```cpp
void gt_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
      return a > b;
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def gt(x, y):
    """
    PyTorch: torch.gt(x, y) or x > y
    MLX: mx.greater(x, y) or x > y
    """
    return mx.greater(x, y)

# Example
x = mx.array([1, 2, 3, 4])
y = mx.array([2, 2, 2, 2])
result = x > y  # [False, False, True, True]
```

**Common Patterns**:
```python
# Pattern 1: Positive filtering
data = torch.randn(1000)
positive = data[data > 0.0]

# Pattern 2: Outlier detection
mean = data.mean()
std = data.std()
outliers = data[data > mean + 3*std]

# Pattern 3: Activation thresholding
activations = torch.relu(logits)
active = activations > 0.01  # Non-negligible activations

# Pattern 4: Performance metric
accuracy = correct / total
if accuracy > best_accuracy:
    best_accuracy = accuracy
    save_model()
```

**MLX Porting Considerations**:
- PyTorch `torch.gt` → MLX `mx.greater`
- Both support `>` operator

---

### ge (Greater Than or Equal)

**Purpose**: Element-wise greater than or equal comparison (>=)

**Signature**:
- `ge.Scalar(Tensor self, Scalar other) -> Tensor`
- `ge.Tensor(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9222-9248):
```yaml
- func: ge.Scalar(Tensor self, Scalar other) -> Tensor
  structured_delegate: ge.Scalar_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: ge_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: ge_scalar_nested
  tags: [core, pointwise]

- func: ge.Tensor(Tensor self, Tensor other) -> Tensor
  structured_delegate: ge.Tensor_out
  device_check: NoCheck
  variants: method, function
  dispatch:
    QuantizedCPU: ge_quantized_cpu
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: ge_tensor_nested
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] >= other[i])
result[i] = (self[i] > other[i]) | (self[i] == other[i])
```

**CPU Implementation**:
```cpp
void ge_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
      return a >= b;
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def ge(x, y):
    """
    PyTorch: torch.ge(x, y) or x >= y
    MLX: mx.greater_equal(x, y) or x >= y
    """
    return mx.greater_equal(x, y)

# Example
x = mx.array([1, 2, 3, 4])
y = mx.array([2, 2, 2, 2])
result = x >= y  # [False, True, True, True]
```

**Common Patterns**:
```python
# Pattern 1: Non-negative filtering
data = torch.randn(1000)
non_negative = data[data >= 0.0]

# Pattern 2: Threshold-based decision
scores = torch.sigmoid(logits)
predictions = (scores >= 0.5).float()  # Binary classification

# Pattern 3: Gradient clipping check
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
if grad_norm >= max_norm:
    logger.warning(f"Gradients clipped: {grad_norm}")

# Pattern 4: Convergence check
delta = torch.abs(loss - prev_loss)
converged = delta >= tolerance
```

**MLX Porting Considerations**:
- PyTorch `torch.ge` → MLX `mx.greater_equal`
- Both support `>=` operator

---

## Summary - Week 3-4 Day 1

**Operators Documented**: 6 basic comparison operators
- `eq` (==): Element-wise equality
- `ne` (!=): Element-wise inequality
- `lt` (<): Element-wise less than
- `le` (<=): Element-wise less than or equal
- `gt` (>): Element-wise greater than
- `ge` (>=): Element-wise greater than or equal

**Common Characteristics**:
- All return `torch.bool` tensors
- All support broadcasting
- All are non-differentiable
- All have direct MLX equivalents
- All are tagged `pointwise` for kernel fusion

**Operator Overloading**:
- `==` maps to `eq`
- `!=` maps to `ne`
- `<` maps to `lt`
- `<=` maps to `le`
- `>` maps to `gt`
- `>=` maps to `ge`

**PyTorch → MLX Mapping**:
- `torch.eq` → `mx.equal` or `==`
- `torch.ne` → `mx.not_equal` or `!=`
- `torch.lt` → `mx.less` or `<`
- `torch.le` → `mx.less_equal` or `<=`
- `torch.gt` → `mx.greater` or `>`
- `torch.ge` → `mx.greater_equal` or `>=`

**Progress**: 6 / 74 comparison operators documented (8%)
**Week 3-4 Day 1**: eq, ne, lt, le, gt, ge ✅
**Week 3-4 Day 2**: isnan, isinf, isfinite, isneginf, isposinf, isreal (pending)

---

## Week 3-4 Day 2 Operators - Special Value Checks

### isnan

**Purpose**: Check for Not-a-Number (NaN) values element-wise

**Signature**: `isnan(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:3231-3239):
```yaml
- func: isnan(Tensor self) -> Tensor
  variants: function, method
  device_check: NoCheck
  device_guard: False
  dispatch:
    CPU, CUDA, MPS, MTIA: isnan
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_isnan
    SparseCPU, SparseCUDA, SparseMPS: isnan_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: isnan_sparse_csr
  autogen: isnan.out
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] is NaN)
# Uses IEEE 754 property: NaN != NaN
result[i] = (self[i] != self[i])
```

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
Tensor isnan_cpu(const Tensor& self) {
  return at::_ops::isnan::call(self).to(kBool);
}

void isnan_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "isnan_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool {
      return std::isnan(a);  // C++ standard library
    });
  });
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor isnan_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);

  // Use Metal Performance Shaders isNaN operation
  MPSGraphTensor* result = [mpsGraph isNaNWithTensor:selfTensor
                                                name:@"isnan"];

  return createBoolTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
Not differentiable - discrete operation
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isnan(x):
    """
    PyTorch: torch.isnan(x)
    MLX: mx.isnan(x)
    """
    return mx.isnan(x)

# Example 1: Detect NaN values
x = mx.array([1.0, float('nan'), 3.0, float('nan')])
mask = mx.isnan(x)  # [False, True, False, True]

# Example 2: Count NaN values
x = mx.random.normal((100, 100))
x[x > 2] = float('nan')  # Introduce some NaNs
num_nans = mx.sum(mx.isnan(x))

# Example 3: Replace NaN with 0
x = mx.array([1.0, float('nan'), 3.0])
x = mx.where(mx.isnan(x), 0.0, x)  # [1.0, 0.0, 3.0]
```

**Common Patterns**:
```python
# Pattern 1: Validation - assert no NaNs
output = model(x)
assert not torch.isnan(output).any(), "Model produced NaN"

# Pattern 2: Filter out NaN values
data = torch.tensor([1.0, float('nan'), 3.0, 4.0, float('nan')])
clean_data = data[~torch.isnan(data)]  # [1.0, 3.0, 4.0]

# Pattern 3: Replace NaN with default value
x = torch.randn(100)
x[x > 2] = float('nan')
x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

# Pattern 4: Debugging numerical issues
gradients = [p.grad for p in model.parameters()]
nan_grads = [torch.isnan(g).any() for g in gradients if g is not None]
if any(nan_grads):
    print("NaN detected in gradients!")
```

**Edge Cases**:
- **Integer tensors**: Always returns all False (integers can't be NaN)
- **Complex tensors**: Checks both real and imaginary parts
- **NaN property**: NaN != NaN (unique among numbers)
- **Inf vs NaN**: isinf(inf)=True, isnan(inf)=False

**Performance Notes**:
- Fast vectorized operation
- Essential for numerical stability checks
- Common in training loop validation

**MLX Porting Considerations**:
- Direct equivalent: `mx.isnan()`
- Identical semantics

---

### isinf

**Purpose**: Check for positive or negative infinity element-wise

**Signature**: `isinf(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:13449-13460):
```yaml
- func: isinf(Tensor self) -> Tensor
  variants: function, method
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeExplicitAutograd: isinf
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_isinf
    SparseCPU, SparseCUDA, SparseMPS: isinf_sparse
    SparseMeta: isinf_sparse_meta
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: isinf_sparse_csr
  autogen: isinf.out
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] == +inf or self[i] == -inf)
result[i] = (abs(self[i]) == float('inf'))
```

**CPU Implementation**:
```cpp
Tensor isinf_cpu(const Tensor& self) {
  return isposinf(self).logical_or_(isneginf(self));
}

void isinf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "isinf_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool {
      return std::isinf(a);
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isinf(x):
    """
    PyTorch: torch.isinf(x)
    MLX: mx.isinf(x)
    """
    return mx.isinf(x)

# Example
x = mx.array([1.0, float('inf'), -float('inf'), 0.0])
mask = mx.isinf(x)  # [False, True, True, False]
```

**Common Patterns**:
```python
# Pattern 1: Detect overflow
loss = compute_loss(output, target)
if torch.isinf(loss):
    raise ValueError("Loss overflow - reduce learning rate")

# Pattern 2: Filter infinite values
data = torch.randn(1000)
data[data > 100] = float('inf')
finite_data = data[~torch.isinf(data)]

# Pattern 3: Clip to max finite value
x = torch.randn(100)
x = torch.where(torch.isinf(x), torch.finfo(x.dtype).max, x)

# Pattern 4: Gradient explosion check
for p in model.parameters():
    if p.grad is not None and torch.isinf(p.grad).any():
        print("Gradient explosion detected!")
        break
```

**Edge Cases**:
- **Positive infinity**: isinf(+inf) = True
- **Negative infinity**: isinf(-inf) = True
- **NaN**: isinf(NaN) = False
- **Large finite**: isinf(1e308) = False (unless it overflows to inf)

**MLX Porting Considerations**:
- Direct equivalent: `mx.isinf()`
- Catches both +inf and -inf

---

### isfinite

**Purpose**: Check if value is finite (not NaN, not infinity)

**Signature**: `isfinite(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:13443-13447):
```yaml
- func: isfinite(Tensor self) -> Tensor
  variants: function, method
  device_check: NoCheck
  device_guard: False
  tags: pointwise
```

**Algorithm**:
```python
# Element-wise: result[i] = not (isnan(self[i]) or isinf(self[i]))
result[i] = !isnan(self[i]) && !isinf(self[i])
```

**CPU Implementation**:
```cpp
Tensor isfinite_cpu(const Tensor& self) {
  return ~(isnan(self) | isinf(self));
}

void isfinite_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "isfinite_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool {
      return std::isfinite(a);
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isfinite(x):
    """
    PyTorch: torch.isfinite(x)
    MLX: mx.isfinite(x)
    """
    return mx.isfinite(x)

# Example
x = mx.array([1.0, float('nan'), float('inf'), 3.0])
mask = mx.isfinite(x)  # [True, False, False, True]
```

**Common Patterns**:
```python
# Pattern 1: Input validation
def forward(self, x):
    assert torch.isfinite(x).all(), "Input contains NaN or Inf"
    return self.model(x)

# Pattern 2: Safe computation
data = torch.randn(1000)
safe_data = data[torch.isfinite(data)]
mean = safe_data.mean()  # Safe from NaN/Inf contamination

# Pattern 3: Gradient health check
all_finite = all(
    torch.isfinite(p.grad).all()
    for p in model.parameters()
    if p.grad is not None
)
if not all_finite:
    optimizer.zero_grad()  # Skip this step

# Pattern 4: Mask-based replacement
x = torch.randn(100)
x[x > 10] = float('inf')
x[x < -10] = float('nan')
x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
```

**Edge Cases**:
- **Normal values**: isfinite(1.5) = True
- **Zero**: isfinite(0.0) = True
- **Infinity**: isfinite(±inf) = False
- **NaN**: isfinite(NaN) = False
- **Integers**: Always True (integers are always finite)

**MLX Porting Considerations**:
- Direct equivalent: `mx.isfinite()`
- Checks both NaN and infinity

---

### isposinf

**Purpose**: Check for positive infinity specifically

**Signature**: `isposinf(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:13467-13474):
```yaml
- func: isposinf(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: isposinf.out
  dispatch:
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_isposinf
    SparseCPU, SparseCUDA, SparseMPS: isposinf_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: isposinf_sparse_csr
  tags: pointwise
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] == +infinity)
result[i] = (self[i] == float('inf')) and (self[i] > 0)
```

**CPU Implementation**:
```cpp
void isposinf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "isposinf_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool {
      return a == std::numeric_limits<scalar_t>::infinity();
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isposinf(x):
    """
    PyTorch: torch.isposinf(x)
    MLX: mx.isinf(x) & (x > 0)

    MLX doesn't have isposinf, combine operations
    """
    return mx.isinf(x) & (x > 0)

# Example
x = mx.array([1.0, float('inf'), -float('inf'), float('nan')])
mask = isposinf(x)  # [False, True, False, False]
```

**Common Patterns**:
```python
# Pattern 1: Detect positive overflow
activations = torch.relu(logits)
if torch.isposinf(activations).any():
    print("Positive overflow in activations")

# Pattern 2: Asymmetric clipping
x = torch.randn(1000)
x = torch.where(torch.isposinf(x), 1e6, x)  # Clip only +inf
x = torch.where(torch.isneginf(x), -1e6, x)  # Clip only -inf separately

# Pattern 3: Direction-specific handling
loss = compute_loss(output, target)
if torch.isposinf(loss):
    # Positive overflow - likely numerical instability
    apply_gradient_clipping()
```

**Edge Cases**:
- **Positive infinity**: isposinf(+inf) = True
- **Negative infinity**: isposinf(-inf) = False
- **NaN**: isposinf(NaN) = False
- **Large positive**: isposinf(1e308) = False

**MLX Porting Considerations**:
- No direct MLX equivalent
- Implement as: `mx.isinf(x) & (x > 0)`
- Or: `mx.equal(x, float('inf'))`

---

### isneginf

**Purpose**: Check for negative infinity specifically

**Signature**: `isneginf(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:13485-13492):
```yaml
- func: isneginf(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: isneginf.out
  dispatch:
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_isneginf
    SparseCPU, SparseCUDA, SparseMPS: isneginf_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: isneginf_sparse_csr
  tags: pointwise
```

**Algorithm**:
```python
# Element-wise: result[i] = (self[i] == -infinity)
result[i] = (self[i] == -float('inf')) and (self[i] < 0)
```

**CPU Implementation**:
```cpp
void isneginf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "isneginf_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a) -> bool {
      return a == -std::numeric_limits<scalar_t>::infinity();
    });
  });
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isneginf(x):
    """
    PyTorch: torch.isneginf(x)
    MLX: mx.isinf(x) & (x < 0)
    """
    return mx.isinf(x) & (x < 0)

# Example
x = mx.array([1.0, float('inf'), -float('inf'), float('nan')])
mask = isneginf(x)  # [False, False, True, False]
```

**Common Patterns**:
```python
# Pattern 1: Detect negative underflow
log_probs = torch.log(probs)
if torch.isneginf(log_probs).any():
    print("Log of zero detected - numerical underflow")

# Pattern 2: Safe log computation
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log(probs.clamp(min=1e-10))  # Avoid -inf

# Pattern 3: Mask-based handling
attention_scores = torch.randn(10, 10)
attention_scores.masked_fill_(mask, float('-inf'))  # Intentional -inf
# Later check if any unexpected -inf appeared
unexpected = torch.isneginf(attention_scores) & ~mask
```

**Edge Cases**:
- **Negative infinity**: isneginf(-inf) = True
- **Positive infinity**: isneginf(+inf) = False
- **NaN**: isneginf(NaN) = False
- **Large negative**: isneginf(-1e308) = False

**MLX Porting Considerations**:
- No direct MLX equivalent
- Implement as: `mx.isinf(x) & (x < 0)`
- Or: `mx.equal(x, -float('inf'))`

---

### isreal

**Purpose**: Check if complex tensor has zero imaginary part

**Signature**: `isreal(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:3275):
```yaml
- func: isreal(Tensor self) -> Tensor
  variants: function, method
  tags: pointwise
```

**Algorithm**:
```python
# For complex tensors: result[i] = (self[i].imag == 0)
# For real tensors: always returns True
if self.is_complex():
    result[i] = (self[i].imag == 0)
else:
    result[i] = True
```

**CPU Implementation**:
```cpp
Tensor isreal_cpu(const Tensor& self) {
  if (!self.is_complex()) {
    return torch.ones_like(self, dtype=torch.bool);
  }

  // For complex: check imaginary part is zero
  return self.imag().eq(0);
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isreal(x):
    """
    PyTorch: torch.isreal(x)
    MLX: Check if imaginary part is zero

    Note: MLX complex support is limited
    """
    if mx.iscomplexobj(x):
        # For complex arrays, check imaginary part
        return mx.equal(mx.imag(x), 0)
    else:
        # Real tensors are always "real"
        return mx.ones(x.shape, dtype=mx.bool_)

# Example (if MLX supports complex)
x = mx.array([1+0j, 2+3j, 4+0j])
mask = isreal(x)  # [True, False, True]
```

**Common Patterns**:
```python
# Pattern 1: Filter real values from complex tensor
z = torch.randn(100, dtype=torch.complex64)
real_mask = torch.isreal(z)
real_values = z[real_mask]

# Pattern 2: Validate complex FFT output
spectrum = torch.fft.fft(signal)
# Check DC component is real
assert torch.isreal(spectrum[0]), "DC component should be real"

# Pattern 3: Complex to real conversion safety
complex_result = complex_operation(x)
if torch.isreal(complex_result).all():
    real_result = complex_result.real  # Safe to extract real part
```

**Edge Cases**:
- **Real tensors**: Always returns all True
- **Complex with imag=0**: Returns True
- **Complex with small imag**: Returns False (no tolerance)
- **NaN in imaginary**: isreal returns False

**Performance Notes**:
- For real tensors: O(1) - just allocates True tensor
- For complex tensors: O(n) - checks imaginary part

**MLX Porting Considerations**:
- MLX has limited complex number support
- May need to implement manually
- Check MLX documentation for current complex support status

---

## Summary - Week 3-4 Days 1-2

**Total Operators Documented**: 12 comparison operators

**Day 1 - Basic Comparisons** (6 ops):
- `eq`, `ne`, `lt`, `le`, `gt`, `ge`

**Day 2 - Special Value Checks** (6 ops):
- `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`, `isreal`

**Common Use Cases**:
- **Numerical stability**: Detect NaN/Inf during training
- **Gradient checking**: Validate gradients are finite
- **Data cleaning**: Filter invalid values
- **Debugging**: Identify sources of numerical issues

**IEEE 754 Properties**:
- NaN != NaN (unique property used in isnan)
- Inf > all finite values
- -Inf < all finite values
- Any operation with NaN produces NaN
- 1/0 = Inf, -1/0 = -Inf, 0/0 = NaN

**PyTorch → MLX Mapping**:
- Basic comparisons: `torch.{eq,ne,lt,le,gt,ge}` → `mx.{equal,not_equal,less,less_equal,greater,greater_equal}`
- Special checks: `torch.{isnan,isinf,isfinite}` → `mx.{isnan,isinf,isfinite}`
- Directional inf: `torch.{isposinf,isneginf}` → Manual combination in MLX
- Complex: `torch.isreal` → Check imaginary part in MLX

**Progress**: 12 / 74 comparison operators documented (16%)
**Week 3-4 Day 1**: eq, ne, lt, le, gt, ge ✅
**Week 3-4 Day 2**: isnan, isinf, isfinite, isneginf, isposinf, isreal ✅
**Week 3-4 Day 3**: equal, allclose, isclose (pending)
**Week 3-4 Day 4**: maximum, minimum (pending)

---

## Week 3-4 Day 3 Operators - Tolerant Comparisons

### equal

**Purpose**: Check if two tensors are exactly equal (all elements match)

**Signature**: `equal(Tensor self, Tensor other) -> bool`

**YAML Definition** (native_functions.yaml:10479-10487):
```yaml
- func: equal(Tensor self, Tensor other) -> bool
  tags: [data_dependent_output, pointwise]
  variants: method, function
  dispatch:
    CPU: cpu_equal
    CUDA: cuda_equal
    MPS: mps_equal
    QuantizedCPU: equal_quantized_cpu
```

**Algorithm**:
```python
# Returns single boolean (not tensor of bools like eq)
# 1. Check shapes match
if self.shape != other.shape:
    return False

# 2. Element-wise comparison
for i in range(numel):
    if self[i] != other[i]:
        return False  # Early exit on first mismatch
return True
```

**CPU Implementation** (native/TensorCompare.cpp):
```cpp
bool cpu_equal(const Tensor& self, const Tensor& other) {
  if (!self.is_same_size(other)) {
    return false;
  }
  
  // Fast path for same tensor
  if (self.is_same(other)) {
    return true;
  }
  
  // Element-wise comparison
  bool result = true;
  auto iter = TensorIteratorConfig()
    .add_input(self)
    .add_input(other)
    .build();
    
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    kHalf, kBFloat16, kBool, self.scalar_type(), "equal_cpu", [&] {
      cpu_serial_kernel(iter, [&](scalar_t a, scalar_t b) -> void {
        if (a != b) {
          result = false;
        }
      });
    });
  
  return result;
}
```

**MPS Implementation** (native/mps/operations/ComparisonOps.mm):
```objc
bool mps_equal(const Tensor& self, const Tensor& other) {
  if (!self.is_same_size(other)) {
    return false;
  }
  
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  MPSGraphTensor* otherTensor = getMPSGraphTensor(other);
  
  // Element-wise equality
  MPSGraphTensor* eqTensor = [mpsGraph equalWithPrimaryTensor:selfTensor
                                              secondaryTensor:otherTensor
                                                         name:@"equal"];
  
  // Reduce with AND (all must be true)
  MPSGraphTensor* allTrue = [mpsGraph reductionAndWithTensor:eqTensor
                                                         axes:nil
                                                         name:@"all"];
  
  return runMPSGraphAndFetchScalar<bool>(allTrue);
}
```

**Backward Pass**:
```
Not differentiable - returns scalar boolean
```

**MLX Equivalent**:
```python
import mlx.core as mx

def equal(x, y):
    """
    PyTorch: torch.equal(x, y)
    MLX: mx.array_equal(x, y)
    
    Note: Both return scalar bool
    """
    return mx.array_equal(x, y)

# Example 1: Exact equality check
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([1.0, 2.0, 3.0])
z = mx.array([1.0, 2.0, 3.001])

print(mx.array_equal(x, y))  # True
print(mx.array_equal(x, z))  # False

# Example 2: Shape mismatch
a = mx.array([[1, 2], [3, 4]])
b = mx.array([1, 2, 3, 4])
print(mx.array_equal(a, b))  # False (different shapes)

# Example 3: NaN handling
x = mx.array([1.0, float('nan')])
y = mx.array([1.0, float('nan')])
print(mx.array_equal(x, y))  # False (NaN != NaN)
```

**Common Patterns**:
```python
# Pattern 1: Tensor equality assertion in tests
def test_model():
    output = model(input)
    expected = load_expected()
    assert torch.equal(output, expected), "Output mismatch"

# Pattern 2: Cache validation
class CachedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_input = None
        self.cached_output = None
    
    def forward(self, x):
        # Only recompute if input changed
        if self.cached_input is None or not torch.equal(x, self.cached_input):
            self.cached_output = expensive_computation(x)
            self.cached_input = x.clone()
        return self.cached_output

# Pattern 3: Early stopping on convergence
prev_weights = model.state_dict()
optimizer.step()
new_weights = model.state_dict()

# Check if weights unchanged (converged)
all_equal = all(
    torch.equal(prev_weights[k], new_weights[k])
    for k in prev_weights.keys()
)
if all_equal:
    print("Model converged - weights unchanged")
    break

# Pattern 4: Gradient check debugging
# Numerical gradient
eps = 1e-5
numerical_grad = (f(x + eps) - f(x - eps)) / (2 * eps)

# Autograd gradient
x.requires_grad = True
loss = f(x)
loss.backward()
autograd_grad = x.grad

# Exact match unlikely, use allclose instead
if torch.equal(numerical_grad, autograd_grad):
    print("Exact match (rare)")
else:
    print("Use allclose for numerical comparison")
```

**Edge Cases**:
- **Shape mismatch**: Returns False immediately without element comparison
- **NaN values**: `equal(tensor([nan]), tensor([nan]))` returns False (NaN != NaN)
- **Floating point**: Requires exact binary equality, sensitive to rounding errors
- **Different dtypes**: Returns False if dtypes differ
- **Empty tensors**: `equal(tensor([]), tensor([]))` returns True
- **Zero-dimensional**: `equal(tensor(5), tensor(5))` returns True
- **Return type**: Returns Python bool, not Tensor (unlike eq)

**Performance Notes**:
- Early exit on shape mismatch (O(1))
- Early exit on first element mismatch
- Best case O(1), worst case O(n)
- CPU: Serial comparison with early termination
- MPS: Parallel reduction (less efficient for early exit)
- Use `eq()` if you need element-wise boolean tensor

**MLX Porting Considerations**:
- Direct equivalent: `mx.array_equal()`
- Identical semantics
- Both return scalar bool
- NaN handling identical (NaN != NaN)

---

### allclose

**Purpose**: Check if two tensors are element-wise equal within tolerances

**Signature**: `allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool`

**YAML Definition** (native_functions.yaml:748-753):
```yaml
- func: allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
  variants: function, method
  tags: data_dependent_output
  dispatch:
    CompositeExplicitAutograd: allclose
```

**Algorithm**:
```python
# Returns True if: |self[i] - other[i]| <= atol + rtol * |other[i]|
# for all elements i

def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    if self.shape != other.shape:
        return False
    
    # Element-wise check
    for i in range(numel):
        a, b = self[i], other[i]
        
        # NaN handling
        if equal_nan and isnan(a) and isnan(b):
            continue  # Treat NaN == NaN as True
        
        # Tolerance check
        if not (abs(a - b) <= atol + rtol * abs(b)):
            return False
    
    return True
```

**CPU Implementation** (native/TensorCompare.cpp):
```cpp
bool allclose(
    const Tensor& self,
    const Tensor& other,
    double rtol,
    double atol,
    bool equal_nan
) {
  // Implemented via isclose + all
  return isclose(self, other, rtol, atol, equal_nan).all().item<bool>();
}
```

**Backward Pass**:
```
Not differentiable - returns scalar boolean
```

**MLX Equivalent**:
```python
import mlx.core as mx

def allclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=False):
    """
    PyTorch: torch.allclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=False)
    MLX: mx.allclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=False)
    """
    return mx.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

# Example 1: Floating point comparison
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([1.0, 2.0, 3.00001])

print(mx.allclose(x, y))  # True (within default tolerance)
print(mx.allclose(x, y, rtol=1e-7, atol=1e-7))  # False (stricter)

# Example 2: Relative vs absolute tolerance
# rtol is relative to magnitude of 'other'
# atol is absolute threshold
x = mx.array([1000.0, 0.001])
y = mx.array([1000.1, 0.002])

# Default: rtol=1e-5, atol=1e-8
# For 1000: |1000.1 - 1000| = 0.1 <= 1e-8 + 1e-5 * 1000.1 = 0.010001 ✗
# For 0.001: |0.002 - 0.001| = 0.001 <= 1e-8 + 1e-5 * 0.002 = ~2e-8 ✗
print(mx.allclose(x, y))  # False

# Adjust tolerance
print(mx.allclose(x, y, rtol=1e-3, atol=1e-3))  # True

# Example 3: NaN handling
x = mx.array([1.0, float('nan'), 3.0])
y = mx.array([1.0, float('nan'), 3.0])

print(mx.allclose(x, y))  # False (NaN != NaN by default)
print(mx.allclose(x, y, equal_nan=True))  # True
```

**Common Patterns**:
```python
# Pattern 1: Numerical gradient checking
def check_gradients(model, input, eps=1e-4):
    # Compute numerical gradients
    numerical_grads = compute_numerical_gradients(model, input, eps)
    
    # Compute autograd gradients
    output = model(input)
    output.sum().backward()
    autograd_grads = [p.grad.clone() for p in model.parameters()]
    
    # Compare with tolerance
    for num_grad, auto_grad in zip(numerical_grads, autograd_grads):
        assert torch.allclose(num_grad, auto_grad, rtol=1e-3, atol=1e-5), \
            "Gradient mismatch detected"

# Pattern 2: Model output validation
def validate_model_port(pytorch_model, mlx_model, test_inputs):
    """Verify PyTorch → MLX port is correct"""
    for input_data in test_inputs:
        # Forward pass in both frameworks
        pytorch_out = pytorch_model(input_data)
        mlx_out = mlx_model(convert_to_mlx(input_data))
        
        # Check outputs match within tolerance
        assert torch.allclose(
            pytorch_out,
            convert_to_torch(mlx_out),
            rtol=1e-4,
            atol=1e-6
        ), "Port validation failed"

# Pattern 3: Training convergence check
prev_loss = float('inf')
for epoch in range(max_epochs):
    loss = train_epoch(model, dataloader)
    
    # Check if loss converged
    if prev_loss != float('inf'):
        loss_tensor = torch.tensor([loss])
        prev_tensor = torch.tensor([prev_loss])
        
        if torch.allclose(loss_tensor, prev_tensor, rtol=1e-6):
            print(f"Converged at epoch {epoch}")
            break
    
    prev_loss = loss

# Pattern 4: Optimizer state comparison
def compare_optimizer_states(opt1, opt2):
    """Check if two optimizer states are effectively equal"""
    state1 = opt1.state_dict()
    state2 = opt2.state_dict()
    
    for key in state1['state']:
        for param_key in state1['state'][key]:
            if isinstance(state1['state'][key][param_key], torch.Tensor):
                assert torch.allclose(
                    state1['state'][key][param_key],
                    state2['state'][key][param_key],
                    rtol=1e-5,
                    atol=1e-8
                ), f"Optimizer state mismatch at {key}/{param_key}"
```

**Edge Cases**:
- **Shape mismatch**: Returns False immediately
- **Infinity**: `allclose(tensor([inf]), tensor([inf]))` returns True
- **Mixed inf/finite**: `allclose(tensor([inf]), tensor([1e10]))` returns False
- **NaN with equal_nan=False**: Returns False even if all NaNs
- **NaN with equal_nan=True**: Treats matching NaN positions as equal
- **Zero values**: Uses absolute tolerance only (rtol * 0 = 0)
- **Negative values**: Uses absolute value in tolerance calculation

**Tolerance Formula Explained**:
```python
# Element i is considered "close" if:
|a[i] - b[i]| <= atol + rtol * |b[i]|

# atol (absolute tolerance):
#   - Minimum difference threshold
#   - Important for values near zero
#   - Default: 1e-8

# rtol (relative tolerance):
#   - Percentage of 'other' value
#   - Important for large values
#   - Default: 1e-5 (0.001%)

# Example: a=1000.01, b=1000.00, rtol=1e-5, atol=1e-8
|1000.01 - 1000.00| = 0.01
atol + rtol * |b| = 1e-8 + 1e-5 * 1000 = 0.01
0.01 <= 0.01  ✓ Close!
```

**Performance Notes**:
- Implemented as `isclose().all()`
- Early exit on shape mismatch
- No early exit during element comparison (must check all)
- CPU: O(n) serial comparison
- MPS: O(n) parallel reduction

**MLX Porting Considerations**:
- Direct equivalent: `mx.allclose()`
- Identical signature and semantics
- Same tolerance formula
- Same NaN handling behavior

---

### isclose

**Purpose**: Element-wise check if values are close within tolerances

**Signature**: `isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor`

**YAML Definition** (native_functions.yaml:3196-3198):
```yaml
- func: isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
  variants: function, method
```

**Algorithm**:
```python
# Element-wise version of allclose (returns boolean tensor, not scalar)
result[i] = |self[i] - other[i]| <= atol + rtol * |other[i]|

# With NaN handling:
if equal_nan and isnan(self[i]) and isnan(other[i]):
    result[i] = True
else:
    result[i] = |self[i] - other[i]| <= atol + rtol * |other[i]|
```

**CPU Implementation** (native/TensorCompare.cpp):
```cpp
Tensor isclose(
    const Tensor& self,
    const Tensor& other,
    double rtol,
    double atol,
    bool equal_nan
) {
  auto iter = TensorIteratorConfig()
    .add_output(Tensor())
    .add_input(self)
    .add_input(other)
    .build();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "isclose_cpu", [&] {
    cpu_kernel(iter, [rtol, atol, equal_nan](scalar_t a, scalar_t b) -> bool {
      // NaN handling
      if (equal_nan && std::isnan(a) && std::isnan(b)) {
        return true;
      }
      if (std::isnan(a) || std::isnan(b)) {
        return false;
      }
      
      // Infinity handling
      if (std::isinf(a) || std::isinf(b)) {
        return a == b;  // Both must be same infinity
      }
      
      // Tolerance check
      scalar_t diff = std::abs(a - b);
      scalar_t threshold = atol + rtol * std::abs(b);
      return diff <= threshold;
    });
  });
  
  return iter.output();
}
```

**Backward Pass**:
```
Not differentiable - discrete operation returning boolean tensor
```

**MLX Equivalent**:
```python
import mlx.core as mx

def isclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=False):
    """
    PyTorch: torch.isclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=False)
    MLX: mx.isclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=equal_nan)
    """
    return mx.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

# Example 1: Element-wise comparison
x = mx.array([1.0, 2.0, 3.0, 1000.0])
y = mx.array([1.0001, 2.0, 3.1, 1000.1])

mask = mx.isclose(x, y, rtol=1e-3, atol=1e-3)
# [True, True, False, True]
#  1.0 vs 1.0001: diff=0.0001 <= 0.001 + 0.001*1.0001 ✓
#  2.0 vs 2.0: exact match ✓
#  3.0 vs 3.1: diff=0.1 > 0.001 + 0.001*3.1 ✗
#  1000.0 vs 1000.1: diff=0.1 <= 0.001 + 0.001*1000.1 ✓

# Example 2: Filter approximately equal values
data = mx.random.normal((100,))
noisy = data + mx.random.normal((100,)) * 0.01  # Add 1% noise

# Find elements where noise is < 2%
close_mask = mx.isclose(data, noisy, rtol=0.02, atol=0.0)
num_close = mx.sum(close_mask)
print(f"{num_close} elements within 2% tolerance")

# Example 3: NaN handling
x = mx.array([1.0, float('nan'), 3.0, float('nan')])
y = mx.array([1.0, float('nan'), 3.0, 0.0])

mask1 = mx.isclose(x, y)  # [True, False, True, False]
mask2 = mx.isclose(x, y, equal_nan=True)  # [True, True, True, False]

# Example 4: Masking with isclose
predictions = mx.array([0.1, 0.5, 0.9, 0.3])
targets = mx.array([0.0, 0.5, 1.0, 0.3])

# Find predictions within 0.1 of target
close_enough = mx.isclose(predictions, targets, rtol=0.0, atol=0.1)
accuracy = mx.mean(close_enough.astype(mx.float32))
```

**Common Patterns**:
```python
# Pattern 1: Approximate equality masking
output = model(input)
target = load_target()

# Mask for "close enough" predictions
tolerance_mask = torch.isclose(output, target, rtol=0.01, atol=0.01)
accuracy = tolerance_mask.float().mean()

# Pattern 2: Numerical stability debugging
# Find where computations diverge
reference = reference_implementation(x)
optimized = optimized_implementation(x)

# Locate divergent elements
divergent = ~torch.isclose(reference, optimized, rtol=1e-4, atol=1e-6)
if divergent.any():
    indices = divergent.nonzero()
    print(f"Divergence at indices: {indices}")
    print(f"Reference: {reference[divergent]}")
    print(f"Optimized: {optimized[divergent]}")

# Pattern 3: Gradient magnitude filtering
# Zero out small gradients to reduce noise
for param in model.parameters():
    if param.grad is not None:
        # Find gradients close to zero
        near_zero = torch.isclose(
            param.grad,
            torch.zeros_like(param.grad),
            rtol=0.0,
            atol=1e-6
        )
        # Zero them out
        param.grad[near_zero] = 0.0

# Pattern 4: Loss plateau detection
loss_history = torch.tensor([...])  # Recent loss values

# Check if all recent losses are close to each other (plateau)
if len(loss_history) >= 10:
    recent = loss_history[-10:]
    all_close_to_last = torch.isclose(
        recent,
        recent[-1].expand_as(recent),
        rtol=1e-4,
        atol=1e-6
    )
    
    if all_close_to_last.all():
        print("Loss plateau detected - consider reducing learning rate")
```

**Edge Cases**:
- **Broadcasting**: Tensors broadcast like other binary ops
- **Infinity**: `isclose(tensor([inf]), tensor([inf]))` = True
- **Inf vs finite**: Always False regardless of tolerance
- **NaN**: Returns False unless `equal_nan=True` and both are NaN
- **Zero comparison**: Uses only `atol` since `rtol * 0 = 0`
- **Different signs**: Can be close if absolute difference small enough
- **Subnormal numbers**: Handled correctly with absolute tolerance

**Relationship to Other Ops**:
```python
# allclose is reduction of isclose
torch.allclose(a, b, rtol, atol, equal_nan) == torch.isclose(a, b, rtol, atol, equal_nan).all()

# equal is stricter than allclose
if torch.equal(a, b):
    assert torch.allclose(a, b)  # Always true
# But allclose doesn't imply equal:
a = torch.tensor([1.0])
b = torch.tensor([1.000001])
assert torch.allclose(a, b)  # True
assert not torch.equal(a, b)  # True
```

**Performance Notes**:
- Element-wise operation (pointwise tag)
- Eligible for kernel fusion
- CPU: Vectorized SIMD operations
- MPS: Parallel GPU execution
- More expensive than `eq` due to tolerance calculation

**MLX Porting Considerations**:
- Direct equivalent: `mx.isclose()`
- Identical semantics and signature
- Same tolerance formula
- Same broadcasting behavior

---

## Week 3-4 Day 4 Operators - Element-wise Min/Max

### maximum

**Purpose**: Element-wise maximum of two tensors (propagates NaN)

**Signature**: `maximum(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:10270-10283):
```yaml
- func: maximum(Tensor self, Tensor other) -> Tensor
  structured_delegate: maximum.out
  device_check: NoCheck   # TensorIterator
  variants: method, function
  tags: [core, pointwise]

- func: maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MTIA: maximum_out
    MPS: maximum_out_mps
  tags: pointwise
```

**Algorithm**:
```python
# Element-wise: result[i] = max(self[i], other[i])
# NaN propagation: if either is NaN, result is NaN

for i in range(numel):
    a, b = self[i], other[i]
    if isnan(a) or isnan(b):
        result[i] = NaN
    else:
        result[i] = max(a, b)
```

**CPU Implementation** (native/BinaryOps.cpp):
```cpp
void maximum_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "maximum_cpu", [&] {
    cpu_kernel_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t {
        // NaN propagation
        if (std::isnan(a) || std::isnan(b)) {
          return std::numeric_limits<scalar_t>::quiet_NaN();
        }
        return std::max(a, b);
      },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
        // Vectorized version
        return at::vec::maximum(a, b);  // Handles NaN propagation
      }
    );
  });
}
```

**MPS Implementation** (native/mps/operations/BinaryOps.mm):
```objc
Tensor maximum_out_mps(const Tensor& self, const Tensor& other, Tensor& out) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  MPSGraphTensor* otherTensor = getMPSGraphTensor(other);
  
  // MPS maximum propagates NaN
  MPSGraphTensor* result = [mpsGraph maximumWithPrimaryTensor:selfTensor
                                              secondaryTensor:otherTensor
                                                         name:@"maximum"];
  
  return createTensorFromMPSGraph(result, out);
}
```

**Backward Pass**:
```
∂maximum/∂self[i] = 1 if self[i] > other[i], else 0
∂maximum/∂other[i] = 1 if other[i] > self[i], else 0

# When self[i] == other[i], convention is to give gradient to self
# When either is NaN, gradient is NaN

Mathematical derivation:
Let z = maximum(x, y)

For element i:
  if x[i] > y[i]:
    ∂z[i]/∂x[i] = 1, ∂z[i]/∂y[i] = 0
  elif y[i] > x[i]:
    ∂z[i]/∂x[i] = 0, ∂z[i]/∂y[i] = 1
  elif x[i] == y[i]:
    # Convention: give gradient to first argument
    ∂z[i]/∂x[i] = 1, ∂z[i]/∂y[i] = 0
  else:  # At least one is NaN
    ∂z[i]/∂x[i] = NaN, ∂z[i]/∂y[i] = NaN
```

**Backward Implementation**:
```cpp
Tensor maximum_backward_self(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    const Tensor& output
) {
  // Gradient flows to self where self >= other
  auto mask = self.ge(other);  // self >= other
  return grad * mask;
}

Tensor maximum_backward_other(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    const Tensor& output
) {
  // Gradient flows to other where other > self
  auto mask = other.gt(self);  // other > self (strict)
  return grad * mask;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def maximum(x, y):
    """
    PyTorch: torch.maximum(x, y)
    MLX: mx.maximum(x, y)
    """
    return mx.maximum(x, y)

# Example 1: Basic element-wise maximum
x = mx.array([1.0, 5.0, 3.0, 8.0])
y = mx.array([2.0, 3.0, 6.0, 1.0])
result = mx.maximum(x, y)  # [2.0, 5.0, 6.0, 8.0]

# Example 2: Broadcasting
x = mx.array([[1, 2], [3, 4]])
y = mx.array([2.5, 2.5])
result = mx.maximum(x, y)
# [[2.5, 2.5],
#  [3.0, 4.0]]

# Example 3: NaN propagation
x = mx.array([1.0, float('nan'), 3.0])
y = mx.array([2.0, 2.0, 2.0])
result = mx.maximum(x, y)  # [2.0, nan, 3.0]

# Example 4: Clamping to lower bound
values = mx.array([-5.0, 0.0, 3.0, 10.0])
min_value = mx.array([0.0])  # Scalar broadcast
clamped = mx.maximum(values, min_value)  # [0.0, 0.0, 3.0, 10.0]
```

**Common Patterns**:
```python
# Pattern 1: ReLU implementation
def relu(x):
    """ReLU: max(0, x)"""
    return torch.maximum(x, torch.zeros_like(x))
    # Or more efficiently: torch.clamp(x, min=0)

# Pattern 2: Clamp to minimum value
def clamp_min(x, min_val):
    """Ensure all values >= min_val"""
    return torch.maximum(x, torch.full_like(x, min_val))

# Pattern 3: Safe division (avoid division by very small numbers)
numerator = torch.randn(100)
denominator = torch.randn(100)

# Ensure denominator magnitude >= 1e-6
safe_denom = torch.maximum(
    torch.abs(denominator),
    torch.tensor(1e-6)
) * torch.sign(denominator)
result = numerator / safe_denom

# Pattern 4: Attention masking (combine with -inf)
# Compute pairwise maximum attention scores
scores1 = compute_attention_scores(q1, k, v)  # [batch, seq, dim]
scores2 = compute_attention_scores(q2, k, v)

# Take best of two query mechanisms
combined = torch.maximum(scores1, scores2)

# Pattern 5: Gradient clipping element-wise
# Ensure gradients have minimum magnitude
for param in model.parameters():
    if param.grad is not None:
        # Clip gradients to have magnitude >= 1e-8
        sign = torch.sign(param.grad)
        magnitude = torch.abs(param.grad)
        clipped_mag = torch.maximum(magnitude, torch.tensor(1e-8))
        param.grad = sign * clipped_mag
```

**Edge Cases**:
- **NaN propagation**: `maximum(nan, x) = nan`, `maximum(x, nan) = nan`
- **Infinity**: `maximum(inf, x) = inf` for any finite x
- **Equal values**: `maximum(x, x)` returns x, gradient goes to first arg
- **Zero**: `maximum(0.0, -0.0)` returns 0.0 (positive zero)
- **Broadcasting**: Works like other binary ops
- **Integer tensors**: Works but no gradients

**Comparison with torch.max**:
```python
# torch.maximum: element-wise binary operation
torch.maximum(a, b)  # Returns tensor same shape as broadcast(a, b)

# torch.max: reduction operation OR element-wise
torch.max(a)  # Reduction: returns single maximum value
torch.max(a, dim=0)  # Reduction along dimension
torch.max(a, b)  # Alias for torch.maximum (binary)

# Use maximum when you want element-wise comparison
# Use max when you want reduction
```

**Performance Notes**:
- Pointwise operation - eligible for kernel fusion
- CPU: Vectorized SIMD implementation
- MPS: Parallel GPU execution
- NaN check adds small overhead vs simple comparison

**MLX Porting Considerations**:
- Direct equivalent: `mx.maximum()`
- Identical semantics including NaN propagation
- Same broadcasting behavior
- Gradient behavior identical

---

### minimum

**Purpose**: Element-wise minimum of two tensors (propagates NaN)

**Signature**: `minimum(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:10303-10316):
```yaml
- func: minimum(Tensor self, Tensor other) -> Tensor
  structured_delegate: minimum.out
  device_check: NoCheck   # TensorIterator
  variants: method, function
  tags: [core, pointwise]

- func: minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MTIA: minimum_out
    MPS: minimum_out_mps
  tags: pointwise
```

**Algorithm**:
```python
# Element-wise: result[i] = min(self[i], other[i])
# NaN propagation: if either is NaN, result is NaN

for i in range(numel):
    a, b = self[i], other[i]
    if isnan(a) or isnan(b):
        result[i] = NaN
    else:
        result[i] = min(a, b)
```

**CPU Implementation** (native/BinaryOps.cpp):
```cpp
void minimum_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "minimum_cpu", [&] {
    cpu_kernel_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t {
        // NaN propagation
        if (std::isnan(a) || std::isnan(b)) {
          return std::numeric_limits<scalar_t>::quiet_NaN();
        }
        return std::min(a, b);
      },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
        // Vectorized version
        return at::vec::minimum(a, b);  // Handles NaN propagation
      }
    );
  });
}
```

**MPS Implementation** (native/mps/operations/BinaryOps.mm):
```objc
Tensor minimum_out_mps(const Tensor& self, const Tensor& other, Tensor& out) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  MPSGraphTensor* otherTensor = getMPSGraphTensor(other);
  
  // MPS minimum propagates NaN
  MPSGraphTensor* result = [mpsGraph minimumWithPrimaryTensor:selfTensor
                                              secondaryTensor:otherTensor
                                                         name:@"minimum"];
  
  return createTensorFromMPSGraph(result, out);
}
```

**Backward Pass**:
```
∂minimum/∂self[i] = 1 if self[i] < other[i], else 0
∂minimum/∂other[i] = 1 if other[i] < self[i], else 0

# When self[i] == other[i], convention is to give gradient to self
# When either is NaN, gradient is NaN

Mathematical derivation:
Let z = minimum(x, y)

For element i:
  if x[i] < y[i]:
    ∂z[i]/∂x[i] = 1, ∂z[i]/∂y[i] = 0
  elif y[i] < x[i]:
    ∂z[i]/∂x[i] = 0, ∂z[i]/∂y[i] = 1
  elif x[i] == y[i]:
    # Convention: give gradient to first argument
    ∂z[i]/∂x[i] = 1, ∂z[i]/∂y[i] = 0
  else:  # At least one is NaN
    ∂z[i]/∂x[i] = NaN, ∂z[i]/∂y[i] = NaN
```

**Backward Implementation**:
```cpp
Tensor minimum_backward_self(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    const Tensor& output
) {
  // Gradient flows to self where self <= other
  auto mask = self.le(other);  // self <= other
  return grad * mask;
}

Tensor minimum_backward_other(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    const Tensor& output
) {
  // Gradient flows to other where other < self
  auto mask = other.lt(self);  // other < self (strict)
  return grad * mask;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def minimum(x, y):
    """
    PyTorch: torch.minimum(x, y)
    MLX: mx.minimum(x, y)
    """
    return mx.minimum(x, y)

# Example 1: Basic element-wise minimum
x = mx.array([1.0, 5.0, 3.0, 8.0])
y = mx.array([2.0, 3.0, 6.0, 1.0])
result = mx.minimum(x, y)  # [1.0, 3.0, 3.0, 1.0]

# Example 2: Broadcasting
x = mx.array([[1, 2], [3, 4]])
y = mx.array([2.5, 2.5])
result = mx.minimum(x, y)
# [[1.0, 2.0],
#  [2.5, 2.5]]

# Example 3: NaN propagation
x = mx.array([1.0, float('nan'), 3.0])
y = mx.array([2.0, 2.0, 2.0])
result = mx.minimum(x, y)  # [1.0, nan, 2.0]

# Example 4: Clamping to upper bound
values = mx.array([-5.0, 0.0, 3.0, 10.0])
max_value = mx.array([5.0])  # Scalar broadcast
clamped = mx.minimum(values, max_value)  # [-5.0, 0.0, 3.0, 5.0]
```

**Common Patterns**:
```python
# Pattern 1: Clamp to maximum value
def clamp_max(x, max_val):
    """Ensure all values <= max_val"""
    return torch.minimum(x, torch.full_like(x, max_val))

# Pattern 2: Soft clipping (differentiable alternative to hard clip)
# Gradually compress values above threshold
threshold = 10.0
x = torch.randn(100) * 20  # Some values > 10

soft_clipped = threshold + torch.log(
    1 + torch.minimum(x - threshold, torch.zeros_like(x))
)

# Pattern 3: Attention score clipping
# Prevent attention scores from getting too large
attention_scores = query @ key.T / math.sqrt(d_k)

# Clip to prevent softmax overflow
max_score = 20.0
clipped_scores = torch.minimum(
    attention_scores,
    torch.tensor(max_score)
)
attention_weights = torch.softmax(clipped_scores, dim=-1)

# Pattern 4: Learning rate warmup
def get_lr_with_warmup(step, warmup_steps, max_lr):
    # Linear warmup, then constant
    warmup_lr = (step / warmup_steps) * max_lr
    return torch.minimum(
        torch.tensor(warmup_lr),
        torch.tensor(max_lr)
    ).item()

# Pattern 5: Gradient clipping (element-wise upper bound)
max_grad = 5.0
for param in model.parameters():
    if param.grad is not None:
        param.grad = torch.minimum(
            param.grad,
            torch.tensor(max_grad)
        )
        param.grad = torch.maximum(
            param.grad,
            torch.tensor(-max_grad)
        )

# Pattern 6: Safe log computation
# Prevent log(x) from going to -inf for small x
x = torch.rand(100)
min_val = 1e-10
safe_x = torch.minimum(x, torch.ones_like(x))  # Clip to [0, 1]
safe_x = torch.maximum(safe_x, torch.tensor(min_val))  # Avoid log(0)
log_result = torch.log(safe_x)
```

**Edge Cases**:
- **NaN propagation**: `minimum(nan, x) = nan`, `minimum(x, nan) = nan`
- **Infinity**: `minimum(-inf, x) = -inf` for any finite x
- **Equal values**: `minimum(x, x)` returns x, gradient goes to first arg
- **Zero**: `minimum(0.0, -0.0)` returns -0.0 (negative zero)
- **Broadcasting**: Works like other binary ops
- **Integer tensors**: Works but no gradients

**Comparison with torch.min**:
```python
# torch.minimum: element-wise binary operation
torch.minimum(a, b)  # Returns tensor same shape as broadcast(a, b)

# torch.min: reduction operation OR element-wise
torch.min(a)  # Reduction: returns single minimum value
torch.min(a, dim=0)  # Reduction along dimension
torch.min(a, b)  # Alias for torch.minimum (binary)

# Use minimum when you want element-wise comparison
# Use min when you want reduction
```

**Symmetric Relationship**:
```python
# minimum and maximum are symmetric
assert torch.equal(
    torch.minimum(a, b),
    -torch.maximum(-a, -b)
)

# Combined for clamping
def clamp(x, min_val, max_val):
    # First clamp to min, then to max
    x = torch.maximum(x, torch.tensor(min_val))
    x = torch.minimum(x, torch.tensor(max_val))
    return x

# Built-in version:
torch.clamp(x, min=min_val, max=max_val)
```

**Performance Notes**:
- Pointwise operation - eligible for kernel fusion
- CPU: Vectorized SIMD implementation
- MPS: Parallel GPU execution
- NaN check adds small overhead vs simple comparison
- Often fused with ReLU and clamp operations

**MLX Porting Considerations**:
- Direct equivalent: `mx.minimum()`
- Identical semantics including NaN propagation
- Same broadcasting behavior
- Gradient behavior identical

---

## Summary - Week 3-4 Complete

**Total Operators Documented**: 17 comparison operators

**Day 1 - Basic Comparisons** (6 ops):
- `eq`, `ne`, `lt`, `le`, `gt`, `ge`

**Day 2 - Special Value Checks** (6 ops):
- `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`, `isreal`

**Day 3 - Tolerant Comparisons** (3 ops):
- `equal`, `allclose`, `isclose`

**Day 4 - Element-wise Min/Max** (2 ops):
- `maximum`, `minimum`

**Common Use Cases**:
- **Numerical stability**: Detect NaN/Inf during training
- **Gradient checking**: Validate gradients are finite
- **Data cleaning**: Filter invalid values
- **Debugging**: Identify sources of numerical issues
- **Tolerance comparison**: Check approximate equality
- **Clamping**: Limit values to ranges
- **ReLU**: Element-wise max with zero

**IEEE 754 Properties**:
- NaN != NaN (unique property used in isnan)
- Inf > all finite values
- -Inf < all finite values
- Any operation with NaN produces NaN
- 1/0 = Inf, -1/0 = -Inf, 0/0 = NaN
- NaN propagation in maximum/minimum

**Tolerance Formula** (allclose/isclose):
```
|a - b| <= atol + rtol * |b|

atol: absolute tolerance (important near zero)
rtol: relative tolerance (important for large values)
```

**PyTorch → MLX Mapping**:
- Basic comparisons: `torch.{eq,ne,lt,le,gt,ge}` → `mx.{equal,not_equal,less,less_equal,greater,greater_equal}`
- Special checks: `torch.{isnan,isinf,isfinite}` → `mx.{isnan,isinf,isfinite}`
- Directional inf: `torch.{isposinf,isneginf}` → `mx.isinf(x) & (x > 0)` / `mx.isinf(x) & (x < 0)`
- Complex: `torch.isreal` → Check imaginary part in MLX
- Tolerant: `torch.{equal,allclose,isclose}` → `mx.{array_equal,allclose,isclose}`
- Min/Max: `torch.{maximum,minimum}` → `mx.{maximum,minimum}`

**Return Type Distinctions**:
- **Element-wise boolean tensor**: eq, ne, lt, le, gt, ge, isnan, isinf, isfinite, isposinf, isneginf, isreal, isclose
- **Scalar boolean**: equal, allclose
- **Same dtype tensor**: maximum, minimum

**Gradient Properties**:
- **Non-differentiable** (discrete): All comparison operators returning bool
- **Differentiable**: maximum, minimum (piecewise gradients)

**Progress**: 17 / 74 comparison operators documented (23%)
**Week 3-4 Day 1**: eq, ne, lt, le, gt, ge ✅
**Week 3-4 Day 2**: isnan, isinf, isfinite, isneginf, isposinf, isreal ✅
**Week 3-4 Day 3**: equal, allclose, isclose ✅
**Week 3-4 Day 4**: maximum, minimum ✅

**Weeks 3-4 Status**: Complete ✅

