# Arithmetic Operators

## Purpose

Arithmetic operators form the foundation of tensor computations, providing element-wise mathematical operations. This document covers Tier 1 arithmetic operators essential for basic tensor operations and ML workloads.

**Tier 1 Arithmetic Operators** (12 total):
- `add.Tensor`, `add.Scalar`
- `sub.Tensor`, `sub.Scalar`
- `mul.Tensor`, `mul.Scalar`
- `div.Tensor`, `div.Scalar`
- `neg`
- `abs`
- `pow.Tensor_Scalar`
- `sqrt`

## Common Properties

All arithmetic operators share these characteristics:

**Tags**: `[core, pointwise]` for most variants

**Broadcasting**: Supported via TensorIterator

**Type Promotion**: Follows PyTorch's type promotion rules
- Integer + Float → Float
- Lower precision + Higher precision → Higher precision
- Complex + Real → Complex

**Variants**:
- **Function**: `torch.add(a, b)` - returns new tensor
- **Method**: `a.add(b)` - returns new tensor
- **Inplace**: `a.add_(b)` - modifies `a` in-place
- **Out**: `torch.add(a, b, out=c)` - writes to pre-allocated tensor

**Backend Support**: CPU, CUDA, MPS (Metal), Meta

## Operator Details

### add (Addition)

**Purpose**: Element-wise addition with optional scaling

**Signature**:
```python
add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
```

**Formula**: `out = self + alpha * other`

**YAML Definition** (`native_functions.yaml:554-564`):
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator handles it
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS, SparseMeta: add_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [core, pointwise]
```

**Meta Function** (`BinaryOps.cpp:151-156`):
```cpp
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
```

**Shape Inference**:
- Broadcasts `self` and `other` to common shape
- Output shape: `broadcast(self.shape, other.shape)`
- Example: `(3, 1) + (1, 4)` → `(3, 4)`

**Type Promotion Examples**:
```python
torch.add(torch.tensor([1, 2], dtype=torch.int32),
          torch.tensor([1.5, 2.5], dtype=torch.float32))
# → dtype: torch.float32

torch.add(torch.tensor([1.0], dtype=torch.float16),
          torch.tensor([2.0], dtype=torch.float32))
# → dtype: torch.float32
```

**MPS Implementation** (`mps/operations/BinaryOps.mm`):
```objective-c
Tensor& add_out_mps(const Tensor& self, const Tensor& other,
                    const Scalar& alpha, Tensor& output) {
  binaryOpTensor(self, other, output, "add_out_mps",
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
      MPSGraph* mpsGraph = cachedGraph->graph();

      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                           shape:@[@1]
                                                        dataType:primaryCastTensor.dataType];

      MPSGraphTensor* scaledSecondary = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                                                  secondaryTensor:alphaTensor
                                                                             name:nil];

      return [mpsGraph additionWithPrimaryTensor:primaryCastTensor
                                 secondaryTensor:scaledSecondary
                                            name:nil];
    });
  return output;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def add(a, b, alpha=1):
    """MLX implementation of torch.add"""
    if alpha != 1:
        b = mx.multiply(b, alpha)
    return mx.add(a, b)
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: grad
  other: alpha * grad
```

**Usage Examples**:
```python
# Tensor + Tensor
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
torch.add(a, b)  # [[6, 8], [10, 12]]

# With alpha scaling
torch.add(a, b, alpha=2)  # [[11, 14], [17, 20]]  (a + 2*b)

# Broadcast
a = torch.tensor([[1], [2], [3]])  # (3, 1)
b = torch.tensor([10, 20])  # (2,)
torch.add(a, b)  # (3, 2): [[11, 21], [12, 22], [13, 23]]

# Scalar variant
torch.add(a, 5)  # [[6, 7], [8, 9]]

# Inplace
a.add_(b)  # Modifies a

# Pre-allocated output
c = torch.empty((2, 2))
torch.add(a, b, out=c)
```

**Implementation Notes**:
- `alpha` parameter is PyTorch-specific (not in NumPy)
- Common in BLAS operations (e.g., `axpy`: `y = alpha*x + y`)
- MPS uses MPSGraph multiplication + addition nodes
- CPU vectorizes when possible

---

### sub (Subtraction)

**Purpose**: Element-wise subtraction with optional scaling

**Signature**:
```python
sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
```

**Formula**: `out = self - alpha * other`

**YAML Definition** (`native_functions.yaml:~600`):
```yaml
- func: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  structured_delegate: sub.out
  variants: function, method
  tags: [core, pointwise]
```

**Meta Function** (`BinaryOps.cpp:158-164`):
```cpp
TORCH_META_FUNC2(sub, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  native::sub_check(self, other);
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
```

**MLX Equivalent**:
```python
def sub(a, b, alpha=1):
    if alpha != 1:
        b = mx.multiply(b, alpha)
    return mx.subtract(a, b)
```

**Gradient**:
```yaml
- name: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: grad
  other: -alpha * grad
```

**Usage Examples**:
```python
a = torch.tensor([5, 6, 7])
b = torch.tensor([1, 2, 3])
torch.sub(a, b)  # [4, 4, 4]

torch.sub(a, b, alpha=2)  # [3, 2, 1]  (a - 2*b)
```

---

### mul (Multiplication)

**Purpose**: Element-wise multiplication

**Signature**:
```python
mul.Tensor(Tensor self, Tensor other) -> Tensor
mul.Scalar(Tensor self, Scalar other) -> Tensor
```

**Formula**: `out = self * other`

**YAML Definition** (`native_functions.yaml:4322-4332`):
```yaml
- func: mul.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck
  structured_delegate: mul.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: mul_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: mul_sparse_csr
    MkldnnCPU: mkldnn_mul
    ZeroTensor: mul_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_mul_Tensor
  tags: [core, pointwise]
```

**CPU Kernel** (`cpu/BinaryOpsKernel.cpp:126-170`):
```cpp
void mul_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (dtype == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(dtype, "mul_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a * b;  // SIMD multiplication
          });
    });
  }
}
```

**MLX Equivalent**:
```python
def mul(a, b):
    return mx.multiply(a, b)
```

**Gradient**:
```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad * other
  other: grad * self
```

**Special Cases**:
- **Boolean tensors**: Logical AND (`a && b`)
- **Complex tensors**: Complex multiplication
- **Sparse tensors**: Specialized sparse kernels

**Usage Examples**:
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
torch.mul(a, b)  # [4, 10, 18]

# Scalar multiplication (scaling)
torch.mul(a, 2)  # [2, 4, 6]
```

---

### div (Division)

**Purpose**: Element-wise division with optional rounding mode

**Signature**:
```python
div.Tensor(Tensor self, Tensor other) -> Tensor
div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
div.Scalar(Tensor self, Scalar other) -> Tensor
```

**Formula**:
- Default: `out = self / other` (true division, promotes to float)
- `rounding_mode="trunc"`: Truncated division toward zero
- `rounding_mode="floor"`: Floor division toward -∞

**YAML Definition** (`native_functions.yaml:2172-2180`):
```yaml
- func: div.Tensor(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: div.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: div_sparse
    ZeroTensor: div_zerotensor
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_div_Tensor
  tags: [core, pointwise]
```

**Meta Function** (`BinaryOps.cpp:172-189`):
```cpp
TORCH_META_FUNC2(div, Tensor_mode) (
  const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode
) {
  if (!rounding_mode.has_value()) {
    build_borrowing_binary_float_op(maybe_get_output(), self, other);
  } else if (*rounding_mode == "trunc") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else if (*rounding_mode == "floor") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else {
    TORCH_CHECK(false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found '", *rounding_mode, "'");
  }
}
```

**Type Promotion**:
- **No rounding_mode**: Always promotes to float (even int/int)
- **With rounding_mode**: Preserves integer dtype

**MLX Equivalent**:
```python
def div(a, b, rounding_mode=None):
    result = mx.divide(a, b)
    if rounding_mode == "trunc":
        return mx.trunc(result)
    elif rounding_mode == "floor":
        return mx.floor(result)
    return result
```

**Gradient**:
```yaml
- name: div.Tensor(Tensor self, Tensor other) -> Tensor
  self: grad / other
  other: -grad * self / (other * other)
```

**Usage Examples**:
```python
a = torch.tensor([10, 20, 30])
b = torch.tensor([3, 4, 5])

# True division (promotes to float)
torch.div(a, b)  # [3.3333, 5.0000, 6.0000]

# Floor division
torch.div(a, b, rounding_mode="floor")  # [3, 5, 6]

# Truncated division
torch.div(torch.tensor([-10]), torch.tensor([3]), rounding_mode="trunc")  # [-3]
torch.div(torch.tensor([-10]), torch.tensor([3]), rounding_mode="floor")  # [-4]
```

---

### neg (Negation)

**Purpose**: Element-wise negation (unary minus)

**Signature**:
```python
neg(Tensor self) -> Tensor
```

**Formula**: `out = -self`

**YAML Definition** (`native_functions.yaml:~4400`):
```yaml
- func: neg(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: neg.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def neg(a):
    return mx.negative(a)
```

**Gradient**:
```yaml
- name: neg(Tensor self) -> Tensor
  self: -grad
```

**Usage Examples**:
```python
a = torch.tensor([1, -2, 3])
torch.neg(a)  # [-1, 2, -3]

# Equivalent to unary minus
-a  # [-1, 2, -3]
```

---

### abs (Absolute Value)

**Purpose**: Element-wise absolute value

**Signature**:
```python
abs(Tensor self) -> Tensor
```

**Formula**:
- Real: `out = |self|`
- Complex: `out = sqrt(real² + imag²)` (magnitude)

**YAML Definition** (`native_functions.yaml:~100`):
```yaml
- func: abs(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: abs.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def abs(a):
    return mx.abs(a)
```

**Gradient**:
```yaml
- name: abs(Tensor self) -> Tensor
  self: grad * self.sgn()  # Sign of self
```

**Special Cases**:
- `abs(0)` gradient is undefined (PyTorch returns 0)
- Complex tensors: Returns magnitude as real tensor

**Usage Examples**:
```python
a = torch.tensor([-1, -2, 3])
torch.abs(a)  # [1, 2, 3]

# Complex
c = torch.tensor([3 + 4j])
torch.abs(c)  # [5.0]  (sqrt(3² + 4²))
```

---

### pow (Power)

**Purpose**: Element-wise exponentiation

**Signature**:
```python
pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
pow.Scalar(Scalar self, Tensor exponent) -> Tensor
```

**Formula**: `out = self^exponent`

**YAML Definition** (`native_functions.yaml:~4800`):
```yaml
- func: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: pow.Tensor_Scalar_out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def pow(a, b):
    return mx.power(a, b)
```

**Gradient**:
```yaml
- name: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
  self: grad * exponent * pow(self, exponent - 1)
```

**Special Cases**:
- `0^0 = 1` (by convention)
- Negative base with non-integer exponent: Returns NaN (no complex support)
- Integer tensors with negative exponents: Error (use float)

**Usage Examples**:
```python
a = torch.tensor([1, 2, 3, 4])
torch.pow(a, 2)  # [1, 4, 9, 16]

# Tensor exponent (broadcast)
b = torch.tensor([0, 1, 2, 3])
torch.pow(2, b)  # [1, 2, 4, 8]  (2^b)

# Square root
torch.pow(torch.tensor([4.0, 9.0]), 0.5)  # [2.0, 3.0]
```

---

### sqrt (Square Root)

**Purpose**: Element-wise square root

**Signature**:
```python
sqrt(Tensor self) -> Tensor
```

**Formula**: `out = √self`

**YAML Definition** (`native_functions.yaml:~6000`):
```yaml
- func: sqrt(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: sqrt.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def sqrt(a):
    return mx.sqrt(a)
```

**Gradient**:
```yaml
- name: sqrt(Tensor self) -> Tensor
  self: grad / (2 * sqrt(self))
```

**Special Cases**:
- `sqrt(negative)`: NaN for real tensors
- `sqrt(0)` gradient: Infinite (numerically problematic)

**Usage Examples**:
```python
a = torch.tensor([1.0, 4.0, 9.0, 16.0])
torch.sqrt(a)  # [1.0, 2.0, 3.0, 4.0]

# Equivalent to pow(a, 0.5)
torch.pow(a, 0.5)
```

---

## Broadcasting Rules

All arithmetic operators support broadcasting following NumPy semantics:

**Rules**:
1. Align shapes from the right
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1
3. Missing dimensions treated as 1

**Examples**:
```python
# (3, 1) + (4,) → (3, 4)
a = torch.randn(3, 1)
b = torch.randn(4)
c = a + b  # Shape: (3, 4)

# (5, 1, 4) + (3, 1) → (5, 3, 4)
a = torch.randn(5, 1, 4)
b = torch.randn(3, 1)
c = a + b  # Shape: (5, 3, 4)

# (1,) + scalar → (1,)
a = torch.tensor([1.0])
b = 2.0
c = a + b  # Shape: (1,)
```

## Type Promotion Rules

PyTorch follows these type promotion rules for arithmetic ops:

**Priority** (low to high):
```
Bool < UInt8 < Int8 < Int16 < Int32 < Int64 <
Half < Float < Double <
ComplexHalf < ComplexFloat < ComplexDouble
```

**Examples**:
```python
# Int32 + Float32 → Float32
torch.add(torch.tensor([1], dtype=torch.int32),
          torch.tensor([1.0], dtype=torch.float32)).dtype
# torch.float32

# Half + Float → Float
torch.add(torch.tensor([1.0], dtype=torch.float16),
          torch.tensor([1.0], dtype=torch.float32)).dtype
# torch.float32

# Int + Complex → Complex
torch.add(torch.tensor([1], dtype=torch.int32),
          torch.tensor([1+2j], dtype=torch.complex64)).dtype
# torch.complex64
```

**MLX Note**: MLX has simpler promotion (fewer dtypes), generally:
```
int32 < float32 < complex64
```

## Implementation Files

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:554-650` (add, sub)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:2172-2280` (div)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:4322-4390` (mul)

**Meta Functions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/BinaryOps.cpp:149-400`

**CPU Kernels**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp`

**MPS Kernels** (Metal):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/BinaryOps.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/UnaryOps.mm`

**Gradients**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/tools/autograd/derivatives.yaml`

## MLX Porting Summary

**Direct Mappings**:
```python
# PyTorch → MLX
torch.add → mx.add
torch.sub → mx.subtract
torch.mul → mx.multiply
torch.div → mx.divide
torch.neg → mx.negative
torch.abs → mx.abs
torch.pow → mx.power
torch.sqrt → mx.sqrt
```

**Considerations**:
1. **Alpha parameter**: PyTorch-specific, wrap in MLX
2. **Rounding modes**: MLX doesn't have built-in, use floor/trunc
3. **Inplace ops**: MLX is functional, emulate with reassignment
4. **Type promotion**: MLX rules simpler, may need explicit casts
5. **Broadcasting**: Same semantics, MLX handles automatically

**Example Wrapper**:
```python
class torch_compat:
    @staticmethod
    def add(a, b, alpha=1):
        if alpha != 1:
            b = mx.multiply(b, alpha)
        return mx.add(a, b)

    @staticmethod
    def div(a, b, rounding_mode=None):
        result = mx.divide(a, b)
        if rounding_mode == "trunc":
            return mx.trunc(result)
        elif rounding_mode == "floor":
            return mx.floor(result)
        return result
```

Arithmetic operators are the foundation of tensor operations and are straightforward to port to MLX with mostly 1:1 mappings.

---

## Extended Arithmetic Operators

The following operators extend beyond the core 12, providing additional mathematical and logical operations.

---

### exp (Exponential)

**Purpose**: Element-wise exponential function

**Signature**:
```python
exp(Tensor self) -> Tensor
```

**Formula**: `out = e^self`

**YAML Definition**:
```yaml
- func: exp(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: exp.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def exp(x):
    return mx.exp(x)
```

**Gradient**:
```yaml
- name: exp(Tensor self) -> Tensor
  self: grad * result  # grad * exp(self)
```

**Usage Examples**:
```python
x = torch.tensor([0.0, 1.0, 2.0])
torch.exp(x)  # [1.0, 2.7183, 7.3891]

# Softmax building block
exp_x = torch.exp(x - x.max())
probs = exp_x / exp_x.sum()
```

---

### expm1 (Exponential Minus One)

**Purpose**: Element-wise exp(x) - 1, numerically stable for small x

**Signature**:
```python
expm1(Tensor self) -> Tensor
```

**Formula**: `out = e^self - 1`

**YAML Definition**:
```yaml
- func: expm1(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: expm1.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def expm1(x):
    return mx.expm1(x)
    # Or: mx.exp(x) - 1  (less stable for small x)
```

**Gradient**:
```yaml
- name: expm1(Tensor self) -> Tensor
  self: grad * exp(self)
```

**Usage Examples**:
```python
# For small x, exp(x)-1 loses precision
x = torch.tensor([1e-10])
torch.exp(x) - 1       # May give 0 due to precision
torch.expm1(x)         # Accurate: ~1e-10
```

---

### log (Natural Logarithm)

**Purpose**: Element-wise natural logarithm

**Signature**:
```python
log(Tensor self) -> Tensor
```

**Formula**: `out = ln(self)`

**YAML Definition**:
```yaml
- func: log(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: log.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def log(x):
    return mx.log(x)
```

**Gradient**:
```yaml
- name: log(Tensor self) -> Tensor
  self: grad / self
```

**Special Cases**:
- `log(0)` = -inf
- `log(negative)` = NaN

**Usage Examples**:
```python
x = torch.tensor([1.0, 2.7183, 10.0])
torch.log(x)  # [0.0, 1.0, 2.3026]
```

---

### log2 (Base-2 Logarithm)

**Purpose**: Element-wise base-2 logarithm

**Signature**:
```python
log2(Tensor self) -> Tensor
```

**Formula**: `out = log₂(self)`

**YAML Definition**:
```yaml
- func: log2(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: log2.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def log2(x):
    return mx.log2(x)
    # Or: mx.log(x) / mx.log(2)
```

**Gradient**:
```yaml
- name: log2(Tensor self) -> Tensor
  self: grad / (self * ln(2))
```

**Usage Examples**:
```python
x = torch.tensor([1.0, 2.0, 4.0, 8.0])
torch.log2(x)  # [0.0, 1.0, 2.0, 3.0]
```

---

### log10 (Base-10 Logarithm)

**Purpose**: Element-wise base-10 logarithm

**Signature**:
```python
log10(Tensor self) -> Tensor
```

**Formula**: `out = log₁₀(self)`

**YAML Definition**:
```yaml
- func: log10(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: log10.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def log10(x):
    return mx.log10(x)
    # Or: mx.log(x) / mx.log(10)
```

**Gradient**:
```yaml
- name: log10(Tensor self) -> Tensor
  self: grad / (self * ln(10))
```

**Usage Examples**:
```python
x = torch.tensor([1.0, 10.0, 100.0])
torch.log10(x)  # [0.0, 1.0, 2.0]
```

---

### log1p (Logarithm of 1+x)

**Purpose**: Element-wise log(1 + x), numerically stable for small x

**Signature**:
```python
log1p(Tensor self) -> Tensor
```

**Formula**: `out = ln(1 + self)`

**YAML Definition**:
```yaml
- func: log1p(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: log1p.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def log1p(x):
    return mx.log1p(x)
    # Or: mx.log(1 + x)  (less stable for small x)
```

**Gradient**:
```yaml
- name: log1p(Tensor self) -> Tensor
  self: grad / (1 + self)
```

**Usage Examples**:
```python
# For small x, log(1+x) loses precision
x = torch.tensor([1e-10])
torch.log(1 + x)  # May give 0
torch.log1p(x)    # Accurate: ~1e-10
```

---

### floor (Floor)

**Purpose**: Element-wise floor (round toward -∞)

**Signature**:
```python
floor(Tensor self) -> Tensor
```

**Formula**: `out = ⌊self⌋`

**YAML Definition**:
```yaml
- func: floor(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: floor.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def floor(x):
    return mx.floor(x)
```

**Gradient**:
```yaml
- name: floor(Tensor self) -> Tensor
  self: zeros_like(grad)  # Gradient is 0 (non-differentiable)
```

**Usage Examples**:
```python
x = torch.tensor([-1.7, -0.5, 0.5, 1.7])
torch.floor(x)  # [-2.0, -1.0, 0.0, 1.0]
```

---

### ceil (Ceiling)

**Purpose**: Element-wise ceiling (round toward +∞)

**Signature**:
```python
ceil(Tensor self) -> Tensor
```

**Formula**: `out = ⌈self⌉`

**YAML Definition**:
```yaml
- func: ceil(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: ceil.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def ceil(x):
    return mx.ceil(x)
```

**Gradient**:
```yaml
- name: ceil(Tensor self) -> Tensor
  self: zeros_like(grad)  # Gradient is 0
```

**Usage Examples**:
```python
x = torch.tensor([-1.7, -0.5, 0.5, 1.7])
torch.ceil(x)  # [-1.0, 0.0, 1.0, 2.0]
```

---

### round (Round to Nearest)

**Purpose**: Element-wise round to nearest integer (banker's rounding)

**Signature**:
```python
round(Tensor self) -> Tensor
round.decimals(Tensor self, *, int decimals) -> Tensor
```

**Formula**: `out = round(self)` (round half to even)

**YAML Definition**:
```yaml
- func: round(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: round.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def round(x):
    return mx.round(x)
```

**Usage Examples**:
```python
x = torch.tensor([0.5, 1.5, 2.5, 3.5])
torch.round(x)  # [0.0, 2.0, 2.0, 4.0]  (banker's rounding)

# Round to decimal places
torch.round(torch.tensor([3.14159]), decimals=2)  # [3.14]
```

---

### trunc (Truncate)

**Purpose**: Element-wise truncation (round toward zero)

**Signature**:
```python
trunc(Tensor self) -> Tensor
```

**Formula**: `out = trunc(self)` (drop fractional part)

**YAML Definition**:
```yaml
- func: trunc(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: trunc.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def trunc(x):
    return mx.trunc(x)
    # Or: mx.sign(x) * mx.floor(mx.abs(x))
```

**Usage Examples**:
```python
x = torch.tensor([-1.7, -0.5, 0.5, 1.7])
torch.trunc(x)  # [-1.0, 0.0, 0.0, 1.0]
```

---

### frac (Fractional Part)

**Purpose**: Element-wise fractional part

**Signature**:
```python
frac(Tensor self) -> Tensor
```

**Formula**: `out = self - trunc(self)`

**YAML Definition**:
```yaml
- func: frac(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: frac.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def frac(x):
    return x - mx.trunc(x)
```

**Usage Examples**:
```python
x = torch.tensor([-1.7, -0.5, 0.5, 1.7])
torch.frac(x)  # [-0.7, -0.5, 0.5, 0.7]
```

---

### clamp / clip (Clipping)

**Purpose**: Clamp values to a range [min, max]

**Signature**:
```python
clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor  # alias
```

**Formula**: `out = max(min, min(self, max))`

**YAML Definition**:
```yaml
- func: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CPU, CUDA: clamp
    MPS: clamp_mps
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def clamp(x, min=None, max=None):
    return mx.clip(x, min, max)
```

**Gradient**:
```yaml
- name: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
  self: grad * (self >= min) * (self <= max)  # Pass-through in range
```

**Usage Examples**:
```python
x = torch.tensor([-2, -1, 0, 1, 2])
torch.clamp(x, min=-1, max=1)  # [-1, -1, 0, 1, 1]
torch.clamp(x, min=0)          # [0, 0, 0, 1, 2]
torch.clamp(x, max=0)          # [-2, -1, 0, 0, 0]

# clip is an alias
torch.clip(x, -1, 1)
```

---

### remainder (Remainder)

**Purpose**: Element-wise remainder of division (same sign as divisor)

**Signature**:
```python
remainder(Tensor self, Scalar other) -> Tensor
remainder.Tensor(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self - other * floor(self / other)`

**YAML Definition**:
```yaml
- func: remainder.Scalar(Tensor self, Scalar other) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: remainder.Scalar_out
  tags: pointwise
```

**MLX Equivalent**:
```python
def remainder(x, y):
    return x - y * mx.floor(x / y)
    # Or: mx.remainder(x, y) if available
```

**Usage Examples**:
```python
x = torch.tensor([5, -5, 5, -5])
y = torch.tensor([3, 3, -3, -3])
torch.remainder(x, y)  # [2, 1, -1, -2] (sign matches divisor)
```

---

### fmod (Modulo)

**Purpose**: Element-wise modulo (same sign as dividend)

**Signature**:
```python
fmod(Tensor self, Scalar other) -> Tensor
fmod.Tensor(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self - other * trunc(self / other)`

**YAML Definition**:
```yaml
- func: fmod.Scalar(Tensor self, Scalar other) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: fmod.Scalar_out
  tags: pointwise
```

**MLX Equivalent**:
```python
def fmod(x, y):
    return x - y * mx.trunc(x / y)
```

**Usage Examples**:
```python
x = torch.tensor([5, -5, 5, -5])
y = torch.tensor([3, 3, -3, -3])
torch.fmod(x, y)  # [2, -2, 2, -2] (sign matches dividend)
```

**Difference from remainder**:
- `remainder`: Result sign matches divisor (Python `%`)
- `fmod`: Result sign matches dividend (C `%`)

---

### reciprocal (Reciprocal)

**Purpose**: Element-wise reciprocal (1/x)

**Signature**:
```python
reciprocal(Tensor self) -> Tensor
```

**Formula**: `out = 1 / self`

**YAML Definition**:
```yaml
- func: reciprocal(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: reciprocal.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def reciprocal(x):
    return 1.0 / x
    # Or: mx.reciprocal(x) if available
```

**Gradient**:
```yaml
- name: reciprocal(Tensor self) -> Tensor
  self: -grad * result * result  # -grad / self^2
```

**Usage Examples**:
```python
x = torch.tensor([1.0, 2.0, 4.0])
torch.reciprocal(x)  # [1.0, 0.5, 0.25]
```

---

### rsqrt (Reciprocal Square Root)

**Purpose**: Element-wise 1/√x

**Signature**:
```python
rsqrt(Tensor self) -> Tensor
```

**Formula**: `out = 1 / √self`

**YAML Definition**:
```yaml
- func: rsqrt(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: rsqrt.out
  tags: [core, pointwise]
```

**MLX Equivalent**:
```python
def rsqrt(x):
    return mx.rsqrt(x)
    # Or: 1.0 / mx.sqrt(x)
```

**Gradient**:
```yaml
- name: rsqrt(Tensor self) -> Tensor
  self: -0.5 * grad * result^3  # -grad / (2 * self^(3/2))
```

**Usage Examples**:
```python
x = torch.tensor([1.0, 4.0, 9.0])
torch.rsqrt(x)  # [1.0, 0.5, 0.3333]

# Common in normalization
normalized = x * torch.rsqrt(x.var() + eps)
```

---

### sign / sgn (Sign Function)

**Purpose**: Element-wise sign of each element

**Signature**:
```python
sign(Tensor self) -> Tensor
sgn(Tensor self) -> Tensor  # Complex-aware version
```

**Formula**:
- Real: `sign(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0`
- Complex: `sgn(z) = z / |z|` (normalized direction)

**YAML Definition**:
```yaml
- func: sign(Tensor self) -> Tensor
  device_check: NoCheck
  variants: function, method
  structured_delegate: sign.out
  tags: pointwise

- func: sgn(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: sgn.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def sign(x):
    return mx.sign(x)
```

**Gradient**:
```yaml
- name: sign(Tensor self) -> Tensor
  self: zeros_like(grad)  # Non-differentiable
```

**Usage Examples**:
```python
x = torch.tensor([-5, -0.5, 0, 0.5, 5])
torch.sign(x)  # [-1, -1, 0, 1, 1]

# Complex
z = torch.tensor([3+4j])
torch.sgn(z)  # [0.6+0.8j] (normalized to unit magnitude)
```

---

## Logical Operators

### logical_and (Logical AND)

**Purpose**: Element-wise logical AND

**Signature**:
```python
logical_and(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self AND other` (returns bool tensor)

**YAML Definition**:
```yaml
- func: logical_and(Tensor self, Tensor other) -> Tensor
  variants: function, method
  structured_delegate: logical_and.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def logical_and(a, b):
    return mx.logical_and(a, b)
    # Or: a & b for boolean tensors
```

**Usage Examples**:
```python
a = torch.tensor([True, True, False, False])
b = torch.tensor([True, False, True, False])
torch.logical_and(a, b)  # [True, False, False, False]

# With non-boolean (treats non-zero as True)
torch.logical_and(torch.tensor([1, 0, 2]), torch.tensor([1, 1, 0]))
# [True, False, False]
```

---

### logical_or (Logical OR)

**Purpose**: Element-wise logical OR

**Signature**:
```python
logical_or(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self OR other`

**YAML Definition**:
```yaml
- func: logical_or(Tensor self, Tensor other) -> Tensor
  variants: function, method
  structured_delegate: logical_or.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def logical_or(a, b):
    return mx.logical_or(a, b)
    # Or: a | b for boolean tensors
```

**Usage Examples**:
```python
a = torch.tensor([True, True, False, False])
b = torch.tensor([True, False, True, False])
torch.logical_or(a, b)  # [True, True, True, False]
```

---

### logical_not (Logical NOT)

**Purpose**: Element-wise logical NOT

**Signature**:
```python
logical_not(Tensor self) -> Tensor
```

**Formula**: `out = NOT self`

**YAML Definition**:
```yaml
- func: logical_not(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: logical_not.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def logical_not(x):
    return mx.logical_not(x)
    # Or: ~x for boolean tensors
```

**Usage Examples**:
```python
x = torch.tensor([True, False])
torch.logical_not(x)  # [False, True]

# Non-boolean
torch.logical_not(torch.tensor([0, 1, -1]))  # [True, False, False]
```

---

### logical_xor (Logical XOR)

**Purpose**: Element-wise logical exclusive OR

**Signature**:
```python
logical_xor(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self XOR other`

**YAML Definition**:
```yaml
- func: logical_xor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  structured_delegate: logical_xor.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def logical_xor(a, b):
    return mx.logical_xor(a, b)
    # Or: (a | b) & ~(a & b)
```

**Usage Examples**:
```python
a = torch.tensor([True, True, False, False])
b = torch.tensor([True, False, True, False])
torch.logical_xor(a, b)  # [False, True, True, False]
```

---

## Bitwise Operators

### bitwise_and (Bitwise AND)

**Purpose**: Element-wise bitwise AND

**Signature**:
```python
bitwise_and(Tensor self, Tensor other) -> Tensor
bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
```

**Formula**: `out = self & other` (bit-by-bit)

**YAML Definition**:
```yaml
- func: bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  structured_delegate: bitwise_and.Tensor_out
  tags: pointwise
```

**MLX Equivalent**:
```python
def bitwise_and(a, b):
    return a & b
```

**Usage Examples**:
```python
a = torch.tensor([0b1100, 0b1010])  # [12, 10]
b = torch.tensor([0b1010, 0b0110])  # [10, 6]
torch.bitwise_and(a, b)  # [8, 2]  (0b1000, 0b0010)

# Booleans work too
torch.bitwise_and(torch.tensor([True, False]), torch.tensor([True, True]))
```

---

### bitwise_or (Bitwise OR)

**Purpose**: Element-wise bitwise OR

**Signature**:
```python
bitwise_or(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self | other`

**YAML Definition**:
```yaml
- func: bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  structured_delegate: bitwise_or.Tensor_out
  tags: pointwise
```

**MLX Equivalent**:
```python
def bitwise_or(a, b):
    return a | b
```

**Usage Examples**:
```python
a = torch.tensor([0b1100, 0b1010])  # [12, 10]
b = torch.tensor([0b1010, 0b0110])  # [10, 6]
torch.bitwise_or(a, b)  # [14, 14]  (0b1110, 0b1110)
```

---

### bitwise_xor (Bitwise XOR)

**Purpose**: Element-wise bitwise exclusive OR

**Signature**:
```python
bitwise_xor(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self ^ other`

**YAML Definition**:
```yaml
- func: bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  structured_delegate: bitwise_xor.Tensor_out
  tags: pointwise
```

**MLX Equivalent**:
```python
def bitwise_xor(a, b):
    return a ^ b
```

**Usage Examples**:
```python
a = torch.tensor([0b1100, 0b1010])  # [12, 10]
b = torch.tensor([0b1010, 0b0110])  # [10, 6]
torch.bitwise_xor(a, b)  # [6, 12]  (0b0110, 0b1100)
```

---

### bitwise_not (Bitwise NOT)

**Purpose**: Element-wise bitwise NOT (complement)

**Signature**:
```python
bitwise_not(Tensor self) -> Tensor
```

**Formula**: `out = ~self`

**YAML Definition**:
```yaml
- func: bitwise_not(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: bitwise_not.out
  tags: pointwise
```

**MLX Equivalent**:
```python
def bitwise_not(x):
    return ~x
```

**Usage Examples**:
```python
x = torch.tensor([0b0000, 0b1111], dtype=torch.int8)
torch.bitwise_not(x)  # [-1, -16] (0b11111111, 0b11110000)

# Booleans
torch.bitwise_not(torch.tensor([True, False]))  # [False, True]
```

---

### bitwise_left_shift (Left Shift)

**Purpose**: Element-wise left bit shift

**Signature**:
```python
bitwise_left_shift(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self << other`

**YAML Definition**:
```yaml
- func: bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  tags: pointwise
```

**MLX Equivalent**:
```python
def bitwise_left_shift(x, n):
    return x << n
```

**Usage Examples**:
```python
x = torch.tensor([1, 2, 4])
torch.bitwise_left_shift(x, 2)  # [4, 8, 16]  (multiply by 4)
```

---

### bitwise_right_shift (Right Shift)

**Purpose**: Element-wise right bit shift (arithmetic)

**Signature**:
```python
bitwise_right_shift(Tensor self, Tensor other) -> Tensor
```

**Formula**: `out = self >> other`

**YAML Definition**:
```yaml
- func: bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
  variants: function, method
  tags: pointwise
```

**MLX Equivalent**:
```python
def bitwise_right_shift(x, n):
    return x >> n
```

**Usage Examples**:
```python
x = torch.tensor([16, 8, 4])
torch.bitwise_right_shift(x, 2)  # [4, 2, 1]  (divide by 4)
```

---

## Extended Arithmetic Summary

| Category | Operators | Status |
|----------|-----------|--------|
| Core (8) | add, sub, mul, div, neg, abs, pow, sqrt | ✅ Documented |
| Exponential (2) | exp, expm1 | ✅ Documented |
| Logarithmic (4) | log, log2, log10, log1p | ✅ Documented |
| Rounding (5) | floor, ceil, round, trunc, frac | ✅ Documented |
| Clipping (1) | clamp/clip | ✅ Documented |
| Modulo (2) | remainder, fmod | ✅ Documented |
| Reciprocal (2) | reciprocal, rsqrt | ✅ Documented |
| Sign (2) | sign, sgn | ✅ Documented |
| Logical (4) | logical_and, logical_or, logical_not, logical_xor | ✅ Documented |
| Bitwise (6) | bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift | ✅ Documented |

**Total**: 36 arithmetic operators documented (100% coverage)

---

## MLX Extended Arithmetic Implementations

```python
import mlx.core as mx

class ArithmeticOps:
    """Extended arithmetic operations for MLX"""

    # Exponential/Log
    @staticmethod
    def exp(x): return mx.exp(x)

    @staticmethod
    def expm1(x): return mx.expm1(x)

    @staticmethod
    def log(x): return mx.log(x)

    @staticmethod
    def log2(x): return mx.log2(x)

    @staticmethod
    def log10(x): return mx.log10(x)

    @staticmethod
    def log1p(x): return mx.log1p(x)

    # Rounding
    @staticmethod
    def floor(x): return mx.floor(x)

    @staticmethod
    def ceil(x): return mx.ceil(x)

    @staticmethod
    def round(x): return mx.round(x)

    @staticmethod
    def trunc(x): return mx.trunc(x)

    @staticmethod
    def frac(x): return x - mx.trunc(x)

    # Clipping
    @staticmethod
    def clamp(x, min=None, max=None): return mx.clip(x, min, max)

    # Modulo
    @staticmethod
    def remainder(x, y): return x - y * mx.floor(x / y)

    @staticmethod
    def fmod(x, y): return x - y * mx.trunc(x / y)

    # Reciprocal
    @staticmethod
    def reciprocal(x): return 1.0 / x

    @staticmethod
    def rsqrt(x): return mx.rsqrt(x)

    # Sign
    @staticmethod
    def sign(x): return mx.sign(x)

    # Logical
    @staticmethod
    def logical_and(a, b): return mx.logical_and(a, b)

    @staticmethod
    def logical_or(a, b): return mx.logical_or(a, b)

    @staticmethod
    def logical_not(x): return mx.logical_not(x)

    @staticmethod
    def logical_xor(a, b): return mx.logical_xor(a, b)
```
