# Broadcasting & Type Promotion

## Purpose

Broadcasting and type promotion are fundamental mechanisms in PyTorch (and NumPy) that allow operations on tensors of different shapes and data types. Understanding these rules is **critical** for correctly implementing operators in MLX and ensuring numerical compatibility with PyTorch.

**Key Points**:
- **Broadcasting**: Automatically expands tensors to compatible shapes for element-wise operations
- **Type Promotion**: Automatically determines the output dtype when mixing different input dtypes
- **NumPy Compatibility**: PyTorch follows NumPy's semantics for both mechanisms

This document provides the complete rules, implementation details, and edge cases.

---

## Broadcasting Semantics

### Core Concept

Broadcasting allows operations between tensors of different shapes without explicit copying:

```python
# Example:
a = torch.tensor([[1, 2, 3]])      # Shape: (1, 3)
b = torch.tensor([[10], [20]])      # Shape: (2, 1)
c = a + b                           # Shape: (2, 3)

# Result:
# [[11, 12, 13],
#  [21, 22, 23]]
```

**How it works**: Dimensions are aligned from the **right** (trailing dimensions), and dimensions of size 1 are "stretched" to match the corresponding dimension.

### Broadcasting Rules

**Rule 1: Right Alignment**
- Compare shapes from right to left (trailing dimensions first)
- Missing dimensions are treated as size 1

**Rule 2: Dimension Compatibility**
Two dimensions are compatible if:
1. They are equal, OR
2. One of them is 1

**Rule 3: Output Shape**
- The output shape is the element-wise maximum of input shapes
- Dimensions of size 1 are expanded to match the larger dimension

### Examples

#### Example 1: Simple Broadcasting

```python
A: (3,)      # Implicitly (1, 1, 3)
B: (2, 1, 3)
---
Result: (2, 1, 3)

# Dimension-by-dimension:
# Dim 2: 3 == 3 ✓
# Dim 1: 1 broadcasts to 1 ✓
# Dim 0: 1 broadcasts to 2 ✓
```

#### Example 2: Multi-Dimension Broadcasting

```python
A: (5, 1, 4, 1)
B:    (3, 1, 6)
---
Result: (5, 3, 4, 6)

# Right-aligned:
# A: (5, 1, 4, 1)
# B: (1, 3, 1, 6)  ← implicitly padded with 1 on left
# Out: (5, 3, 4, 6)
```

#### Example 3: Incompatible Shapes (Error)

```python
A: (3, 5)
B: (3, 4)
---
ERROR: Dimension 1: 5 != 4 and neither is 1
```

#### Example 4: Scalar Broadcasting

```python
A: ()        # Scalar (0-dimensional tensor)
B: (2, 3, 4)
---
Result: (2, 3, 4)

# Scalar broadcasts to any shape
```

### Implementation in PyTorch

**Location**: `aten/src/ATen/ExpandUtils.h`, `aten/src/ATen/TensorIterator.cpp`

#### Shape Inference Algorithm

```cpp
DimVector infer_size(IntArrayRef shape_a, IntArrayRef shape_b) {
  size_t ndim_a = shape_a.size();
  size_t ndim_b = shape_b.size();
  size_t ndim = std::max(ndim_a, ndim_b);

  DimVector result(ndim);

  // Iterate from right (trailing dims)
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t dim_a = (i < ndim - ndim_a) ? 1 : shape_a[i - (ndim - ndim_a)];
    int64_t dim_b = (i < ndim - ndim_b) ? 1 : shape_b[i - (ndim - ndim_b)];

    if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
      result[i] = std::max(dim_a, dim_b);
    } else {
      TORCH_CHECK(false,
        "The size of tensor a (", dim_a,
        ") must match the size of tensor b (", dim_b,
        ") at non-singleton dimension ", i);
    }
  }

  return result;
}
```

#### Memory-Efficient Broadcasting

PyTorch **doesn't copy data** for broadcasting. Instead, it uses **stride manipulation**:

```cpp
// Original tensor: shape=[3], strides=[1]
// After broadcasting to shape=[2, 3]:
//   - Dimension 0 (size 2): stride = 0  ← Reuses same data!
//   - Dimension 1 (size 3): stride = 1

// TensorIterator sets stride=0 for broadcast dimensions
operands_[i].stride_bytes[dim] = 0;  // Broadcast dimension
```

**How Stride-0 Works**:
- When iterating, offset never advances along broadcast dimensions
- Same element is read/written repeatedly
- Zero memory overhead!

### Broadcasting in Backward Pass

Gradients must be **reduced** when broadcasting occurs in the forward pass:

```python
# Forward:
a = torch.tensor([[1, 2, 3]], requires_grad=True)  # (1, 3)
b = torch.tensor([[10], [20]], requires_grad=True) # (2, 1)
c = a + b  # (2, 3)

# Backward:
c.sum().backward()

# a.grad: (1, 3) ← Sum over dimension 0 (reduced from (2, 3))
# b.grad: (2, 1) ← Sum over dimension 1 (reduced from (2, 3))
```

**Implementation**:
- Autograd tracks which dimensions were broadcast
- Gradients are summed along those dimensions using `.sum(dim, keepdim=True)`
- See `torch/csrc/autograd/functions/accumulate_grad.cpp`

---

## Type Promotion

### Core Concept

When operating on tensors with different dtypes, PyTorch automatically promotes them to a common dtype:

```python
a = torch.tensor([1, 2], dtype=torch.int32)
b = torch.tensor([3.5, 4.5], dtype=torch.float32)
c = a + b  # Result is float32

# int32 + float32 → float32
```

### Type Categories

PyTorch organizes types into a hierarchy:

```
bool < integer < floating < complex
```

**Promotion Direction**: Always promote "upward" in the hierarchy.

#### Category Definitions

**Boolean**:
- `bool`

**Integer** (signed):
- `int8`, `int16`, `int32`, `int64`

**Integer** (unsigned):
- `uint8`, `uint16`, `uint32`, `uint64`

**Floating Point**:
- `float16` (half)
- `bfloat16`
- `float32` (float)
- `float64` (double)

**Complex**:
- `complex32` (complex half)
- `complex64` (complex float)
- `complex128` (complex double)

**Special** (no promotion):
- Quantized types: `qint8`, `quint8`, `qint32`
- Float8 types: `float8_e4m3fn`, `float8_e5m2`, etc.
- Bits types: `bits1x8`, `bits2x4`, etc.

### Promotion Rules

#### Rule 1: Same Category → Wider Type

Within the same category, promote to the type with more bits:

```python
int8 + int32 → int32
float16 + float32 → float32
complex64 + complex128 → complex128
```

#### Rule 2: Cross-Category → Higher Category

Across categories, promote to the higher category:

```python
bool + int32 → int32
int32 + float32 → float32
float32 + complex64 → complex64
```

#### Rule 3: Complex Preserves Value Type

When promoting to complex, preserve the value type of the floating operand:

```python
float32 + complex64 → complex64  # float32 has same value type
float64 + complex64 → complex128  # float64 needs complex128
```

#### Rule 4: Integer + Complex → Complex

Integer operands are promoted to the complex type:

```python
int32 + complex64 → complex64
```

### Promotion Matrix

**Location**: `c10/core/ScalarType.cpp:109-128`

The complete type promotion lookup table:

```cpp
//           u1   i1   i2   i4   i8   f2   f4   f8   c2   c4   c8   b1   bf
/* u1 */  { u1,  i2,  i2,  i4,  i8,  f2,  f4,  f8,  c2,  c4,  c8,  u1,  bf  },
/* i1 */  { i2,  i1,  i2,  i4,  i8,  f2,  f4,  f8,  c2,  c4,  c8,  i1,  bf  },
/* i2 */  { i2,  i2,  i2,  i4,  i8,  f2,  f4,  f8,  c2,  c4,  c8,  i2,  bf  },
/* i4 */  { i4,  i4,  i4,  i4,  i8,  f2,  f4,  f8,  c2,  c4,  c8,  i4,  bf  },
/* i8 */  { i8,  i8,  i8,  i8,  i8,  f2,  f4,  f8,  c2,  c4,  c8,  i8,  bf  },
/* f2 */  { f2,  f2,  f2,  f2,  f2,  f2,  f4,  f8,  c2,  c4,  c8,  f2,  f4  },
/* f4 */  { f4,  f4,  f4,  f4,  f4,  f4,  f4,  f8,  c4,  c4,  c8,  f4,  f4  },
/* f8 */  { f8,  f8,  f8,  f8,  f8,  f8,  f8,  f8,  c8,  c8,  c8,  f8,  f8  },
/* c2 */  { c2,  c2,  c2,  c2,  c2,  c2,  c4,  c8,  c2,  c4,  c8,  c2,  c4  },
/* c4 */  { c4,  c4,  c4,  c4,  c4,  c4,  c4,  c8,  c4,  c4,  c8,  c4,  c4  },
/* c8 */  { c8,  c8,  c8,  c8,  c8,  c8,  c8,  c8,  c8,  c8,  c8,  c8,  c8  },
/* b1 */  { u1,  i1,  i2,  i4,  i8,  f2,  f4,  f8,  c2,  c4,  c8,  b1,  bf  },
/* bf */  { bf,  bf,  bf,  bf,  bf,  f4,  f4,  f8,  c4,  c4,  c8,  bf,  bf  },
```

**Key**:
- `u1` = uint8, `i1` = int8, `i2` = int16, `i4` = int32, `i8` = int64
- `f2` = float16, `f4` = float32, `f8` = float64
- `c2` = complex32, `c4` = complex64, `c8` = complex128
- `b1` = bool, `bf` = bfloat16

### Special Cases

#### BFloat16 Promotion

BFloat16 promotes to float32 when mixed with float16:

```python
bfloat16 + float16 → float32
```

This prevents precision loss since bfloat16 and float16 have different range/precision tradeoffs.

#### Unsigned Integer Handling

**Problem**: PyTorch historically promoted `uint8` to `int64` (not `uint64`) for NumPy compatibility. This can overflow for large values.

**Current Behavior**:
- `uint8` + integer → Follows promotion table (may promote to int)
- `uint16`, `uint32`, `uint64` + integer → **ERROR** (not supported except with floating types)

```python
uint8 + int32 → int32  # OK (legacy behavior)
uint16 + int32 → ERROR  # Not supported
uint16 + float32 → float32  # OK
```

#### Quantized and Float8 Types

These types **do not support promotion**:

```python
qint8 + qint8 → ERROR  # Promotion not defined
float8_e4m3fn + float32 → ERROR  # Promotion not supported
```

**Rationale**: These are specialized types for inference/training optimization; mixing them with standard types is rarely meaningful.

### `result_type` Function

PyTorch provides `torch.result_type()` to query the promoted type without computation:

```python
result = torch.result_type(tensor1, tensor2)
```

**Implementation**: `aten/src/ATen/native/TypeProperties.cpp:141-176`

#### Result Type for Scalars vs Tensors

PyTorch distinguishes between:
1. **Dimensional tensors** (ndim > 0): Full participation in promotion
2. **0-D tensors** (ndim == 0): Same as dimensional tensors
3. **Wrapped numbers** (Python scalars wrapped as tensors): Use default dtype
4. **Python scalars**: Use default dtype for that category

```cpp
struct ResultTypeState {
  ScalarType dimResult = ScalarType::Undefined;      // From dimensional tensors
  ScalarType zeroResult = ScalarType::Undefined;     // From 0-D tensors
  ScalarType wrappedResult = ScalarType::Undefined;  // From scalars
};

ScalarType result_type(const ResultTypeState& state) {
  return combine_categories(
    state.dimResult,
    combine_categories(state.zeroResult, state.wrappedResult)
  );
}
```

**Priority**: `dimResult` > `zeroResult` > `wrappedResult`

#### Example

```python
a = torch.tensor([1, 2], dtype=torch.int32)        # Dimensional
b = torch.tensor(3.0, dtype=torch.float32)         # 0-D tensor
c = 4.0                                            # Python scalar

result = torch.result_type(a, b, c)
# dimResult: int32
# zeroResult: float32
# wrappedResult: float64 (default for Python float)
# Final: combine_categories(int32, combine_categories(float32, float64))
#      = combine_categories(int32, float64)
#      = float64
```

### Implementation in TensorIterator

TensorIterator handles type promotion automatically:

```cpp
auto iter = TensorIteratorConfig()
  .add_output(Tensor())              // Output will use common dtype
  .add_input(int_tensor)             // int32
  .add_input(float_tensor)           // float32
  .promote_inputs_to_common_dtype()  // Enable promotion
  .build();

ScalarType common = iter.common_dtype();  // Returns float32
```

**What Happens**:
1. TensorIterator computes common dtype using `promoteTypes()`
2. Inputs are internally converted to common dtype (if needed)
3. Output is allocated with common dtype
4. Operation proceeds on promoted types

---

## Expansion vs Broadcasting

### Expansion (Explicit)

`Tensor.expand()` explicitly creates a view with broadcast semantics:

```python
a = torch.tensor([[1, 2, 3]])  # (1, 3)
b = a.expand(4, 3)              # (4, 3) - view with stride 0

# b shares storage with a
# Modifying a affects b
```

**Memory**: Creates a **view** with stride=0 for broadcast dimensions (no data copy).

### Broadcasting (Implicit)

Operations automatically broadcast without explicit expansion:

```python
a = torch.tensor([[1, 2, 3]])  # (1, 3)
b = torch.tensor([[10], [20], [30], [40]])  # (4, 1)
c = a + b  # (4, 3) - broadcasting happens internally
```

**Memory**: TensorIterator handles broadcasting via stride manipulation (no intermediate tensor created).

### When to Use Each

**Use Expand**:
- Need a view for indexing or further operations
- Want to explicitly control shape transformation
- Debugging (makes broadcasting visible)

**Use Broadcasting**:
- Element-wise operations (add, mul, etc.)
- More concise code
- No performance difference (both use stride tricks)

---

## Edge Cases & Pitfalls

### Pitfall 1: Silent Precision Loss

```python
# Be careful mixing int and float!
a = torch.tensor([1000000000], dtype=torch.int64)  # Large integer
b = torch.tensor([1.5], dtype=torch.float32)

c = a + b  # Result is float32
# int64 → float32 loses precision for large integers!
# float32 can only represent ~7 significant digits
```

**Solution**: Be explicit about dtypes when precision matters.

### Pitfall 2: Complex Promotion Surprises

```python
a = torch.tensor([1.0], dtype=torch.float64)
b = torch.tensor([1.0 + 2.0j], dtype=torch.complex64)

c = a + b  # complex128, not complex64!
# float64 promotes to complex128 to preserve precision
```

### Pitfall 3: Broadcasting with Reduction

```python
a = torch.tensor([[1, 2, 3]])  # (1, 3)
b = torch.tensor([[10], [20]]) # (2, 1)

# Forward
c = a + b  # (2, 3)

# If a and b require gradients:
a.grad  # Will be (1, 3) - summed from (2, 3)
b.grad  # Will be (2, 1) - summed from (2, 3)
```

**Key**: Gradients are automatically reduced to match original shapes.

### Pitfall 4: Bool Promotion

```python
a = torch.tensor([True, False], dtype=torch.bool)
b = torch.tensor([1, 2], dtype=torch.int32)

c = a + b  # int32
# bool promoted to int32, then: [1, 0] + [1, 2] = [2, 2]
```

**Surprise**: `True` becomes `1`, `False` becomes `0`.

### Pitfall 5: In-Place Operations Don't Promote

```python
a = torch.tensor([1, 2], dtype=torch.int32)
b = torch.tensor([3.5, 4.5], dtype=torch.float32)

a += b  # ERROR! Can't cast float32 to int32 in-place
```

**In-place operations** require compatible dtypes (no widening allowed).

### Pitfall 6: NumPy Compatibility Edge Cases

PyTorch follows NumPy promotion rules **mostly**, but with differences:

```python
# NumPy: uint8 + int → uint
# PyTorch: uint8 + int → int (for historical reasons)

# NumPy: Promotes uint16/32/64 with int
# PyTorch: Raises error for uint16/32/64 + int
```

---

## MLX Porting Considerations

### Broadcasting

**MLX Implementation**: MLX supports NumPy-compatible broadcasting.

```cpp
// MLX broadcasting example
array a = array(std::vector<float>{1, 2, 3}, {1, 3});  // (1, 3)
array b = array(std::vector<float>{10, 20}, {2, 1});   // (2, 1)
array c = a + b;  // (2, 3) - automatic broadcasting
```

**Recommendations**:
1. ✅ **Adopt PyTorch's broadcasting rules** (they match NumPy)
2. ✅ **Use stride manipulation** for efficiency (zero-copy broadcasting)
3. ✅ **Implement `infer_size()` algorithm** for shape inference
4. ✅ **Handle backward pass correctly** (sum gradients along broadcast dimensions)

### Type Promotion

**MLX Current State**: MLX has simpler type system than PyTorch.

**MLX Types**:
- `bool_`, `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float16`, `bfloat16`, `float32`
- `complex64`

**Recommendations**:
1. ✅ **Implement promotion matrix** for MLX types (use PyTorch's table as reference)
2. ✅ **Follow category hierarchy**: `bool < int < float < complex`
3. ⚠️ **Decide on unsigned promotion**: PyTorch's behavior is inconsistent (error vs promote to int)
   - **Suggestion**: Promote `uint + int → uint` for larger type (more intuitive than PyTorch)
4. ⚠️ **BFloat16 handling**: Decide `bfloat16 + float16 → ?`
   - **Suggestion**: Promote to `float32` (same as PyTorch) to avoid precision issues
5. ❌ **Skip quantized/float8**: MLX doesn't need these specialized types (yet)

### Example MLX Promotion Table

```cpp
// Simplified MLX promotion table
//        b1   i8   i16  i32  i64  u8   u16  u32  u64  f16  bf16 f32  c64
/* b1 */ { b1,  i8,  i16, i32, i64, u8,  u16, u32, u64, f16, bf16, f32, c64 },
/* i8 */ { i8,  i8,  i16, i32, i64, i16, i16, i32, i64, f16, bf16, f32, c64 },
// ... (fill in following PyTorch's rules)
```

### Key Differences from PyTorch

| Aspect | PyTorch | MLX Recommendation |
|--------|---------|---------------------|
| **uint + int** | Error (except uint8) | Promote to unsigned of larger size |
| **float16 + bfloat16** | float32 | float32 (same) |
| **Quantized types** | No promotion (error) | N/A (no quantized types) |
| **Complex types** | complex32, complex64, complex128 | Only complex64 currently |

---

## Testing Promotion & Broadcasting

### Test Cases for Broadcasting

```python
# Test 1: Simple broadcast
assert (torch.tensor([1]) + torch.tensor([[2], [3]])).shape == (2, 1)

# Test 2: Multi-dimension
a = torch.randn(5, 1, 4, 1)
b = torch.randn(3, 1, 6)
assert (a + b).shape == (5, 3, 4, 6)

# Test 3: Scalar
assert (torch.tensor(5) + torch.randn(3, 4)).shape == (3, 4)

# Test 4: Incompatible (should error)
try:
    torch.tensor([[1, 2]]) + torch.tensor([[1], [2], [3]])  # (1,2) + (3,1) → (3,2) ✓
except RuntimeError:
    pass  # Expected error
```

### Test Cases for Type Promotion

```python
# Test 1: Same category
assert torch.result_type(torch.int32, torch.int64) == torch.int64

# Test 2: Cross category
assert torch.result_type(torch.int32, torch.float32) == torch.float32

# Test 3: Complex preservation
assert torch.result_type(torch.float64, torch.complex64) == torch.complex128

# Test 4: Bool promotion
assert torch.result_type(torch.bool, torch.int32) == torch.int32
```

---

## Critical Files Reference

**Broadcasting**:
- [aten/src/ATen/ExpandUtils.h](../reference/pytorch/aten/src/ATen/ExpandUtils.h) - `infer_size()` implementation
- [aten/src/ATen/TensorIterator.cpp](../reference/pytorch/aten/src/ATen/TensorIterator.cpp) - Stride computation for broadcasting

**Type Promotion**:
- [c10/core/ScalarType.cpp](../reference/pytorch/c10/core/ScalarType.cpp) - `promoteTypes()` lookup table (lines 43-129)
- [aten/src/ATen/native/TypeProperties.cpp](../reference/pytorch/aten/src/ATen/native/TypeProperties.cpp) - `result_type()` implementation
- [c10/core/ScalarType.h](../reference/pytorch/c10/core/ScalarType.h) - Type utilities

**Testing**:
- `test/test_type_promotion.py` - Comprehensive promotion tests
- `test/test_ops.py` - Broadcasting tests

---

## Summary

**Broadcasting**:
- ✅ Aligns shapes from the right (trailing dimensions)
- ✅ Dimensions of size 1 are broadcast to match
- ✅ Uses stride=0 for zero-copy broadcasting
- ✅ Gradients are automatically summed along broadcast dimensions

**Type Promotion**:
- ✅ Follows category hierarchy: `bool < int < float < complex`
- ✅ Within category: promotes to wider type
- ✅ Across category: promotes to higher category
- ✅ Complex preserves value type of float operand
- ⚠️ Unsigned integer promotion is inconsistent (PyTorch limitation)
- ❌ Quantized and float8 types don't support promotion

**For MLX**:
- Adopt PyTorch's broadcasting rules exactly (NumPy-compatible)
- Implement type promotion matrix for MLX types
- Consider fixing unsigned integer promotion (cleaner than PyTorch)
- Use stride manipulation for efficient broadcasting

**Next Steps**:
1. Implement `infer_size()` for broadcasting in MLX
2. Create promotion matrix for MLX types
3. Test edge cases thoroughly (especially unsigned types, bfloat16)
4. Ensure gradient reduction works correctly for broadcast dimensions
