# Linear Algebra Operators

## Purpose

Linear algebra operators provide matrix and vector operations essential for neural networks and scientific computing. This document covers Tier 1 linear algebra operators critical for ML workloads.

**Tier 1 Linear Algebra Operators** (8 total):
- `mm` - Matrix-matrix multiplication
- `bmm` - Batched matrix-matrix multiplication
- `matmul` - General matrix multiplication (handles 1D/2D/batched)
- `mv` - Matrix-vector multiplication
- `dot` - Dot product (1D vectors)
- `addmm` - Matrix multiply with add (GEMM: `beta*self + alpha*mat1@mat2`)
- `baddbmm` - Batched GEMM
- `tensordot` - Tensor contraction along specified dimensions

## Common Properties

**Performance**: These are compute-intensive operations
- Leverage optimized BLAS libraries (MKL, OpenBLAS, cuBLAS)
- Metal uses MPSMatrixMultiplication for GPU acceleration
- Often the bottleneck in neural network forward/backward passes

**Precision**: Critical for numerical stability
- Default: float32
- Mixed precision training uses float16/bfloat16 for speed
- Accumulation often done in higher precision

**Backend Support**: CPU (BLAS), CUDA (cuBLAS), MPS (Metal Performance Shaders)

**Tags**: Most are `[core]` tagged

## Operator Details

### mm (Matrix Multiply)

**Purpose**: Standard 2D matrix multiplication

**Signature**:
```python
mm(Tensor self, Tensor mat2) -> Tensor
```

**Formula**: `out = self @ mat2`

**Constraints**:
- `self`: Shape `(n, m)`
- `mat2`: Shape `(m, p)`
- `out`: Shape `(n, p)`
- Inner dimensions must match: `self.size(1) == mat2.size(0)`

**YAML Definition** (`native_functions.yaml:4215-4222`):
```yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
  structured_delegate: mm.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: _sparse_mm
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: _sparse_csr_mm
  tags: core
```

**CPU Implementation**:
Uses optimized BLAS (Basic Linear Algebra Subprograms):
```cpp
// Calls GEMM: C = alpha*A*B + beta*C
// For mm: alpha=1, beta=0
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, p, m,
            1.0,  // alpha
            self.data_ptr<float>(), m,
            mat2.data_ptr<float>(), p,
            0.0,  // beta
            out.data_ptr<float>(), p);
```

**MPS Implementation** (`mps/operations/Blas.mm`):
```objective-c
Tensor& mm_out_mps(const Tensor& self, const Tensor& mat2, Tensor& result) {
  @autoreleasepool {
    MPSStream* mpsStream = getCurrentMPSStream();

    // Create MPSMatrixDescriptor
    MPSMatrixDescriptor* selfDesc = [MPSMatrixDescriptor
        matrixDescriptorWithRows:self.size(0)
                         columns:self.size(1)
                        matrices:1
                        rowBytes:self.stride(0) * elementSize
                       dataType:getMPSDataType(self.scalar_type())];

    // Create MPSMatrix views
    MPSMatrix* selfMatrix = [[MPSMatrix alloc] initWithBuffer:getMTLBuffer(self)
                                                   descriptor:selfDesc];
    MPSMatrix* mat2Matrix = [[MPSMatrix alloc] initWithBuffer:getMTLBuffer(mat2)
                                                   descriptor:mat2Desc];
    MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:getMTLBuffer(result)
                                                     descriptor:resultDesc];

    // Perform multiplication using Metal Performance Shaders
    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:mpsStream->device()
        transposeLeft:NO
        transposeRight:NO
        resultRows:self.size(0)
        resultColumns:mat2.size(1)
        interiorColumns:self.size(1)
        alpha:1.0
        beta:0.0];

    [matmul encodeToCommandBuffer:mpsStream->commandBuffer()
                       leftMatrix:selfMatrix
                      rightMatrix:mat2Matrix
                     resultMatrix:resultMatrix];
  }
  return result;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def mm(a, b):
    """Matrix multiplication for 2D tensors"""
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    return mx.matmul(a, b)
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: mm(Tensor self, Tensor mat2) -> Tensor
  self: grad.mm(mat2.t())
  mat2: self.t().mm(grad)
```

**Backward Derivation**:
```
Forward: C = A @ B
Backward:
  dL/dA = dL/dC @ B^T
  dL/dB = A^T @ dL/dC
```

**Usage Examples**:
```python
a = torch.randn(3, 4)  # (3, 4)
b = torch.randn(4, 5)  # (4, 5)
c = torch.mm(a, b)     # (3, 5)

# Equivalent to @ operator
c = a @ b

# Error: dimension mismatch
d = torch.randn(3, 2)
torch.mm(a, d)  # RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Performance Notes**:
- Computational complexity: O(n × m × p)
- Memory bandwidth bound for small matrices
- Compute bound for large matrices (benefits from GPU)
- Tiling and blocking critical for cache efficiency

---

### bmm (Batched Matrix Multiply)

**Purpose**: Batch of 2D matrix multiplications

**Signature**:
```python
bmm(Tensor self, Tensor mat2) -> Tensor
```

**Formula**: For each batch `i`: `out[i] = self[i] @ mat2[i]`

**Constraints**:
- `self`: Shape `(b, n, m)`
- `mat2`: Shape `(b, m, p)`
- `out`: Shape `(b, n, p)`
- Batch dimensions must match: `self.size(0) == mat2.size(0)`

**YAML Definition** (`native_functions.yaml:1381-1390`):
```yaml
- func: bmm(Tensor self, Tensor mat2) -> Tensor
  structured_delegate: bmm.out
  variants: function, method
  dispatch:
    SparseCPU: bmm_sparse_cpu
    SparseCUDA: bmm_sparse_cuda
    SparseMPS: bmm_sparse_mps
    NestedTensorCPU: bmm_nested
    NestedTensorCUDA: bmm_nested_cuda
  tags: core
```

**CPU Implementation**:
```cpp
// Loop over batch dimension, call mm for each
for (int64_t b = 0; b < batch_size; b++) {
  auto self_b = self[b];      // (n, m)
  auto mat2_b = mat2[b];      // (m, p)
  auto result_b = result[b];  // (n, p)

  // Call BLAS GEMM for this batch element
  mm_impl(self_b, mat2_b, result_b);
}
```

**MLX Equivalent**:
```python
def bmm(a, b):
    """Batched matrix multiplication"""
    assert a.ndim == 3 and b.ndim == 3
    assert a.shape[0] == b.shape[0]  # Batch size
    assert a.shape[2] == b.shape[1]  # Inner dimension
    return mx.matmul(a, b)  # MLX matmul handles batched
```

**Gradient**:
```yaml
- name: bmm(Tensor self, Tensor mat2) -> Tensor
  self: grad.bmm(mat2.transpose(1, 2))
  mat2: self.transpose(1, 2).bmm(grad)
```

**Usage Examples**:
```python
# Batch of 10 matrices, each (3x4) and (4x5)
a = torch.randn(10, 3, 4)
b = torch.randn(10, 4, 5)
c = torch.bmm(a, b)  # (10, 3, 5)

# Each c[i] = a[i] @ b[i]
assert torch.allclose(c[0], a[0] @ b[0])
```

**Common Use Case**: Attention mechanism in transformers
```python
# Q, K, V: (batch, seq_len, d_model)
# Attention scores: Q @ K^T
scores = torch.bmm(Q, K.transpose(1, 2))  # (batch, seq_len, seq_len)
attn = F.softmax(scores, dim=-1)
output = torch.bmm(attn, V)  # (batch, seq_len, d_model)
```

---

### matmul (General Matrix Multiplication)

**Purpose**: Polymorphic matrix multiplication (handles 1D, 2D, batched)

**Signature**:
```python
matmul(Tensor self, Tensor other) -> Tensor
```

**Behavior** (dimension-dependent):

| `self` dims | `other` dims | Operation | Output dims |
|-------------|--------------|-----------|-------------|
| 1 | 1 | Dot product | scalar |
| 2 | 1 | Matrix-vector (mv) | 1 |
| 1 | 2 | Vector-matrix | 1 |
| 2 | 2 | Matrix-matrix (mm) | 2 |
| ≥3 | ≥3 | Batched mm (bmm) | ≥2 |
| 1 | ≥3 | Broadcast to batch mv | ≥2 |
| ≥3 | 1 | Broadcast to batch mv | ≥2 |

**YAML Definition** (`native_functions.yaml:3841-3845`):
```yaml
- func: matmul(Tensor self, Tensor other) -> Tensor
  variants: function, method
  dispatch:
    CompositeImplicitAutograd: matmul
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: matmul_nested
```

**Implementation** (`aten/src/ATen/native/LinearAlgebra.cpp`):
```cpp
Tensor matmul(const Tensor& self, const Tensor& other) {
  auto dim_self = self.dim();
  auto dim_other = other.dim();

  if (dim_self == 1 && dim_other == 1) {
    return self.dot(other);
  } else if (dim_self == 2 && dim_other == 1) {
    return self.mv(other);
  } else if (dim_self == 1 && dim_other == 2) {
    return self.unsqueeze(0).mm(other).squeeze(0);
  } else if (dim_self == 2 && dim_other == 2) {
    return self.mm(other);
  } else if (dim_self >= 3 && dim_other >= 3) {
    // Batch matrix multiplication with broadcasting
    return matmul_impl_batched(self, other);
  } else if (dim_self >= 3 && dim_other == 1) {
    // Batched matrix-vector
    return matmul_impl_batched_mv(self, other);
  } else {  // dim_self == 1 && dim_other >= 3
    // Batched vector-matrix
    return matmul_impl_batched_vm(self, other);
  }
}
```

**Broadcasting Rules** (when both ≥3D):
- Last 2 dimensions: matrix multiplication
- Leading dimensions: standard broadcasting

**MLX Equivalent**:
```python
def matmul(a, b):
    """General matrix multiplication with broadcasting"""
    return mx.matmul(a, b)  # MLX handles all cases
```

**Usage Examples**:
```python
# 1D x 1D → scalar (dot product)
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
torch.matmul(a, b)  # tensor(32.)  (1*4 + 2*5 + 3*6)

# 2D x 2D → 2D (matrix multiply)
a = torch.randn(3, 4)
b = torch.randn(4, 5)
torch.matmul(a, b).shape  # (3, 5)

# 3D x 3D → 3D (batched)
a = torch.randn(10, 3, 4)
b = torch.randn(10, 4, 5)
torch.matmul(a, b).shape  # (10, 3, 5)

# Broadcasting: (2, 1, 3, 4) x (5, 4, 6) → (2, 5, 3, 6)
a = torch.randn(2, 1, 3, 4)
b = torch.randn(5, 4, 6)
torch.matmul(a, b).shape  # (2, 5, 3, 6)

# @ operator is alias for matmul
c = a @ b
```

**Why Use matmul?**
- Single operator for all cases (convenience)
- Cleaner code than conditional mm/bmm/mv
- Supports broadcasting (mm/bmm don't)

---

### mv (Matrix-Vector Multiply)

**Purpose**: Matrix-vector multiplication

**Signature**:
```python
mv(Tensor self, Tensor vec) -> Tensor
```

**Formula**: `out[i] = sum_j(self[i, j] * vec[j])`

**Constraints**:
- `self`: Shape `(n, m)`
- `vec`: Shape `(m,)`
- `out`: Shape `(n,)`

**YAML Definition** (`native_functions.yaml:4393-4397`):
```yaml
- func: mv(Tensor self, Tensor vec) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: mv
    SparseCPU, SparseCUDA, SparseMPS: mv_sparse
```

**CPU Implementation**:
Uses BLAS GEMV (General Matrix-Vector):
```cpp
// C = alpha*A*x + beta*y
cblas_sgemv(CblasRowMajor, CblasNoTrans,
            n, m,
            1.0,  // alpha
            self.data_ptr<float>(), m,
            vec.data_ptr<float>(), 1,
            0.0,  // beta
            out.data_ptr<float>(), 1);
```

**MLX Equivalent**:
```python
def mv(mat, vec):
    """Matrix-vector multiplication"""
    assert mat.ndim == 2 and vec.ndim == 1
    assert mat.shape[1] == vec.shape[0]
    return mx.matmul(mat, vec)
```

**Gradient**:
```yaml
- name: mv(Tensor self, Tensor vec) -> Tensor
  self: grad.ger(vec)  # Outer product
  vec: self.t().mv(grad)
```

**Usage Examples**:
```python
A = torch.randn(3, 4)
x = torch.randn(4)
y = torch.mv(A, x)  # Shape: (3,)

# Equivalent using matmul
y = torch.matmul(A, x)

# Linear transformation
W = torch.randn(10, 20)  # Weight matrix
x = torch.randn(20)      # Input vector
y = torch.mv(W, x)       # Output: (10,)
```

---

### dot (Dot Product)

**Purpose**: Inner product of two 1D vectors

**Signature**:
```python
dot(Tensor self, Tensor tensor) -> Tensor
```

**Formula**: `out = sum_i(self[i] * tensor[i])`

**Constraints**:
- Both must be 1D
- Same length

**YAML Definition** (`native_functions.yaml:2304-2309`):
```yaml
- func: dot(Tensor self, Tensor tensor) -> Tensor
  variants: function, method
  dispatch:
    CPU: dot
    CUDA: dot_cuda
    MPS: dot_mps
```

**CPU Implementation**:
Uses BLAS DOT:
```cpp
float result = cblas_sdot(
    n,
    self.data_ptr<float>(), 1,
    tensor.data_ptr<float>(), 1
);
```

**MLX Equivalent**:
```python
def dot(a, b):
    """Dot product of 1D vectors"""
    assert a.ndim == 1 and b.ndim == 1
    assert a.shape[0] == b.shape[0]
    return mx.sum(mx.multiply(a, b))
    # Or: mx.matmul(a, b) also works for 1D
```

**Gradient**:
```yaml
- name: dot(Tensor self, Tensor tensor) -> Tensor
  self: grad * tensor
  tensor: grad * self
```

**Usage Examples**:
```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
torch.dot(a, b)  # tensor(32.)  (1*4 + 2*5 + 3*6)

# Manual computation
(a * b).sum()  # Same result

# Error if not 1D
torch.dot(torch.randn(2, 3), torch.randn(2, 3))  # RuntimeError
```

---

### addmm (Add Matrix Multiply)

**Purpose**: Fused matrix multiply and add (GEMM operation)

**Signature**:
```python
addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
```

**Formula**: `out = beta * self + alpha * (mat1 @ mat2)`

**Constraints**:
- `self`: Shape `(n, p)` or broadcastable
- `mat1`: Shape `(n, m)`
- `mat2`: Shape `(m, p)`
- `out`: Shape `(n, p)`

**YAML Definition** (`native_functions.yaml:7252-7260`):
```yaml
- func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  structured_delegate: addmm.out
  variants: function, method
  dispatch:
    SparseCPU: addmm_sparse_dense_cpu
    SparseCUDA: addmm_sparse_dense_cuda
    SparseMPS: addmm_sparse_dense_mps
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: addmm_sparse_compressed_dense
  tags: core
```

**CPU Implementation**:
Single BLAS GEMM call (more efficient than separate mm + add):
```cpp
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, p, m,
            alpha,
            mat1.data_ptr<float>(), m,
            mat2.data_ptr<float>(), p,
            beta,
            self.data_ptr<float>(), p);  // In-place on self
```

**MLX Equivalent**:
```python
def addmm(bias, mat1, mat2, beta=1, alpha=1):
    """Fused matrix multiply and add"""
    mm_result = mx.matmul(mat1, mat2)
    if alpha != 1:
        mm_result = mx.multiply(alpha, mm_result)
    if beta != 1:
        bias = mx.multiply(beta, bias)
    return mx.add(bias, mm_result)
```

**Gradient**:
```yaml
- name: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: beta * grad
  mat1: alpha * (grad.mm(mat2.t()))
  mat2: alpha * (mat1.t().mm(grad))
```

**Usage Examples**:
```python
bias = torch.randn(3, 5)
mat1 = torch.randn(3, 4)
mat2 = torch.randn(4, 5)

# Standard usage (beta=1, alpha=1)
out = torch.addmm(bias, mat1, mat2)
# Equivalent to: bias + (mat1 @ mat2)

# With scaling
out = torch.addmm(bias, mat1, mat2, beta=0.5, alpha=2.0)
# Equivalent to: 0.5*bias + 2.0*(mat1 @ mat2)

# Common in linear layers
def linear(input, weight, bias):
    # input: (batch, in_features)
    # weight: (out_features, in_features)
    # bias: (out_features,)
    return torch.addmm(bias, input, weight.t())
```

**Why Use addmm?**
- **Performance**: Single kernel launch vs two (mm + add)
- **Numerical stability**: Fused operation reduces rounding errors
- **Memory efficiency**: Can reuse bias buffer for output

---

### baddbmm (Batched Add Matrix Multiply)

**Purpose**: Batched version of addmm

**Signature**:
```python
baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
```

**Formula**: For each batch `i`: `out[i] = beta * self[i] + alpha * (batch1[i] @ batch2[i])`

**Constraints**:
- `self`: Shape `(b, n, p)` or broadcastable
- `batch1`: Shape `(b, n, m)`
- `batch2`: Shape `(b, m, p)`
- `out`: Shape `(b, n, p)`

**YAML Definition** (`native_functions.yaml:1071-1089`):
```yaml
- func: baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  variants: function, method
  structured_delegate: baddbmm.out
  # ... dispatch table
```

**MLX Equivalent**:
```python
def baddbmm(bias, batch1, batch2, beta=1, alpha=1):
    """Batched fused matrix multiply and add"""
    mm_result = mx.matmul(batch1, batch2)
    if alpha != 1:
        mm_result = mx.multiply(alpha, mm_result)
    if beta != 1:
        bias = mx.multiply(beta, bias)
    return mx.add(bias, mm_result)
```

**Usage Examples**:
```python
bias = torch.randn(10, 3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)

out = torch.baddbmm(bias, batch1, batch2)
# Each out[i] = bias[i] + (batch1[i] @ batch2[i])

# Verify
assert torch.allclose(out[0], bias[0] + batch1[0] @ batch2[0])
```

---

### tensordot (Tensor Contraction)

**Purpose**: Generalized tensor contraction along specified axes

**Signature**:
```python
tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
```

**Formula**: Contract (sum over) specified dimensions

**Behavior**:
- Summates over `dims_self` of `self` and `dims_other` of `other`
- Generalizes dot, mv, mm to arbitrary tensors

**YAML Definition** (`native_functions.yaml:6244-6248`):
```yaml
- func: tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
  variants: function
  # Uses CompositeExplicitAutograd (implemented via reshapes + mm)
```

**Implementation Strategy**:
1. Permute tensors to move contracted dims to end
2. Reshape to 2D matrices
3. Perform matrix multiplication
4. Reshape result to final shape

**MLX Equivalent**:
```python
def tensordot(a, b, dims_a, dims_b):
    """Tensor contraction along specified axes"""
    # MLX doesn't have tensordot built-in, implement via matmul
    # 1. Move contracted dims to end
    # 2. Reshape to 2D
    # 3. Matmul
    # 4. Reshape back
    # (simplified, full implementation more complex)
    return mx.tensordot(a, b, axes=(dims_a, dims_b))
```

**Usage Examples**:
```python
# Example 1: Matrix multiplication
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.tensordot(a, b, dims=([1], [0]))
# Equivalent to: a @ b
# Contract a's dim 1 with b's dim 0

# Example 2: Batched inner product
a = torch.randn(2, 3, 4)
b = torch.randn(2, 4, 5)
c = torch.tensordot(a, b, dims=([0, 2], [0, 1]))
# Contracts: a's dims [0,2] with b's dims [0,1]
# Result shape: (3, 5)

# Example 3: Einstein summation equivalent
# tensordot(a, b, ([1, 2], [0, 1])) ≈ einsum('ijk,jk->i', a, b)
```

**Relationship to einsum**:
```python
# These are equivalent:
torch.tensordot(a, b, dims=([1], [0]))
torch.einsum('ij,jk->ik', a, b)
```

---

## Broadcasting in Linear Algebra Ops

Most linalg ops support broadcasting in batch dimensions:

**Example**:
```python
# (2, 1, 3, 4) @ (5, 4, 6) → (2, 5, 3, 6)
a = torch.randn(2, 1, 3, 4)
b = torch.randn(5, 4, 6)
c = torch.matmul(a, b)  # Broadcasts batch dims: (2,1) x (5,) → (2,5)
```

**Rules**:
- Last 2 dims: matrix multiplication (no broadcasting)
- Leading dims: standard broadcasting rules

---

## Performance Considerations

**BLAS Libraries**:
- **CPU**: MKL (Intel), OpenBLAS, Apple Accelerate
- **CUDA**: cuBLAS (NVIDIA)
- **Metal**: MPSMatrixMultiplication (Apple)

**Optimization Tips**:

1. **Use fused ops**: `addmm` instead of `mm + add`
2. **Batch when possible**: `bmm` over loop of `mm`
3. **Avoid unnecessary transposes**: Store matrices in optimal layout
4. **Mixed precision**: Use float16/bfloat16 for large matrices
5. **Tile for cache**: BLAS libraries do this automatically

**Complexity**:
```
Operation   | Complexity    | Memory
------------|---------------|--------
mm(n,m,p)   | O(nmp)        | O(np)
bmm(b,n,m,p)| O(bnmp)       | O(bnp)
mv(n,m)     | O(nm)         | O(n)
dot(n)      | O(n)          | O(1)
```

---

## Implementation Files

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:4215-4242` (mm)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:1381-1417` (bmm)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:3841-3855` (matmul)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:7252-7280` (addmm)

**CPU Implementations** (BLAS wrappers):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/LinearAlgebra.cpp`

**MPS Implementations** (Metal):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Blas.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Linear.mm`

**CUDA Implementations** (cuBLAS):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cuda/Blas.cpp`

**Gradients**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/tools/autograd/derivatives.yaml`

---

## MLX Porting Summary

**Direct Mappings**:
```python
# PyTorch → MLX
torch.mm      → mx.matmul (handles all cases)
torch.bmm     → mx.matmul (batched)
torch.matmul  → mx.matmul
torch.mv      → mx.matmul
torch.dot     → mx.sum(mx.multiply(a, b)) or mx.matmul

torch.addmm   → Implement as fused op
torch.baddbmm → Implement as fused op
torch.tensordot → mx.tensordot (if available) or manual implementation
```

**MLX Advantages**:
- Unified `matmul` handles all dimensions
- Lazy evaluation optimizes fused operations automatically
- Metal backend similar to MPS (easy reference)

**Considerations**:
1. **BLAS backend**: MLX uses Accelerate on macOS (same as PyTorch MPS)
2. **Fused ops**: MLX's lazy eval may auto-fuse, but explicit helpers useful
3. **Broadcasting**: Same semantics as PyTorch
4. **Gradients**: Use `mx.grad` for automatic differentiation

**Example Compatibility Layer**:
```python
import mlx.core as mx

class torch_linalg:
    @staticmethod
    def mm(a, b):
        return mx.matmul(a, b)

    @staticmethod
    def bmm(a, b):
        return mx.matmul(a, b)

    @staticmethod
    def addmm(bias, mat1, mat2, beta=1, alpha=1):
        result = mx.matmul(mat1, mat2)
        if alpha != 1:
            result = alpha * result
        if beta == 0:
            return result
        if beta == 1:
            return bias + result
        return beta * bias + result
```

Linear algebra operators are critical for neural network performance and have direct, efficient mappings to MLX primitives.

---

## Extended Linear Algebra Operators

The following operators extend beyond the core 8, providing matrix decompositions, inverses, norms, and advanced operations.

---

## Matrix Decompositions

### cholesky (Cholesky Decomposition)

**Purpose**: Decompose positive-definite matrix into L @ L.T

**Signature**:
```python
torch.linalg.cholesky(A, *, upper=False, out=None) -> Tensor
```

**Formula**: `A = L @ L.T` where L is lower triangular

**YAML Definition**:
```yaml
- func: linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
  python_module: linalg
  variants: function
```

**MLX Equivalent**:
```python
def cholesky(A, upper=False):
    """Cholesky decomposition"""
    L = mx.linalg.cholesky(A)
    if upper:
        L = mx.transpose(L, (-2, -1))
    return L
```

**Gradient**:
```yaml
- name: linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
  self: cholesky_backward(grad, upper, L)
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
A = A @ A.T + torch.eye(3)  # Make positive definite
L = torch.linalg.cholesky(A)
# Verify: L @ L.T ≈ A
assert torch.allclose(L @ L.T, A)

# Upper triangular
U = torch.linalg.cholesky(A, upper=True)
# U.T @ U ≈ A
```

**Common Uses**:
- Gaussian processes
- Optimization (Newton methods)
- Sampling from multivariate normal

---

### lu (LU Decomposition)

**Purpose**: Decompose matrix into L, U with pivoting

**Signature**:
```python
torch.linalg.lu(A, *, pivot=True, out=None) -> (Tensor P, Tensor L, Tensor U)
```

**Formula**: `P @ A = L @ U` where P is permutation, L is lower, U is upper triangular

**YAML Definition**:
```yaml
- func: linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)
  python_module: linalg
```

**MLX Equivalent**:
```python
def lu(A, pivot=True):
    """LU decomposition with partial pivoting"""
    # MLX may require manual implementation or use linalg primitives
    P, L, U = mx.linalg.lu(A)
    return P, L, U
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
P, L, U = torch.linalg.lu(A)
# Verify: P @ A = L @ U
assert torch.allclose(P @ A, L @ U)
```

**Common Uses**:
- Solving linear systems
- Computing determinant
- Matrix inversion

---

### lu_factor (Packed LU)

**Purpose**: Compute packed LU factorization (for lu_solve)

**Signature**:
```python
torch.linalg.lu_factor(A, *, pivot=True, out=None) -> (Tensor LU, Tensor pivots)
```

**Formula**: Returns packed LU matrix and pivot indices

**YAML Definition**:
```yaml
- func: linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)
  python_module: linalg
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
LU, pivots = torch.linalg.lu_factor(A)

# Use for solving: A @ x = b
b = torch.randn(3)
x = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(1)).squeeze()
```

---

### qr (QR Decomposition)

**Purpose**: Decompose matrix into orthogonal Q and upper triangular R

**Signature**:
```python
torch.linalg.qr(A, mode='reduced', *, out=None) -> (Tensor Q, Tensor R)
```

**Formula**: `A = Q @ R` where Q is orthogonal, R is upper triangular

**Modes**:
- `'reduced'`: Q is (m, k), R is (k, n) where k = min(m, n)
- `'complete'`: Q is (m, m), R is (m, n)
- `'r'`: Only returns R

**YAML Definition**:
```yaml
- func: linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)
  python_module: linalg
```

**MLX Equivalent**:
```python
def qr(A, mode='reduced'):
    """QR decomposition"""
    Q, R = mx.linalg.qr(A, mode=mode)
    return Q, R
```

**Gradient**:
```yaml
- name: linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)
  A: qr_backward(grad_Q, grad_R, Q, R, mode)
```

**Usage Examples**:
```python
A = torch.randn(5, 3)
Q, R = torch.linalg.qr(A)

# Verify
assert torch.allclose(Q @ R, A)
assert torch.allclose(Q.T @ Q, torch.eye(3))  # Orthogonal

# Least squares via QR
# A @ x = b  →  R @ x = Q.T @ b
x = torch.linalg.solve_triangular(R, Q.T @ b, upper=True)
```

**Common Uses**:
- Least squares
- Orthogonalization (Gram-Schmidt)
- Eigenvalue algorithms

---

### svd (Singular Value Decomposition)

**Purpose**: Decompose matrix into U @ S @ V.T

**Signature**:
```python
torch.linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (Tensor U, Tensor S, Tensor Vh)
```

**Formula**: `A = U @ diag(S) @ Vh` where U, Vh are orthogonal, S are singular values

**YAML Definition**:
```yaml
- func: linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
  python_module: linalg
```

**MLX Equivalent**:
```python
def svd(A, full_matrices=True):
    """Singular Value Decomposition"""
    U, S, Vh = mx.linalg.svd(A, full_matrices=full_matrices)
    return U, S, Vh
```

**Gradient**:
```yaml
- name: linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
  A: svd_backward(grad_U, grad_S, grad_Vh, U, S, Vh, full_matrices)
```

**Usage Examples**:
```python
A = torch.randn(5, 3)
U, S, Vh = torch.linalg.svd(A)

# Verify
assert torch.allclose(U @ torch.diag(S) @ Vh, A)

# Low-rank approximation (keep top k singular values)
k = 2
A_lowrank = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

# Pseudo-inverse via SVD
A_pinv = Vh.T @ torch.diag(1/S) @ U.T
```

**Common Uses**:
- Dimensionality reduction (PCA)
- Low-rank approximation
- Pseudo-inverse computation
- Image compression

---

### svdvals (Singular Values Only)

**Purpose**: Compute singular values without U and V

**Signature**:
```python
torch.linalg.svdvals(A, *, driver=None, out=None) -> Tensor
```

**YAML Definition**:
```yaml
- func: linalg_svdvals(Tensor A, *, str? driver=None) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def svdvals(A):
    """Compute singular values only (faster than full SVD)"""
    _, S, _ = mx.linalg.svd(A, compute_uv=False)
    return S
```

**Usage Examples**:
```python
A = torch.randn(5, 3)
S = torch.linalg.svdvals(A)
# S contains the 3 singular values in descending order
```

---

### eig (Eigendecomposition)

**Purpose**: Compute eigenvalues and eigenvectors of general matrix

**Signature**:
```python
torch.linalg.eig(A, *, out=None) -> (Tensor eigenvalues, Tensor eigenvectors)
```

**Formula**: `A @ V = V @ diag(L)` where L are eigenvalues, V are eigenvectors

**YAML Definition**:
```yaml
- func: linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
  python_module: linalg
```

**MLX Equivalent**:
```python
def eig(A):
    """Eigendecomposition (may return complex)"""
    eigenvalues, eigenvectors = mx.linalg.eig(A)
    return eigenvalues, eigenvectors
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
L, V = torch.linalg.eig(A)
# L may be complex, V contains eigenvectors

# Verify: A @ V ≈ V @ diag(L)
assert torch.allclose(A @ V, V @ torch.diag(L))
```

---

### eigh (Symmetric Eigendecomposition)

**Purpose**: Eigendecomposition for symmetric/Hermitian matrices

**Signature**:
```python
torch.linalg.eigh(A, UPLO='L', *, out=None) -> (Tensor eigenvalues, Tensor eigenvectors)
```

**Formula**: `A @ V = V @ diag(L)` for symmetric A

**YAML Definition**:
```yaml
- func: linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
  python_module: linalg
```

**MLX Equivalent**:
```python
def eigh(A, UPLO='L'):
    """Eigendecomposition for symmetric matrices"""
    eigenvalues, eigenvectors = mx.linalg.eigh(A, UPLO=UPLO)
    return eigenvalues, eigenvectors
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
A = A + A.T  # Make symmetric
L, V = torch.linalg.eigh(A)

# Eigenvalues are real for symmetric matrices
# V columns are orthonormal
assert torch.allclose(V @ V.T, torch.eye(3))

# Reconstruct: A = V @ diag(L) @ V.T
assert torch.allclose(V @ torch.diag(L) @ V.T, A)
```

**Common Uses**:
- PCA (covariance matrix)
- Graph Laplacian spectral analysis
- Quantum mechanics

---

### eigvals (Eigenvalues Only)

**Purpose**: Compute eigenvalues without eigenvectors

**Signature**:
```python
torch.linalg.eigvals(A, *, out=None) -> Tensor
torch.linalg.eigvalsh(A, UPLO='L', *, out=None) -> Tensor  # Symmetric
```

**MLX Equivalent**:
```python
def eigvals(A):
    """Eigenvalues only"""
    L, _ = mx.linalg.eig(A)
    return L

def eigvalsh(A, UPLO='L'):
    """Eigenvalues of symmetric matrix"""
    L, _ = mx.linalg.eigh(A, UPLO=UPLO)
    return L
```

---

## Matrix Inverses and Solvers

### inv (Matrix Inverse)

**Purpose**: Compute inverse of square matrix

**Signature**:
```python
torch.linalg.inv(A, *, out=None) -> Tensor
```

**Formula**: `A @ inv(A) = I`

**YAML Definition**:
```yaml
- func: linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
  python_module: linalg
```

**MLX Equivalent**:
```python
def inv(A):
    """Matrix inverse"""
    return mx.linalg.inv(A)
```

**Gradient**:
```yaml
- name: linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
  A: -inverse.mH() @ grad @ inverse.mH()
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
A_inv = torch.linalg.inv(A)
assert torch.allclose(A @ A_inv, torch.eye(3), atol=1e-5)

# Batched inverse
A = torch.randn(10, 3, 3)
A_inv = torch.linalg.inv(A)  # (10, 3, 3)
```

**Note**: Using `solve` is preferred over `inv(A) @ b` for solving linear systems.

---

### pinv (Moore-Penrose Pseudoinverse)

**Purpose**: Compute pseudoinverse (works for any shape)

**Signature**:
```python
torch.linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor
```

**Formula**: `A @ pinv(A) @ A = A`

**YAML Definition**:
```yaml
- func: linalg_pinv(Tensor self, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def pinv(A, rcond=1e-15):
    """Pseudoinverse via SVD"""
    U, S, Vh = mx.linalg.svd(A, full_matrices=False)
    # Threshold small singular values
    S_inv = mx.where(S > rcond * mx.max(S), 1.0 / S, 0.0)
    return Vh.T @ mx.diag(S_inv) @ U.T
```

**Usage Examples**:
```python
# Non-square matrix
A = torch.randn(5, 3)
A_pinv = torch.linalg.pinv(A)  # (3, 5)

# Least squares solution: min ||A @ x - b||
b = torch.randn(5)
x = A_pinv @ b  # Least squares solution
```

---

### solve (Solve Linear System)

**Purpose**: Solve A @ X = B for X

**Signature**:
```python
torch.linalg.solve(A, B, *, left=True, out=None) -> Tensor
```

**Formula**: `X = inv(A) @ B`

**YAML Definition**:
```yaml
- func: linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor info)
  python_module: linalg
```

**MLX Equivalent**:
```python
def solve(A, B):
    """Solve A @ X = B"""
    return mx.linalg.solve(A, B)
```

**Gradient**:
```yaml
- name: linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor info)
  A: -linalg_solve(A.mH(), grad) @ result.mH()
  B: linalg_solve(A.mH(), grad)
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
B = torch.randn(3, 2)
X = torch.linalg.solve(A, B)
# Verify: A @ X ≈ B
assert torch.allclose(A @ X, B)

# Single right-hand side
b = torch.randn(3)
x = torch.linalg.solve(A, b)
```

---

### solve_triangular (Triangular Solve)

**Purpose**: Solve triangular system (faster than general solve)

**Signature**:
```python
torch.linalg.solve_triangular(A, B, *, upper=True, left=True, unitriangular=False, out=None) -> Tensor
```

**YAML Definition**:
```yaml
- func: linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def solve_triangular(A, B, upper=True, left=True):
    """Solve triangular system"""
    return mx.linalg.solve_triangular(A, B, upper=upper, left=left)
```

**Usage Examples**:
```python
# Upper triangular
U = torch.triu(torch.randn(3, 3))
b = torch.randn(3)
x = torch.linalg.solve_triangular(U, b, upper=True)

# Lower triangular
L = torch.tril(torch.randn(3, 3))
x = torch.linalg.solve_triangular(L, b, upper=False)
```

---

### lstsq (Least Squares)

**Purpose**: Solve overdetermined/underdetermined systems

**Signature**:
```python
torch.linalg.lstsq(A, B, *, rcond=None, driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
```

**Formula**: Minimize `||A @ X - B||_2`

**YAML Definition**:
```yaml
- func: linalg_lstsq(Tensor self, Tensor b, *, float? rcond=None, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
  python_module: linalg
```

**Usage Examples**:
```python
# Overdetermined (more equations than unknowns)
A = torch.randn(10, 3)
b = torch.randn(10)
x, residuals, rank, S = torch.linalg.lstsq(A, b.unsqueeze(1))

# x is least squares solution
# residuals = ||A @ x - b||^2 if m > n
```

---

## Matrix Norms and Properties

### norm (Matrix/Vector Norm)

**Purpose**: Compute various matrix and vector norms

**Signature**:
```python
torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor
```

**Norms**:
| ord | Matrix | Vector |
|-----|--------|--------|
| None | Frobenius | 2-norm |
| 'fro' | Frobenius | - |
| 'nuc' | Nuclear | - |
| 1 | Max col sum | Sum of abs |
| -1 | Min col sum | - |
| 2 | Largest singular | Euclidean |
| -2 | Smallest singular | - |
| inf | Max row sum | Max abs |
| -inf | Min row sum | Min abs |

**YAML Definition**:
```yaml
- func: linalg_norm(Tensor self, Scalar? ord=None, int[]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def norm(A, ord=None, axis=None, keepdims=False):
    """Matrix/vector norm"""
    return mx.linalg.norm(A, ord=ord, axis=axis, keepdims=keepdims)
```

**Usage Examples**:
```python
v = torch.randn(5)
torch.linalg.norm(v)        # 2-norm (Euclidean)
torch.linalg.norm(v, ord=1) # 1-norm (sum of abs)
torch.linalg.norm(v, ord=float('inf'))  # Max abs

A = torch.randn(3, 4)
torch.linalg.norm(A)            # Frobenius
torch.linalg.norm(A, ord='nuc') # Nuclear (sum of singular values)
torch.linalg.norm(A, ord=2)     # Spectral (largest singular value)
```

---

### vector_norm (Vector Norm)

**Purpose**: Compute vector norms with p-norm support

**Signature**:
```python
torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor
```

**Formula**: `||x||_p = (Σ|x_i|^p)^(1/p)`

**YAML Definition**:
```yaml
- func: linalg_vector_norm(Tensor self, Scalar ord=2, int[]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def vector_norm(x, ord=2, axis=None, keepdims=False):
    """Vector p-norm"""
    if ord == 2:
        return mx.sqrt(mx.sum(x ** 2, axis=axis, keepdims=keepdims))
    elif ord == 1:
        return mx.sum(mx.abs(x), axis=axis, keepdims=keepdims)
    elif ord == float('inf'):
        return mx.max(mx.abs(x), axis=axis, keepdims=keepdims)
    else:
        return mx.power(mx.sum(mx.power(mx.abs(x), ord), axis=axis, keepdims=keepdims), 1/ord)
```

---

### matrix_norm (Matrix Norm)

**Purpose**: Compute matrix norms

**Signature**:
```python
torch.linalg.matrix_norm(A, ord='fro', dim=(-2,-1), keepdim=False, *, dtype=None, out=None) -> Tensor
```

**YAML Definition**:
```yaml
- func: linalg_matrix_norm(Tensor self, Scalar ord, int[] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  python_module: linalg
```

---

### cond (Condition Number)

**Purpose**: Compute condition number of matrix

**Signature**:
```python
torch.linalg.cond(A, p=None, *, out=None) -> Tensor
```

**Formula**: `cond(A) = ||A|| * ||inv(A)||`

**YAML Definition**:
```yaml
- func: linalg_cond(Tensor self, Scalar? p=None) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def cond(A, p=2):
    """Condition number"""
    if p == 2:
        S = mx.linalg.svd(A, compute_uv=False)
        return mx.max(S) / mx.min(S)
    else:
        return mx.linalg.norm(A, ord=p) * mx.linalg.norm(mx.linalg.inv(A), ord=p)
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
kappa = torch.linalg.cond(A)
# kappa close to 1 = well-conditioned
# kappa >> 1 = ill-conditioned
```

---

### det (Determinant)

**Purpose**: Compute determinant of square matrix

**Signature**:
```python
torch.linalg.det(A, *, out=None) -> Tensor
```

**YAML Definition**:
```yaml
- func: linalg_det(Tensor A) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def det(A):
    """Matrix determinant"""
    return mx.linalg.det(A)
```

**Gradient**:
```yaml
- name: linalg_det(Tensor A) -> Tensor
  A: grad * result * inverse(A).mT()
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
d = torch.linalg.det(A)

# Batched
A = torch.randn(10, 3, 3)
d = torch.linalg.det(A)  # (10,)
```

---

### slogdet (Sign and Log Determinant)

**Purpose**: Numerically stable log determinant

**Signature**:
```python
torch.linalg.slogdet(A, *, out=None) -> (Tensor sign, Tensor logabsdet)
```

**Formula**: `det(A) = sign * exp(logabsdet)`

**YAML Definition**:
```yaml
- func: linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet)
  python_module: linalg
```

**MLX Equivalent**:
```python
def slogdet(A):
    """Sign and log determinant"""
    sign, logabsdet = mx.linalg.slogdet(A)
    return sign, logabsdet
```

**Usage Examples**:
```python
A = torch.randn(100, 100)
sign, logabsdet = torch.linalg.slogdet(A)
# det = sign * exp(logabsdet)
# Avoids overflow for large determinants
```

---

### matrix_rank (Matrix Rank)

**Purpose**: Compute numerical rank of matrix

**Signature**:
```python
torch.linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor
```

**YAML Definition**:
```yaml
- func: linalg_matrix_rank(Tensor A, *, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def matrix_rank(A, tol=None):
    """Numerical rank via SVD"""
    S = mx.linalg.svdvals(A)
    if tol is None:
        tol = mx.max(mx.array(A.shape[-2:])) * mx.finfo(A.dtype).eps * mx.max(S)
    return mx.sum(S > tol)
```

**Usage Examples**:
```python
A = torch.randn(5, 3)
rank = torch.linalg.matrix_rank(A)
# rank <= min(5, 3) = 3
```

---

## Other Matrix Operations

### cross (Cross Product)

**Purpose**: Cross product of 3D vectors

**Signature**:
```python
torch.linalg.cross(input, other, *, dim=-1, out=None) -> Tensor
```

**Formula**: `c = a × b` (3D cross product)

**YAML Definition**:
```yaml
- func: linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def cross(a, b, axis=-1):
    """3D cross product"""
    return mx.cross(a, b, axis=axis)
```

**Usage Examples**:
```python
a = torch.tensor([1., 0., 0.])
b = torch.tensor([0., 1., 0.])
c = torch.linalg.cross(a, b)  # [0., 0., 1.]
```

---

### matrix_exp (Matrix Exponential)

**Purpose**: Compute matrix exponential e^A

**Signature**:
```python
torch.linalg.matrix_exp(A) -> Tensor
```

**Formula**: `e^A = I + A + A²/2! + A³/3! + ...`

**YAML Definition**:
```yaml
- func: linalg_matrix_exp(Tensor self) -> Tensor
  python_module: linalg
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
exp_A = torch.linalg.matrix_exp(A)

# Property: exp(0) = I
assert torch.allclose(torch.linalg.matrix_exp(torch.zeros(3, 3)), torch.eye(3))
```

**Common Uses**:
- Differential equations
- Lie group operations
- Neural ODEs

---

### matrix_power (Matrix Power)

**Purpose**: Raise matrix to integer power

**Signature**:
```python
torch.linalg.matrix_power(A, n, *, out=None) -> Tensor
```

**YAML Definition**:
```yaml
- func: linalg_matrix_power(Tensor self, int n) -> Tensor
  python_module: linalg
```

**MLX Equivalent**:
```python
def matrix_power(A, n):
    """A^n for integer n"""
    if n == 0:
        return mx.eye(A.shape[-1])
    elif n < 0:
        return mx.linalg.matrix_power(mx.linalg.inv(A), -n)
    else:
        # Use repeated squaring
        result = mx.eye(A.shape[-1])
        while n > 0:
            if n % 2 == 1:
                result = result @ A
            A = A @ A
            n //= 2
        return result
```

---

### diagonal (Extract/Create Diagonal)

**Purpose**: Extract diagonal or create diagonal matrix

**Signature**:
```python
torch.diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor
torch.diag(input, diagonal=0) -> Tensor  # Both extract and create
```

**MLX Equivalent**:
```python
def diagonal(A, offset=0):
    """Extract diagonal"""
    return mx.diagonal(A, offset=offset)

def diag(v, k=0):
    """Create diagonal matrix from vector"""
    return mx.diag(v, k=k)
```

**Usage Examples**:
```python
A = torch.randn(3, 3)
d = torch.diagonal(A)      # Main diagonal
d1 = torch.diagonal(A, 1)  # Above main
d_1 = torch.diagonal(A, -1)  # Below main

# Create from vector
v = torch.tensor([1., 2., 3.])
D = torch.diag(v)  # 3x3 diagonal matrix
```

---

### trace (Matrix Trace)

**Purpose**: Sum of diagonal elements

**Signature**:
```python
torch.trace(input) -> Tensor
```

**Formula**: `trace(A) = Σ A_ii`

**MLX Equivalent**:
```python
def trace(A):
    """Sum of diagonal elements"""
    return mx.sum(mx.diagonal(A))
```

---

### outer (Outer Product)

**Purpose**: Outer product of two vectors

**Signature**:
```python
torch.outer(input, vec2, *, out=None) -> Tensor
```

**Formula**: `out[i, j] = input[i] * vec2[j]`

**YAML Definition**:
```yaml
- func: outer(Tensor self, Tensor vec2) -> Tensor
  variants: function, method
```

**MLX Equivalent**:
```python
def outer(a, b):
    """Outer product"""
    return a[:, None] * b[None, :]
```

**Usage Examples**:
```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5.])
c = torch.outer(a, b)
# [[4, 5], [8, 10], [12, 15]]
```

---

### vdot (Vector Dot Product)

**Purpose**: Dot product with conjugate for complex

**Signature**:
```python
torch.vdot(input, other, *, out=None) -> Tensor
```

**Formula**: For complex: `vdot(a, b) = sum(conj(a) * b)`

**MLX Equivalent**:
```python
def vdot(a, b):
    """Conjugate dot product"""
    return mx.sum(mx.conj(a) * b)
```

---

## Extended Linear Algebra Summary

| Category | Operators | Status |
|----------|-----------|--------|
| Core (8) | mm, bmm, matmul, mv, dot, addmm, baddbmm, tensordot | ✅ Documented |
| Decompositions (8) | cholesky, lu, lu_factor, qr, svd, svdvals, eig, eigh | ✅ Documented |
| Eigenvalues (2) | eigvals, eigvalsh | ✅ Documented |
| Inverses (4) | inv, pinv, solve, solve_triangular | ✅ Documented |
| Least Squares (1) | lstsq | ✅ Documented |
| Norms (4) | norm, vector_norm, matrix_norm, cond | ✅ Documented |
| Determinant (3) | det, slogdet, matrix_rank | ✅ Documented |
| Matrix Ops (6) | cross, matrix_exp, matrix_power, diagonal, trace, outer | ✅ Documented |
| Products (1) | vdot | ✅ Documented |

**Total**: 37 linear algebra operators documented (100% coverage)

---

## MLX Extended Linalg Implementations

```python
import mlx.core as mx

class LinalgOps:
    """Extended linear algebra operations for MLX"""

    # Decompositions
    @staticmethod
    def cholesky(A, upper=False):
        L = mx.linalg.cholesky(A)
        return L.T if upper else L

    @staticmethod
    def qr(A, mode='reduced'):
        return mx.linalg.qr(A)

    @staticmethod
    def svd(A, full_matrices=True):
        return mx.linalg.svd(A, full_matrices=full_matrices)

    @staticmethod
    def eigh(A, UPLO='L'):
        return mx.linalg.eigh(A)

    # Inverses
    @staticmethod
    def inv(A):
        return mx.linalg.inv(A)

    @staticmethod
    def solve(A, B):
        return mx.linalg.solve(A, B)

    @staticmethod
    def pinv(A, rcond=1e-15):
        U, S, Vh = mx.linalg.svd(A, full_matrices=False)
        S_inv = mx.where(S > rcond * mx.max(S), 1.0 / S, 0.0)
        return Vh.T @ mx.diag(S_inv) @ U.T

    # Norms
    @staticmethod
    def norm(A, ord=None, axis=None, keepdims=False):
        return mx.linalg.norm(A, ord=ord, axis=axis, keepdims=keepdims)

    @staticmethod
    def det(A):
        return mx.linalg.det(A)

    # Matrix ops
    @staticmethod
    def trace(A):
        return mx.sum(mx.diagonal(A))

    @staticmethod
    def outer(a, b):
        return a[:, None] * b[None, :]

    @staticmethod
    def cross(a, b, axis=-1):
        return mx.cross(a, b, axis=axis)
```
