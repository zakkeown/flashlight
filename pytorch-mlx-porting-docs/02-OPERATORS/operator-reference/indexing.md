# Indexing and Selection Operators

## Purpose

Indexing operators enable flexible data access and manipulation patterns essential for deep learning. This document covers Tier 1 indexing operators used in attention mechanisms, embedding lookups, and tensor manipulation.

**Tier 1 Indexing Operators** (5 total):
- `gather` - Gather values along an axis using indices
- `scatter` - Scatter values to specific indices
- `index_select` - Select slices along a dimension
- `masked_fill` - Fill elements where mask is true
- `where` - Select elements based on condition

## Common Properties

**Tags**: Most are `[core]`

**Index Tensors**: Usually `int64` (LongTensor)

**Broadcasting**: Supported with careful dimension matching

**Gradients**: Flow through indices for gather/scatter operations

**Performance**: Often memory-bound, benefit from coalescing

## Operator Details

### gather (Gather Along Dimension)

**Purpose**: Collect values from tensor using index tensor

**Signature**:
```python
gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
```

**Behavior**:
```python
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

**Constraints**:
- `index.dtype` must be `int64`
- `out.shape == index.shape`
- `index` values must be in range `[0, self.size(dim))`

**YAML Definition** (`native_functions.yaml:9564-9573`):
```yaml
- func: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
  variants: method, function
  structured_delegate: gather.out
  tags: core
```

**CPU Implementation** (`native/cpu/IndexKernel.cpp`):
```cpp
void gather_kernel(const Tensor& result, const Tensor& self,
                   int64_t dim, const Tensor& index) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "gather_cpu", [&] {
    cpu_gather_kernel<scalar_t>(result, self, dim, index);
  });
}

template <typename scalar_t>
void cpu_gather_kernel(const Tensor& result, const Tensor& self,
                       int64_t dim, const Tensor& index) {
  auto self_data = self.data_ptr<scalar_t>();
  auto result_data = result.data_ptr<scalar_t>();
  auto index_data = index.data_ptr<int64_t>();

  // Iterate over index tensor
  for (int64_t idx = 0; idx < index.numel(); ++idx) {
    // Compute position in result
    auto result_coords = unravel_index(idx, result.sizes());

    // Replace coordinate at dim with index value
    auto self_coords = result_coords;
    self_coords[dim] = index_data[idx];

    // Gather value
    result_data[idx] = self_data[ravel_index(self_coords, self.strides())];
  }
}
```

**MPS Implementation** (`mps/operations/ScatterGather.mm`):
```objective-c
Tensor& gather_out_mps(const Tensor& self, int64_t dim, const Tensor& index,
                       bool sparse_grad, Tensor& result) {
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSGraph* graph = make_mps_graph();

    auto selfTensor = mpsGraphRankedPlaceHolder(graph, self);
    auto indexTensor = mpsGraphRankedPlaceHolder(graph, index);

    // MPSGraph gather operation
    MPSGraphTensor* outputTensor = [graph gatherWithUpdatesTensor:selfTensor
                                                    indicesTensor:indexTensor
                                                             axis:dim
                                                   batchDimensions:0
                                                             name:nil];

    runMPSGraph(stream, graph,
                @{selfTensor: getMPSData(self),
                  indexTensor: getMPSData(index)},
                outputTensor, result);
  }
  return result;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def gather(input, dim, index):
    """Gather operation along dimension"""
    return mx.take_along_axis(input, index, axis=dim)
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
  self: gather_backward(grad, self, dim, index, sparse_grad)
```

**Gradient Implementation**:
Scatter gradient back to source positions using same indices.

**Usage Examples**:
```python
# Example 1: Simple 1D gather
input = torch.tensor([10, 20, 30, 40, 50])
index = torch.tensor([0, 2, 4])
torch.gather(input, 0, index)  # tensor([10, 30, 50])

# Example 2: 2D gather along dim=1 (columns)
input = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
index = torch.tensor([[0, 2],
                      [1, 0]])
torch.gather(input, dim=1, index=index)
# tensor([[1, 3],
#         [5, 4]])
# Explanation:
# out[0][0] = input[0][index[0][0]] = input[0][0] = 1
# out[0][1] = input[0][index[0][1]] = input[0][2] = 3
# out[1][0] = input[1][index[1][0]] = input[1][1] = 5
# out[1][1] = input[1][index[1][1]] = input[1][0] = 4

# Example 3: Attention score gathering
# Useful for selecting top-k attention scores
scores = torch.randn(32, 128)  # (batch, seq_len)
top_k = 10
values, indices = torch.topk(scores, top_k, dim=1)
# indices shape: (32, 10)
# Can gather other tensors using same indices
```

**Common Use Cases**:
- **Attention mechanisms**: Gather values based on attention weights
- **Embedding lookup**: Gather embeddings for token indices
- **Top-k selection**: Extract top-k elements

---

### scatter (Scatter Values to Indices)

**Purpose**: Write values to specific indices (inverse of gather)

**Signature**:
```python
scatter(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
scatter(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
scatter(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
```

**Behavior**:
```python
# Tensor variant
out = self.clone()
out[i][j][k] = src[i][j][k]  # at position index[i][j][k]

# Scalar variant
out = self.clone()
out[i][j][k] = value  # at position index[i][j][k]
```

**Reduce Modes**:
- `None` (default): Overwrite
- `"add"`: Add to existing value
- `"multiply"`: Multiply with existing value

**YAML Definition** (`native_functions.yaml:8459-8489`):
```yaml
- func: scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
  structured_delegate: scatter.src_out
  variants: function, method
  tags: core

- func: scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
  structured_delegate: scatter.value_out
  variants: function, method
  tags: core

- func: scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
  structured_delegate: scatter.reduce_out
  variants: function, method
```

**CPU Implementation**:
```cpp
void scatter_kernel(Tensor& self, int64_t dim, const Tensor& index,
                    const Tensor& src) {
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "scatter_cpu", [&] {
    auto self_data = self.data_ptr<scalar_t>();
    auto src_data = src.data_ptr<scalar_t>();
    auto index_data = index.data_ptr<int64_t>();

    for (int64_t idx = 0; idx < index.numel(); ++idx) {
      auto index_coords = unravel_index(idx, index.sizes());

      // Get target coordinate
      auto self_coords = index_coords;
      self_coords[dim] = index_data[idx];

      // Scatter value
      int64_t self_offset = ravel_index(self_coords, self.strides());
      self_data[self_offset] = src_data[idx];
    }
  });
}
```

**MLX Equivalent**:
```python
def scatter(input, dim, index, src):
    """Scatter operation"""
    # MLX doesn't have direct scatter, implement via advanced indexing
    # Or use scatter_add for additive scatter
    result = input.copy()
    # Would need custom implementation
    return result

def scatter_add(input, dim, index, src):
    """Additive scatter"""
    return mx.scatter_add(input, index, src, axis=dim)
```

**Gradient**:
```yaml
- name: scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
  self: grad
  src: gather(grad, dim, index)
```

**Usage Examples**:
```python
# Example 1: Basic scatter
input = torch.zeros(3, 5)
index = torch.tensor([[0, 1, 2, 0]])
src = torch.ones(1, 4)
output = input.scatter(1, index, src)
# output[0][0] = 1.0 (from src[0][0])
# output[0][1] = 1.0 (from src[0][1])
# output[0][2] = 1.0 (from src[0][2])

# Example 2: Scatter with scalar value
input = torch.zeros(3, 5)
index = torch.tensor([[0, 1, 2]])
output = input.scatter(1, index, 99.0)
# Sets input[0][0], input[0][1], input[0][2] to 99.0

# Example 3: One-hot encoding
labels = torch.tensor([0, 2, 1, 3])
num_classes = 4
one_hot = torch.zeros(4, num_classes)
one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
# tensor([[1., 0., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 0., 1.]])

# Example 4: Scatter with reduction (accumulate)
input = torch.ones(3, 5)
index = torch.tensor([[0, 0, 1]])  # Duplicate index 0
src = torch.tensor([[1.0, 2.0, 3.0]])
output = input.scatter(1, index, src, reduce='add')
# output[0][0] = 1.0 + 1.0 + 2.0 = 4.0 (accumulated)
# output[0][1] = 1.0 + 3.0 = 4.0
```

**Common Use Cases**:
- **One-hot encoding**: Create one-hot vectors
- **Histogram**: Accumulate counts at indices
- **Gradient accumulation**: Sum gradients for duplicate indices

---

### index_select (Select by Index)

**Purpose**: Select entire slices along a dimension

**Signature**:
```python
index_select(Tensor self, int dim, Tensor index) -> Tensor
```

**Behavior**: Equivalent to `self[:, :, index, :]` (for dim=2)

**Difference from gather**:
- `gather`: Can select different indices for each position
- `index_select`: Selects entire slices uniformly

**YAML Definition** (`native_functions.yaml:9486-9497`):
```yaml
- func: index_select(Tensor self, int dim, Tensor index) -> Tensor
  variants: method, function
  dispatch:
    CPU: index_select_cpu_
    QuantizedCPU: index_select_quantized_cpu_
    CUDA: index_select_cuda
    QuantizedCUDA: index_select_quantized_cuda
    SparseCPU: index_select_sparse_cpu
    SparseCUDA: index_select_sparse_cuda
    SparseMPS: index_select_sparse_mps
    MPS: index_select_mps
  tags: core
```

**CPU Implementation**:
```cpp
Tensor index_select_cpu(const Tensor& self, int64_t dim, const Tensor& index) {
  TORCH_CHECK(index.dim() == 1, "index_select(): index must be 1D");
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "dim out of range");

  auto result_shape = self.sizes().vec();
  result_shape[dim] = index.numel();

  Tensor result = at::empty(result_shape, self.options());

  auto index_data = index.data_ptr<int64_t>();

  // Copy slices
  for (int64_t i = 0; i < index.numel(); ++i) {
    int64_t idx = index_data[i];
    result.select(dim, i).copy_(self.select(dim, idx));
  }

  return result;
}
```

**MLX Equivalent**:
```python
def index_select(input, dim, index):
    """Index select operation"""
    return mx.take(input, index, axis=dim)
```

**Gradient**:
```yaml
- name: index_select(Tensor self, int dim, Tensor index) -> Tensor
  self: index_select_backward(grad, self.sizes(), dim, index)
```

**Usage Examples**:
```python
# Example 1: Select specific rows
input = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
index = torch.tensor([0, 2])
torch.index_select(input, dim=0, index=index)
# tensor([[1, 2, 3],
#         [7, 8, 9]])

# Example 2: Select specific columns
index = torch.tensor([0, 2])
torch.index_select(input, dim=1, index=index)
# tensor([[1, 3],
#         [4, 6],
#         [7, 9]])

# Example 3: Reorder dimensions
index = torch.tensor([2, 1, 0])
torch.index_select(input, dim=0, index=index)
# Reverses row order

# Example 4: Duplicate selection
index = torch.tensor([0, 0, 1])
torch.index_select(input, dim=0, index=index)
# tensor([[1, 2, 3],
#         [1, 2, 3],  # Duplicate
#         [4, 5, 6]])
```

**Performance**: More efficient than gather for selecting entire slices

**Common Use Cases**:
- **Embedding lookup**: `embedding.weight.index_select(0, indices)`
- **Batch selection**: Select specific samples from batch
- **Dimension reordering**: Permute or duplicate slices

---

### masked_fill (Conditional Fill)

**Purpose**: Fill tensor elements where mask is True

**Signature**:
```python
masked_fill(Tensor self, Tensor mask, Scalar value) -> Tensor
masked_fill(Tensor self, Tensor mask, Tensor value) -> Tensor
```

**Behavior**:
```python
out = self.clone()
out[mask] = value  # Where mask is True
```

**YAML Definition** (`native_functions.yaml:8287-8311`):
```yaml
- func: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: masked_fill
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_masked_fill
  tags: pointwise

- func: masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: masked_fill
```

**CPU Implementation**:
```cpp
Tensor masked_fill_cpu(const Tensor& self, const Tensor& mask, const Scalar& value) {
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_fill: mask must be bool");

  Tensor result = self.clone();

  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_input(result)
      .add_input(mask)
      .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "masked_fill_cpu", [&] {
    scalar_t fill_value = value.to<scalar_t>();
    cpu_kernel(iter, [fill_value](scalar_t a, bool mask_val) -> scalar_t {
      return mask_val ? fill_value : a;
    });
  });

  return result;
}
```

**MLX Equivalent**:
```python
def masked_fill(input, mask, value):
    """Masked fill operation"""
    return mx.where(mask, value, input)
```

**Gradient**:
```yaml
- name: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
  self: grad.masked_fill(mask, 0)  # Zero gradient where filled
```

**Usage Examples**:
```python
# Example 1: Fill negative values with zero
input = torch.tensor([1.0, -2.0, 3.0, -4.0])
mask = input < 0
input.masked_fill(mask, 0.0)
# tensor([1.0, 0.0, 3.0, 0.0])

# Example 2: Attention mask (set padding to -inf)
# Typical in transformers for masking padding tokens
attention_scores = torch.randn(32, 10, 10)  # (batch, seq, seq)
padding_mask = torch.zeros(32, 10, 10, dtype=torch.bool)
padding_mask[:, :, 5:] = True  # Mask positions 5-9
masked_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
# After softmax, -inf positions become 0

# Example 3: Causal masking (autoregressive)
seq_len = 5
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# tensor([[False,  True,  True,  True,  True],
#         [False, False,  True,  True,  True],
#         [False, False, False,  True,  True],
#         [False, False, False, False,  True],
#         [False, False, False, False, False]])
scores = torch.randn(seq_len, seq_len)
scores.masked_fill(causal_mask, float('-inf'))
# Prevents attending to future positions

# Example 4: NaN masking
input = torch.tensor([1.0, float('nan'), 3.0, float('nan')])
mask = torch.isnan(input)
input.masked_fill(mask, 0.0)
# tensor([1.0, 0.0, 3.0, 0.0])
```

**Common Use Cases**:
- **Attention masking**: Prevent attention to padding/future tokens
- **Invalid value replacement**: Replace NaN, inf, or out-of-range values
- **Conditional initialization**: Fill specific positions

---

### where (Conditional Selection)

**Purpose**: Element-wise selection based on condition

**Signature**:
```python
where(Tensor condition, Tensor self, Tensor other) -> Tensor
where(Tensor condition) -> Tuple[Tensor]  # Returns indices where True
```

**Behavior**:
```python
# Ternary variant
out[i] = self[i] if condition[i] else other[i]

# Index variant
out = nonzero(condition)  # Indices where condition is True
```

**YAML Definition** (`native_functions.yaml:6713-6739`):
```yaml
- func: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
  device_check: NoCheck
  variants: function, method
  dispatch:
    CPU, CUDA, MPS, MTIA: where
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_where
  tags: [core, pointwise]

- func: where(Tensor condition) -> Tensor[]
  device_check: NoCheck
  variants: function
```

**CPU Implementation**:
```cpp
Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(condition.dtype() == ScalarType::Bool, "where: condition must be bool");

  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_input(condition)
      .add_input(self)
      .add_input(other)
      .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "where_cpu", [&] {
    cpu_kernel(iter, [](bool cond, scalar_t a, scalar_t b) -> scalar_t {
      return cond ? a : b;
    });
  });

  return result;
}
```

**MLX Equivalent**:
```python
def where(condition, x=None, y=None):
    """Where operation"""
    if x is None and y is None:
        # Index variant: return indices where True
        return mx.nonzero(condition)
    else:
        # Ternary variant
        return mx.where(condition, x, y)
```

**Gradient**:
```yaml
- name: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
  self: grad.masked_fill(~condition, 0)
  other: grad.masked_fill(condition, 0)
```

**Usage Examples**:
```python
# Example 1: Ternary selection
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([10.0, 20.0, 30.0, 40.0])
condition = x > 2.5
torch.where(condition, x, y)
# tensor([10.0, 20.0, 3.0, 4.0])
# Explanation: x > 2.5 is [False, False, True, True]
# So output = [y[0], y[1], x[2], x[3]]

# Example 2: Clamp using where
x = torch.tensor([-1.0, 0.5, 2.0, 3.5])
# Clamp to [0, 3]
torch.where(x < 0, torch.zeros_like(x),
            torch.where(x > 3, 3 * torch.ones_like(x), x))
# tensor([0.0, 0.5, 2.0, 3.0])

# Example 3: Replace NaN
x = torch.tensor([1.0, float('nan'), 3.0])
torch.where(torch.isnan(x), torch.zeros_like(x), x)
# tensor([1.0, 0.0, 3.0])

# Example 4: Index variant (find indices where True)
x = torch.tensor([1, 5, 3, 8, 2])
indices = torch.where(x > 3)
# (tensor([1, 3]),)  # Indices where x > 3

# Example 5: Multi-dimensional
x = torch.tensor([[1, 2], [3, 4]])
indices = torch.where(x > 2)
# (tensor([1, 1]), tensor([0, 1]))
# Position (1, 0) has value 3
# Position (1, 1) has value 4
```

**Broadcasting**:
```python
# condition, x, and y can have different shapes (broadcast)
condition = torch.tensor([True, False, True])
x = torch.tensor([[1, 2, 3]])  # (1, 3)
y = torch.tensor([[10], [20], [30]])  # (3, 1)
torch.where(condition, x, y)
# Broadcasts to (3, 3)
```

**Common Use Cases**:
- **Conditional computation**: Select values based on conditions
- **NaN/inf handling**: Replace invalid values
- **Piecewise functions**: Implement different formulas for different regions
- **Masking**: Efficient alternative to masked_fill for two alternatives

---

## Advanced Indexing Patterns

### Combining Operations

**Attention Mechanism Pattern**:
```python
# 1. Compute attention scores
scores = query @ key.transpose(-2, -1)  # (batch, heads, seq, seq)

# 2. Apply causal mask
mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))

# 3. Softmax
attn_weights = F.softmax(scores, dim=-1)

# 4. Gather values
output = attn_weights @ value  # (batch, heads, seq, dim)
```

**Top-k Selection**:
```python
# Find top-k values and gather related tensors
values, indices = torch.topk(scores, k=10, dim=1)  # (batch, 10)
selected_embeddings = torch.gather(embeddings, 1,
                                   indices.unsqueeze(-1).expand(-1, -1, embed_dim))
```

### Performance Considerations

**Gather vs Index Select**:
```python
# index_select is more efficient for selecting entire slices
indices = torch.tensor([0, 2, 5])

# Faster (entire rows)
result = input.index_select(0, indices)

# Slower (element-wise, but more flexible)
indices_expanded = indices.view(-1, 1).expand(-1, input.size(1))
result = input.gather(0, indices_expanded)
```

**Scatter Performance**:
```python
# Avoid duplicate indices when possible (causes race conditions on GPU)
# OK: Unique indices
indices = torch.tensor([0, 1, 2])
output = input.scatter(0, indices, src)

# Warning: Duplicate indices (non-deterministic on GPU)
indices = torch.tensor([0, 0, 1])  # Index 0 appears twice
output = input.scatter(0, indices, src)  # Use scatter_add for determinism
```

---

## Implementation Files

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:9564-9584` (gather)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:8459-8520` (scatter)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:9486-9502` (index_select)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:8276-8311` (masked_fill)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:6713-6739` (where)

**CPU Kernels**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/IndexKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/IndexingUtils.h`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/Indexing.cpp`

**MPS Kernels** (Metal):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/ScatterGather.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Indexing.mm`

**Gradients**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/tools/autograd/derivatives.yaml`

---

## MLX Porting Summary

**Direct Mappings**:
```python
# PyTorch → MLX
torch.gather        → mx.take_along_axis
torch.scatter       → Custom implementation (MLX scatter_add available)
torch.index_select  → mx.take
torch.masked_fill   → mx.where(mask, value, input)
torch.where         → mx.where (ternary) / mx.nonzero (indices)
```

**Considerations**:

1. **Index Types**: PyTorch uses int64, MLX typically uses int32 (may need conversion)

2. **Scatter Implementation**: MLX has `scatter_add` but not general scatter with all reduce modes

3. **Broadcasting**: Same semantics as PyTorch

4. **Performance**: Metal backend provides efficient implementations

**Example Compatibility Layer**:
```python
import mlx.core as mx

class torch_indexing:
    @staticmethod
    def gather(input, dim, index):
        return mx.take_along_axis(input, index, axis=dim)

    @staticmethod
    def index_select(input, dim, index):
        return mx.take(input, index, axis=dim)

    @staticmethod
    def masked_fill(input, mask, value):
        return mx.where(mask, value, input)

    @staticmethod
    def where(condition, x=None, y=None):
        if x is None:
            return mx.nonzero(condition)
        return mx.where(condition, x, y)

    @staticmethod
    def scatter_add(input, dim, index, src):
        return mx.scatter_add(input, index, src, axis=dim)
```

**Custom Scatter Implementation** (if needed):
```python
def scatter(input, dim, index, src, reduce=None):
    """Custom scatter for MLX"""
    result = input.copy()

    if reduce == 'add':
        return mx.scatter_add(result, index, src, axis=dim)
    elif reduce is None:
        # Overwrite mode - requires custom implementation
        # Can use advanced indexing or loops
        pass
    else:
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    return result
```

Indexing operators are essential for flexible data access patterns in deep learning and have mostly direct mappings to MLX primitives, with some operations requiring custom implementations.
