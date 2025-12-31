# PyTorch to MLX Implementation Roadmap

## Purpose

This document provides a **comprehensive, actionable roadmap** for porting PyTorch functionality to MLX (Apple's Machine Learning framework for Apple Silicon). Unlike theoretical comparisons, this roadmap gives concrete:

1. **Phased implementation plan** with clear milestones
2. **Dependency-ordered task lists** to avoid circular dependencies
3. **Tier-based operator prioritization** focused on real-world models
4. **Testing strategies** to validate each phase
5. **Resource estimates** for effort planning

## Porting Philosophy

### Bottom-Up vs Top-Down Approaches

**Bottom-Up (Recommended)**:
```
c10/TensorImpl → ATen Operators → Autograd → nn.Module → Training
```
- Validates each layer before building on top
- Easier to test incrementally
- Matches MLX's existing architecture

**Top-Down (Risky)**:
```
Training API → nn.Module → Autograd → Operators → Tensor Core
```
- Higher-level features depend on unstable foundation
- Harder to isolate failures
- More refactoring when lower layers change

**This roadmap follows bottom-up approach.**

## Phase Overview

```
┌────────────────────────────────────────────────────────────┐
│ Phase 1: Tensor Core (Weeks 1-2)                          │
│ Foundation: Array abstraction, memory model, dtypes       │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 2: Essential Operators (Weeks 3-4)                  │
│ Core ops: 50 tier-1 operators for basic neural networks   │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 3: Autograd System (Weeks 5-6)                      │
│ Differentiation: Backward pass, gradient accumulation     │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 4: Neural Network Layers (Weeks 7-8)                │
│ nn.Module: Layer implementations, parameter management    │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 5: Training Infrastructure (Weeks 9-10)             │
│ Optimizers: SGD, Adam, learning rate scheduling           │
└────────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│ Phase 6: Model Zoo & Validation (Weeks 11-12)             │
│ Reference models: ResNet, Transformer, validation suite   │
└────────────────────────────────────────────────────────────┘
```

## Phase 1: Tensor Core (Weeks 1-2)

### Goal
Establish PyTorch-compatible tensor abstraction on top of MLX arrays.

### Deliverables

#### 1.1 Tensor Wrapper Class
**Status**: MLX already has `mlx.core.array` - need PyTorch-like interface

**Implementation**:
```python
# File: mlx_compat/tensor.py

import mlx.core as mx

class Tensor:
    """PyTorch-compatible Tensor wrapping MLX array."""

    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        # MLX arrays are always on unified memory
        # 'device' parameter ignored (CPU/GPU transparent)
        if isinstance(data, mx.array):
            self._array = data
        else:
            self._array = mx.array(data, dtype=dtype)

        self.requires_grad = requires_grad
        self.grad = None  # Gradient tensor
        self.grad_fn = None  # Backward function (Phase 3)

    # Shape properties (PyTorch compatibility)
    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    # Device (always returns 'mps' for compatibility)
    @property
    def device(self):
        return 'mps'  # Or 'cpu' - doesn't matter with unified memory

    # Item access
    def item(self):
        """Get Python scalar (single element tensors only)."""
        if self._array.size != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self._array.item()

    # Indexing
    def __getitem__(self, key):
        return Tensor(self._array[key], requires_grad=self.requires_grad)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value._array
        self._array[key] = value

    # Arithmetic (implement in Phase 2)
    def __add__(self, other):
        raise NotImplementedError("Implement in Phase 2")

    # String representation
    def __repr__(self):
        return f"Tensor({self._array})"
```

**Test Plan**:
```python
def test_tensor_creation():
    # From list
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)

    # From MLX array
    a = mx.array([1.0, 2.0, 3.0])
    t = Tensor(a)
    assert t.shape == (3,)

    # With dtype
    t = Tensor([1, 2, 3], dtype=mx.float32)
    assert t.dtype == mx.float32

    # requires_grad
    t = Tensor([1.0], requires_grad=True)
    assert t.requires_grad == True
```

#### 1.2 Storage and View Semantics
**Status**: MLX already supports views - need PyTorch API compatibility

**Key Operations**:
- `view(shape)`: Reshape without copying
- `reshape(shape)`: Reshape (may copy if necessary)
- `transpose(dim0, dim1)`: Swap dimensions
- `permute(*dims)`: Arbitrary dimension reordering
- `squeeze()` / `unsqueeze(dim)`: Remove/add dimensions

**Implementation Pattern**:
```python
class Tensor:
    def view(self, *shape):
        """Reshape tensor (must be contiguous)."""
        reshaped = mx.reshape(self._array, shape)
        result = Tensor(reshaped, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = ViewBackward(self, shape)  # Phase 3
        return result

    def reshape(self, *shape):
        """Reshape tensor (may copy)."""
        reshaped = mx.reshape(self._array, shape)
        return Tensor(reshaped, requires_grad=self.requires_grad)

    def transpose(self, dim0, dim1):
        """Swap two dimensions."""
        # Convert to list of dims
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)

    def permute(self, *dims):
        """Permute dimensions."""
        permuted = mx.transpose(self._array, dims)
        return Tensor(permuted, requires_grad=self.requires_grad)
```

**Test Plan**:
```python
def test_view_semantics():
    t = Tensor([[1, 2], [3, 4]])

    # View
    v = t.view(4)
    assert v.shape == (4,)

    # Transpose
    t_t = t.transpose(0, 1)
    assert t_t.shape == (2, 2)
    assert t_t[0, 1].item() == 2

    # Permute
    t = Tensor([[[1, 2], [3, 4]]])  # Shape: (1, 2, 2)
    p = t.permute(2, 0, 1)  # Shape: (2, 1, 2)
    assert p.shape == (2, 1, 2)
```

#### 1.3 Type System
**Status**: Map PyTorch dtypes to MLX dtypes

**Mapping Table**:
| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.float32` | `mx.float32` | Default floating point |
| `torch.float16` | `mx.float16` | Half precision |
| `torch.bfloat16` | `mx.bfloat16` | Brain float (if supported) |
| `torch.int64` | `mx.int64` | Default integer |
| `torch.int32` | `mx.int32` | 32-bit integer |
| `torch.int16` | `mx.int16` | 16-bit integer |
| `torch.int8` | `mx.int8` | 8-bit integer |
| `torch.uint8` | `mx.uint8` | Unsigned 8-bit |
| `torch.bool` | `mx.bool_` | Boolean |
| `torch.complex64` | `mx.complex64` | Complex float32 |

**Implementation**:
```python
# File: mlx_compat/dtypes.py

import mlx.core as mx

# PyTorch dtype constants
float32 = mx.float32
float16 = mx.float16
bfloat16 = mx.bfloat16
int64 = mx.int64
int32 = mx.int32
int16 = mx.int16
int8 = mx.int8
uint8 = mx.uint8
bool = mx.bool_
complex64 = mx.complex64

# Type promotion rules (match PyTorch)
def promote_types(type1, type2):
    """Promote two types to common type."""
    # Simplified promotion table
    promotion_table = {
        (mx.int32, mx.float32): mx.float32,
        (mx.int64, mx.float32): mx.float32,
        (mx.float16, mx.float32): mx.float32,
        # ... full table ...
    }
    return promotion_table.get((type1, type2), type1)
```

### Success Criteria (Phase 1)
- [ ] Tensor creation from Python lists, NumPy arrays, MLX arrays
- [ ] Basic indexing and slicing
- [ ] View operations (view, reshape, transpose, permute)
- [ ] Type conversions
- [ ] Shape manipulation (squeeze, unsqueeze)
- [ ] 100% pass rate on tensor core unit tests

## Phase 2: Essential Operators (Weeks 3-4)

### Goal
Implement **Tier 1 operators** - the 50 critical operations needed for basic neural networks.

### Tier 1 Operator List

#### Arithmetic (12 operators)
Priority: CRITICAL
```python
# Binary arithmetic
add(a, b)        # Element-wise addition
sub(a, b)        # Element-wise subtraction
mul(a, b)        # Element-wise multiplication
div(a, b)        # Element-wise division

# Matrix operations
matmul(a, b)     # Matrix multiplication
mm(a, b)         # Matrix-matrix product
bmm(a, b)        # Batch matrix-matrix product
addmm(a, b, c)   # a + b @ c

# Scalar operations
add(a, scalar)   # Scalar addition
mul(a, scalar)   # Scalar multiplication
div(a, scalar)   # Scalar division
pow(a, exp)      # Element-wise power
```

**Implementation Example**:
```python
# File: mlx_compat/ops/arithmetic.py

def add(input, other, *, alpha=1):
    """Add two tensors element-wise."""
    if isinstance(other, (int, float)):
        # Scalar addition
        result_array = input._array + other
    elif isinstance(other, Tensor):
        # Tensor addition
        if alpha != 1:
            other_array = other._array * alpha
        else:
            other_array = other._array
        result_array = input._array + other_array
    else:
        raise TypeError(f"Cannot add Tensor and {type(other)}")

    result = Tensor(result_array)
    if input.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
        result.requires_grad = True
        result.grad_fn = AddBackward(input, other, alpha)  # Phase 3
    return result

# Register as method
Tensor.__add__ = lambda self, other: add(self, other)
Tensor.add = add
```

#### Activations (7 operators)
Priority: CRITICAL
```python
relu(x)          # ReLU: max(0, x)
gelu(x)          # GELU: Gaussian Error Linear Unit
sigmoid(x)       # Sigmoid: 1 / (1 + exp(-x))
tanh(x)          # Hyperbolic tangent
softmax(x, dim)  # Softmax normalization
log_softmax(x, dim)  # Log-softmax (numerically stable)
silu(x)          # SiLU/Swish: x * sigmoid(x)
```

#### Reductions (8 operators)
Priority: CRITICAL
```python
sum(x, dim=None, keepdim=False)   # Sum across dimensions
mean(x, dim=None, keepdim=False)  # Mean across dimensions
max(x, dim=None, keepdim=False)   # Max across dimensions
min(x, dim=None, keepdim=False)   # Min across dimensions
argmax(x, dim)                     # Index of max element
argmin(x, dim)                     # Index of min element
var(x, dim, unbiased=True)        # Variance
std(x, dim, unbiased=True)        # Standard deviation
```

#### Indexing & Gathering (5 operators)
Priority: HIGH
```python
gather(input, dim, index)     # Gather values along dimension
scatter(input, dim, index, src)  # Scatter values along dimension
index_select(input, dim, index)  # Select indices along dimension
where(condition, x, y)        # Element-wise selection
masked_fill(input, mask, value)  # Fill masked elements
```

#### Shape Manipulation (8 operators)
Priority: HIGH
```python
cat(tensors, dim)            # Concatenate tensors
stack(tensors, dim)          # Stack tensors (new dimension)
split(tensor, size, dim)     # Split tensor into chunks
chunk(tensor, chunks, dim)   # Split into equal chunks
expand(tensor, *sizes)       # Broadcast to new shape
repeat(tensor, *sizes)       # Repeat tensor elements
flatten(tensor, start, end)  # Flatten dimensions
unflatten(tensor, dim, sizes)  # Unflatten dimension
```

#### Convolution (3 operators)
Priority: CRITICAL
```python
conv1d(input, weight, bias, stride, padding, dilation, groups)
conv2d(input, weight, bias, stride, padding, dilation, groups)
conv3d(input, weight, bias, stride, padding, dilation, groups)
```

**Note**: Convolution is complex - may use MLX's native conv ops directly.

#### Other Essential (7 operators)
Priority: HIGH
```python
embedding(input, weight)     # Embedding lookup
linear(input, weight, bias)  # Fully connected layer
dropout(input, p, training)  # Dropout regularization
layer_norm(input, normalized_shape, weight, bias, eps)
batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
max_pool2d(input, kernel_size, stride, padding)
adaptive_avg_pool2d(input, output_size)
```

### Implementation Strategy

#### Step 1: Operator Skeleton
```python
# File: mlx_compat/ops/relu.py

def relu(input, inplace=False):
    """Apply ReLU activation."""
    if inplace:
        # MLX doesn't support true inplace, simulate
        input._array = mx.maximum(input._array, 0)
        return input
    else:
        result_array = mx.maximum(input._array, 0)
        result = Tensor(result_array)
        if input.requires_grad:
            result.requires_grad = True
            result.grad_fn = ReluBackward(input)  # Phase 3
        return result
```

#### Step 2: Metal Kernel (if needed)
For operations not in MLX:
```python
# Custom Metal kernel for unsupported ops
def custom_op(input):
    # Compile Metal shader
    kernel_source = """
    kernel void custom_kernel(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    {
        output[id] = custom_function(input[id]);
    }
    """
    # Use MLX's Metal compilation interface
    # ...
```

#### Step 3: Unit Tests
```python
def test_add():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = add(a, b)
    assert c.shape == (3,)
    assert c[0].item() == 5.0
    assert c[1].item() == 7.0
    assert c[2].item() == 9.0

def test_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = matmul(a, b)
    assert c.shape == (2, 2)
    # a @ b = [[19, 22], [43, 50]]
    assert c[0, 0].item() == 19.0
    assert c[1, 1].item() == 50.0
```

### Success Criteria (Phase 2)
- [ ] 50 Tier 1 operators implemented
- [ ] MLX equivalents identified for all ops
- [ ] Unit tests for each operator
- [ ] Benchmarks vs PyTorch (within 20% performance)
- [ ] Can run simple MLP forward pass

## Phase 3: Autograd System (Weeks 5-6)

### Goal
Implement automatic differentiation compatible with PyTorch's tape-based system.

### Deliverables

#### 3.1 Computational Graph
**Status**: MLX uses function transforms - need tape-based graph for PyTorch compatibility

**Implementation**:
```python
# File: mlx_compat/autograd/function.py

class Function:
    """Base class for differentiable functions."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

class AddBackward(Function):
    @staticmethod
    def forward(ctx, a, b, alpha=1):
        ctx.save_for_backward(a, b)
        ctx.alpha = alpha
        return a._array + alpha * b._array

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output if a.requires_grad else None
        grad_b = grad_output * ctx.alpha if b.requires_grad else None
        return grad_a, grad_b
```

**Backward Pass Example**:
```python
# Forward pass builds graph
a = Tensor([1.0, 2.0], requires_grad=True)
b = Tensor([3.0, 4.0], requires_grad=True)
c = a + b  # c.grad_fn = AddBackward
d = c * 2  # d.grad_fn = MulBackward
loss = d.sum()  # loss.grad_fn = SumBackward

# Backward pass traverses graph
loss.backward()

# Gradients accumulated
assert a.grad is not None
assert b.grad is not None
```

#### 3.2 Gradient Formulas
Implement backward methods for all Tier 1 operators.

**Examples**:
```python
# ReLU backward
class ReluBackward(Function):
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (input._array > 0)
        return Tensor(grad_input)

# MatMul backward
class MatMulBackward(Function):
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.transpose(-1, -2) if a.requires_grad else None
        grad_b = a.transpose(-1, -2) @ grad_output if b.requires_grad else None
        return grad_a, grad_b
```

#### 3.3 backward() Method
```python
class Tensor:
    def backward(self, gradient=None, retain_graph=False):
        """Compute gradients via backpropagation."""
        if not self.requires_grad:
            raise RuntimeError("Calling backward on tensor that doesn't require grad")

        if gradient is None:
            # Scalar tensor (loss)
            if self._array.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = Tensor(mx.ones_like(self._array))

        # Topological sort of computation graph
        topo_order = self._topological_sort()

        # Backward pass
        self.grad = gradient
        for node in reversed(topo_order):
            if node.grad_fn is not None:
                grads = node.grad_fn.backward(node.grad)
                # Accumulate gradients for inputs
                for inp, grad in zip(node.grad_fn.inputs, grads):
                    if inp.requires_grad:
                        if inp.grad is None:
                            inp.grad = grad
                        else:
                            inp.grad = inp.grad + grad  # Accumulate

        if not retain_graph:
            # Clear graph to save memory
            self._clear_graph()
```

### Success Criteria (Phase 3)
- [ ] Backward methods for all Tier 1 operators
- [ ] Gradient accumulation
- [ ] Gradient checking (numerical vs analytical)
- [ ] Can train simple MLP end-to-end

## Phase 4: Neural Network Layers (Weeks 7-8)

### Goal
Implement `nn.Module` and common layer types.

### Tier 1 Layers

#### 4.1 Core Layers (Week 7)
```python
nn.Linear(in_features, out_features, bias=True)
nn.Conv1d(in_channels, out_channels, kernel_size, ...)
nn.Conv2d(in_channels, out_channels, kernel_size, ...)
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.Dropout(p=0.5)
```

#### 4.2 Activation Modules (Week 7)
```python
nn.ReLU(inplace=False)
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=-1)
nn.LogSoftmax(dim=-1)
```

#### 4.3 Pooling & Containers (Week 8)
```python
nn.MaxPool2d(kernel_size, stride, padding)
nn.AvgPool2d(kernel_size, stride, padding)
nn.AdaptiveAvgPool2d(output_size)
nn.Sequential(*layers)
nn.ModuleList(modules)
nn.ModuleDict(modules)
```

**Implementation Reference**: See [04-NEURAL-NETWORKS/module-system.md](../04-NEURAL-NETWORKS/module-system.md)

### Success Criteria (Phase 4)
- [ ] nn.Module base class
- [ ] 15+ layer types implemented
- [ ] Can define ResNet-18 architecture
- [ ] Can run forward pass on CIFAR-10

## Phase 5: Training Infrastructure (Weeks 9-10)

### Goal
Implement optimizers, loss functions, and training utilities.

### Deliverables

#### 5.1 Optimizers (Week 9)
```python
optim.SGD(params, lr, momentum, weight_decay)
optim.Adam(params, lr, betas, eps, weight_decay)
optim.AdamW(params, lr, betas, eps, weight_decay)
```

**Implementation Reference**: See [05-TRAINING/optimizer-base.md](../05-TRAINING/optimizer-base.md)

#### 5.2 Loss Functions (Week 9)
```python
nn.CrossEntropyLoss(reduction='mean')
nn.MSELoss(reduction='mean')
nn.BCEWithLogitsLoss(reduction='mean')
nn.NLLLoss(reduction='mean')
```

#### 5.3 Learning Rate Schedulers (Week 10)
```python
optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience)
```

### Success Criteria (Phase 5)
- [ ] 3+ optimizers implemented
- [ ] 4+ loss functions implemented
- [ ] Can train ResNet-18 on CIFAR-10 to >90% accuracy
- [ ] Training convergence matches PyTorch

## Phase 6: Model Zoo & Validation (Weeks 11-12)

### Goal
Validate porting with reference models and comprehensive test suite.

### Reference Models

#### 6.1 Computer Vision (Week 11)
```python
# ResNet variants
models.resnet18(pretrained=False)
models.resnet50(pretrained=False)

# EfficientNet
models.efficientnet_b0(pretrained=False)

# ViT (Vision Transformer)
models.vit_b_16(pretrained=False)
```

#### 6.2 Natural Language Processing (Week 12)
```python
# GPT-2
models.gpt2(config)

# BERT
models.bert_base_uncased(config)

# Transformer blocks
nn.TransformerEncoder(...)
nn.MultiheadAttention(...)
```

### Validation Strategy

#### Numerical Accuracy Tests
```python
def test_resnet18_parity():
    """Verify MLX ResNet-18 matches PyTorch output."""
    torch_model = torchvision.models.resnet18()
    mlx_model = mlx_compat.models.resnet18()

    # Copy weights from PyTorch to MLX
    mlx_model.load_state_dict(torch_model.state_dict())

    # Forward pass
    input = torch.randn(1, 3, 224, 224)
    torch_output = torch_model(input)

    mlx_input = mlx_compat.from_torch(input)
    mlx_output = mlx_model(mlx_input)

    # Check outputs match within tolerance
    diff = torch.abs(torch_output - mlx_compat.to_torch(mlx_output))
    assert diff.max() < 1e-5  # Numerical tolerance
```

#### Performance Benchmarks
```python
def benchmark_training_speed():
    """Compare training throughput: MLX vs PyTorch."""
    model = models.resnet50()
    dataloader = create_cifar10_loader(batch_size=128)

    # Measure time per epoch
    start = time.time()
    for epoch in range(3):
        for batch in dataloader:
            loss = train_step(model, batch)
    mlx_time = time.time() - start

    # Compare with PyTorch baseline
    assert mlx_time < pytorch_baseline * 1.2  # Within 20%
```

### Success Criteria (Phase 6)
- [ ] 5+ reference models run successfully
- [ ] Numerical parity with PyTorch (<1e-5 error)
- [ ] Performance within 20% of PyTorch on Apple Silicon
- [ ] Can fine-tune pretrained models
- [ ] 95%+ test coverage

## MLX Capability Gaps

### Known Limitations to Address

#### 1. Sparse Tensors
**Status**: MLX doesn't support sparse formats (COO, CSR)
**Workaround**: Implement dense equivalents, warn user
**Priority**: Low (most models use dense tensors)

#### 2. Quantization
**Status**: MLX has limited quantization support
**Plan**: Implement INT8/INT4 quantization primitives
**Priority**: Medium (needed for deployment)

#### 3. Distributed Training
**Status**: MLX is single-GPU only
**Plan**: Not applicable for Apple Silicon (single unified GPU)
**Priority**: N/A

#### 4. Compilation (torch.compile)
**Status**: MLX uses lazy evaluation + graph fusion
**Plan**: Map torch.compile to MLX's graph compilation
**Priority**: High (major PyTorch 2.0 feature)

#### 5. Complex Numbers
**Status**: MLX supports complex64, limited ops
**Plan**: Implement missing complex operations
**Priority**: Medium (needed for FFT, signal processing)

## Testing Strategy

### Unit Tests (Per Phase)
```bash
# Run phase-specific tests
pytest tests/phase1_tensor_core/
pytest tests/phase2_operators/
pytest tests/phase3_autograd/
pytest tests/phase4_nn_modules/
pytest tests/phase5_training/
```

### Integration Tests (Cross-Phase)
```python
def test_end_to_end_mnist():
    """Full training loop on MNIST."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(5):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch['image'])
            loss = criterion(output, batch['label'])
            loss.backward()
            optimizer.step()

    # Validation
    accuracy = evaluate(model, test_loader)
    assert accuracy > 0.95
```

### Gradient Checking
```python
def gradient_check(op, *inputs, eps=1e-5):
    """Verify analytical gradients match numerical gradients."""
    # Analytical gradient (autograd)
    output = op(*inputs)
    output.backward()
    analytical_grads = [inp.grad for inp in inputs]

    # Numerical gradient (finite differences)
    numerical_grads = []
    for inp in inputs:
        grad = torch.zeros_like(inp)
        for i in range(inp.numel()):
            # +eps
            inp.data.view(-1)[i] += eps
            output_plus = op(*inputs)
            # -eps
            inp.data.view(-1)[i] -= 2 * eps
            output_minus = op(*inputs)
            # Gradient
            grad.view(-1)[i] = (output_plus - output_minus) / (2 * eps)
            # Reset
            inp.data.view(-1)[i] += eps
        numerical_grads.append(grad)

    # Compare
    for analytical, numerical in zip(analytical_grads, numerical_grads):
        diff = torch.abs(analytical - numerical).max()
        assert diff < 1e-4  # Tolerance
```

## Resource Estimates

### Development Effort

| Phase | Duration | Developers | Complexity |
|-------|----------|------------|------------|
| Phase 1: Tensor Core | 2 weeks | 1-2 | Medium |
| Phase 2: Operators | 2 weeks | 2-3 | High |
| Phase 3: Autograd | 2 weeks | 2 | High |
| Phase 4: NN Layers | 2 weeks | 2-3 | Medium |
| Phase 5: Training | 2 weeks | 1-2 | Medium |
| Phase 6: Validation | 2 weeks | 2 | Low |
| **Total** | **12 weeks** | **2-3** | - |

### Dependencies

**External**:
- MLX SDK (from Apple)
- Metal Performance Shaders
- Python 3.9+
- NumPy (for testing)

**Internal**:
- Phase 2 depends on Phase 1
- Phase 3 depends on Phase 2
- Phase 4 depends on Phase 3
- Phase 5 depends on Phase 4
- Phase 6 depends on all phases

### Risk Mitigation

#### Risk 1: MLX API Instability
**Likelihood**: Medium
**Impact**: High
**Mitigation**: Pin MLX version, abstract MLX calls behind interface

#### Risk 2: Performance Regressions
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**: Continuous benchmarking, profile bottlenecks

#### Risk 3: Gradient Bugs
**Likelihood**: High
**Impact**: Critical
**Mitigation**: Extensive gradient checking, numerical validation

## Critical File References

### PyTorch Source (For Reference)
- [aten/src/ATen/native/native_functions.yaml](../reference/pytorch/aten/src/ATen/native/native_functions.yaml) - All operators
- [torch/nn/modules/module.py](../reference/pytorch/torch/nn/modules/module.py) - nn.Module
- [torch/optim/optimizer.py](../reference/pytorch/torch/optim/optimizer.py) - Optimizer base
- [torch/autograd/function.py](../reference/pytorch/torch/autograd/function.py) - Autograd functions

### Related Documentation
- [01-FOUNDATIONS/tensor-core.md](../01-FOUNDATIONS/tensor-core.md) - Tensor internals
- [02-OPERATORS/operator-categories.md](../02-OPERATORS/operator-categories.md) - Operator tiers
- [03-AUTOGRAD/autograd-overview.md](../03-AUTOGRAD/autograd-overview.md) - Autograd architecture
- [04-NEURAL-NETWORKS/module-system.md](../04-NEURAL-NETWORKS/module-system.md) - nn.Module system
- [06-BACKENDS/metal-mps-backend.md](../06-BACKENDS/metal-mps-backend.md) - Metal reference

## Next Steps

1. **Immediate Actions**:
   - Set up MLX development environment
   - Create project structure (`mlx_compat/` package)
   - Implement Phase 1.1: Tensor Wrapper Class
   - Write first unit tests

2. **Week 1 Goals**:
   - Complete Phase 1 (Tensor Core)
   - Begin Phase 2 (Arithmetic operators)

3. **Month 1 Checkpoint**:
   - Phases 1-2 complete
   - Can run simple forward passes
   - Begin autograd implementation

4. **Month 3 Target**:
   - All phases complete
   - Can train ResNet-18 on CIFAR-10
   - Numerical parity with PyTorch

## Summary

This roadmap provides a structured, dependency-ordered plan for porting PyTorch to MLX. The bottom-up approach ensures each layer is validated before building on top, minimizing rework and integration issues.

**Key Principles**:
1. **Incremental**: Each phase delivers working functionality
2. **Testable**: Comprehensive unit and integration tests
3. **Pragmatic**: Focus on common use cases (Tier 1 operators first)
4. **Validated**: Numerical parity and performance benchmarks

**Success Metrics**:
- Can run 5+ popular model architectures
- Numerical error <1e-5 vs PyTorch
- Performance within 20% on Apple Silicon
- 95%+ test coverage

The porting effort is estimated at **12 weeks with 2-3 developers**, assuming familiarity with both PyTorch and MLX. Adjust timeline based on team size and experience.
