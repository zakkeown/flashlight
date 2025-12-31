# Activation Functions

## Purpose

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. This document covers Tier 1 activation operators essential for modern ML architectures.

**Tier 1 Activation Operators** (7 total):
- `relu` - Rectified Linear Unit
- `gelu` - Gaussian Error Linear Unit
- `sigmoid` - Logistic sigmoid
- `tanh` - Hyperbolic tangent
- `softmax` - Normalized exponential
- `log_softmax` - Log of softmax
- `silu` - Sigmoid Linear Unit (Swish)

## Common Properties

**Tags**: Most are `[core, pointwise]` or `[pointwise]`

**Differentiability**: All are differentiable (some have special cases)

**Element-wise**: Most operate independently on each element

**Exceptions**: `softmax`/`log_softmax` operate across a dimension

**Performance**: Generally memory-bound, benefit from vectorization

## Operator Details

### relu (Rectified Linear Unit)

**Purpose**: Most widely used activation, simple and effective

**Signature**:
```python
relu(Tensor self) -> Tensor
```

**Formula**: `out = max(0, self)`

**YAML Definition** (`native_functions.yaml:5181-5194`):
```yaml
- func: relu(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CPU, CUDA: relu
    MPS: relu_mps
    MTIA: relu_mtia
    MkldnnCPU: mkldnn_relu
    QuantizedCPU: relu_quantized_cpu
    QuantizedCUDA: relu_quantized_cuda
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_relu
    SparseCPU, SparseCUDA, SparseMPS: relu_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: relu_sparse_csr
  tags: [core, pointwise]
```

**CPU Implementation** (`native/cpu/ActivationKernel.cpp`):
```cpp
void relu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "relu_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::max(scalar_t(0), a); },
        [=](Vectorized<scalar_t> a) { return a.relu(); }  // SIMD
    );
  });
}
```

**MPS Implementation** (`mps/operations/Activation.mm`):
```objective-c
Tensor& relu_out_mps(const Tensor& self, Tensor& result) {
  @autoreleasepool {
    MPSStream* stream = getCurrentMPSStream();
    MPSGraph* graph = make_mps_graph();

    auto selfPlaceholder = mpsGraphRankedPlaceHolder(graph, self);

    // Use MPSGraph ReLU operation
    MPSGraphTensor* outputTensor = [graph reLUWithTensor:selfPlaceholder
                                                    name:nil];

    runMPSGraph(stream, graph, @{selfPlaceholder: getMPSData(self)},
                outputTensor, result);
  }
  return result;
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def relu(x):
    """ReLU activation"""
    return mx.maximum(x, 0)
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: relu(Tensor self) -> Tensor
  self: grad * (self > 0).to(grad.dtype())
```

**Gradient Formula**:
```
∂ReLU/∂x = {1 if x > 0, 0 if x ≤ 0}
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
torch.relu(x)  # [0.0, 0.0, 0.0, 1.0, 2.0]

# Inplace variant (memory efficient)
x.relu_()

# As module in nn
import torch.nn as nn
relu = nn.ReLU()
out = relu(x)
```

**Properties**:
- **Non-saturating**: Gradient doesn't vanish for positive inputs
- **Dead neurons**: Neurons can "die" if always negative (zero gradient)
- **Sparse activation**: Many zeros in output (good for efficiency)

**Variants**:
- **Leaky ReLU**: `max(alpha*x, x)` for small alpha (prevents dead neurons)
- **Parametric ReLU (PReLU)**: Learnable alpha per channel
- **ReLU6**: `min(max(0, x), 6)` - bounded for mobile quantization

---

### gelu (Gaussian Error Linear Unit)

**Purpose**: Smooth approximation to ReLU, used in transformers (BERT, GPT)

**Signature**:
```python
gelu(Tensor self, *, str approximate='none') -> Tensor
```

**Formula**:

**Exact**:
```
GELU(x) = x * Φ(x)
        = x * (1/2) * [1 + erf(x/√2)]
```

**Tanh Approximation** (`approximate='tanh'`):
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

Where:
- `Φ(x)` is the cumulative distribution function of standard normal
- `erf` is the error function

**YAML Definition** (`native_functions.yaml:5254-5263`):
```yaml
- func: gelu(Tensor self, *, str approximate='none') -> Tensor
  structured_delegate: gelu.out
  device_check: NoCheck   # TensorIterator
  python_module: nn
  dispatch:
    MkldnnCPU: mkldnn_gelu
    QuantizedCPU: gelu_quantized_cpu
    QuantizedCUDA: gelu_quantized_cuda
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_gelu
  tags: [core, pointwise]
```

**CPU Implementation** (`native/cpu/ActivationKernel.cpp`):
```cpp
void gelu_kernel(TensorIteratorBase& iter, GeluType approximate) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "gelu_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    if (approximate == GeluType::Tanh) {
      // Tanh approximation
      const scalar_t kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
      const scalar_t kKappa = 0.044715;
      cpu_kernel_vec(
          iter,
          [=](scalar_t x) -> scalar_t {
            auto x_cube = x * x * x;
            auto inner = kBeta * (x + kKappa * x_cube);
            return scalar_t(0.5) * x * (scalar_t(1) + std::tanh(inner));
          },
          [=](Vec x) {
            auto x_cube = x * x * x;
            auto inner = kBeta * (x + kKappa * x_cube);
            return Vec(0.5) * x * (Vec(1) + inner.tanh());
          });
    } else {
      // Exact via erf
      cpu_kernel_vec(
          iter,
          [=](scalar_t x) -> scalar_t {
            constexpr scalar_t kAlpha = M_SQRT1_2;  // 1/√2
            return x * scalar_t(0.5) * (scalar_t(1) + std::erf(x * kAlpha));
          },
          [=](Vec x) {
            const Vec kAlpha(M_SQRT1_2);
            return x * Vec(0.5) * (Vec(1) + (x * kAlpha).erf());
          });
    }
  });
}
```

**MLX Equivalent**:
```python
def gelu(x, approximate='none'):
    """GELU activation"""
    if approximate == 'tanh':
        # Tanh approximation
        import math
        kBeta = math.sqrt(2.0 / math.pi)
        kKappa = 0.044715
        inner = kBeta * (x + kKappa * mx.power(x, 3))
        return 0.5 * x * (1.0 + mx.tanh(inner))
    else:
        # Exact (if MLX has erf)
        return x * 0.5 * (1.0 + mx.erf(x / mx.sqrt(2.0)))
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: gelu(Tensor self, *, str approximate='none') -> Tensor
  self: gelu_backward(grad, self, approximate=approximate)
```

**Gradient Formula** (exact):
```
∂GELU/∂x = Φ(x) + x * φ(x)
         = Φ(x) + (x/√(2π)) * exp(-x²/2)
```

Where `φ(x)` is the standard normal probability density function.

**Usage Examples**:
```python
x = torch.randn(3, 4)

# Exact GELU
torch.nn.functional.gelu(x)

# Tanh approximation (faster)
torch.nn.functional.gelu(x, approximate='tanh')

# As module
gelu = nn.GELU()
out = gelu(x)
```

**Properties**:
- **Smooth**: Differentiable everywhere (unlike ReLU)
- **Non-monotonic**: Can have negative outputs for negative inputs
- **Probabilistic interpretation**: Multiplies input by probability of being greater than a random sample
- **Popular in transformers**: BERT, GPT models use GELU

---

### sigmoid (Logistic Sigmoid)

**Purpose**: Squashes values to (0, 1) range, used for binary classification and gates

**Signature**:
```python
sigmoid(Tensor self) -> Tensor
```

**Formula**: `σ(x) = 1 / (1 + exp(-x))`

**YAML Definition** (`native_functions.yaml:5444-5451`):
```yaml
- func: sigmoid(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: sigmoid.out
  variants: function, method
  dispatch:
    QuantizedCPU: sigmoid_quantized_cpu
    MkldnnCPU: mkldnn_sigmoid
  tags: [core, pointwise]
```

**CPU Implementation** (`native/cpu/UnaryOpsKernel.cpp`):
```cpp
void sigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "sigmoid_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-a));
        },
        [=](Vectorized<scalar_t> a) {
          return a.neg().exp().add(1).reciprocal();
        });
  });
}
```

**MLX Equivalent**:
```python
def sigmoid(x):
    """Sigmoid activation"""
    return mx.sigmoid(x)
    # Or manually: 1 / (1 + mx.exp(-x))
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: sigmoid(Tensor self) -> Tensor
  self: sigmoid_backward(grad, result)
```

**Gradient Formula**:
```
∂σ/∂x = σ(x) * (1 - σ(x))
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
torch.sigmoid(x)  # [0.1192, 0.2689, 0.5000, 0.7311, 0.8808]

# Binary classification output
logits = model(input)
probs = torch.sigmoid(logits)  # Probabilities in (0, 1)

# Gate in LSTM
gate = torch.sigmoid(W @ x + b)
```

**Properties**:
- **Range**: (0, 1)
- **Saturates**: Gradients vanish for large |x|
- **Centered at 0.5**: σ(0) = 0.5
- **Symmetric**: σ(-x) = 1 - σ(x)

**Common Uses**:
- Binary classification (output layer)
- Attention gates (LSTMs, GRUs)
- Gating mechanisms

---

### tanh (Hyperbolic Tangent)

**Purpose**: Squashes values to (-1, 1) range, zero-centered alternative to sigmoid

**Signature**:
```python
tanh(Tensor self) -> Tensor
```

**Formula**: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**YAML Definition** (`native_functions.yaml:6211-6221`):
```yaml
- func: tanh(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: tanh.out
  variants: function, method
  dispatch:
    QuantizedCPU: tanh_quantized_cpu
    MkldnnCPU: mkldnn_tanh
    SparseCPU, SparseCUDA, SparseMPS: tanh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: tanh_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_tanh
  tags: [core, pointwise]
```

**CPU Implementation**:
```cpp
void tanh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "tanh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::tanh(a); },
        [=](Vectorized<scalar_t> a) { return a.tanh(); }
    );
  });
}
```

**MLX Equivalent**:
```python
def tanh(x):
    """Tanh activation"""
    return mx.tanh(x)
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: tanh(Tensor self) -> Tensor
  self: tanh_backward(grad, result)
```

**Gradient Formula**:
```
∂tanh/∂x = 1 - tanh²(x)
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
torch.tanh(x)  # [-0.9640, -0.7616, 0.0000, 0.7616, 0.9640]

# RNN hidden state
h_t = torch.tanh(W @ x_t + U @ h_prev + b)
```

**Properties**:
- **Range**: (-1, 1)
- **Zero-centered**: tanh(0) = 0 (better than sigmoid)
- **Saturates**: Gradients vanish for large |x|
- **Relation to sigmoid**: tanh(x) = 2*σ(2x) - 1

**Comparison to Sigmoid**:
- Tanh is zero-centered → better for hidden layers
- Sigmoid outputs probabilities → better for output layers

---

### softmax (Normalized Exponential)

**Purpose**: Converts logits to probability distribution

**Signature**:
```python
softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
```

**Formula**:
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**Numerically Stable Version**:
```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

**YAML Definition** (`native_functions.yaml:5699-5715`):
```yaml
- func: softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
  variants: function, method
  # Delegates to _softmax

- func: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  structured_delegate: _softmax.out
  dispatch:
    MkldnnCPU: mkldnn_softmax
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: softmax_nested
  tags: core
```

**CPU Implementation** (`native/cpu/SoftMaxKernel.cpp`):
```cpp
void softmax_kernel(const Tensor& output, const Tensor& input, int64_t dim) {
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;

  // Calculate sizes for parallelization
  for (int64_t i = 0; i < dim; ++i) outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i) inner_size *= input.size(i);

  parallel_for(0, outer_size * inner_size, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t outer_idx = i / inner_size;
      int64_t inner_idx = i % inner_size;

      // Find max for numerical stability
      scalar_t max_val = input[outer_idx][0][inner_idx];
      for (int64_t d = 1; d < dim_size; ++d) {
        max_val = std::max(max_val, input[outer_idx][d][inner_idx]);
      }

      // Compute exp(x - max) and sum
      scalar_t sum = 0;
      for (int64_t d = 0; d < dim_size; ++d) {
        scalar_t exp_val = std::exp(input[outer_idx][d][inner_idx] - max_val);
        output[outer_idx][d][inner_idx] = exp_val;
        sum += exp_val;
      }

      // Normalize
      for (int64_t d = 0; d < dim_size; ++d) {
        output[outer_idx][d][inner_idx] /= sum;
      }
    }
  });
}
```

**MLX Equivalent**:
```python
def softmax(x, axis=-1):
    """Softmax activation"""
    return mx.softmax(x, axis=axis)
```

**Gradient**:
```
∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
```

Where `δ_ij` is Kronecker delta (1 if i==j, else 0).

**Usage Examples**:
```python
# Logits for 3 classes
logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)
# [0.6590, 0.2424, 0.0986]  (sums to 1.0)

# Batch of logits
logits = torch.randn(32, 10)  # Batch=32, Classes=10
probs = torch.softmax(logits, dim=1)  # Softmax over classes

# Temperature scaling (sharper/softer)
temperature = 0.5
probs = torch.softmax(logits / temperature, dim=1)
```

**Properties**:
- **Outputs probabilities**: Σ softmax(x_i) = 1
- **Differentiable**: Smooth function
- **Monotonic**: Preserves order of inputs
- **Translation invariant**: softmax(x + c) = softmax(x)

**Common Uses**:
- Multi-class classification (output layer)
- Attention weights
- Mixture models

---

### log_softmax (Log Softmax)

**Purpose**: Numerically stable log of softmax, used with NLL loss

**Signature**:
```python
log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
```

**Formula**:
```
log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
                 = x_i - max(x) - log(Σ_j exp(x_j - max(x)))
```

**Why Not `log(softmax(x))`?**
- Numerical stability: exp(large number) can overflow
- Precision: log(small number) loses precision
- Efficiency: Combined operation is faster

**YAML Definition** (`native_functions.yaml:3756-3776`):
```yaml
- func: log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
  variants: function, method
  # Delegates to _log_softmax

- func: _log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  structured_delegate: _log_softmax.out
  tags: core
```

**CPU Implementation** (similar to softmax but computes log):
```cpp
void log_softmax_kernel(const Tensor& output, const Tensor& input, int64_t dim) {
  // Similar structure to softmax but:
  // output[i] = input[i] - max - log(sum(exp(input - max)))

  scalar_t max_val = input.max_along_dim(dim);
  scalar_t log_sum_exp = std::log(
      (input - max_val).exp().sum_along_dim(dim)
  );
  output = input - max_val - log_sum_exp;
}
```

**MLX Equivalent**:
```python
def log_softmax(x, axis=-1):
    """Log softmax activation"""
    return mx.log_softmax(x, axis=axis)
    # Or: x - mx.logsumexp(x, axis=axis, keepdims=True)
```

**Gradient**:
```
∂log_softmax_i/∂x_j = δ_ij - softmax_j
```

**Usage Examples**:
```python
logits = torch.randn(32, 10)

# Method 1: log_softmax (recommended)
log_probs = torch.log_softmax(logits, dim=1)

# Method 2: log(softmax) (numerically unstable)
log_probs_unstable = torch.log(torch.softmax(logits, dim=1))

# Used with NLLLoss
loss = nn.NLLLoss()
target = torch.tensor([3])  # Class index
loss_val = loss(log_probs[0:1], target)

# Equivalent to CrossEntropyLoss
# CrossEntropyLoss = log_softmax + NLLLoss
```

**Properties**:
- **Numerically stable**: Avoids overflow/underflow
- **Log-space**: Values in (-∞, 0]
- **Efficient**: Fused operation faster than separate log+softmax

---

### silu (Sigmoid Linear Unit / Swish)

**Purpose**: Smooth activation, self-gated (x * σ(x)), used in modern architectures

**Signature**:
```python
silu(Tensor self) -> Tensor
```

**Formula**: `SiLU(x) = x * σ(x) = x / (1 + exp(-x))`

**Also Known As**: Swish activation (discovered by Google)

**YAML Definition** (`native_functions.yaml:5380-5401`):
```yaml
- func: silu(Tensor self) -> Tensor
  structured_delegate: silu.out
  python_module: nn
  dispatch:
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_silu
  tags: pointwise
```

**CPU Implementation**:
```cpp
void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "silu_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return a / (scalar_t(1) + std::exp(-a));
        },
        [=](Vectorized<scalar_t> a) {
          return a * a.neg().exp().add(1).reciprocal();
        });
  });
}
```

**MLX Equivalent**:
```python
def silu(x):
    """SiLU/Swish activation"""
    return x * mx.sigmoid(x)
```

**Gradient** (from `derivatives.yaml`):
```yaml
- name: silu(Tensor self) -> Tensor
  self: silu_backward(grad, self)
```

**Gradient Formula**:
```
∂SiLU/∂x = σ(x) + x * σ(x) * (1 - σ(x))
         = σ(x) * (1 + x * (1 - σ(x)))
```

**Usage Examples**:
```python
x = torch.randn(3, 4)
torch.nn.functional.silu(x)

# Equivalent using sigmoid
x * torch.sigmoid(x)

# As module
silu = nn.SiLU()
out = silu(x)
```

**Properties**:
- **Smooth**: Differentiable everywhere
- **Non-monotonic**: Can decrease for small negative values
- **Self-gated**: Multiplies input by its sigmoid
- **Unbounded above**: No upper saturation
- **Approaches linear** for large positive x

**Comparison to ReLU**:
- Smooth (no kink at 0)
- Small negative values preserved (not zeroed)
- Slightly more compute (sigmoid)

**Common Uses**:
- Transformer models (GPT-3 variants)
- Vision models (EfficientNet)
- Alternative to ReLU in modern architectures

---

## Activation Function Comparison

| Activation | Range | Smooth | Zero-Centered | Saturates | Common Use |
|------------|-------|--------|---------------|-----------|------------|
| ReLU | [0, ∞) | No | No | No (above) | CNNs, general |
| GELU | (-∞, ∞) | Yes | Yes | No | Transformers |
| Sigmoid | (0, 1) | Yes | No | Both | Binary class, gates |
| Tanh | (-1, 1) | Yes | Yes | Both | RNNs, hidden layers |
| Softmax | (0, 1)* | Yes | No | No | Multi-class output |
| Log Softmax | (-∞, 0] | Yes | No | No | With NLL loss |
| SiLU | (-∞, ∞) | Yes | No | No (above) | Modern architectures |

*Softmax: sum = 1

**Computational Cost** (relative):
```
ReLU:        Cheapest  (1 comparison)
Tanh:        Medium    (exp calls)
Sigmoid:     Medium    (exp call)
GELU:        Expensive (erf or tanh + powers)
SiLU:        Medium    (exp call)
Softmax:     Expensive (many exp + sum + div)
Log Softmax: Expensive (exp + log + sum)
```

---

## Implementation Files

**YAML Definitions**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:5181-5210` (relu)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:5254-5280` (gelu)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:5444-5467` (sigmoid)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:6211-6242` (tanh)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:5699-5720` (softmax)
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/native_functions.yaml:5380-5418` (silu)

**CPU Kernels**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/ActivationKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/cpu/SoftMaxKernel.cpp`

**MPS Kernels** (Metal):
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/Activation.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/ActivationKernel.mm`
- `/Users/zakkeown/Code/flashlight/reference/pytorch/aten/src/ATen/native/mps/operations/SoftMax.mm`

**Gradients**:
- `/Users/zakkeown/Code/flashlight/reference/pytorch/tools/autograd/derivatives.yaml`

---

## MLX Porting Summary

**Direct Mappings**:
```python
# PyTorch → MLX
torch.relu       → mx.maximum(x, 0)
torch.gelu       → mx.gelu (if available) or manual
torch.sigmoid    → mx.sigmoid
torch.tanh       → mx.tanh
torch.softmax    → mx.softmax
torch.log_softmax→ mx.log_softmax
torch.silu       → x * mx.sigmoid(x)
```

**Implementation Strategy**:
1. **Element-wise ops** (relu, sigmoid, tanh, silu): Direct mapping or simple expressions
2. **GELU**: May need manual implementation if MLX lacks erf
3. **Softmax/LogSoftmax**: Use MLX primitives or implement via logsumexp

**Considerations**:
- MLX likely has efficient implementations for common activations
- Softmax requires reduction primitives (sum along axis)
- Gradients handled automatically by `mx.grad`
- Metal backend provides hardware-accelerated implementations

**Example Compatibility Layer**:
```python
import mlx.core as mx

class F:  # torch.nn.functional equivalent
    @staticmethod
    def relu(x):
        return mx.maximum(x, 0)

    @staticmethod
    def gelu(x, approximate='none'):
        if approximate == 'tanh':
            import math
            return 0.5 * x * (1 + mx.tanh(
                math.sqrt(2/math.pi) * (x + 0.044715 * x**3)
            ))
        return x * 0.5 * (1 + mx.erf(x / mx.sqrt(2.0)))

    @staticmethod
    def sigmoid(x):
        return mx.sigmoid(x)

    @staticmethod
    def tanh(x):
        return mx.tanh(x)

    @staticmethod
    def softmax(x, dim=-1):
        return mx.softmax(x, axis=dim)

    @staticmethod
    def log_softmax(x, dim=-1):
        return mx.log_softmax(x, axis=dim)

    @staticmethod
    def silu(x):
        return x * mx.sigmoid(x)
```

Activation functions are fundamental to neural networks and have straightforward mappings to MLX, either as direct primitives or simple compositions of basic operations.

---

## Extended Activation Functions

The following activation functions extend beyond the core 7, providing specialized behavior for various architectures and use cases.

---

### elu (Exponential Linear Unit)

**Purpose**: Smooth alternative to ReLU with negative values, reduces "dying ReLU" problem

**Signature**:
```python
elu(Tensor input, Scalar alpha=1.0, bool inplace=False) -> Tensor
```

**Formula**:
```
ELU(x) = x                  if x > 0
       = α * (exp(x) - 1)   if x ≤ 0
```

**YAML Definition** (`native_functions.yaml`):
```yaml
- func: elu(Tensor self, Scalar alpha=1) -> Tensor
  structured_delegate: elu.out
  python_module: nn
  tags: pointwise
```

**CPU Implementation**:
```cpp
void elu_kernel(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "elu_cpu", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return a > 0 ? a : alpha * (std::exp(a) - scalar_t(1));
        },
        [=](Vectorized<scalar_t> a) {
          auto zero = Vectorized<scalar_t>(0);
          auto one = Vectorized<scalar_t>(1);
          auto alpha_vec = Vectorized<scalar_t>(alpha);
          return Vectorized<scalar_t>::blendv(
              alpha_vec * (a.exp() - one), a, a > zero);
        });
  });
}
```

**MLX Equivalent**:
```python
def elu(x, alpha=1.0):
    """ELU activation"""
    return mx.where(x > 0, x, alpha * (mx.exp(x) - 1))
```

**Gradient Formula**:
```
∂ELU/∂x = 1                     if x > 0
        = α * exp(x) = ELU(x) + α   if x ≤ 0
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
F.elu(x, alpha=1.0)  # [-0.8647, -0.6321, 0.0, 1.0, 2.0]

# With different alpha
F.elu(x, alpha=0.5)  # [-0.4323, -0.3161, 0.0, 1.0, 2.0]
```

**Properties**:
- **Negative outputs**: Non-zero for negative inputs (unlike ReLU)
- **Smooth**: Differentiable everywhere
- **Mean activations closer to zero**: Better gradient flow
- **Slower than ReLU**: Requires exp() computation

---

### selu (Scaled Exponential Linear Unit)

**Purpose**: Self-normalizing activation for deep networks, maintains mean~0 and variance~1

**Signature**:
```python
selu(Tensor input, bool inplace=False) -> Tensor
```

**Formula**:
```
SELU(x) = λ * x                      if x > 0
        = λ * α * (exp(x) - 1)       if x ≤ 0

where:
  λ ≈ 1.0507009873554804934193349852946
  α ≈ 1.6732632423543772848170429916717
```

**YAML Definition**:
```yaml
- func: selu(Tensor self) -> Tensor
  structured_delegate: selu.out
  python_module: nn
  tags: pointwise
```

**CPU Implementation**:
```cpp
void selu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "selu_cpu", [&]() {
    const scalar_t alpha = 1.6732632423543772848170429916717;
    const scalar_t scale = 1.0507009873554804934193349852946;
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return scale * (a > 0 ? a : alpha * (std::exp(a) - 1));
        },
        [=](Vectorized<scalar_t> a) {
          auto zero = Vectorized<scalar_t>(0);
          auto one = Vectorized<scalar_t>(1);
          auto alpha_vec = Vectorized<scalar_t>(alpha);
          auto scale_vec = Vectorized<scalar_t>(scale);
          auto pos = a * scale_vec;
          auto neg = scale_vec * alpha_vec * (a.exp() - one);
          return Vectorized<scalar_t>::blendv(neg, pos, a > zero);
        });
  });
}
```

**MLX Equivalent**:
```python
def selu(x):
    """SELU activation (self-normalizing)"""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * mx.where(x > 0, x, alpha * (mx.exp(x) - 1))
```

**Usage Examples**:
```python
x = torch.randn(100, 100)
out = F.selu(x)
# With proper initialization, mean ≈ 0, std ≈ 1

# Requires lecun_normal initialization
nn.init.lecun_normal_(layer.weight)
```

**Properties**:
- **Self-normalizing**: Preserves mean and variance through layers
- **Requires specific initialization**: lecun_normal
- **Dropout variant**: Use AlphaDropout, not regular Dropout
- **Fixed parameters**: α and λ are derived mathematically

---

### celu (Continuously Differentiable ELU)

**Purpose**: Smooth ELU variant with continuous derivative at x=0

**Signature**:
```python
celu(Tensor input, Scalar alpha=1.0, bool inplace=False) -> Tensor
```

**Formula**:
```
CELU(x) = max(0, x) + min(0, α * (exp(x/α) - 1))
```

**YAML Definition**:
```yaml
- func: celu(Tensor self, Scalar alpha=1.0) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def celu(x, alpha=1.0):
    """CELU activation (continuously differentiable ELU)"""
    return mx.maximum(x, 0) + mx.minimum(0, alpha * (mx.exp(x / alpha) - 1))
```

**Gradient Formula**:
```
∂CELU/∂x = 1                  if x ≥ 0
         = exp(x/α)           if x < 0
```

---

### leaky_relu (Leaky ReLU)

**Purpose**: ReLU variant with small negative slope to prevent dead neurons

**Signature**:
```python
leaky_relu(Tensor input, Scalar negative_slope=0.01, bool inplace=False) -> Tensor
```

**Formula**:
```
LeakyReLU(x) = x                    if x ≥ 0
             = negative_slope * x    if x < 0
```

**YAML Definition** (`native_functions.yaml`):
```yaml
- func: leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
  structured_delegate: leaky_relu.out
  python_module: nn
  dispatch:
    QuantizedCPU: leaky_relu_quantized_cpu
  tags: pointwise
```

**CPU Implementation**:
```cpp
void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "leaky_relu_cpu", [&]() {
    auto negval = negval_scalar.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return a > 0 ? a : a * negval;
        },
        [=](Vectorized<scalar_t> a) {
          auto zero = Vectorized<scalar_t>(0);
          auto negval_vec = Vectorized<scalar_t>(negval);
          return Vectorized<scalar_t>::blendv(a * negval_vec, a, a > zero);
        });
  });
}
```

**MLX Equivalent**:
```python
def leaky_relu(x, negative_slope=0.01):
    """Leaky ReLU activation"""
    return mx.where(x >= 0, x, negative_slope * x)
    # Or: mx.maximum(x, 0) + negative_slope * mx.minimum(x, 0)
```

**Gradient Formula**:
```
∂LeakyReLU/∂x = 1                if x > 0
              = negative_slope    if x ≤ 0
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
F.leaky_relu(x, 0.01)  # [-0.02, -0.01, 0.0, 1.0, 2.0]
F.leaky_relu(x, 0.1)   # [-0.2, -0.1, 0.0, 1.0, 2.0]
```

**Properties**:
- **No dead neurons**: Small gradient for negative values
- **Almost as fast as ReLU**: Simple comparison and multiply
- **Typical slopes**: 0.01 to 0.3

---

### prelu (Parametric ReLU)

**Purpose**: Learnable negative slope ReLU, parameters trained with backprop

**Signature**:
```python
prelu(Tensor input, Tensor weight) -> Tensor
```

**Formula**:
```
PReLU(x) = x           if x ≥ 0
         = weight * x   if x < 0
```

**YAML Definition**:
```yaml
- func: prelu(Tensor self, Tensor weight) -> Tensor
  python_module: nn
  dispatch:
    CPU: prelu_cpu
    CUDA: prelu_cuda
    MPS: prelu_mps
  tags: pointwise
```

**MLX Equivalent**:
```python
def prelu(x, weight):
    """PReLU activation with learnable negative slope"""
    # weight shape: (1,) for shared or (num_channels,) for per-channel
    return mx.where(x >= 0, x, weight * x)
```

**Gradient Formula**:
```
∂PReLU/∂x = 1       if x > 0
          = weight   if x ≤ 0

∂PReLU/∂weight = x    if x < 0
               = 0    otherwise
```

**Usage Examples**:
```python
# Single learnable parameter (shared across channels)
prelu = nn.PReLU()

# Per-channel parameters
prelu = nn.PReLU(num_parameters=64)  # 64 channels

x = torch.randn(1, 64, 32, 32)
out = prelu(x)
```

---

### rrelu (Randomized Leaky ReLU)

**Purpose**: Leaky ReLU with randomized negative slope during training

**Signature**:
```python
rrelu(Tensor input, Scalar lower=1./8, Scalar upper=1./3, bool training=False, bool inplace=False) -> Tensor
```

**Formula**:
```
Training:
  RReLU(x) = x              if x ≥ 0
           = a * x           if x < 0    (a sampled uniformly from [lower, upper])

Inference:
  RReLU(x) = x              if x ≥ 0
           = (lower+upper)/2 * x    if x < 0
```

**YAML Definition**:
```yaml
- func: rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def rrelu(x, lower=1./8, upper=1./3, training=False, key=None):
    """Randomized Leaky ReLU"""
    if training:
        # Sample random slope for each element
        a = mx.random.uniform(lower, upper, x.shape, key=key)
        return mx.where(x >= 0, x, a * x)
    else:
        # Use mean slope at inference
        slope = (lower + upper) / 2
        return mx.where(x >= 0, x, slope * x)
```

---

### hardsigmoid (Hard Sigmoid)

**Purpose**: Piecewise linear approximation to sigmoid, faster to compute

**Signature**:
```python
hardsigmoid(Tensor input, bool inplace=False) -> Tensor
```

**Formula**:
```
Hardsigmoid(x) = 0                    if x ≤ -3
               = 1                    if x ≥ 3
               = x/6 + 0.5            otherwise
```

**YAML Definition**:
```yaml
- func: hardsigmoid(Tensor self) -> Tensor
  structured_delegate: hardsigmoid.out
  python_module: nn
  tags: pointwise
```

**CPU Implementation**:
```cpp
void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "hardsigmoid_cpu", [&]() {
    const scalar_t zero(0.0f);
    const scalar_t one_sixth(1.0f / 6.0f);
    const scalar_t three(3.0f);
    const scalar_t six(6.0f);
    cpu_kernel_vec(
        iter,
        [=](scalar_t x) -> scalar_t {
          return std::min(std::max(x + three, zero), six) * one_sixth;
        },
        [=](Vectorized<scalar_t> x) {
          return vec::minimum(
              vec::maximum(x + Vectorized<scalar_t>(three), Vectorized<scalar_t>(zero)),
              Vectorized<scalar_t>(six)
          ) * Vectorized<scalar_t>(one_sixth);
        });
  });
}
```

**MLX Equivalent**:
```python
def hardsigmoid(x):
    """Hard sigmoid activation"""
    return mx.clip(x / 6 + 0.5, 0, 1)
    # Or: mx.minimum(mx.maximum(x + 3, 0), 6) / 6
```

**Gradient Formula**:
```
∂Hardsigmoid/∂x = 0      if x ≤ -3 or x ≥ 3
                = 1/6    otherwise
```

**Properties**:
- **Faster than sigmoid**: No exp() computation
- **Quantization-friendly**: Simple operations
- **Used in MobileNets**: Efficient mobile architectures

---

### hardswish (Hard Swish)

**Purpose**: Piecewise linear approximation to swish/silu, used in MobileNetV3

**Signature**:
```python
hardswish(Tensor input, bool inplace=False) -> Tensor
```

**Formula**:
```
Hardswish(x) = 0                            if x ≤ -3
             = x                            if x ≥ 3
             = x * (x + 3) / 6              otherwise
```

**YAML Definition**:
```yaml
- func: hardswish(Tensor self) -> Tensor
  structured_delegate: hardswish.out
  python_module: nn
  tags: pointwise
```

**CPU Implementation**:
```cpp
void hardswish_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "hardswish_cpu", [&]() {
    const scalar_t zero(0.0f);
    const scalar_t three(3.0f);
    const scalar_t six(6.0f);
    cpu_kernel_vec(
        iter,
        [=](scalar_t x) -> scalar_t {
          return x * std::min(std::max(x + three, zero), six) / six;
        },
        [=](Vectorized<scalar_t> x) {
          return x * vec::minimum(
              vec::maximum(x + Vectorized<scalar_t>(three), Vectorized<scalar_t>(zero)),
              Vectorized<scalar_t>(six)
          ) / Vectorized<scalar_t>(six);
        });
  });
}
```

**MLX Equivalent**:
```python
def hardswish(x):
    """Hard swish activation (MobileNetV3)"""
    return x * mx.clip(x + 3, 0, 6) / 6
```

**Gradient Formula**:
```
∂Hardswish/∂x = 0                  if x ≤ -3
              = 1                  if x ≥ 3
              = (2x + 3) / 6       otherwise
```

**Properties**:
- **Approximates SiLU**: x * hardsigmoid(x)
- **Used in MobileNetV3**: Efficient inference
- **No exp() computation**: Fast on mobile devices

---

### hardtanh (Hard Tanh)

**Purpose**: Piecewise linear approximation to tanh, bounded output

**Signature**:
```python
hardtanh(Tensor input, Scalar min_val=-1, Scalar max_val=1, bool inplace=False) -> Tensor
```

**Formula**:
```
Hardtanh(x) = min_val    if x < min_val
            = max_val    if x > max_val
            = x          otherwise
```

**YAML Definition**:
```yaml
- func: hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
  python_module: nn
  dispatch:
    CPU, CUDA: hardtanh
    QuantizedCPU: hardtanh_quantized_cpu
  tags: pointwise
```

**MLX Equivalent**:
```python
def hardtanh(x, min_val=-1.0, max_val=1.0):
    """Hard tanh activation (clipping)"""
    return mx.clip(x, min_val, max_val)
```

**Gradient Formula**:
```
∂Hardtanh/∂x = 0    if x < min_val or x > max_val
             = 1    otherwise
```

**Usage Examples**:
```python
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
F.hardtanh(x)  # [-1.0, -1.0, 0.0, 1.0, 1.0]

# Custom bounds
F.hardtanh(x, min_val=-2, max_val=2)  # [-2.0, -1.0, 0.0, 1.0, 2.0]
```

---

### mish (Mish Activation)

**Purpose**: Smooth self-regularizing activation, better than ReLU/Swish in some cases

**Signature**:
```python
mish(Tensor input, bool inplace=False) -> Tensor
```

**Formula**:
```
Mish(x) = x * tanh(softplus(x))
        = x * tanh(ln(1 + exp(x)))
```

**YAML Definition**:
```yaml
- func: mish(Tensor self) -> Tensor
  structured_delegate: mish.out
  python_module: nn
  tags: pointwise
```

**CPU Implementation**:
```cpp
void mish_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "mish_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t {
          return a * std::tanh(std::log1p(std::exp(a)));
        },
        [=](Vectorized<scalar_t> a) {
          return a * (a.exp().log1p()).tanh();
        });
  });
}
```

**MLX Equivalent**:
```python
def mish(x):
    """Mish activation"""
    return x * mx.tanh(mx.log(1 + mx.exp(x)))
    # Or with softplus: x * mx.tanh(softplus(x))
```

**Gradient Formula**:
```
∂Mish/∂x = sech²(softplus(x)) * sigmoid(x) + Mish(x)/x
```

Where sech(x) = 1/cosh(x)

**Properties**:
- **Smooth**: Infinitely differentiable
- **Non-monotonic**: Allows negative values
- **Unbounded above**: No saturation for positive x
- **Self-regularizing**: Implicit regularization effect
- **Expensive**: Requires exp, log, and tanh

---

### softplus (Softplus)

**Purpose**: Smooth approximation to ReLU, always positive output

**Signature**:
```python
softplus(Tensor input, Scalar beta=1, Scalar threshold=20) -> Tensor
```

**Formula**:
```
Softplus(x) = (1/β) * log(1 + exp(β * x))

For numerical stability, when β*x > threshold:
Softplus(x) = x
```

**YAML Definition**:
```yaml
- func: softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
  python_module: nn
  tags: pointwise
```

**CPU Implementation**:
```cpp
void softplus_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "softplus_cpu", [&]() {
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
      auto z = a * beta;
      return z > threshold ? a : std::log1p(std::exp(z)) / beta;
    });
  });
}
```

**MLX Equivalent**:
```python
def softplus(x, beta=1.0, threshold=20.0):
    """Softplus activation"""
    scaled = beta * x
    return mx.where(scaled > threshold, x, mx.log(1 + mx.exp(scaled)) / beta)
```

**Gradient Formula**:
```
∂Softplus/∂x = sigmoid(β * x)
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
F.softplus(x)  # [0.1269, 0.3133, 0.6931, 1.3133, 2.1269]

# Different beta (sharper transition)
F.softplus(x, beta=2)  # [0.0635, 0.1566, 0.3466, 0.6566, 1.0635]
```

**Properties**:
- **Smooth ReLU**: Differentiable everywhere
- **Always positive**: Range (0, ∞)
- **Approaches ReLU**: As β → ∞

---

### softsign (Softsign)

**Purpose**: Tanh alternative with slower saturation

**Signature**:
```python
softsign(Tensor input) -> Tensor
```

**Formula**:
```
Softsign(x) = x / (1 + |x|)
```

**YAML Definition**:
```yaml
- func: softsign(Tensor self) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def softsign(x):
    """Softsign activation"""
    return x / (1 + mx.abs(x))
```

**Gradient Formula**:
```
∂Softsign/∂x = 1 / (1 + |x|)²
```

**Properties**:
- **Range**: (-1, 1)
- **Slower saturation than tanh**: Polynomial vs exponential decay
- **Cheaper to compute**: No exp()

---

### softshrink (Soft Shrinkage)

**Purpose**: Shrinkage function for sparse representations

**Signature**:
```python
softshrink(Tensor input, Scalar lambd=0.5) -> Tensor
```

**Formula**:
```
Softshrink(x) = x - λ    if x > λ
              = x + λ    if x < -λ
              = 0        otherwise
```

**YAML Definition**:
```yaml
- func: softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def softshrink(x, lambd=0.5):
    """Soft shrinkage activation"""
    return mx.where(x > lambd, x - lambd,
                   mx.where(x < -lambd, x + lambd, 0))
```

**Gradient Formula**:
```
∂Softshrink/∂x = 1    if |x| > λ
               = 0    otherwise
```

---

### hardshrink (Hard Shrinkage)

**Purpose**: Hard thresholding, sets small values to zero

**Signature**:
```python
hardshrink(Tensor input, Scalar lambd=0.5) -> Tensor
```

**Formula**:
```
Hardshrink(x) = x    if |x| > λ
              = 0    otherwise
```

**YAML Definition**:
```yaml
- func: hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def hardshrink(x, lambd=0.5):
    """Hard shrinkage activation"""
    return mx.where(mx.abs(x) > lambd, x, 0)
```

**Gradient Formula**:
```
∂Hardshrink/∂x = 1    if |x| > λ
               = 0    otherwise
```

---

### tanhshrink (Tanh Shrink)

**Purpose**: Identity minus tanh, preserves sign for large values

**Signature**:
```python
tanhshrink(Tensor input) -> Tensor
```

**Formula**:
```
Tanhshrink(x) = x - tanh(x)
```

**YAML Definition**:
```yaml
- func: tanhshrink(Tensor self) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def tanhshrink(x):
    """Tanh shrink activation"""
    return x - mx.tanh(x)
```

**Gradient Formula**:
```
∂Tanhshrink/∂x = tanh²(x)
```

---

### threshold (Threshold)

**Purpose**: Simple thresholding, values below threshold become fixed value

**Signature**:
```python
threshold(Tensor input, Scalar threshold, Scalar value, bool inplace=False) -> Tensor
```

**Formula**:
```
Threshold(x) = x        if x > threshold
             = value    otherwise
```

**YAML Definition**:
```yaml
- func: threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
  python_module: nn
  dispatch:
    CPU, CUDA: threshold
    QuantizedCPU, QuantizedCUDA: threshold_quantized_cpu
  tags: pointwise
```

**MLX Equivalent**:
```python
def threshold(x, threshold_val, value):
    """Threshold activation"""
    return mx.where(x > threshold_val, x, value)
```

**Usage Examples**:
```python
x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
F.threshold(x, threshold=0.5, value=0.0)  # [0.0, 0.0, 0.0, 1.0, 2.0]

# Different replacement value
F.threshold(x, threshold=0.5, value=-1.0)  # [-1.0, -1.0, -1.0, 1.0, 2.0]
```

---

### glu (Gated Linear Unit)

**Purpose**: Split input and gate one half with sigmoid of other half

**Signature**:
```python
glu(Tensor input, int dim=-1) -> Tensor
```

**Formula**:
```
GLU(x) = a * σ(b)

where x is split into a and b along dim
```

**YAML Definition**:
```yaml
- func: glu(Tensor self, int dim=-1) -> Tensor
  python_module: nn
```

**Implementation Logic**:
```cpp
Tensor glu(const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  auto chunks = self.chunk(2, wrap_dim);
  return chunks[0] * chunks[1].sigmoid();
}
```

**MLX Equivalent**:
```python
def glu(x, axis=-1):
    """Gated Linear Unit"""
    a, b = mx.split(x, 2, axis=axis)
    return a * mx.sigmoid(b)
```

**Gradient**:
```
∂GLU/∂a = σ(b)
∂GLU/∂b = a * σ(b) * (1 - σ(b))
```

**Usage Examples**:
```python
# Input must have even size along dim
x = torch.randn(32, 128)  # Split 128 → 64 + 64
out = F.glu(x, dim=-1)    # Output shape: (32, 64)

# Common in language models
linear = nn.Linear(512, 1024)  # Double output for GLU
x = F.glu(linear(input))  # Output: 512 features
```

**Properties**:
- **Output halves input**: Reduces dimension by 2
- **Gating mechanism**: Learns which features to pass through
- **Used in Transformers**: Gated FFN variants

---

### relu6 (ReLU6)

**Purpose**: ReLU capped at 6, useful for quantization and mobile models

**Signature**:
```python
relu6(Tensor input, bool inplace=False) -> Tensor
```

**Formula**:
```
ReLU6(x) = min(max(0, x), 6)
```

**YAML Definition**:
```yaml
- func: relu6(Tensor self) -> Tensor
  python_module: nn
  tags: pointwise
```

**MLX Equivalent**:
```python
def relu6(x):
    """ReLU6 activation (bounded ReLU)"""
    return mx.clip(x, 0, 6)
    # Or: mx.minimum(mx.maximum(x, 0), 6)
```

**Gradient Formula**:
```
∂ReLU6/∂x = 1    if 0 < x < 6
          = 0    otherwise
```

**Usage Examples**:
```python
x = torch.tensor([-2.0, 0.0, 3.0, 6.0, 10.0])
F.relu6(x)  # [0.0, 0.0, 3.0, 6.0, 6.0]

# Used in MobileNet architectures
```

**Properties**:
- **Bounded output**: Range [0, 6]
- **Quantization-friendly**: Known output range
- **Mobile-optimized**: Used in MobileNetV1/V2

---

## Extended Activation Comparison

| Activation | Range | Formula | Use Case |
|------------|-------|---------|----------|
| elu | (-α, ∞) | α*(exp(x)-1) for x≤0 | Deep networks |
| selu | (-λα, ∞) | Self-normalizing ELU | Self-normalizing nets |
| celu | (-α, ∞) | Continuous ELU | Smooth alternative |
| leaky_relu | (-∞, ∞) | αx for x<0 | Prevent dead neurons |
| prelu | (-∞, ∞) | Learnable slope | Adaptive activation |
| rrelu | (-∞, ∞) | Random slope | Regularization |
| hardsigmoid | [0, 1] | Linear approx | Mobile/quantized |
| hardswish | ~[-0.4, ∞) | x*hardsigmoid | MobileNetV3 |
| hardtanh | [min, max] | Clipping | Bounded output |
| mish | ~[-0.3, ∞) | x*tanh(softplus(x)) | State-of-art |
| softplus | (0, ∞) | log(1+exp(x)) | Smooth ReLU |
| softsign | (-1, 1) | x/(1+|x|) | Tanh alternative |
| softshrink | (-∞, ∞) | Soft threshold | Sparsity |
| hardshrink | (-∞, ∞) | Hard threshold | Sparsity |
| tanhshrink | (-∞, ∞) | x - tanh(x) | Specialized |
| threshold | (-∞, ∞) | Conditional | Thresholding |
| glu | (-∞, ∞) | Gated | Language models |
| relu6 | [0, 6] | Bounded ReLU | Mobile/quantized |

---

## MLX Extended Activation Implementations

```python
import mlx.core as mx

class ActivationFunctions:
    """Extended activation functions for MLX"""

    @staticmethod
    def elu(x, alpha=1.0):
        return mx.where(x > 0, x, alpha * (mx.exp(x) - 1))

    @staticmethod
    def selu(x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * mx.where(x > 0, x, alpha * (mx.exp(x) - 1))

    @staticmethod
    def celu(x, alpha=1.0):
        return mx.maximum(x, 0) + mx.minimum(0, alpha * (mx.exp(x / alpha) - 1))

    @staticmethod
    def leaky_relu(x, negative_slope=0.01):
        return mx.where(x >= 0, x, negative_slope * x)

    @staticmethod
    def prelu(x, weight):
        return mx.where(x >= 0, x, weight * x)

    @staticmethod
    def hardsigmoid(x):
        return mx.clip(x / 6 + 0.5, 0, 1)

    @staticmethod
    def hardswish(x):
        return x * mx.clip(x + 3, 0, 6) / 6

    @staticmethod
    def hardtanh(x, min_val=-1.0, max_val=1.0):
        return mx.clip(x, min_val, max_val)

    @staticmethod
    def mish(x):
        return x * mx.tanh(mx.log(1 + mx.exp(x)))

    @staticmethod
    def softplus(x, beta=1.0, threshold=20.0):
        scaled = beta * x
        return mx.where(scaled > threshold, x, mx.log(1 + mx.exp(scaled)) / beta)

    @staticmethod
    def softsign(x):
        return x / (1 + mx.abs(x))

    @staticmethod
    def softshrink(x, lambd=0.5):
        return mx.where(x > lambd, x - lambd,
                       mx.where(x < -lambd, x + lambd, 0))

    @staticmethod
    def hardshrink(x, lambd=0.5):
        return mx.where(mx.abs(x) > lambd, x, 0)

    @staticmethod
    def tanhshrink(x):
        return x - mx.tanh(x)

    @staticmethod
    def threshold(x, threshold_val, value):
        return mx.where(x > threshold_val, x, value)

    @staticmethod
    def glu(x, axis=-1):
        a, b = mx.split(x, 2, axis=axis)
        return a * mx.sigmoid(b)

    @staticmethod
    def relu6(x):
        return mx.clip(x, 0, 6)
```

---

## Activation Coverage Summary

| Category | Operators | Status |
|----------|-----------|--------|
| Core (7) | relu, gelu, sigmoid, tanh, softmax, log_softmax, silu | ✅ Documented |
| ELU Family (3) | elu, selu, celu | ✅ Documented |
| Leaky Family (3) | leaky_relu, prelu, rrelu | ✅ Documented |
| Hard Family (4) | hardsigmoid, hardswish, hardtanh, relu6 | ✅ Documented |
| Soft Family (3) | softplus, softsign, softshrink | ✅ Documented |
| Shrink Family (2) | hardshrink, tanhshrink | ✅ Documented |
| Special (3) | threshold, glu, mish | ✅ Documented |

**Total**: 25 activation functions documented (100% coverage)
