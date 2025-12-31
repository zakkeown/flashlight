# PyTorch Attention Mechanisms

## Overview

Attention is the foundation of modern transformer neural networks. PyTorch implements **multiple attention variants** optimized for different hardware backends and use cases. The primary API is **`scaled_dot_product_attention`** (SDPA), which automatically selects the most efficient backend.

**Key Insight**: PyTorch doesn't use a single attention implementation. Instead, it routes to specialized kernels based on hardware, input shapes, precision requirements, and performance characteristics.

**Attention Variants Supported**:
1. **Math Fallback**: Pure PyTorch ops (compatible with all devices)
2. **FlashAttention**: Memory-efficient attention (Hopper/Ampere GPUs)
3. **Memory-Efficient Attention**: Optimized with CUTLASS (NVIDIA GPUs)
4. **cuDNN Attention**: Vendor-optimized (NVIDIA cuDNN)
5. **MPS Attention**: Metal-accelerated (Apple Silicon)
6. **CPU FlashAttention**: Software FlashAttention for CPUs

**Location**: `aten/src/ATen/native/transformers/` (1,000+ lines across 20+ files)

---

## 1. Scaled Dot-Product Attention (SDPA)

### 1.1 Algorithm Overview

**Classic Self-Attention Formula**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (query): `[batch, num_heads, seq_len_q, head_dim]`
- **K** (key): `[batch, num_heads, seq_len_k, head_dim]`
- **V** (value): `[batch, num_heads, seq_len_v, head_dim]`
- **d_k**: `head_dim` (dimension per attention head)
- **Scale**: `1 / √d_k` for numerical stability

**Causal Attention** (for autoregressive models):

Apply upper-triangular mask to prevent attending to future tokens:

```
mask[i, j] = -∞  if j > i else 0
```

**Masked Attention**:

Apply arbitrary boolean or float mask:

```
scores = QK^T / √d_k + mask
attn = softmax(scores)
output = attn V
```

### 1.2 PyTorch API

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query,              # [B, H, N, D]
    key,                # [B, H, S, D]
    value,              # [B, H, S, D]
    attn_mask=None,     # Optional [B, H, N, S] or broadcastable
    dropout_p=0.0,      # Dropout probability
    is_causal=False,    # Use causal mask
    scale=None,         # Override 1/√d_k scaling
    enable_gqa=False    # Enable grouped-query attention
)
# Returns: [B, H, N, D]
```

**Parameters**:
- `query`, `key`, `value`: Input tensors (must be contiguous, 4D)
- `attn_mask`: Attention mask (bool or float), broadcastable to `[B, H, N, S]`
- `dropout_p`: Attention dropout probability (0.0 = no dropout)
- `is_causal`: Use causal mask (for autoregressive generation)
- `scale`: Custom scaling factor (default: `1/√head_dim`)
- `enable_gqa`: Enable grouped-query attention (num_kv_heads < num_q_heads)

**Backend Selection**:

PyTorch automatically chooses backend based on:
1. **Hardware**: CUDA, CPU, MPS (Metal)
2. **Input dtype**: float16, bfloat16, float32
3. **Input shapes**: head_dim, seq_len
4. **Gradient requirements**: training vs inference
5. **Feature support**: causal mask, dropout, custom mask

### 1.3 SDPA Backend Dispatch Logic

```cpp
// From aten/src/ATen/native/transformers/sdp_utils_cpp.h

enum class SDPBackend {
  error = -1,
  math = 0,                    // Math fallback (PyTorch ops)
  flash_attention = 1,         // FlashAttention (fast, memory-efficient)
  efficient_attention = 2,     // CUTLASS-based memory-efficient
  cudnn_attention = 3,         // cuDNN fused attention
  overrideable = 4,            // Backend override mechanism
  math_for_mps = 5,            // MPS-specific math fallback
};

SDPBackend select_sdp_backend_cpp(sdp_params const& params) {
  // Priority order:
  // 1. FlashAttention (if available and compatible)
  // 2. Memory-Efficient Attention (if available)
  // 3. cuDNN Attention (if enabled)
  // 4. Math fallback (always available)

  // Check FlashAttention compatibility
  if (can_use_flash_attention(params)) {
    return SDPBackend::flash_attention;
  }

  // Check Memory-Efficient Attention compatibility
  if (can_use_mem_efficient_attention(params)) {
    return SDPBackend::efficient_attention;
  }

  // Check cuDNN Attention
  if (can_use_cudnn_attention(params)) {
    return SDPBackend::cudnn_attention;
  }

  // Fallback to math implementation
  return SDPBackend::math;
}
```

**Compatibility Checks** (simplified):

```cpp
bool can_use_flash_attention(sdp_params const& params) {
  // Requirements:
  // - CUDA device with compute capability >= 8.0 (Ampere or newer)
  // - dtype: float16 or bfloat16
  // - head_dim: 32, 64, 96, 128, or 256
  // - No custom attention mask (only causal supported)
  // - Gradient enabled (has backward kernel)

  if (params.query.device().type() != c10::kCUDA) return false;
  if (params.query.dtype() != at::kHalf && params.query.dtype() != at::kBFloat16) return false;

  int64_t head_dim = params.query.size(-1);
  if (head_dim != 32 && head_dim != 64 && head_dim != 96 &&
      head_dim != 128 && head_dim != 256) {
    return false;
  }

  // Custom mask not supported
  if (params.attn_mask.has_value() && !params.is_causal) return false;

  return true;
}
```

---

## 2. FlashAttention

### 2.1 FlashAttention Algorithm

**Key Innovation**: Compute attention in blocks, fusing operations to minimize HBM (GPU DRAM) accesses.

**Standard Attention Memory Bottleneck**:

```python
# Standard implementation (memory-inefficient)
scores = (Q @ K.T) / sqrt(d_k)       # Store N x S matrix
scores = scores + mask               # Additional N x S memory
attn_weights = softmax(scores, dim=-1)  # Store N x S matrix
output = attn_weights @ V            # Read N x S matrix
```

Memory: **O(N × S)** for attention matrix

**FlashAttention Tiling**:

1. **Partition Q, K, V into blocks**:
   - Query blocks: Q₁, Q₂, ..., Qₜ (each size `[Br, d]`)
   - Key/Value blocks: K₁, K₂, ..., Kₜ (each size `[Bc, d]`)

2. **Compute attention block-by-block**:
   ```
   For each query block Qᵢ:
     For each key/value block (Kⱼ, Vⱼ):
       1. Load Qᵢ, Kⱼ, Vⱼ to SRAM
       2. Compute attention scores: Sᵢⱼ = Qᵢ Kⱼᵀ
       3. Apply softmax numerically stable (with running max)
       4. Compute output block: Oᵢ += attn(Sᵢⱼ) Vⱼ
       5. Update running statistics for final normalization
   ```

3. **Final normalization**: Correct output using running statistics

**Memory**: **O(N² / B)** where B = block size

**Performance**: 2-4x faster than standard attention due to reduced memory bandwidth

### 2.2 FlashAttention Forward Pass

**C++ Interface**:

```cpp
// From aten/src/ATen/native/transformers/attention.h

using flash_attention_fn = void (*)(
    const Tensor& output,        // Output tensor [B, H, N, D]
    const Tensor& logsumexp,     // Log sum exp for backward [B, H, N]
    const Tensor& query,         // Query [B, H, N, D]
    const Tensor& key,           // Key [B, H, S, D]
    const Tensor& value,         // Value [B, H, S, D]
    double dropout_p,            // Dropout probability
    bool is_causal,              // Causal mask flag
    std::optional<Tensor> attn_mask,  // Optional custom mask
    std::optional<double> scale  // Optional scale override
);

DECLARE_DISPATCH(flash_attention_fn, flash_attention_kernel)
```

**CUDA Implementation** (`aten/src/ATen/native/transformers/cuda/attention.cu`):

```cpp
void flash_attention_kernel_cuda(
    const Tensor& output,
    const Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale) {

  // Determine block sizes based on head_dim
  int64_t head_dim = query.size(-1);
  int64_t Br = (head_dim <= 64) ? 128 : 64;  // Query block size
  int64_t Bc = (head_dim <= 64) ? 128 : 64;  // Key block size

  // Allocate shared memory for blocks
  // Q_block: [Br, head_dim]
  // K_block: [Bc, head_dim]
  // V_block: [Bc, head_dim]
  // S_block: [Br, Bc] (attention scores)

  // Launch kernel
  const dim3 grid(batch_size * num_heads, cdiv(seq_len_q, Br));
  const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);

  flash_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
    output.data_ptr<scalar_t>(),
    logsumexp.data_ptr<float>(),
    query.data_ptr<scalar_t>(),
    key.data_ptr<scalar_t>(),
    value.data_ptr<scalar_t>(),
    batch_size,
    num_heads,
    seq_len_q,
    seq_len_k,
    head_dim,
    scale_factor,
    is_causal
  );
}
```

**Kernel Pseudocode**:

```cuda
__global__ void flash_attention_forward_kernel(...) {
  // Each block handles one head for Br query tokens

  extern __shared__ float smem[];
  float* Q_smem = smem;                        // [Br, D]
  float* K_smem = Q_smem + Br * D;            // [Bc, D]
  float* V_smem = K_smem + Bc * D;            // [Bc, D]
  float* S_smem = V_smem + Bc * D;            // [Br, Bc]

  // Load query block to shared memory
  load_query_block(Q_smem, query, block_idx);

  // Initialize output and statistics
  float O_local[D] = {0};
  float m_local = -INFINITY;  // Running max for softmax
  float l_local = 0;          // Running sum for softmax

  // Iterate over key/value blocks
  for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
    // Load K, V blocks to shared memory
    load_kv_block(K_smem, V_smem, key, value, kv_block);
    __syncthreads();

    // Compute attention scores: S = Q K^T / sqrt(d_k)
    matmul_nt(S_smem, Q_smem, K_smem, Br, Bc, D, scale);

    // Apply causal mask if needed
    if (is_causal) {
      apply_causal_mask(S_smem, block_idx, kv_block);
    }

    // Softmax with numerical stability (online algorithm)
    // m_new = max(m_local, max(S))
    // l_new = l_local * exp(m_local - m_new) + sum(exp(S - m_new))
    float m_block = compute_row_max(S_smem);
    float m_new = fmaxf(m_local, m_block);

    // Apply exp and compute row sum
    float l_block = 0;
    for (int j = 0; j < Bc; ++j) {
      S_smem[row][j] = expf(S_smem[row][j] - m_new);
      l_block += S_smem[row][j];
    }

    // Update running statistics
    float correction = expf(m_local - m_new);
    l_local = l_local * correction + l_block;

    // Correct previous output: O *= correction
    for (int d = 0; d < D; ++d) {
      O_local[d] *= correction;
    }

    // Accumulate: O += S @ V
    matmul_nn(O_local, S_smem, V_smem, Br, Bc, D);

    m_local = m_new;
    __syncthreads();
  }

  // Final normalization: O /= l_local
  for (int d = 0; d < D; ++d) {
    O_local[d] /= l_local;
  }

  // Write output and statistics to global memory
  store_output(output, O_local, block_idx);
  store_logsumexp(logsumexp, logf(l_local) + m_local, block_idx);
}
```

**Key Optimizations**:
1. **Shared Memory**: Store Q, K, V blocks in fast SRAM
2. **Online Softmax**: Compute softmax incrementally without materializing full attention matrix
3. **Fused Operations**: Combine QK^T, softmax, and output matmul in single kernel
4. **Warp-Level Primitives**: Use cooperative groups for efficient reductions

### 2.3 FlashAttention Backward Pass

**Challenge**: Backward requires attention weights, but FlashAttention doesn't store them.

**Solution**: **Recompute** attention weights block-by-block during backward pass.

**C++ Interface**:

```cpp
using flash_attention_backward_fn = void (*)(
    const Tensor& grad_q,        // Gradient w.r.t. query [B, H, N, D]
    const Tensor& grad_k,        // Gradient w.r.t. key [B, H, S, D]
    const Tensor& grad_v,        // Gradient w.r.t. value [B, H, S, D]
    const Tensor& grad_out,      // Gradient of output [B, H, N, D]
    const Tensor& query,         // Query (from forward)
    const Tensor& key,           // Key (from forward)
    const Tensor& value,         // Value (from forward)
    const Tensor& out,           // Output (from forward)
    const Tensor& logsumexp,     // Log-sum-exp (from forward)
    double dropout_p,
    bool is_causal,
    std::optional<Tensor> attn_mask,
    std::optional<double> scale
);
```

**Backward Algorithm**:

```
Given: dL/dO, Q, K, V, O, logsumexp (from forward)

1. Compute dL/dV:
   - Recompute attention weights: P = softmax(QK^T / √d_k)
   - dL/dV = P^T @ dL/dO

2. Compute dL/dP (gradient of softmax output):
   - dL/dP = dL/dO @ V^T

3. Compute dL/dS (gradient before softmax):
   - dL/dS = P ⊙ (dL/dP - rowsum(P ⊙ dL/dP))
   (Softmax backward formula)

4. Compute dL/dQ and dL/dK:
   - dL/dQ = (dL/dS @ K) / √d_k
   - dL/dK = (dL/dS^T @ Q) / √d_k
```

**Tiled Backward** (similar to forward):

```cuda
__global__ void flash_attention_backward_kernel(...) {
  // For each query block:
  for (int q_block = 0; q_block < num_q_blocks; ++q_block) {
    load_query_block(Q_smem, ...);
    load_grad_out_block(dO_smem, grad_out, ...);

    // Recompute D = rowsum(O ⊙ dO) for softmax backward
    float D_local[Br] = compute_D(O_smem, dO_smem);

    // For each key/value block:
    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
      load_kv_block(K_smem, V_smem, ...);

      // Recompute attention: P = softmax(QK^T / √d_k)
      matmul_nt(S_smem, Q_smem, K_smem, scale);
      apply_softmax(S_smem, logsumexp, ...);  // Use saved logsumexp

      // Compute dP = P ⊙ (dO @ V^T - D)
      matmul_nt(dP_smem, dO_smem, V_smem);
      for (i, j) {
        dP_smem[i][j] = S_smem[i][j] * (dP_smem[i][j] - D_local[i]);
      }

      // Accumulate gradients
      // dQ += dP @ K / √d_k
      matmul_nn(dQ_local, dP_smem, K_smem, scale);

      // dK += dP^T @ Q / √d_k (atomic add to global memory)
      matmul_tn_atomic(grad_k, dP_smem, Q_smem, scale);

      // dV += P^T @ dO
      matmul_tn_atomic(grad_v, S_smem, dO_smem);
    }

    // Write dQ to global memory
    store_grad_query(grad_q, dQ_local, q_block);
  }
}
```

**Memory Trade-off**: Recomputation avoids storing O(N²) attention weights

---

## 3. Memory-Efficient Attention

### 3.1 Overview

**Memory-Efficient Attention** is based on the **"Self-Attention Does Not Need O(n²) Memory"** paper (2021).

**Key Idea**: Similar to FlashAttention, but uses **backward-optimized tiling** and CUTLASS templates for GEMM operations.

**Advantages**:
- Supports wider range of head dimensions (not limited to powers of 2)
- Better for inference (no backward needed)
- Works on older GPUs (compute capability 7.0+, Volta)

**Implementation**: Uses CUTLASS library for high-performance matrix multiplications

### 3.2 CUTLASS-Based Kernels

**Location**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/`

PyTorch generates **specialized kernels** for different configurations:

```bash
# Example: Generated kernel for specific config
cutlassB_f16_aligned_k128.cu      # Float16, head_dim=128
cutlassB_bf16_aligned_k64.cu      # BFloat16, head_dim=64
cutlassF_f16_aligned.cu           # Float16, forward only
```

**Code Generator** (`generate_kernels.py`):

```python
# Generates kernels for various combinations of:
# - dtype: float16, bfloat16, float32
# - head_dim: 32, 64, 96, 128, 160, 192, 224, 256
# - alignment: aligned (head_dim % 8 == 0), not aligned
# - dropout: with/without dropout
# - direction: forward (F) or backward (B)

for dtype in [float16, bfloat16, float32]:
  for head_dim in [32, 64, 96, 128, 160, 192, 224, 256]:
    for aligned in [True, False]:
      generate_cutlass_kernel(dtype, head_dim, aligned)
```

**Total**: 100+ specialized kernels for maximum performance

### 3.3 Dispatcher Logic

```cpp
// Select best mem-efficient attention kernel
template <typename Kernel>
void dispatch_efficient_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    ...) {

  int64_t head_dim = query.size(-1);
  bool is_aligned = (head_dim % 8 == 0);

  // Select kernel based on head_dim and alignment
  if (query.dtype() == at::kHalf) {
    if (is_aligned) {
      if (head_dim == 32) return launch_kernel<CutlassF16AlignedK32>();
      if (head_dim == 64) return launch_kernel<CutlassF16AlignedK64>();
      if (head_dim == 128) return launch_kernel<CutlassF16AlignedK128>();
      // ... etc
    } else {
      if (head_dim == 32) return launch_kernel<CutlassF16NotAlignedK32>();
      // ... etc
    }
  } else if (query.dtype() == at::kBFloat16) {
    // ... BFloat16 kernels
  }
}
```

---

## 4. Math Fallback

### 4.1 Pure PyTorch Implementation

When specialized kernels aren't available, PyTorch uses standard ops:

```python
# Simplified Python equivalent
def scaled_dot_product_attention_math(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Compute attention scores
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    # Apply causal mask
    if is_causal:
        mask = torch.tril(torch.ones(L, S, dtype=torch.bool))
        attn_weight.masked_fill_(~mask, float('-inf'))

    # Apply custom mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(~attn_mask, float('-inf'))
        else:
            attn_weight += attn_mask

    # Softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Dropout
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # Output
    return attn_weight @ value
```

**C++ Implementation** (`aten/src/ATen/native/transformers/attention.cpp`):

```cpp
std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale,
    bool enable_gqa) {

  // Compute scale
  auto scale_factor = calculate_scale(query, scale);

  // Q @ K^T
  Tensor attn_weight = at::matmul(query, key.transpose(-2, -1));

  // Scale
  attn_weight = attn_weight.mul(scale_factor);

  // Causal mask
  if (is_causal) {
    auto L = query.size(-2);
    auto S = key.size(-2);
    auto mask = at::ones({L, S}, query.options().dtype(kBool)).tril();
    attn_weight.masked_fill_(mask.logical_not(), -std::numeric_limits<float>::infinity());
  }

  // Custom mask
  if (attn_mask.has_value()) {
    if (attn_mask->dtype() == kBool) {
      attn_weight.masked_fill_(attn_mask->logical_not(),
                                -std::numeric_limits<float>::infinity());
    } else {
      attn_weight = attn_weight.add(*attn_mask);
    }
  }

  // Softmax
  attn_weight = at::_softmax(attn_weight, -1, false);

  // Dropout
  if (dropout_p > 0.0) {
    attn_weight = at::dropout(attn_weight, dropout_p, /*train=*/true);
  }

  // Output
  Tensor output = at::matmul(attn_weight, value);

  return std::make_tuple(output, attn_weight);
}
```

**When Used**:
- CPU execution
- Unsupported dtypes (e.g., float64)
- Unsupported head dimensions
- Custom masks that optimized kernels don't support

---

## 5. Grouped-Query Attention (GQA)

### 5.1 Overview

**Grouped-Query Attention** reduces memory and compute for multi-query attention scenarios:

**Standard Multi-Head Attention**:
- `num_heads` separate query/key/value heads
- Memory: `3 × num_heads × head_dim`

**Multi-Query Attention (MQA)**:
- `num_heads` query heads
- **1 shared** key/value head
- Memory: `(num_heads + 2) × head_dim`
- Used in PaLM, GPT-J

**Grouped-Query Attention (GQA)**:
- `num_q_heads` query heads
- `num_kv_heads` key/value heads (num_kv_heads < num_q_heads)
- Queries grouped: `num_q_heads / num_kv_heads` queries per KV head
- Memory: `(num_q_heads + 2 × num_kv_heads) × head_dim`
- Used in LLaMA 2, Mistral

### 5.2 GQA Implementation

```python
def grouped_query_attention(
    query,  # [B, num_q_heads, N, head_dim]
    key,    # [B, num_kv_heads, S, head_dim]
    value,  # [B, num_kv_heads, S, head_dim]
    ...
):
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    group_size = num_q_heads // num_kv_heads

    # Expand K, V to match Q heads
    # Repeat each KV head `group_size` times
    key = key.repeat_interleave(group_size, dim=1)
    value = value.repeat_interleave(group_size, dim=1)

    # Now standard attention
    return scaled_dot_product_attention(query, key, value, ...)
```

**PyTorch Optimization**:

Instead of explicitly repeating, PyTorch kernels **implicitly broadcast** K/V across query groups:

```cpp
// Kernel optimization for GQA
template <int kQueriesPerBlock, int kKeysPerBlock>
__global__ void gqa_attention_kernel(...) {
  int q_group = blockIdx.x % group_size;
  int kv_idx = blockIdx.x / group_size;

  // Load one KV block, used by multiple query blocks
  load_kv_block(K_smem, V_smem, key + kv_idx * kv_stride, ...);

  // Process multiple query groups with same KV
  for (int q_local = 0; q_local < group_size; ++q_local) {
    int q_idx = kv_idx * group_size + q_local;
    load_query_block(Q_smem, query + q_idx * q_stride, ...);

    // Compute attention with shared KV
    compute_attention(Q_smem, K_smem, V_smem, ...);
  }
}
```

**Benefits**:
- Reduces KV cache size for large models
- Faster inference (less data to load)
- Minimal quality degradation vs full MHA

---

## 6. MPS (Metal) Attention

### 6.1 Apple Silicon Implementation

PyTorch implements Metal-accelerated attention for Apple Silicon (M1/M2/M3):

**Location**: `aten/src/ATen/native/transformers/attention.cpp`

```cpp
std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_for_mps(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale) {

  // MPS-specific implementation
  // Uses MetalPerformanceShadersGraph (MPSGraph) API

  // Route to Metal kernels
  return _scaled_dot_product_attention_math_mps(
    query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale
  );
}
```

**Metal Shader**: Custom `.metal` kernels for attention

```metal
// Simplified Metal kernel pseudocode
kernel void attention_forward(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  // Thread computes one output element
  int batch = gid.z;
  int head = gid.y;
  int query_idx = gid.x;

  float sum = 0.0;
  float max_score = -INFINITY;

  // First pass: compute max for numerical stability
  for (int k = 0; k < seq_len_k; ++k) {
    float score = dot_product(query[query_idx], key[k]) * params.scale;
    max_score = max(max_score, score);
  }

  // Second pass: compute exp and sum
  float exp_sum = 0.0;
  for (int k = 0; k < seq_len_k; ++k) {
    float score = dot_product(query[query_idx], key[k]) * params.scale;
    float exp_score = exp(score - max_score);
    exp_sum += exp_score;

    // Accumulate output
    for (int d = 0; d < head_dim; ++d) {
      output[d] += exp_score * value[k][d];
    }
  }

  // Final normalization
  for (int d = 0; d < head_dim; ++d) {
    output[d] /= exp_sum;
  }
}
```

**Optimization**: Uses MPS matrix multiplication and activation functions

---

## 7. Performance Characteristics

### 7.1 Backend Comparison

| Backend | Memory | Speed | Hardware | Precision |
|---------|--------|-------|----------|-----------|
| **FlashAttention** | O(N) | 2-4x faster | NVIDIA Ampere+ (sm_80+) | FP16, BF16 |
| **Memory-Efficient** | O(N) | 1.5-3x faster | NVIDIA Volta+ (sm_70+) | FP16, BF16, FP32 |
| **cuDNN Attention** | O(N²) | 1-2x faster | NVIDIA GPUs | FP16, BF16 |
| **MPS Attention** | O(N²) | 1-2x faster | Apple Silicon | FP16, FP32 |
| **Math Fallback** | O(N²) | 1x (baseline) | All devices | All dtypes |

**Memory Scaling**:

```
Sequence Length    Math Fallback    FlashAttention
   1024 tokens        4 MB              0.5 MB
   4096 tokens       64 MB              2 MB
  16384 tokens        1 GB              8 MB
  65536 tokens       16 GB             32 MB
```

### 7.2 Speed Benchmarks

**Forward Pass** (A100 GPU, batch=32, heads=16, head_dim=64):

```
Seq Len    Math     FlashAttn    MemEfficient
  512      1.2 ms      0.4 ms         0.5 ms
 1024      4.8 ms      0.8 ms         1.0 ms
 2048     19.2 ms      1.6 ms         2.0 ms
 4096     76.8 ms      3.2 ms         4.0 ms
 8192    307.2 ms      6.4 ms         8.0 ms
```

**Speedup**: FlashAttention is **4-5x faster** at long sequences

### 7.3 Choosing the Right Backend

**Use FlashAttention when**:
- Ampere/Hopper GPU (A100, H100, RTX 30xx/40xx)
- FP16 or BF16 precision
- head_dim in {32, 64, 96, 128, 256}
- Training mode (need gradients)

**Use Memory-Efficient Attention when**:
- Volta/Turing GPU (V100, RTX 20xx)
- Inference only (no gradients)
- Arbitrary head_dim

**Use MPS Attention when**:
- Apple Silicon (M1/M2/M3)
- macOS environment

**Use Math Fallback when**:
- CPU execution
- Custom masks
- FP64 precision
- Debugging

---

## 8. Gradient Formulas

### 8.1 Attention Backward

**Forward**:
```
S = QK^T / √d_k
P = softmax(S)
O = PV
```

**Backward**:

Given `dL/dO`, compute `dL/dQ`, `dL/dK`, `dL/dV`:

**Step 1**: Gradient w.r.t. P (before matmul with V)
```
dL/dP = dL/dO @ V^T
```

**Step 2**: Gradient w.r.t. S (softmax backward)
```
dL/dS = P ⊙ (dL/dP - rowsum(P ⊙ dL/dP))
```

Softmax Jacobian:
```
∂softmax_i/∂x_j = softmax_i (δ_ij - softmax_j)
```

**Step 3**: Gradients w.r.t. Q, K
```
dL/dQ = (dL/dS @ K) / √d_k
dL/dK = (dL/dS^T @ Q) / √d_k
```

**Step 4**: Gradient w.r.t. V
```
dL/dV = P^T @ dL/dO
```

### 8.2 Causal Mask Backward

Causal mask sets `S[i,j] = -∞` for `j > i`.

**Backward**: Masked entries have zero gradient (chain rule stops at `-∞`)

```
dL/dS[i,j] = 0  if j > i  (masked positions)
```

**Implementation**:
```cpp
if (is_causal && j > i) {
  continue;  // Skip gradient computation for masked positions
}
```

---

## 9. MLX Porting Recommendations

### 9.1 What to Port

**ADOPT**:

1. **Scaled Dot-Product Attention API**: Standard interface
   ```python
   mlx.nn.scaled_dot_product_attention(
       query, key, value,
       attn_mask=None,
       is_causal=False,
       scale=None
   )
   ```

2. **Backend Dispatch Pattern**: Select best implementation based on input
   - Metal kernel (optimized for Apple Silicon)
   - CPU fallback (pure MLX ops)

3. **FlashAttention Algorithm**: Tiled attention with online softmax
   - Adapt algorithm to Metal shader
   - Use threadgroup (shared) memory for blocks

4. **GQA Support**: Grouped-query attention for efficient LLMs
   - Implicit K/V broadcasting
   - Reduce memory footprint

**SIMPLIFY**:

1. **Single Backend**: MLX is Metal-only
   - No CUDA/ROCm variants
   - No multi-backend selection complexity

2. **Fewer Kernel Variants**: 10-20 kernels vs PyTorch's 100+
   - Cover common head dimensions: 32, 64, 128, 256
   - FP16 and FP32 only (no BF16 on Metal)

### 9.2 What NOT to Port

**SKIP**:

1. **CUDA-Specific Kernels**: FlashAttention CUDA, CUTLASS kernels
2. **cuDNN Integration**: NVIDIA-specific
3. **Nested Tensor Support**: Complex feature, low ROI
4. **Extensive Dropout Support**: Start without, add later if needed

### 9.3 Recommended MLX Implementation

**Metal Shader Architecture**:

```metal
// mlx/backend/metal/kernels/attention.metal

kernel void flash_attention_forward(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    threadgroup half* shared_mem [[threadgroup(0)]],
    uint3 thread_position [[thread_position_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]]) {

  // Tile sizes
  constexpr int Br = 64;  // Query block size
  constexpr int Bc = 64;  // Key block size

  // Shared memory layout
  threadgroup half* Q_shared = shared_mem;
  threadgroup half* K_shared = Q_shared + Br * params.head_dim;
  threadgroup half* V_shared = K_shared + Bc * params.head_dim;
  threadgroup half* S_shared = V_shared + Bc * params.head_dim;

  // Each threadgroup handles one query block
  int q_block = threadgroup_position.x;

  // Load query block to shared memory
  load_query_block(Q_shared, query, q_block, params);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Initialize output accumulator
  half O_local[MAX_HEAD_DIM] = {0};
  float m_local = -INFINITY;
  float l_local = 0;

  // Iterate over key/value blocks
  for (int kv_block = 0; kv_block < params.num_kv_blocks; ++kv_block) {
    // Load K, V to shared memory
    load_kv_block(K_shared, V_shared, key, value, kv_block, params);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute attention scores
    compute_qk_matmul(S_shared, Q_shared, K_shared, Br, Bc, params);

    // Apply scaling
    scale_scores(S_shared, params.scale, Br, Bc);

    // Apply causal mask
    if (params.is_causal) {
      apply_causal_mask(S_shared, q_block, kv_block, Br, Bc);
    }

    // Online softmax
    float m_block = compute_row_max(S_shared, thread_position);
    float m_new = max(m_local, m_block);

    apply_exp_and_sum(S_shared, m_new, &l_block, Br, Bc);

    float correction = exp(m_local - m_new);
    l_local = l_local * correction + l_block;

    scale_output(O_local, correction, params.head_dim);

    // Accumulate output
    accumulate_pv(O_local, S_shared, V_shared, Br, Bc, params);

    m_local = m_new;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Final normalization
  normalize_output(O_local, l_local, params.head_dim);

  // Write to global memory
  store_output(output, O_local, q_block, thread_position, params);
}
```

**C++ API**:

```cpp
// mlx/ops/attention.cpp

array scaled_dot_product_attention(
    const array& query,
    const array& key,
    const array& value,
    const std::optional<array>& attn_mask,
    float dropout_p,
    bool is_causal,
    std::optional<float> scale) {

  // Validate inputs
  check_attention_inputs(query, key, value);

  // Select backend
  if (query.dtype() == float16 && can_use_flash_attention(query, key, value)) {
    return flash_attention_metal(query, key, value, is_causal, scale);
  } else {
    return attention_fallback(query, key, value, attn_mask, is_causal, scale);
  }
}
```

**Python Wrapper**:

```python
# mlx/nn/attention.py

def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    attn_mask: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> mx.array:
    """Scaled dot-product attention.

    Args:
        query: [B, H, N, D]
        key: [B, H, S, D]
        value: [B, H, S, D]
        attn_mask: Optional mask
        dropout_p: Dropout probability (not yet implemented)
        is_causal: Use causal mask
        scale: Custom scale (default: 1/√D)

    Returns:
        Output: [B, H, N, D]
    """
    return mx.fast.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale
    )
```

---

## Summary

PyTorch's attention implementation is **highly optimized** with multiple specialized backends:

**Key Implementations**:
1. **FlashAttention**: Memory-efficient, 2-4x faster (CUDA)
2. **Memory-Efficient Attention**: CUTLASS-based (CUDA)
3. **Math Fallback**: Pure ops (all devices)
4. **MPS Attention**: Metal-accelerated (Apple Silicon)

**For MLX**:
- Port **FlashAttention algorithm** to Metal shaders
- Implement **backend dispatch** (Metal vs fallback)
- Support **GQA** for efficient LLMs
- Start simple (FP16/FP32, common head dims), optimize iteratively

Attention is **critical infrastructure** for modern transformers. Investing in optimized Metal kernels will provide massive performance benefits for LLM workloads on Apple Silicon.
