# Adam Optimizer Family: PyTorch Implementation & MLX Porting Guide

## Overview

The Adam (Adaptive Moment Estimation) optimizer family represents the most widely-used optimization algorithms in deep learning. This document covers three main variants:

1. **Adam** - Original adaptive learning rate optimizer
2. **AdamW** - Adam with decoupled weight decay
3. **Adamax** - Adam variant using infinity norm

All three optimizers use adaptive learning rates based on first and second moment estimates of gradients, enabling faster convergence and better performance across diverse model architectures.

---

## Table of Contents

- [Adam Algorithm](#adam-algorithm)
- [AdamW (Decoupled Weight Decay)](#adamw-decoupled-weight-decay)
- [Adamax (Infinity Norm)](#adamax-infinity-norm)
- [Implementation Details](#implementation-details)
- [State Management](#state-management)
- [Optimization Variants](#optimization-variants)
- [MLX Porting Recommendations](#mlx-porting-recommendations)

---

## Adam Algorithm

### Mathematical Formulation

Adam maintains exponential moving averages of both the gradient (first moment `m`) and the squared gradient (second moment `v`).

**Algorithm** (from `adam.py:275-309`):

```
Input: θ₀ (initial parameters), f(θ) (objective function)
Hyperparameters: α (learning rate, default: 1e-3)
                 β₁, β₂ (exponential decay rates, default: 0.9, 0.999)
                 ε (numerical stability constant, default: 1e-8)
                 λ (weight decay, default: 0)
                 amsgrad (use AMSGrad variant, default: False)

Initialize: m₀ ← 0 (first moment vector)
            v₀ ← 0 (second moment vector)
            v̂₀ ← 0 (max of second moments, if amsgrad)

For t = 1 to ... do:
    g_t ← ∇_θ f_t(θ_{t-1})                    # Compute gradient

    if maximize:
        g_t ← -g_t                              # For maximization

    if λ ≠ 0:
        g_t ← g_t + λ θ_{t-1}                   # L2 regularization

    m_t ← β₁ m_{t-1} + (1 - β₁) g_t            # Update biased first moment
    v_t ← β₂ v_{t-1} + (1 - β₂) g_t²           # Update biased second moment

    m̂_t ← m_t / (1 - β₁^t)                     # Bias-corrected first moment

    if amsgrad:
        v̂_t ← max(v̂_{t-1}, v_t)               # Maintain max of second moments
        v̂_t ← v̂_t / (1 - β₂^t)                 # Bias-corrected max second moment
    else:
        v̂_t ← v_t / (1 - β₂^t)                 # Bias-corrected second moment

    θ_t ← θ_{t-1} - α · m̂_t / (√v̂_t + ε)     # Parameter update

Return θ_t
```

**Key Properties**:
- **Adaptive learning rates**: Each parameter has its own effective learning rate
- **Momentum**: First moment `m` provides directional averaging
- **RMSProp-like scaling**: Second moment `v` normalizes by gradient magnitude
- **Bias correction**: Accounts for initialization at zero

### PyTorch API

**Class Definition** (`adam.py:33-101`):

```python
torch.optim.Adam(
    params,                      # Iterable of parameters or param groups
    lr=1e-3,                     # Learning rate (float or Tensor)
    betas=(0.9, 0.999),          # Coefficients (β₁, β₂) for moment estimates
    eps=1e-8,                    # Term for numerical stability
    weight_decay=0,              # L2 penalty coefficient
    amsgrad=False,               # Use AMSGrad variant
    foreach=None,                # Use foreach (multi-tensor) implementation
    maximize=False,              # Maximize objective instead of minimize
    capturable=False,            # Enable CUDA graph capture
    differentiable=False,        # Allow differentiation through optimizer
    fused=None,                  # Use fused kernel (CUDA only)
    decoupled_weight_decay=False # Use AdamW-style weight decay
)
```

**Parameters**:
- `betas`: Tuple `(β₁, β₂)` controls exponential decay of moment estimates
  - `β₁`: First moment decay (typical: 0.9)
  - `β₂`: Second moment decay (typical: 0.999)
- `eps`: Added to denominator for numerical stability (prevents division by zero)
- `weight_decay`: When `decoupled_weight_decay=False`, adds `λ·θ` to gradient
- `amsgrad`: If True, maintains maximum of second moments for better convergence

### Core Update Implementation

**Single-Tensor Update** (`adam.py:395-548`):

```python
def _single_tensor_adam(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
    *, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize,
    capturable, differentiable, decoupled_weight_decay):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]          # First moment estimate (m)
        exp_avg_sq = exp_avg_sqs[i]    # Second moment estimate (v)
        step_t = state_steps[i]

        # Increment step count
        step_t += 1

        # Weight decay
        if weight_decay != 0:
            if decoupled_weight_decay:
                # AdamW: θ ← θ(1 - α·λ)  (decoupled from gradient)
                param.mul_(1 - lr * weight_decay)
            else:
                # Adam: g ← g + λ·θ  (couples with gradient)
                grad = grad.add(param, alpha=weight_decay)

        # Handle complex parameters
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # Update first moment: m_t = β₁·m_{t-1} + (1-β₁)·g_t
        exp_avg.lerp_(grad, 1 - beta1)

        # Update second moment: v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            # Tensor-based bias correction (for CUDA graphs)
            bias_correction1 = 1 - beta1 ** step_t
            bias_correction2 = 1 - beta2 ** step_t

            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # AMSGrad: v̂_t = max(v̂_{t-1}, v_t)
                max_exp_avg_sqs[i].copy_(torch.maximum(
                    max_exp_avg_sqs[i], exp_avg_sq
                ))

                denom = (
                    max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt
                ).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            # Update: θ ← θ - α·m̂/√v̂
            param.addcdiv_(exp_avg, denom, value=-step_size)
        else:
            # Scalar-based bias correction (faster for non-capturable)
            step = step_t.item()  # Convert to Python scalar

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2 ** 0.5

            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq,
                             out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)
```

**Key Implementation Details**:

1. **Bias Correction**: Accounts for initialization at zero
   - Without correction, early updates would be biased toward zero
   - Correction factor: `1 / (1 - β^t)` grows from ∞ to 1 as `t` increases
   - After ~1000 steps, correction is negligible (<0.1% effect)

2. **Linear Interpolation (`lerp`)**: `exp_avg.lerp_(grad, 1 - beta1)`
   - Equivalent to: `exp_avg = beta1 * exp_avg + (1 - beta1) * grad`
   - Single fused operation, more efficient than separate mul/add

3. **In-place Operations**: Minimize memory allocations
   - `mul_()`, `addcmul_()`, `addcdiv_()` modify tensors in-place
   - Critical for memory efficiency with large models

4. **Complex Parameter Handling**: Use `view_as_real` to treat complex tensors as real

---

## AdamW (Decoupled Weight Decay)

### Algorithm

AdamW fixes a flaw in the original Adam: **weight decay should not accumulate in momentum**.

**Key Difference** (`adamw.py:59-93`):

```
Standard Adam with weight decay:
    g_t ← g_t + λ θ_{t-1}           # Add weight decay to gradient
    m_t ← β₁ m_{t-1} + (1 - β₁) g_t  # Weight decay accumulates in momentum
    ...

AdamW (decoupled weight decay):
    θ_t ← θ_{t-1} (1 - α λ)          # Apply weight decay directly to parameters
    m_t ← β₁ m_{t-1} + (1 - β₁) g_t  # Momentum unaffected by weight decay
    v_t ← β₂ v_{t-1} + (1 - β₂) g_t²
    θ_t ← θ_t - α · m̂_t / (√v̂_t + ε)
```

**Why AdamW is Better**:

1. **Cleaner separation**: Weight decay acts as true regularization, not gradient modification
2. **Better generalization**: Empirically performs better on many tasks
3. **Interpretable λ**: Weight decay strength independent of learning rate

### PyTorch API

**Class Definition** (`adamw.py:19-48`):

```python
torch.optim.AdamW(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2,    # Default: 0.01 (higher than Adam's 0)
    amsgrad=False,
    maximize=False,
    foreach=None,
    capturable=False,
    differentiable=False,
    fused=None
)
```

**Implementation**: AdamW is implemented as a thin wrapper around Adam with `decoupled_weight_decay=True`:

```python
class AdamW(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, **kwargs):
        super().__init__(
            params, lr, betas, eps, weight_decay, amsgrad,
            decoupled_weight_decay=True,  # Key difference!
            **kwargs
        )
```

**When to Use AdamW vs Adam**:
- **AdamW**: Default choice for most modern architectures (transformers, vision models)
- **Adam**: Legacy codebases, specific architectures where it performs better

---

## Adamax (Infinity Norm)

### Algorithm

Adamax replaces the second moment estimate with an **infinity norm** of gradients.

**Mathematical Formulation** (`adamax.py:177-200`):

```
Initialize: m₀ ← 0 (first moment)
            u₀ ← 0 (infinity norm)

For t = 1 to ... do:
    g_t ← ∇_θ f_t(θ_{t-1})

    if λ ≠ 0:
        g_t ← g_t + λ θ_{t-1}

    m_t ← β₁ m_{t-1} + (1 - β₁) g_t           # First moment (same as Adam)
    u_t ← max(β₂ u_{t-1}, |g_t| + ε)          # Infinity norm (different!)

    θ_t ← θ_{t-1} - (α / (1 - β₁^t)) · m_t / u_t

Return θ_t
```

**Key Differences from Adam**:

1. **No square root**: Simpler denominator computation
2. **Max operation**: `u_t = max(β₂·u_{t-1}, |g_t|)` instead of exponential average
3. **No second bias correction**: Infinity norm doesn't need it
4. **Different default LR**: 2e-3 instead of 1e-3 (due to different scaling)

### PyTorch API

**Class Definition** (`adamax.py:28-65`):

```python
torch.optim.Adamax(
    params,
    lr=2e-3,              # Higher default than Adam
    betas=(0.9, 0.999),   # Same as Adam
    eps=1e-8,
    weight_decay=0,
    foreach=None,
    maximize=False,
    differentiable=False,
    capturable=False
)
```

### Core Update Implementation

**Single-Tensor Update** (`adamax.py:225-303`):

```python
def _single_tensor_adamax(params, grads, exp_avgs, exp_infs, state_steps,
                          *, eps, beta1, beta2, lr, weight_decay,
                          maximize, differentiable, capturable):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]     # First moment (m)
        exp_inf = exp_infs[i]     # Infinity norm (u)
        step_t = state_steps[i]

        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_inf = torch.view_as_real(exp_inf)

        # Update first moment: m_t = β₁·m_{t-1} + (1-β₁)·g_t
        exp_avg.lerp_(grad, 1 - beta1)

        # Update infinity norm: u_t = max(β₂·u_{t-1}, |g_t| + ε)
        if not differentiable:
            torch.maximum(
                exp_inf.mul_(beta2),
                grad.abs().add_(eps),
                out=exp_inf
            )
        else:
            # Differentiable version (for second-order optimization)
            norm_buf = torch.cat([
                exp_inf.mul_(beta2).unsqueeze(0),
                grad.abs().add_(eps).unsqueeze(0)
            ], 0)
            exp_inf.copy_(torch.amax(norm_buf, 0, keepdim=False))

        if capturable:
            bias_correction = 1 - beta1 ** step_t
            clr = lr / bias_correction
            param.addcdiv_(exp_avg, exp_inf, value=-clr)
        else:
            step = step_t.item()
            bias_correction = 1 - beta1 ** step
            clr = lr / bias_correction
            param.addcdiv_(exp_avg, exp_inf, value=-clr)
```

**When to Use Adamax**:
- Sparse gradients or embeddings
- When gradient magnitudes vary widely
- Alternatives to Adam for specific tasks (NLP, RL)

---

## Implementation Details

### State Management

**Optimizer State** (`adam.py:159-189`):

Each parameter tracked by the optimizer maintains:

```python
state = {
    'step': Tensor | float,        # Step count (for bias correction)
    'exp_avg': Tensor,             # First moment estimate (m_t)
    'exp_avg_sq': Tensor,          # Second moment estimate (v_t)
    'max_exp_avg_sq': Tensor       # Max second moment (if amsgrad=True)
}
```

**Lazy Initialization**: State is created on first `step()` call:

```python
if len(state) == 0:
    # Decide where to host 'step' counter
    state["step"] = (
        torch.zeros((), dtype=torch.float32, device=param.device)
        if capturable or fused
        else torch.tensor(0.0, dtype=torch.float32)  # On CPU for efficiency
    )

    # Initialize moments to zero
    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    if amsgrad:
        state["max_exp_avg_sq"] = torch.zeros_like(param)
```

**Why CPU for `step`?** (`adam.py:165-167`)
- Kernel launches are expensive on CUDA/XLA
- Storing step on CPU avoids device sync unless `capturable=True`

### Multi-Tensor (Foreach) Implementation

**Purpose**: Process all parameters in parallel for better GPU utilization.

**Single-Tensor vs Multi-Tensor**:

```python
# Single-tensor: Sequential processing
for param in params:
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    # ...

# Multi-tensor: Vectorized processing
torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
torch._foreach_mul_(exp_avg_sqs, beta2)
torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)
# ...
```

**Benefits**:
- Single kernel launch for multiple tensors
- Better GPU occupancy
- Reduced Python overhead

**Tradeoffs**:
- Requires all tensors on same device
- Not compatible with `differentiable=True`

### Fused Kernel Implementation

**Purpose**: Combine all Adam operations into a single CUDA kernel.

**Fused vs Unfused**:

```cpp
// Unfused (PyTorch default):
exp_avg.lerp_(grad, 1 - beta1)                        // Kernel 1
exp_avg_sq.mul_(beta2)                                // Kernel 2
exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)      // Kernel 3
denom = exp_avg_sq.sqrt()                             // Kernel 4
param.addcdiv_(exp_avg, denom, value=-step_size)      // Kernel 5

// Fused (CUDA only):
adam_fused_kernel<<<...>>>(
    param, grad, exp_avg, exp_avg_sq, step,
    beta1, beta2, lr, eps, weight_decay
);  // Single kernel launch!
```

**Benefits of Fused**:
- Minimize memory traffic (read param/grad once)
- Fewer kernel launches
- 2-3x speedup on CUDA

**Limitations**:
- CUDA only (no CPU/MPS implementation)
- Not compatible with `differentiable=True` or `foreach=True`

### AMSGrad Variant

**Motivation**: Standard Adam can fail to converge in some cases (proven theoretically).

**Fix**: Maintain maximum of all past second moment estimates:

```python
# Standard Adam:
v̂_t = v_t / (1 - β₂^t)
θ_t ← θ_{t-1} - α · m̂_t / (√v̂_t + ε)

# AMSGrad:
v̂_t = max(v̂_{t-1}, v_t) / (1 - β₂^t)  # Use max!
θ_t ← θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**Effect**:
- Prevents second moment from decreasing
- More conservative updates
- Theoretical convergence guarantees

**In Practice**:
- Marginal improvement in most cases
- Slightly slower per-step
- Not widely used (AdamW more popular)

---

## MLX Porting Recommendations

### 1. Adam Core Implementation

**MLX C++ API** (`mlx/optimizers/adam.h`):

```cpp
namespace mlx::optimizers {

class Adam : public Optimizer {
public:
  Adam(
      float learning_rate = 1e-3,
      std::pair<float, float> betas = {0.9, 0.999},
      float eps = 1e-8,
      float weight_decay = 0.0,
      bool amsgrad = false,
      bool decoupled_weight_decay = false
  );

  void step(const std::vector<array>& parameters,
            const std::vector<array>& gradients) override;

private:
  float lr_;
  float beta1_, beta2_;
  float eps_;
  float weight_decay_;
  bool amsgrad_;
  bool decoupled_weight_decay_;

  // State per parameter
  std::unordered_map<void*, AdamState> state_;
};

struct AdamState {
  int step = 0;
  array exp_avg;      // First moment
  array exp_avg_sq;   // Second moment
  array max_exp_avg_sq;  // For AMSGrad
};

} // namespace mlx::optimizers
```

**Implementation**:

```cpp
void Adam::step(const std::vector<array>& parameters,
                const std::vector<array>& gradients) {
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto& param = parameters[i];
    auto& grad = gradients[i];

    // Get or create state
    auto& state = state_[param.data()];
    if (state.step == 0) {
      state.exp_avg = mlx::core::zeros_like(param);
      state.exp_avg_sq = mlx::core::zeros_like(param);
      if (amsgrad_) {
        state.max_exp_avg_sq = mlx::core::zeros_like(param);
      }
    }

    state.step++;

    // Weight decay
    array grad_wd = grad;
    if (weight_decay_ > 0) {
      if (decoupled_weight_decay_) {
        // AdamW: θ ← θ(1 - α·λ)
        param = param * (1.0f - lr_ * weight_decay_);
      } else {
        // Adam: g ← g + λ·θ
        grad_wd = grad + weight_decay_ * param;
      }
    }

    // Update moments
    state.exp_avg = beta1_ * state.exp_avg + (1 - beta1_) * grad_wd;
    state.exp_avg_sq = beta2_ * state.exp_avg_sq +
                       (1 - beta2_) * mlx::core::square(grad_wd);

    // Bias correction
    float bias_correction1 = 1.0f - std::pow(beta1_, state.step);
    float bias_correction2 = 1.0f - std::pow(beta2_, state.step);

    float step_size = lr_ / bias_correction1;

    // Compute denominator
    array denom;
    if (amsgrad_) {
      state.max_exp_avg_sq = mlx::core::maximum(
          state.max_exp_avg_sq, state.exp_avg_sq
      );
      denom = mlx::core::sqrt(state.max_exp_avg_sq / bias_correction2) + eps_;
    } else {
      denom = mlx::core::sqrt(state.exp_avg_sq / bias_correction2) + eps_;
    }

    // Update parameters: θ ← θ - α · m̂ / √v̂
    param = param - step_size * (state.exp_avg / denom);
  }
}
```

### 2. Metal Shader (Fused Kernel)

**Optimized Metal Implementation**:

```metal
kernel void adam_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* exp_avg [[buffer(2)]],
    device float* exp_avg_sq [[buffer(3)]],
    device float* max_exp_avg_sq [[buffer(4)]],  // nullable
    constant AdamParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {

    const float beta1 = params.beta1;
    const float beta2 = params.beta2;
    const float lr = params.lr;
    const float eps = params.eps;
    const float weight_decay = params.weight_decay;
    const int step = params.step;
    const bool amsgrad = params.amsgrad;
    const bool decoupled_wd = params.decoupled_weight_decay;

    if (tid >= params.num_elements) return;

    float p = param[tid];
    float g = grad[tid];
    float m = exp_avg[tid];
    float v = exp_avg_sq[tid];

    // Weight decay
    if (weight_decay > 0) {
        if (decoupled_wd) {
            p *= (1.0f - lr * weight_decay);  // AdamW
        } else {
            g += weight_decay * p;            // Adam
        }
    }

    // Update moments
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g * g;

    // Bias correction
    float bias_correction1 = 1.0f - pow(beta1, step);
    float bias_correction2 = 1.0f - pow(beta2, step);

    float step_size = lr / bias_correction1;

    // Compute update
    float denom;
    if (amsgrad) {
        float v_max = max_exp_avg_sq[tid];
        v_max = max(v_max, v);
        max_exp_avg_sq[tid] = v_max;
        denom = sqrt(v_max / bias_correction2) + eps;
    } else {
        denom = sqrt(v / bias_correction2) + eps;
    }

    // Apply update
    p -= step_size * m / denom;

    // Write back
    param[tid] = p;
    exp_avg[tid] = m;
    exp_avg_sq[tid] = v;
}
```

### 3. Python API

**MLX Python Wrapper**:

```python
class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}

    def step(self, parameters: list, gradients: list):
        # Call Metal/C++ implementation
        mlx.optimizers.adam_step(
            parameters,
            gradients,
            self.state,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )

class AdamW(Adam):
    """AdamW with decoupled weight decay."""
    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        super().__init__(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False
        )
        self.decoupled_weight_decay = True
```

### 4. Optimization Strategies

**Memory Efficiency**:

1. **Lazy state allocation**: Only create state when parameter receives gradient
2. **In-place updates**: Minimize temporary allocations
3. **Metal shared memory**: Use threadgroup memory for reductions

**Performance**:

1. **Fused kernel**: Single Metal kernel for all operations
2. **Vectorized ops**: Use SIMD for 4-wide float operations
3. **Async execution**: Launch kernel asynchronously, sync only when needed

**Testing**:

1. **Numerical correctness**: Compare against PyTorch with `atol=1e-5`
2. **Gradient checking**: Verify with finite differences
3. **Convergence tests**: Train small models, compare loss curves

---

## Summary

This document provides comprehensive reverse-engineering of the Adam optimizer family:

**Key Algorithms**:
- **Adam**: Adaptive learning rates using first and second moment estimates
- **AdamW**: Adam with decoupled weight decay (preferred for modern architectures)
- **Adamax**: Adam variant using infinity norm (for sparse/embedding tasks)

**Implementation Insights**:
- Bias correction essential for early training
- Decoupled weight decay improves generalization
- Fused kernels provide 2-3x speedup
- Multi-tensor (foreach) improves GPU utilization

**MLX Porting Priorities**:
1. **High Priority**: Adam and AdamW (most widely used)
2. **Medium Priority**: AMSGrad variant
3. **Lower Priority**: Adamax (specialized use cases)

**Next Steps**: Document SGD optimizer with momentum and Nesterov variants.
