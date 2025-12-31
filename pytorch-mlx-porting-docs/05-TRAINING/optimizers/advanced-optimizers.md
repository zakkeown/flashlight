# Advanced Optimizers: LBFGS, NAdam, RAdam, Adagrad

## Overview

This document covers PyTorch's advanced optimization algorithms that extend beyond the commonly used SGD, Adam, AdamW, and RMSprop. These optimizers target specific use cases where standard methods struggle:

- **LBFGS**: Second-order quasi-Newton method for small-batch, full-batch optimization
- **NAdam**: Nesterov-accelerated Adam combining Adam's adaptive rates with Nesterov momentum
- **RAdam**: Rectified Adam addressing the variance warm-up problem
- **Adagrad**: Adaptive learning rates with cumulative squared gradients (foundational algorithm)

**When to Use These**:
- **LBFGS**: Small models, full-batch training, scientific computing
- **NAdam**: Transformers, NLP tasks where Adam is used
- **RAdam**: When Adam training is unstable in early steps
- **Adagrad**: Sparse gradients, feature learning (less common now)

---

## 1. LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)

### Algorithm Description

LBFGS is a **second-order optimization method** that approximates the Newton's method update using a limited history of gradient differences. Unlike first-order methods (SGD, Adam), LBFGS uses curvature information.

**Newton's Method** (full second-order):
```
θ_{t+1} = θ_t - H^{-1} ∇L(θ_t)
```

Where `H` is the Hessian matrix (n×n for n parameters).

**LBFGS Approximation**:
Instead of computing the full Hessian, LBFGS maintains a history of `m` gradient and parameter differences to approximate `H^{-1}` using the two-loop recursion algorithm.

### Mathematical Formulation

**State Variables**:
- `s_k = θ_k - θ_{k-1}` (parameter difference)
- `y_k = ∇L(θ_k) - ∇L(θ_{k-1})` (gradient difference)
- History size: `m` (typically 100)

**Two-Loop Recursion** (Nocedal & Wright, 2006):

```python
# Initialize
q = ∇L(θ_t)
α = []

# First loop (backward)
for i in range(t-1, max(t-m, 0), -1):
    ρ_i = 1 / (y_i · s_i)
    α_i = ρ_i (s_i · q)
    q = q - α_i y_i
    α.append(α_i)

# Scale
γ = (s_{t-1} · y_{t-1}) / (y_{t-1} · y_{t-1})
r = γ q

# Second loop (forward)
for i in range(max(t-m, 0), t):
    β = ρ_i (y_i · r)
    r = r + s_i (α_i - β)

# Search direction
p = -r
```

**Line Search**: LBFGS requires a line search to find optimal step size:

```
θ_{t+1} = θ_t + α p
```

Where `α` is found using Strong Wolfe conditions:
1. **Sufficient decrease** (Armijo): `L(θ + α p) ≤ L(θ) + c_1 α ∇L(θ)·p`
2. **Curvature condition**: `|∇L(θ + α p)·p| ≤ c_2 |∇L(θ)·p|`

Typical values: `c_1 = 1e-4`, `c_2 = 0.9`

### PyTorch API

```python
torch.optim.LBFGS(
    params,
    lr=1,                      # Learning rate (step size multiplier)
    max_iter=20,               # Max iterations per step
    max_eval=None,             # Max function evaluations (default: max_iter * 1.25)
    tolerance_grad=1e-7,       # Termination tolerance on gradient norm
    tolerance_change=1e-9,     # Termination tolerance on function/param change
    history_size=100,          # Update history size (m)
    line_search_fn=None        # 'strong_wolfe' or None
)
```

**Key Differences from Other Optimizers**:
1. **Requires closure**: Must pass a closure that re-evaluates the model
2. **Single parameter group**: Doesn't support per-parameter options
3. **Memory intensive**: Requires `param_bytes * (history_size + 1)` extra memory
4. **Multiple function evaluations per step**: Line search evaluates loss multiple times

### Implementation Details

From [torch/optim/lbfgs.py](reference/pytorch/torch/optim/lbfgs.py):

```python
@torch.no_grad()
def step(self, closure):
    """Perform a single optimization step.

    Args:
        closure (Callable): A closure that reevaluates the model and returns the loss.
    """
    closure = torch.enable_grad()(closure)

    group = self.param_groups[0]
    lr = _to_scalar(group['lr'])
    max_iter = group['max_iter']
    tolerance_grad = group['tolerance_grad']
    line_search_fn = group['line_search_fn']
    history_size = group['history_size']

    state = self.state[self._params[0]]
    state.setdefault('func_evals', 0)
    state.setdefault('n_iter', 0)

    # Evaluate initial f(x) and df/dx
    orig_loss = closure()
    loss = float(orig_loss)
    flat_grad = self._gather_flat_grad()

    # Check optimality
    opt_cond = flat_grad.abs().max() <= tolerance_grad
    if opt_cond:
        return orig_loss

    # Get state
    d = state.get('d')  # Search direction
    old_dirs = state.get('old_dirs', [])  # s_i history
    old_stps = state.get('old_stps', [])  # y_i history
    ro = state.get('ro', [])  # ρ_i = 1/(y_i · s_i)
    H_diag = state.get('H_diag')  # Diagonal Hessian approximation

    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        state['n_iter'] += 1

        # Compute search direction using two-loop recursion
        if state['n_iter'] == 1:
            d = flat_grad.neg()  # First step: steepest descent
            old_dirs = []
            old_stps = []
            ro = []
            H_diag = 1
        else:
            # Two-loop recursion
            q = flat_grad.neg()

            # First loop (backward through history)
            al = []
            for i in range(len(old_dirs) - 1, -1, -1):
                al_i = old_stps[i].dot(q) * ro[i]
                q.add_(old_dirs[i], alpha=-al_i)
                al.append(al_i)
            al.reverse()

            # Scale by H_diag
            r = q.mul(H_diag)

            # Second loop (forward through history)
            for i in range(len(old_dirs)):
                be_i = old_dirs[i].dot(r) * ro[i]
                r.add_(old_stps[i], alpha=al[i] - be_i)

            d = r

        # Directional derivative
        gtd = flat_grad.dot(d)

        # Optional line search
        if line_search_fn == 'strong_wolfe':
            x_init = self._clone_param()

            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d)

            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                obj_func, x_init, t=1, d=d, f=loss, g=flat_grad, gtd=gtd
            )
            self._add_grad(t, d)
            state['func_evals'] += ls_func_evals
        else:
            # Simple backtracking
            t = lr
            self._add_grad(t, d)
            loss = float(closure())
            flat_grad = self._gather_flat_grad()

        # Update history
        y = flat_grad - prev_flat_grad
        s = d.mul(t)
        ys = y.dot(s)

        if ys > 1e-10:  # Curvature condition
            if len(old_dirs) == history_size:
                old_dirs.pop(0)
                old_stps.pop(0)
                ro.pop(0)

            old_dirs.append(s)
            old_stps.append(y)
            ro.append(1.0 / ys)
            H_diag = ys / y.dot(y)

    return orig_loss
```

### Use Cases and Best Practices

**When to Use LBFGS**:
- ✅ Small to medium-sized models (<1M parameters)
- ✅ Full-batch or large-batch training
- ✅ Scientific computing, physics simulations
- ✅ When you need precise convergence (e.g., GANs, style transfer)
- ✅ Deterministic optimization (no stochasticity)

**When NOT to Use LBFGS**:
- ❌ Large models (>10M parameters) - memory prohibitive
- ❌ Mini-batch stochastic training - poor performance
- ❌ Online learning - requires full dataset pass

**Usage Example**:

```python
model = SmallNN()
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10)

def closure():
    optimizer.zero_grad()
    loss = compute_loss(full_batch)  # Must use full batch or large batch
    loss.backward()
    return loss

optimizer.step(closure)
```

**Memory Calculation**:
- Parameters: N floats
- Gradient: N floats
- History: 2 * history_size * N floats (s_i and y_i vectors)
- **Total**: N * (2 + 2 * history_size) floats

For 1M parameters with history_size=100: ~800 MB

### MLX Porting Considerations

**Priority**: **Low** - LBFGS is rarely used in modern deep learning

**Challenges**:
1. Requires line search implementation (complex)
2. Closure-based interface differs from standard optimizers
3. Memory overhead may be prohibitive on mobile/edge devices
4. Limited use cases for typical MLX applications

**Recommendation**: Defer LBFGS implementation unless specific scientific computing use cases arise. Focus on first-order methods (SGD, Adam) first.

---

## 2. NAdam (Nesterov-accelerated Adaptive Moment Estimation)

### Algorithm Description

NAdam combines **Adam's adaptive learning rates** with **Nesterov momentum**, providing better convergence properties than vanilla Adam. The key insight is to apply Nesterov's "look-ahead" to Adam's momentum term.

**Standard Adam**:
```
m_t = β₁ m_{t-1} + (1 - β₁) g_t
v_t = β₂ v_{t-1} + (1 - β₂) g_t²
θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)
```

**NAdam** (Nesterov-accelerated Adam):
```
m_t = β₁ m_{t-1} + (1 - β₁) g_t
v_t = β₂ v_{t-1} + (1 - β₂) g_t²

# Nesterov-style bias correction
μ_t = β₁ (1 - 0.5 * 0.96^(t * ψ))
μ_{t+1} = β₁ (1 - 0.5 * 0.96^((t+1) * ψ))

# Nesterov momentum with bias correction
m̂_t = μ_{t+1} m_t / (1 - ∏μ_i) + (1 - μ_t) g_t / (1 - ∏μ_i)
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)
```

Where:
- `ψ`: Momentum decay (default: 4e-3)
- `μ_t`: Time-dependent momentum coefficient

### PyTorch API

```python
torch.optim.NAdam(
    params,
    lr=2e-3,                           # Higher default than Adam (1e-3)
    betas=(0.9, 0.999),                # Same as Adam
    eps=1e-8,
    weight_decay=0,
    momentum_decay=4e-3,               # ψ (unique to NAdam)
    decoupled_weight_decay=False,      # Use NAdamW variant
    foreach=None,
    maximize=False,
    capturable=False,
    differentiable=False
)
```

### Implementation Details

From [torch/optim/nadam.py:281-379](reference/pytorch/torch/optim/nadam.py#L281-L379):

```python
def _single_tensor_nadam(params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps,
                         *, beta1, beta2, lr, weight_decay, momentum_decay, eps,
                         decoupled_weight_decay, maximize, capturable, differentiable):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step_t = state_steps[i]

        step_t += 1
        step = step_t if capturable else _get_value(step_t)
        bias_correction2 = 1 - beta2 ** step

        # Weight decay
        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)  # AdamW-style (NAdamW)
            else:
                grad = grad.add(param, alpha=weight_decay)  # Adam-style

        # Calculate momentum schedule
        mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))

        # Update mu_product (cumulative product of μ)
        mu_product *= mu

        # Update biased first and second moments
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute denominator
        denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)

        # Nesterov momentum update
        mu_product_next = mu_product * mu_next

        # Update parameters with Nesterov momentum
        param.addcdiv_(grad, denom, value=(-lr * (1.0 - mu) / (1.0 - mu_product)))
        param.addcdiv_(exp_avg, denom, value=(-lr * mu_next / (1.0 - mu_product_next)))
```

**Key Differences from Adam**:
1. **Momentum Decay Schedule**: `μ_t` changes over time based on `momentum_decay`
2. **Two-Part Update**: Separate terms for current gradient and momentum
3. **Cumulative Momentum Product**: Tracks `∏μ_i` for bias correction
4. **Higher Default LR**: 2e-3 vs Adam's 1e-3

### Use Cases

**When to Use NAdam**:
- ✅ NLP / Transformers (often outperforms Adam)
- ✅ Tasks where Adam is already used
- ✅ When training is sensitive to learning rate warm-up

**When NOT to Use NAdam**:
- ❌ Computer vision (SGD often better)
- ❌ When simplicity is preferred (Adam is simpler)

**Empirical Results** (from paper):
- LSTM language modeling: 5-10% better perplexity than Adam
- Machine translation: 0.3-0.5 BLEU improvement
- Image classification: Similar to Adam

### MLX Porting

**Priority**: **Medium** - Useful for Transformer training

**Python API** (MLX):

```python
class NAdam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
    ):
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum_decay = momentum_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.state = {}

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = mx.zeros_like(parameter)
            state["exp_avg_sq"] = mx.zeros_like(parameter)
            state["mu_product"] = 1.0

        state["step"] += 1
        step = state["step"]

        grad = gradient

        # Weight decay
        if self.weight_decay != 0:
            if self.decoupled_weight_decay:
                parameter = parameter * (1 - self.learning_rate * self.weight_decay)
            else:
                grad = grad + self.weight_decay * parameter

        # Momentum schedule
        mu = self.beta1 * (1.0 - 0.5 * (0.96 ** (step * self.momentum_decay)))
        mu_next = self.beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * self.momentum_decay)))

        # Update moments
        state["exp_avg"] = self.beta1 * state["exp_avg"] + (1 - self.beta1) * grad
        state["exp_avg_sq"] = self.beta2 * state["exp_avg_sq"] + (1 - self.beta2) * grad * grad

        # Update mu_product
        state["mu_product"] *= mu

        # Bias correction
        bias_correction2 = 1 - self.beta2 ** step
        denom = mx.sqrt(state["exp_avg_sq"] / bias_correction2) + self.eps

        # Nesterov update
        mu_product_next = state["mu_product"] * mu_next
        update = (
            self.learning_rate * (1.0 - mu) / (1.0 - state["mu_product"]) * grad / denom +
            self.learning_rate * mu_next / (1.0 - mu_product_next) * state["exp_avg"] / denom
        )

        return parameter - update
```

---

## 3. RAdam (Rectified Adam)

### Algorithm Description

RAdam addresses the **variance warm-up problem** in Adam. During early training steps, Adam's adaptive learning rate has high variance due to insufficient samples for estimating the second moment. RAdam rectifies this by:

1. Computing the **length of the approximated SMA** (Simple Moving Average)
2. Using **vanilla SGD** when variance is too high (early steps)
3. Switching to **adaptive learning rate** once variance stabilizes

### Mathematical Formulation

**Standard Adam Update**:
```
m_t = β₁ m_{t-1} + (1 - β₁) g_t
v_t = β₂ v_{t-1} + (1 - β₂) g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)
```

**RAdam** adds variance rectification:

```
# Standard momentum and variance updates
m_t = β₁ m_{t-1} + (1 - β₁) g_t
v_t = β₂ v_{t-1} + (1 - β₂) g_t²
m̂_t = m_t / (1 - β₁^t)

# Compute SMA length
ρ_∞ = 2 / (1 - β₂) - 1
ρ_t = ρ_∞ - 2t β₂^t / (1 - β₂^t)

# Variance rectification
if ρ_t > 5:
    # Variance is reliable - use adaptive learning rate
    l_t = √[(1 - β₂^t) / (√v_t + ε)]
    r_t = √[(ρ_t - 4)(ρ_t - 2)ρ_∞ / ((ρ_∞ - 4)(ρ_∞ - 2)ρ_t)]
    θ_{t+1} = θ_t - α m̂_t r_t l_t
else:
    # Variance is unreliable - use vanilla SGD
    θ_{t+1} = θ_t - α m̂_t
```

Where:
- `ρ_∞`: Maximum SMA length (typically ~5.5 for β₂=0.999)
- `ρ_t`: Current SMA length
- `r_t`: Variance rectification term

**Key Insight**: For β₂=0.999, RAdam uses vanilla SGD for ~5 steps, then gradually transitions to adaptive rates.

### PyTorch API

```python
torch.optim.RAdam(
    params,
    lr=1e-3,                           # Same as Adam
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    decoupled_weight_decay=False,      # Use RAdamW variant
    foreach=None,
    maximize=False,
    capturable=False,
    differentiable=False
)
```

### Implementation Details

From [torch/optim/radam.py:256-360](reference/pytorch/torch/optim/radam.py#L256-L360):

```python
def _single_tensor_radam(params, grads, exp_avgs, exp_avg_sqs, state_steps,
                         *, beta1, beta2, lr, weight_decay, eps,
                         decoupled_weight_decay, maximize, capturable):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step_t += 1
        step = step_t if capturable else _get_value(step_t)

        # Weight decay
        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        # Update biased moments
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        bias_corrected_exp_avg = exp_avg / bias_correction1

        # Compute SMA length
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

        # Variance rectification
        if rho_t > 5.0:
            # Adaptive learning rate
            rect = ((rho_t - 4) * (rho_t - 2) * rho_inf /
                    ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
            adaptive_lr = (bias_correction2 ** 0.5) / (exp_avg_sq.sqrt() + eps)
            param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)
        else:
            # Vanilla SGD (no adaptive rate)
            param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)
```

### Use Cases

**When to Use RAdam**:
- ✅ When Adam training is unstable in early epochs
- ✅ Tasks sensitive to learning rate warm-up
- ✅ When you want Adam's benefits without manual warm-up scheduling

**When NOT to Use RAdam**:
- ❌ When Adam already works well
- ❌ Tasks where warm-up isn't needed

**Empirical Results**:
- Image classification: Similar or slightly better than Adam
- NMT: 0.2-0.4 BLEU improvement over Adam
- Object detection: Better AP than Adam without warm-up

### MLX Porting

**Priority**: **Low** - Nice-to-have, but Adam/AdamW cover most use cases

---

## 4. Adagrad (Adaptive Gradient Algorithm)

### Algorithm Description

Adagrad is the **foundational adaptive learning rate algorithm**. It adapts learning rates based on the cumulative sum of squared gradients, giving frequently updated parameters smaller learning rates and infrequently updated parameters larger learning rates.

**Key Property**: Learning rates are **monotonically decreasing** (never increase).

### Mathematical Formulation

```
G_t = G_{t-1} + g_t²
θ_{t+1} = θ_t - η / (√G_t + ε) · g_t
```

With learning rate decay:
```
η_t = η / (1 + (t-1) * lr_decay)
θ_{t+1} = θ_t - η_t / (√G_t + ε) · g_t
```

Where:
- `G_t`: Cumulative sum of squared gradients (monotonically increasing)
- `η`: Base learning rate
- `ε`: Numerical stability term (default: 1e-10, larger than Adam's 1e-8)

### PyTorch API

```python
torch.optim.Adagrad(
    params,
    lr=1e-2,                           # Higher default than Adam
    lr_decay=0,                        # Learning rate decay
    weight_decay=0,
    initial_accumulator_value=0,       # Initial G_0 (can be > 0)
    eps=1e-10,                         # Larger than Adam's eps
    foreach=None,
    maximize=False,
    differentiable=False
)
```

### Implementation

From [torch/optim/adagrad.py:323-389](reference/pytorch/torch/optim/adagrad.py#L323-L389):

```python
def _single_tensor_adagrad(params, grads, state_sums, state_steps,
                           *, lr, weight_decay, lr_decay, eps, maximize):
    for param, grad, state_sum, step_t in zip(params, grads, state_sums, state_steps):
        step_t += 1
        step = _get_value(step_t)
        grad = grad if not maximize else -grad

        # Weight decay
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Learning rate decay
        clr = lr / (1 + (step - 1) * lr_decay)

        # Update sum of squared gradients (cumulative!)
        state_sum.addcmul_(grad, grad, value=1)

        # Adaptive update
        std = state_sum.sqrt().add_(eps)
        param.addcdiv_(grad, std, value=-clr)
```

**Critical Difference from Adam/RMSprop**:
- **Adagrad**: `G_t = G_{t-1} + g_t²` (cumulative sum, always increases)
- **RMSprop**: `v_t = α v_{t-1} + (1-α) g_t²` (exponential moving average, can decrease)
- **Adam**: Same as RMSprop but with momentum

### Use Cases

**When to Use Adagrad**:
- ✅ **Sparse gradients** (e.g., word embeddings, categorical features)
- ✅ **Feature learning** where different features have different scales
- ✅ **Online learning** with non-stationary objectives

**When NOT to Use Adagrad**:
- ❌ Long training runs (learning rate → 0 due to cumulative sum)
- ❌ Deep neural networks (use Adam/RMSprop instead)
- ❌ Computer vision (use SGD)

**Historical Context**: Adagrad is the precursor to modern adaptive methods:
- Adagrad (2011) → RMSprop (2012) → Adam (2014) → AdamW (2017)

### MLX Porting

**Priority**: **Low** - Historical importance, but superseded by Adam/RMSprop

---

## Comparison Table

| Optimizer | Second Moment | First Moment | Bias Correction | Variance Rectification | Nesterov | Memory | Use Case |
|-----------|---------------|--------------|-----------------|------------------------|----------|--------|----------|
| **SGD** | None | Optional | No | N/A | Optional | 1x | CV, general |
| **Adam** | EMA | EMA | Yes | No | No | 2x | NLP, general |
| **AdamW** | EMA | EMA | Yes | No | No | 2x | Transformers |
| **RMSprop** | EMA | None | No | No | No | 1x | RNNs, RL |
| **NAdam** | EMA | EMA (Nesterov) | Yes | No | Yes | 2x+ | NLP, Transformers |
| **RAdam** | EMA | EMA | Yes | Yes | No | 2x | Unstable Adam training |
| **Adagrad** | Cumulative | None | No | No | No | 1x | Sparse, online learning |
| **LBFGS** | Quasi-Newton | N/A | N/A | N/A | No | 100x+ | Full-batch, small models |

---

## MLX Implementation Priorities

### High Priority (Implement First)
1. **Adam / AdamW** - Most widely used
2. **SGD** - Essential for CV

### Medium Priority (Implement Second)
3. **RMSprop** - Important for RNNs, RL
4. **NAdam** - Useful for Transformers

### Low Priority (Implement Later or Skip)
5. **RAdam** - Nice-to-have, but Adam + warm-up covers most cases
6. **Adagrad** - Historical, superseded by modern methods
7. **LBFGS** - Niche use cases, complex implementation

---

## References

1. **LBFGS**: Nocedal, J. & Wright, S. (2006). "Numerical Optimization"
2. **NAdam**: Dozat, T. (2016). "Incorporating Nesterov Momentum into Adam"
3. **RAdam**: Liu, L. et al. (2019). "On the Variance of the Adaptive Learning Rate and Beyond"
4. **Adagrad**: Duchi, J. et al. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"

---

## Summary

This document covered four advanced optimizers:

**LBFGS**: Second-order quasi-Newton method for precise, full-batch optimization. Memory intensive (100x parameters), requires line search. Use for small models and scientific computing.

**NAdam**: Nesterov-accelerated Adam with time-dependent momentum schedule. Often outperforms Adam on NLP tasks (5-10% improvement). Higher default LR (2e-3 vs 1e-3).

**RAdam**: Rectified Adam that addresses variance warm-up. Uses vanilla SGD for ~5 steps, then transitions to adaptive rates. Good for unstable Adam training.

**Adagrad**: Foundational adaptive method with cumulative squared gradients. Learning rates monotonically decrease. Good for sparse gradients but superseded by RMSprop/Adam for most tasks.

**For MLX**: Prioritize Adam/AdamW and SGD first, then RMSprop and NAdam. LBFGS, RAdam, and Adagrad are lower priority.

---

## 5. Adadelta (Adaptive Delta)

### Algorithm Description

Adadelta is an extension of Adagrad that **avoids the problem of monotonically decreasing learning rates**. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to a fixed size.

**Key Innovation**: Adadelta doesn't require a learning rate hyperparameter - it uses a ratio of RMS of parameter updates to RMS of gradients.

### Mathematical Formulation

```
# Exponential moving average of squared gradients
E[g²]_t = ρ E[g²]_{t-1} + (1 - ρ) g_t²

# RMS of gradients
RMS[g]_t = √(E[g²]_t + ε)

# Parameter update (before applying to parameters)
Δθ_t = - (RMS[Δθ]_{t-1} / RMS[g]_t) · g_t

# Exponential moving average of squared parameter updates
E[Δθ²]_t = ρ E[Δθ²]_{t-1} + (1 - ρ) Δθ_t²

# RMS of parameter updates
RMS[Δθ]_t = √(E[Δθ²]_t + ε)

# Apply update
θ_{t+1} = θ_t + Δθ_t
```

Where:
- `ρ`: Decay rate (typically 0.9)
- `ε`: Numerical stability (typically 1e-6)

### PyTorch API

```python
torch.optim.Adadelta(
    params,
    lr=1.0,                            # Often 1.0 (algorithm is LR-free in theory)
    rho=0.9,                           # Decay rate
    eps=1e-6,                          # Larger than Adam's eps
    weight_decay=0,
    foreach=None,
    maximize=False,
    differentiable=False
)
```

### Implementation

From [torch/optim/adadelta.py](reference/pytorch/torch/optim/adadelta.py):

```python
def _single_tensor_adadelta(params, grads, square_avgs, acc_deltas, *,
                            lr, rho, eps, weight_decay, maximize):
    for param, grad, square_avg, acc_delta in zip(params, grads, square_avgs, acc_deltas):
        grad = grad if not maximize else -grad

        # Weight decay
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Update running average of squared gradients
        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)

        # Compute update
        std = square_avg.add(eps).sqrt()
        delta = acc_delta.add(eps).sqrt().div_(std).mul_(grad)

        # Apply update
        param.add_(delta, alpha=-lr)

        # Update running average of squared parameter updates
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
```

### Use Cases

**When to Use Adadelta**:
- ✅ When learning rate tuning is difficult
- ✅ As an alternative to RMSprop
- ✅ Tasks where Adagrad's decreasing LR is problematic

**When NOT to Use Adadelta**:
- ❌ Modern deep learning (Adam/AdamW preferred)
- ❌ Computer vision (SGD preferred)

### MLX Equivalent

```python
class Adadelta(Optimizer):
    def __init__(self, learning_rate=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def apply_single(self, gradient, parameter, state):
        if "square_avg" not in state:
            state["square_avg"] = mx.zeros_like(parameter)
            state["acc_delta"] = mx.zeros_like(parameter)

        grad = gradient
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * parameter

        # Update running average of squared gradients
        state["square_avg"] = self.rho * state["square_avg"] + (1 - self.rho) * grad * grad

        # Compute update
        std = mx.sqrt(state["square_avg"] + self.eps)
        delta = mx.sqrt(state["acc_delta"] + self.eps) / std * grad

        # Update running average of squared deltas
        state["acc_delta"] = self.rho * state["acc_delta"] + (1 - self.rho) * delta * delta

        return parameter - self.learning_rate * delta
```

**Priority**: **Low** - Historical importance, superseded by Adam

---

## 6. ASGD (Averaged Stochastic Gradient Descent)

### Algorithm Description

ASGD, or **Polyak-Ruppert averaging**, maintains a running average of parameters in addition to the current parameters. This average often provides better generalization than the final parameters.

**Key Insight**: The average of all parameters visited during optimization can be a better estimator than the final point.

### Mathematical Formulation

```
# Standard SGD update
θ_t = θ_{t-1} - η_t g_t

# Averaged parameters (after t0 steps)
if t > t0:
    θ̄_t = ((t - t0) θ̄_{t-1} + θ_t) / (t - t0 + 1)
else:
    θ̄_t = θ_t

# Learning rate decay
η_t = η / (1 + (t - 1) * λ)^α
```

Where:
- `t0`: Number of steps before starting averaging
- `λ`: Learning rate decay (lambd)
- `α`: Power for learning rate decay (alpha)

### PyTorch API

```python
torch.optim.ASGD(
    params,
    lr=1e-2,                           # Base learning rate
    lambd=1e-4,                        # Decay term
    alpha=0.75,                        # Power for decay
    t0=1e6,                            # Start averaging after t0 steps
    weight_decay=0,
    foreach=None,
    maximize=False,
    differentiable=False
)
```

### Implementation

From [torch/optim/asgd.py](reference/pytorch/torch/optim/asgd.py):

```python
def _single_tensor_asgd(params, grads, axs, mus, etas, state_steps, *,
                        lambd, lr, t0, alpha, weight_decay, maximize):
    for param, grad, ax, mu, eta, step in zip(params, grads, axs, mus, etas, state_steps):
        step += 1
        grad = grad if not maximize else -grad

        # Weight decay
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Learning rate decay
        eta_val = lr / ((1 + lambd * (step - 1)) ** alpha)

        # SGD step
        param.add_(grad, alpha=-eta_val)

        # Update average (after t0 steps)
        if mu != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)

        # Update mu for next step
        new_mu = 1 / max(1, step - t0)
        mus[i].fill_(new_mu)
```

### Use Cases

**When to Use ASGD**:
- ✅ When seeking better generalization
- ✅ Convex optimization problems
- ✅ Traditional machine learning (logistic regression, SVM)

**When NOT to Use ASGD**:
- ❌ Deep learning (Adam/SGD preferred)
- ❌ When fast convergence is more important than generalization

### MLX Equivalent

```python
class ASGD(Optimizer):
    def __init__(self, learning_rate=0.01, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.alpha = alpha
        self.t0 = t0
        self.weight_decay = weight_decay
        self.state = {}

    def apply_single(self, gradient, parameter, state):
        if "step" not in state:
            state["step"] = 0
            state["ax"] = mx.array(parameter)  # Averaged parameter

        state["step"] += 1
        step = state["step"]

        grad = gradient
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * parameter

        # Learning rate decay
        eta = self.learning_rate / ((1 + self.lambd * (step - 1)) ** self.alpha)

        # SGD update
        new_param = parameter - eta * grad

        # Update average
        if step > self.t0:
            mu = 1 / (step - self.t0 + 1)
            state["ax"] = state["ax"] + mu * (new_param - state["ax"])
        else:
            state["ax"] = new_param

        return new_param
```

**Priority**: **Low** - Specialized use cases

---

## 7. Rprop (Resilient Backpropagation)

### Algorithm Description

Rprop uses **only the sign of gradients** to determine update direction, with a magnitude that adapts based on whether gradients agree across steps. It's designed for **full-batch training** where gradient noise is minimal.

**Key Properties**:
- Ignores gradient magnitude (only uses sign)
- Increases step size when gradients agree, decreases when they flip
- Very effective for full-batch optimization

### Mathematical Formulation

```
# Sign-based update
Δ_t(i) = {
    min(Δ_{t-1}(i) * η⁺, Δ_max)  if g_t(i) * g_{t-1}(i) > 0
    max(Δ_{t-1}(i) * η⁻, Δ_min)  if g_t(i) * g_{t-1}(i) < 0
    Δ_{t-1}(i)                    otherwise
}

# Parameter update
θ_{t+1}(i) = θ_t(i) - sign(g_t(i)) * Δ_t(i)
```

Where:
- `η⁺`: Increase factor (typically 1.2)
- `η⁻`: Decrease factor (typically 0.5)
- `Δ_min`, `Δ_max`: Step size bounds

### PyTorch API

```python
torch.optim.Rprop(
    params,
    lr=1e-2,                           # Initial step size
    etas=(0.5, 1.2),                   # (η⁻, η⁺)
    step_sizes=(1e-6, 50),             # (Δ_min, Δ_max)
    foreach=None,
    maximize=False,
    differentiable=False
)
```

### Implementation

From [torch/optim/rprop.py](reference/pytorch/torch/optim/rprop.py):

```python
def _single_tensor_rprop(params, grads, prevs, step_sizes, *,
                         step_size_min, step_size_max, etaminus, etaplus, maximize):
    for param, grad, prev, step_size in zip(params, grads, prevs, step_sizes):
        grad = grad if not maximize else -grad

        # Sign of gradient product
        sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus    # Increase step size
        sign[sign.lt(0)] = etaminus   # Decrease step size
        sign[sign.eq(0)] = 1          # No change

        # Update step sizes
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)

        # Zero gradients where sign flipped
        grad = grad.clone()
        grad[sign.eq(etaminus)] = 0

        # Update parameters
        param.addcmul_(grad.sign(), step_size, value=-1)

        # Store gradient for next step
        prev.copy_(grad)
```

### Use Cases

**When to Use Rprop**:
- ✅ Full-batch training
- ✅ Small datasets
- ✅ When gradient magnitude is noisy but sign is reliable

**When NOT to Use Rprop**:
- ❌ Mini-batch/stochastic training
- ❌ Large-scale deep learning

### MLX Equivalent

```python
class Rprop(Optimizer):
    def __init__(self, learning_rate=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
        self.learning_rate = learning_rate
        self.etaminus, self.etaplus = etas
        self.step_min, self.step_max = step_sizes
        self.state = {}

    def apply_single(self, gradient, parameter, state):
        if "prev" not in state:
            state["prev"] = mx.zeros_like(gradient)
            state["step_size"] = mx.full(gradient.shape, self.learning_rate)

        # Sign of gradient product
        sign = mx.sign(gradient * state["prev"])

        # Update step sizes
        step_size = state["step_size"]
        step_size = mx.where(sign > 0, step_size * self.etaplus, step_size)
        step_size = mx.where(sign < 0, step_size * self.etaminus, step_size)
        step_size = mx.clip(step_size, self.step_min, self.step_max)

        # Zero gradients where sign flipped
        grad = mx.where(sign < 0, 0.0, gradient)

        # Update parameters
        new_param = parameter - mx.sign(grad) * step_size

        # Store state
        state["prev"] = grad
        state["step_size"] = step_size

        return new_param
```

**Priority**: **Low** - Specialized for full-batch training

---

## 8. SparseAdam

### Algorithm Description

SparseAdam is a **variant of Adam designed for sparse gradients**, particularly useful for **embedding layers** in NLP. It only updates the moment estimates for elements that have non-zero gradients.

**Key Difference from Adam**:
- Regular Adam updates all moment estimates even when gradients are zero
- SparseAdam only updates moment estimates for non-zero gradient indices
- More memory and compute efficient for sparse updates

### PyTorch API

```python
torch.optim.SparseAdam(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    maximize=False
)
```

### Implementation

From [torch/optim/sparse_adam.py](reference/pytorch/torch/optim/sparse_adam.py):

```python
def step(self, closure=None):
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if not grad.is_sparse:
                raise RuntimeError("SparseAdam only supports sparse gradients")

            # Get sparse indices and values
            grad = grad.coalesce()
            grad_indices = grad._indices()
            grad_values = grad._values()

            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1
            beta1, beta2 = group['betas']

            # Only update for non-zero gradient indices
            old_exp_avg = state['exp_avg'].sparse_mask(grad)
            old_exp_avg_sq = state['exp_avg_sq'].sparse_mask(grad)

            # Standard Adam update on sparse subset
            new_exp_avg = beta1 * old_exp_avg._values() + (1 - beta1) * grad_values
            new_exp_avg_sq = beta2 * old_exp_avg_sq._values() + (1 - beta2) * grad_values ** 2

            # Bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] / bias_correction1

            # Sparse parameter update
            denom = (new_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            update = new_exp_avg / denom
            p.data.add_(-step_size, torch.sparse_coo_tensor(grad_indices, update, p.shape))

            # Update state (sparse)
            state['exp_avg'].scatter_(0, grad_indices, new_exp_avg)
            state['exp_avg_sq'].scatter_(0, grad_indices, new_exp_avg_sq)
```

### Use Cases

**When to Use SparseAdam**:
- ✅ Embedding layers with sparse updates
- ✅ NLP models with large vocabularies
- ✅ Recommendation systems

**When NOT to Use SparseAdam**:
- ❌ Dense gradients (use regular Adam)
- ❌ Computer vision

**Priority**: **Medium** - Important for NLP embeddings

---

## 9. Adafactor

### Algorithm Description

Adafactor is a **memory-efficient adaptive optimizer** designed for training large models. It uses **factored second-moment estimates** to reduce memory from O(n) to O(√n) for matrices.

**Key Innovations**:
1. **Row and column factorization**: Instead of storing full second moment, store row and column factors
2. **No first moment by default**: Reduces memory further
3. **Relative step size**: Scale updates relative to parameter magnitude

### Mathematical Formulation

```
# For matrices: factorized second moment
R_t = β₂ R_{t-1} + (1 - β₂) mean(g_t², axis=1)  # Row means
C_t = β₂ C_{t-1} + (1 - β₂) mean(g_t², axis=0)  # Column means

# Reconstruct second moment estimate
v_t = R_t[:, None] * C_t[None, :] / mean(R_t)

# RMS clipping
RMS(g_t) = √(mean(g_t²))
g_t = g_t / max(1, RMS(g_t) / d)

# Update
θ_{t+1} = θ_t - α_t g_t / √(v_t + ε)
```

Where:
- `d`: Clipping threshold (typically 1.0)
- `α_t`: Learning rate (can be relative to parameter magnitude)

### PyTorch API

```python
torch.optim.Adafactor(
    params,
    lr=None,                           # Can be None for relative LR
    eps=(1e-30, 1e-3),                 # (eps1, eps2)
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,                        # None = no momentum
    weight_decay=0.0,
    scale_parameter=True,
    relative_step=True,
    warmup_init=False
)
```

### Implementation (Simplified)

```python
def _single_tensor_adafactor(params, grads, state_steps, exp_avgs, exp_avg_sq_rows, exp_avg_sq_cols, *,
                              lr, eps1, eps2, clip_threshold, decay_rate, beta1, weight_decay,
                              scale_parameter, relative_step, warmup_init):
    for param, grad, step in zip(params, grads, state_steps):
        step += 1

        # Compute RMS of gradients
        rms = grad.pow(2).mean().sqrt()

        # Gradient clipping
        if clip_threshold > 0:
            grad = grad / max(1.0, rms / clip_threshold)

        # Decay rate schedule
        rho = min(decay_rate, 1.0 - step ** (-0.8))

        # Relative step size
        if relative_step:
            param_scale = max(eps2, param.abs().mean())
            alpha = param_scale / max(1, step)
        else:
            alpha = lr

        # Factorized second moment (for matrices)
        if param.dim() >= 2:
            # Row and column statistics
            row_mean = grad.pow(2).mean(dim=-1, keepdim=True)
            col_mean = grad.pow(2).mean(dim=-2, keepdim=True)

            exp_avg_sq_rows.mul_(rho).add_(row_mean, alpha=1 - rho)
            exp_avg_sq_cols.mul_(rho).add_(col_mean, alpha=1 - rho)

            # Reconstruct
            v = exp_avg_sq_rows * exp_avg_sq_cols
            v = v / exp_avg_sq_rows.mean()
        else:
            # Vector: standard EMA
            v = grad.pow(2)

        # Update
        u = grad / (v.sqrt() + eps1)

        # Optional momentum
        if beta1 is not None:
            exp_avgs[i].mul_(beta1).add_(u, alpha=1 - beta1)
            u = exp_avgs[i]

        # Apply update
        param.add_(u, alpha=-alpha)
```

### Use Cases

**When to Use Adafactor**:
- ✅ Large transformer models (T5, BART)
- ✅ Memory-constrained training
- ✅ When Adam's memory overhead is too high

**When NOT to Use Adafactor**:
- ❌ Small models (Adam works fine)
- ❌ When training stability is critical

**Memory Savings**:
- For (H, W) matrix: Adam needs 2HW, Adafactor needs H + W
- For 4096x4096 matrix: Adam ~134MB, Adafactor ~33KB

**Priority**: **Medium** - Important for large Transformer training

---

## 10. Muon (Momentum Orthogonalized Update)

### Algorithm Description

Muon is a **research optimizer** that applies orthogonalization to momentum updates. It's designed to improve training dynamics by maintaining update orthogonality.

**Key Innovation**:
- Orthogonalizes momentum across parameters
- Can improve convergence in some architectures
- Experimental/research optimizer

### Mathematical Formulation

```
# Standard momentum
m_t = β m_{t-1} + g_t

# Orthogonalize momentum (Gram-Schmidt or Newton-Schulz)
m_t^{orth} = orthogonalize(m_t)

# Update
θ_{t+1} = θ_t - α m_t^{orth}
```

### PyTorch API

```python
torch.optim.Muon(
    params,
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    backend='newtonschulz5',           # Orthogonalization method
    backend_steps=5
)
```

### Implementation Notes

Muon uses Newton-Schulz iteration for orthogonalization:

```python
def newton_schulz(M, num_iters=5):
    """Compute M @ inv(sqrt(M.T @ M)) using Newton-Schulz iteration"""
    X = M
    for _ in range(num_iters):
        A = X.T @ X
        B = (3 * torch.eye(A.shape[0]) - A) / 2
        X = X @ B
    return X
```

### Use Cases

**When to Use Muon**:
- ✅ Research/experimentation
- ✅ Specific architectures where orthogonality helps
- ✅ Training stability issues

**When NOT to Use Muon**:
- ❌ Production systems (not widely validated)
- ❌ Standard architectures (Adam/SGD preferred)

**Priority**: **Low** - Experimental/research optimizer

---

## Extended Optimizer Summary

| Optimizer | Memory | Sparse Support | Use Case | Priority |
|-----------|--------|----------------|----------|----------|
| LBFGS | 100x+ | No | Full-batch, small models | Low |
| NAdam | 2x+ | No | Transformers, NLP | Medium |
| RAdam | 2x | No | Unstable Adam training | Low |
| Adagrad | 1x | Yes | Sparse, online learning | Low |
| Adadelta | 2x | No | LR-free training | Low |
| ASGD | 2x | No | Better generalization | Low |
| Rprop | 2x | No | Full-batch training | Low |
| SparseAdam | 2x | Yes | NLP embeddings | Medium |
| Adafactor | ~1x | No | Large transformers | Medium |
| Muon | 2x | No | Research | Low |

**Total Optimizers Documented**: 10

---

## MLX Implementation Recommendations

### Tier 1 (Essential)
1. SGD (with momentum, Nesterov)
2. Adam / AdamW

### Tier 2 (Important)
3. RMSprop
4. NAdam
5. SparseAdam
6. Adafactor

### Tier 3 (Nice-to-have)
7. RAdam
8. Adadelta
9. Adagrad
10. ASGD
11. Rprop
12. LBFGS
13. Muon
