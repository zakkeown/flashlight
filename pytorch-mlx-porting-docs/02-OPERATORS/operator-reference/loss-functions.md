# Loss Functions: PyTorch Implementation & MLX Porting Guide

## Overview

Loss functions quantify the difference between model predictions and ground truth labels, providing the training signal for optimization. PyTorch provides a comprehensive set of loss functions for various tasks:

1. **Classification Losses** - Cross-entropy, NLL, binary cross-entropy
2. **Regression Losses** - MSE, L1, Smooth L1 (Huber)
3. **Sequence Losses** - CTC loss for sequence-to-sequence models
4. **Embedding Losses** - Triplet margin, cosine embedding, hinge embedding
5. **Probabilistic Losses** - KL divergence, Poisson NLL

This document provides detailed reverse-engineering of PyTorch's loss function implementations to aid MLX porting efforts.

---

## Table of Contents

- [Reduction Modes](#reduction-modes)
- [Cross-Entropy Loss](#cross-entropy-loss)
- [Negative Log-Likelihood (NLL) Loss](#negative-log-likelihood-nll-loss)
- [Binary Cross-Entropy Loss](#binary-cross-entropy-loss)
- [Mean Squared Error (MSE) Loss](#mean-squared-error-mse-loss)
- [L1 Loss (MAE)](#l1-loss-mae)
- [Smooth L1 Loss (Huber Loss)](#smooth-l1-loss-huber-loss)
- [CTC Loss](#ctc-loss)
- [KL Divergence Loss](#kl-divergence-loss)
- [Margin-Based Losses](#margin-based-losses)
- [MLX Porting Recommendations](#mlx-porting-recommendations)

---

## Reduction Modes

All PyTorch loss functions support three reduction modes that determine how per-sample losses are aggregated:

**Reduction Modes** (`aten/src/ATen/core/Reduction.h`):

```cpp
enum Reduction {
  None,      // No reduction - return per-sample losses
  Mean,      // Average over all samples (default)
  Sum        // Sum over all samples
};
```

**Helper Function** (`Loss.cpp:63-71`):

```cpp
inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;  // Reduction::None
}
```

**Usage Pattern**:
- `reduction='none'` → Returns `[N]` or `[N, ...]` tensor with per-sample losses
- `reduction='mean'` → Returns scalar (average loss)
- `reduction='sum'` → Returns scalar (total loss)

---

## Cross-Entropy Loss

### Algorithm

Cross-entropy loss combines `log_softmax` and `nll_loss` for multi-class classification. It's numerically stable by computing log-softmax directly.

**Mathematical Formulation**:

For input logits `x` (unnormalized scores) and target class `y`:

```
Loss = -log(softmax(x)[y])
     = -log(exp(x[y]) / Σ_j exp(x[j]))
     = -x[y] + log(Σ_j exp(x[j]))
     = -x[y] + logsumexp(x)
```

With class weights `w` and label smoothing `α`:

```
For each sample i:
  1. Compute log-probabilities:
     log_probs[i] = log_softmax(logits[i])

  2. Apply label smoothing (optional):
     If α > 0:
       smooth_target = (1 - α) * one_hot(target[i]) + α / num_classes

  3. Compute loss:
     loss[i] = -w[target[i]] * log_probs[i, target[i]]

  4. Ignore index (optional):
     If target[i] == ignore_index:
       loss[i] = 0
```

### PyTorch API

**Native Function Signature** (`native_functions.yaml:9629`):

```yaml
- func: cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None,
                           int reduction=Mean, SymInt ignore_index=-100,
                           float label_smoothing=0.0) -> Tensor
```

**Python API**:

```python
torch.nn.CrossEntropyLoss(
    weight=None,          # Per-class weights [C]
    size_average=None,    # Deprecated
    ignore_index=-100,    # Target value to ignore
    reduce=None,          # Deprecated
    reduction='mean',     # 'none' | 'mean' | 'sum'
    label_smoothing=0.0   # Label smoothing factor [0, 1]
)
```

**Input Shapes**:
- `input`: `[N, C]` or `[N, C, d1, d2, ...]` (logits, unnormalized scores)
- `target`: `[N]` or `[N, d1, d2, ...]` (class indices, long dtype)
- `weight`: `[C]` (optional per-class weights)

### Implementation

**Cross-Entropy = Log Softmax + NLL Loss** (`LossNLL.cpp:17-28`):

```cpp
Tensor cross_entropy_loss(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {

  // Cross-entropy is decomposed into two steps:
  // 1. Compute log-softmax (numerically stable)
  auto log_probs = at::log_softmax(self, /*dim=*/1);

  // 2. Compute NLL loss
  return at::nll_loss(log_probs, target, weight_opt, reduction, ignore_index);
}
```

**Why This Decomposition?**
1. **Numerical stability**: `log_softmax` uses log-sum-exp trick
2. **Code reuse**: NLL loss handles weights, reduction, ignore_index, label smoothing
3. **Gradient efficiency**: Fused backward pass

### Gradients

**Gradient w.r.t. logits**:

For sample `i`, class `c`:

```
∂L/∂x[i,c] = w[target[i]] * (softmax(x[i])[c] - δ(c == target[i]))
```

Where `δ(·)` is the Kronecker delta (1 if true, 0 if false).

**Simplified**:
- For target class `c = target[i]`: `∂L/∂x[i,c] = w[c] * (p[c] - 1)`
- For other classes `c ≠ target[i]`: `∂L/∂x[i,c] = w[target[i]] * p[c]`

Where `p = softmax(x[i])` is the predicted probability distribution.

---

## Negative Log-Likelihood (NLL) Loss

### Algorithm

NLL loss expects log-probabilities as input (output of `log_softmax`) and computes the negative log-likelihood of the correct class.

**Mathematical Formulation**:

```
For each sample i:
  target_class = target[i]

  If target_class == ignore_index:
    loss[i] = 0
  Else:
    loss[i] = -weight[target_class] * input[i, target_class]
```

With reduction:
```
total_weight = Σ_i weight[target[i]]  (excluding ignored samples)

If reduction == 'mean':
  output = (Σ_i loss[i]) / total_weight
Elif reduction == 'sum':
  output = Σ_i loss[i]
Else:  # 'none'
  output = loss  # [N]
```

### PyTorch API

**Native Function Signature** (`native_functions.yaml:11944`):

```yaml
- func: nll_loss(Tensor self, Tensor target, Tensor? weight=None,
                 int reduction=Mean, SymInt ignore_index=-100) -> Tensor
```

**Python API**:

```python
torch.nn.NLLLoss(
    weight=None,        # Per-class weights [C]
    size_average=None,  # Deprecated
    ignore_index=-100,  # Target value to ignore
    reduce=None,        # Deprecated
    reduction='mean'    # 'none' | 'mean' | 'sum'
)
```

**Input Shapes**:
- `input`: `[N, C]` (log-probabilities)
- `target`: `[N]` (class indices, long dtype)
- `weight`: `[C]` (optional per-class weights)

### Implementation

**Forward Pass** (`LossNLL.cpp:161-300`):

```cpp
template <typename scalar_t, typename target_t>
void nll_loss_out_frame(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {

  const auto n_dims = input.dim();
  const auto n_classes = input.size(-1);

  scalar_t* total_weight_data = total_weight.data_ptr<scalar_t>();
  *total_weight_data = 0;

  const scalar_t* weight_data = optional_data<const scalar_t>(weight);

  // Case 1: No reduction (return per-sample losses)
  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = input.size(0);
    at::native::resize_output(output, {batch_size});

    auto input_acc = input.accessor<const scalar_t, 2>();
    auto target_acc = target.accessor<const target_t, 1>();
    auto output_acc = output.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        const auto cur_target = target_acc[i];

        if (cur_target == ignore_index) {
          output_acc[i] = 0;
          continue;
        }

        TORCH_CHECK_INDEX(
            cur_target >= 0 && cur_target < n_classes,
            "Target ", cur_target, " is out of bounds.");

        scalar_t cur_weight = weight_data != nullptr ? weight_data[cur_target]
                                                     : static_cast<scalar_t>(1);
        // NLL loss formula: -w[target] * log_prob[target]
        output_acc[i] = -input_acc[i][cur_target] * cur_weight;
      }
    });

    return;
  }

  // Case 2: Reduction (mean or sum)
  at::native::resize_output(output, {});

  const scalar_t* input_data = input.contiguous().const_data_ptr<scalar_t>();
  const target_t* target_data = target.contiguous().const_data_ptr<target_t>();

  const int64_t batch_size = n_dims == 1 ? 1 : input.size(0);

  // Cascade sum for numerical stability (8-level reduction tree)
  constexpr int64_t cascade_sum_num_levels = 8;
  scalar_t weight_partial_sums[cascade_sum_num_levels] = {0};
  scalar_t loss_partial_sums[cascade_sum_num_levels] = {0};

  int64_t num_ignored = 0;

  for (const auto b : c10::irange(batch_size)) {
    const int64_t cur_target = target_data[b];

    if (cur_target == ignore_index) {
      ++num_ignored;
      continue;
    }

    TORCH_CHECK_INDEX(
        cur_target >= 0 && cur_target < n_classes,
        "Target ", cur_target, " is out of bounds.");

    const auto data = input_data[b * n_classes + cur_target];

    if (weight_data) {
      const scalar_t weight_val = weight_data[cur_target];
      loss_partial_sums[0] -= data * weight_val;
      weight_partial_sums[0] += weight_val;
    } else {
      loss_partial_sums[0] -= data;
    }

    // Cascade sum (reduction tree to improve numerical precision)
    for (int64_t j = 0; j + 1 < cascade_sum_num_levels; ++j) {
      // ... reduction logic
    }
  }

  // Aggregate partial sums
  const scalar_t total_weight_val = !weight_data ?
    static_cast<scalar_t>(batch_size - num_ignored) :
    std::accumulate(std::begin(weight_partial_sums),
                    std::end(weight_partial_sums),
                    scalar_t{0});

  scalar_t output_val = std::accumulate(std::begin(loss_partial_sums),
                                        std::end(loss_partial_sums),
                                        scalar_t{0});

  if (reduction == Reduction::Mean) {
    output_val /= total_weight_val;
  }

  *output.data_ptr<scalar_t>() = output_val;
  *total_weight_data = total_weight_val;
}
```

**Cascade Sum for Numerical Stability**:

PyTorch uses an 8-level reduction tree to sum losses, which reduces numerical errors from adding many floating-point numbers:

```cpp
// Instead of: loss = loss1 + loss2 + loss3 + ... + lossN
// Use hierarchical sum:
//   Level 0: [l1, l2, l3, l4, l5, l6, l7, l8]
//   Level 1: [l1+l2, l3+l4, l5+l6, l7+l8]
//   Level 2: [(l1+l2)+(l3+l4), (l5+l6)+(l7+l8)]
//   ...
```

### Gradients

**Gradient w.r.t. input (log-probabilities)** (`LossNLL.cpp:339-399`):

```cpp
template <typename scalar_t, typename target_t>
void nll_loss_backward_out_frame(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {

  const auto n_classes = input.size(-1);
  const scalar_t* weight_data = optional_data<const scalar_t>(weight);

  // Initialize grad_input to zero
  grad_input.zero_();

  if (reduction == Reduction::None) {
    // Per-sample gradient
    auto grad_output_acc = grad_output.accessor<const scalar_t, 1>();
    auto grad_input_acc = grad_input.accessor<scalar_t, 2>();

    for (const auto i : c10::irange(batch_size)) {
      auto cur_target = target_acc[i];
      if (cur_target == ignore_index) continue;

      const scalar_t w = weight_data ? weight_data[cur_target] : 1.0;
      grad_input_acc[i][cur_target] = -w * grad_output_acc[i];
    }
  } else {
    // Scalar gradient (mean or sum)
    const scalar_t total_weight_value = *total_weight.const_data_ptr<scalar_t>();
    const scalar_t grad_output_value = *grad_output.const_data_ptr<scalar_t>();

    const auto grad = -(reduction == Reduction::Mean
                        ? grad_output_value / total_weight_value
                        : grad_output_value);

    for (const auto i : c10::irange(batch_size)) {
      auto cur_target = target_acc[i];
      if (cur_target == ignore_index) continue;

      grad_input_acc[i][cur_target] = weight_data ? weight_data[cur_target] * grad
                                                   : grad;
    }
  }
}
```

**Gradient Formula**:

```
For sample i, class c:
  If c == target[i] and target[i] != ignore_index:
    ∂L/∂log_prob[i,c] = -w[c] * grad_out * (1 / total_weight if mean else 1)
  Else:
    ∂L/∂log_prob[i,c] = 0
```

---

## Binary Cross-Entropy Loss

### Algorithm

Binary cross-entropy (BCE) loss for binary classification with probabilities (not logits).

**Mathematical Formulation**:

```
For each element:
  BCE(x, y) = -w * [y * log(x) + (1 - y) * log(1 - x)]
```

Where:
- `x` ∈ [0, 1] is the predicted probability
- `y` ∈ {0, 1} is the target
- `w` is the optional sample weight

### PyTorch API

**Native Function Signature** (`native_functions.yaml`):

```yaml
- func: binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None,
                              int reduction=Mean) -> Tensor
```

**Python API**:

```python
torch.nn.BCELoss(
    weight=None,      # Per-sample weights [N]
    size_average=None,  # Deprecated
    reduce=None,        # Deprecated
    reduction='mean'    # 'none' | 'mean' | 'sum'
)
```

**Input Shapes**:
- `input`: Any shape (probabilities in [0, 1])
- `target`: Same shape as input (values in {0, 1})
- `weight`: Same shape as input (optional)

**Important**: Input must be **probabilities** (post-sigmoid), not logits. For logits, use `BCEWithLogitsLoss`.

### Implementation

**Forward Pass** (`Loss.cpp:261-302`):

```cpp
Tensor& binary_cross_entropy_out_cpu(
    const Tensor& input, const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction, Tensor& loss) {

  Tensor loss_squeezed = at::squeeze(loss);

  auto iter = TensorIteratorConfig()
    .add_output(loss_squeezed)
    .add_owned_const_input(at::squeeze(input))
    .add_owned_const_input(at::squeeze(target))
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      loss.scalar_type(), "binary_cross_entropy",
      [&] {
        at::native::cpu_kernel(
            iter, [](scalar_t input_val, scalar_t target_val) {
              // Validate inputs in [0, 1]
              TORCH_CHECK(
                  (input_val >= 0) && (input_val <= 1),
                  "all elements of input should be between 0 and 1");
              TORCH_CHECK(
                  (target_val >= 0) && (target_val <= 1),
                  "all elements of target should be between 0 and 1");

              // Binary cross entropy formula:
              // L = -w (y ln(x) + (1-y) ln(1-x))
              // Rewritten for numerical stability:
              return (target_val - scalar_t(1)) *
                  std::max(scalar_t(std::log1p(-input_val)), scalar_t(-100)) -
                  target_val *
                  std::max(scalar_t(std::log(input_val)), scalar_t(-100));
            });
      });

  // Apply per-sample weights
  if (weight_opt.has_value() && weight_opt->defined()) {
      loss.mul_(*weight_opt);
  }

  // Apply reduction
  if (reduction != at::Reduction::None) {
      Tensor loss_reduced = apply_loss_reduction(loss, reduction);
      loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }

  return loss;
}
```

**Numerical Stability**:
- Uses `log1p(-x)` for `log(1-x)` (better precision near x=1)
- Clamps log values to -100 to prevent -infinity

### Binary Cross-Entropy with Logits

**For logits (unnormalized scores), use BCEWithLogitsLoss**:

```cpp
Tensor binary_cross_entropy_with_logits(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t reduction) {

  // Numerically stable formula using log-sigmoid:
  // BCE(logit, y) = (1 - y) * logit - log_sigmoid(logit)
  // log_sigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))

  auto log_sigmoid_input = at::log_sigmoid(input);

  if (pos_weight_opt.has_value() && pos_weight_opt->defined()) {
      // pos_weight: weight for positive examples
      auto log_weight = (*pos_weight_opt - 1).mul(target).add_(1);
      log_sigmoid_input.mul_(log_weight);
  }

  Tensor loss = (1 - target).mul_(input).sub_(log_sigmoid_input);

  if (weight_opt.has_value() && weight_opt->defined()) {
      loss.mul_(*weight_opt);
  }

  return apply_loss_reduction(loss, reduction);
}
```

**Why BCEWithLogitsLoss is preferred**:
1. **Numerical stability**: Fused sigmoid + BCE avoids precision loss
2. **Avoids gradient issues**: No division by very small probabilities

### Gradients

**Gradient for BCE**:

```cpp
at::native::cpu_kernel(
    iter,
    [](scalar_t grad_val, scalar_t input_val, scalar_t target_val) {
      // d(BCE)/d(x) = -w (y - x) / (x - x²)
      //             = -w (y - x) / (x(1 - x))
      return grad_val * (input_val - target_val) /
          (scalar_t(std::max(
              (scalar_t(1) - input_val) * input_val,
              scalar_t(EPSILON))));  // Clamp denominator
    });
```

**Gradient Formula**:

```
∂BCE/∂x = -w * (y - x) / (x(1 - x))
```

---

## Mean Squared Error (MSE) Loss

### Algorithm

MSE loss (L2 loss) for regression tasks.

**Mathematical Formulation**:

```
MSE(x, y) = (x - y)²
```

With reduction:
```
If reduction == 'mean':
  output = (1/N) * Σ_i (x_i - y_i)²
Elif reduction == 'sum':
  output = Σ_i (x_i - y_i)²
Else:  # 'none'
  output = (x - y)²  # Element-wise
```

### PyTorch API

**Native Function Signature** (`native_functions.yaml:11862`):

```yaml
- func: mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
```

**Python API**:

```python
torch.nn.MSELoss(
    size_average=None,  # Deprecated
    reduce=None,        # Deprecated
    reduction='mean'    # 'none' | 'mean' | 'sum'
)
```

### Implementation

**Forward Pass** (`Loss.cpp:127-144`):

```cpp
TORCH_IMPL_FUNC(mse_loss_out)
(const Tensor& input, const Tensor& target, int64_t reduction, const Tensor& result) {

  TORCH_CHECK(input.device() == target.device(),
      "Expected all tensors to be on the same device");

  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);

    // Dispatch to backend-specific kernel
    mse_stub(iter.device_type(), iter);

    // Apply reduction
    if (reduction == Reduction::Mean) {
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    mse_stub(device_type(), *this);
  }
}
```

**MSE Kernel** (CPU implementation):

```cpp
void mse_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "mse_cpu", [&] {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
      auto diff = a - b;
      return diff * diff;
    });
  });
}
```

### Gradients

**Gradient Formula**:

```
∂MSE/∂x = 2 * (x - y) * grad_out / N  (if reduction='mean')
        = 2 * (x - y) * grad_out      (if reduction='sum' or 'none')
```

**PyTorch Implementation** (`native_functions.yaml:11873`):

```cpp
Tensor mse_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {

  Tensor grad_input = at::empty_like(self);

  auto iter = TensorIteratorConfig()
    .add_output(grad_input)
    .add_input(grad_output)
    .add_input(self)
    .add_input(target)
    .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      grad_input.scalar_type(), "mse_backward_cpu", [&] {
    cpu_kernel(iter, [](scalar_t grad, scalar_t input, scalar_t target) {
      return 2 * (input - target) * grad;
    });
  });

  if (reduction == Reduction::Mean) {
    grad_input.div_(self.numel());
  }

  return grad_input;
}
```

---

## L1 Loss (MAE)

### Algorithm

L1 loss (Mean Absolute Error) for regression.

**Mathematical Formulation**:

```
L1(x, y) = |x - y|
```

### PyTorch API

**Native Function Signature** (`native_functions.yaml:11879`):

```yaml
- func: l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
```

**Python API**:

```python
torch.nn.L1Loss(
    size_average=None,  # Deprecated
    reduce=None,        # Deprecated
    reduction='mean'    # 'none' | 'mean' | 'sum'
)
```

### Implementation

L1 loss uses the same pattern as MSE but with absolute difference:

```cpp
cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
  return std::abs(a - b);
});
```

### Gradients

**Gradient Formula**:

```
∂L1/∂x = sign(x - y) * grad_out / N  (if reduction='mean')
       = sign(x - y) * grad_out      (otherwise)
```

---

## Smooth L1 Loss (Huber Loss)

### Algorithm

Smooth L1 loss (also called Huber loss) is less sensitive to outliers than MSE, using L2 loss near zero and L1 loss for large errors.

**Mathematical Formulation**:

```
SmoothL1(x, y) = {
  0.5 * (x - y)² / β       if |x - y| < β
  |x - y| - 0.5 * β        otherwise
}
```

Where `β` (beta) controls the transition point (default: 1.0).

**Gradient**:

```
∂SmoothL1/∂x = {
  (x - y) / β              if |x - y| < β
  sign(x - y)              otherwise
}
```

### PyTorch API

**Native Function Signature** (`native_functions.yaml:12018`):

```yaml
- func: smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean,
                       float beta=1.0) -> Tensor
```

**Python API**:

```python
torch.nn.SmoothL1Loss(
    size_average=None,  # Deprecated
    reduce=None,        # Deprecated
    reduction='mean',   # 'none' | 'mean' | 'sum'
    beta=1.0            # Transition point
)
```

### Implementation

**Forward Kernel**:

```cpp
void smooth_l1_kernel(TensorIteratorBase& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      iter.dtype(), "smooth_l1_cpu", [&] {
    const scalar_t beta_val = beta;
    cpu_kernel(iter, [beta_val](scalar_t a, scalar_t b) -> scalar_t {
      auto z = std::abs(a - b);
      return z < beta_val
        ? 0.5 * z * z / beta_val
        : z - 0.5 * beta_val;
    });
  });
}
```

**Backward Kernel**:

```cpp
void smooth_l1_backward_kernel(TensorIteratorBase& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      iter.dtype(), "smooth_l1_backward_cpu", [&] {
    const scalar_t beta_val = beta;
    cpu_kernel(iter,
        [beta_val](scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
      const auto x = input - target;
      const auto abs_x = std::abs(x);

      if (abs_x < beta_val) {
        return grad_output * x / beta_val;
      } else {
        return grad_output * (x < 0 ? -1 : 1);
      }
    });
  });
}
```

**Huber Loss** is a variant with different parameterization (`delta` instead of `beta`):

```yaml
- func: huber_loss(Tensor self, Tensor target, int reduction=Mean,
                   float delta=1.0) -> Tensor
```

---

## CTC Loss

### Algorithm

Connectionist Temporal Classification (CTC) loss for sequence-to-sequence tasks with variable-length inputs/outputs (e.g., speech recognition, OCR).

**Problem Setup**:
- Input: Log-probabilities `[T, N, C]` (time steps, batch, num classes)
- Target: Variable-length sequences (e.g., "HELLO")
- Alignment unknown (e.g., "HH_EE_LL_LLL_O_O" where _ is blank)

**CTC computes the probability of all valid alignments** using dynamic programming (forward-backward algorithm).

**Mathematical Formulation**:

1. **Augmented Target Sequence** `l'`:
   Insert blanks between characters and at boundaries:
   ```
   Target: [H, E, L, L, O]
   Augmented: [BLANK, H, BLANK, E, BLANK, L, BLANK, L, BLANK, O, BLANK]
   Length: 2 * target_length + 1
   ```

2. **Forward Variables** `α[t, s]`:
   Probability of outputting augmented prefix `l'[0:s]` at time `t`

   Initialization (t=0):
   ```
   α[0, 0] = log_probs[0, BLANK]
   α[0, 1] = log_probs[0, l'[1]]  (first character)
   α[0, s] = -∞ for s > 1
   ```

   Recurrence (t > 0):
   ```
   α[t, s] = log_probs[t, l'[s]] + logsumexp(
       α[t-1, s],                              # Stay
       α[t-1, s-1],                            # Move from blank
       α[t-1, s-2] if l'[s] ≠ l'[s-2] else -∞  # Skip blank (no repeat)
   )
   ```

3. **Loss**:
   ```
   Total probability = logsumexp(α[T-1, |l'|-1], α[T-1, |l'|-2])
   CTC Loss = -log(total_probability)
   ```

### PyTorch API

**Native Function Signature** (`native_functions.yaml:2083`):

```yaml
- func: ctc_loss.IntList(Tensor log_probs, Tensor targets,
                         int[] input_lengths, int[] target_lengths,
                         int blank=0, int reduction=Mean,
                         bool zero_infinity=False) -> Tensor
```

**Python API**:

```python
torch.nn.CTCLoss(
    blank=0,            # Blank label index
    reduction='mean',   # 'none' | 'mean' | 'sum'
    zero_infinity=False # Replace infinite losses with 0
)
```

**Input Shapes**:
- `log_probs`: `[T, N, C]` (time, batch, num_classes) - log-probabilities
- `targets`: `[N, S]` or `[sum(target_lengths)]` - target sequences
- `input_lengths`: `[N]` - length of each sequence in batch
- `target_lengths`: `[N]` - length of each target

### Implementation

**Forward Pass - Alpha Calculation** (`LossCTC.cpp:127-226`):

```cpp
template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> ctc_loss_cpu_template(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {

  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  using target_t = std::conditional_t<target_scalar_type == kInt, int, int64_t>;

  int64_t batch_size = log_probs.size(1);

  // Allocate alpha (forward variables) and neg_log_likelihood
  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1},
                                log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  auto log_probs_a = log_probs.permute({1,0,2}).accessor<const scalar_t, 3>();
  auto log_alpha_a = log_alpha.accessor<scalar_t, 3>();
  auto targets_data = targets.const_data_ptr<target_t>();
  auto neg_log_likelihood_a = neg_log_likelihood.accessor<scalar_t, 1>();

  // Initialize first time step
  log_alpha.narrow(1, 0, 1).fill_(neginf);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      int64_t input_length = input_lengths[b];
      int64_t target_length = target_lengths[b];

      if (input_length == 0) {
        neg_log_likelihood_a[b] = target_length == 0 ? 0 : neginf;
        continue;
      }

      // Initialize α[0, 0] and α[0, 1]
      log_alpha_a[b][0][0] = log_probs_a[b][0][BLANK];
      if (target_length > 0) {
        log_alpha_a[b][0][1] = log_probs_a[b][0][get_target_prime(targets_data, b, 1, BLANK)];
      }

      // Forward pass: compute α[t, s] for all t, s
      for (const auto t : c10::irange(1, input_length)) {
        for (const auto s : c10::irange(2*target_length+1)) {
          auto current_target_prime = get_target_prime(targets_data, b, s, BLANK);

          // Three transitions: stay, move from blank, skip blank
          scalar_t la1 = log_alpha_a[b][t-1][s];
          scalar_t lamax = la1;
          scalar_t la2 = (s > 0) ? log_alpha_a[b][t-1][s-1] : neginf;
          if (la2 > lamax) lamax = la2;

          scalar_t la3 = neginf;
          if ((s > 1) && (get_target_prime(targets_data, b, s-2, BLANK) != current_target_prime)) {
            la3 = log_alpha_a[b][t-1][s-2];
            if (la3 > lamax) lamax = la3;
          }

          if (lamax == neginf) lamax = 0;

          // logsumexp(la1, la2, la3) + log_prob[t, current_target']
          log_alpha_a[b][t][s] = std::log(std::exp(la1-lamax) +
                                          std::exp(la2-lamax) +
                                          std::exp(la3-lamax)) + lamax +
                                 log_probs_a[b][t][current_target_prime];
        }
      }

      // Final loss: logsumexp of last two alphas
      if (target_length == 0) {
        neg_log_likelihood_a[b] = -log_alpha_a[b][input_length-1][0];
      } else {
        scalar_t l1 = log_alpha_a[b][input_length-1][target_length*2];
        scalar_t l2 = log_alpha_a[b][input_length-1][target_length*2-1];
        scalar_t m = std::max(l1, l2);
        m = (m == neginf) ? 0 : m;
        scalar_t log_likelihood = std::log(std::exp(l1-m) + std::exp(l2-m)) + m;
        neg_log_likelihood_a[b] = -log_likelihood;
      }
    }
  });

  return std::make_tuple(neg_log_likelihood, log_alpha);
}
```

**Backward Pass - Beta Calculation and Gradient** (`LossCTC.cpp:232-399`):

The backward pass:
1. Computes beta (backward variables) via recurrence (similar to alpha)
2. Computes gradient using: `∂L/∂log_prob[t,c] = exp(log_prob[t,c]) - exp(α[t,s] + β[t,s] - Z)` summed over all `s` where `l'[s] = c`

**Key Implementation Details**:
- Uses log-space calculations to avoid underflow
- Parallel over batch dimension
- Cascade sum for numerical stability

### Gradients

**Gradient Formula** (eq. 16 in Graves et al.):

```
∂L/∂log_prob[t, c] = (exp(log_prob[t, c]) - Σ_{s: l'[s]=c} exp(α[t,s] + β[t,s] - Z)) * grad_out
```

Where `Z = log_likelihood` is the normalizer.

---

## KL Divergence Loss

### Algorithm

KL divergence measures the difference between two probability distributions.

**Mathematical Formulation**:

```
For log_target = False:
  KL(P || Q) = Σ_i P[i] * (log(P[i]) - Q[i])
             = Σ_i P[i] * log(P[i] / exp(Q[i]))

For log_target = True:
  KL(P || Q) = Σ_i exp(Q_log[i]) * (Q_log[i] - P_log[i])
```

Where:
- `Q` is `input` (predicted distribution in log-space)
- `P` is `target` (true distribution, in log-space if `log_target=True`)

### PyTorch API

**Native Function Signature** (`native_functions.yaml:3303`):

```yaml
- func: kl_div(Tensor self, Tensor target, int reduction=Mean, bool log_target=False) -> Tensor
```

**Python API**:

```python
torch.nn.KLDivLoss(
    size_average=None,  # Deprecated
    reduce=None,        # Deprecated
    reduction='mean',   # 'none' | 'mean' | 'sum' | 'batchmean'
    log_target=False    # Whether target is in log-space
)
```

### Implementation

**Forward Pass** (`Loss.cpp:240-253`):

```cpp
Tensor kl_div(const Tensor& input, const Tensor& target,
              int64_t reduction, bool log_target) {

  TORCH_CHECK(!input.is_complex() && !target.is_complex(),
              "kl_div: Complex inputs not supported.");

  Tensor output;
  if (log_target) {
    // Both input and target are in log-space
    output = at::exp(target) * (target - input);
  } else {
    // input is log-space, target is probability
    // xlogy(a, b) = a * log(b), with xlogy(0, 0) = 0
    output = at::xlogy(target, target) - target * input;
  }

  return apply_loss_reduction(output, reduction);
}
```

**Reduction Modes**:
- `reduction='batchmean'`: Divide by batch size (first dimension)
- Other modes: Standard mean/sum/none

---

## Margin-Based Losses

### Triplet Margin Loss

**Algorithm**: Enforces that anchor-positive distance < anchor-negative distance by a margin.

```
Loss = max(0, margin + ||anchor - positive||_p - ||anchor - negative||_p)
```

**PyTorch API** (`native_functions.yaml:6505`):

```python
torch.nn.TripletMarginLoss(
    margin=1.0,      # Margin value
    p=2,             # Norm degree (2 for Euclidean)
    eps=1e-6,        # Small value to avoid division by zero
    swap=False,      # Use distance swap heuristic
    reduction='mean'
)
```

**Implementation** (`Loss.cpp:201-225`):

```cpp
Tensor triplet_margin_loss(
    const Tensor& anchor, const Tensor& positive, const Tensor& negative,
    double margin, double p, double eps, bool swap, int64_t reduction) {

  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);

  if (swap) {
    // Distance swap: use min(dist_neg, dist(positive, negative))
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    dist_neg = at::min(dist_neg, dist_swap);
  }

  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  return apply_loss_reduction(output, reduction);
}
```

### Cosine Embedding Loss

**Algorithm**: Enforces cosine similarity for similar pairs, dissimilarity for different pairs.

```
Loss = {
  1 - cos(x1, x2)                      if y = 1  (similar)
  max(0, cos(x1, x2) - margin)         if y = -1 (dissimilar)
}
```

**PyTorch API** (`native_functions.yaml:1873`):

```python
torch.nn.CosineEmbeddingLoss(
    margin=0.0,      # Margin for dissimilar pairs
    size_average=None,
    reduce=None,
    reduction='mean'
)
```

### Hinge Embedding Loss

**Algorithm**: Minimizes input for target=1, maximizes to margin for target=-1.

```
Loss = {
  x                     if y = 1
  max(0, margin - x)    if y = -1
}
```

---

## MLX Porting Recommendations

### 1. Cross-Entropy Loss

**MLX Equivalent**: `mlx.nn.losses.cross_entropy`

**Implementation Strategy**:

```cpp
// mlx/nn/losses.h
namespace mlx::nn::losses {

array cross_entropy(
    const array& logits,        // [N, C] unnormalized scores
    const array& targets,       // [N] class indices
    const std::vector<float>& weights = {},
    int axis = -1,
    float label_smoothing = 0.0,
    ReductionType reduction = ReductionType::Mean) {

  // Step 1: Log-softmax (numerically stable)
  array log_probs = mlx::core::log_softmax(logits, axis);

  // Step 2: Gather log-probabilities for target classes
  // For each sample i: log_probs[i, targets[i]]
  array nll = -mlx::core::take_along_axis(log_probs, expand_dims(targets, -1), axis);
  nll = squeeze(nll, -1);

  // Step 3: Apply class weights
  if (!weights.empty()) {
    array weight_tensor = array(weights);
    array sample_weights = weight_tensor[targets];
    nll = nll * sample_weights;
  }

  // Step 4: Apply reduction
  return apply_reduction(nll, reduction);
}

} // namespace mlx::nn::losses
```

**Metal Shader** (Log-Softmax kernel):

```metal
kernel void log_softmax_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant LogSoftmaxParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {

  const int batch_idx = gid.y;
  const int num_classes = params.num_classes;

  if (batch_idx >= params.batch_size) return;

  threadgroup float shared_max[256];
  threadgroup float shared_sum[256];

  // Step 1: Find max for numerical stability
  float local_max = -INFINITY;
  for (int c = tid.x; c < num_classes; c += 256) {
    int idx = batch_idx * num_classes + c;
    local_max = max(local_max, input[idx]);
  }
  shared_max[tid.x] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction to find global max
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid.x < stride) {
      shared_max[tid.x] = max(shared_max[tid.x], shared_max[tid.x + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float max_val = shared_max[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 2: Compute exp(x - max) and sum
  float local_sum = 0.0f;
  for (int c = tid.x; c < num_classes; c += 256) {
    int idx = batch_idx * num_classes + c;
    local_sum += exp(input[idx] - max_val);
  }
  shared_sum[tid.x] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction for sum
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid.x < stride) {
      shared_sum[tid.x] += shared_sum[tid.x + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sum_exp = shared_sum[0];
  float log_sum_exp = log(sum_exp);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 3: Compute log_softmax = x - max - log(sum(exp(x - max)))
  for (int c = tid.x; c < num_classes; c += 256) {
    int idx = batch_idx * num_classes + c;
    output[idx] = input[idx] - max_val - log_sum_exp;
  }
}
```

### 2. NLL Loss

**MLX Implementation**:

```cpp
array nll_loss(
    const array& log_probs,
    const array& targets,
    const std::vector<float>& weights = {},
    int ignore_index = -100,
    ReductionType reduction = ReductionType::Mean) {

  int batch_size = log_probs.shape(0);
  int num_classes = log_probs.shape(1);

  // Gather log-probabilities for target classes
  array losses = -mlx::core::take_along_axis(log_probs, expand_dims(targets, -1), 1);
  losses = squeeze(losses, -1);

  // Apply ignore_index mask
  if (ignore_index >= 0) {
    array mask = mlx::core::not_equal(targets, ignore_index);
    losses = losses * mask;
  }

  // Apply class weights
  if (!weights.empty()) {
    array weight_tensor = array(weights);
    array sample_weights = weight_tensor[targets];
    losses = losses * sample_weights;
  }

  // Compute total weight for mean reduction
  array total_weight;
  if (reduction == ReductionType::Mean) {
    if (!weights.empty()) {
      total_weight = mlx::core::sum(sample_weights);
    } else {
      total_weight = array(static_cast<float>(batch_size));
    }
  }

  // Apply reduction
  if (reduction == ReductionType::Mean) {
    return mlx::core::sum(losses) / total_weight;
  } else if (reduction == ReductionType::Sum) {
    return mlx::core::sum(losses);
  } else {
    return losses;
  }
}
```

### 3. MSE Loss

**MLX Implementation**:

```cpp
array mse_loss(const array& input, const array& target,
               ReductionType reduction = ReductionType::Mean) {
  array diff = input - target;
  array squared_diff = diff * diff;

  if (reduction == ReductionType::Mean) {
    return mlx::core::mean(squared_diff);
  } else if (reduction == ReductionType::Sum) {
    return mlx::core::sum(squared_diff);
  } else {
    return squared_diff;
  }
}
```

**Gradient**:

```cpp
array mse_loss_backward(const array& grad_output, const array& input,
                        const array& target, ReductionType reduction) {
  array grad = 2.0f * (input - target) * grad_output;

  if (reduction == ReductionType::Mean) {
    grad = grad / static_cast<float>(input.size());
  }

  return grad;
}
```

### 4. Smooth L1 Loss

**MLX Implementation**:

```cpp
array smooth_l1_loss(const array& input, const array& target,
                     float beta = 1.0,
                     ReductionType reduction = ReductionType::Mean) {
  array diff = mlx::core::abs(input - target);

  // Piecewise: 0.5 * z^2 / beta if z < beta, else z - 0.5 * beta
  array mask = diff < beta;
  array loss = mlx::core::where(
      mask,
      0.5f * diff * diff / beta,
      diff - 0.5f * beta
  );

  return apply_reduction(loss, reduction);
}
```

### 5. CTC Loss

**MLX Recommendation**: Use existing CTC implementation from Apple's CoreML or reference implementation.

**High-level approach**:
1. Implement forward-backward algorithm in C++
2. Use Metal for parallel alpha/beta computation
3. Leverage Apple's Accelerate framework for logsumexp

### 6. General Porting Considerations

**Numerical Stability**:

1. **Log-space calculations**: Always use log-softmax, not softmax + log
2. **Log-sum-exp trick**: `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))`
3. **Clamping**: Prevent -infinity gradients (e.g., clamp log to -100)

**Performance Optimizations**:

1. **Fuse operations**: Combine softmax + NLL in single kernel
2. **Parallel reductions**: Use threadgroup memory for sum/max
3. **Avoid data movement**: Keep intermediate results on GPU

**Testing Strategy**:

1. **Gradient checking**: Finite difference validation
2. **Numerical precision**: Compare against PyTorch with atol=1e-5
3. **Edge cases**: Empty batches, single-sample, extreme values

---

## Summary

This document provides comprehensive reverse-engineering of PyTorch's loss function implementations:

**Classification Losses**:
- **Cross-Entropy**: Fused log-softmax + NLL for numerical stability
- **NLL**: Expects log-probabilities, supports class weights and ignore_index
- **BCE**: For binary classification with probabilities

**Regression Losses**:
- **MSE**: L2 loss, gradient: `2(x - y)`
- **L1**: Mean absolute error, gradient: `sign(x - y)`
- **Smooth L1**: Huber loss, robust to outliers

**Sequence Losses**:
- **CTC**: Forward-backward algorithm for variable-length alignment

**Key Implementation Insights**:
- All losses support reduction modes (none/mean/sum)
- Numerical stability via log-space calculations and clamping
- Cascade sums for better floating-point precision
- Parallel computation over batch dimension

**MLX Porting Priorities**:
1. **High Priority**: Cross-entropy, NLL, MSE (fundamental for training)
2. **Medium Priority**: BCE, Smooth L1, L1
3. **Lower Priority**: CTC, margin losses (specialized use cases)

**Next Steps**: Continue with Phase 5 documentation (optimizers, schedulers, mixed-precision).
