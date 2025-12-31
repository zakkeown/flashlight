# PyTorch Loss Functions (torch.nn)

## Purpose

This document provides comprehensive documentation of PyTorch's loss function classes. Loss functions are critical for training neural networks as they measure the discrepancy between predictions and targets, providing gradients for optimization.

**Source**: [torch/nn/modules/loss.py](../../reference/pytorch/torch/nn/modules/loss.py)

## Architecture Overview

### Loss Function Hierarchy

All loss functions inherit from a base class hierarchy that provides common reduction functionality:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Module                                   │
│                            ↑                                     │
│                         _Loss                                    │
│                    (reduction: str)                              │
│                            ↑                                     │
│              ┌─────────────┴─────────────┐                      │
│         _WeightedLoss              (other losses)                │
│      (+ weight buffer)                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Reduction Modes

All loss functions support a `reduction` parameter:

| Mode | Behavior |
|------|----------|
| `'none'` | Return per-element loss |
| `'mean'` | Return mean of all losses (default) |
| `'sum'` | Return sum of all losses |

---

## Loss Function Categories

### Quick Reference

| Category | Loss Functions |
|----------|----------------|
| **Regression** | L1Loss, MSELoss, SmoothL1Loss, HuberLoss |
| **Classification** | CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss |
| **Probabilistic** | KLDivLoss, PoissonNLLLoss, GaussianNLLLoss |
| **Embedding/Metric** | CosineEmbeddingLoss, TripletMarginLoss, MarginRankingLoss, HingeEmbeddingLoss |
| **Multi-label** | MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss |
| **Sequence** | CTCLoss |
| **Other** | SoftMarginLoss |

---

## 1. Regression Losses

### L1Loss (Mean Absolute Error)

Measures the mean absolute error between predictions and targets.

**Formula**:
```
L(x, y) = |x - y|
```

```python
class L1Loss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)
```

**Usage**:
```python
loss = nn.L1Loss()
output = loss(predictions, targets)
```

**Shape**: Input and target can be any shape `(*)`.

---

### MSELoss (Mean Squared Error)

Measures the mean squared error (L2 loss) between predictions and targets.

**Formula**:
```
L(x, y) = (x - y)²
```

```python
class MSELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)
```

**Usage**:
```python
loss = nn.MSELoss()
output = loss(predictions, targets)
```

**Shape**: Input and target can be any shape `(*)`.

---

### SmoothL1Loss

Combines L1 and L2 loss - uses squared loss near zero (smooth gradient) and L1 loss for larger errors (outlier robust).

**Formula**:
```
         ┌ 0.5(x - y)² / β   if |x - y| < β
L(x,y) = │
         └ |x - y| - 0.5β    otherwise
```

**Parameters**:
- `beta` (float): Threshold where loss transitions from L2 to L1. Default: 1.0

```python
loss = nn.SmoothL1Loss(beta=1.0)
```

**Used in**: Fast R-CNN, object detection bounding box regression.

---

### HuberLoss

Similar to SmoothL1Loss but with delta-scaled L1 region.

**Formula**:
```
         ┌ 0.5(x - y)²              if |x - y| < δ
L(x,y) = │
         └ δ(|x - y| - 0.5δ)        otherwise
```

**Parameters**:
- `delta` (float): Threshold for L2→L1 transition. Default: 1.0

**Note**: When `delta=1`, HuberLoss equals SmoothL1Loss.

---

## 2. Classification Losses

### CrossEntropyLoss

The most commonly used loss for multi-class classification. Combines LogSoftmax and NLLLoss.

**Formula** (with class indices):
```
L(x, y) = -log(exp(x[y]) / Σⱼ exp(x[j]))
        = -x[y] + log(Σⱼ exp(x[j]))
```

**Parameters**:
- `weight` (Tensor): Manual rescaling weight for each class
- `ignore_index` (int): Target value to ignore. Default: -100
- `label_smoothing` (float): Smoothing factor in [0, 1]. Default: 0.0

```python
class CrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
```

**Usage**:
```python
# Basic usage with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5)  # (batch, num_classes)
target = torch.tensor([1, 0, 4])  # class indices
output = loss(input, target)

# With label smoothing
loss = nn.CrossEntropyLoss(label_smoothing=0.1)

# With class weights for imbalanced datasets
weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0])
loss = nn.CrossEntropyLoss(weight=weights)

# With soft targets (class probabilities)
target_probs = torch.softmax(torch.randn(3, 5), dim=1)
output = loss(input, target_probs)
```

**Shape**:
- Input: `(N, C)` or `(N, C, d₁, d₂, ..., dₖ)` for K-dimensional loss
- Target (indices): `(N)` or `(N, d₁, d₂, ..., dₖ)`
- Target (probabilities): Same shape as input

---

### NLLLoss (Negative Log Likelihood)

Expects log-probabilities as input (typically from LogSoftmax).

**Formula**:
```
L(x, y) = -w[y] × x[y]
```

**Parameters**:
- `weight` (Tensor): Per-class weights
- `ignore_index` (int): Target value to ignore

```python
log_softmax = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()

input = torch.randn(3, 5)
target = torch.tensor([1, 0, 4])
output = loss(log_softmax(input), target)
```

**Note**: `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`

---

### BCELoss (Binary Cross Entropy)

For binary classification when input is probabilities (after sigmoid).

**Formula**:
```
L(x, y) = -[y × log(x) + (1-y) × log(1-x)]
```

**Important**: Input must be probabilities in [0, 1]. Apply sigmoid first!

```python
m = nn.Sigmoid()
loss = nn.BCELoss()

input = torch.randn(3, 2)
target = torch.rand(3, 2)  # Probabilities in [0, 1]
output = loss(m(input), target)
```

**Warning**: BCELoss clamps log outputs to >= -100 to avoid infinite loss when x=0 or x=1.

---

### BCEWithLogitsLoss

Combines Sigmoid and BCELoss for numerical stability (log-sum-exp trick).

**Parameters**:
- `weight` (Tensor): Per-element weight
- `pos_weight` (Tensor): Weight of positive examples (for class imbalance)

```python
loss = nn.BCEWithLogitsLoss()

input = torch.randn(3, 2)  # Logits (raw scores)
target = torch.empty(3, 2).random_(2)  # Binary targets
output = loss(input, target)

# For imbalanced classes (3x more negatives than positives)
pos_weight = torch.tensor([3.0, 3.0])
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Advantage**: More numerically stable than `Sigmoid` + `BCELoss`.

---

## 3. Probabilistic/Divergence Losses

### KLDivLoss (Kullback-Leibler Divergence)

Measures divergence between two probability distributions.

**Formula**:
```
L(y_pred, y_true) = y_true × (log(y_true) - log(y_pred))
```

**Parameters**:
- `log_target` (bool): Whether target is in log-space. Default: False

**Important**:
- Input should be **log-probabilities** (use `log_softmax`)
- Use `reduction='batchmean'` for mathematically correct KL divergence

```python
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Input should be log-probabilities
input = F.log_softmax(torch.randn(3, 5), dim=1)
# Target should be probabilities
target = F.softmax(torch.randn(3, 5), dim=1)

output = kl_loss(input, target)
```

**Warning**: `reduction='mean'` doesn't give true KL divergence - use `'batchmean'`.

---

### PoissonNLLLoss

Negative log likelihood for Poisson-distributed targets.

**Formula** (when `log_input=True`):
```
L(x, y) = exp(x) - y × x
```

**Parameters**:
- `log_input` (bool): If True, input is log(λ). Default: True
- `full` (bool): Include Stirling approximation term. Default: False
- `eps` (float): Small value for numerical stability. Default: 1e-8

```python
loss = nn.PoissonNLLLoss()
log_lambda = torch.randn(5, 2)  # log of Poisson rate parameter
counts = torch.randint(0, 10, (5, 2)).float()  # Count data
output = loss(log_lambda, counts)
```

---

### GaussianNLLLoss

Negative log likelihood for Gaussian-distributed targets with predicted variance.

**Formula**:
```
L(μ, y, σ²) = 0.5 × [log(σ²) + (y - μ)² / σ²]
```

**Parameters**:
- `full` (bool): Include constant term. Default: False
- `eps` (float): Minimum variance. Default: 1e-6

```python
loss = nn.GaussianNLLLoss()

mean = torch.randn(5, 2)  # Predicted mean
target = torch.randn(5, 2)  # Observed values
var = torch.ones(5, 2)  # Predicted variance (heteroscedastic)

output = loss(mean, target, var)
```

---

## 4. Embedding/Metric Learning Losses

### CosineEmbeddingLoss

Measures cosine similarity between two embeddings, pushing similar pairs together and dissimilar pairs apart.

**Formula**:
```
         ┌ 1 - cos(x₁, x₂)                  if y = 1
L(x₁,x₂) = │
         └ max(0, cos(x₁, x₂) - margin)    if y = -1
```

**Parameters**:
- `margin` (float): Margin for negative pairs. Default: 0

```python
loss = nn.CosineEmbeddingLoss(margin=0.5)

x1 = torch.randn(3, 128)
x2 = torch.randn(3, 128)
target = torch.tensor([1, -1, 1])  # 1=similar, -1=dissimilar

output = loss(x1, x2, target)
```

---

### TripletMarginLoss

For metric learning with (anchor, positive, negative) triplets.

**Formula**:
```
L(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)
```
where d is the p-norm distance.

**Parameters**:
- `margin` (float): Margin value. Default: 1.0
- `p` (float): Norm degree for distance. Default: 2
- `eps` (float): Numerical stability. Default: 1e-6
- `swap` (bool): Use distance swap optimization. Default: False

```python
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

anchor = torch.randn(100, 128)
positive = torch.randn(100, 128)  # Similar to anchor
negative = torch.randn(100, 128)  # Dissimilar to anchor

output = triplet_loss(anchor, positive, negative)
```

---

### TripletMarginWithDistanceLoss

Like TripletMarginLoss but with custom distance function.

```python
# With cosine distance
triplet_loss = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
)

# With custom L-infinity distance
def l_infinity(x1, x2):
    return torch.max(torch.abs(x1 - x2), dim=1).values

triplet_loss = nn.TripletMarginWithDistanceLoss(
    distance_function=l_infinity,
    margin=1.5
)
```

---

### MarginRankingLoss

For ranking tasks - ensures x1 ranks higher than x2 when y=1.

**Formula**:
```
L(x1, x2, y) = max(0, -y × (x1 - x2) + margin)
```

```python
loss = nn.MarginRankingLoss(margin=0.5)

x1 = torch.randn(3)
x2 = torch.randn(3)
target = torch.tensor([1, -1, 1])  # 1 means x1 should be larger

output = loss(x1, x2, target)
```

---

### HingeEmbeddingLoss

For binary classification with embeddings.

**Formula**:
```
         ┌ x                          if y = 1
L(x, y) = │
         └ max(0, margin - x)        if y = -1
```

```python
loss = nn.HingeEmbeddingLoss(margin=1.0)

input = torch.randn(3, 5)
target = torch.randn(3, 5).sign()  # -1 or 1

output = loss(input, target)
```

---

## 5. Multi-Label Losses

### MultiLabelMarginLoss

Multi-class multi-label hinge loss.

**Formula**:
```
L(x, y) = Σᵢⱼ max(0, 1 - (x[y[j]] - x[i])) / |x|
```

```python
loss = nn.MultiLabelMarginLoss()

x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
# y specifies which classes are present; -1 ends the list
y = torch.LongTensor([[3, 0, -1, 1]])  # Classes 3 and 0 are present

output = loss(x, y)
```

---

### MultiLabelSoftMarginLoss

One-versus-all loss for multi-label classification.

**Formula**:
```
L(x, y) = -1/C × Σᵢ [yᵢ × log(σ(xᵢ)) + (1-yᵢ) × log(1-σ(xᵢ))]
```

```python
loss = nn.MultiLabelSoftMarginLoss()

input = torch.randn(3, 5)
target = torch.empty(3, 5).random_(2)  # Binary targets for each class

output = loss(input, target)
```

---

### MultiMarginLoss

Multi-class classification hinge loss.

**Formula**:
```
L(x, y) = Σᵢ max(0, margin - x[y] + x[i])^p / |x|
```

**Parameters**:
- `p` (int): 1 or 2. Default: 1
- `margin` (float): Default: 1.0
- `weight` (Tensor): Per-class weights

```python
loss = nn.MultiMarginLoss(p=1, margin=1.0)

x = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
y = torch.tensor([3])  # Correct class index

output = loss(x, y)
```

---

## 6. Sequence Losses

### CTCLoss (Connectionist Temporal Classification)

For sequence-to-sequence tasks without pre-aligned targets (e.g., speech recognition, OCR).

**Parameters**:
- `blank` (int): Blank label index. Default: 0
- `zero_infinity` (bool): Zero infinite losses. Default: False

**Shape**:
- Log_probs: `(T, N, C)` - T=time, N=batch, C=classes (including blank)
- Targets: `(N, S)` or concatenated `(sum(target_lengths),)`
- Input_lengths: `(N,)` - actual lengths of each input sequence
- Target_lengths: `(N,)` - actual lengths of each target sequence

```python
ctc_loss = nn.CTCLoss(blank=0)

T = 50  # Input sequence length
C = 20  # Number of classes (including blank)
N = 16  # Batch size
S = 30  # Target sequence length

# Log probabilities from network
log_probs = torch.randn(T, N, C).log_softmax(2)

# Target sequences (class indices, no blank)
targets = torch.randint(1, C, (N, S), dtype=torch.long)

# Lengths
input_lengths = torch.full((N,), T, dtype=torch.long)
target_lengths = torch.randint(10, S, (N,), dtype=torch.long)

loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

---

## 7. Other Losses

### SoftMarginLoss

Two-class logistic loss (sigmoid + cross entropy for binary labels).

**Formula**:
```
L(x, y) = log(1 + exp(-y × x)) / |x|
```

Target should be 1 or -1.

```python
loss = nn.SoftMarginLoss()

input = torch.randn(3, 5)
target = torch.randn(3, 5).sign()  # -1 or 1

output = loss(input, target)
```

---

## MLX Mapping

### Direct Equivalents

| PyTorch | MLX |
|---------|-----|
| `nn.L1Loss()` | `mx.abs(pred - target).mean()` |
| `nn.MSELoss()` | `mx.square(pred - target).mean()` |
| `nn.CrossEntropyLoss()` | `mx.nn.losses.cross_entropy()` |
| `nn.BCELoss()` | `mx.nn.losses.binary_cross_entropy()` |

### MLX Implementation Patterns

```python
import mlx.core as mx
import mlx.nn as nn

# MSE Loss
def mse_loss(pred, target, reduction='mean'):
    loss = mx.square(pred - target)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# L1 Loss
def l1_loss(pred, target, reduction='mean'):
    loss = mx.abs(pred - target)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# Cross Entropy Loss (with logits)
def cross_entropy_loss(logits, targets, reduction='mean'):
    """
    logits: (N, C) raw scores
    targets: (N,) class indices
    """
    # Log-softmax for numerical stability
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Gather correct class log-probabilities
    nll = -mx.take_along_axis(log_probs, targets[:, None], axis=1).squeeze(-1)
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    return nll

# Binary Cross Entropy with Logits
def bce_with_logits_loss(logits, targets, reduction='mean'):
    """Numerically stable BCE with logits."""
    # log(sigmoid(x)) = x - softplus(x)
    # log(1 - sigmoid(x)) = -softplus(x)
    pos_loss = mx.maximum(-logits, 0) + mx.log1p(mx.exp(-mx.abs(logits)))
    neg_loss = logits + pos_loss
    loss = targets * neg_loss + (1 - targets) * pos_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# Smooth L1 Loss
def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    diff = mx.abs(pred - target)
    loss = mx.where(
        diff < beta,
        0.5 * mx.square(diff) / beta,
        diff - 0.5 * beta
    )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# Triplet Margin Loss
def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, reduction='mean'):
    d_pos = mx.sum(mx.power(mx.abs(anchor - positive), p), axis=-1) ** (1/p)
    d_neg = mx.sum(mx.power(mx.abs(anchor - negative), p), axis=-1) ** (1/p)
    loss = mx.maximum(d_pos - d_neg + margin, 0)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# KL Divergence
def kl_div_loss(log_pred, target, reduction='batchmean'):
    """
    log_pred: log-probabilities from model
    target: probabilities (ground truth distribution)
    """
    loss = target * (mx.log(target + 1e-10) - log_pred)
    if reduction == 'batchmean':
        return loss.sum() / log_pred.shape[0]
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Built-in losses** | ~23 loss classes | ~5 loss functions |
| **Class vs function** | Class-based with state | Functional style |
| **Reduction** | Via class parameter | Via function parameter |
| **Weight support** | Built-in for many losses | Manual implementation |
| **ignore_index** | Built-in for classification | Manual masking |
| **Label smoothing** | Built-in for CE | Manual implementation |

---

## Summary

### When to Use Each Loss

| Task | Recommended Loss |
|------|------------------|
| Multi-class classification | CrossEntropyLoss |
| Binary classification | BCEWithLogitsLoss |
| Multi-label classification | BCEWithLogitsLoss or MultiLabelSoftMarginLoss |
| Regression | MSELoss or L1Loss |
| Robust regression | HuberLoss or SmoothL1Loss |
| Sequence labeling | CTCLoss |
| Metric learning | TripletMarginLoss or CosineEmbeddingLoss |
| Distribution matching | KLDivLoss |
| Ranking | MarginRankingLoss |

### Quick Reference Table

| Loss | Input | Target | Use Case |
|------|-------|--------|----------|
| L1Loss | Any | Same as input | MAE regression |
| MSELoss | Any | Same as input | L2 regression |
| CrossEntropyLoss | (N, C) logits | (N,) indices or (N, C) probs | Classification |
| BCEWithLogitsLoss | Any | Same as input, in [0,1] | Binary classification |
| KLDivLoss | Log-probs | Probs | Distribution matching |
| TripletMarginLoss | (N, D) | anchor, pos, neg | Metric learning |
| CTCLoss | (T, N, C) log-probs | Sequences | Speech/OCR |
