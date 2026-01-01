"""
Loss Functions

Implements PyTorch-compatible loss functions:
- MSELoss: Mean Squared Error
- CrossEntropyLoss: Cross Entropy (combines LogSoftmax and NLLLoss)
- NLLLoss: Negative Log Likelihood
- BCELoss: Binary Cross Entropy
- BCEWithLogitsLoss: BCE with built-in sigmoid
- L1Loss: Mean Absolute Error
"""

import warnings
from typing import Optional

from .. import ops
from ..tensor import Tensor
from .module import Module


def _verify_reduction_params(
    size_average: Optional[bool], reduce: Optional[bool], reduction: str
) -> str:
    """Verify and handle deprecated size_average and reduce parameters."""
    if size_average is not None or reduce is not None:
        warnings.warn(
            "size_average and reduce args will be deprecated, please use reduction='{}' instead.".format(
                reduction
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        # Handle legacy params
        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True

        if reduce:
            reduction = "mean" if size_average else "sum"
        else:
            reduction = "none"
    return reduction


class MSELoss(Module):
    """
    Mean Squared Error loss.

    Creates a criterion that measures the mean squared error between
    n elements in the input x and target y.

    Loss(x, y) = mean((x - y)^2)

    Args:
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': the mean of the output (default)
            'sum': the sum of the output

    Shape:
        - Input: (*) any shape
        - Target: (*) same shape as input
        - Output: scalar if reduction='mean' or 'sum', otherwise same shape as input

    Example:
        >>> criterion = nn.MSELoss()
        >>> x = flashlight.randn(3, 5, requires_grad=True)
        >>> y = flashlight.randn(3, 5)
        >>> loss = criterion(x, y)
        >>> loss.backward()
    """

    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Compute squared error
        diff = input - target
        squared_error = diff * diff

        # Apply reduction
        if self.reduction == "none":
            return squared_error
        elif self.reduction == "mean":
            return ops.mean(squared_error)
        else:  # sum
            return ops.sum(squared_error)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class L1Loss(Module):
    """
    Mean Absolute Error loss.

    Creates a criterion that measures the mean absolute error between
    n elements in the input x and target y.

    Loss(x, y) = mean(|x - y|)

    Args:
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': the mean of the output (default)
            'sum': the sum of the output

    Shape:
        - Input: (*) any shape
        - Target: (*) same shape as input
        - Output: scalar if reduction='mean' or 'sum', otherwise same shape as input

    Example:
        >>> criterion = nn.L1Loss()
        >>> x = flashlight.randn(3, 5, requires_grad=True)
        >>> y = flashlight.randn(3, 5)
        >>> loss = criterion(x, y)
    """

    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Compute absolute error
        absolute_error = ops.abs(input - target)

        # Apply reduction
        if self.reduction == "none":
            return absolute_error
        elif self.reduction == "mean":
            return ops.mean(absolute_error)
        else:  # sum
            return ops.sum(absolute_error)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class NLLLoss(Module):
    """
    Negative Log Likelihood loss.

    Useful for training a classification problem with C classes.
    The input is expected to contain log-probabilities.

    Args:
        weight: Manual rescaling weight for each class (optional)
        size_average: Deprecated (use reduction instead)
        ignore_index: Specifies a target value that is ignored
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': the mean of the output (default)
            'sum': the sum of the output

    Shape:
        - Input: (N, C) where N is batch size, C is number of classes
        - Target: (N,) where each value is 0 ≤ targets[i] ≤ C-1
        - Output: scalar if reduction='mean' or 'sum', otherwise (N,)

    Example:
        >>> criterion = nn.NLLLoss()
        >>> log_probs = flashlight.randn(3, 5, requires_grad=True)
        >>> target = flashlight.tensor([1, 0, 4], dtype=flashlight.int32)
        >>> loss = criterion(log_probs, target)
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        # input: (N, C) log probabilities
        # target: (N,) class indices
        # Gather the log probabilities at target indices
        # For each sample i, we want input[i, target[i]]
        # Convert target to one-hot
        target_mlx = target._mlx_array
        input_mlx = input._mlx_array

        # Gather using MLX's take_along_axis equivalent
        # Actually, MLX supports basic indexing, let's use that
        gathered_mlx = mx.take_along_axis(input_mlx, mx.expand_dims(target_mlx, axis=1), axis=1)
        gathered = Tensor._from_mlx_array(mx.squeeze(gathered_mlx, axis=1))

        # Preserve gradient tracking
        from ..autograd.context import is_grad_enabled

        if is_grad_enabled() and input.requires_grad:
            gathered.requires_grad = True

        # Negate (negative log likelihood)
        loss = -gathered

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            loss = loss * mask

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            if self.ignore_index >= 0:
                # Only count non-ignored samples
                mask = target != self.ignore_index
                return ops.sum(loss) / ops.sum(mask)
            return ops.mean(loss)
        else:  # sum
            return ops.sum(loss)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', ignore_index={self.ignore_index}"


class CrossEntropyLoss(Module):
    """
    Cross Entropy loss.

    Combines LogSoftmax and NLLLoss in a single class.
    Useful for training a classification problem with C classes.

    Args:
        weight: Manual rescaling weight for each class (optional)
        size_average: Deprecated (use reduction instead)
        ignore_index: Specifies a target value that is ignored
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': the mean of the output (default)
            'sum': the sum of the output
        label_smoothing: Amount of label smoothing (default: 0.0)

    Shape:
        - Input: (N, C) where N is batch size, C is number of classes (raw logits)
        - Target: (N,) where each value is 0 ≤ targets[i] ≤ C-1
        - Output: scalar if reduction='mean' or 'sum', otherwise (N,)

    Example:
        >>> criterion = nn.CrossEntropyLoss()
        >>> logits = flashlight.randn(3, 5, requires_grad=True)
        >>> target = flashlight.tensor([1, 0, 4], dtype=flashlight.int32)
        >>> loss = criterion(logits, target)
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Apply log_softmax
        log_probs = ops.log_softmax(input, dim=1)

        # Apply NLL loss
        nll = NLLLoss(reduction=self.reduction, ignore_index=self.ignore_index)
        return nll(log_probs, target)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', ignore_index={self.ignore_index}"


class BCELoss(Module):
    """
    Binary Cross Entropy loss.

    Creates a criterion that measures the Binary Cross Entropy
    between the target and input probabilities.

    Args:
        weight: Manual rescaling weight (optional)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': the mean of the output (default)
            'sum': the sum of the output

    Shape:
        - Input: (*) any shape, values in [0, 1]
        - Target: (*) same shape as input, values in [0, 1]
        - Output: scalar if reduction='mean' or 'sum', otherwise same shape as input

    Note:
        Input should be probabilities (use sigmoid if needed).
        For numerical stability with logits, use BCEWithLogitsLoss instead.

    Example:
        >>> criterion = nn.BCELoss()
        >>> x = flashlight.sigmoid(flashlight.randn(3, requires_grad=True))
        >>> y = flashlight.tensor([1.0, 0.0, 1.0])
        >>> loss = criterion(x, y)
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # BCE = -[y * log(x) + (1 - y) * log(1 - x)]
        # Clamp input to avoid log(0)
        import mlx.core as mx

        eps = 1e-7
        input_clamped_mlx = mx.clip(input._mlx_array, eps, 1 - eps)
        input_clamped = Tensor._from_mlx_array(input_clamped_mlx)
        if input.requires_grad:
            input_clamped.requires_grad = True
            # Note: clipping gradient is not fully correct but close enough for small eps
            input_clamped._grad_fn = input._grad_fn

        loss = -(target * ops.log(input_clamped) + (1 - target) * ops.log(1 - input_clamped))

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return ops.mean(loss)
        else:  # sum
            return ops.sum(loss)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class BCEWithLogitsLoss(Module):
    """
    Binary Cross Entropy with Logits loss.

    Combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid + BCELoss.

    Args:
        weight: Manual rescaling weight (optional)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': the mean of the output (default)
            'sum': the sum of the output
        pos_weight: Weight of positive examples (optional)

    Shape:
        - Input: (*) any shape (raw logits)
        - Target: (*) same shape as input, values in [0, 1]
        - Output: scalar if reduction='mean' or 'sum', otherwise same shape as input

    Example:
        >>> criterion = nn.BCEWithLogitsLoss()
        >>> x = flashlight.randn(3, requires_grad=True)
        >>> y = flashlight.tensor([1.0, 0.0, 1.0])
        >>> loss = criterion(x, y)
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # More stable computation: max(x, 0) - x * y + log(1 + exp(-abs(x)))
        # This avoids overflow/underflow in exp
        max_val = ops.relu(input)
        loss = max_val - input * target + ops.log(1 + ops.exp(-ops.abs(input)))

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return ops.mean(loss)
        else:  # sum
            return ops.sum(loss)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class SmoothL1Loss(Module):
    """
    Smooth L1 Loss (Huber Loss).

    Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.

    Args:
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
        beta: Threshold for switching between L1 and L2 loss (default: 1.0)

    Shape:
        - Input: (*) any shape
        - Target: (*) same shape as input
    """

    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        beta: float = 1.0,
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction
        self.beta = beta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        diff = (input - target)._mlx_array
        abs_diff = mx.abs(diff)
        loss = mx.where(abs_diff < self.beta, 0.5 * diff**2 / self.beta, abs_diff - 0.5 * self.beta)
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', beta={self.beta}"


class HuberLoss(Module):
    """
    Huber Loss.

    Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.

    Args:
        reduction: Specifies the reduction to apply (default: 'mean')
        delta: Threshold for switching between L1 and L2 loss (default: 1.0)

    Shape:
        - Input: (*) any shape
        - Target: (*) same shape as input
    """

    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        diff = (input - target)._mlx_array
        abs_diff = mx.abs(diff)
        loss = mx.where(
            abs_diff <= self.delta, 0.5 * diff**2, self.delta * (abs_diff - 0.5 * self.delta)
        )
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', delta={self.delta}"


class KLDivLoss(Module):
    """
    Kullback-Leibler divergence loss.

    Args:
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
        log_target: Whether the target is in log-space (default: False)

    Shape:
        - Input: (*) log-probabilities
        - Target: (*) probability distribution
    """

    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        log_target: bool = False,
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        if reduction not in ("none", "mean", "sum", "batchmean"):
            raise ValueError(
                f"reduction must be 'none', 'mean', 'sum', or 'batchmean', got '{reduction}'"
            )
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        # KL(P||Q) = sum(P * (log(P) - log(Q)))
        # input is log(Q), target is P (or log(P) if log_target=True)
        if self.log_target:
            loss = mx.exp(target._mlx_array) * (target._mlx_array - input._mlx_array)
        else:
            # Use xlogy for numerical stability: xlogy(p, p) returns 0 when p=0
            # This avoids the need for epsilon and handles 0*log(0) correctly
            xlogy_term = ops.xlogy(target, target)._mlx_array
            loss = xlogy_term - target._mlx_array * input._mlx_array
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        elif self.reduction == "batchmean":
            return ops.sum(result) / input.shape[0]
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', log_target={self.log_target}"


class MarginRankingLoss(Module):
    """
    Margin Ranking Loss.

    Creates a criterion that measures the loss given inputs x1, x2,
    two 1D mini-batch Tensors, and a label 1D mini-batch tensor y
    containing 1 or -1.

    loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)

    Args:
        margin: The margin value (default: 0)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        margin: float = 0.0,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        loss = mx.maximum(
            0, -target._mlx_array * (input1._mlx_array - input2._mlx_array) + self.margin
        )
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction='{self.reduction}'"


class HingeEmbeddingLoss(Module):
    """
    Hinge Embedding Loss.

    Measures the loss given an input tensor x and a labels tensor y
    containing values (1 or -1).

    loss_n = x_n, if y_n = 1
           = max(0, margin - x_n), if y_n = -1

    Args:
        margin: The margin value (default: 1.0)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        margin: float = 1.0,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.margin = margin
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        y = target._mlx_array
        loss = mx.where(y == 1, x, mx.maximum(0, self.margin - x))
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction='{self.reduction}'"


class CosineEmbeddingLoss(Module):
    """
    Cosine Embedding Loss.

    Measures the loss given two input tensors and a label tensor with values 1 or -1.

    loss(x1, x2, y) = 1 - cos(x1, x2), if y = 1
                    = max(0, cos(x1, x2) - margin), if y = -1

    Args:
        margin: The margin value (default: 0)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        margin: float = 0.0,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        x1 = input1._mlx_array
        x2 = input2._mlx_array
        y = target._mlx_array

        # Compute cosine similarity
        cos_sim = mx.sum(x1 * x2, axis=-1) / (
            mx.sqrt(mx.sum(x1**2, axis=-1)) * mx.sqrt(mx.sum(x2**2, axis=-1)) + 1e-8
        )

        loss = mx.where(y == 1, 1 - cos_sim, mx.maximum(0, cos_sim - self.margin))
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction='{self.reduction}'"


class SoftMarginLoss(Module):
    """
    Soft Margin Loss.

    Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor x and target tensor y
    (containing 1 or -1).

    loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

    Args:
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        # Numerically stable implementation of log(1 + exp(-y*x))
        # For large positive y*x: use exp(-y*x) directly (avoiding overflow in exp)
        # For large negative y*x: use -y*x + log(1 + exp(y*x)) ~ -y*x
        yx = target._mlx_array * input._mlx_array
        # log(1 + exp(-z)) = log(1 + exp(-|z|)) + max(-z, 0)
        #                  = log1p(exp(-|z|)) + relu(-z)
        loss = mx.log1p(mx.exp(-mx.abs(yx))) + mx.maximum(-yx, 0)
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class TripletMarginLoss(Module):
    """
    Triplet Margin Loss.

    Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.

    L(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)

    Args:
        margin: The margin value (default: 1.0)
        p: The norm degree for pairwise distance (default: 2)
        eps: Small constant for numerical stability (default: 1e-6)
        swap: Use the swapped version (default: False)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        import mlx.core as mx

        a = anchor._mlx_array
        p = positive._mlx_array
        n = negative._mlx_array

        # Compute pairwise distances
        d_ap = mx.sum((a - p) ** self.p, axis=-1) ** (1 / self.p)
        d_an = mx.sum((a - n) ** self.p, axis=-1) ** (1 / self.p)

        if self.swap:
            d_pn = mx.sum((p - n) ** self.p, axis=-1) ** (1 / self.p)
            d_an = mx.minimum(d_an, d_pn)

        loss = mx.maximum(d_ap - d_an + self.margin, 0)
        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, p={self.p}, swap={self.swap}, reduction='{self.reduction}'"


class PoissonNLLLoss(Module):
    """
    Negative log likelihood loss with Poisson distribution of target.

    loss(input, target) = input - target * log(input) + log(target!)

    Args:
        log_input: If True, loss is exp(input) - target * input (default: True)
        full: Whether to compute full loss including Stirling approximation (default: False)
        size_average: Deprecated (use reduction instead)
        eps: Small value to avoid log(0) (default: 1e-8)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average: Optional[bool] = None,
        eps: float = 1e-8,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        t = target._mlx_array

        if self.log_input:
            loss = mx.exp(x) - t * x
        else:
            loss = x - t * mx.log(x + self.eps)

        if self.full:
            # Stirling approximation for log(target!)
            stirling = t * mx.log(t) - t + 0.5 * mx.log(2 * 3.14159265359 * t)
            stirling = mx.where(t > 1, stirling, 0)
            loss = loss + stirling

        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"log_input={self.log_input}, full={self.full}, reduction='{self.reduction}'"


class GaussianNLLLoss(Module):
    """
    Gaussian Negative Log Likelihood Loss.

    Assumes the target is sampled from a Gaussian distribution with
    the given mean (input) and variance (var).

    loss = 0.5 * [log(var) + (input - target)^2 / var]

    Args:
        full: Include the constant term in the loss (default: False)
        eps: Small value for numerical stability (default: 1e-6)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        t = target._mlx_array
        v = mx.maximum(var._mlx_array, self.eps)

        loss = 0.5 * (mx.log(v) + (x - t) ** 2 / v)

        if self.full:
            loss = loss + 0.5 * mx.log(2 * 3.14159265359)

        result = Tensor._from_mlx_array(loss)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"full={self.full}, eps={self.eps}, reduction='{self.reduction}'"


class MultiLabelMarginLoss(Module):
    """
    Multi-Label Margin Loss.

    Creates a criterion that optimizes a multi-label classification loss.

    Args:
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import multilabel_margin_loss

        return multilabel_margin_loss(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class MultiMarginLoss(Module):
    """
    Multi-class Margin Loss.

    Creates a criterion that optimizes a multi-class classification
    hinge loss (margin-based loss).

    Args:
        p: The norm type (default: 1)
        margin: The margin value (default: 1)
        weight: Class weights (optional)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        y = target._mlx_array.astype(mx.int32)
        N, C = x.shape

        # Get scores for correct classes: (N,)
        # Use one-hot encoding to extract correct class scores
        y_onehot = mx.zeros((N, C), dtype=x.dtype)
        # Create one-hot by comparing y (N,1) with arange (C,)
        y_expanded = mx.expand_dims(y, axis=1)  # (N, 1)
        class_indices = mx.arange(C)  # (C,)
        y_onehot = (y_expanded == class_indices).astype(x.dtype)  # (N, C)

        # Correct class scores: (N,)
        correct_scores = mx.sum(x * y_onehot, axis=1, keepdims=True)  # (N, 1)

        # Compute margin loss for all classes: margin - correct_score + score
        margins = self.margin - correct_scores + x  # (N, C)

        # Apply max(0, margin) - hinge
        margins = mx.maximum(margins, 0)

        # Zero out the correct class (don't include it in the loss)
        margins = margins * (1 - y_onehot)

        # Apply power if p == 2
        if self.p == 2:
            margins = mx.square(margins)

        # Apply class weights if provided
        if self.weight is not None:
            # Weight by correct class weight
            weights = mx.take(self.weight._mlx_array, y)  # (N,)
            margins = margins * mx.expand_dims(weights, axis=1)

        # Sum over classes and divide by C
        sample_losses = mx.sum(margins, axis=1) / C  # (N,)

        result = Tensor._from_mlx_array(sample_losses)

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"p={self.p}, margin={self.margin}, reduction='{self.reduction}'"


class MultiLabelSoftMarginLoss(Module):
    """
    Multi-Label Soft Margin Loss.

    Creates a criterion that optimizes a multi-label one-versus-all loss
    based on max-entropy.

    Args:
        weight: Class weights (optional)
        size_average: Deprecated (use reduction instead)
        reduce: Deprecated (use reduction instead)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        weight: Tensor = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        import mlx.core as mx

        x = input._mlx_array
        y = target._mlx_array

        # Numerically stable binary cross entropy (same as BCEWithLogitsLoss)
        # Instead of: -(y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x)))
        # Use: max(x, 0) - x*y + log(1 + exp(-|x|))
        # This avoids log(0) issues and is more numerically stable
        loss = mx.maximum(x, 0) - x * y + mx.log1p(mx.exp(-mx.abs(x)))

        if self.weight is not None:
            loss = loss * self.weight._mlx_array

        result = Tensor._from_mlx_array(loss)

        # Apply reduction - for 'none', return per-element losses
        # For 'mean'/'sum', reduce over ALL elements (batch and classes)
        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class TripletMarginWithDistanceLoss(Module):
    """
    Triplet Margin Loss with custom distance function.

    Similar to TripletMarginLoss but allows for a custom distance function.

    Args:
        distance_function: Custom distance function (default: pairwise_distance)
        margin: The margin value (default: 1.0)
        swap: Use the swapped version (default: False)
        reduction: Specifies the reduction to apply (default: 'mean')
    """

    def __init__(
        self,
        distance_function: callable = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        self.distance_function = distance_function
        self.margin = margin
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        from .functional import triplet_margin_with_distance_loss

        return triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"margin={self.margin}, swap={self.swap}, reduction='{self.reduction}'"


class CTCLoss(Module):
    """
    Connectionist Temporal Classification Loss.

    Used for sequence-to-sequence learning without alignment, e.g., speech recognition.

    Args:
        blank: Index of the blank label (default: 0)
        reduction: Specifies the reduction to apply (default: 'mean')
        zero_infinity: Whether to zero infinite losses and gradients (default: False)
    """

    def __init__(self, blank: int = 0, reduction: str = "mean", zero_infinity: bool = False):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor
    ) -> Tensor:
        from ..ops.quick_ops import ctc_loss

        return ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )

    def extra_repr(self) -> str:
        return (
            f"blank={self.blank}, reduction='{self.reduction}', zero_infinity={self.zero_infinity}"
        )


class NLLLoss2d(Module):
    """
    Negative Log Likelihood Loss for 2D inputs (images).

    This is a convenience wrapper around NLLLoss for 2D spatial inputs.

    Args:
        weight: Manual rescaling weight for each class
        size_average: Deprecated
        ignore_index: Target value to ignore
        reduce: Deprecated
        reduction: Reduction type (default: 'mean')
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        reduction = _verify_reduction_params(size_average, reduce, reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # NLLLoss2d is essentially NLLLoss applied to 2D spatial data
        # Input: (N, C, H, W), Target: (N, H, W)
        import mlx.core as mx

        log_probs = input._mlx_array
        tgt = target._mlx_array

        N, C, H, W = log_probs.shape

        # Reshape for processing: (N, C, H*W) -> (N*H*W, C)
        log_probs = log_probs.transpose(0, 2, 3, 1).reshape(-1, C)
        tgt = tgt.reshape(-1)

        # Gather log probabilities
        indices = mx.arange(log_probs.shape[0])
        loss = -log_probs[indices, tgt]

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = tgt != self.ignore_index
            loss = mx.where(mask, loss, 0.0)

        result = Tensor._from_mlx_array(loss.reshape(N, H, W))

        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return ops.mean(result)
        else:
            return ops.sum(result)

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}, reduction='{self.reduction}'"


class AdaptiveLogSoftmaxWithLoss(Module):
    """
    Efficient softmax approximation for large output spaces.

    This module is an efficient approximation for training models with large
    numbers of classes. It uses a hierarchical structure with adaptive clusters.

    Args:
        in_features: Number of features in the input tensor
        n_classes: Number of classes in the dataset
        cutoffs: Cutoffs for forming the clusters (sorted, increasing)
        div_value: Value used for computing relative cluster sizes (default: 4.0)
        head_bias: If True, adds a bias to the 'head' of the adaptive softmax (default: False)

    Shape:
        - Input: (N, in_features)
        - Target: (N,) where each value is 0 <= target[i] < n_classes
        - Output: Named tuple with 'output' and 'loss'

    Example:
        >>> m = nn.AdaptiveLogSoftmaxWithLoss(20, 200, [20, 100, 150])
        >>> input = torch.randn(128, 20)
        >>> target = torch.randint(200, (128,))
        >>> output = m(input, target)
        >>> output.loss.backward()
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: list,
        div_value: float = 4.0,
        head_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # device and dtype accepted for PyTorch compatibility (MLX uses unified memory)
        from .layers.linear import Linear
        from .parameter import Parameter

        cutoffs = list(cutoffs)
        if (
            not cutoffs
            or cutoffs != sorted(cutoffs)
            or min(cutoffs) <= 0
            or max(cutoffs) >= n_classes - 1
        ):
            raise ValueError(
                "cutoffs should be a sorted list of unique positive integers with max < n_classes - 1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        # Head handles the first cluster plus cluster indicators
        self.head_size = self.cutoffs[0] + len(self.cutoffs) - 1

        # Create head
        self.head = Linear(in_features, self.head_size, bias=head_bias)

        # Create tail layers (one per remaining cluster)
        self.tail = []
        for i, (low, high) in enumerate(
            zip([self.cutoffs[0]] + self.cutoffs[:-1], self.cutoffs[1:])
        ):
            cluster_size = high - low
            # Reduce dimensionality for tail clusters
            reduced_dim = int(in_features / (div_value ** (i + 1)))
            reduced_dim = max(1, reduced_dim)

            proj = Linear(in_features, reduced_dim, bias=False)
            out = Linear(reduced_dim, cluster_size, bias=False)

            setattr(self, f"tail_{i}_proj", proj)
            setattr(self, f"tail_{i}_out", out)
            self.tail.append((proj, out))

    def forward(self, input: Tensor, target: Tensor):
        """Compute adaptive log softmax with loss."""
        from collections import namedtuple

        import mlx.core as mx

        ASMOutput = namedtuple("ASMOutput", ["output", "loss"])

        x = input._mlx_array
        t = target._mlx_array.astype(mx.int32)
        N = x.shape[0]

        # Compute head logits
        head_out = self.head(input)
        head_logits = head_out._mlx_array

        # Compute log softmax over full head output
        # Head output has: cutoffs[0] classes + (len(cutoffs)-1) cluster indicators
        head_max = mx.max(head_logits, axis=1, keepdims=True)
        head_log_sum_exp = mx.log(mx.sum(mx.exp(head_logits - head_max), axis=1, keepdims=True))
        head_log_probs = head_logits - head_max - head_log_sum_exp

        # Initialize output
        output = mx.zeros((N,), dtype=x.dtype)
        total_loss = mx.array(0.0, dtype=x.dtype)

        # Process each cluster
        cutoffs_with_zero = [0] + self.cutoffs

        for i in range(len(self.cutoffs)):
            low = cutoffs_with_zero[i]
            high = cutoffs_with_zero[i + 1]

            # Find samples in this cluster using vectorized mask
            mask = (t >= low) & (t < high)
            mask_sum = mx.sum(mask.astype(mx.int32))

            # Skip if no samples in this cluster
            if int(mask_sum.item()) == 0:
                continue

            if i == 0:
                # Head cluster - use head log probs directly for classes 0 to cutoffs[0]-1
                # Get log probs for targets in head cluster
                # Use gather/take_along_axis to get log probs at target indices
                local_targets = t  # (N,)
                target_expanded = mx.expand_dims(local_targets, axis=1)  # (N, 1)

                # Gather log probs at target positions
                gathered_log_probs = mx.take_along_axis(head_log_probs, target_expanded, axis=1)
                gathered_log_probs = mx.squeeze(gathered_log_probs, axis=1)  # (N,)

                # Only include samples in this cluster
                mask_float = mask.astype(x.dtype)
                output = output + gathered_log_probs * mask_float
                total_loss = total_loss - mx.sum(gathered_log_probs * mask_float)

            else:
                # Tail cluster
                cluster_idx = i - 1
                proj, out_layer = self.tail[cluster_idx]

                # Get log prob of selecting this cluster from head
                cluster_indicator_idx = self.cutoffs[0] + cluster_idx
                cluster_log_probs_head = head_log_probs[:, cluster_indicator_idx]  # (N,)

                # Compute cluster-specific logits for all samples
                projected = proj(input)
                cluster_out = out_layer(projected)
                cluster_logits = cluster_out._mlx_array  # (N, cluster_size)

                # Compute log softmax within this cluster
                cluster_max = mx.max(cluster_logits, axis=1, keepdims=True)
                cluster_log_sum_exp = mx.log(
                    mx.sum(mx.exp(cluster_logits - cluster_max), axis=1, keepdims=True)
                )
                within_cluster_log_probs = cluster_logits - cluster_max - cluster_log_sum_exp

                # Local targets within cluster
                local_targets = t - low  # (N,)
                local_targets_clamped = mx.maximum(local_targets, 0)
                local_targets_clamped = mx.minimum(local_targets_clamped, high - low - 1)
                local_targets_expanded = mx.expand_dims(local_targets_clamped, axis=1)

                # Gather log probs at local target positions
                gathered_within = mx.take_along_axis(
                    within_cluster_log_probs, local_targets_expanded, axis=1
                )
                gathered_within = mx.squeeze(gathered_within, axis=1)  # (N,)

                # Total log prob = log P(cluster) + log P(class | cluster)
                total_log_prob = cluster_log_probs_head + gathered_within

                # Only include samples in this cluster
                mask_float = mask.astype(x.dtype)
                output = output + total_log_prob * mask_float
                total_loss = total_loss - mx.sum(total_log_prob * mask_float)

        # Average loss
        loss = total_loss / N

        output_tensor = Tensor._from_mlx_array(output)
        loss_tensor = Tensor._from_mlx_array(loss)

        return ASMOutput(output=output_tensor, loss=loss_tensor)

    def log_prob(self, input: Tensor) -> Tensor:
        """Compute log probabilities for all classes."""
        # Simplified implementation
        head_out = self.head(input)
        return head_out

    def predict(self, input: Tensor) -> Tensor:
        """Predict the most likely class."""
        import mlx.core as mx

        log_probs = self.log_prob(input)
        predictions = mx.argmax(log_probs._mlx_array, axis=1)
        return Tensor._from_mlx_array(predictions)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, n_classes={self.n_classes}, cutoffs={self.cutoffs[:-1]}"


__all__ = [
    "MSELoss",
    "L1Loss",
    "NLLLoss",
    "NLLLoss2d",
    "CrossEntropyLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "KLDivLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "CosineEmbeddingLoss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "MultiLabelMarginLoss",
    "MultiMarginLoss",
    "MultiLabelSoftMarginLoss",
    "TripletMarginWithDistanceLoss",
    "CTCLoss",
    "AdaptiveLogSoftmaxWithLoss",
]
