"""
RMSprop Optimizer

Implements RMSprop algorithm (Root Mean Square Propagation).
"""

from typing import Iterable, Dict, Any, Union, Optional
import mlx.core as mx
from .optimizer import Optimizer
from ..nn.parameter import Parameter
from ..tensor import Tensor


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Implements RMSprop algorithm proposed by G. Hinton in his Coursera lecture.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.01)
        alpha: Smoothing constant (default: 0.99)
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        momentum: Momentum factor (default: 0)
        centered: If True, compute centered RMSprop (default: False)
        capturable: Whether this instance is safe to capture in a CUDA graph (default: False)
            Note: Not used in MLX, included for PyTorch API compatibility.
        foreach: Whether to use the foreach implementation (default: None)
            Note: Not used in MLX, included for PyTorch API compatibility.
        maximize: Maximize the objective instead of minimizing (default: False)
        differentiable: Whether autograd should occur through the optimizer step (default: False)
            Note: Not used in MLX, included for PyTorch API compatibility.

    Example:
        >>> optimizer = RMSprop(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            capturable=capturable,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            centered = group['centered']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = Tensor._from_mlx_array(
                        grad._mlx_array + weight_decay * p._mlx_array
                    )

                param_state = self.state[id(p)]

                # Initialize state
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['square_avg'] = Tensor._from_mlx_array(mx.zeros_like(p._mlx_array))
                    if momentum > 0:
                        param_state['momentum_buffer'] = Tensor._from_mlx_array(mx.zeros_like(p._mlx_array))
                    if centered:
                        param_state['grad_avg'] = Tensor._from_mlx_array(mx.zeros_like(p._mlx_array))

                square_avg = param_state['square_avg']
                param_state['step'] += 1

                # Update square average
                # v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
                square_avg_mlx = alpha * square_avg._mlx_array + (1 - alpha) * (grad._mlx_array ** 2)
                square_avg = Tensor._from_mlx_array(square_avg_mlx)
                param_state['square_avg'] = square_avg

                if centered:
                    grad_avg = param_state['grad_avg']
                    # g_avg_t = alpha * g_avg_{t-1} + (1 - alpha) * g_t
                    grad_avg_mlx = alpha * grad_avg._mlx_array + (1 - alpha) * grad._mlx_array
                    grad_avg = Tensor._from_mlx_array(grad_avg_mlx)
                    param_state['grad_avg'] = grad_avg
                    # avg = v_t - g_avg^2
                    avg_mlx = square_avg._mlx_array - grad_avg._mlx_array ** 2
                else:
                    avg_mlx = square_avg._mlx_array

                if momentum > 0:
                    buf = param_state['momentum_buffer']
                    # buf = momentum * buf + g / sqrt(avg + eps)
                    buf_mlx = momentum * buf._mlx_array + grad._mlx_array / (mx.sqrt(avg_mlx) + eps)
                    buf = Tensor._from_mlx_array(buf_mlx)
                    param_state['momentum_buffer'] = buf
                    # p = p - lr * buf
                    p._mlx_array = p._mlx_array - lr * buf._mlx_array
                else:
                    # p = p - lr * g / sqrt(avg + eps)
                    p._mlx_array = p._mlx_array - lr * grad._mlx_array / (mx.sqrt(avg_mlx) + eps)

        return loss

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"RMSprop (lr={self.defaults['lr']}, alpha={self.defaults['alpha']}, " \
               f"eps={self.defaults['eps']}, weight_decay={self.defaults['weight_decay']}, " \
               f"momentum={self.defaults['momentum']}, centered={self.defaults['centered']})"


__all__ = ['RMSprop']
