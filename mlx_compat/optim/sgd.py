"""
Stochastic Gradient Descent (SGD) Optimizer

Implements SGD with optional momentum, weight decay, and Nesterov momentum.
"""

from typing import Iterable, Dict, Any, Optional, Union
import mlx.core as mx
from .optimizer import Optimizer
from ..nn.parameter import Parameter
from ..tensor import Tensor


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Implements SGD with optional momentum, weight decay, dampening, and Nesterov momentum.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (required)
        momentum: Momentum factor (default: 0)
        dampening: Dampening for momentum (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        nesterov: Enable Nesterov momentum (default: False)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.001,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None
    ):
        # maximize, foreach, differentiable, fused are accepted for PyTorch compatibility
        # but are either not supported or ignored in MLX
        if maximize:
            import warnings
            warnings.warn("maximize=True is not supported in MLX, will be ignored")
        if foreach:
            import warnings
            warnings.warn("foreach is not supported in MLX, will be ignored")
        if differentiable:
            import warnings
            warnings.warn("differentiable=True is not supported in MLX, will be ignored")
        if fused:
            import warnings
            warnings.warn("fused=True is not supported in MLX, will be ignored")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            loss value if closure is provided, otherwise None

        Note:
            Optimized to use raw MLX arrays internally to reduce Python object overhead.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Work directly with raw MLX arrays for performance
                param = p._mlx_array
                grad = p.grad._mlx_array

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                # Apply momentum
                if momentum != 0:
                    param_state = self.state[id(p)]

                    if 'momentum_buffer' not in param_state:
                        # Initialize momentum buffer as raw MLX array
                        param_state['momentum_buffer'] = mx.zeros_like(grad)

                    buf = param_state['momentum_buffer']

                    # Update momentum buffer: v = momentum * v + (1 - dampening) * grad
                    buf = momentum * buf + (1 - dampening) * grad
                    param_state['momentum_buffer'] = buf

                    if nesterov:
                        # Nesterov momentum: grad = grad + momentum * buf
                        grad = grad + momentum * buf
                    else:
                        # Standard momentum: grad = buf
                        grad = buf

                # Update parameters: θ = θ - lr * grad
                p._mlx_array = param - lr * grad

        return loss

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"SGD (lr={self.defaults['lr']}, momentum={self.defaults['momentum']}, " \
               f"weight_decay={self.defaults['weight_decay']}, nesterov={self.defaults['nesterov']})"


__all__ = ['SGD']
