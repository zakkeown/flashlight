"""
Adagrad Optimizer

Implements Adagrad algorithm (Adaptive Gradient).
"""

from typing import Iterable, Dict, Any, Union, Optional
import mlx.core as mx
from .optimizer import Optimizer
from ..nn.parameter import Parameter
from ..tensor import Tensor


class Adagrad(Optimizer):
    """
    Adagrad optimizer.

    Implements Adagrad algorithm from "Adaptive Subgradient Methods for
    Online Learning and Stochastic Optimization" (Duchi et al., 2011).

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-2)
        lr_decay: Learning rate decay (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        initial_accumulator_value: Initial value for the sum of squared gradients (default: 0)
        eps: Term added to denominator to improve numerical stability (default: 1e-10)

    Example:
        >>> optimizer = Adagrad(model.parameters(), lr=1e-2)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial_accumulator_value: {initial_accumulator_value}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps
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
            lr_decay = group['lr_decay']
            weight_decay = group['weight_decay']
            initial_accumulator_value = group['initial_accumulator_value']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                param_state = self.state[id(p)]

                # Initialize state
                if 'step' not in param_state:
                    param_state['step'] = 0
                    # Sum of squared gradients
                    param_state['sum'] = Tensor._from_mlx_array(
                        mx.full(p._mlx_array.shape, initial_accumulator_value)
                    )

                param_state['step'] += 1
                step = param_state['step']

                # Apply weight decay
                if weight_decay != 0:
                    grad = Tensor._from_mlx_array(
                        grad._mlx_array + weight_decay * p._mlx_array
                    )

                # Apply learning rate decay
                clr = lr / (1 + (step - 1) * lr_decay)

                # Update sum of squared gradients
                sum_sq = param_state['sum']
                sum_sq_mlx = sum_sq._mlx_array + grad._mlx_array ** 2
                sum_sq = Tensor._from_mlx_array(sum_sq_mlx)
                param_state['sum'] = sum_sq

                # Compute step: p = p - clr * g / sqrt(sum + eps)
                std = mx.sqrt(sum_sq._mlx_array) + eps
                p._mlx_array = p._mlx_array - clr * grad._mlx_array / std

        return loss

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"Adagrad (lr={self.defaults['lr']}, lr_decay={self.defaults['lr_decay']}, " \
               f"weight_decay={self.defaults['weight_decay']}, eps={self.defaults['eps']})"


__all__ = ['Adagrad']
