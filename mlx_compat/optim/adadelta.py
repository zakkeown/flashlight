"""
Adadelta Optimizer

Implements Adadelta algorithm (Adaptive Learning Rate Method).
"""

from typing import Iterable, Dict, Any, Union, Optional
import mlx.core as mx
from .optimizer import Optimizer
from ..nn.parameter import Parameter
from ..tensor import Tensor


class Adadelta(Optimizer):
    """
    Adadelta optimizer.

    Implements Adadelta algorithm from "ADADELTA: An Adaptive Learning Rate Method"
    (Zeiler, 2012).

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Coefficient that scales delta before applying to parameters (default: 1.0)
        rho: Coefficient for computing running average of squared gradients (default: 0.9)
        eps: Term added to denominator to improve numerical stability (default: 1e-6)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        foreach: Whether to use foreach implementation (ignored, for compatibility)
        maximize: Whether to maximize instead of minimize (default: False)
        differentiable: Whether autograd should occur through optimizer step (ignored)

    Example:
        >>> optimizer = Adadelta(model.parameters(), lr=1.0, rho=0.9)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        *,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
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
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array

                # Negate gradient for maximize
                if maximize:
                    grad = -grad

                param_state = self.state[id(p)]

                # Initialize state
                if 'step' not in param_state:
                    param_state['step'] = 0
                    # Running average of squared gradients
                    param_state['square_avg'] = mx.zeros_like(p._mlx_array)
                    # Running average of squared parameter updates
                    param_state['acc_delta'] = mx.zeros_like(p._mlx_array)

                param_state['step'] += 1

                square_avg = param_state['square_avg']
                acc_delta = param_state['acc_delta']

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad + weight_decay * p._mlx_array

                # Update running average of squared gradients
                square_avg = rho * square_avg + (1 - rho) * (grad ** 2)
                param_state['square_avg'] = square_avg

                # Compute update
                std = mx.sqrt(square_avg + eps)
                delta = mx.sqrt(acc_delta + eps) / std * grad

                # Update running average of squared deltas
                acc_delta = rho * acc_delta + (1 - rho) * (delta ** 2)
                param_state['acc_delta'] = acc_delta

                # Apply update
                p._mlx_array = p._mlx_array - lr * delta

        return loss

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"Adadelta (lr={self.defaults['lr']}, rho={self.defaults['rho']}, " \
               f"eps={self.defaults['eps']}, weight_decay={self.defaults['weight_decay']})"


__all__ = ['Adadelta']
