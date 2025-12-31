"""
Adafactor Optimizer

Implements Adafactor algorithm for memory-efficient adaptive learning rate optimization.
From "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (Shazeer & Stern, 2018).
"""

from typing import Any, Dict, Iterable, Optional, Tuple, Union
import math
import mlx.core as mx
from .optimizer import Optimizer
from ..nn.parameter import Parameter
from ..tensor import Tensor


class Adafactor(Optimizer):
    """
    Adafactor optimizer.

    Implements Adafactor algorithm from "Adafactor: Adaptive Learning Rates with
    Sublinear Memory Cost" (Shazeer and Stern, 2018).

    Adafactor is a memory-efficient variant of Adam that uses factored second moment
    estimates for parameters with 2+ dimensions, significantly reducing memory usage.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.01)
        beta2_decay: Coefficient for computing running averages of square gradient (default: -0.8)
        eps: Regularization constants for square gradient and parameter scale (default: (None, 0.001))
        d: Coefficient for computing the learning rate (default: 1.0)
        weight_decay: Weight decay (L2 penalty) (default: 0.0)
        foreach: Whether to use the foreach implementation (default: None, not supported in MLX)
        maximize: Whether to maximize the objective (default: False, not supported in MLX)

    Example:
        >>> optimizer = Adafactor(model.parameters())
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

        >>> # With explicit learning rate
        >>> optimizer = Adafactor(model.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.01,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (None, 0.001),
        d: float = 1.0,
        weight_decay: float = 0.0,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
    ):
        # Handle PyTorch compatibility params
        if foreach:
            import warnings
            warnings.warn("foreach is not supported in MLX, will be ignored")
        if maximize:
            import warnings
            warnings.warn("maximize=True is not supported in MLX, will be ignored")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=eps,
            d=d,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(group: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Compute the learning rate for a parameter group."""
        return group['lr']

    @staticmethod
    def _get_options(group: Dict[str, Any], shape: Tuple[int, ...]) -> bool:
        """Determine if factored second moment should be used."""
        factored = len(shape) >= 2
        return factored

    @staticmethod
    def _rms(tensor: mx.array) -> mx.array:
        """Compute the root mean square of a tensor."""
        return mx.sqrt(mx.mean(tensor ** 2))

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: mx.array, exp_avg_sq_col: mx.array) -> mx.array:
        """Approximate second moment from factored representation."""
        r_factor = (exp_avg_sq_row / mx.mean(exp_avg_sq_row)).reshape(-1, 1)
        c_factor = exp_avg_sq_col.reshape(1, -1)
        return r_factor * c_factor

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
            eps1, eps2 = group['eps']
            beta2_decay = group['beta2_decay']
            d = group['d']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array
                param = p._mlx_array

                # Get shape and options
                shape = param.shape
                factored = self._get_options(group, shape)

                param_state = self.state[id(p)]

                # Initialize state
                if 'step' not in param_state:
                    param_state['step'] = 0
                    if factored:
                        param_state['exp_avg_sq_row'] = mx.zeros(shape[:-1])
                        param_state['exp_avg_sq_col'] = mx.zeros(shape[:-2] + (shape[-1],))
                    else:
                        param_state['exp_avg_sq'] = mx.zeros_like(param)
                    param_state['RMS'] = mx.array(0.0)

                param_state['step'] += 1
                step = param_state['step']

                # Compute learning rate
                lr = self._get_lr(group, param_state)

                # Compute RMS of parameters for scaling
                param_state['RMS'] = self._rms(param)
                param_scale = mx.maximum(eps2 if eps2 is not None else 0.001, param_state['RMS'])

                # Compute rho_t (decay rate for second moment)
                rho_t = min(1.0, (step ** beta2_decay))

                # Update second moment estimate
                grad_sqr = grad ** 2 + (eps1 if eps1 is not None else 1e-30)

                if factored:
                    exp_avg_sq_row = param_state['exp_avg_sq_row']
                    exp_avg_sq_col = param_state['exp_avg_sq_col']

                    exp_avg_sq_row = rho_t * exp_avg_sq_row + (1 - rho_t) * mx.mean(grad_sqr, axis=-1)
                    exp_avg_sq_col = rho_t * exp_avg_sq_col + (1 - rho_t) * mx.mean(grad_sqr, axis=-2)

                    param_state['exp_avg_sq_row'] = exp_avg_sq_row
                    param_state['exp_avg_sq_col'] = exp_avg_sq_col

                    # Approximate second moment
                    exp_avg_sq = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                else:
                    exp_avg_sq = param_state['exp_avg_sq']
                    exp_avg_sq = rho_t * exp_avg_sq + (1 - rho_t) * grad_sqr
                    param_state['exp_avg_sq'] = exp_avg_sq

                # Compute update
                exp_avg_sq_sqrt = mx.sqrt(exp_avg_sq)
                update = grad / exp_avg_sq_sqrt

                # Apply gradient clipping (using default threshold of 1.0)
                update_rms = self._rms(update)
                update = update / mx.maximum(mx.array(1.0), update_rms / 1.0)

                # Apply weight decay
                if weight_decay != 0:
                    param = param - weight_decay * param * lr

                # Apply scaled update
                update = update * lr * param_scale
                p._mlx_array = param - update

        return loss

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return (
            f"Adafactor (lr={self.defaults['lr']}, beta2_decay={self.defaults['beta2_decay']}, "
            f"eps={self.defaults['eps']}, d={self.defaults['d']}, "
            f"weight_decay={self.defaults['weight_decay']})"
        )


__all__ = ['Adafactor']
