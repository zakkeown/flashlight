"""
Adam Optimizer

Implements Adam algorithm (Adaptive Moment Estimation).
"""

from typing import Iterable, Dict, Any, Union, Tuple, Optional
import mlx.core as mx
from .optimizer import Optimizer
from ..nn.parameter import Parameter
from ..tensor import Tensor


class Adam(Optimizer):
    """
    Adam optimizer.

    Implements Adam algorithm from "Adam: A Method for Stochastic Optimization"
    (Kingma and Ba, 2014).

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        amsgrad: Whether to use AMSGrad variant (default: False)

    Example:
        >>> optimizer = Adam(model.parameters(), lr=1e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        decoupled_weight_decay: bool = False
    ):
        # foreach, maximize, capturable, differentiable, fused are accepted for
        # PyTorch compatibility but are either not supported or ignored in MLX
        if maximize:
            import warnings
            warnings.warn("maximize=True is not supported in MLX, will be ignored")
        if foreach:
            import warnings
            warnings.warn("foreach is not supported in MLX, will be ignored")
        if capturable:
            import warnings
            warnings.warn("capturable=True is not supported in MLX, will be ignored")
        if differentiable:
            import warnings
            warnings.warn("differentiable=True is not supported in MLX, will be ignored")
        if fused:
            import warnings
            warnings.warn("fused=True is not supported in MLX, will be ignored")

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused
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
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = Tensor._from_mlx_array(
                        grad._mlx_array + weight_decay * p._mlx_array
                    )

                param_state = self.state[id(p)]

                # Initialize state
                if 'step' not in param_state:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = Tensor._from_mlx_array(mx.zeros_like(p._mlx_array))
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = Tensor._from_mlx_array(mx.zeros_like(p._mlx_array))
                    if amsgrad:
                        # Max of exp_avg_sq
                        param_state['max_exp_avg_sq'] = Tensor._from_mlx_array(mx.zeros_like(p._mlx_array))

                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                param_state['step'] += 1
                step = param_state['step']

                # Decay the first and second moment running average coefficient
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg_mlx = beta1 * exp_avg._mlx_array + (1 - beta1) * grad._mlx_array
                exp_avg = Tensor._from_mlx_array(exp_avg_mlx)
                param_state['exp_avg'] = exp_avg

                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq_mlx = beta2 * exp_avg_sq._mlx_array + (1 - beta2) * (grad._mlx_array ** 2)
                exp_avg_sq = Tensor._from_mlx_array(exp_avg_sq_mlx)
                param_state['exp_avg_sq'] = exp_avg_sq

                if amsgrad:
                    max_exp_avg_sq = param_state['max_exp_avg_sq']
                    # max_v_t = max(max_v_{t-1}, v_t)
                    max_exp_avg_sq_mlx = mx.maximum(max_exp_avg_sq._mlx_array, exp_avg_sq._mlx_array)
                    max_exp_avg_sq = Tensor._from_mlx_array(max_exp_avg_sq_mlx)
                    param_state['max_exp_avg_sq'] = max_exp_avg_sq
                    denom_mlx = mx.sqrt(max_exp_avg_sq._mlx_array) + eps
                else:
                    denom_mlx = mx.sqrt(exp_avg_sq._mlx_array) + eps

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute step size with bias correction
                step_size = lr / bias_correction1

                # Update parameters: θ = θ - step_size * m_t / (sqrt(v_t) + eps)
                # With bias correction: θ = θ - lr * m_hat_t / (sqrt(v_hat_t) + eps)
                # where m_hat_t = m_t / (1 - beta1^t), v_hat_t = v_t / (1 - beta2^t)
                update_mlx = step_size * exp_avg._mlx_array / (mx.sqrt(denom_mlx ** 2 / bias_correction2) + eps)
                p._mlx_array = p._mlx_array - update_mlx

        return loss

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"Adam (lr={self.defaults['lr']}, betas={self.defaults['betas']}, " \
               f"eps={self.defaults['eps']}, weight_decay={self.defaults['weight_decay']})"


__all__ = ['Adam']
