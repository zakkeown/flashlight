"""
Additional Optimizers

Implements additional PyTorch-compatible optimizers for MLX.
"""

from typing import Callable, Iterable, Optional, Tuple

import mlx.core as mx

from ..tensor import Tensor
from .optimizer import Optimizer


class Adamax(Optimizer):
    """
    Adamax optimizer (variant of Adam based on infinity norm).

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 2e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        foreach: Use foreach implementation (ignored in MLX)
        maximize: Maximize the objective (ignored in MLX)
        differentiable: Enable differentiable optimizer (ignored in MLX)
        capturable: Capture optimizer state (ignored in MLX)
    """

    def __init__(
        self,
        params,
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        # foreach, maximize, differentiable, capturable are for PyTorch compatibility
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = mx.zeros_like(p._mlx_array)
                    state["exp_inf"] = mx.zeros_like(p._mlx_array)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_inf = state["exp_inf"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p._mlx_array

                # Update biased first moment estimate
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # Update exponentially weighted infinity norm
                exp_inf = mx.maximum(beta2 * exp_inf, mx.abs(grad))

                state["exp_avg"] = exp_avg
                state["exp_inf"] = exp_inf

                # Bias correction for first moment
                bias_correction1 = 1 - beta1 ** state["step"]
                step_size = lr / bias_correction1

                p._mlx_array = p._mlx_array - step_size * exp_avg / (exp_inf + eps)

        return loss


class RAdam(Optimizer):
    """
    Rectified Adam optimizer.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay: Use decoupled weight decay (default: False)
        foreach: Use foreach implementation (ignored in MLX)
        maximize: Maximize the objective (ignored in MLX)
        capturable: Capture optimizer state (ignored in MLX)
        differentiable: Enable differentiable optimizer (ignored in MLX)
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        decoupled_weight_decay: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):
        # These are for PyTorch compatibility
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = mx.zeros_like(p._mlx_array)
                    state["exp_avg_sq"] = mx.zeros_like(p._mlx_array)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p._mlx_array

                # Update biased moments
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Maximum length of approximated SMA
                rho_inf = 2 / (1 - beta2) - 1
                # Current length of approximated SMA
                rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

                if rho_t > 5:
                    # Variance is tractable
                    rect = (
                        (rho_t - 4)
                        * (rho_t - 2)
                        * rho_inf
                        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    ) ** 0.5
                    step_size = lr * rect / bias_correction1
                    denom = mx.sqrt(exp_avg_sq / bias_correction2) + eps
                    p._mlx_array = p._mlx_array - step_size * exp_avg / denom
                else:
                    # Variance is not tractable, use unadapted step
                    step_size = lr / bias_correction1
                    p._mlx_array = p._mlx_array - step_size * exp_avg

        return loss


class NAdam(Optimizer):
    """
    NAdam optimizer (Nesterov-accelerated Adam).

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 2e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        momentum_decay: Momentum decay (default: 4e-3)
        decoupled_weight_decay: Use decoupled weight decay (default: False)
        foreach: Use foreach implementation (ignored in MLX)
        maximize: Maximize the objective (ignored in MLX)
        capturable: Capture optimizer state (ignored in MLX)
        differentiable: Enable differentiable optimizer (ignored in MLX)
    """

    def __init__(
        self,
        params,
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):
        # These are for PyTorch compatibility
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = mx.zeros_like(p._mlx_array)
                    state["exp_avg_sq"] = mx.zeros_like(p._mlx_array)
                    state["mu_product"] = 1.0

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay != 0:
                    grad = grad + weight_decay * p._mlx_array

                step = state["step"]

                # Update biased first moment estimate
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # Update biased second raw moment estimate
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Nesterov momentum
                mu = beta1 * (1 - 0.5 * (0.96 ** (step * group["momentum_decay"])))
                mu_next = beta1 * (1 - 0.5 * (0.96 ** ((step + 1) * group["momentum_decay"])))

                state["mu_product"] *= mu
                mu_product_next = state["mu_product"] * mu_next

                exp_avg_hat = mu_next * exp_avg / (1 - mu_product_next) + (1 - mu) * grad / (
                    1 - state["mu_product"]
                )
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                p._mlx_array = p._mlx_array - lr * exp_avg_hat / (mx.sqrt(exp_avg_sq_hat) + eps)

        return loss


class ASGD(Optimizer):
    """
    Averaged Stochastic Gradient Descent optimizer.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-2)
        lambd: Decay term (default: 1e-4)
        alpha: Power for eta update (default: 0.75)
        t0: Starting point for averaging (default: 1e6)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        foreach: Use foreach implementation (ignored in MLX)
        maximize: Maximize the objective (ignored in MLX)
        differentiable: Enable differentiable optimizer (ignored in MLX)
        capturable: Capture optimizer state (ignored in MLX)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        # These are for PyTorch compatibility
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lambd = group["lambd"]
            alpha = group["alpha"]
            t0 = group["t0"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["eta"] = lr
                    state["mu"] = 1.0
                    state["ax"] = mx.zeros_like(p._mlx_array)

                state["step"] += 1

                if weight_decay != 0:
                    grad = grad + weight_decay * p._mlx_array

                # Decay learning rate
                state["eta"] = lr / ((1 + lambd * state["step"]) ** alpha)

                # Update parameter
                p._mlx_array = p._mlx_array - state["eta"] * grad

                # Update averaged parameter
                if state["step"] > t0:
                    state["mu"] += 1
                    state["ax"] = state["ax"] + (p._mlx_array - state["ax"]) / state["mu"]

        return loss


class Rprop(Optimizer):
    """
    Resilient backpropagation optimizer.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-2)
        etas: Pair of (etaminus, etaplus) (default: (0.5, 1.2))
        step_sizes: Pair of (min_step, max_step) (default: (1e-6, 50))
        capturable: Capture optimizer state (ignored in MLX)
        foreach: Use foreach implementation (ignored in MLX)
        maximize: Maximize the objective (ignored in MLX)
        differentiable: Enable differentiable optimizer (ignored in MLX)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-06, 50),
        *,
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        # These are for PyTorch compatibility
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad._mlx_array
                state = self.state[id(p)]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = mx.zeros_like(p._mlx_array)
                    # Use mx.full instead of mx.full_like (MLX doesn't have full_like)
                    state["step_size"] = mx.full(p._mlx_array.shape, lr, dtype=p._mlx_array.dtype)

                state["step"] += 1
                prev_grad = state["prev_grad"]
                step_size = state["step_size"]

                # Compute sign changes
                sign = grad * prev_grad
                pos_sign = sign > 0
                neg_sign = sign < 0

                # Update step sizes
                step_size = mx.where(
                    pos_sign, mx.minimum(step_size * etaplus, step_size_max), step_size
                )
                step_size = mx.where(
                    neg_sign, mx.maximum(step_size * etaminus, step_size_min), step_size
                )

                # Update gradient (set to 0 where sign changed)
                grad = mx.where(neg_sign, 0, grad)

                state["prev_grad"] = grad
                state["step_size"] = step_size

                # Update parameter
                p._mlx_array = p._mlx_array - step_size * mx.sign(grad)

        return loss


__all__ = [
    "Adamax",
    "RAdam",
    "NAdam",
    "ASGD",
    "Rprop",
]
