"""KL Divergence"""

from typing import Callable, Dict, Tuple, Type
import mlx.core as mx
import math

from ..tensor import Tensor
from .distribution import Distribution


# Registry for KL divergence implementations
_KL_REGISTRY: Dict[Tuple[Type, Type], Callable] = {}


def register_kl(type_p: Type, type_q: Type):
    """
    Decorator to register a KL divergence implementation.

    Usage:
        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            ...
    """
    def decorator(fn: Callable):
        _KL_REGISTRY[(type_p, type_q)] = fn
        return fn
    return decorator


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """
    Compute KL divergence KL(p || q).

    Args:
        p: First distribution
        q: Second distribution

    Returns:
        KL divergence tensor
    """
    # Look up in registry
    key = (type(p), type(q))
    if key in _KL_REGISTRY:
        return _KL_REGISTRY[key](p, q)

    # Try parent classes
    for (type_p, type_q), fn in _KL_REGISTRY.items():
        if isinstance(p, type_p) and isinstance(q, type_q):
            return fn(p, q)

    raise NotImplementedError(f"KL divergence not implemented for {type(p).__name__} and {type(q).__name__}")


# Register common KL divergences

@register_kl(Distribution, Distribution)
def _kl_generic(p, q):
    """Generic Monte Carlo KL estimation."""
    samples = p.sample((100,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)
    if isinstance(log_p, Tensor):
        log_p = log_p._mlx_array
    if isinstance(log_q, Tensor):
        log_q = log_q._mlx_array
    return Tensor(mx.mean(log_p - log_q, axis=0))


def _register_normal_kl():
    """Register Normal-Normal KL after Normal is defined."""
    from .normal import Normal

    @register_kl(Normal, Normal)
    def _kl_normal_normal(p: Normal, q: Normal) -> Tensor:
        var_ratio = (p.scale / q.scale) ** 2
        t1 = ((p.loc - q.loc) / q.scale) ** 2
        return Tensor(0.5 * (var_ratio + t1 - 1 - mx.log(var_ratio)))


def _register_bernoulli_kl():
    """Register Bernoulli-Bernoulli KL after Bernoulli is defined."""
    from .bernoulli import Bernoulli

    @register_kl(Bernoulli, Bernoulli)
    def _kl_bernoulli_bernoulli(p: Bernoulli, q: Bernoulli) -> Tensor:
        t1 = p.probs * (mx.log(p.probs + 1e-10) - mx.log(q.probs + 1e-10))
        t2 = (1 - p.probs) * (mx.log(1 - p.probs + 1e-10) - mx.log(1 - q.probs + 1e-10))
        return Tensor(t1 + t2)


def _register_categorical_kl():
    """Register Categorical-Categorical KL after Categorical is defined."""
    from .categorical import Categorical

    @register_kl(Categorical, Categorical)
    def _kl_categorical_categorical(p: Categorical, q: Categorical) -> Tensor:
        t = p.probs * (mx.log(p.probs + 1e-10) - mx.log(q.probs + 1e-10))
        return Tensor(mx.sum(t, axis=-1))


def _register_uniform_kl():
    """Register Uniform-Uniform KL after Uniform is defined."""
    from .uniform import Uniform

    @register_kl(Uniform, Uniform)
    def _kl_uniform_uniform(p: Uniform, q: Uniform) -> Tensor:
        return Tensor(mx.log((q.high - q.low) / (p.high - p.low)))


def _register_exponential_kl():
    """Register Exponential-Exponential KL."""
    from .exponential import Exponential

    @register_kl(Exponential, Exponential)
    def _kl_exponential_exponential(p: Exponential, q: Exponential) -> Tensor:
        return Tensor(mx.log(q.rate / p.rate) + p.rate / q.rate - 1)


def _register_gamma_kl():
    """Register Gamma-Gamma KL."""
    from .gamma import Gamma
    from ..ops.special import lgamma, digamma

    @register_kl(Gamma, Gamma)
    def _kl_gamma_gamma(p: Gamma, q: Gamma) -> Tensor:
        psi_p = digamma(p.concentration)
        log_gamma_p = lgamma(p.concentration)
        log_gamma_q = lgamma(q.concentration)
        return Tensor(
            (p.concentration - q.concentration) * psi_p -
            log_gamma_p + log_gamma_q +
            q.concentration * (mx.log(p.rate) - mx.log(q.rate)) +
            p.concentration * (q.rate / p.rate - 1)
        )


def _register_beta_kl():
    """Register Beta-Beta KL."""
    from .beta import Beta
    from ..ops.special import betaln, digamma

    @register_kl(Beta, Beta)
    def _kl_beta_beta(p: Beta, q: Beta) -> Tensor:
        a_p, b_p = p.concentration1, p.concentration0
        a_q, b_q = q.concentration1, q.concentration0
        log_beta_p = betaln(a_p, b_p)
        log_beta_q = betaln(a_q, b_q)
        psi_sum_p = digamma(a_p + b_p)
        psi_a_p = digamma(a_p)
        psi_b_p = digamma(b_p)
        return Tensor(
            log_beta_q - log_beta_p +
            (a_p - a_q) * psi_a_p +
            (b_p - b_q) * psi_b_p +
            (a_q + b_q - a_p - b_p) * psi_sum_p
        )


# Initialize registrations
try:
    _register_normal_kl()
except ImportError:
    pass

try:
    _register_bernoulli_kl()
except ImportError:
    pass

try:
    _register_categorical_kl()
except ImportError:
    pass

try:
    _register_uniform_kl()
except ImportError:
    pass

try:
    _register_exponential_kl()
except ImportError:
    pass

try:
    _register_gamma_kl()
except ImportError:
    pass

try:
    _register_beta_kl()
except ImportError:
    pass


__all__ = ['kl_divergence', 'register_kl']
