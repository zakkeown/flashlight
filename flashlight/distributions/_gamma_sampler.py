"""
Gamma distribution sampler for MLX.

Implements gamma sampling since mx.random.gamma doesn't exist yet.
Uses Marsaglia and Tsang's method for alpha >= 1, and Ahrens-Dieter
method for alpha < 1.

Reference:
    Marsaglia, G., & Tsang, W. W. (2000). A simple method for generating
    gamma variables. ACM Transactions on Mathematical Software, 26(3), 363-372.
"""

import mlx.core as mx


def _gamma_sample_alpha_ge_1(alpha: mx.array, shape: tuple) -> mx.array:
    """
    Sample from Gamma(alpha, 1) where alpha >= 1 using Marsaglia-Tsang method.

    This is an efficient rejection sampling method that works well for alpha >= 1.
    """
    # Marsaglia-Tsang parameters
    d = alpha - 1.0 / 3.0
    c = 1.0 / mx.sqrt(9.0 * d)

    # We need to loop until we get valid samples
    # For MLX, we'll use vectorized rejection sampling with oversampling
    result = mx.zeros(shape)
    remaining_mask = mx.ones(shape, dtype=mx.bool_)

    max_iterations = 100
    for _ in range(max_iterations):
        # Count remaining samples needed
        n_remaining = int(mx.sum(remaining_mask).item())
        if n_remaining == 0:
            break

        # Generate candidate samples (oversample by 2x for efficiency)
        n_candidates = max(n_remaining * 2, 1000)

        # Generate standard normal samples
        z = mx.random.normal(shape=(n_candidates,))
        u = mx.random.uniform(shape=(n_candidates,))

        # Marsaglia-Tsang transformation
        v = (1.0 + c * z) ** 3

        # Rejection conditions
        # Accept if: z > -1/c AND log(u) < 0.5*z^2 + d - d*v + d*log(v)
        valid_v = v > 0  # equivalent to z > -1/c

        # For valid v, compute acceptance criterion
        log_accept = 0.5 * z * z + d - d * v + d * mx.log(mx.maximum(v, 1e-10))
        accept = valid_v & (mx.log(u) < log_accept)

        # Get accepted samples
        accepted_vals = d * v
        accepted_vals = mx.where(accept, accepted_vals, mx.zeros_like(accepted_vals))

        # Fill in remaining positions
        # This is a simplified approach - fill from accepted samples
        accept_indices = mx.where(accept)[0]
        if accept_indices.size > 0:
            # Get flat indices of remaining positions
            remaining_flat = remaining_mask.reshape(-1)
            result_flat = result.reshape(-1)

            fill_count = min(int(mx.sum(remaining_flat).item()), accept_indices.size)
            if fill_count > 0:
                # Get positions to fill
                remaining_indices = mx.where(remaining_flat)[0][:fill_count]
                accepted_values = accepted_vals[accept_indices[:fill_count]]

                # Update result
                for i in range(fill_count):
                    idx = int(remaining_indices[i].item())
                    result_flat = result_flat.at[idx].add(accepted_values[i])

                # Update mask
                for i in range(fill_count):
                    idx = int(remaining_indices[i].item())
                    remaining_flat = remaining_flat.at[idx].add(-remaining_flat[idx])

                result = result_flat.reshape(shape)
                remaining_mask = remaining_flat.reshape(shape)

    return result


def _gamma_sample_vectorized(alpha: mx.array, shape: tuple) -> mx.array:
    """
    Vectorized gamma sampling using Marsaglia-Tsang with boosting for alpha < 1.

    For alpha < 1, we use the identity:
        Gamma(alpha, 1) = Gamma(alpha + 1, 1) * U^(1/alpha)
    where U ~ Uniform(0, 1)
    """
    # Handle alpha < 1 by boosting
    boost_mask = alpha < 1.0
    alpha_boosted = mx.where(boost_mask, alpha + 1.0, alpha)

    # Marsaglia-Tsang parameters
    d = alpha_boosted - 1.0 / 3.0
    c = 1.0 / mx.sqrt(9.0 * d)

    # Broadcast d and c to output shape
    d = mx.broadcast_to(d, shape)
    c = mx.broadcast_to(c, shape)
    boost_mask = mx.broadcast_to(boost_mask, shape)
    alpha = mx.broadcast_to(alpha, shape)

    # Simple rejection sampling loop
    result = mx.zeros(shape)
    done = mx.zeros(shape, dtype=mx.bool_)

    max_iterations = 200
    for _ in range(max_iterations):
        # Generate candidates for all positions
        z = mx.random.normal(shape=shape)
        u = mx.random.uniform(shape=shape)

        # Marsaglia-Tsang transformation
        v = (1.0 + c * z) ** 3

        # Rejection conditions
        valid_v = v > 0
        log_accept = 0.5 * z * z + d - d * v + d * mx.log(mx.maximum(v, 1e-10))
        accept = valid_v & (mx.log(u) < log_accept) & (~done)

        # Update result where accepted
        samples = d * v
        result = mx.where(accept, samples, result)
        done = done | accept

        if mx.all(done):
            break

    # Apply boost correction for alpha < 1
    # Gamma(alpha, 1) = Gamma(alpha + 1, 1) * U^(1/alpha)
    u_boost = mx.random.uniform(shape=shape)
    boost_factor = mx.power(u_boost, 1.0 / alpha)
    result = mx.where(boost_mask, result * boost_factor, result)

    return result


def random_gamma(alpha: mx.array, shape: tuple = None) -> mx.array:
    """
    Sample from Gamma(alpha, scale=1) distribution.

    This provides a pure MLX implementation of gamma sampling since
    mx.random.gamma doesn't exist yet in MLX.

    Args:
        alpha: Shape/concentration parameter (must be positive)
        shape: Output shape. If None, uses alpha's shape.

    Returns:
        Samples from Gamma(alpha, 1) distribution

    Note:
        For Gamma(alpha, rate=beta), divide the result by beta:
        Gamma(alpha, beta) = Gamma(alpha, 1) / beta
    """
    if shape is None:
        shape = alpha.shape

    return _gamma_sample_vectorized(alpha, shape)


__all__ = ["random_gamma"]
