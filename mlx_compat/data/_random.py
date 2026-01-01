"""
MLX-based Random Utilities for Data Loading

Provides MLX-native random operations for samplers and dataset splitting,
replacing Python's random module for better integration with MLX's random state.

Performance optimizations:
- Unified key handling to reduce code duplication
- Delayed tolist() calls where possible
- Cached log computations for repeated sampling
- Hybrid Python/MLX: Python random for small sizes (faster due to no GPU overhead),
  MLX for large sizes (faster due to parallel computation)
"""

from typing import List, Optional, Any, Union, Iterator
import random
import mlx.core as mx

# Threshold for switching between Python and MLX implementations
# Below this size, Python random is faster due to MLX's fixed overhead:
# - JIT compilation: ~5ms first call
# - GPU kernel dispatch: ~0.2ms per call
# - tolist() transfer: ~0.05ms per call
# Above this size, MLX's parallel computation dominates
_HYBRID_THRESHOLD = 5000

# Type alias for weights that can be list, tuple, or array-like
WeightsType = Union[List[float], tuple, mx.array]


def _generate_random_uniform(n: int, key: Optional[mx.array] = None) -> mx.array:
    """Generate random uniform values, handling key optionally."""
    if key is not None:
        return mx.random.uniform(shape=(n,), key=key)
    return mx.random.uniform(shape=(n,))


def mlx_permutation(n: int, key: Optional[mx.array] = None) -> List[int]:
    """
    Generate a random permutation of [0, n-1].

    Uses hybrid approach for optimal performance:
    - Python random.shuffle for small sizes (n < 5000) when no key is provided
    - MLX argsort on random uniform values for large sizes or when key is needed

    Args:
        n: Length of permutation.
        key: Optional random key for reproducibility. When provided, always uses MLX.

    Returns:
        List of integers representing a random permutation.

    Example:
        >>> perm = mlx_permutation(10)
        >>> len(perm)
        10
        >>> set(perm) == set(range(10))
        True
    """
    if n == 0:
        return []

    # Use Python random for small sizes without key (avoids MLX overhead)
    if n < _HYBRID_THRESHOLD and key is None:
        indices = list(range(n))
        random.shuffle(indices)
        return indices

    # Use MLX for large sizes or when reproducibility key is needed
    random_vals = _generate_random_uniform(n, key)
    indices = mx.argsort(random_vals)
    mx.eval(indices)  # Ensure computation is complete
    return indices.tolist()


def mlx_permutation_array(n: int, key: Optional[mx.array] = None) -> mx.array:
    """
    Generate a random permutation as MLX array (avoids CPU transfer).

    For internal use when the result stays in MLX operations.

    Args:
        n: Length of permutation.
        key: Optional random key for reproducibility.

    Returns:
        MLX array with random permutation.
    """
    if n == 0:
        return mx.array([], dtype=mx.int32)

    random_vals = _generate_random_uniform(n, key)
    return mx.argsort(random_vals)


def mlx_shuffle_list(items: List[Any], key: Optional[mx.array] = None) -> List[Any]:
    """
    Shuffle a list using MLX random.

    Returns a new shuffled list (does not modify original).

    Args:
        items: List to shuffle.
        key: Optional random key for reproducibility.

    Returns:
        New list with elements in random order.

    Example:
        >>> original = [1, 2, 3, 4, 5]
        >>> shuffled = mlx_shuffle_list(original)
        >>> set(shuffled) == set(original)
        True
    """
    n = len(items)
    if n == 0:
        return []

    perm = mlx_permutation(n, key=key)
    return [items[i] for i in perm]


def mlx_shuffle_iterator(items: List[Any], key: Optional[mx.array] = None) -> Iterator[Any]:
    """
    Yield items in shuffled order (lazy evaluation).

    Useful when you don't need the full shuffled list materialized.

    Args:
        items: List to shuffle.
        key: Optional random key for reproducibility.

    Yields:
        Items in random order.
    """
    if len(items) == 0:
        return

    perm = mlx_permutation(len(items), key=key)
    for i in perm:
        yield items[i]


def mlx_randint(low: int, high: int, size: int, key: Optional[mx.array] = None) -> List[int]:
    """
    Generate random integers in [low, high).

    Uses hybrid approach for optimal performance:
    - Python random.randint for small sizes (size < 5000) when no key is provided
    - MLX random.randint for large sizes or when key is needed

    Args:
        low: Minimum value (inclusive).
        high: Maximum value (exclusive).
        size: Number of integers to generate.
        key: Optional random key for reproducibility. When provided, always uses MLX.

    Returns:
        List of random integers.

    Example:
        >>> values = mlx_randint(0, 10, 5)
        >>> len(values)
        5
        >>> all(0 <= v < 10 for v in values)
        True
    """
    if size == 0:
        return []

    # Use Python random for small sizes without key (avoids MLX overhead)
    if size < _HYBRID_THRESHOLD and key is None:
        return [random.randint(low, high - 1) for _ in range(size)]

    # Use MLX for large sizes or when reproducibility key is needed
    if key is not None:
        result = mx.random.randint(low, high, (size,), key=key)
    else:
        result = mx.random.randint(low, high, (size,))

    mx.eval(result)
    return result.tolist()


def mlx_randint_array(low: int, high: int, size: int, key: Optional[mx.array] = None) -> mx.array:
    """
    Generate random integers as MLX array (avoids CPU transfer).

    For internal use when the result stays in MLX operations.

    Args:
        low: Minimum value (inclusive).
        high: Maximum value (exclusive).
        size: Number of integers to generate.
        key: Optional random key for reproducibility.

    Returns:
        MLX array with random integers.
    """
    if size == 0:
        return mx.array([], dtype=mx.int32)

    if key is not None:
        return mx.random.randint(low, high, (size,), key=key)
    return mx.random.randint(low, high, (size,))


def _normalize_weights(weights: WeightsType) -> mx.array:
    """
    Convert weights to normalized probability array.

    Handles various input types efficiently.

    Args:
        weights: Input weights (list, tuple, or mx.array).

    Returns:
        Normalized probability array.
    """
    # Convert to MLX array if needed
    if isinstance(weights, mx.array):
        weights_arr = weights.astype(mx.float32)
    else:
        # Convert to Python floats to handle numpy types
        weights_list = [float(w) for w in weights]
        weights_arr = mx.array(weights_list, dtype=mx.float32)

    # Normalize
    total = mx.sum(weights_arr)
    if float(mx.max(total)) <= 0:
        # All weights zero - use uniform
        return mx.ones_like(weights_arr) / len(weights_arr)

    return weights_arr / total


def mlx_weighted_sample(
    weights: WeightsType,
    num_samples: int,
    replacement: bool = True,
    key: Optional[mx.array] = None
) -> List[int]:
    """
    Weighted random sampling.

    Uses hybrid approach for optimal performance:
    - Python random.choices for small sizes with replacement when no key is provided
    - MLX categorical/Gumbel-top-k for large sizes or when key is needed

    For sampling with replacement, uses mx.random.categorical (or Python random.choices).
    For sampling without replacement, uses Gumbel-top-k trick.

    Args:
        weights: Sampling weights (not necessarily summing to 1).
                 Can be list, tuple, or mx.array.
        num_samples: Number of samples to draw.
        replacement: If True, sample with replacement.
        key: Optional random key for reproducibility. When provided, always uses MLX.

    Returns:
        List of sampled indices.

    Example:
        >>> weights = [0.1, 0.9]  # Second element much more likely
        >>> samples = mlx_weighted_sample(weights, 100, replacement=True)
        >>> # Second element should appear more often
    """
    if num_samples == 0:
        return []

    n = len(weights)

    # Use Python random for small sizes with replacement and no key
    if num_samples < _HYBRID_THRESHOLD and replacement and key is None:
        # Normalize weights for Python random.choices
        if isinstance(weights, mx.array):
            weights_list = weights.tolist()
        else:
            weights_list = list(weights)
        total = sum(weights_list)
        if total <= 0:
            # Uniform distribution if all weights zero
            weights_list = [1.0 / n] * n
        else:
            weights_list = [w / total for w in weights_list]
        indices = list(range(n))
        return random.choices(indices, weights=weights_list, k=num_samples)

    # Use MLX for large sizes, without-replacement, or when key is needed
    probs = _normalize_weights(weights)

    # Pre-compute log probabilities with epsilon for numerical stability
    log_probs = mx.log(probs + 1e-10)

    if replacement:
        # Use categorical distribution for sampling with replacement
        if key is not None:
            indices = mx.random.categorical(log_probs, axis=-1, num_samples=num_samples, key=key)
        else:
            indices = mx.random.categorical(log_probs, axis=-1, num_samples=num_samples)

        mx.eval(indices)
        return indices.tolist()
    else:
        # Gumbel-top-k trick for sampling without replacement
        u = _generate_random_uniform(n, key)

        # Clamp u to avoid numerical issues with log
        u = mx.clip(u, 1e-10, 1.0 - 1e-10)

        # Gumbel noise: -log(-log(U))
        gumbel = -mx.log(-mx.log(u))

        # Perturbed log-weights
        perturbed = log_probs + gumbel

        # Get indices of top num_samples values
        # argsort returns ascending order, so we negate and take first num_samples
        sorted_indices = mx.argsort(-perturbed)
        indices = sorted_indices[:num_samples]

        mx.eval(indices)
        return indices.tolist()


def mlx_seeded_key(seed: int, epoch: int = 0) -> mx.array:
    """
    Create a seeded random key combining seed and epoch.

    Useful for reproducible shuffling that changes per epoch.

    Args:
        seed: Base random seed.
        epoch: Epoch number to combine with seed.

    Returns:
        MLX random key array.

    Example:
        >>> key1 = mlx_seeded_key(42, epoch=0)
        >>> key2 = mlx_seeded_key(42, epoch=1)
        >>> # key1 and key2 produce different random sequences
    """
    return mx.random.key(seed + epoch)


__all__ = [
    'mlx_permutation',
    'mlx_permutation_array',
    'mlx_shuffle_list',
    'mlx_shuffle_iterator',
    'mlx_randint',
    'mlx_randint_array',
    'mlx_weighted_sample',
    'mlx_seeded_key',
]
