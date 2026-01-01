"""
PyTorch-compatible benchmarking utilities for MLX.

Provides Timer class and comparison utilities matching torch.utils.benchmark.

Example:
    >>> from mlx_compat.utils.benchmark import Timer, Compare
    >>>
    >>> # Time a simple operation
    >>> t = Timer(
    ...     stmt="mx.matmul(a, b)",
    ...     setup="import mlx.core as mx; a = mx.random.normal((100, 100)); b = mx.random.normal((100, 100))",
    ...     label="matmul",
    ...     sub_label="100x100",
    ... )
    >>> m = t.blocked_autorange()
    >>> print(m)
    >>>
    >>> # Compare multiple operations
    >>> results = []
    >>> for size in [100, 500, 1000]:
    ...     t = Timer(
    ...         stmt="mx.matmul(a, b)",
    ...         setup=f"import mlx.core as mx; a = mx.random.normal(({size}, {size})); b = mx.random.normal(({size}, {size}))",
    ...         label="matmul",
    ...         sub_label=f"{size}x{size}",
    ...     )
    ...     results.append(t.blocked_autorange())
    >>> compare = Compare(results)
    >>> compare.print()
"""

from .common import (
    TaskSpec,
    Measurement,
    select_unit,
    unit_to_english,
    trim_sigfig,
    ordered_unique,
)
from .timer import Timer, timer
from .compare import Compare, Colorize

__all__ = [
    # Core classes
    "Timer",
    "Measurement",
    "Compare",
    # Utilities
    "TaskSpec",
    "Colorize",
    "timer",
    "select_unit",
    "unit_to_english",
    "trim_sigfig",
    "ordered_unique",
]
