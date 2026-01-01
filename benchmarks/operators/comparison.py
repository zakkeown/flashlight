"""
Comparison operator benchmarks.

Benchmarks for:
- eq, ne, lt, le, gt, ge
- maximum, minimum
- isclose, allclose
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.operators.base import (
    OperatorBenchmark,
    BinaryOperatorBenchmark,
)


class EqOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise equality comparison."""

    op_name = "eq"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.eq(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.eq(a, b)


class NeOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise inequality comparison."""

    op_name = "ne"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.ne(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.ne(a, b)


class LtOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise less than comparison."""

    op_name = "lt"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.lt(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.lt(a, b)


class LeOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise less than or equal comparison."""

    op_name = "le"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.le(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.le(a, b)


class GtOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise greater than comparison."""

    op_name = "gt"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.gt(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.gt(a, b)


class GeOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise greater than or equal comparison."""

    op_name = "ge"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.ge(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.ge(a, b)


class MaximumOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise maximum."""

    op_name = "maximum"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.maximum(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.maximum(a, b)


class MinimumOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise minimum."""

    op_name = "minimum"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.minimum(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.minimum(a, b)


class IscloseOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise isclose."""

    op_name = "isclose"

    def mlx_operation(self, a, b):
        import mlx_compat
        return mlx_compat.isclose(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.isclose(a, b)


COMPARISON_BENCHMARKS = [
    EqOperatorBenchmark,
    NeOperatorBenchmark,
    LtOperatorBenchmark,
    LeOperatorBenchmark,
    GtOperatorBenchmark,
    GeOperatorBenchmark,
    MaximumOperatorBenchmark,
    MinimumOperatorBenchmark,
    IscloseOperatorBenchmark,
]
