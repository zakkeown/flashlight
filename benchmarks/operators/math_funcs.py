"""
Math functions operator benchmarks.

Benchmarks for:
- clamp, floor, ceil, round, trunc
- cumsum, cumprod
- logical_and, logical_or, logical_not
- reciprocal, rsqrt, square
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.operators.base import (
    OperatorBenchmark,
    UnaryOperatorBenchmark,
    BinaryOperatorBenchmark,
)


class ClampOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for clamp operation."""

    op_name = "clamp"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.clamp(x, min=-1.0, max=1.0)

    def pytorch_operation(self, x):
        import torch
        return torch.clamp(x, min=-1.0, max=1.0)


class FloorOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for floor operation."""

    op_name = "floor"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.floor(x)

    def pytorch_operation(self, x):
        import torch
        return torch.floor(x)


class CeilOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for ceil operation."""

    op_name = "ceil"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.ceil(x)

    def pytorch_operation(self, x):
        import torch
        return torch.ceil(x)


class RoundOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for round operation."""

    op_name = "round"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.round(x)

    def pytorch_operation(self, x):
        import torch
        return torch.round(x)


class TruncOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for trunc operation."""

    op_name = "trunc"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.trunc(x)

    def pytorch_operation(self, x):
        import torch
        return torch.trunc(x)


class ReciprocalOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for reciprocal operation."""

    op_name = "reciprocal"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        # Use values away from zero to avoid division issues
        x = flashlight.randn(*config["shape"]) + 2.0
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device) + 2.0
        return (x,)

    def mlx_operation(self, x):
        import flashlight
        return flashlight.reciprocal(x)

    def pytorch_operation(self, x):
        import torch
        return torch.reciprocal(x)


class RsqrtOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for rsqrt (reciprocal square root) operation."""

    op_name = "rsqrt"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        # Use positive values for sqrt
        x = flashlight.abs(flashlight.randn(*config["shape"])) + 0.1
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.abs(torch.randn(*config["shape"], device=device)) + 0.1
        return (x,)

    def mlx_operation(self, x):
        import flashlight
        return flashlight.rsqrt(x)

    def pytorch_operation(self, x):
        import torch
        return torch.rsqrt(x)


class SquareOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for square operation."""

    op_name = "square"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.square(x)

    def pytorch_operation(self, x):
        import torch
        return torch.square(x)


class CumsumOperatorBenchmark(OperatorBenchmark):
    """Benchmark for cumulative sum operation."""

    name = "cumsum"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000,), "dim": 0},
            {"shape": (1000, 1000), "dim": 1},
            {"shape": (100, 100, 100), "dim": 2},
            {"shape": (32, 256, 256), "dim": 1},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["dim"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["dim"])

    def mlx_operation(self, x, dim):
        import flashlight
        return flashlight.cumsum(x, dim=dim)

    def pytorch_operation(self, x, dim):
        import torch
        return torch.cumsum(x, dim=dim)


class CumprodOperatorBenchmark(OperatorBenchmark):
    """Benchmark for cumulative product operation."""

    name = "cumprod"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000,), "dim": 0},
            {"shape": (1000, 1000), "dim": 1},
            {"shape": (100, 100, 100), "dim": 2},
            {"shape": (32, 256, 256), "dim": 1},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        # Use small values to avoid overflow
        x = flashlight.randn(*config["shape"]) * 0.1 + 1.0
        return (x, config["dim"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device) * 0.1 + 1.0
        return (x, config["dim"])

    def mlx_operation(self, x, dim):
        import flashlight
        return flashlight.cumprod(x, dim=dim)

    def pytorch_operation(self, x, dim):
        import torch
        return torch.cumprod(x, dim=dim)


class LogicalAndOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for logical AND operation."""

    op_name = "logical_and"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        a = flashlight.randn(*config["shape"]) > 0
        b = flashlight.randn(*config["shape"]) > 0
        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        a = torch.randn(*config["shape"], device=device) > 0
        b = torch.randn(*config["shape"], device=device) > 0
        return (a, b)

    def mlx_operation(self, a, b):
        import flashlight
        return flashlight.logical_and(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.logical_and(a, b)


class LogicalOrOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for logical OR operation."""

    op_name = "logical_or"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        a = flashlight.randn(*config["shape"]) > 0
        b = flashlight.randn(*config["shape"]) > 0
        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        a = torch.randn(*config["shape"], device=device) > 0
        b = torch.randn(*config["shape"], device=device) > 0
        return (a, b)

    def mlx_operation(self, a, b):
        import flashlight
        return flashlight.logical_or(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.logical_or(a, b)


class LogicalNotOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for logical NOT operation."""

    op_name = "logical_not"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"]) > 0
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device) > 0
        return (x,)

    def mlx_operation(self, x):
        import flashlight
        return flashlight.logical_not(x)

    def pytorch_operation(self, x):
        import torch
        return torch.logical_not(x)


MATH_FUNCS_BENCHMARKS = [
    ClampOperatorBenchmark,
    FloorOperatorBenchmark,
    CeilOperatorBenchmark,
    RoundOperatorBenchmark,
    TruncOperatorBenchmark,
    ReciprocalOperatorBenchmark,
    RsqrtOperatorBenchmark,
    SquareOperatorBenchmark,
    CumsumOperatorBenchmark,
    CumprodOperatorBenchmark,
    LogicalAndOperatorBenchmark,
    LogicalOrOperatorBenchmark,
    LogicalNotOperatorBenchmark,
]
