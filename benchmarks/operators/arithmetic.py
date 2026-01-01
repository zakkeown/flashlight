"""
Arithmetic operator benchmarks.

Benchmarks for:
- matmul, mm, bmm
- add, sub, mul, div
- pow, sqrt, exp, log
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.operators.base import (
    OperatorBenchmark,
    UnaryOperatorBenchmark,
    BinaryOperatorBenchmark,
    MatmulBenchmark,
)


class MatmulOperatorBenchmark(MatmulBenchmark):
    """Benchmark for matrix multiplication (@ operator)."""

    name = "matmul"
    level = BenchmarkLevel.OPERATOR

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight

        m, k, n = config["m"], config["k"], config["n"]
        batch = config.get("batch", None)

        if batch:
            a = flashlight.randn(batch, m, k)
            b = flashlight.randn(batch, k, n)
        else:
            a = flashlight.randn(m, k)
            b = flashlight.randn(k, n)

        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        m, k, n = config["m"], config["k"], config["n"]
        batch = config.get("batch", None)

        if batch:
            a = torch.randn(batch, m, k, device=device)
            b = torch.randn(batch, k, n, device=device)
        else:
            a = torch.randn(m, k, device=device)
            b = torch.randn(k, n, device=device)

        return (a, b)

    def mlx_operation(self, a, b):
        return a @ b

    def pytorch_operation(self, a, b):
        return a @ b


class AddOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise addition."""

    op_name = "add"

    def mlx_operation(self, a, b):
        return a + b

    def pytorch_operation(self, a, b):
        return a + b


class SubOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise subtraction."""

    op_name = "sub"

    def mlx_operation(self, a, b):
        return a - b

    def pytorch_operation(self, a, b):
        return a - b


class MulOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise multiplication."""

    op_name = "mul"

    def mlx_operation(self, a, b):
        return a * b

    def pytorch_operation(self, a, b):
        return a * b


class DivOperatorBenchmark(BinaryOperatorBenchmark):
    """Benchmark for element-wise division."""

    op_name = "div"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        a = flashlight.randn(*config["shape"])
        # Avoid division by zero
        b = flashlight.randn(*config["shape"]).abs() + 0.1
        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        a = torch.randn(*config["shape"], device=device)
        b = torch.randn(*config["shape"], device=device).abs() + 0.1
        return (a, b)

    def mlx_operation(self, a, b):
        return a / b

    def pytorch_operation(self, a, b):
        return a / b


class PowOperatorBenchmark(OperatorBenchmark):
    """Benchmark for power operation."""

    name = "pow"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000, 1000), "exponent": 2},
            {"shape": (1000, 1000), "exponent": 0.5},
            {"shape": (32, 256, 256), "exponent": 2},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"]).abs() + 0.1
        return (x, config["exponent"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device).abs() + 0.1
        return (x, config["exponent"])

    def mlx_operation(self, x, exp):
        return x ** exp

    def pytorch_operation(self, x, exp):
        return x ** exp


class SqrtOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for square root."""

    op_name = "sqrt"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"]).abs() + 0.1
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device).abs() + 0.1
        return (x,)

    def mlx_operation(self, x):
        import flashlight
        return flashlight.sqrt(x)

    def pytorch_operation(self, x):
        import torch
        return torch.sqrt(x)


class ExpOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for exponential."""

    op_name = "exp"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        # Clamp to avoid overflow
        x = flashlight.randn(*config["shape"]).clamp(-10, 10)
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device).clamp(-10, 10)
        return (x,)

    def mlx_operation(self, x):
        import flashlight
        return flashlight.exp(x)

    def pytorch_operation(self, x):
        import torch
        return torch.exp(x)


class LogOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for natural logarithm."""

    op_name = "log"

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"]).abs() + 0.1
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device).abs() + 0.1
        return (x,)

    def mlx_operation(self, x):
        import flashlight
        return flashlight.log(x)

    def pytorch_operation(self, x):
        import torch
        return torch.log(x)


class AbsOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for absolute value."""

    op_name = "abs"

    def mlx_operation(self, x):
        import flashlight
        return flashlight.abs(x)

    def pytorch_operation(self, x):
        import torch
        return torch.abs(x)


class NegOperatorBenchmark(UnaryOperatorBenchmark):
    """Benchmark for negation."""

    op_name = "neg"

    def mlx_operation(self, x):
        return -x

    def pytorch_operation(self, x):
        return -x


# List of all arithmetic benchmarks
ARITHMETIC_BENCHMARKS = [
    MatmulOperatorBenchmark,
    AddOperatorBenchmark,
    SubOperatorBenchmark,
    MulOperatorBenchmark,
    DivOperatorBenchmark,
    PowOperatorBenchmark,
    SqrtOperatorBenchmark,
    ExpOperatorBenchmark,
    LogOperatorBenchmark,
    AbsOperatorBenchmark,
    NegOperatorBenchmark,
]
