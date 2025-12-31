"""
Reduction operator benchmarks.

Benchmarks for:
- sum, mean, max, min
- argmax, argmin
- var, std
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark


class SumBenchmark(OperatorBenchmark):
    """Benchmark for sum reduction."""

    name = "sum"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (10000000,), "dim": None},  # Full reduction
            {"shape": (1000, 1000), "dim": None},
            {"shape": (1000, 1000), "dim": 0},  # Reduce along axis
            {"shape": (1000, 1000), "dim": 1},
            {"shape": (32, 256, 256), "dim": -1},
            {"shape": (32, 256, 256), "dim": (1, 2)},  # Multiple dims
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        x = mlx_compat.randn(*config["shape"])
        return (x, config["dim"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["dim"])

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.sum()
        return x.sum(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.sum()
        return x.sum(dim=dim)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "elem/sec"

        shape = config["shape"]
        elements = 1
        for d in shape:
            elements *= d

        elements_per_sec = elements / (time_ms / 1000.0)

        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        return elements_per_sec, "elem/sec"


class MeanBenchmark(SumBenchmark):
    """Benchmark for mean reduction."""

    name = "mean"

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.mean()
        return x.mean(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.mean()
        return x.mean(dim=dim)


class MaxBenchmark(SumBenchmark):
    """Benchmark for max reduction."""

    name = "max"

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.max()
        return x.max(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.max()
        # PyTorch returns (values, indices) when dim specified
        return x.max(dim=dim).values


class MinBenchmark(SumBenchmark):
    """Benchmark for min reduction."""

    name = "min"

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.min()
        return x.min(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.min()
        return x.min(dim=dim).values


class ArgmaxBenchmark(OperatorBenchmark):
    """Benchmark for argmax."""

    name = "argmax"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (10000000,), "dim": None},
            {"shape": (1000, 1000), "dim": 0},
            {"shape": (1000, 1000), "dim": 1},
            {"shape": (32, 256, 256), "dim": -1},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        x = mlx_compat.randn(*config["shape"])
        return (x, config["dim"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["dim"])

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.argmax()
        return x.argmax(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.argmax()
        return x.argmax(dim=dim)


class ArgminBenchmark(ArgmaxBenchmark):
    """Benchmark for argmin."""

    name = "argmin"

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.argmin()
        return x.argmin(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.argmin()
        return x.argmin(dim=dim)


class VarBenchmark(SumBenchmark):
    """Benchmark for variance."""

    name = "var"

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.var()
        return x.var(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.var()
        return x.var(dim=dim)


class StdBenchmark(SumBenchmark):
    """Benchmark for standard deviation."""

    name = "std"

    def mlx_operation(self, x, dim):
        if dim is None:
            return x.std()
        return x.std(dim=dim)

    def pytorch_operation(self, x, dim):
        if dim is None:
            return x.std()
        return x.std(dim=dim)


# List of all reduction benchmarks
REDUCTION_BENCHMARKS = [
    SumBenchmark,
    MeanBenchmark,
    MaxBenchmark,
    MinBenchmark,
    ArgmaxBenchmark,
    ArgminBenchmark,
    VarBenchmark,
    StdBenchmark,
]
