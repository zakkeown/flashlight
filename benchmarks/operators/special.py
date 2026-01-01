"""
Special function benchmarks.

Benchmarks for torch.special functions:
- gammaln, digamma, polygamma
- bessel functions (j0, j1, i0, i1)
- ndtr, ndtri
- chebyshev, hermite, laguerre, legendre polynomials
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark


class GammalnBenchmark(OperatorBenchmark):
    """Benchmark for gammaln operation."""

    name = "gammaln"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        # Use positive values for gammaln
        x = mlx_compat.rand(config["size"]) * 10 + 0.1
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.rand(config["size"], device=device) * 10 + 0.1
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.gammaln(x)

    def pytorch_operation(self, x):
        import torch

        return torch.special.gammaln(x)


class DigammaBenchmark(OperatorBenchmark):
    """Benchmark for digamma operation."""

    name = "digamma"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.rand(config["size"]) * 10 + 1.0
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.rand(config["size"], device=device) * 10 + 1.0
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.digamma(x)

    def pytorch_operation(self, x):
        import torch

        return torch.special.digamma(x)


class BesselJ0Benchmark(OperatorBenchmark):
    """Benchmark for bessel_j0 operation."""

    name = "bessel_j0"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.rand(config["size"]) * 20
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.rand(config["size"], device=device) * 20
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.bessel_j0(x)

    def pytorch_operation(self, x):
        import torch

        return torch.special.bessel_j0(x)


class ModifiedBesselI0Benchmark(OperatorBenchmark):
    """Benchmark for modified_bessel_i0 operation."""

    name = "modified_bessel_i0"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.rand(config["size"]) * 5
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.rand(config["size"], device=device) * 5
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.modified_bessel_i0(x)

    def pytorch_operation(self, x):
        import torch

        return torch.special.i0(x)


class NdtrBenchmark(OperatorBenchmark):
    """Benchmark for ndtr (normal CDF) operation."""

    name = "ndtr"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["size"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["size"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.ndtr(x)

    def pytorch_operation(self, x):
        import torch

        return torch.special.ndtr(x)


class NdtriBenchmark(OperatorBenchmark):
    """Benchmark for ndtri (inverse normal CDF) operation."""

    name = "ndtri"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        # Values in (0, 1) for ndtri
        x = mlx_compat.rand(config["size"]) * 0.98 + 0.01
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.rand(config["size"], device=device) * 0.98 + 0.01
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.ndtri(x)

    def pytorch_operation(self, x):
        import torch

        return torch.special.ndtri(x)


class ErfBenchmark(OperatorBenchmark):
    """Benchmark for erf operation."""

    name = "erf"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"size": 1000},
            {"size": 10000},
            {"size": 100000},
            {"size": 1000000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["size"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["size"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.erf(x)

    def pytorch_operation(self, x):
        import torch

        return torch.erf(x)


class SoftmaxBenchmark(OperatorBenchmark):
    """Benchmark for softmax operation."""

    name = "softmax"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "features": 1000},
            {"batch": 64, "features": 1000},
            {"batch": 128, "features": 1000},
            {"batch": 32, "features": 10000},
            {"batch": 64, "features": 10000},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["batch"], config["features"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["batch"], config["features"], device=device)
        return (x,)

    def mlx_operation(self, x):
        import mlx_compat

        return mlx_compat.special.softmax(x, dim=-1)

    def pytorch_operation(self, x):
        import torch

        return torch.softmax(x, dim=-1)


# Export all benchmark classes
BENCHMARKS = [
    GammalnBenchmark,
    DigammaBenchmark,
    BesselJ0Benchmark,
    ModifiedBesselI0Benchmark,
    NdtrBenchmark,
    NdtriBenchmark,
    ErfBenchmark,
    SoftmaxBenchmark,
]
