"""
Linear algebra operator benchmarks.

Benchmarks for:
- einsum, tensordot
- mm, bmm
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark


class EinsumBenchmark(OperatorBenchmark):
    """Benchmark for einsum operation."""

    name = "einsum"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Matrix multiplication via einsum
            {"equation": "ij,jk->ik", "shapes": [(512, 512), (512, 512)]},
            {"equation": "ij,jk->ik", "shapes": [(1024, 1024), (1024, 1024)]},
            # Batch matmul
            {"equation": "bij,bjk->bik", "shapes": [(32, 64, 64), (32, 64, 64)]},
            # Transpose
            {"equation": "ij->ji", "shapes": [(1024, 1024)]},
            # Trace
            {"equation": "ii->", "shapes": [(1024, 1024)]},
            # Outer product
            {"equation": "i,j->ij", "shapes": [(1024,), (1024,)]},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight

        tensors = [flashlight.randn(*shape) for shape in config["shapes"]]
        return (config["equation"], tensors)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        tensors = [torch.randn(*shape, device=device) for shape in config["shapes"]]
        return (config["equation"], tensors)

    def mlx_operation(self, equation, tensors):
        import flashlight
        return flashlight.einsum(equation, *tensors)

    def pytorch_operation(self, equation, tensors):
        import torch
        return torch.einsum(equation, *tensors)


class BmmBenchmark(OperatorBenchmark):
    """Benchmark for batch matrix multiplication."""

    name = "bmm"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "m": 64, "k": 64, "n": 64},
            {"batch": 32, "m": 128, "k": 128, "n": 128},
            {"batch": 64, "m": 64, "k": 64, "n": 64},
            {"batch": 16, "m": 256, "k": 256, "n": 256},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight

        a = flashlight.randn(config["batch"], config["m"], config["k"])
        b = flashlight.randn(config["batch"], config["k"], config["n"])
        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        a = torch.randn(config["batch"], config["m"], config["k"], device=device)
        b = torch.randn(config["batch"], config["k"], config["n"], device=device)
        return (a, b)

    def mlx_operation(self, a, b):
        import flashlight
        return flashlight.bmm(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.bmm(a, b)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        batch = config["batch"]
        m, k, n = config["m"], config["k"], config["n"]
        flops = batch * 2 * m * k * n
        gflops = flops / (time_ms / 1000.0) / 1e9

        return gflops, "GFLOPS"


class MmBenchmark(OperatorBenchmark):
    """Benchmark for matrix multiplication (mm)."""

    name = "mm"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"m": 256, "k": 256, "n": 256},
            {"m": 512, "k": 512, "n": 512},
            {"m": 1024, "k": 1024, "n": 1024},
            {"m": 2048, "k": 2048, "n": 2048},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight

        a = flashlight.randn(config["m"], config["k"])
        b = flashlight.randn(config["k"], config["n"])
        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        a = torch.randn(config["m"], config["k"], device=device)
        b = torch.randn(config["k"], config["n"], device=device)
        return (a, b)

    def mlx_operation(self, a, b):
        import flashlight
        return flashlight.mm(a, b)

    def pytorch_operation(self, a, b):
        import torch
        return torch.mm(a, b)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        m, k, n = config["m"], config["k"], config["n"]
        flops = 2 * m * k * n
        gflops = flops / (time_ms / 1000.0) / 1e9

        return gflops, "GFLOPS"


# List of all linalg benchmarks
LINALG_BENCHMARKS = [
    EinsumBenchmark,
    BmmBenchmark,
    MmBenchmark,
]
