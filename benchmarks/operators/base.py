"""
Base class for operator benchmarks.
"""

from typing import List, Dict, Any, Tuple
from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BaseBenchmark


class OperatorBenchmark(BaseBenchmark):
    """
    Base class for operator-level benchmarks.

    Operator benchmarks focus on individual operations like matmul,
    add, relu, etc. They test raw operation performance without
    the overhead of module initialization.

    Subclasses should implement:
    - get_input_configs(): Return configurations to benchmark
    - create_mlx_inputs(config): Create mlx_compat tensors
    - create_pytorch_inputs(config, device): Create PyTorch tensors
    - mlx_operation(*inputs): The MLX operation
    - pytorch_operation(*inputs): The PyTorch operation
    """

    level = BenchmarkLevel.OPERATOR

    def format_input_shape(self, config: Dict[str, Any]) -> str:
        """Format input configuration as a readable string."""
        parts = []
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                parts.append(f"{key}={list(value)}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)


class UnaryOperatorBenchmark(OperatorBenchmark):
    """Base class for unary operators (single input)."""

    # Override in subclass
    op_name: str = "unary_op"
    mlx_op = None  # Callable
    torch_op = None  # Callable

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.name = self.op_name

    def get_input_configs(self) -> List[Dict[str, Any]]:
        """Standard configurations for unary operators."""
        return [
            {"shape": (1000,)},
            {"shape": (1000, 1000)},
            {"shape": (100, 100, 100)},
            {"shape": (32, 256, 256)},
            {"shape": (16, 128, 128, 128)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        x = mlx_compat.randn(*config["shape"])
        return (x,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x,)

    def mlx_operation(self, x):
        return self.mlx_op(x)

    def pytorch_operation(self, x):
        return self.torch_op(x)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate elements per second."""
        if time_ms <= 0:
            return 0.0, "elem/sec"

        shape = config["shape"]
        elements = 1
        for dim in shape:
            elements *= dim

        elements_per_sec = elements / (time_ms / 1000.0)

        # Format with SI prefix
        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        else:
            return elements_per_sec, "elem/sec"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        """Calculate bytes for bandwidth estimation."""
        shape = config["shape"]
        elements = 1
        for dim in shape:
            elements *= dim

        # Read input + write output (both float32)
        return elements * 4 * 2


class BinaryOperatorBenchmark(OperatorBenchmark):
    """Base class for binary operators (two inputs of same shape)."""

    op_name: str = "binary_op"
    mlx_op = None
    torch_op = None

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.name = self.op_name

    def get_input_configs(self) -> List[Dict[str, Any]]:
        """Standard configurations for binary operators."""
        return [
            {"shape": (1000,)},
            {"shape": (1000, 1000)},
            {"shape": (100, 100, 100)},
            {"shape": (32, 256, 256)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        a = mlx_compat.randn(*config["shape"])
        b = mlx_compat.randn(*config["shape"])
        return (a, b)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        a = torch.randn(*config["shape"], device=device)
        b = torch.randn(*config["shape"], device=device)
        return (a, b)

    def mlx_operation(self, a, b):
        return self.mlx_op(a, b)

    def pytorch_operation(self, a, b):
        return self.torch_op(a, b)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "elem/sec"

        shape = config["shape"]
        elements = 1
        for dim in shape:
            elements *= dim

        elements_per_sec = elements / (time_ms / 1000.0)

        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        else:
            return elements_per_sec, "elem/sec"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        shape = config["shape"]
        elements = 1
        for dim in shape:
            elements *= dim
        # Read two inputs + write output
        return elements * 4 * 3


class MatmulBenchmark(OperatorBenchmark):
    """Base class for matrix multiplication benchmarks."""

    def get_input_configs(self) -> List[Dict[str, Any]]:
        """Standard matmul configurations."""
        return [
            # Square matrices
            {"m": 128, "k": 128, "n": 128},
            {"m": 256, "k": 256, "n": 256},
            {"m": 512, "k": 512, "n": 512},
            {"m": 1024, "k": 1024, "n": 1024},
            {"m": 2048, "k": 2048, "n": 2048},
            # Rectangular
            {"m": 1024, "k": 256, "n": 1024},
            {"m": 4096, "k": 1024, "n": 4096},
            # Batched
            {"batch": 32, "m": 64, "k": 64, "n": 64},
            {"batch": 32, "m": 256, "k": 256, "n": 256},
        ]

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate GFLOPS for matmul."""
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        m, k, n = config["m"], config["k"], config["n"]
        batch = config.get("batch", 1)

        # FLOPs = 2 * M * K * N * batch (multiply-add)
        flops = 2 * m * k * n * batch
        gflops = flops / (time_ms / 1000.0) / 1e9

        return gflops, "GFLOPS"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        m, k, n = config["m"], config["k"], config["n"]
        batch = config.get("batch", 1)

        # Read A (m*k) + B (k*n), write C (m*n)
        elements = (m * k + k * n + m * n) * batch
        return elements * 4  # float32
