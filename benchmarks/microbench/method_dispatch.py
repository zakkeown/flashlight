"""
Method dispatch overhead benchmarks.

Measures the overhead of calling tensor.method() vs ops.function(tensor).
"""

from typing import List, Dict, Any, Tuple
from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BaseBenchmark


class MethodVsFunctionBenchmark(BaseBenchmark):
    """Compare tensor.sum() method vs flashlight.sum(tensor) function."""

    name = "method_vs_function"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000,), "op": "sum"},
            {"shape": (1000, 1000), "op": "sum"},
            {"shape": (1000, 1000), "op": "mean"},
            {"shape": (1000, 1000), "op": "max"},
            {"shape": (32, 256, 256), "op": "sum"},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["op"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["op"])

    def mlx_operation(self, x, op):
        """Use tensor method."""
        if op == "sum":
            return x.sum()
        elif op == "mean":
            return x.mean()
        elif op == "max":
            return x.max()

    def pytorch_operation(self, x, op):
        """Use tensor method."""
        if op == "sum":
            return x.sum()
        elif op == "mean":
            return x.mean()
        elif op == "max":
            return x.max()


class FunctionCallBenchmark(BaseBenchmark):
    """Benchmark using function call instead of method."""

    name = "function_call"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000,), "op": "sum"},
            {"shape": (1000, 1000), "op": "sum"},
            {"shape": (1000, 1000), "op": "mean"},
            {"shape": (32, 256, 256), "op": "sum"},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["op"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["op"])

    def mlx_operation(self, x, op):
        """Use module function."""
        import flashlight
        if op == "sum":
            return flashlight.sum(x)
        elif op == "mean":
            return flashlight.mean(x)
        elif op == "max":
            return flashlight.max(x)

    def pytorch_operation(self, x, op):
        """Use module function."""
        import torch
        if op == "sum":
            return torch.sum(x)
        elif op == "mean":
            return torch.mean(x)
        elif op == "max":
            return torch.max(x)


class TensorCreationBenchmark(BaseBenchmark):
    """Measure overhead of Tensor wrapper creation."""

    name = "tensor_creation"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (100,)},
            {"shape": (1000,)},
            {"shape": (1000, 1000)},
            {"shape": (100, 100, 100)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import numpy as np
        data = np.random.randn(*config["shape"]).astype(np.float32)
        return (data,)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import numpy as np
        data = np.random.randn(*config["shape"]).astype(np.float32)
        return (data, device)

    def mlx_operation(self, data):
        import flashlight
        return flashlight.tensor(data)

    def pytorch_operation(self, data, device):
        import torch
        return torch.tensor(data, device=device)


class PropertyAccessBenchmark(BaseBenchmark):
    """Measure overhead of accessing tensor properties."""

    name = "property_access"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"shape": (1000, 1000), "property": "shape"},
            {"shape": (1000, 1000), "property": "dtype"},
            {"shape": (1000, 1000), "property": "ndim"},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"])
        return (x, config["property"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device)
        return (x, config["property"])

    def mlx_operation(self, x, prop):
        """Access property multiple times to measure overhead."""
        for _ in range(1000):
            _ = getattr(x, prop)
        return getattr(x, prop)

    def pytorch_operation(self, x, prop):
        """Access property multiple times to measure overhead."""
        for _ in range(1000):
            _ = getattr(x, prop)
        return getattr(x, prop)


# List of benchmark classes for registration
DISPATCH_BENCHMARKS = [
    MethodVsFunctionBenchmark,
    FunctionCallBenchmark,
    TensorCreationBenchmark,
    PropertyAccessBenchmark,
]
