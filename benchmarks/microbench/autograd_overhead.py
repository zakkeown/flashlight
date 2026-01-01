"""
Autograd overhead benchmarks.

Measures the overhead of gradient tracking and tape construction.
"""

from typing import List, Dict, Any, Tuple
from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BaseBenchmark


class NoGradVsGradBenchmark(BaseBenchmark):
    """Compare forward pass with and without gradient tracking."""

    name = "no_grad_vs_grad"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._mode = "grad"  # "grad" or "no_grad"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Small ops where overhead is proportionally larger
            {"shape": (100,), "op": "add", "mode": "no_grad"},
            {"shape": (100,), "op": "add", "mode": "grad"},
            {"shape": (100,), "op": "mul", "mode": "no_grad"},
            {"shape": (100,), "op": "mul", "mode": "grad"},
            # Large ops where overhead is proportionally smaller
            {"shape": (1000, 1000), "op": "add", "mode": "no_grad"},
            {"shape": (1000, 1000), "op": "add", "mode": "grad"},
            {"shape": (1000, 1000), "op": "matmul", "mode": "no_grad"},
            {"shape": (1000, 1000), "op": "matmul", "mode": "grad"},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        shape = config["shape"]
        requires_grad = config["mode"] == "grad"

        if config["op"] == "matmul":
            a = flashlight.randn(shape[0], shape[0], requires_grad=requires_grad)
            b = flashlight.randn(shape[0], shape[0], requires_grad=requires_grad)
        else:
            a = flashlight.randn(*shape, requires_grad=requires_grad)
            b = flashlight.randn(*shape, requires_grad=requires_grad)

        return (a, b, config["op"], config["mode"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        shape = config["shape"]
        requires_grad = config["mode"] == "grad"

        if config["op"] == "matmul":
            a = torch.randn(shape[0], shape[0], device=device, requires_grad=requires_grad)
            b = torch.randn(shape[0], shape[0], device=device, requires_grad=requires_grad)
        else:
            a = torch.randn(*shape, device=device, requires_grad=requires_grad)
            b = torch.randn(*shape, device=device, requires_grad=requires_grad)

        return (a, b, config["op"], config["mode"])

    def mlx_operation(self, a, b, op, mode):
        import flashlight

        if mode == "no_grad":
            with flashlight.no_grad():
                if op == "add":
                    return a + b
                elif op == "mul":
                    return a * b
                elif op == "matmul":
                    return a @ b
        else:
            if op == "add":
                return a + b
            elif op == "mul":
                return a * b
            elif op == "matmul":
                return a @ b

    def pytorch_operation(self, a, b, op, mode):
        import torch

        if mode == "no_grad":
            with torch.no_grad():
                if op == "add":
                    return a + b
                elif op == "mul":
                    return a * b
                elif op == "matmul":
                    return a @ b
        else:
            if op == "add":
                return a + b
            elif op == "mul":
                return a * b
            elif op == "matmul":
                return a @ b


class TapeConstructionBenchmark(BaseBenchmark):
    """Measure overhead of autograd tape construction for various op counts."""

    name = "tape_construction"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"op_count": 1, "shape": (1000, 1000)},
            {"op_count": 5, "shape": (1000, 1000)},
            {"op_count": 10, "shape": (1000, 1000)},
            {"op_count": 20, "shape": (1000, 1000)},
            {"op_count": 50, "shape": (1000, 1000)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"], requires_grad=True)
        return (x, config["op_count"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device, requires_grad=True)
        return (x, config["op_count"])

    def mlx_operation(self, x, op_count):
        """Chain of operations to build up tape."""
        result = x
        for i in range(op_count):
            if i % 3 == 0:
                result = result + 0.1
            elif i % 3 == 1:
                result = result * 1.01
            else:
                result = result - 0.05
        return result

    def pytorch_operation(self, x, op_count):
        """Chain of operations to build up tape."""
        result = x
        for i in range(op_count):
            if i % 3 == 0:
                result = result + 0.1
            elif i % 3 == 1:
                result = result * 1.01
            else:
                result = result - 0.05
        return result


class BackwardPassBenchmark(BaseBenchmark):
    """Measure backward pass overhead for different graph sizes."""

    name = "backward_pass"
    level = BenchmarkLevel.OPERATOR

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"op_count": 5, "shape": (1000, 1000)},
            {"op_count": 10, "shape": (1000, 1000)},
            {"op_count": 20, "shape": (1000, 1000)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import flashlight
        x = flashlight.randn(*config["shape"], requires_grad=True)
        return (x, config["op_count"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(*config["shape"], device=device, requires_grad=True)
        return (x, config["op_count"])

    def mlx_operation(self, x, op_count):
        """Build graph and run backward."""
        result = x
        for i in range(op_count):
            if i % 3 == 0:
                result = result + 0.1
            elif i % 3 == 1:
                result = result * 1.01
            else:
                result = result - 0.05

        loss = result.sum()
        loss.backward()
        return x.grad

    def pytorch_operation(self, x, op_count):
        """Build graph and run backward."""
        result = x
        for i in range(op_count):
            if i % 3 == 0:
                result = result + 0.1
            elif i % 3 == 1:
                result = result * 1.01
            else:
                result = result - 0.05

        loss = result.sum()
        loss.backward()
        return x.grad


# List of benchmark classes for registration
AUTOGRAD_BENCHMARKS = [
    NoGradVsGradBenchmark,
    TapeConstructionBenchmark,
    BackwardPassBenchmark,
]
