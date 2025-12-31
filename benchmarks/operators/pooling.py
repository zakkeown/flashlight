"""
Pooling operator benchmarks.

Benchmarks for:
- max_pool2d, avg_pool2d
- adaptive_avg_pool2d
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark


class MaxPool2dBenchmark(OperatorBenchmark):
    """Benchmark for 2D max pooling."""

    name = "max_pool2d"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 64, "h": 112, "w": 112, "k": 3, "stride": 2, "padding": 1},
            {"batch": 32, "channels": 64, "h": 224, "w": 224, "k": 2, "stride": 2, "padding": 0},
            {"batch": 64, "channels": 128, "h": 56, "w": 56, "k": 2, "stride": 2, "padding": 0},
            {"batch": 64, "channels": 256, "h": 28, "w": 28, "k": 2, "stride": 2, "padding": 0},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        x = mlx_compat.randn(config["batch"], config["channels"], config["h"], config["w"])
        return (x, config["k"], config["stride"], config["padding"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(config["batch"], config["channels"], config["h"], config["w"], device=device)
        return (x, config["k"], config["stride"], config["padding"])

    def mlx_operation(self, x, k, stride, padding):
        import mlx_compat.nn.functional as F
        return F.max_pool2d(x, kernel_size=k, stride=stride, padding=padding)

    def pytorch_operation(self, x, k, stride, padding):
        import torch.nn.functional as F
        return F.max_pool2d(x, kernel_size=k, stride=stride, padding=padding)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "elem/sec"

        batch = config["batch"]
        channels = config["channels"]
        h = config["h"]
        w = config["w"]
        elements = batch * channels * h * w

        elements_per_sec = elements / (time_ms / 1000.0)

        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        return elements_per_sec, "elem/sec"


class AvgPool2dBenchmark(MaxPool2dBenchmark):
    """Benchmark for 2D average pooling."""

    name = "avg_pool2d"

    def mlx_operation(self, x, k, stride, padding):
        import mlx_compat.nn.functional as F
        return F.avg_pool2d(x, kernel_size=k, stride=stride, padding=padding)

    def pytorch_operation(self, x, k, stride, padding):
        import torch.nn.functional as F
        return F.avg_pool2d(x, kernel_size=k, stride=stride, padding=padding)


class AdaptiveAvgPool2dBenchmark(OperatorBenchmark):
    """Benchmark for adaptive average pooling."""

    name = "adaptive_avg_pool2d"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 512, "h": 7, "w": 7, "output_size": (1, 1)},
            {"batch": 32, "channels": 2048, "h": 7, "w": 7, "output_size": (1, 1)},
            {"batch": 64, "channels": 256, "h": 14, "w": 14, "output_size": (1, 1)},
            {"batch": 32, "channels": 512, "h": 14, "w": 14, "output_size": (7, 7)},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        x = mlx_compat.randn(config["batch"], config["channels"], config["h"], config["w"])
        return (x, config["output_size"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(config["batch"], config["channels"], config["h"], config["w"], device=device)
        return (x, config["output_size"])

    def mlx_operation(self, x, output_size):
        import mlx_compat.nn.functional as F
        return F.adaptive_avg_pool2d(x, output_size)

    def pytorch_operation(self, x, output_size):
        import torch.nn.functional as F
        return F.adaptive_avg_pool2d(x, output_size)


class MaxPool1dBenchmark(OperatorBenchmark):
    """Benchmark for 1D max pooling."""

    name = "max_pool1d"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 64, "length": 1024, "k": 2, "stride": 2},
            {"batch": 64, "channels": 128, "length": 512, "k": 2, "stride": 2},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        x = mlx_compat.randn(config["batch"], config["channels"], config["length"])
        return (x, config["k"], config["stride"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch
        x = torch.randn(config["batch"], config["channels"], config["length"], device=device)
        return (x, config["k"], config["stride"])

    def mlx_operation(self, x, k, stride):
        import mlx_compat.nn.functional as F
        return F.max_pool1d(x, kernel_size=k, stride=stride)

    def pytorch_operation(self, x, k, stride):
        import torch.nn.functional as F
        return F.max_pool1d(x, kernel_size=k, stride=stride)


# List of all pooling benchmarks
POOLING_BENCHMARKS = [
    MaxPool2dBenchmark,
    AvgPool2dBenchmark,
    AdaptiveAvgPool2dBenchmark,
    MaxPool1dBenchmark,
]
