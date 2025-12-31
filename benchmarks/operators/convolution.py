"""
Convolution operator benchmarks.

Benchmarks for:
- conv1d, conv2d, conv3d
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.operators.base import OperatorBenchmark
from benchmarks.core.memory import calculate_conv2d_bytes


class Conv2dBenchmark(OperatorBenchmark):
    """Benchmark for 2D convolution."""

    name = "conv2d"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Standard vision configurations
            {"batch": 1, "c_in": 3, "c_out": 64, "h": 224, "w": 224, "k": 7, "stride": 2, "padding": 3},
            {"batch": 8, "c_in": 3, "c_out": 64, "h": 224, "w": 224, "k": 7, "stride": 2, "padding": 3},
            {"batch": 32, "c_in": 3, "c_out": 64, "h": 224, "w": 224, "k": 3, "stride": 1, "padding": 1},
            # Deeper layers
            {"batch": 32, "c_in": 64, "c_out": 128, "h": 56, "w": 56, "k": 3, "stride": 1, "padding": 1},
            {"batch": 32, "c_in": 128, "c_out": 256, "h": 28, "w": 28, "k": 3, "stride": 1, "padding": 1},
            {"batch": 32, "c_in": 256, "c_out": 512, "h": 14, "w": 14, "k": 3, "stride": 1, "padding": 1},
            # Small input (CIFAR-10 style)
            {"batch": 64, "c_in": 3, "c_out": 32, "h": 32, "w": 32, "k": 3, "stride": 1, "padding": 1},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat
        import mlx_compat.nn.functional as F

        x = mlx_compat.randn(config["batch"], config["c_in"], config["h"], config["w"])
        weight = mlx_compat.randn(config["c_out"], config["c_in"], config["k"], config["k"])

        return (x, weight, config["stride"], config["padding"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["batch"], config["c_in"], config["h"], config["w"], device=device)
        weight = torch.randn(config["c_out"], config["c_in"], config["k"], config["k"], device=device)

        return (x, weight, config["stride"], config["padding"])

    def mlx_operation(self, x, weight, stride, padding):
        import mlx_compat.nn.functional as F
        return F.conv2d(x, weight, stride=stride, padding=padding)

    def pytorch_operation(self, x, weight, stride, padding):
        import torch.nn.functional as F
        return F.conv2d(x, weight, stride=stride, padding=padding)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate GFLOPS for conv2d."""
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        batch = config["batch"]
        c_in = config["c_in"]
        c_out = config["c_out"]
        h = config["h"]
        w = config["w"]
        k = config["k"]
        stride = config["stride"]

        # Output spatial dimensions
        h_out = (h + 2 * config["padding"] - k) // stride + 1
        w_out = (w + 2 * config["padding"] - k) // stride + 1

        # FLOPs per output element: 2 * c_in * k * k (multiply-add)
        flops = batch * c_out * h_out * w_out * (2 * c_in * k * k)
        gflops = flops / (time_ms / 1000.0) / 1e9

        return gflops, "GFLOPS"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        return calculate_conv2d_bytes(
            config["batch"], config["c_in"], config["c_out"],
            config["h"], config["w"], config["k"],
        )


class Conv1dBenchmark(OperatorBenchmark):
    """Benchmark for 1D convolution."""

    name = "conv1d"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "c_in": 64, "c_out": 128, "length": 1024, "k": 3, "stride": 1, "padding": 1},
            {"batch": 32, "c_in": 128, "c_out": 256, "length": 512, "k": 3, "stride": 1, "padding": 1},
            {"batch": 64, "c_in": 256, "c_out": 512, "length": 256, "k": 3, "stride": 1, "padding": 1},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["batch"], config["c_in"], config["length"])
        weight = mlx_compat.randn(config["c_out"], config["c_in"], config["k"])

        return (x, weight, config["stride"], config["padding"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["batch"], config["c_in"], config["length"], device=device)
        weight = torch.randn(config["c_out"], config["c_in"], config["k"], device=device)

        return (x, weight, config["stride"], config["padding"])

    def mlx_operation(self, x, weight, stride, padding):
        import mlx_compat.nn.functional as F
        return F.conv1d(x, weight, stride=stride, padding=padding)

    def pytorch_operation(self, x, weight, stride, padding):
        import torch.nn.functional as F
        return F.conv1d(x, weight, stride=stride, padding=padding)


class Conv3dBenchmark(OperatorBenchmark):
    """Benchmark for 3D convolution."""

    name = "conv3d"
    level = BenchmarkLevel.OPERATOR

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 4, "c_in": 3, "c_out": 32, "d": 16, "h": 32, "w": 32, "k": 3, "stride": 1, "padding": 1},
            {"batch": 8, "c_in": 32, "c_out": 64, "d": 8, "h": 16, "w": 16, "k": 3, "stride": 1, "padding": 1},
        ]

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        import mlx_compat

        x = mlx_compat.randn(config["batch"], config["c_in"], config["d"], config["h"], config["w"])
        weight = mlx_compat.randn(config["c_out"], config["c_in"], config["k"], config["k"], config["k"])

        return (x, weight, config["stride"], config["padding"])

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        import torch

        x = torch.randn(config["batch"], config["c_in"], config["d"], config["h"], config["w"], device=device)
        weight = torch.randn(config["c_out"], config["c_in"], config["k"], config["k"], config["k"], device=device)

        return (x, weight, config["stride"], config["padding"])

    def mlx_operation(self, x, weight, stride, padding):
        import mlx_compat.nn.functional as F
        return F.conv3d(x, weight, stride=stride, padding=padding)

    def pytorch_operation(self, x, weight, stride, padding):
        import torch.nn.functional as F
        return F.conv3d(x, weight, stride=stride, padding=padding)


# List of all convolution benchmarks
CONVOLUTION_BENCHMARKS = [
    Conv2dBenchmark,
    Conv1dBenchmark,
    Conv3dBenchmark,
]
