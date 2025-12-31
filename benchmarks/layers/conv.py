"""
Convolutional layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark
from benchmarks.core.memory import calculate_conv2d_bytes


class Conv2dLayerBenchmark(LayerBenchmark):
    """Benchmark for nn.Conv2d layer."""

    name = "Conv2d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # First conv (RGB input)
            {"batch": 32, "in_channels": 3, "out_channels": 64, "h": 224, "w": 224, "kernel": 7, "stride": 2, "padding": 3},
            # Standard 3x3 convs
            {"batch": 32, "in_channels": 64, "out_channels": 64, "h": 56, "w": 56, "kernel": 3, "stride": 1, "padding": 1},
            {"batch": 32, "in_channels": 64, "out_channels": 128, "h": 56, "w": 56, "kernel": 3, "stride": 2, "padding": 1},
            {"batch": 32, "in_channels": 128, "out_channels": 256, "h": 28, "w": 28, "kernel": 3, "stride": 2, "padding": 1},
            {"batch": 32, "in_channels": 256, "out_channels": 512, "h": 14, "w": 14, "kernel": 3, "stride": 2, "padding": 1},
            # CIFAR-10 style
            {"batch": 64, "in_channels": 3, "out_channels": 32, "h": 32, "w": 32, "kernel": 3, "stride": 1, "padding": 1},
            # 1x1 convs (pointwise)
            {"batch": 32, "in_channels": 256, "out_channels": 64, "h": 28, "w": 28, "kernel": 1, "stride": 1, "padding": 0},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.Conv2d(
            config["in_channels"], config["out_channels"],
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
        )

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.Conv2d(
            config["in_channels"], config["out_channels"],
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
        )
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["in_channels"], config["h"], config["w"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["in_channels"], config["h"], config["w"], device=device)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        batch = config["batch"]
        c_in = config["in_channels"]
        c_out = config["out_channels"]
        h = config["h"]
        w = config["w"]
        k = config["kernel"]
        stride = config["stride"]

        h_out = (h + 2 * config["padding"] - k) // stride + 1
        w_out = (w + 2 * config["padding"] - k) // stride + 1

        flops = batch * c_out * h_out * w_out * (2 * c_in * k * k)
        gflops = flops / (time_ms / 1000.0) / 1e9

        return gflops, "GFLOPS"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        return calculate_conv2d_bytes(
            config["batch"], config["in_channels"], config["out_channels"],
            config["h"], config["w"], config["kernel"],
        )


class Conv1dLayerBenchmark(LayerBenchmark):
    """Benchmark for nn.Conv1d layer."""

    name = "Conv1d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "in_channels": 64, "out_channels": 128, "length": 1024, "kernel": 3, "stride": 1, "padding": 1},
            {"batch": 64, "in_channels": 128, "out_channels": 256, "length": 512, "kernel": 3, "stride": 1, "padding": 1},
            {"batch": 32, "in_channels": 256, "out_channels": 512, "length": 256, "kernel": 3, "stride": 2, "padding": 1},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.Conv1d(
            config["in_channels"], config["out_channels"],
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
        )

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.Conv1d(
            config["in_channels"], config["out_channels"],
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
        )
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["in_channels"], config["length"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["in_channels"], config["length"], device=device)


# List of all conv benchmarks
CONV_BENCHMARKS = [
    Conv2dLayerBenchmark,
    Conv1dLayerBenchmark,
]
