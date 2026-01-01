"""
Pooling layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class MaxPool2dBenchmark(LayerBenchmark):
    """Benchmark for nn.MaxPool2d layer."""

    name = "MaxPool2d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Small feature maps
            {"batch": 32, "channels": 64, "height": 32, "width": 32, "kernel": 2, "stride": 2},
            {"batch": 64, "channels": 128, "height": 16, "width": 16, "kernel": 2, "stride": 2},
            # Larger feature maps
            {"batch": 16, "channels": 64, "height": 112, "width": 112, "kernel": 3, "stride": 2},
            {"batch": 32, "channels": 256, "height": 56, "width": 56, "kernel": 2, "stride": 2},
            # ResNet-style
            {"batch": 32, "channels": 64, "height": 112, "width": 112, "kernel": 3, "stride": 2},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.MaxPool2d(kernel_size=config["kernel"], stride=config["stride"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.MaxPool2d(kernel_size=config["kernel"], stride=config["stride"])
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(
            config["batch"], config["channels"], config["height"], config["width"]
        )

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(
            config["batch"], config["channels"], config["height"], config["width"],
            device=device
        )

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GB/s"

        bytes_processed = self.calculate_bytes(config)
        gb_per_s = (bytes_processed / (time_ms / 1000.0)) / 1e9
        return gb_per_s, "GB/s"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        b = config["batch"]
        c = config["channels"]
        h = config["height"]
        w = config["width"]
        k = config["kernel"]
        s = config["stride"]

        input_elements = b * c * h * w
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        output_elements = b * c * out_h * out_w

        return (input_elements + output_elements) * 4


class AvgPool2dBenchmark(MaxPool2dBenchmark):
    """Benchmark for nn.AvgPool2d layer."""

    name = "AvgPool2d"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.AvgPool2d(kernel_size=config["kernel"], stride=config["stride"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.AvgPool2d(kernel_size=config["kernel"], stride=config["stride"])
        return layer.to(device)


class AdaptiveAvgPool2dBenchmark(LayerBenchmark):
    """Benchmark for nn.AdaptiveAvgPool2d layer."""

    name = "AdaptiveAvgPool2d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 512, "height": 7, "width": 7, "out_size": 1},
            {"batch": 64, "channels": 2048, "height": 7, "width": 7, "out_size": 1},
            {"batch": 32, "channels": 256, "height": 14, "width": 14, "out_size": 1},
            {"batch": 16, "channels": 512, "height": 28, "width": 28, "out_size": 7},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.AdaptiveAvgPool2d((config["out_size"], config["out_size"]))

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.AdaptiveAvgPool2d((config["out_size"], config["out_size"]))
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(
            config["batch"], config["channels"], config["height"], config["width"]
        )

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(
            config["batch"], config["channels"], config["height"], config["width"],
            device=device
        )

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GB/s"
        bytes_processed = self.calculate_bytes(config)
        gb_per_s = (bytes_processed / (time_ms / 1000.0)) / 1e9
        return gb_per_s, "GB/s"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        b = config["batch"]
        c = config["channels"]
        h = config["height"]
        w = config["width"]
        out = config["out_size"]
        return (b * c * h * w + b * c * out * out) * 4


POOLING_BENCHMARKS = [
    MaxPool2dBenchmark,
    AvgPool2dBenchmark,
    AdaptiveAvgPool2dBenchmark,
]
