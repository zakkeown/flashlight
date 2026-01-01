"""
Padding layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class ZeroPad2dBenchmark(LayerBenchmark):
    """Benchmark for nn.ZeroPad2d layer."""

    name = "ZeroPad2d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 64, "height": 56, "width": 56, "padding": 1},
            {"batch": 32, "channels": 128, "height": 28, "width": 28, "padding": 2},
            {"batch": 16, "channels": 256, "height": 14, "width": 14, "padding": 3},
            {"batch": 8, "channels": 512, "height": 7, "width": 7, "padding": 1},
            # Asymmetric padding
            {"batch": 32, "channels": 64, "height": 32, "width": 32, "padding": (1, 2, 1, 2)},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.ZeroPad2d(config["padding"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.ZeroPad2d(config["padding"])

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(
            config["batch"], config["channels"],
            config["height"], config["width"]
        )

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(
            config["batch"], config["channels"],
            config["height"], config["width"],
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
        batch = config["batch"]
        channels = config["channels"]
        h = config["height"]
        w = config["width"]
        padding = config["padding"]

        if isinstance(padding, int):
            out_h = h + 2 * padding
            out_w = w + 2 * padding
        else:
            out_h = h + padding[2] + padding[3]
            out_w = w + padding[0] + padding[1]

        input_bytes = batch * channels * h * w * 4
        output_bytes = batch * channels * out_h * out_w * 4
        return input_bytes + output_bytes


class ReflectionPad2dBenchmark(ZeroPad2dBenchmark):
    """Benchmark for nn.ReflectionPad2d layer."""

    name = "ReflectionPad2d"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.ReflectionPad2d(config["padding"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.ReflectionPad2d(config["padding"])


class ReplicationPad2dBenchmark(ZeroPad2dBenchmark):
    """Benchmark for nn.ReplicationPad2d layer."""

    name = "ReplicationPad2d"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.ReplicationPad2d(config["padding"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.ReplicationPad2d(config["padding"])


class ConstantPad2dBenchmark(LayerBenchmark):
    """Benchmark for nn.ConstantPad2d layer."""

    name = "ConstantPad2d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 64, "height": 56, "width": 56, "padding": 1, "value": 0.5},
            {"batch": 32, "channels": 128, "height": 28, "width": 28, "padding": 2, "value": -1.0},
            {"batch": 16, "channels": 256, "height": 14, "width": 14, "padding": 3, "value": 0.0},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.ConstantPad2d(config["padding"], config["value"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.ConstantPad2d(config["padding"], config["value"])

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(
            config["batch"], config["channels"],
            config["height"], config["width"]
        )

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(
            config["batch"], config["channels"],
            config["height"], config["width"],
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
        batch = config["batch"]
        channels = config["channels"]
        h = config["height"]
        w = config["width"]
        padding = config["padding"]

        out_h = h + 2 * padding
        out_w = w + 2 * padding

        input_bytes = batch * channels * h * w * 4
        output_bytes = batch * channels * out_h * out_w * 4
        return input_bytes + output_bytes


class ZeroPad1dBenchmark(LayerBenchmark):
    """Benchmark for nn.ZeroPad1d layer."""

    name = "ZeroPad1d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 64, "length": 256, "padding": 4},
            {"batch": 64, "channels": 128, "length": 512, "padding": 8},
            {"batch": 32, "channels": 256, "length": 1024, "padding": 16},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.ZeroPad1d(config["padding"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.ZeroPad1d(config["padding"])

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(
            config["batch"], config["channels"], config["length"]
        )

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(
            config["batch"], config["channels"], config["length"],
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
        batch = config["batch"]
        channels = config["channels"]
        length = config["length"]
        padding = config["padding"]

        out_length = length + 2 * padding
        input_bytes = batch * channels * length * 4
        output_bytes = batch * channels * out_length * 4
        return input_bytes + output_bytes


PADDING_BENCHMARKS = [
    ZeroPad2dBenchmark,
    ReflectionPad2dBenchmark,
    ReplicationPad2dBenchmark,
    ConstantPad2dBenchmark,
    ZeroPad1dBenchmark,
]
