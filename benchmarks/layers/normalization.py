"""
Normalization layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class BatchNorm2dBenchmark(LayerBenchmark):
    """Benchmark for nn.BatchNorm2d layer."""

    name = "BatchNorm2d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 64, "h": 56, "w": 56},
            {"batch": 32, "channels": 128, "h": 28, "w": 28},
            {"batch": 32, "channels": 256, "h": 14, "w": 14},
            {"batch": 32, "channels": 512, "h": 7, "w": 7},
            {"batch": 64, "channels": 64, "h": 224, "w": 224},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        layer = nn.BatchNorm2d(config["channels"])
        layer.eval()  # Use eval mode for consistent benchmarking
        return layer

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.BatchNorm2d(config["channels"])
        layer.eval()
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["channels"], config["h"], config["w"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["channels"], config["h"], config["w"], device=device)

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


class BatchNorm1dBenchmark(LayerBenchmark):
    """Benchmark for nn.BatchNorm1d layer."""

    name = "BatchNorm1d"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 64, "features": 256},
            {"batch": 64, "features": 512},
            {"batch": 128, "features": 1024},
            {"batch": 256, "features": 2048},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        layer = nn.BatchNorm1d(config["features"])
        layer.eval()
        return layer

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.BatchNorm1d(config["features"])
        layer.eval()
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["features"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["features"], device=device)


class LayerNormBenchmark(LayerBenchmark):
    """Benchmark for nn.LayerNorm layer."""

    name = "LayerNorm"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Transformer-style
            {"batch": 32, "seq_len": 128, "hidden": 768},
            {"batch": 32, "seq_len": 512, "hidden": 768},
            {"batch": 16, "seq_len": 512, "hidden": 1024},
            {"batch": 8, "seq_len": 2048, "hidden": 768},
            # Large models
            {"batch": 16, "seq_len": 128, "hidden": 4096},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.LayerNorm(config["hidden"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.LayerNorm(config["hidden"]).to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["seq_len"], config["hidden"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["seq_len"], config["hidden"], device=device)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "elem/sec"

        elements = config["batch"] * config["seq_len"] * config["hidden"]
        elements_per_sec = elements / (time_ms / 1000.0)

        if elements_per_sec >= 1e9:
            return elements_per_sec / 1e9, "Gelem/sec"
        elif elements_per_sec >= 1e6:
            return elements_per_sec / 1e6, "Melem/sec"
        return elements_per_sec, "elem/sec"


class GroupNormBenchmark(LayerBenchmark):
    """Benchmark for nn.GroupNorm layer."""

    name = "GroupNorm"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "groups": 32, "channels": 256, "h": 28, "w": 28},
            {"batch": 32, "groups": 32, "channels": 512, "h": 14, "w": 14},
            {"batch": 16, "groups": 32, "channels": 1024, "h": 7, "w": 7},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.GroupNorm(config["groups"], config["channels"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.GroupNorm(config["groups"], config["channels"]).to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["channels"], config["h"], config["w"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["channels"], config["h"], config["w"], device=device)


# List of all normalization benchmarks
NORMALIZATION_BENCHMARKS = [
    BatchNorm2dBenchmark,
    BatchNorm1dBenchmark,
    LayerNormBenchmark,
    GroupNormBenchmark,
]
