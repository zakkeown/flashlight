"""
Activation layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class ReLUBenchmark(LayerBenchmark):
    """Benchmark for nn.ReLU layer."""

    name = "ReLU"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # 2D inputs
            {"shape": (32, 512)},
            {"shape": (64, 1024)},
            {"shape": (128, 4096)},
            # 3D inputs (transformer-style)
            {"shape": (32, 128, 768)},
            {"shape": (64, 256, 1024)},
            # 4D inputs (CNN-style)
            {"shape": (32, 64, 56, 56)},
            {"shape": (64, 256, 28, 28)},
            {"shape": (32, 512, 14, 14)},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.ReLU()

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.ReLU()

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(*config["shape"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(*config["shape"], device=device)

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
        import numpy as np
        elements = int(np.prod(config["shape"]))
        return elements * 4 * 2  # input + output


class GELUBenchmark(ReLUBenchmark):
    """Benchmark for nn.GELU layer."""

    name = "GELU"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.GELU()

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.GELU()


class SiLUBenchmark(ReLUBenchmark):
    """Benchmark for nn.SiLU layer."""

    name = "SiLU"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.SiLU()

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.SiLU()


class SoftmaxBenchmark(LayerBenchmark):
    """Benchmark for nn.Softmax layer."""

    name = "Softmax"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Classification heads
            {"shape": (32, 1000)},
            {"shape": (64, 10000)},
            {"shape": (128, 50000)},
            # Attention weights
            {"shape": (32, 12, 128, 128)},
            {"shape": (16, 16, 512, 512)},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.Softmax(dim=-1)

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.Softmax(dim=-1)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(*config["shape"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(*config["shape"], device=device)

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
        import numpy as np
        elements = int(np.prod(config["shape"]))
        return elements * 4 * 2


ACTIVATION_BENCHMARKS = [
    ReLUBenchmark,
    GELUBenchmark,
    SiLUBenchmark,
    SoftmaxBenchmark,
]
