"""
Loss function benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class CrossEntropyLossBenchmark(LayerBenchmark):
    """Benchmark for nn.CrossEntropyLoss."""

    name = "CrossEntropyLoss"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # ImageNet-style
            {"batch": 32, "num_classes": 1000},
            {"batch": 64, "num_classes": 1000},
            {"batch": 128, "num_classes": 1000},
            # LLM-style (vocabulary)
            {"batch": 32, "num_classes": 32000},
            {"batch": 64, "num_classes": 50000},
            # Small classification
            {"batch": 128, "num_classes": 10},
            {"batch": 256, "num_classes": 100},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.CrossEntropyLoss()

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.CrossEntropyLoss()

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        import mlx.core as mx
        logits = flashlight.randn(config["batch"], config["num_classes"])
        targets = flashlight.tensor(
            mx.random.randint(0, config["num_classes"], (config["batch"],))
        )
        return (logits, targets)

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        logits = torch.randn(config["batch"], config["num_classes"], device=device)
        targets = torch.randint(0, config["num_classes"], (config["batch"],), device=device)
        return (logits, targets)

    def run_mlx(self, layer, inputs):
        import mlx.core as mx
        logits, targets = inputs
        result = layer(logits, targets)
        mx.eval(result._mlx_array if hasattr(result, '_mlx_array') else result)
        return result

    def run_pytorch(self, layer, inputs):
        logits, targets = inputs
        result = layer(logits, targets)
        return result

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
        num_classes = config["num_classes"]
        # logits + targets (int64) + output
        return batch * num_classes * 4 + batch * 8 + 4


class MSELossBenchmark(LayerBenchmark):
    """Benchmark for nn.MSELoss."""

    name = "MSELoss"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "features": 1024},
            {"batch": 64, "features": 4096},
            {"batch": 128, "features": 8192},
            {"batch": 256, "features": 1024},
            # Regression with many features
            {"batch": 32, "features": 65536},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.MSELoss()

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        return nn.MSELoss()

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        pred = flashlight.randn(config["batch"], config["features"])
        target = flashlight.randn(config["batch"], config["features"])
        return (pred, target)

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        pred = torch.randn(config["batch"], config["features"], device=device)
        target = torch.randn(config["batch"], config["features"], device=device)
        return (pred, target)

    def run_mlx(self, layer, inputs):
        import mlx.core as mx
        pred, target = inputs
        result = layer(pred, target)
        mx.eval(result._mlx_array if hasattr(result, '_mlx_array') else result)
        return result

    def run_pytorch(self, layer, inputs):
        pred, target = inputs
        result = layer(pred, target)
        return result

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
        features = config["features"]
        # pred + target + output
        return batch * features * 4 * 2 + 4


LOSS_BENCHMARKS = [
    CrossEntropyLossBenchmark,
    MSELossBenchmark,
]
