"""
Linear layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class LinearLayerBenchmark(LayerBenchmark):
    """Benchmark for nn.Linear layer."""

    name = "Linear"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Small layers
            {"batch": 32, "in_features": 64, "out_features": 64},
            {"batch": 64, "in_features": 128, "out_features": 128},
            # Medium layers
            {"batch": 32, "in_features": 256, "out_features": 256},
            {"batch": 64, "in_features": 512, "out_features": 512},
            # Large layers (LLM-style)
            {"batch": 32, "in_features": 1024, "out_features": 1024},
            {"batch": 32, "in_features": 4096, "out_features": 4096},
            # Asymmetric
            {"batch": 32, "in_features": 768, "out_features": 3072},  # FFN up
            {"batch": 32, "in_features": 3072, "out_features": 768},  # FFN down
            # Large batch
            {"batch": 128, "in_features": 512, "out_features": 512},
            {"batch": 256, "in_features": 256, "out_features": 256},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.Linear(config["in_features"], config["out_features"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.Linear(config["in_features"], config["out_features"])
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["in_features"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["in_features"], device=device)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate GFLOPS for linear layer."""
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        batch = config["batch"]
        in_f = config["in_features"]
        out_f = config["out_features"]

        # FLOPs = 2 * batch * in * out (multiply-add for each output element)
        flops = 2 * batch * in_f * out_f
        gflops = flops / (time_ms / 1000.0) / 1e9

        return gflops, "GFLOPS"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        batch = config["batch"]
        in_f = config["in_features"]
        out_f = config["out_features"]

        # Input: batch * in_features
        # Weights: in_features * out_features
        # Output: batch * out_features
        elements = batch * in_f + in_f * out_f + batch * out_f
        return elements * 4  # float32


class LinearWithBiasBenchmark(LinearLayerBenchmark):
    """Benchmark for Linear layer with bias disabled."""

    name = "Linear_no_bias"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.Linear(config["in_features"], config["out_features"], bias=False)

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.Linear(config["in_features"], config["out_features"], bias=False)
        return layer.to(device)


# List of all linear benchmarks
LINEAR_BENCHMARKS = [
    LinearLayerBenchmark,
    LinearWithBiasBenchmark,
]
