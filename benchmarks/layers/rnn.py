"""
RNN layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class LSTMBenchmark(LayerBenchmark):
    """Benchmark for nn.LSTM layer."""

    name = "LSTM"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Small LSTM
            {"batch": 32, "seq_len": 64, "input_size": 128, "hidden_size": 256},
            {"batch": 64, "seq_len": 128, "input_size": 256, "hidden_size": 512},
            # Medium LSTM
            {"batch": 32, "seq_len": 256, "input_size": 512, "hidden_size": 512},
            {"batch": 16, "seq_len": 512, "input_size": 256, "hidden_size": 256},
            # Large LSTM
            {"batch": 16, "seq_len": 256, "input_size": 1024, "hidden_size": 1024},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.LSTM(config["input_size"], config["hidden_size"], batch_first=True)

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.LSTM(config["input_size"], config["hidden_size"], batch_first=True)
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["seq_len"], config["input_size"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(
            config["batch"], config["seq_len"], config["input_size"],
            device=device
        )

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        # LSTM has 4 gates, each with input-hidden and hidden-hidden matrix multiplies
        batch = config["batch"]
        seq_len = config["seq_len"]
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]

        # Each timestep: 4 * (batch * input_size * hidden_size + batch * hidden_size * hidden_size)
        # Multiply by 2 for multiply-add
        flops_per_step = 2 * 4 * batch * (input_size * hidden_size + hidden_size * hidden_size)
        total_flops = flops_per_step * seq_len

        gflops = total_flops / (time_ms / 1000.0) / 1e9
        return gflops, "GFLOPS"

    def calculate_bytes(self, config: Dict[str, Any]) -> int:
        batch = config["batch"]
        seq_len = config["seq_len"]
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]

        input_bytes = batch * seq_len * input_size * 4
        output_bytes = batch * seq_len * hidden_size * 4
        # Weights: 4 gates * (input_size * hidden_size + hidden_size * hidden_size)
        weight_bytes = 4 * (input_size * hidden_size + hidden_size * hidden_size) * 4

        return input_bytes + output_bytes + weight_bytes


class GRUBenchmark(LSTMBenchmark):
    """Benchmark for nn.GRU layer."""

    name = "GRU"

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.GRU(config["input_size"], config["hidden_size"], batch_first=True)

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.GRU(config["input_size"], config["hidden_size"], batch_first=True)
        return layer.to(device)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "GFLOPS"

        # GRU has 3 gates instead of 4
        batch = config["batch"]
        seq_len = config["seq_len"]
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]

        flops_per_step = 2 * 3 * batch * (input_size * hidden_size + hidden_size * hidden_size)
        total_flops = flops_per_step * seq_len

        gflops = total_flops / (time_ms / 1000.0) / 1e9
        return gflops, "GFLOPS"


RNN_BENCHMARKS = [
    LSTMBenchmark,
    GRUBenchmark,
]
