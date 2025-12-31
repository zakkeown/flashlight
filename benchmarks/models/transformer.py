"""
Transformer model benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.models.base import ModelBenchmark


class TransformerEncoderBenchmark(ModelBenchmark):
    """Benchmark for Transformer encoder layer forward pass."""

    name = "TransformerEncoder_forward"
    level = BenchmarkLevel.MODEL
    mode = "forward"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # BERT-base style
            {"batch": 32, "seq_len": 128, "d_model": 768, "nhead": 12, "dim_feedforward": 3072},
            {"batch": 16, "seq_len": 512, "d_model": 768, "nhead": 12, "dim_feedforward": 3072},
            # Smaller transformer
            {"batch": 32, "seq_len": 128, "d_model": 256, "nhead": 8, "dim_feedforward": 1024},
            {"batch": 64, "seq_len": 64, "d_model": 512, "nhead": 8, "dim_feedforward": 2048},
            # Longer sequences
            {"batch": 8, "seq_len": 1024, "d_model": 512, "nhead": 8, "dim_feedforward": 2048},
        ]

    def create_mlx_model(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            batch_first=True,
        )

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            batch_first=True,
        )
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["seq_len"], config["d_model"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["seq_len"], config["d_model"], device=device)

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        model = self.create_mlx_model(config)
        model.eval()
        x = self.create_mlx_input(config)
        return (model, x, None, config)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        model = self.create_pytorch_model(config, device)
        model.eval()
        x = self.create_pytorch_input(config, device)
        return (model, x, None, config)

    def mlx_operation(self, model, x, target, config):
        return model(x)

    def pytorch_operation(self, model, x, target, config):
        return model(x)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate tokens/second throughput."""
        if time_ms <= 0:
            return 0.0, "tokens/sec"

        tokens = config["batch"] * config["seq_len"]
        tokens_per_sec = tokens / (time_ms / 1000.0)

        if tokens_per_sec >= 1e6:
            return tokens_per_sec / 1e6, "Mtokens/sec"
        elif tokens_per_sec >= 1e3:
            return tokens_per_sec / 1e3, "Ktokens/sec"
        return tokens_per_sec, "tokens/sec"


class TransformerStackBenchmark(ModelBenchmark):
    """Benchmark for stacked Transformer encoder layers."""

    name = "TransformerStack_forward"
    level = BenchmarkLevel.MODEL
    mode = "forward"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # 6-layer encoder (like BERT-base encoder)
            {"batch": 16, "seq_len": 128, "d_model": 512, "nhead": 8, "dim_feedforward": 2048, "num_layers": 6},
            {"batch": 8, "seq_len": 256, "d_model": 768, "nhead": 12, "dim_feedforward": 3072, "num_layers": 6},
            # 12-layer encoder
            {"batch": 8, "seq_len": 128, "d_model": 768, "nhead": 12, "dim_feedforward": 3072, "num_layers": 12},
        ]

    def create_mlx_model(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            batch_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            batch_first=True,
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        return model.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["seq_len"], config["d_model"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["seq_len"], config["d_model"], device=device)

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        model = self.create_mlx_model(config)
        model.eval()
        x = self.create_mlx_input(config)
        return (model, x, None, config)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        model = self.create_pytorch_model(config, device)
        model.eval()
        x = self.create_pytorch_input(config, device)
        return (model, x, None, config)

    def mlx_operation(self, model, x, target, config):
        return model(x)

    def pytorch_operation(self, model, x, target, config):
        return model(x)

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        if time_ms <= 0:
            return 0.0, "tokens/sec"

        tokens = config["batch"] * config["seq_len"]
        tokens_per_sec = tokens / (time_ms / 1000.0)

        if tokens_per_sec >= 1e6:
            return tokens_per_sec / 1e6, "Mtokens/sec"
        elif tokens_per_sec >= 1e3:
            return tokens_per_sec / 1e3, "Ktokens/sec"
        return tokens_per_sec, "tokens/sec"


# List of all Transformer benchmarks
TRANSFORMER_BENCHMARKS = [
    TransformerEncoderBenchmark,
    TransformerStackBenchmark,
]
