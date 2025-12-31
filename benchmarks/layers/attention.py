"""
Attention layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class MultiHeadAttentionBenchmark(LayerBenchmark):
    """Benchmark for nn.MultiheadAttention layer."""

    name = "MultiHeadAttention"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Standard transformer configs
            {"batch": 32, "seq_len": 128, "embed_dim": 512, "num_heads": 8},
            {"batch": 32, "seq_len": 256, "embed_dim": 512, "num_heads": 8},
            {"batch": 16, "seq_len": 512, "embed_dim": 768, "num_heads": 12},
            # Larger models
            {"batch": 8, "seq_len": 512, "embed_dim": 1024, "num_heads": 16},
            {"batch": 4, "seq_len": 1024, "embed_dim": 768, "num_heads": 12},
            # Vision transformer style
            {"batch": 32, "seq_len": 197, "embed_dim": 768, "num_heads": 12},  # ViT-B
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            batch_first=True,
        )

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            batch_first=True,
        )
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        x = mlx_compat.randn(config["batch"], config["seq_len"], config["embed_dim"])
        return x

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        x = torch.randn(config["batch"], config["seq_len"], config["embed_dim"], device=device)
        return x

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        layer = self.create_mlx_layer(config)
        x = self.create_mlx_input(config)
        return (layer, x, x, x)  # query, key, value

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        layer = self.create_pytorch_layer(config, device)
        x = self.create_pytorch_input(config, device)
        return (layer, x, x, x)

    def mlx_operation(self, layer, query, key, value):
        output, _ = layer(query, key, value)
        return output

    def pytorch_operation(self, layer, query, key, value):
        output, _ = layer(query, key, value)
        return output

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Estimate throughput in tokens/second."""
        if time_ms <= 0:
            return 0.0, "tokens/sec"

        tokens = config["batch"] * config["seq_len"]
        tokens_per_sec = tokens / (time_ms / 1000.0)

        if tokens_per_sec >= 1e6:
            return tokens_per_sec / 1e6, "Mtokens/sec"
        elif tokens_per_sec >= 1e3:
            return tokens_per_sec / 1e3, "Ktokens/sec"
        return tokens_per_sec, "tokens/sec"


class SelfAttentionBenchmark(LayerBenchmark):
    """Benchmark for self-attention (Q=K=V)."""

    name = "SelfAttention"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "seq_len": 128, "embed_dim": 512, "num_heads": 8},
            {"batch": 16, "seq_len": 512, "embed_dim": 768, "num_heads": 12},
            {"batch": 8, "seq_len": 1024, "embed_dim": 768, "num_heads": 12},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn
        return nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            batch_first=True,
        )

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            batch_first=True,
        )
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["seq_len"], config["embed_dim"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["seq_len"], config["embed_dim"], device=device)

    def mlx_operation(self, layer, x):
        output, _ = layer(x, x, x)
        return output

    def pytorch_operation(self, layer, x):
        output, _ = layer(x, x, x)
        return output


# List of all attention benchmarks
ATTENTION_BENCHMARKS = [
    MultiHeadAttentionBenchmark,
    SelfAttentionBenchmark,
]
