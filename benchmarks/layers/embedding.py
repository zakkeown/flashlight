"""
Embedding layer benchmarks.
"""

from typing import List, Dict, Any, Tuple

from benchmarks.core.config import BenchmarkLevel
from benchmarks.layers.base import LayerBenchmark


class EmbeddingBenchmark(LayerBenchmark):
    """Benchmark for nn.Embedding layer."""

    name = "Embedding"
    level = BenchmarkLevel.LAYER

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Small vocabulary
            {"batch": 32, "seq_len": 128, "num_embeddings": 1000, "embedding_dim": 256},
            {"batch": 64, "seq_len": 256, "num_embeddings": 10000, "embedding_dim": 512},
            # LLM-style
            {"batch": 32, "seq_len": 512, "num_embeddings": 50000, "embedding_dim": 768},
            {"batch": 16, "seq_len": 1024, "num_embeddings": 32000, "embedding_dim": 1024},
            {"batch": 8, "seq_len": 2048, "num_embeddings": 32000, "embedding_dim": 4096},
            # Large vocabulary
            {"batch": 32, "seq_len": 128, "num_embeddings": 100000, "embedding_dim": 512},
        ]

    def create_mlx_layer(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return nn.Embedding(config["num_embeddings"], config["embedding_dim"])

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        layer = nn.Embedding(config["num_embeddings"], config["embedding_dim"])
        return layer.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        import mlx.core as mx
        return flashlight.tensor(
            mx.random.randint(0, config["num_embeddings"], (config["batch"], config["seq_len"]))
        )

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randint(
            0, config["num_embeddings"],
            (config["batch"], config["seq_len"]),
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
        seq_len = config["seq_len"]
        embedding_dim = config["embedding_dim"]
        # Output: batch * seq_len * embedding_dim floats
        # We also need to read from the embedding table
        output_elements = batch * seq_len * embedding_dim
        return output_elements * 4


EMBEDDING_BENCHMARKS = [
    EmbeddingBenchmark,
]
