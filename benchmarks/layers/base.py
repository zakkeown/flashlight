"""
Base class for layer benchmarks.
"""

from typing import List, Dict, Any, Tuple
from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BaseBenchmark


class LayerBenchmark(BaseBenchmark):
    """
    Base class for layer-level benchmarks.

    Layer benchmarks focus on nn.Module instances including:
    - Module initialization overhead
    - Forward pass with weights
    - (Optionally) Backward pass

    Subclasses should implement:
    - get_input_configs(): Return configurations to benchmark
    - create_mlx_layer(config): Create flashlight layer
    - create_pytorch_layer(config, device): Create PyTorch layer
    - create_mlx_input(config): Create input tensor
    - create_pytorch_input(config, device): Create input tensor
    """

    level = BenchmarkLevel.LAYER

    def create_mlx_layer(self, config: Dict[str, Any]):
        """Create flashlight layer."""
        raise NotImplementedError

    def create_pytorch_layer(self, config: Dict[str, Any], device: str):
        """Create PyTorch layer."""
        raise NotImplementedError

    def create_mlx_input(self, config: Dict[str, Any]):
        """Create flashlight input tensor."""
        raise NotImplementedError

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        """Create PyTorch input tensor."""
        raise NotImplementedError

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        """Create layer and input for MLX benchmark."""
        layer = self.create_mlx_layer(config)
        x = self.create_mlx_input(config)
        return (layer, x)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        """Create layer and input for PyTorch benchmark."""
        layer = self.create_pytorch_layer(config, device)
        x = self.create_pytorch_input(config, device)
        return (layer, x)

    def mlx_operation(self, layer, x):
        """Forward pass through MLX layer."""
        return layer(x)

    def pytorch_operation(self, layer, x):
        """Forward pass through PyTorch layer."""
        return layer(x)

    def format_config(self, config: Dict[str, Any]) -> str:
        """Format configuration as readable string."""
        parts = []
        for key, value in config.items():
            if key != "batch":
                parts.append(f"{key}={value}")
        return ", ".join(parts)
