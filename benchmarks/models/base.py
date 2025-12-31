"""
Base class for model benchmarks.
"""

from typing import List, Dict, Any, Tuple
from benchmarks.core.config import BenchmarkConfig, BenchmarkLevel
from benchmarks.core.runner import BaseBenchmark


class ModelBenchmark(BaseBenchmark):
    """
    Base class for model-level benchmarks.

    Model benchmarks test complete neural network architectures
    including all layers, activations, and (optionally) loss
    and optimizer steps.

    Modes:
    - forward: Forward pass only (inference)
    - training: Forward + backward + optimizer step

    Subclasses should implement:
    - get_input_configs(): Return configurations to benchmark
    - create_mlx_model(config): Create mlx_compat model
    - create_pytorch_model(config, device): Create PyTorch model
    - create_mlx_input(config): Create input tensor
    - create_pytorch_input(config, device): Create input tensor
    """

    level = BenchmarkLevel.MODEL
    mode: str = "forward"  # "forward" or "training"

    def create_mlx_model(self, config: Dict[str, Any]):
        """Create mlx_compat model."""
        raise NotImplementedError

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        """Create PyTorch model."""
        raise NotImplementedError

    def create_mlx_input(self, config: Dict[str, Any]):
        """Create mlx_compat input tensor."""
        raise NotImplementedError

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        """Create PyTorch input tensor."""
        raise NotImplementedError

    def create_mlx_target(self, config: Dict[str, Any]):
        """Create mlx_compat target for training."""
        return None

    def create_pytorch_target(self, config: Dict[str, Any], device: str):
        """Create PyTorch target for training."""
        return None

    def create_mlx_inputs(self, config: Dict[str, Any]) -> Tuple:
        """Create model and inputs for MLX benchmark."""
        model = self.create_mlx_model(config)

        if self.mode == "forward":
            model.eval()
        else:
            model.train()

        x = self.create_mlx_input(config)
        target = self.create_mlx_target(config) if self.mode == "training" else None

        return (model, x, target, config)

    def create_pytorch_inputs(self, config: Dict[str, Any], device: str) -> Tuple:
        """Create model and inputs for PyTorch benchmark."""
        model = self.create_pytorch_model(config, device)

        if self.mode == "forward":
            model.eval()
        else:
            model.train()

        x = self.create_pytorch_input(config, device)
        target = self.create_pytorch_target(config, device) if self.mode == "training" else None

        return (model, x, target, config)

    def mlx_operation(self, model, x, target, config):
        """Forward/training pass through MLX model."""
        if self.mode == "forward":
            return model(x)
        else:
            # Training step
            import mlx_compat
            import mlx_compat.nn as nn
            import mlx_compat.optim as optim

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            return output

    def pytorch_operation(self, model, x, target, config):
        """Forward/training pass through PyTorch model."""
        if self.mode == "forward":
            return model(x)
        else:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            return output

    def calculate_throughput(
        self,
        config: Dict[str, Any],
        time_ms: float,
    ) -> Tuple[float, str]:
        """Calculate samples/second."""
        if time_ms <= 0:
            return 0.0, "samples/sec"

        batch = config.get("batch", 1)
        samples_per_sec = batch / (time_ms / 1000.0)

        if samples_per_sec >= 1e6:
            return samples_per_sec / 1e6, "Msamples/sec"
        elif samples_per_sec >= 1e3:
            return samples_per_sec / 1e3, "Ksamples/sec"
        return samples_per_sec, "samples/sec"
