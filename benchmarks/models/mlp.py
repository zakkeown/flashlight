"""
MLP model benchmarks.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from benchmarks.core.config import BenchmarkLevel
from benchmarks.models.base import ModelBenchmark


class MLPForwardBenchmark(ModelBenchmark):
    """Benchmark for MLP forward pass."""

    name = "MLP_forward"
    level = BenchmarkLevel.MODEL
    mode = "forward"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # MNIST-style
            {"batch": 32, "input_size": 784, "hidden_sizes": [256, 128], "num_classes": 10},
            {"batch": 64, "input_size": 784, "hidden_sizes": [256, 128], "num_classes": 10},
            {"batch": 128, "input_size": 784, "hidden_sizes": [256, 128], "num_classes": 10},
            # Larger MLPs
            {"batch": 32, "input_size": 1024, "hidden_sizes": [512, 256, 128], "num_classes": 100},
            {"batch": 64, "input_size": 2048, "hidden_sizes": [1024, 512, 256], "num_classes": 100},
        ]

    def create_mlx_model(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn

        layers = []
        in_features = config["input_size"]

        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, config["num_classes"]))

        return nn.Sequential(*layers)

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn

        layers = []
        in_features = config["input_size"]

        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, config["num_classes"]))

        model = nn.Sequential(*layers)
        return model.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["input_size"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["input_size"], device=device)


class MLPTrainingBenchmark(ModelBenchmark):
    """Benchmark for MLP training step."""

    name = "MLP_training"
    level = BenchmarkLevel.MODEL
    mode = "training"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "input_size": 784, "hidden_sizes": [256, 128], "num_classes": 10},
            {"batch": 64, "input_size": 784, "hidden_sizes": [256, 128], "num_classes": 10},
            {"batch": 128, "input_size": 784, "hidden_sizes": [256, 128], "num_classes": 10},
        ]

    def create_mlx_model(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn

        layers = []
        in_features = config["input_size"]

        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, config["num_classes"]))

        return nn.Sequential(*layers)

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn

        layers = []
        in_features = config["input_size"]

        for hidden_size in config["hidden_sizes"]:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, config["num_classes"]))

        model = nn.Sequential(*layers)
        return model.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["input_size"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["input_size"], device=device)

    def create_mlx_target(self, config: Dict[str, Any]):
        import mlx_compat
        labels = np.random.randint(0, config["num_classes"], config["batch"])
        return mlx_compat.tensor(labels)

    def create_pytorch_target(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randint(0, config["num_classes"], (config["batch"],), device=device)


# List of all MLP benchmarks
MLP_BENCHMARKS = [
    MLPForwardBenchmark,
    MLPTrainingBenchmark,
]
