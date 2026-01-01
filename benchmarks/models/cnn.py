"""
CNN model benchmarks.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from benchmarks.core.config import BenchmarkLevel
from benchmarks.models.base import ModelBenchmark


class CNNForwardBenchmark(ModelBenchmark):
    """Benchmark for CNN forward pass."""

    name = "CNN_forward"
    level = BenchmarkLevel.MODEL
    mode = "forward"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # CIFAR-10 style
            {"batch": 32, "channels": 3, "height": 32, "width": 32, "num_classes": 10},
            {"batch": 64, "channels": 3, "height": 32, "width": 32, "num_classes": 10},
            # ImageNet style (smaller)
            {"batch": 16, "channels": 3, "height": 224, "width": 224, "num_classes": 1000},
            {"batch": 32, "channels": 3, "height": 224, "width": 224, "num_classes": 1000},
        ]

    def _create_simple_cnn(self, nn_module, config: Dict[str, Any]):
        """Create a simple CNN architecture."""
        layers = []

        # Conv block 1
        layers.extend([
            nn_module.Conv2d(config["channels"], 32, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.Conv2d(32, 64, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.MaxPool2d(2, 2),
        ])

        # Conv block 2
        layers.extend([
            nn_module.Conv2d(64, 128, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.Conv2d(128, 128, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.MaxPool2d(2, 2),
        ])

        # Calculate feature map size after pooling
        h = config["height"] // 4
        w = config["width"] // 4
        flat_features = 128 * h * w

        # Classifier
        layers.extend([
            nn_module.Flatten(),
            nn_module.Linear(flat_features, 256),
            nn_module.ReLU(),
            nn_module.Linear(256, config["num_classes"]),
        ])

        return nn_module.Sequential(*layers)

    def create_mlx_model(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return self._create_simple_cnn(nn, config)

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        model = self._create_simple_cnn(nn, config)
        return model.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["channels"], config["height"], config["width"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["channels"], config["height"], config["width"], device=device)


class CNNTrainingBenchmark(ModelBenchmark):
    """Benchmark for CNN training step."""

    name = "CNN_training"
    level = BenchmarkLevel.MODEL
    mode = "training"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 3, "height": 32, "width": 32, "num_classes": 10},
            {"batch": 64, "channels": 3, "height": 32, "width": 32, "num_classes": 10},
        ]

    def _create_simple_cnn(self, nn_module, config: Dict[str, Any]):
        """Create a simple CNN architecture."""
        layers = []

        layers.extend([
            nn_module.Conv2d(config["channels"], 32, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.Conv2d(32, 64, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.MaxPool2d(2, 2),
        ])

        layers.extend([
            nn_module.Conv2d(64, 128, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.Conv2d(128, 128, kernel_size=3, padding=1),
            nn_module.ReLU(),
            nn_module.MaxPool2d(2, 2),
        ])

        h = config["height"] // 4
        w = config["width"] // 4
        flat_features = 128 * h * w

        layers.extend([
            nn_module.Flatten(),
            nn_module.Linear(flat_features, 256),
            nn_module.ReLU(),
            nn_module.Linear(256, config["num_classes"]),
        ])

        return nn_module.Sequential(*layers)

    def create_mlx_model(self, config: Dict[str, Any]):
        import flashlight.nn as nn
        return self._create_simple_cnn(nn, config)

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn
        model = self._create_simple_cnn(nn, config)
        return model.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import flashlight
        return flashlight.randn(config["batch"], config["channels"], config["height"], config["width"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["channels"], config["height"], config["width"], device=device)

    def create_mlx_target(self, config: Dict[str, Any]):
        import flashlight
        labels = np.random.randint(0, config["num_classes"], config["batch"])
        return flashlight.tensor(labels)

    def create_pytorch_target(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randint(0, config["num_classes"], (config["batch"],), device=device)


# List of all CNN benchmarks
CNN_BENCHMARKS = [
    CNNForwardBenchmark,
    CNNTrainingBenchmark,
]
