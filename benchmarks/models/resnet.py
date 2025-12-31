"""
ResNet model benchmarks.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from benchmarks.core.config import BenchmarkLevel
from benchmarks.models.base import ModelBenchmark


def _create_basic_block_mlx(nn, in_channels, out_channels, stride=1):
    """Create a basic residual block for mlx_compat."""
    class BasicBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = self.relu(out)
            return out

    return BasicBlock()


def _create_basic_block_torch(nn, in_channels, out_channels, stride=1):
    """Create a basic residual block for PyTorch."""
    class BasicBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = self.relu(out)
            return out

    return BasicBlock()


class ResNetForwardBenchmark(ModelBenchmark):
    """Benchmark for ResNet forward pass."""

    name = "ResNet_forward"
    level = BenchmarkLevel.MODEL
    mode = "forward"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            # Small ResNet for CIFAR-10
            {"batch": 32, "channels": 3, "height": 32, "width": 32, "num_classes": 10, "num_blocks": [2, 2, 2, 2]},
            {"batch": 64, "channels": 3, "height": 32, "width": 32, "num_classes": 10, "num_blocks": [2, 2, 2, 2]},
            # ResNet-18 style for ImageNet
            {"batch": 8, "channels": 3, "height": 224, "width": 224, "num_classes": 1000, "num_blocks": [2, 2, 2, 2]},
            {"batch": 16, "channels": 3, "height": 224, "width": 224, "num_classes": 1000, "num_blocks": [2, 2, 2, 2]},
        ]

    def create_mlx_model(self, config: Dict[str, Any]):
        import mlx_compat.nn as nn

        class SmallResNet(nn.Module):
            def __init__(self, num_classes, num_blocks):
                super().__init__()
                self.in_channels = 64

                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()

                self.layer1 = self._make_layer(nn, 64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(nn, 128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(nn, 256, num_blocks[2], stride=2)
                self.layer4 = self._make_layer(nn, 512, num_blocks[3], stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)

            def _make_layer(self, nn, out_channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for s in strides:
                    layers.append(_create_basic_block_mlx(nn, self.in_channels, out_channels, s))
                    self.in_channels = out_channels
                return nn.Sequential(*layers)

            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out

        return SmallResNet(config["num_classes"], config["num_blocks"])

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        import torch.nn as nn

        class SmallResNet(nn.Module):
            def __init__(self, num_classes, num_blocks):
                super().__init__()
                self.in_channels = 64

                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()

                self.layer1 = self._make_layer(nn, 64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(nn, 128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(nn, 256, num_blocks[2], stride=2)
                self.layer4 = self._make_layer(nn, 512, num_blocks[3], stride=2)

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)

            def _make_layer(self, nn, out_channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for s in strides:
                    layers.append(_create_basic_block_torch(nn, self.in_channels, out_channels, s))
                    self.in_channels = out_channels
                return nn.Sequential(*layers)

            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out

        model = SmallResNet(config["num_classes"], config["num_blocks"])
        return model.to(device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["channels"], config["height"], config["width"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["channels"], config["height"], config["width"], device=device)


class ResNetTrainingBenchmark(ModelBenchmark):
    """Benchmark for ResNet training step."""

    name = "ResNet_training"
    level = BenchmarkLevel.MODEL
    mode = "training"

    def get_input_configs(self) -> List[Dict[str, Any]]:
        return [
            {"batch": 32, "channels": 3, "height": 32, "width": 32, "num_classes": 10, "num_blocks": [2, 2, 2, 2]},
            {"batch": 16, "channels": 3, "height": 32, "width": 32, "num_classes": 10, "num_blocks": [2, 2, 2, 2]},
        ]

    def create_mlx_model(self, config: Dict[str, Any]):
        # Reuse the same model creation from forward benchmark
        forward_bench = ResNetForwardBenchmark(self.config)
        return forward_bench.create_mlx_model(config)

    def create_pytorch_model(self, config: Dict[str, Any], device: str):
        forward_bench = ResNetForwardBenchmark(self.config)
        return forward_bench.create_pytorch_model(config, device)

    def create_mlx_input(self, config: Dict[str, Any]):
        import mlx_compat
        return mlx_compat.randn(config["batch"], config["channels"], config["height"], config["width"])

    def create_pytorch_input(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randn(config["batch"], config["channels"], config["height"], config["width"], device=device)

    def create_mlx_target(self, config: Dict[str, Any]):
        import mlx_compat
        labels = np.random.randint(0, config["num_classes"], config["batch"])
        return mlx_compat.tensor(labels)

    def create_pytorch_target(self, config: Dict[str, Any], device: str):
        import torch
        return torch.randint(0, config["num_classes"], (config["batch"],), device=device)


# List of all ResNet benchmarks
RESNET_BENCHMARKS = [
    ResNetForwardBenchmark,
    ResNetTrainingBenchmark,
]
