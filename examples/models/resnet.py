"""
ResNet Architecture

Implements ResNet models for image classification.
"""

import sys
sys.path.insert(0, '../..')

import flashlight
import flashlight.nn as nn


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions.

    Used in ResNet-18 and ResNet-34.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for first convolution (default: 1)
        downsample: Optional downsample layer for skip connection
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # First conv block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture.

    Args:
        block: Block type (BasicBlock or Bottleneck)
        layers: List of number of blocks in each layer
        num_classes: Number of output classes (default: 1000)
    """

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with multiple blocks."""
        downsample = None

        # Create downsample layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        # Classifier
        x = self.fc(x)

        return x


def resnet18(num_classes=1000):
    """
    ResNet-18 model.

    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)

    Returns:
        ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    """
    ResNet-34 model.

    Args:
        num_classes: Number of output classes (default: 1000)

    Returns:
        ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


class SmallResNet(nn.Module):
    """
    Smaller ResNet for CIFAR-10 (32x32 images).

    Adapted ResNet-18 for smaller images without initial downsampling.

    Args:
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 16

        # Initial convolution (no stride-2 for small images)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with multiple blocks."""
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        # Classifier
        x = self.fc(x)

        return x


__all__ = ['BasicBlock', 'ResNet', 'resnet18', 'resnet34', 'SmallResNet']
