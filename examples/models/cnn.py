"""
Convolutional Neural Networks

Implements CNN architectures for image classification.
"""

import sys
sys.path.insert(0, '../..')

import mlx_compat
import mlx_compat.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10.

    A basic convolutional neural network with 2 conv layers followed by
    2 fully connected layers.

    Architecture:
        Conv2d(3, 32, 3) -> ReLU -> MaxPool2d(2) ->
        Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(64*6*6, 128) -> ReLU -> Dropout ->
        Linear(128, num_classes)

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # After 2 pools: 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [N, 3, 32, 32]

        Returns:
            Output logits of shape [N, num_classes]
        """
        # Conv block 1: 3x32x32 -> 32x32x32 -> 32x16x16
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 2: 32x16x16 -> 64x16x16 -> 64x8x8
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten: 64x8x8 -> 4096
        x = x.reshape(x.shape[0], -1)

        # FC block: 4096 -> 128 -> 10
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class LeNet5(nn.Module):
    """
    LeNet-5 architecture adapted for CIFAR-10.

    Classic convolutional neural network introduced by Yann LeCun.
    Adapted for 32x32 RGB images (CIFAR-10).

    Architecture:
        Conv2d(3, 6, 5) -> ReLU -> MaxPool2d(2) ->
        Conv2d(6, 16, 5) -> ReLU -> MaxPool2d(2) ->
        Flatten -> Linear(16*5*5, 120) -> ReLU ->
        Linear(120, 84) -> ReLU ->
        Linear(84, num_classes)

    Args:
        num_classes: Number of output classes (default: 10)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        # Input: 32x32, after conv1 (k=5): 28x28, after pool: 14x14
        # After conv2 (k=5): 10x10, after pool: 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [N, 3, 32, 32]

        Returns:
            Output logits of shape [N, num_classes]
        """
        # Feature extraction
        x = self.pool(self.relu(self.conv1(x)))  # 3x32x32 -> 6x14x14
        x = self.pool(self.relu(self.conv2(x)))  # 6x14x14 -> 16x5x5

        # Flatten
        x = x.reshape(x.shape[0], -1)  # 16x5x5 -> 400

        # Classifier
        x = self.relu(self.fc1(x))  # 400 -> 120
        x = self.relu(self.fc2(x))  # 120 -> 84
        x = self.fc3(x)  # 84 -> num_classes

        return x


class VGGBlock(nn.Module):
    """
    VGG-style block with repeated conv layers.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_convs: Number of conv layers in block
    """

    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()

        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SmallVGG(nn.Module):
    """
    Small VGG-style network for CIFAR-10.

    A simplified VGG architecture suitable for 32x32 images.

    Args:
        num_classes: Number of output classes (default: 10)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),    # 32x32 -> 16x16
            VGGBlock(64, 128, 2),  # 16x16 -> 8x8
            VGGBlock(128, 256, 2), # 8x8 -> 4x4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [N, 3, 32, 32]

        Returns:
            Output logits of shape [N, num_classes]
        """
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


__all__ = ['SimpleCNN', 'LeNet5', 'VGGBlock', 'SmallVGG']
