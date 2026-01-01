"""
Multi-Layer Perceptron (MLP) for MNIST

A simple feedforward neural network for classifying MNIST digits.
"""

import sys
sys.path.insert(0, '../..')

import flashlight as mx
import flashlight.nn as nn


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST classification.

    Architecture:
        Input (784) -> FC (256) -> ReLU -> FC (128) -> ReLU -> FC (10)

    Args:
        input_size: Size of input features (default: 784 for MNIST)
        hidden_sizes: List of hidden layer sizes (default: [256, 128])
        num_classes: Number of output classes (default: 10)
        dropout: Dropout probability (default: 0.2)

    Example:
        >>> model = MLP()
        >>> x = mx.randn(32, 784)  # batch of 32 images
        >>> logits = model(x)
        >>> print(logits.shape)  # (32, 10)
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: list = None,
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 784) or (batch_size, 28, 28)

        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        return self.network(x)


class SimpleMLP(nn.Module):
    """
    Even simpler MLP with just one hidden layer.

    Architecture:
        Input (784) -> FC (128) -> ReLU -> FC (10)

    Example:
        >>> model = SimpleMLP()
        >>> x = mx.randn(32, 784)
        >>> logits = model(x)
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Test the model
    print("Testing MLP model...")

    # Create model
    model = MLP()
    print(f"Model: {model}")

    # Test forward pass
    batch_size = 32
    x = mx.randn(batch_size, 28, 28)
    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test with requires_grad
    x_grad = mx.randn(batch_size, 784, requires_grad=True)
    output = model(x_grad)
    loss = mx.sum(output)
    loss.backward()

    print(f"\nGradient test passed!")
    print(f"Input gradient shape: {x_grad.grad.shape}")
