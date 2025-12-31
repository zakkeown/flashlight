"""
MNIST Training Example

Trains an MLP on MNIST dataset to demonstrate the training pipeline.
"""

import sys
sys.path.insert(0, '../..')

import mlx_compat
import mlx_compat.nn as nn
import mlx_compat.optim as optim

# Import model
sys.path.insert(0, '..')
from models.mlp import MLP, SimpleMLP


def create_synthetic_mnist_batch(batch_size=32):
    """
    Create synthetic MNIST-like data for demonstration.

    In a real scenario, you would load actual MNIST data.

    Returns:
        x: Input images [batch_size, 1, 28, 28]
        y: Target labels [batch_size]
    """
    # Random images
    x = mlx_compat.randn(batch_size, 784)

    # Random labels (0-9)
    y = mlx_compat.tensor(
        mlx_compat.randint(0, 10, (batch_size,))._mlx_array.astype(mlx_compat.int32._mlx_dtype),
        requires_grad=False
    )

    return x, y


def train_epoch(model, optimizer, criterion, num_batches=100, batch_size=32):
    """
    Train for one epoch.

    Args:
        model: Neural network model
        optimizer: Optimizer
        criterion: Loss function
        num_batches: Number of batches per epoch
        batch_size: Batch size

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch_idx in range(num_batches):
        # Get batch
        x, y = create_synthetic_mnist_batch(batch_size)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += float(loss.numpy())

        if (batch_idx + 1) % 20 == 0:
            print(f'  Batch [{batch_idx + 1}/{num_batches}], Loss: {float(loss.numpy()):.4f}')

    return total_loss / num_batches


def evaluate(model, criterion, num_batches=20, batch_size=32):
    """
    Evaluate the model.

    Args:
        model: Neural network model
        criterion: Loss function
        num_batches: Number of batches to evaluate
        batch_size: Batch size

    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(num_batches):
        # Get batch
        x, y = create_synthetic_mnist_batch(batch_size)

        # Forward pass (no gradient)
        outputs = model(x)
        loss = criterion(outputs, y)

        # Calculate accuracy
        predictions = mlx_compat.argmax(outputs, dim=1)
        correct += int(mlx_compat.sum(predictions == y).numpy())
        total += batch_size

        total_loss += float(loss.numpy())

    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    """Main training loop."""
    print('=' * 60)
    print('MNIST Training Example')
    print('=' * 60)

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 64

    # Create model
    print('\nCreating model...')
    model = SimpleMLP()
    print(model)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f'\nTraining for {num_epochs} epochs...')
    print(f'Learning rate: {learning_rate}')
    print(f'Batch size: {batch_size}')
    print(f'Optimizer: Adam')

    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')

        # Train
        train_loss = train_epoch(model, optimizer, criterion, num_batches=50, batch_size=batch_size)
        print(f'  Training Loss: {train_loss:.4f}')

        # Evaluate
        val_loss, val_accuracy = evaluate(model, criterion, num_batches=10, batch_size=batch_size)
        print(f'  Validation Loss: {val_loss:.4f}')
        print(f'  Validation Accuracy: {val_accuracy:.2f}%')

    print('\n' + '=' * 60)
    print('Training complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
