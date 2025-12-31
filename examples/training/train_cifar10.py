"""
CIFAR-10 Training Example

Trains a CNN on CIFAR-10 dataset to demonstrate the training pipeline.
"""

import sys
sys.path.insert(0, '../..')

import mlx_compat
import mlx_compat.nn as nn
import mlx_compat.optim as optim

# Import model
sys.path.insert(0, '..')
from models.cnn import SimpleCNN, LeNet5


def create_synthetic_cifar10_batch(batch_size=32):
    """
    Create synthetic CIFAR-10-like data for demonstration.

    In a real scenario, you would load actual CIFAR-10 data.

    Returns:
        x: Input images [batch_size, 3, 32, 32]
        y: Target labels [batch_size]
    """
    # Random images (3 channels, 32x32)
    x = mlx_compat.randn(batch_size, 3, 32, 32)

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
        x, y = create_synthetic_cifar10_batch(batch_size)

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
        x, y = create_synthetic_cifar10_batch(batch_size)

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
    print('CIFAR-10 Training Example')
    print('=' * 60)

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 32

    # Create model
    print('\nCreating model...')
    model = LeNet5(num_classes=10)
    print(model)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Optional: Learning rate scheduler
    scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f'\nTraining for {num_epochs} epochs...')
    print(f'Learning rate: {learning_rate}')
    print(f'Batch size: {batch_size}')
    print(f'Optimizer: Adam')
    print(f'LR Scheduler: StepLR (step_size=2, gamma=0.5)')

    # Training loop
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] (LR: {current_lr:.6f})')

        # Train
        train_loss = train_epoch(model, optimizer, criterion, num_batches=50, batch_size=batch_size)
        print(f'  Training Loss: {train_loss:.4f}')

        # Evaluate
        val_loss, val_accuracy = evaluate(model, criterion, num_batches=10, batch_size=batch_size)
        print(f'  Validation Loss: {val_loss:.4f}')
        print(f'  Validation Accuracy: {val_accuracy:.2f}%')

        # Step scheduler
        scheduler.step()

    print('\n' + '=' * 60)
    print('Training complete!')
    print('=' * 60)

    # Final evaluation
    print('\nFinal Evaluation:')
    final_loss, final_accuracy = evaluate(model, criterion, num_batches=20, batch_size=batch_size)
    print(f'  Test Loss: {final_loss:.4f}')
    print(f'  Test Accuracy: {final_accuracy:.2f}%')


if __name__ == '__main__':
    main()
