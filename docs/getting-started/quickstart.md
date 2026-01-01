# Quick Start

This guide will help you get started with Flashlight in just a few minutes.

## Creating Tensors

Flashlight tensors work just like PyTorch tensors:

```python
import flashlight as fl

# Create tensors
x = fl.zeros(3, 4)           # 3x4 tensor of zeros
y = fl.ones(3, 4)            # 3x4 tensor of ones
z = fl.randn(3, 4)           # Random normal distribution
w = fl.tensor([[1, 2], [3, 4]])  # From Python list

# Tensor operations
result = x + y * 2
result = fl.matmul(x, z.T)
```

## Building Neural Networks

Use the familiar `nn.Module` API:

```python
import flashlight.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
```

## Training Loop

```python
import flashlight as fl
import flashlight.nn as nn
import flashlight.optim as optim

# Model and optimizer
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training step
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
```

## Next Steps

- Read the [Migration Guide](migration.md) for moving PyTorch code to Flashlight
- Explore the [API Reference](../reference/) for detailed documentation
- Check out the [Best Practices](../guide/best-practices.md) for optimization tips
