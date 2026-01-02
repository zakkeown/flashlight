# Training Models

This guide covers everything you need to train neural networks in Flashlight: optimizers, learning rate schedulers, gradient handling, and complete training loops.

## Training Loop Basics

A standard training loop has these components:

```python
import flashlight
import flashlight.nn as nn

# Model, loss, optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = flashlight.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Optimizers

### SGD

Stochastic Gradient Descent with optional momentum and weight decay:

```python
# Basic SGD
optimizer = flashlight.optim.SGD(model.parameters(), lr=0.01)

# With momentum
optimizer = flashlight.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)

# With Nesterov momentum
optimizer = flashlight.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True,
)

# With weight decay (L2 regularization)
optimizer = flashlight.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)
```

### Adam

Adaptive moment estimation, good default choice:

```python
optimizer = flashlight.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

### AdamW

Adam with decoupled weight decay (recommended for transformers):

```python
optimizer = flashlight.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01,  # Decoupled from gradient update
)
```

### Parameter Groups

Apply different learning rates to different parts of the model:

```python
optimizer = flashlight.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3},
])

# Access and modify groups
for group in optimizer.param_groups:
    group['lr'] *= 0.1  # Reduce all learning rates
```

## Learning Rate Schedulers

### StepLR

Decay learning rate by factor every N epochs:

```python
scheduler = flashlight.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,    # Decay every 10 epochs
    gamma=0.1,       # Multiply LR by 0.1
)

for epoch in range(100):
    train_epoch(model, dataloader, optimizer)
    scheduler.step()  # Update LR
```

### MultiStepLR

Decay at specific milestones:

```python
scheduler = flashlight.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # Decay at these epochs
    gamma=0.1,
)
```

### ExponentialLR

Exponential decay every epoch:

```python
scheduler = flashlight.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95,  # LR = LR * 0.95 each epoch
)
```

### CosineAnnealingLR

Cosine annealing to minimum LR:

```python
scheduler = flashlight.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,        # Total epochs
    eta_min=1e-6,     # Minimum LR
)
```

### ReduceLROnPlateau

Reduce LR when metric stops improving:

```python
scheduler = flashlight.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 'min' for loss, 'max' for accuracy
    factor=0.1,       # Reduce LR by factor
    patience=10,      # Wait this many epochs
    threshold=1e-4,   # Minimum change to qualify as improvement
)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)  # Pass metric value
```

### OneCycleLR

Super-convergence schedule:

```python
scheduler = flashlight.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=len(train_loader),
)

for epoch in range(100):
    for batch in train_loader:
        train_step(model, batch, optimizer)
        scheduler.step()  # Step after each batch
```

### Chaining Schedulers

Apply multiple schedulers:

```python
scheduler1 = flashlight.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)
scheduler2 = flashlight.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.9
)
scheduler = flashlight.optim.lr_scheduler.ChainedScheduler(
    [scheduler1, scheduler2]
)
```

## Gradient Handling

### Gradient Clipping

Prevent exploding gradients:

```python
# Clip by norm
flashlight.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

# Clip by value
flashlight.nn.utils.clip_grad_value_(
    model.parameters(),
    clip_value=0.5,
)

# In training loop
optimizer.zero_grad()
loss.backward()
flashlight.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(dataloader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Disabling Gradients

For inference or freezing layers:

```python
# Context manager
with flashlight.no_grad():
    output = model(x)  # No gradient tracking

# Inference decorator
@flashlight.inference_mode()
def predict(model, x):
    return model(x)

# Freeze specific layers
for param in model.backbone.parameters():
    param.requires_grad = False
```

### Inspecting Gradients

```python
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm().item():.4f}")
```

## Checkpointing

### Saving Checkpoints

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    flashlight.save(checkpoint, path)
```

### Loading Checkpoints

```python
def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = flashlight.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

### Best Model Tracking

```python
best_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    if val_loss < best_loss:
        best_loss = val_loss
        flashlight.save(model.state_dict(), 'best_model.pth')

    scheduler.step()
```

## Complete Training Example

Here's a complete training script for image classification:

```python
import flashlight
import flashlight.nn as nn

# Configuration
config = {
    'epochs': 100,
    'batch_size': 128,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
}

# Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = SimpleCNN(num_classes=10)

# Optimizer and scheduler
optimizer = flashlight.optim.SGD(
    model.parameters(),
    lr=config['lr'],
    momentum=config['momentum'],
    weight_decay=config['weight_decay'],
)
scheduler = flashlight.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config['epochs']
)
criterion = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with flashlight.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(loader), 100. * correct / total

# Training loop
best_acc = 0
for epoch in range(config['epochs']):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch+1}/{config['epochs']}")
    print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        flashlight.save(model.state_dict(), 'best_model.pth')

print(f"Best validation accuracy: {best_acc:.2f}%")
```

## PyTorch Migration Notes

### What Works the Same

- All optimizer APIs (SGD, Adam, AdamW)
- All scheduler APIs
- `zero_grad()`, `backward()`, `step()` pattern
- `state_dict()` for saving/loading
- Gradient clipping utilities

### Key Differences

1. **No CUDA synchronization**: No need for `torch.cuda.synchronize()`
2. **Unified memory**: No `.to(device)` calls needed
3. **No AMP**: Mixed precision handled differently (use dtype directly)

### Common Migration Pattern

```python
# PyTorch
model = model.cuda()
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    ...

# Flashlight
for x, y in loader:
    ...  # No device transfer needed
```
