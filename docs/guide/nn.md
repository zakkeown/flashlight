# Neural Networks

Flashlight's `nn` module provides a PyTorch-compatible API for building neural networks. This guide covers the module system, available layers, and model construction patterns.

## The Module System

### Basic Module

All neural network components inherit from `nn.Module`:

```python
import flashlight
import flashlight.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = MyModel()
output = model(input_tensor)
```

### Parameters and State

```python
# Get all parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Get parameter list (for optimizers)
params = list(model.parameters())

# Get state dict (for saving/loading)
state = model.state_dict()
model.load_state_dict(state)
```

### Training vs Evaluation Mode

```python
model.train()   # Enable training mode (affects Dropout, BatchNorm)
model.eval()    # Enable evaluation mode

# Context manager
with flashlight.no_grad():
    output = model(input)  # No gradient tracking
```

## Building Networks

### Sequential

For simple feed-forward networks:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

### ModuleList

When you need indexed access to layers:

```python
class ResidualStack(nn.Module):
    def __init__(self, num_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = x + self.relu(layer(x))  # Residual connection
        return x
```

### ModuleDict

For named submodules:

```python
class MultiHeadModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Linear(input_dim, 256)
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(256, 10),
            'regression': nn.Linear(256, 1),
        })

    def forward(self, x, task='classification'):
        x = self.shared(x)
        return self.heads[task](x)
```

## Layer Types

### Linear Layers

```python
# Basic linear layer
linear = nn.Linear(in_features=256, out_features=128)

# Without bias
linear_no_bias = nn.Linear(256, 128, bias=False)

# Bilinear layer
bilinear = nn.Bilinear(in1_features=64, in2_features=32, out_features=16)
output = bilinear(x1, x2)
```

### Convolutional Layers

```python
# 1D convolution (for sequences)
conv1d = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

# 2D convolution (for images)
conv2d = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
)

# 3D convolution (for video/volumetric)
conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3)

# Transposed convolution (upsampling)
conv_transpose = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
```

!!! note "Layout Handling"
    Flashlight automatically handles NCHW (PyTorch) ↔ NHWC (MLX) conversion for convolutions. Your code uses PyTorch's NCHW format; Flashlight handles the rest.

### Normalization Layers

```python
# Batch normalization
bn1d = nn.BatchNorm1d(num_features=256)
bn2d = nn.BatchNorm2d(num_features=64)

# Layer normalization
ln = nn.LayerNorm(normalized_shape=256)
ln_multi = nn.LayerNorm([64, 32])  # Multi-dimensional

# Group normalization
gn = nn.GroupNorm(num_groups=8, num_channels=64)

# Instance normalization
instance_norm = nn.InstanceNorm2d(num_features=64)
```

### Activation Functions

```python
# As modules
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
gelu = nn.GELU()
silu = nn.SiLU()  # Swish
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=-1)

# As functions
import flashlight.nn.functional as F

x = F.relu(x)
x = F.gelu(x)
x = F.softmax(x, dim=-1)
```

### Dropout

```python
dropout = nn.Dropout(p=0.5)
dropout2d = nn.Dropout2d(p=0.1)  # Drops entire channels

# Only active during training
model.train()
output = dropout(x)   # Drops elements

model.eval()
output = dropout(x)   # Identity (no dropping)
```

### Pooling Layers

```python
# Max pooling
maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive pooling (output size specified)
adaptive_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
adaptive_max = nn.AdaptiveMaxPool2d(output_size=(7, 7))
```

### Recurrent Layers

```python
# RNN
rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
output, hidden = rnn(x)  # x: (batch, seq, features)

# LSTM
lstm = nn.LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
)
output, (h_n, c_n) = lstm(x)

# GRU
gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
output, hidden = gru(x)
```

### Attention

```python
# Multi-head attention
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True,
)

# Self-attention
attn_output, attn_weights = mha(query, key, value)

# With attention mask
attn_output, _ = mha(query, key, value, attn_mask=mask)
```

### Embedding

```python
# Embedding layer
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)
embedded = embedding(token_ids)  # token_ids: (batch, seq_len) -> (batch, seq_len, 256)

# With padding index
embedding = nn.Embedding(10000, 256, padding_idx=0)
```

## Loss Functions

```python
# Classification
cross_entropy = nn.CrossEntropyLoss()
loss = cross_entropy(logits, targets)  # logits: (N, C), targets: (N,)

nll_loss = nn.NLLLoss()
bce = nn.BCELoss()
bce_with_logits = nn.BCEWithLogitsLoss()

# Regression
mse = nn.MSELoss()
l1 = nn.L1Loss()
smooth_l1 = nn.SmoothL1Loss()
huber = nn.HuberLoss(delta=1.0)

# With reduction options
mse_sum = nn.MSELoss(reduction='sum')
mse_none = nn.MSELoss(reduction='none')  # No reduction
```

## Model Serialization

### Saving and Loading

```python
# Save state dict (recommended)
flashlight.save(model.state_dict(), 'model.pth')

# Load state dict
state_dict = flashlight.load('model.pth')
model.load_state_dict(state_dict)

# Save entire model
flashlight.save(model, 'model_full.pth')
model = flashlight.load('model_full.pth')
```

### Checkpointing

```python
# Save checkpoint with optimizer state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
flashlight.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = flashlight.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## Example: Complete CNN

```python
import flashlight
import flashlight.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Usage
model = CNN(num_classes=10)
x = flashlight.randn(32, 3, 32, 32)  # Batch of 32 RGB 32x32 images
output = model(x)  # (32, 10)
```

## PyTorch Migration Notes

### What Works the Same

- `nn.Module` subclassing and `forward()` method
- All standard layer types
- `parameters()` and `named_parameters()`
- `state_dict()` and `load_state_dict()`
- `train()` and `eval()` modes

### Key Differences

1. **Weight format for Conv2d**: Internal format differs, but API is identical
2. **No CUDA**: No `.cuda()` method; use unified memory
3. **Immutable operations**: Some internal optimizations differ

### Common Migration Pattern

```python
# PyTorch
model = model.cuda()
output = model(x.cuda())

# Flashlight
output = model(x)  # Just works - unified memory
```
