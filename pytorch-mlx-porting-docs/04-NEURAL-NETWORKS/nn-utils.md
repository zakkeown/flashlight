# Neural Network Utilities (torch.nn.utils)

## Overview

`torch.nn.utils` provides essential utilities for neural network training and deployment:

- **Gradient Clipping**: Prevent exploding gradients
- **Weight Normalization**: Reparameterize weights for faster training
- **Spectral Normalization**: Stabilize GAN training
- **RNN Utilities**: Handle variable-length sequences
- **Parameter Utilities**: Convert between flat/structured representations
- **Pruning**: Remove network connections

**Source**: `torch/nn/utils/`

---

## Gradient Clipping

### clip_grad_norm_

Clips gradient norm to prevent exploding gradients.

```python
torch.nn.utils.clip_grad_norm_(
    parameters: Iterable[Tensor] | Tensor,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> Tensor
```

**Source**: `torch/nn/utils/clip_grad.py:186-233`

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `parameters` | Iterable/Tensor | Parameters with gradients |
| `max_norm` | float | Maximum allowed gradient norm |
| `norm_type` | float | Type of norm (2.0 = L2, inf = infinity) |
| `error_if_nonfinite` | bool | Raise error on NaN/Inf gradients |
| `foreach` | bool | Use vectorized implementation |

**Formula**:
```
grad = grad * min(max_norm / (total_norm + 1e-6), 1)
```

**Example**:
```python
import torch.nn.utils as utils

# Standard gradient clipping
optimizer.zero_grad()
loss.backward()
total_norm = utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# With different norm types
utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)  # L2
utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=float('inf'))  # L∞

# Error on non-finite gradients
utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
```

### clip_grad_value_

Clips gradient values element-wise to a range.

```python
torch.nn.utils.clip_grad_value_(
    parameters: Iterable[Tensor] | Tensor,
    clip_value: float,
    foreach: bool | None = None,
) -> None
```

**Source**: `torch/nn/utils/clip_grad.py:257-299`

**Example**:
```python
# Clip gradients to [-1, 1]
utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

---

## Weight Normalization

Reparameterizes weights as `w = g * v / ||v||` for faster convergence.

### weight_norm (Deprecated)

```python
torch.nn.utils.weight_norm(
    module: Module,
    name: str = 'weight',
    dim: int = 0,
) -> Module
```

**Note**: Deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.

**Source**: `torch/nn/utils/weight_norm.py`

**Formula**:
```
w = g * (v / ||v||)
```
Where:
- `g` is the magnitude (scalar per output channel)
- `v` is the direction (original weight shape)

**Example**:
```python
from torch.nn.utils import weight_norm

# Apply weight normalization
layer = nn.Linear(20, 40)
layer = weight_norm(layer, name='weight')

# Accesses weight_g and weight_v parameters
print(layer.weight_g.shape)  # (40, 1)
print(layer.weight_v.shape)  # (40, 20)

# Remove normalization
from torch.nn.utils import remove_weight_norm
remove_weight_norm(layer, name='weight')
```

### Modern API (parametrizations)

```python
from torch.nn.utils.parametrizations import weight_norm

layer = nn.Linear(20, 40)
layer = weight_norm(layer, name='weight')
```

---

## Spectral Normalization

Normalizes weights by their spectral norm (largest singular value) for GAN stability.

### spectral_norm

```python
torch.nn.utils.spectral_norm(
    module: Module,
    name: str = 'weight',
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int = 0,
) -> Module
```

**Source**: `torch/nn/utils/spectral_norm.py`

**Algorithm**:
1. Initialize random vectors `u` and `v`
2. Power iteration: `v = W^T u / ||W^T u||`, `u = W v / ||W v||`
3. Spectral norm: `σ = u^T W v`
4. Normalized weight: `W_norm = W / σ`

**Example**:
```python
from torch.nn.utils import spectral_norm, remove_spectral_norm

# Apply spectral normalization
layer = nn.Conv2d(3, 64, 3)
layer = spectral_norm(layer)

# The weight is now normalized by spectral norm
# During forward pass, weight is recomputed

# Remove normalization
remove_spectral_norm(layer)
```

**GAN Discriminator Example**:
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.fc = spectral_norm(nn.Linear(256 * 4 * 4, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return self.fc(x.flatten(1))
```

---

## RNN Utilities

### PackedSequence

Container for variable-length sequences in RNNs.

```python
class PackedSequence(NamedTuple):
    data: Tensor           # Packed data
    batch_sizes: Tensor    # Elements per timestep
    sorted_indices: Tensor | None
    unsorted_indices: Tensor | None
```

**Source**: `torch/nn/utils/rnn.py:38-199`

### pack_padded_sequence

Pack a padded batch of variable-length sequences.

```python
torch.nn.utils.rnn.pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence
```

**Example**:
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sequences of different lengths (batch_first=True)
sequences = torch.randn(3, 10, 5)  # (batch, max_len, features)
lengths = torch.tensor([10, 7, 4])  # Actual lengths

# Pack sequences
packed = pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=True)

# Process with RNN
output, hidden = rnn(packed)

# Unpack
unpacked, lengths = pad_packed_sequence(output, batch_first=True)
```

### pad_packed_sequence

Unpack a PackedSequence to padded tensor.

```python
torch.nn.utils.rnn.pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Tensor, Tensor]
```

### pad_sequence

Pad a list of variable-length tensors.

```python
torch.nn.utils.rnn.pad_sequence(
    sequences: list[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor
```

**Example**:
```python
from torch.nn.utils.rnn import pad_sequence

# Variable-length sequences
seqs = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(8, 10)]

# Pad to same length
padded = pad_sequence(seqs, batch_first=True, padding_value=0)
# Shape: (3, 8, 10) - padded to max length
```

### pack_sequence

Pack a list of variable-length tensors.

```python
torch.nn.utils.rnn.pack_sequence(
    sequences: list[Tensor],
    enforce_sorted: bool = True,
) -> PackedSequence
```

---

## Parameter Utilities

### parameters_to_vector

Flatten all parameters into a single vector.

```python
torch.nn.utils.parameters_to_vector(
    parameters: Iterable[Tensor],
) -> Tensor
```

**Source**: `torch/nn/utils/convert_parameters.py`

**Example**:
```python
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Flatten parameters
flat_params = parameters_to_vector(model.parameters())
print(flat_params.shape)  # (total_params,)

# Modify flat vector
flat_params *= 0.9

# Unflatten back to model
vector_to_parameters(flat_params, model.parameters())
```

### vector_to_parameters

Copy a vector to model parameters.

```python
torch.nn.utils.vector_to_parameters(
    vec: Tensor,
    parameters: Iterable[Tensor],
) -> None
```

---

## Pruning

Remove connections from neural networks.

**Source**: `torch/nn/utils/prune.py`

### Basic Pruning Methods

```python
import torch.nn.utils.prune as prune

# L1 unstructured pruning (remove smallest weights)
prune.l1_unstructured(module, name='weight', amount=0.3)

# Random unstructured pruning
prune.random_unstructured(module, name='weight', amount=0.3)

# L1 structured pruning (remove entire channels)
prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)

# Random structured pruning
prune.random_structured(module, name='weight', amount=0.3, dim=0)

# Global pruning (across multiple layers)
parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc, 'weight'),
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
```

### Custom Pruning

```python
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        return torch.abs(t) > self.threshold

# Apply custom pruning
prune.custom_from_mask(module, name='weight', mask=custom_mask)
```

### Pruning Operations

```python
# Check if pruned
prune.is_pruned(module)

# Remove pruning reparameterization (make permanent)
prune.remove(module, 'weight')

# Access pruning mask
module.weight_mask
```

---

## Parametrizations

Modern API for parameter transformations.

**Source**: `torch/nn/utils/parametrize.py`, `torch/nn/utils/parametrizations.py`

### register_parametrization

```python
torch.nn.utils.parametrize.register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    unsafe: bool = False,
) -> Module
```

**Example**:
```python
from torch.nn.utils import parametrize

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).T

# Register parametrization
layer = nn.Linear(5, 5)
parametrize.register_parametrization(layer, 'weight', Symmetric())

# Now layer.weight is always symmetric
```

### Built-in Parametrizations

```python
from torch.nn.utils.parametrizations import (
    orthogonal,
    spectral_norm,
    weight_norm,
)

# Orthogonal weights
layer = nn.Linear(10, 10)
layer = orthogonal(layer)  # W^T W = I

# Spectral normalization (new API)
layer = spectral_norm(layer)

# Weight normalization (new API)
layer = weight_norm(layer)
```

---

## Fusion Utilities

Fuse operations for inference optimization.

**Source**: `torch/nn/utils/fusion.py`

```python
from torch.nn.utils.fusion import fuse_conv_bn_eval

# Fuse Conv2d and BatchNorm2d for inference
fused = fuse_conv_bn_eval(conv, bn)
```

---

## MLX Implementation

### Gradient Clipping

```python
import mlx.core as mx

def clip_grad_norm_(grads: dict, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients by norm for MLX."""
    # Compute total norm
    if norm_type == float('inf'):
        norms = [mx.max(mx.abs(g)) for g in grads.values()]
        total_norm = mx.max(mx.stack(norms))
    else:
        norms = [mx.sum(mx.abs(g) ** norm_type) for g in grads.values()]
        total_norm = mx.sum(mx.stack(norms)) ** (1.0 / norm_type)

    # Compute clip coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = mx.minimum(clip_coef, mx.array(1.0))

    # Scale gradients
    clipped_grads = {k: v * clip_coef for k, v in grads.items()}

    return clipped_grads, float(total_norm)

def clip_grad_value_(grads: dict, clip_value: float) -> dict:
    """Clip gradients by value for MLX."""
    return {k: mx.clip(v, -clip_value, clip_value) for k, v in grads.items()}
```

### Weight Normalization

```python
import mlx.core as mx
import mlx.nn as nn

class WeightNorm(nn.Module):
    """Weight normalization for MLX."""
    def __init__(self, module: nn.Module, name: str = 'weight', dim: int = 0):
        super().__init__()
        self.module = module
        self.name = name
        self.dim = dim

        # Get original weight
        weight = getattr(module, name)

        # Compute g and v
        self.g = mx.linalg.norm(weight, axis=self._get_norm_axes(weight))
        self.v = weight

    def _get_norm_axes(self, weight):
        return tuple(i for i in range(weight.ndim) if i != self.dim)

    def _compute_weight(self):
        norm = mx.linalg.norm(self.v, axis=self._get_norm_axes(self.v), keepdims=True)
        return self.g * (self.v / (norm + 1e-12))

    def __call__(self, *args, **kwargs):
        # Compute normalized weight before forward
        weight = self._compute_weight()
        # Temporarily set weight
        original = getattr(self.module, self.name)
        setattr(self.module, self.name, weight)
        result = self.module(*args, **kwargs)
        setattr(self.module, self.name, original)
        return result
```

### Spectral Normalization

```python
class SpectralNorm(nn.Module):
    """Spectral normalization for MLX."""
    def __init__(self, module: nn.Module, name: str = 'weight',
                 n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps

        weight = getattr(module, name)
        height = weight.shape[dim]
        width = weight.size // height

        # Initialize u and v
        self.u = mx.random.normal((height,))
        self.u = self.u / mx.linalg.norm(self.u)
        self.v = mx.random.normal((width,))
        self.v = self.v / mx.linalg.norm(self.v)

    def _reshape_weight(self, weight):
        if self.dim != 0:
            # Move dim to front
            perm = [self.dim] + [i for i in range(weight.ndim) if i != self.dim]
            weight = mx.transpose(weight, perm)
        return weight.reshape(weight.shape[0], -1)

    def _compute_weight(self):
        weight = getattr(self.module, self.name)
        weight_mat = self._reshape_weight(weight)

        # Power iteration
        u, v = self.u, self.v
        for _ in range(self.n_power_iterations):
            v = weight_mat.T @ u
            v = v / (mx.linalg.norm(v) + self.eps)
            u = weight_mat @ v
            u = u / (mx.linalg.norm(u) + self.eps)

        self.u, self.v = u, v

        # Compute spectral norm
        sigma = mx.sum(u * (weight_mat @ v))

        return weight / sigma

    def __call__(self, *args, **kwargs):
        weight = self._compute_weight()
        original = getattr(self.module, self.name)
        setattr(self.module, self.name, weight)
        result = self.module(*args, **kwargs)
        setattr(self.module, self.name, original)
        return result
```

### Sequence Packing (Simplified)

```python
def pack_padded_sequence(input, lengths, batch_first=False):
    """Simple pack implementation for MLX."""
    if not batch_first:
        input = mx.transpose(input, (1, 0, 2))

    batch_size, max_len, features = input.shape

    # Sort by length (descending)
    sorted_indices = mx.argsort(-lengths)
    sorted_lengths = lengths[sorted_indices]
    sorted_input = input[sorted_indices]

    # Pack data
    data_list = []
    batch_sizes = []

    for t in range(max_len):
        mask = sorted_lengths > t
        n_valid = mx.sum(mask)
        if n_valid == 0:
            break
        data_list.append(sorted_input[:n_valid, t, :])
        batch_sizes.append(int(n_valid))

    data = mx.concatenate(data_list, axis=0)
    batch_sizes = mx.array(batch_sizes)

    return data, batch_sizes, sorted_indices
```

---

## Implementation Files

- `torch/nn/utils/__init__.py` - Module exports
- `torch/nn/utils/clip_grad.py` - Gradient clipping
- `torch/nn/utils/weight_norm.py` - Weight normalization
- `torch/nn/utils/spectral_norm.py` - Spectral normalization
- `torch/nn/utils/rnn.py` - RNN utilities
- `torch/nn/utils/prune.py` - Network pruning
- `torch/nn/utils/parametrize.py` - Parametrization API
- `torch/nn/utils/parametrizations.py` - Built-in parametrizations
- `torch/nn/utils/convert_parameters.py` - Parameter conversion
- `torch/nn/utils/fusion.py` - Layer fusion

---

## Summary

| Utility | Purpose | MLX Status |
|---------|---------|------------|
| `clip_grad_norm_` | Gradient clipping | Implement manually |
| `clip_grad_value_` | Value clipping | Implement manually |
| `weight_norm` | Weight reparameterization | Implement manually |
| `spectral_norm` | GAN stability | Implement manually |
| `pack_padded_sequence` | RNN variable-length | Implement manually |
| `prune.*` | Network pruning | Implement manually |
| `parametrize` | Parameter transforms | Implement manually |
