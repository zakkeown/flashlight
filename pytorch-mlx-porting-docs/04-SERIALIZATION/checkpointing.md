# Model Checkpointing and Serialization

## Overview

PyTorch provides `torch.save()` and `torch.load()` for serializing and deserializing tensors, models, and arbitrary Python objects. The serialization system handles device mapping, storage sharing, and security through weights-only loading.

**Reference File:** `torch/serialization.py`

## Core Functions

```
Serialization API
├── torch.save(obj, f)           - Serialize any object to file
├── torch.load(f, map_location)  - Deserialize from file
├── Module.state_dict()          - Extract module parameters
├── Module.load_state_dict()     - Restore module parameters
└── Optimizer.state_dict()       - Extract optimizer state
```

---

## torch.save()

Serializes any Python object to a file.

### Function Signature

```python
def save(
    obj: object,                           # Object to serialize
    f: FileLike,                           # File path or file-like object
    pickle_module: Any = pickle,           # Custom pickle module
    pickle_protocol: int = 2,              # Pickle protocol version
    _use_new_zipfile_serialization: bool = True  # Use zipfile format
) -> None
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `obj` | Any Python object (tensors, models, dicts, etc.) |
| `f` | File path (str/PathLike) or file-like object with `write()` |
| `pickle_module` | Module for pickling (default: pickle, can use dill) |
| `pickle_protocol` | Protocol version (default: 2 for compatibility) |

### Basic Usage

```python
import torch

# Save a tensor
x = torch.randn(10, 10)
torch.save(x, 'tensor.pt')

# Save a dictionary of tensors
checkpoint = {
    'weights': torch.randn(100, 100),
    'bias': torch.randn(100),
}
torch.save(checkpoint, 'checkpoint.pt')

# Save to a buffer
import io
buffer = io.BytesIO()
torch.save(x, buffer)
```

### File Format

Modern PyTorch (1.6+) uses a **zipfile-based format**:

```
checkpoint.pt (zip archive)
├── data.pkl           # Pickled object structure
├── data/0             # Storage for first tensor
├── data/1             # Storage for second tensor
├── byteorder          # Endianness marker
├── .format_version    # Format version (currently "1")
└── .storage_alignment # Storage alignment (default: 64)
```

The zipfile format provides:
- Better cross-platform compatibility
- Efficient memory mapping
- Storage deduplication
- Integrity verification (CRC32)

---

## torch.load()

Deserializes objects saved with `torch.save()`.

### Function Signature

```python
def load(
    f: FileLike,
    map_location: MAP_LOCATION = None,     # Device remapping
    pickle_module: Any = None,              # Custom unpickler
    *,
    weights_only: bool | None = None,       # Security: restrict unpickling
    mmap: bool | None = None,               # Memory-map file
    **pickle_load_args: Any                 # Additional pickle args
) -> Any
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `f` | File path or file-like object |
| `map_location` | Device remapping (see below) |
| `weights_only` | Restrict to safe types (default: True in PyTorch 2.6+) |
| `mmap` | Memory-map storages for lazy loading |

### map_location Options

```python
# String: load all tensors to specific device
data = torch.load('model.pt', map_location='cpu')
data = torch.load('model.pt', map_location='cuda:0')
data = torch.load('model.pt', map_location='mps')

# torch.device object
data = torch.load('model.pt', map_location=torch.device('cpu'))

# Dict: remap specific devices
data = torch.load('model.pt', map_location={'cuda:0': 'cuda:1'})
data = torch.load('model.pt', map_location={'cuda:0': 'cpu'})

# Callable: custom remapping logic
def my_map(storage, location):
    if location.startswith('cuda'):
        return storage.cuda(0)  # All GPU tensors to GPU 0
    return storage

data = torch.load('model.pt', map_location=my_map)
```

### weights_only Mode (Security)

**CRITICAL FOR SECURITY**: `weights_only=True` restricts unpickling to safe types only.

```python
# Safe loading (recommended for untrusted sources)
data = torch.load('model.pt', weights_only=True)

# Unsafe loading (only for trusted sources!)
data = torch.load('model.pt', weights_only=False)
```

**Default behavior (PyTorch 2.6+):** `weights_only=True`

**Allowed types in weights_only mode:**
- Tensors, StorageTypes
- Primitive types (int, float, str, bool, None)
- Collections (dict, list, tuple, set)
- OrderedDict
- Some torch-specific types

**Adding custom safe globals:**

```python
# Register globally
torch.serialization.add_safe_globals([MyCustomClass])

# Or use context manager
with torch.serialization.safe_globals([MyCustomClass]):
    data = torch.load('model.pt', weights_only=True)

# Check what globals are in a checkpoint
unsafe = torch.serialization.get_unsafe_globals_in_checkpoint('model.pt')
```

### Memory Mapping

For large models, use `mmap=True` for lazy loading:

```python
# Tensor data loaded on-demand, not all at once
model = torch.load('large_model.pt', mmap=True)
```

Benefits:
- Reduced peak memory usage
- Faster initial load time
- Efficient for partial model access

---

## Module state_dict

The `state_dict()` method returns a dictionary of all learnable parameters and buffers.

### Module.state_dict()

```python
def state_dict(
    self,
    *,
    destination: dict = None,  # Pre-allocated dict
    prefix: str = '',          # Key prefix
    keep_vars: bool = False    # Keep as Parameters (vs Tensors)
) -> dict[str, Any]
```

### What's Included

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.BatchNorm1d(20),
    nn.Linear(20, 5)
)

state = model.state_dict()
# {
#   '0.weight': tensor(...),
#   '0.bias': tensor(...),
#   '1.weight': tensor(...),
#   '1.bias': tensor(...),
#   '1.running_mean': tensor(...),   # Buffer
#   '1.running_var': tensor(...),    # Buffer
#   '1.num_batches_tracked': tensor(...),  # Buffer
#   '2.weight': tensor(...),
#   '2.bias': tensor(...),
# }
```

**Included:**
- All `nn.Parameter` instances (learnable weights)
- All persistent buffers (registered with `register_buffer()`)

**Not included:**
- Non-persistent buffers (`register_buffer(..., persistent=False)`)
- Regular Python attributes
- Parameters/buffers set to `None`

### Module.load_state_dict()

```python
def load_state_dict(
    self,
    state_dict: Mapping[str, Any],
    strict: bool = True,       # Require exact key match
    assign: bool = False       # Assign vs copy tensors
) -> NamedTuple
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `strict` | If True, keys must exactly match |
| `assign` | If True, assign tensors directly (preserves tensor properties from state_dict) |

### Return Value

Returns a `NamedTuple` with:
- `missing_keys`: Keys in model but not in state_dict
- `unexpected_keys`: Keys in state_dict but not in model

### Usage Examples

```python
# Save model state
torch.save(model.state_dict(), 'model_weights.pt')

# Load model state
model = MyModel()
state_dict = torch.load('model_weights.pt', weights_only=True)
model.load_state_dict(state_dict)

# Partial loading (ignore missing/extra keys)
result = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {result.missing_keys}")
print(f"Unexpected: {result.unexpected_keys}")

# Load with device mapping
state_dict = torch.load('model_weights.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.to('cuda')  # Then move to GPU
```

---

## Optimizer State

Optimizers also have `state_dict()` and `load_state_dict()`.

### Optimizer.state_dict()

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for a while...

opt_state = optimizer.state_dict()
# {
#   'state': {
#       0: {'step': tensor(100), 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
#       1: {'step': tensor(100), 'exp_avg': tensor(...), 'exp_avg_sq': tensor(...)},
#       ...
#   },
#   'param_groups': [
#       {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, ...}
#   ]
# }
```

### Saving/Loading Optimizer

```python
# Save
torch.save(optimizer.state_dict(), 'optimizer.pt')

# Load
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(torch.load('optimizer.pt', weights_only=True))
```

---

## Complete Checkpoint Pattern

### Saving a Training Checkpoint

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, path)
```

### Loading a Training Checkpoint

```python
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore RNG state for reproducibility
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint.get('cuda_rng_state') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
```

### Training Loop with Checkpointing

```python
def train(model, train_loader, epochs, checkpoint_path=None, resume_path=None):
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        start_epoch, _ = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            loss = train_step(model, batch, optimizer)
            epoch_loss += loss.item()

        scheduler.step()

        # Save checkpoint every N epochs
        if checkpoint_path and (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                epoch_loss / len(train_loader),
                f"{checkpoint_path}_epoch{epoch+1}.pt"
            )
```

---

## Storage Sharing

PyTorch preserves storage sharing across serialization:

```python
# Original tensors share storage
a = torch.randn(10)
b = a[2:5]  # View into a

torch.save({'a': a, 'b': b}, 'shared.pt')

# After loading, sharing is preserved
data = torch.load('shared.pt', weights_only=True)
data['a'][3] = 999
print(data['b'][1])  # Also 999 - storage is shared
```

---

## Device Mapping Internals

### Location Tags

Each storage is tagged with its original device:
- `'cpu'` - CPU tensor
- `'cuda:N'` - CUDA device N
- `'mps'` - Apple Metal Performance Shaders
- `'meta'` - Meta tensors (no data)

### Custom Device Registration

```python
def ipu_tag(storage):
    if storage.device.type == 'ipu':
        return 'ipu'
    return None

def ipu_deserialize(storage, location):
    if location.startswith('ipu'):
        return storage.ipu(location)
    return None

torch.serialization.register_package(11, ipu_tag, ipu_deserialize)
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` | Force `weights_only=True` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Force `weights_only=False` |

---

## MLX Mapping

### Saving/Loading in MLX

MLX uses different serialization formats. Common approaches:

```python
import mlx.core as mx
import numpy as np

# Method 1: NumPy format
def save_mlx_model(weights, path):
    np_weights = {k: np.array(v) for k, v in weights.items()}
    np.savez(path, **np_weights)

def load_mlx_model(path):
    data = np.load(path)
    return {k: mx.array(v) for k, v in data.items()}

# Method 2: SafeTensors (recommended for security)
from safetensors import safe_open
from safetensors.numpy import save_file

def save_safetensors(weights, path):
    np_weights = {k: np.array(v) for k, v in weights.items()}
    save_file(np_weights, path)

def load_safetensors(path):
    with safe_open(path, framework="numpy") as f:
        return {k: mx.array(f.get_tensor(k)) for k in f.keys()}
```

### Converting PyTorch Checkpoints to MLX

```python
import torch
import mlx.core as mx
import numpy as np

def convert_pytorch_to_mlx(pytorch_path, mlx_path):
    """Convert PyTorch checkpoint to MLX-compatible format."""
    # Load PyTorch checkpoint
    state_dict = torch.load(pytorch_path, map_location='cpu', weights_only=True)

    # Convert to MLX arrays
    mlx_weights = {}
    for key, tensor in state_dict.items():
        # Convert to numpy then MLX
        np_array = tensor.numpy()
        mlx_weights[key] = mx.array(np_array)

    # Save using safetensors or numpy
    save_safetensors(mlx_weights, mlx_path)

    return mlx_weights
```

### Key Differences

| PyTorch | MLX |
|---------|-----|
| `torch.save()` / `torch.load()` | safetensors or numpy |
| Pickle-based | No pickle (safer) |
| Device mapping built-in | Manual handling |
| state_dict pattern | Similar pattern |

---

## Best Practices

1. **Always use `weights_only=True`** for untrusted checkpoints
2. **Use `.pt` extension** by convention
3. **Save state_dict, not entire model** for flexibility
4. **Include metadata** (epoch, loss, config) in checkpoints
5. **Use `map_location='cpu'`** then move to device for predictable loading
6. **Save optimizer state** for training resumption
7. **Save RNG state** for perfect reproducibility
8. **Use safetensors** for cross-framework compatibility

---

## Common Patterns

### Inference-Only Save

```python
# Save only what's needed for inference
torch.save(model.state_dict(), 'model_inference.pt')

# Or save the entire model (less flexible)
torch.save(model, 'model_full.pt')  # Not recommended
```

### Transfer Learning

```python
# Load pretrained weights, ignore classifier
pretrained = torch.load('pretrained.pt', weights_only=True)
model.load_state_dict(pretrained, strict=False)  # Ignore missing/extra keys
```

### Multi-GPU Checkpoint

```python
# Save from DataParallel/DDP
if hasattr(model, 'module'):
    torch.save(model.module.state_dict(), 'model.pt')
else:
    torch.save(model.state_dict(), 'model.pt')
```

### Selective Loading

```python
# Load only specific layers
state_dict = torch.load('model.pt', weights_only=True)
filtered = {k: v for k, v in state_dict.items() if 'encoder' in k}
model.encoder.load_state_dict(filtered, strict=False)
```
