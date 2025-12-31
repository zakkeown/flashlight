# Pretrained Models and torch.hub

## Overview

PyTorch Hub (`torch.hub`) provides a unified API for loading pretrained models from GitHub repositories. It handles downloading, caching, and instantiating models with their pretrained weights.

**Reference File:** `torch/hub.py`

## Architecture

```
torch.hub Workflow
├── Repository Discovery
│   ├── GitHub API queries
│   └── hubconf.py parsing
├── Caching
│   ├── Model code cache
│   └── Weights cache
├── Model Loading
│   ├── Entrypoint execution
│   └── State dict loading
└── Security
    ├── Trusted repos
    └── weights_only mode
```

---

## Core Functions

### torch.hub.load()

Load a model from a GitHub repo or local directory.

```python
def load(
    repo_or_dir: str,              # 'owner/repo[:ref]' or local path
    model: str,                    # Entrypoint name in hubconf.py
    *args,                         # Args for the entrypoint
    source: str = 'github',        # 'github' or 'local'
    trust_repo: bool | str = None, # Trust level: True, False, 'check'
    force_reload: bool = False,    # Force fresh download
    verbose: bool = True,          # Show download messages
    skip_validation: bool = False, # Skip GitHub validation
    **kwargs                       # Kwargs for the entrypoint
) -> Any
```

### Basic Usage

```python
import torch

# Load ResNet50 with pretrained weights
model = torch.hub.load(
    'pytorch/vision:v0.10.0',    # Repo with version tag
    'resnet50',                   # Model name (entrypoint)
    weights='ResNet50_Weights.IMAGENET1K_V1'
)

# Load from main branch
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

# Load from local directory
model = torch.hub.load('/path/to/repo', 'my_model', source='local')
```

### Trust Levels

```python
# Trust the repo (adds to trusted list)
model = torch.hub.load('pytorch/vision', 'resnet50', trust_repo=True)

# Check against trusted list
model = torch.hub.load('pytorch/vision', 'resnet50', trust_repo='check')

# Prompt user for trust decision
model = torch.hub.load('user/repo', 'model', trust_repo=False)
```

---

### torch.hub.list()

List available entrypoints in a repository.

```python
def list(
    github: str,                   # 'owner/repo[:ref]'
    force_reload: bool = False,
    skip_validation: bool = False,
    trust_repo: bool | str = None
) -> list[str]
```

**Example:**

```python
# List all available models
models = torch.hub.list('pytorch/vision')
print(models)
# ['alexnet', 'densenet121', 'densenet169', ..., 'vgg19_bn', 'wide_resnet50_2']
```

---

### torch.hub.help()

Get docstring for a specific entrypoint.

```python
def help(
    github: str,                   # 'owner/repo[:ref]'
    model: str,                    # Entrypoint name
    force_reload: bool = False,
    skip_validation: bool = False,
    trust_repo: bool | str = None
) -> str
```

**Example:**

```python
# Get model documentation
help_text = torch.hub.help('pytorch/vision', 'resnet50')
print(help_text)
```

---

### torch.hub.load_state_dict_from_url()

Download and load a state dict from a URL.

```python
def load_state_dict_from_url(
    url: str,                      # URL to download
    model_dir: str | None = None,  # Cache directory
    map_location: MAP_LOCATION = None,  # Device mapping
    progress: bool = True,         # Show progress bar
    check_hash: bool = False,      # Verify SHA256 hash
    file_name: str | None = None,  # Override filename
    weights_only: bool = False     # Security: weights only
) -> dict[str, Any]
```

**Example:**

```python
# Download weights directly
state_dict = torch.hub.load_state_dict_from_url(
    'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    progress=True,
    weights_only=True  # Recommended for security
)

# Apply to model
model = MyModel()
model.load_state_dict(state_dict)
```

---

### torch.hub.download_url_to_file()

Download a file from URL to local path.

```python
def download_url_to_file(
    url: str,                      # Source URL
    dst: str,                      # Destination path
    hash_prefix: str | None = None, # Expected hash prefix
    progress: bool = True          # Show progress
) -> None
```

**Example:**

```python
# Download file
torch.hub.download_url_to_file(
    'https://example.com/weights.pth',
    '/path/to/save/weights.pth',
    progress=True
)
```

---

## Cache Management

### Get/Set Cache Directory

```python
# Get current cache directory
cache_dir = torch.hub.get_dir()
print(cache_dir)  # ~/.cache/torch/hub

# Set custom cache directory
torch.hub.set_dir('/custom/cache/path')
```

### Cache Structure

```
~/.cache/torch/
├── hub/
│   ├── pytorch_vision_v0.10.0/   # Downloaded repo
│   │   ├── hubconf.py
│   │   └── ...
│   ├── checkpoints/               # Downloaded weights
│   │   ├── resnet50-0676ba61.pth
│   │   └── vgg16-397923af.pth
│   └── trusted_list                # Trusted repos
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TORCH_HOME` | Override default cache (~/.cache/torch) |
| `XDG_CACHE_HOME` | XDG cache base directory |
| `GITHUB_TOKEN` | GitHub API token for rate limits |

---

## Creating a Hub Repository

### hubconf.py

Each repository must have a `hubconf.py` at the root:

```python
# hubconf.py

# Optional: declare dependencies
dependencies = ['torch', 'torchvision']

def my_model(pretrained=False, **kwargs):
    """
    My custom model.

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional model arguments

    Returns:
        nn.Module: The model
    """
    from .models import MyModel

    model = MyModel(**kwargs)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            'https://example.com/my_model_weights.pth',
            progress=True,
            weights_only=True
        )
        model.load_state_dict(state_dict)

    return model


def another_model(num_classes=1000, pretrained=False):
    """Another model entrypoint."""
    from .models import AnotherModel
    return AnotherModel(num_classes=num_classes)
```

### Best Practices for Hub Repos

1. **Declare dependencies** - List required packages
2. **Provide docstrings** - Document each entrypoint
3. **Support pretrained** - Offer pretrained weight loading
4. **Use weights_only** - For secure weight loading
5. **Version your weights** - Host at stable URLs
6. **Test locally** - Use `source='local'` during development

---

## Loading Weights into Custom Models

### Pattern 1: Direct State Dict

```python
# Load weights
state_dict = torch.hub.load_state_dict_from_url(
    url,
    weights_only=True,
    map_location='cpu'
)

# Create model and load
model = MyModel()
model.load_state_dict(state_dict)
```

### Pattern 2: Partial Loading

```python
# Load pretrained backbone
state_dict = torch.hub.load_state_dict_from_url(url, weights_only=True)

# Filter keys for backbone only
backbone_state = {
    k.replace('backbone.', ''): v
    for k, v in state_dict.items()
    if k.startswith('backbone.')
}

# Load partially
model.backbone.load_state_dict(backbone_state, strict=False)
```

### Pattern 3: Transfer Learning

```python
# Load full pretrained model
pretrained = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

# Extract features
backbone = torch.nn.Sequential(*list(pretrained.children())[:-1])

# New classifier
model = torch.nn.Sequential(
    backbone,
    torch.nn.Flatten(),
    torch.nn.Linear(2048, num_classes)
)
```

---

## Security Considerations

### weights_only Mode

Always use `weights_only=True` for untrusted weights:

```python
# SAFE: Only tensor data loaded
state_dict = torch.hub.load_state_dict_from_url(
    url,
    weights_only=True  # Prevents arbitrary code execution
)

# UNSAFE: Can execute arbitrary code in pickle
state_dict = torch.hub.load_state_dict_from_url(
    url,
    weights_only=False  # Only for trusted sources
)
```

### Trusted Repositories

The following organizations are auto-trusted:
- `pytorch`
- `facebookresearch`
- `facebookincubator`
- `fairinternal`

For other repos, explicitly set trust:

```python
# Prompt user
model = torch.hub.load('user/repo', 'model', trust_repo=False)

# Or trust explicitly
model = torch.hub.load('user/repo', 'model', trust_repo=True)
```

---

## MLX Mapping

### Loading Pretrained Models in MLX

MLX doesn't have a direct equivalent to torch.hub, but provides patterns for loading weights:

```python
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

# Method 1: From safetensors
def load_mlx_model(model_path):
    model = MyMLXModel()

    with safe_open(model_path, framework="numpy") as f:
        weights = {k: mx.array(f.get_tensor(k)) for k in f.keys()}

    model.load_weights(weights)
    return model

# Method 2: From NumPy files
def load_from_numpy(model, npz_path):
    weights = dict(np.load(npz_path))
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    model.load_weights(mlx_weights)
    return model
```

### Converting PyTorch Hub Models to MLX

```python
import torch
import mlx.core as mx
import numpy as np

def convert_pytorch_hub_to_mlx(repo, model_name, **kwargs):
    """Convert a PyTorch Hub model to MLX format."""

    # Load PyTorch model
    pt_model = torch.hub.load(repo, model_name, **kwargs)
    pt_model.eval()

    # Extract weights
    state_dict = pt_model.state_dict()

    # Convert to MLX arrays
    mlx_weights = {}
    for name, param in state_dict.items():
        # Convert to numpy then MLX
        np_array = param.cpu().numpy()
        mlx_weights[name] = mx.array(np_array)

    return mlx_weights


# Usage
weights = convert_pytorch_hub_to_mlx(
    'pytorch/vision',
    'resnet50',
    pretrained=True
)

# Save in MLX format
mx.save('resnet50_mlx.npz', weights)
```

### MLX-Native Model Hub

For MLX-specific models, use Hugging Face Hub with MLX format:

```python
from huggingface_hub import hf_hub_download
import mlx.core as mx

# Download MLX weights
weights_path = hf_hub_download(
    repo_id="mlx-community/model-name",
    filename="weights.safetensors"
)

# Load directly
model = load_mlx_model(weights_path)
```

### Key Differences

| Aspect | PyTorch Hub | MLX |
|--------|-------------|-----|
| Central API | `torch.hub` | No direct equivalent |
| Weight format | .pth (pickle) | .safetensors, .npz |
| Code loading | hubconf.py | Separate model code |
| Caching | Automatic | Manual |
| Security | weights_only mode | safetensors (inherently safe) |

---

## Common Patterns

### Listing and Loading Available Models

```python
# List models
models = torch.hub.list('pytorch/vision')

# Get help
for model in models[:5]:
    print(f"\n{model}:")
    print(torch.hub.help('pytorch/vision', model)[:200])
```

### Force Reload

```python
# Clear cache and reload
model = torch.hub.load(
    'pytorch/vision',
    'resnet50',
    force_reload=True,  # Re-download repo
    pretrained=True
)
```

### Offline Usage

```python
# First run (online): downloads and caches
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

# Later runs (offline): uses cache
# Works if repo and weights were previously downloaded
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
```

### Custom Model Directory

```python
import os

# Set custom model directory
os.environ['TORCH_HOME'] = '/my/custom/cache'

# Or programmatically
torch.hub.set_dir('/my/custom/cache')

# Now loads/saves to custom location
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
```

---

## Summary

| Function | Purpose |
|----------|---------|
| `torch.hub.load()` | Load model from repo |
| `torch.hub.list()` | List available models |
| `torch.hub.help()` | Get model documentation |
| `torch.hub.load_state_dict_from_url()` | Download weights |
| `torch.hub.download_url_to_file()` | Download any file |
| `torch.hub.get_dir()` | Get cache directory |
| `torch.hub.set_dir()` | Set cache directory |

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `repo_or_dir` | GitHub repo ('owner/repo:tag') or local path |
| `source` | 'github' or 'local' |
| `trust_repo` | Security trust level |
| `force_reload` | Force fresh download |
| `weights_only` | Secure weight loading |
| `map_location` | Device mapping for weights |
