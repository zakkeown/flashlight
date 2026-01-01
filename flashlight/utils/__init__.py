"""
Utilities module (torch.utils compatible).

Provides PyTorch-compatible utility functions and submodules for MLX:

- data: Data loading utilities (DataLoader, Dataset, Samplers)
- hooks: Hook management (RemovableHandle, BackwardHook)
- benchmark: Benchmarking utilities (Timer, Compare, Measurement)
- checkpoint: Gradient checkpointing for memory efficiency
- tensorboard: TensorBoard integration (optional)
- model_zoo: Download pretrained model weights
- collect_env: Environment information for debugging
- weak: Weak reference utilities for tensors
- mobile_optimizer: Mobile/edge deployment optimization

Example:
    >>> from flashlight.utils.data import DataLoader, TensorDataset
    >>> from flashlight.utils.benchmark import Timer, Compare
    >>> from flashlight.utils import checkpoint
    >>> from flashlight.utils.model_zoo import load_url
    >>>
    >>> # Create a checkpoint of a model layer
    >>> out = checkpoint(model.layer, x)
    >>>
    >>> # Load pretrained weights
    >>> state_dict = load_url('https://example.com/model.safetensors')
"""

# Import submodules
from . import benchmark, collect_env, data, hooks, mobile_optimizer, model_zoo, weak
from .checkpoint import CheckpointFunction, checkpoint, checkpoint_sequential
from .collect_env import get_env_info, get_pretty_env_info

# Re-export commonly used items
from .hooks import BackwardHook, RemovableHandle, unserializable_hook, warn_if_has_hooks
from .mobile_optimizer import (
    LintCode,
    MobileOptimizerType,
    generate_mobile_module_lints,
    optimize_for_mobile,
)
from .model_zoo import get_dir, load_url, set_dir
from .weak import TensorWeakRef, WeakIdKeyDictionary, WeakIdRef, WeakRef, WeakTensorKeyDictionary

# Optional tensorboard import
try:
    from . import tensorboard

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

__all__ = [
    # Submodules
    "data",
    "hooks",
    "benchmark",
    "model_zoo",
    "collect_env",
    "weak",
    "mobile_optimizer",
    # Checkpoint functions
    "checkpoint",
    "checkpoint_sequential",
    "CheckpointFunction",
    # Hook utilities
    "RemovableHandle",
    "BackwardHook",
    "unserializable_hook",
    "warn_if_has_hooks",
    # Model zoo
    "load_url",
    "get_dir",
    "set_dir",
    # Collect env
    "get_env_info",
    "get_pretty_env_info",
    # Weak references
    "WeakRef",
    "TensorWeakRef",
    "WeakIdRef",
    "WeakIdKeyDictionary",
    "WeakTensorKeyDictionary",
    # Mobile optimizer
    "optimize_for_mobile",
    "generate_mobile_module_lints",
    "MobileOptimizerType",
    "LintCode",
]

if _TENSORBOARD_AVAILABLE:
    __all__.append("tensorboard")
