"""
Serialization Functions

Implements PyTorch-compatible save/load functions for models and tensors.
"""

import os
import pickle
from typing import IO, Any, Dict, Union

import mlx.core as mx

from .tensor import Tensor


def save(
    obj: Any,
    f: Union[str, os.PathLike, IO[bytes]],
    pickle_module=pickle,
    pickle_protocol: int = 2,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    """
    Save an object to a file.

    Saves a Python object using pickle serialization. For Tensor objects,
    the underlying MLX array is saved.

    Args:
        obj: Object to save (can be a dict, Tensor, Module state_dict, etc.)
        f: A file-like object or a string/path-like object containing a filename
        pickle_module: Module to use for pickling (default: pickle). For PyTorch compatibility.
        pickle_protocol: Pickle protocol to use (default: 2 for compatibility)
        _use_new_zipfile_serialization: Use new zipfile format (ignored in MLX)
        _disable_byteorder_record: Disable byteorder record (ignored in MLX)

    Example:
        >>> # Save a model's state dict
        >>> flashlight.save(model.state_dict(), 'model.pt')
        >>>
        >>> # Save a tensor
        >>> x = flashlight.randn(10, 20)
        >>> flashlight.save(x, 'tensor.pt')
        >>>
        >>> # Save a dictionary
        >>> checkpoint = {'epoch': 10, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        >>> flashlight.save(checkpoint, 'checkpoint.pt')
    """
    # Convert Tensors to serializable format
    serializable = _to_serializable(obj)

    # Handle file path vs file object
    if isinstance(f, (str, os.PathLike)):
        with open(f, "wb") as file:
            pickle_module.dump(serializable, file, protocol=pickle_protocol)
    else:
        pickle_module.dump(serializable, f, protocol=pickle_protocol)


def load(
    f: Union[str, os.PathLike, IO[bytes]],
    map_location=None,
    pickle_module=None,
    weights_only=None,
    mmap=None,
    **pickle_load_args,
) -> Any:
    """
    Load an object from a file.

    Loads an object that was saved with flashlight.save() or torch.save().

    Args:
        f: A file-like object or a string/path-like object containing a filename
        map_location: Device to map tensors to (ignored for MLX, kept for compatibility)
        pickle_module: Module to use for unpickling (default: pickle)
        weights_only: If True, only load weights without executing any code (default: None)
        mmap: Memory-map the file (ignored for MLX, kept for compatibility)
        **pickle_load_args: Additional arguments to pass to pickle.load

    Returns:
        Loaded object

    Example:
        >>> # Load a model's state dict
        >>> state_dict = flashlight.load('model.pt')
        >>> model.load_state_dict(state_dict)
        >>>
        >>> # Load a tensor
        >>> x = flashlight.load('tensor.pt')
        >>>
        >>> # Load a checkpoint
        >>> checkpoint = flashlight.load('checkpoint.pt')
        >>> model.load_state_dict(checkpoint['model'])
        >>> optimizer.load_state_dict(checkpoint['optimizer'])
        >>> epoch = checkpoint['epoch']
    """
    # Use default pickle module if not specified
    if pickle_module is None:
        pickle_module = pickle

    # Handle file path vs file object
    if isinstance(f, (str, os.PathLike)):
        with open(f, "rb") as file:
            obj = pickle_module.load(file, **pickle_load_args)
    else:
        obj = pickle_module.load(f, **pickle_load_args)

    # Convert from serializable format back to Tensors
    return _from_serializable(obj)


def _to_serializable(obj: Any) -> Any:
    """
    Convert an object to a serializable format.

    Recursively converts Tensors to dictionaries with array data.
    """
    if isinstance(obj, Tensor):
        # Convert Tensor to serializable dict
        return {
            "__mlx_tensor__": True,
            "data": obj._mlx_array.tolist(),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "requires_grad": obj.requires_grad,
        }
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [_to_serializable(item) for item in obj]
        return type(obj)(result) if isinstance(obj, tuple) else result
    elif hasattr(obj, "_mlx_array"):
        # Handle Parameter objects
        return {
            "__mlx_tensor__": True,
            "data": obj._mlx_array.tolist(),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "requires_grad": getattr(obj, "requires_grad", False),
        }
    else:
        return obj


def _from_serializable(obj: Any) -> Any:
    """
    Convert from serializable format back to Tensors.

    Recursively converts serializable dicts back to Tensors.
    """
    if isinstance(obj, dict):
        if obj.get("__mlx_tensor__", False):
            # Reconstruct Tensor
            data = obj["data"]
            dtype_str = obj.get("dtype", "float32")
            requires_grad = obj.get("requires_grad", False)

            # Map dtype string to MLX dtype
            dtype_map = {
                "float32": mx.float32,
                "float16": mx.float16,
                "bfloat16": mx.bfloat16,
                "float64": mx.float32,  # MLX doesn't support float64, use float32
                "int32": mx.int32,
                "int64": mx.int64,
                "int16": mx.int16,
                "int8": mx.int8,
                "uint8": mx.uint8,
                "uint16": mx.uint16,
                "uint32": mx.uint32,
                "uint64": mx.uint64,
                "bool": mx.bool_,
            }
            dtype = dtype_map.get(dtype_str, mx.float32)

            arr = mx.array(data, dtype=dtype)
            tensor = Tensor._from_mlx_array(arr)
            tensor.requires_grad = requires_grad
            return tensor
        else:
            return {k: _from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [_from_serializable(item) for item in obj]
        return type(obj)(result) if isinstance(obj, tuple) else result
    else:
        return obj


__all__ = ["save", "load"]
