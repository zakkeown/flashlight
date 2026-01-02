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
    Supports loading PyTorch checkpoint files directly.

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

    # First, try to load as flashlight format
    try:
        if isinstance(f, (str, os.PathLike)):
            with open(f, "rb") as file:
                obj = pickle_module.load(file, **pickle_load_args)
        else:
            obj = pickle_module.load(f, **pickle_load_args)

        # Convert from serializable format back to Tensors
        return _from_serializable(obj)
    except Exception as flashlight_error:
        pass

    # If that fails, try to load as PyTorch format
    try:
        return _load_pytorch_checkpoint(f, weights_only=weights_only)
    except ImportError:
        raise RuntimeError(
            "Failed to load file. It may be a PyTorch checkpoint, but PyTorch "
            "is not installed. Install PyTorch to load PyTorch checkpoints: "
            "pip install torch"
        )
    except Exception as pytorch_error:
        raise RuntimeError(
            f"Failed to load file. Tried flashlight format and PyTorch format.\n"
            f"Flashlight error: {flashlight_error}\n"
            f"PyTorch error: {pytorch_error}"
        )


def _load_pytorch_checkpoint(
    f: Union[str, os.PathLike, IO[bytes]],
    weights_only: bool = None,
) -> Any:
    """
    Load a PyTorch checkpoint and convert tensors to flashlight Tensors.

    Args:
        f: File path or file object
        weights_only: If True, only load weights (passed to torch.load)

    Returns:
        Loaded object with PyTorch tensors converted to flashlight Tensors
    """
    import torch

    # Build kwargs for torch.load
    load_kwargs = {"map_location": "cpu"}
    if weights_only is not None:
        load_kwargs["weights_only"] = weights_only

    # Load with PyTorch
    obj = torch.load(f, **load_kwargs)

    # Convert PyTorch tensors to flashlight Tensors
    return _convert_pytorch_object(obj)


def _convert_pytorch_object(obj: Any) -> Any:
    """
    Recursively convert PyTorch tensors to flashlight Tensors.
    """
    # Check if it's a PyTorch tensor
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return _convert_pytorch_tensor(obj)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _convert_pytorch_object(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [_convert_pytorch_object(item) for item in obj]
        return type(obj)(result) if isinstance(obj, tuple) else result
    else:
        return obj


def _convert_pytorch_tensor(torch_tensor) -> Tensor:
    """
    Convert a PyTorch tensor to a flashlight Tensor.

    Args:
        torch_tensor: A PyTorch tensor

    Returns:
        A flashlight Tensor with the same data
    """
    import numpy as np

    # Move to CPU and convert to numpy
    numpy_array = torch_tensor.detach().cpu().numpy()

    # Map dtype
    dtype_map = {
        np.float32: mx.float32,
        np.float16: mx.float16,
        np.float64: mx.float32,  # MLX doesn't support float64
        np.int32: mx.int32,
        np.int64: mx.int64,
        np.int16: mx.int16,
        np.int8: mx.int8,
        np.uint8: mx.uint8,
        np.uint16: mx.uint16,
        np.uint32: mx.uint32,
        np.uint64: mx.uint64,
        np.bool_: mx.bool_,
    }

    # Get target dtype
    numpy_dtype = numpy_array.dtype.type
    mlx_dtype = dtype_map.get(numpy_dtype, mx.float32)

    # Handle bfloat16 specially (numpy doesn't have it)
    if str(torch_tensor.dtype) == "torch.bfloat16":
        # Convert to float32 first, then to bfloat16
        numpy_array = torch_tensor.float().detach().cpu().numpy()
        mlx_dtype = mx.bfloat16

    # Create MLX array
    arr = mx.array(numpy_array, dtype=mlx_dtype)

    # Create flashlight Tensor
    tensor = Tensor._from_mlx_array(arr)

    # Note: We don't preserve requires_grad for loaded tensors
    # (consistent with PyTorch's default behavior when loading)

    return tensor


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
