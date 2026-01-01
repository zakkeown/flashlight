"""
Parameter Classes

PyTorch-compatible torch.nn.parameter module for neural network parameters and buffers.
"""

from collections import OrderedDict
from typing import Iterator, Optional

import mlx.core as mx

from ...tensor import Tensor

__all__ = [
    "Parameter",
    "Buffer",
    "UninitializedParameter",
    "UninitializedBuffer",
    "UninitializedTensorMixin",
    "is_lazy",
]


class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.

    Parameters are Tensor subclasses, that have a very special property when used with
    Modules - when they're assigned as Module attributes they are automatically added
    to the list of its parameters.

    Args:
        data: Parameter tensor data
        requires_grad: If the parameter requires gradient (default: True)

    Example:
        >>> weight = nn.Parameter(flashlight.randn(10, 5))
        >>> weight.requires_grad
        True
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a parameter.

        Args:
            *args, **kwargs: Flexible signature matching PyTorch
                data: Initial tensor data
                requires_grad: Whether this parameter requires gradients
        """
        # Parse args/kwargs to match PyTorch's flexible signature
        if len(args) >= 1:
            data = args[0]
            args = args[1:]
        else:
            data = kwargs.pop("data", None)

        requires_grad = kwargs.pop("requires_grad", True)

        if data is None:
            raise ValueError("Parameter data cannot be None")

        # Initialize as a Tensor
        if isinstance(data, Tensor):
            # If it's already a Tensor, use its data
            super().__init__(data._mlx_array, requires_grad=requires_grad)
        else:
            # Otherwise create a new tensor
            super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        """String representation showing it's a parameter."""
        return f"Parameter containing:\n{super().__repr__()}"

    def __reduce_ex__(self, proto):
        """Support for pickling."""
        # Return a tuple (callable, args) to reconstruct the object
        return (Parameter, (self.data, self.requires_grad))


class Buffer(Tensor):
    """
    A buffer is a tensor that is registered with a module but is not considered
    a parameter (i.e., it doesn't require gradients by default).

    Buffers are typically used for things like running statistics in batch normalization.

    Args:
        data: Buffer tensor data (can be None for lazy initialization)
        requires_grad: If the buffer requires gradient (default: False)
        persistent: If True, the buffer will be saved during state_dict (default: True)

    Example:
        >>> buffer = nn.Buffer(torch.zeros(10))
        >>> buffer.requires_grad
        False
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a buffer.

        Args:
            *args, **kwargs: Flexible signature matching PyTorch
        """
        # Parse args/kwargs
        if len(args) >= 1:
            data = args[0]
            args = args[1:]
        else:
            data = kwargs.pop("data", None)

        requires_grad = kwargs.pop("requires_grad", False)
        persistent = kwargs.pop("persistent", True)
        self._persistent = persistent

        if data is None:
            # Create empty tensor for None data
            super().__init__(mx.array([]), requires_grad=requires_grad)
        elif isinstance(data, Tensor):
            super().__init__(data._mlx_array, requires_grad=requires_grad)
        else:
            super().__init__(data, requires_grad=requires_grad)

    @property
    def persistent(self) -> bool:
        """Whether this buffer is persistent (included in state_dict)."""
        return self._persistent

    def __repr__(self):
        """String representation showing it's a buffer."""
        return f"Buffer containing:\n{super().__repr__()}"

    def __reduce_ex__(self, proto):
        """Support for pickling."""
        return (Buffer, (self.data, self.requires_grad, self._persistent))


class UninitializedTensorMixin:
    """
    Mixin class for uninitialized tensors used in lazy modules.

    An uninitialized tensor is a placeholder that will be replaced with a properly
    initialized tensor once the input shape is known.
    """

    _is_uninitialized: bool = True

    def materialize(self, shape, device=None, dtype=None):
        """
        Create a tensor with the given shape.

        Args:
            shape: The shape of the tensor to create
            device: Device (ignored in MLX)
            dtype: Data type for the tensor

        Returns:
            The materialized tensor
        """
        raise NotImplementedError("Subclasses must implement materialize()")

    @property
    def is_materialized(self) -> bool:
        """Check if the tensor has been materialized."""
        return not getattr(self, "_is_uninitialized", False)


class UninitializedParameter(UninitializedTensorMixin, Parameter):
    """
    A parameter that is not yet initialized.

    Used in lazy modules where the parameter shape depends on the input shape.
    Once the input shape is known, the parameter is materialized.

    Args:
        requires_grad: If the parameter requires gradient (default: True)
        device: Device (ignored in MLX, uses unified memory)
        dtype: Data type for the parameter when materialized

    Example:
        >>> param = nn.UninitializedParameter()
        >>> param.is_materialized
        False
        >>> param.materialize((10, 5))
        >>> param.is_materialized
        True
    """

    cls_to_become = Parameter

    def __init__(self, *args, **kwargs):
        """
        Initialize an uninitialized parameter.

        Args:
            *args, **kwargs: Flexible signature matching PyTorch
        """
        requires_grad = kwargs.pop("requires_grad", True)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        # Store configuration for materialization
        self._requires_grad = requires_grad
        self._device = device
        self._dtype = dtype or mx.float32
        self._is_uninitialized = True

        # Create a placeholder array
        placeholder = mx.array([0.0], dtype=self._dtype)
        # Initialize parent with placeholder
        Tensor.__init__(self, placeholder, requires_grad=requires_grad)

    def materialize(self, shape, device=None, dtype=None):
        """
        Create the actual parameter tensor with the given shape.

        Args:
            shape: The shape of the parameter
            device: Device (ignored in MLX)
            dtype: Data type (uses stored dtype if not provided)

        Returns:
            Self, now materialized as a proper Parameter
        """
        dtype = dtype or self._dtype
        # Initialize with random values (standard initialization)
        new_array = mx.random.normal(shape=shape).astype(dtype)
        self._mlx_array = new_array
        self._is_uninitialized = False
        return self

    def __repr__(self):
        if self._is_uninitialized:
            return f"UninitializedParameter(requires_grad={self._requires_grad})"
        return super().__repr__()


class UninitializedBuffer(UninitializedTensorMixin, Buffer):
    """
    A buffer that is not yet initialized.

    Used in lazy modules where the buffer shape depends on the input shape.

    Args:
        requires_grad: If the buffer requires gradient (default: False)
        device: Device (ignored in MLX, uses unified memory)
        dtype: Data type for the buffer when materialized

    Example:
        >>> buffer = nn.UninitializedBuffer()
        >>> buffer.is_materialized
        False
    """

    cls_to_become = Buffer

    def __init__(self, *args, **kwargs):
        """
        Initialize an uninitialized buffer.

        Args:
            *args, **kwargs: Flexible signature matching PyTorch
        """
        requires_grad = kwargs.pop("requires_grad", False)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        self._requires_grad = requires_grad
        self._device = device
        self._dtype = dtype or mx.float32
        self._is_uninitialized = True

        # Create a placeholder array
        placeholder = mx.array([0.0], dtype=self._dtype)
        # Initialize parent with placeholder
        Tensor.__init__(self, placeholder, requires_grad=requires_grad)
        self._persistent = True

    def materialize(self, shape, device=None, dtype=None):
        """
        Create the actual buffer tensor with the given shape.

        Args:
            shape: The shape of the buffer
            device: Device (ignored in MLX)
            dtype: Data type (uses stored dtype if not provided)

        Returns:
            Self, now materialized as a proper Buffer
        """
        dtype = dtype or self._dtype
        # Initialize with zeros (typical for buffers)
        new_array = mx.zeros(shape, dtype=dtype)
        self._mlx_array = new_array
        self._is_uninitialized = False
        return self

    def __repr__(self):
        if self._is_uninitialized:
            return f"UninitializedBuffer(requires_grad={self._requires_grad})"
        return super().__repr__()


def is_lazy(param) -> bool:
    """
    Check if a parameter or buffer is uninitialized (lazy).

    Args:
        param: A Parameter, Buffer, or tensor to check

    Returns:
        True if the parameter is uninitialized, False otherwise

    Example:
        >>> param = nn.UninitializedParameter()
        >>> nn.parameter.is_lazy(param)
        True
        >>> param.materialize((10, 5))
        >>> nn.parameter.is_lazy(param)
        False
    """
    return isinstance(param, UninitializedTensorMixin) and param._is_uninitialized


# Re-export for compatibility
# These are sometimes imported from torch.nn.parameter
torch = None  # Placeholder for module reference
OrderedDict = OrderedDict
