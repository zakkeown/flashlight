"""
PyTorch-compatible Tensor wrapper for MLX arrays.

This module implements the core Tensor class that wraps MLX arrays while
providing a PyTorch-compatible API.

Reference:
- pytorch-mlx-porting-docs/01-FOUNDATIONS/tensor-core.md
- pytorch-mlx-porting-docs/08-PORTING-GUIDE/mlx-mapping.md
"""

from typing import Union, Optional, Tuple, List, Sequence
import warnings

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

import numpy as np

from .dtype import DType, get_dtype, get_default_dtype
from .device import Device, get_default_device


class Tensor:
    """
    PyTorch-compatible tensor wrapper around MLX arrays.

    This class provides a PyTorch-like interface while using MLX arrays
    internally. It handles:
    - Automatic differentiation metadata (requires_grad, grad, grad_fn)
    - View tracking (base tensor for views)
    - Device and dtype management
    - Operator overloading

    Attributes:
        data: The underlying MLX array
        requires_grad: Whether to track gradients
        grad: Accumulated gradients (None if not computed)
        grad_fn: Backward function for autograd (Phase 3)
        device: Device object (compatibility)
        dtype: Data type object
        shape: Tensor shape
        ndim: Number of dimensions
        size: Alias for shape (PyTorch compatibility)

    Example:
        >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x * 2
        >>> print(y.shape)
        (3,)
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Create a new tensor.

        Args:
            data: Input data (list, numpy array, MLX array, or another Tensor)
            dtype: Data type (default: float32)
            device: Device specification (for compatibility)
            requires_grad: Whether to track gradients

        Note:
            Signature is (*args, **kwargs) to match PyTorch exactly.
            Internally parses: Tensor(data, dtype=None, device=None, requires_grad=False)
        """
        # Parse arguments to match PyTorch's flexible signature
        if len(args) == 0:
            data = kwargs.pop('data', None)
            if data is None:
                raise TypeError("Tensor() requires 'data' argument")
        else:
            data = args[0]
            args = args[1:]

        # Handle remaining positional args (rare but possible in PyTorch)
        dtype = kwargs.pop('dtype', args[0] if len(args) > 0 else None)
        device = kwargs.pop('device', args[1] if len(args) > 1 else None)
        requires_grad = kwargs.pop('requires_grad', args[2] if len(args) > 2 else False)

        # Ignore unknown kwargs for PyTorch compatibility
        # (PyTorch accepts things like pin_memory, etc.)
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX is not available. Please install MLX: pip install mlx")

        # Convert input to MLX array
        if isinstance(data, Tensor):
            self._mlx_array = data._mlx_array
            if dtype is None:
                dtype = data.dtype
        elif isinstance(data, np.ndarray):
            self._mlx_array = mx.array(data)
        elif hasattr(data, '__array__'):  # NumPy-like array interface
            self._mlx_array = mx.array(np.array(data))
        else:
            # List, tuple, scalar, or MLX array
            self._mlx_array = mx.array(data)

        # Handle dtype
        if dtype is not None:
            dtype = get_dtype(dtype)
            # Cast to requested dtype
            self._mlx_array = mx.array(self._mlx_array, dtype=dtype._mlx_dtype)
            self._dtype = dtype
        else:
            # Infer dtype from MLX array
            self._dtype = self._infer_dtype(self._mlx_array)

        # Device (compatibility - MLX uses unified memory)
        self._device = Device(device) if device is not None else get_default_device()

        # Autograd metadata (Phase 3)
        self.requires_grad = requires_grad
        self._grad = None  # Accumulated gradients
        self._grad_fn = None  # Backward function (Phase 3)

        # View tracking
        self._base = None  # Base tensor if this is a view
        self._is_view = False

        # Layout tracking for NHWC optimization
        self._layout = None  # None = infer from shape and mode

    @classmethod
    def _from_mlx_array(
        cls,
        mlx_array,
        requires_grad: bool = False,
        grad_fn=None,
        layout=None,
    ) -> 'Tensor':
        """
        Internal: Create tensor directly from MLX array.

        Used by operators to create output tensors efficiently.

        Args:
            mlx_array: The MLX array to wrap
            requires_grad: Whether to track gradients
            grad_fn: Backward function for autograd
            layout: Optional layout hint (Layout enum from layout.py)
        """
        tensor = cls.__new__(cls)
        tensor._mlx_array = mlx_array
        tensor._dtype = tensor._infer_dtype(mlx_array)
        tensor._device = get_default_device()
        tensor.requires_grad = requires_grad
        tensor._grad = None
        tensor._grad_fn = grad_fn
        tensor._base = None
        tensor._is_view = False
        tensor._layout = layout
        return tensor

    def _infer_dtype(self, mlx_array) -> DType:
        """Infer mlx_compat dtype from MLX array dtype."""
        from . import dtype as dtype_module

        # Map MLX dtype to our dtype
        mlx_dtype = mlx_array.dtype
        dtype_map = {
            mx.float32: dtype_module.float32,
            mx.float16: dtype_module.float16,
            mx.bfloat16: dtype_module.bfloat16,
            mx.int8: dtype_module.int8,
            mx.int16: dtype_module.int16,
            mx.int32: dtype_module.int32,
            mx.int64: dtype_module.int64,
            mx.uint8: dtype_module.uint8,
            mx.uint16: dtype_module.uint16,
            mx.uint32: dtype_module.uint32,
            mx.uint64: dtype_module.uint64,
            mx.bool_: dtype_module.bool,
        }

        result = dtype_map.get(mlx_dtype)
        if result is None:
            # Default to float32 for unknown types
            warnings.warn(f"Unknown MLX dtype {mlx_dtype}, defaulting to float32")
            return dtype_module.float32
        return result

    # ==================== Properties ====================

    @property
    def data(self):
        """The underlying MLX array."""
        return self._mlx_array

    @data.setter
    def data(self, value):
        """Set the underlying data (creates new array due to MLX immutability)."""
        if isinstance(value, Tensor):
            self._mlx_array = value._mlx_array
        elif isinstance(value, np.ndarray):
            self._mlx_array = mx.array(value)
        else:
            self._mlx_array = mx.array(value)

    @property
    def dtype(self) -> DType:
        """Data type of the tensor."""
        return self._dtype

    @property
    def device(self) -> Device:
        """Device of the tensor (compatibility)."""
        return self._device

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor."""
        return tuple(self._mlx_array.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._mlx_array.shape)

    @property
    def size(self) -> Tuple[int, ...]:
        """Alias for shape (PyTorch compatibility)."""
        return self.shape

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return int(self._mlx_array.size)

    @property
    def grad(self):
        """Gradient tensor (None if not computed)."""
        return self._grad

    @grad.setter
    def grad(self, value):
        """Set gradient tensor."""
        if value is not None and not isinstance(value, Tensor):
            value = Tensor(value)
        self._grad = value

    @property
    def is_leaf(self) -> bool:
        """
        Check if this is a leaf tensor in the computation graph.

        A tensor is a leaf if:
        - It was created by the user (not by an operation), OR
        - It doesn't require gradients
        """
        return self._grad_fn is None

    @property
    def is_view(self) -> bool:
        """Check if this tensor is a view of another tensor."""
        return self._is_view

    @property
    def layout(self):
        """
        Get the memory layout of this tensor.

        Returns:
            Layout enum value (NCHW, NHWC, etc.) or None for non-spatial tensors.
        """
        from .layout import infer_layout
        return infer_layout(self)

    def to_nhwc(self) -> 'Tensor':
        """
        Convert 4D tensor to NHWC layout if not already.

        Returns:
            Tensor in NHWC layout.
        """
        from .layout import ensure_nhwc
        return ensure_nhwc(self)

    def to_nchw(self) -> 'Tensor':
        """
        Convert 4D tensor to NCHW layout if not already.

        Returns:
            Tensor in NCHW layout.
        """
        from .layout import ensure_nchw
        return ensure_nchw(self)

    def to_layout(self, target_layout) -> 'Tensor':
        """
        Convert tensor to specified layout.

        Args:
            target_layout: Target Layout enum value

        Returns:
            Tensor in target layout.
        """
        from .layout import convert_layout
        return convert_layout(self, target_layout)

    # ==================== Type Conversion ====================

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array(self._mlx_array)

    def tolist(self) -> List:
        """Convert to nested Python list."""
        return self.numpy().tolist()

    def item(self):
        """Get Python scalar (only for single-element tensors)."""
        if self.numel != 1:
            raise ValueError(
                f"Can only convert tensors with exactly one element to Python scalars. "
                f"Tensor has {self.numel} elements."
            )
        return self.numpy().item()

    def to(
        self,
        *args,
        dtype: Optional[Union[DType, str]] = None,
        device: Optional[Union[Device, str]] = None,
        **kwargs
    ) -> 'Tensor':
        """
        Convert tensor to different dtype/device.

        Args:
            dtype: Target dtype
            device: Target device (ignored in MLX - unified memory)

        Returns:
            New tensor with specified dtype/device
        """
        # Parse positional arguments (can be dtype or device)
        for arg in args:
            if isinstance(arg, (DType, str)):
                if dtype is None:
                    dtype = arg
            elif isinstance(arg, (Device, str)):
                if device is None:
                    device = arg

        # Convert dtype if specified
        if dtype is not None:
            dtype = get_dtype(dtype)
            if dtype != self.dtype:
                new_array = mx.array(self._mlx_array, dtype=dtype._mlx_dtype)
                return Tensor._from_mlx_array(
                    new_array,
                    requires_grad=self.requires_grad,
                )

        # Device conversion is a no-op in MLX (unified memory)
        if device is not None:
            device = Device(device)
            # Just update device attribute for compatibility
            result = Tensor._from_mlx_array(
                self._mlx_array,
                requires_grad=self.requires_grad,
            )
            result._device = device
            return result

        return self

    def type(self, dtype: Union[DType, str]) -> 'Tensor':
        """Convert to specified dtype (alias for .to(dtype=...))."""
        return self.to(dtype=dtype)

    def cpu(self) -> 'Tensor':
        """Move to CPU (no-op in MLX, for compatibility)."""
        return self.to(device='cpu')

    def cuda(self, device: Optional[int] = None) -> 'Tensor':
        """Move to CUDA (no-op in MLX, for compatibility)."""
        dev = f'cuda:{device}' if device is not None else 'cuda'
        return self.to(device=dev)

    def float(self) -> 'Tensor':
        """Convert to float32."""
        from . import dtype as dtype_module
        return self.to(dtype=dtype_module.float32)

    def double(self) -> 'Tensor':
        """Convert to float64 (warns and uses float32 in MLX)."""
        from . import dtype as dtype_module
        return self.to(dtype=dtype_module.float64)

    def half(self) -> 'Tensor':
        """Convert to float16."""
        from . import dtype as dtype_module
        return self.to(dtype=dtype_module.float16)

    def int(self) -> 'Tensor':
        """Convert to int32."""
        from . import dtype as dtype_module
        return self.to(dtype=dtype_module.int32)

    def long(self) -> 'Tensor':
        """Convert to int64."""
        from . import dtype as dtype_module
        return self.to(dtype=dtype_module.int64)

    def bool(self) -> 'Tensor':
        """Convert to bool."""
        from . import dtype as dtype_module
        return self.to(dtype=dtype_module.bool)

    # ==================== String Representation ====================

    def __repr__(self) -> str:
        """String representation."""
        # Get string representation of data
        data_str = str(self.numpy())

        # Build metadata
        meta = []
        if self.requires_grad:
            meta.append("requires_grad=True")
        if self.dtype:
            meta.append(f"dtype={self.dtype}")
        if self._grad_fn is not None:
            meta.append(f"grad_fn=<{self._grad_fn.__class__.__name__}>")

        # Format
        if meta:
            meta_str = ", ".join(meta)
            return f"tensor({data_str}, {meta_str})"
        else:
            return f"tensor({data_str})"

    def __str__(self) -> str:
        """String representation (user-friendly)."""
        return self.__repr__()

    # ==================== Operator Overloading (Arithmetic) ====================
    # These will be fully implemented in Phase 2 with autograd support

    def __add__(self, other):
        """Element-wise addition (x + y)."""
        from .ops.arithmetic import add
        return add(self, other)

    def __radd__(self, other):
        """Reverse addition (y + x where y is not a Tensor)."""
        return self.__add__(other)

    def __sub__(self, other):
        """Element-wise subtraction (x - y)."""
        from .ops.arithmetic import sub
        return sub(self, other)

    def __rsub__(self, other):
        """Reverse subtraction (y - x where y is not a Tensor)."""
        result_array = other - self._mlx_array
        return Tensor._from_mlx_array(result_array)

    def __mul__(self, other):
        """Element-wise multiplication (x * y)."""
        from .ops.arithmetic import mul
        return mul(self, other)

    def __rmul__(self, other):
        """Reverse multiplication (y * x where y is not a Tensor)."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element-wise division (x / y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array / other._mlx_array
        else:
            result_array = self._mlx_array / other
        return Tensor._from_mlx_array(result_array)

    def __rtruediv__(self, other):
        """Reverse division (y / x where y is not a Tensor)."""
        result_array = other / self._mlx_array
        return Tensor._from_mlx_array(result_array)

    def __matmul__(self, other):
        """Matrix multiplication (x @ y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array @ other._mlx_array
        else:
            result_array = self._mlx_array @ other
        return Tensor._from_mlx_array(result_array)

    def __rmatmul__(self, other):
        """Reverse matrix multiplication (y @ x where y is not a Tensor)."""
        result_array = other @ self._mlx_array
        return Tensor._from_mlx_array(result_array)

    def __neg__(self):
        """Negation (-x)."""
        from .ops.arithmetic import neg
        return neg(self)

    def __pow__(self, other):
        """Power operator (x ** y)."""
        from .ops.arithmetic import pow
        return pow(self, other)

    def __rpow__(self, other):
        """Reverse power operator (y ** x where y is not a Tensor)."""
        from .ops.arithmetic import pow
        return pow(other, self)

    # ==================== Comparison Operators ====================

    def __eq__(self, other):
        """Element-wise equality (x == y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array == other._mlx_array
        else:
            result_array = self._mlx_array == other
        return Tensor._from_mlx_array(result_array)

    def __ne__(self, other):
        """Element-wise inequality (x != y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array != other._mlx_array
        else:
            result_array = self._mlx_array != other
        return Tensor._from_mlx_array(result_array)

    def __lt__(self, other):
        """Element-wise less than (x < y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array < other._mlx_array
        else:
            result_array = self._mlx_array < other
        return Tensor._from_mlx_array(result_array)

    def __le__(self, other):
        """Element-wise less than or equal (x <= y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array <= other._mlx_array
        else:
            result_array = self._mlx_array <= other
        return Tensor._from_mlx_array(result_array)

    def __gt__(self, other):
        """Element-wise greater than (x > y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array > other._mlx_array
        else:
            result_array = self._mlx_array > other
        return Tensor._from_mlx_array(result_array)

    def __ge__(self, other):
        """Element-wise greater than or equal (x >= y)."""
        if isinstance(other, Tensor):
            result_array = self._mlx_array >= other._mlx_array
        else:
            result_array = self._mlx_array >= other
        return Tensor._from_mlx_array(result_array)

    # ==================== Indexing ====================

    def __getitem__(self, key):
        """Indexing and slicing."""
        # Handle boolean mask indexing
        if isinstance(key, Tensor):
            from . import dtype as dtype_module
            if key.dtype == dtype_module.bool:
                from .ops.indexing import masked_select
                return masked_select(self, key)

        result_array = self._mlx_array[key]
        result = Tensor._from_mlx_array(result_array, requires_grad=self.requires_grad)

        # Mark as view
        result._is_view = True
        result._base = self if self._base is None else self._base

        return result

    def __setitem__(self, key, value):
        """
        Item assignment (creates new array due to MLX immutability).

        Note: This doesn't modify in-place like PyTorch. It creates a new
        array and reassigns the internal reference.
        """
        if isinstance(value, Tensor):
            value = value._mlx_array

        # MLX arrays are immutable, so we need to create a new array
        # This is a workaround - actual implementation would use mx ops
        arr_np = np.array(self._mlx_array)
        arr_np[key] = np.array(value) if hasattr(value, '__array__') else value
        self._mlx_array = mx.array(arr_np)

    # ==================== View Operations (Instance Methods) ====================

    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor to new shape."""
        from .view_ops import reshape as reshape_fn
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape_fn(self, shape)

    def view(self, *shape: int) -> 'Tensor':
        """View tensor with new shape (alias for reshape)."""
        from .view_ops import view as view_fn
        return view_fn(self, *shape)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        """Transpose two dimensions."""
        from .view_ops import transpose as transpose_fn
        return transpose_fn(self, dim0, dim1)

    def t(self) -> 'Tensor':
        """Transpose (2D tensors only - swaps dimensions 0 and 1)."""
        if self.ndim != 2:
            raise RuntimeError(f"t() expects a 2D tensor, but got {self.ndim}D")
        return self.transpose(0, 1)

    def permute(self, *dims: int) -> 'Tensor':
        """Permute dimensions."""
        from .view_ops import permute as permute_fn
        return permute_fn(self, *dims)

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        from .view_ops import squeeze as squeeze_fn
        return squeeze_fn(self, dim)

    def unsqueeze(self, dim: int) -> 'Tensor':
        """Add a dimension of size 1."""
        from .view_ops import unsqueeze as unsqueeze_fn
        return unsqueeze_fn(self, dim)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        """Flatten dimensions."""
        from .view_ops import flatten as flatten_fn
        return flatten_fn(self, start_dim, end_dim)

    def contiguous(self) -> 'Tensor':
        """Return contiguous tensor (no-op in MLX)."""
        from .view_ops import contiguous as contiguous_fn
        return contiguous_fn(self)

    # ==================== Autograd Methods (Stubs for Phase 3) ====================

    def backward(self, gradient=None):
        """
        Compute gradients (Phase 3 - Autograd).

        Currently a stub that will be implemented in Phase 3.
        """
        raise NotImplementedError(
            "Autograd not yet implemented (Phase 3). "
            "Currently only tensor creation and operations are supported."
        )

    def detach(self) -> 'Tensor':
        """
        Create a new tensor detached from the computation graph.

        Returns a new tensor with the same data but requires_grad=False.
        """
        result = Tensor._from_mlx_array(self._mlx_array)
        result.requires_grad = False
        result._grad_fn = None
        return result

    def requires_grad_(self, requires_grad: bool = True) -> 'Tensor':
        """
        Set requires_grad flag in-place.

        Args:
            requires_grad: Whether to track gradients

        Returns:
            self (modified in-place)
        """
        self.requires_grad = requires_grad
        return self

    def zero_grad(self):
        """Zero out the gradient."""
        self._grad = None

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """
        Compute gradients via backpropagation.

        This method triggers the backward pass from this tensor, computing gradients
        for all tensors in the computation graph that have requires_grad=True.

        Args:
            gradient: Gradient tensor w.r.t. this tensor. If None, assumes scalar output (gradient=1).
            retain_graph: If True, the computation graph will be retained for additional backward passes.
            create_graph: If True, enables higher-order derivative computation.

        Raises:
            RuntimeError: If this tensor doesn't require gradients or if gradient shape doesn't match.

        Example:
            >>> x = mlx_compat.tensor([1.0, 2.0, 3.0], requires_grad=True)
            >>> y = (x ** 2).sum()
            >>> y.backward()
            >>> print(x.grad)  # Should be [2.0, 4.0, 6.0]
        """
        from .autograd.engine import backward as engine_backward
        engine_backward(self, gradient, retain_graph, create_graph)

    def mean(self, dim=None, keepdim=False):
        """
        Compute mean.

        Args:
            dim: Dimension(s) to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Mean tensor.
        """
        from . import ops
        return ops.mean(self, dim=dim, keepdim=keepdim)

    def var(self, dim=None, unbiased=True, keepdim=False):
        """
        Compute variance.

        Args:
            dim: Dimension(s) to reduce. If None, compute over all dimensions.
            unbiased: If True, use Bessel's correction (divide by N-1).
            keepdim: If True, keep reduced dimensions.

        Returns:
            Variance tensor.
        """
        from . import ops
        return ops.var(self, dim=dim, unbiased=unbiased, keepdim=keepdim)

    def std(self, dim=None, unbiased=True, keepdim=False):
        """
        Compute standard deviation.

        Args:
            dim: Dimension(s) to reduce. If None, compute over all dimensions.
            unbiased: If True, use Bessel's correction (divide by N-1).
            keepdim: If True, keep reduced dimensions.

        Returns:
            Standard deviation tensor.
        """
        from . import ops
        return ops.std(self, dim=dim, unbiased=unbiased, keepdim=keepdim)

    def sum(self, dim=None, keepdim=False):
        """
        Sum of elements.

        Args:
            dim: Dimension(s) to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Sum tensor.
        """
        from . import ops
        return ops.sum(self, dim=dim, keepdim=keepdim)

    def prod(self, dim=None, keepdim=False):
        """
        Product of elements.

        Args:
            dim: Dimension to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Product tensor.
        """
        from . import ops
        return ops.prod(self, dim=dim, keepdim=keepdim)

    def max(self, dim=None, keepdim=False):
        """
        Maximum value(s) of tensor.

        Args:
            dim: Dimension to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            If dim is None: single tensor with max value.
            If dim is specified: tuple of (values, indices).
        """
        from . import ops
        return ops.max(self, dim=dim, keepdim=keepdim)

    def min(self, dim=None, keepdim=False):
        """
        Minimum value(s) of tensor.

        Args:
            dim: Dimension to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            If dim is None: single tensor with min value.
            If dim is specified: tuple of (values, indices).
        """
        from . import ops
        return ops.min(self, dim=dim, keepdim=keepdim)

    def argmax(self, dim=None, keepdim=False):
        """
        Indices of maximum values.

        Args:
            dim: Dimension to reduce. If None, flatten first.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Tensor of indices.
        """
        from . import ops
        return ops.argmax(self, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        """
        Indices of minimum values.

        Args:
            dim: Dimension to reduce. If None, flatten first.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Tensor of indices.
        """
        from . import ops
        return ops.argmin(self, dim=dim, keepdim=keepdim)

    def all(self, dim=None, keepdim=False):
        """
        Test if all elements are True.

        Args:
            dim: Dimension to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Boolean tensor.
        """
        from . import ops
        return ops.all(self, dim=dim, keepdim=keepdim)

    def any(self, dim=None, keepdim=False):
        """
        Test if any element is True.

        Args:
            dim: Dimension to reduce. If None, compute over all dimensions.
            keepdim: If True, keep reduced dimensions.

        Returns:
            Boolean tensor.
        """
        from . import ops
        return ops.any(self, dim=dim, keepdim=keepdim)

    def abs(self):
        """
        Compute absolute value element-wise.

        Returns:
            Tensor with absolute values.
        """
        from . import ops
        return ops.abs(self)

    def clamp(self, min=None, max=None):
        """
        Clamp tensor values to a range.

        Args:
            min: Minimum value (None means no lower bound).
            max: Maximum value (None means no upper bound).

        Returns:
            Clamped tensor.
        """
        from . import ops
        return ops.clamp(self, min=min, max=max)

    def zero_(self):
        """
        Zero out this tensor's values in-place.

        Returns:
            self

        Note:
            MLX arrays are immutable, so this creates a new array internally
            but maintains the same tensor object.
        """
        import mlx.core as mx
        self._mlx_array = mx.zeros_like(self._mlx_array)
        return self

    def fill_(self, value):
        """
        Fill this tensor with a scalar value in-place.

        Args:
            value: Scalar value to fill with

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.full(self._mlx_array.shape, value, dtype=self._mlx_array.dtype)
        return self

    def uniform_(self, from_=0.0, to=1.0):
        """
        Fill this tensor with uniformly distributed random values in-place.

        Args:
            from_: Lower bound of uniform distribution
            to: Upper bound of uniform distribution

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.random.uniform(
            low=from_, high=to,
            shape=self._mlx_array.shape,
            dtype=self._mlx_array.dtype
        )
        return self

    def normal_(self, mean=0.0, std=1.0):
        """
        Fill this tensor with normally distributed random values in-place.

        Args:
            mean: Mean of normal distribution
            std: Standard deviation of normal distribution

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.random.normal(
            shape=self._mlx_array.shape,
            dtype=self._mlx_array.dtype
        ) * std + mean
        return self

    def clamp_(self, min=None, max=None):
        """
        Clamp all elements in this tensor to [min, max] in-place.

        Args:
            min: Lower bound. If None, no lower clamping.
            max: Upper bound. If None, no upper clamping.

        Returns:
            self
        """
        import mlx.core as mx
        result = self._mlx_array
        if min is not None:
            result = mx.maximum(result, min)
        if max is not None:
            result = mx.minimum(result, max)
        self._mlx_array = result
        return self

    def relu_(self):
        """
        Apply ReLU activation in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.maximum(self._mlx_array, 0)
        return self

    def sigmoid_(self):
        """
        Apply sigmoid activation in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.sigmoid(self._mlx_array)
        return self

    def tanh_(self):
        """
        Apply tanh activation in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.tanh(self._mlx_array)
        return self

    def exp_(self):
        """
        Apply exponential function in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.exp(self._mlx_array)
        return self

    def log_(self):
        """
        Apply natural logarithm in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.log(self._mlx_array)
        return self

    def sqrt_(self):
        """
        Apply square root in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.sqrt(self._mlx_array)
        return self

    def abs_(self):
        """
        Apply absolute value in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.abs(self._mlx_array)
        return self

    def neg_(self):
        """
        Negate values in-place.

        Returns:
            self
        """
        self._mlx_array = -self._mlx_array
        return self

    def add_(self, other, alpha=1):
        """
        Add a tensor or scalar to this tensor in-place.

        Args:
            other: Tensor or scalar to add
            alpha: Multiplier for other

        Returns:
            self
        """
        if isinstance(other, Tensor):
            other_array = other._mlx_array
        else:
            other_array = other
        self._mlx_array = self._mlx_array + alpha * other_array
        return self

    def sub_(self, other, alpha=1):
        """
        Subtract a tensor or scalar from this tensor in-place.

        Args:
            other: Tensor or scalar to subtract
            alpha: Multiplier for other

        Returns:
            self
        """
        if isinstance(other, Tensor):
            other_array = other._mlx_array
        else:
            other_array = other
        self._mlx_array = self._mlx_array - alpha * other_array
        return self

    def mul_(self, other):
        """
        Multiply this tensor by another tensor or scalar in-place.

        Args:
            other: Tensor or scalar to multiply by

        Returns:
            self
        """
        if isinstance(other, Tensor):
            other_array = other._mlx_array
        else:
            other_array = other
        self._mlx_array = self._mlx_array * other_array
        return self

    def div_(self, other):
        """
        Divide this tensor by another tensor or scalar in-place.

        Args:
            other: Tensor or scalar to divide by

        Returns:
            self
        """
        if isinstance(other, Tensor):
            other_array = other._mlx_array
        else:
            other_array = other
        self._mlx_array = self._mlx_array / other_array
        return self

    def pow_(self, exponent):
        """
        Raise this tensor to a power in-place.

        Args:
            exponent: Exponent value

        Returns:
            self
        """
        import mlx.core as mx
        if isinstance(exponent, Tensor):
            exponent = exponent._mlx_array
        self._mlx_array = mx.power(self._mlx_array, exponent)
        return self

    def copy_(self, src, non_blocking=False):
        """
        Copy elements from source tensor in-place.

        Args:
            src: Source tensor
            non_blocking: Ignored (for API compatibility)

        Returns:
            self
        """
        if isinstance(src, Tensor):
            self._mlx_array = src._mlx_array
        else:
            import mlx.core as mx
            self._mlx_array = mx.array(src, dtype=self._mlx_array.dtype)
        return self

    def floor_(self):
        """
        Apply floor function in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.floor(self._mlx_array)
        return self

    def ceil_(self):
        """
        Apply ceiling function in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.ceil(self._mlx_array)
        return self

    def round_(self):
        """
        Round values in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.round(self._mlx_array)
        return self

    def sin_(self):
        """
        Apply sine function in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.sin(self._mlx_array)
        return self

    def cos_(self):
        """
        Apply cosine function in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.cos(self._mlx_array)
        return self

    def erf_(self):
        """
        Apply error function in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.erf(self._mlx_array)
        return self

    def erfc_(self):
        """
        Apply complementary error function in-place.

        erfc(x) = 1 - erf(x)

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = 1 - mx.erf(self._mlx_array)
        return self

    def exp2_(self):
        """
        Apply base-2 exponential in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.power(mx.array(2.0, dtype=self._mlx_array.dtype), self._mlx_array)
        return self

    def reciprocal_(self):
        """
        Apply reciprocal (1/x) in-place.

        Returns:
            self
        """
        self._mlx_array = 1.0 / self._mlx_array
        return self

    def rsqrt_(self):
        """
        Apply reciprocal square root (1/sqrt(x)) in-place.

        Returns:
            self
        """
        import mlx.core as mx
        self._mlx_array = mx.rsqrt(self._mlx_array)
        return self


__all__ = ['Tensor']
