"""
Quantized Tensor

Represents a tensor stored in quantized format using MLX's quantization.
"""

from typing import Optional, Tuple

import mlx.core as mx

from ..tensor import Tensor


class QuantizedTensor:
    """
    Tensor with quantized storage.

    MLX uses group-wise quantization with configurable bit width and group size.
    This class wraps the quantized representation and provides dequantization.

    Attributes:
        _data: The quantized integer data
        _scales: Per-group scale factors
        _biases: Per-group bias values (zero points)
        _dtype: Original dtype before quantization
        _shape: Original tensor shape
        _bits: Number of bits used for quantization (2, 4, or 8)
        _group_size: Number of elements per quantization group

    Example:
        >>> x = flashlight.randn(128, 256)
        >>> qx = flashlight.quantization.quantize_per_tensor(x, bits=4)
        >>> x_restored = qx.dequantize()
    """

    def __init__(
        self,
        data: mx.array,
        scales: mx.array,
        biases: mx.array,
        shape: Tuple[int, ...],
        dtype: "DType",
        bits: int = 4,
        group_size: int = 64,
    ):
        """
        Initialize a QuantizedTensor.

        Args:
            data: Quantized integer data
            scales: Per-group scale factors
            biases: Per-group bias values
            shape: Original tensor shape
            dtype: Original data type
            bits: Bits per element (2, 4, or 8)
            group_size: Elements per quantization group
        """
        self._data = data
        self._scales = scales
        self._biases = biases
        self._shape = shape
        self._dtype = dtype
        self._bits = bits
        self._group_size = group_size

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the original tensor shape."""
        return self._shape

    @property
    def dtype(self) -> "DType":
        """Return the original data type."""
        return self._dtype

    @property
    def bits(self) -> int:
        """Return the number of bits used for quantization."""
        return self._bits

    @property
    def group_size(self) -> int:
        """Return the quantization group size."""
        return self._group_size

    @property
    def int_repr(self) -> Tensor:
        """Return the integer representation of quantized data."""
        return Tensor._from_mlx_array(self._data)

    def dequantize(self) -> Tensor:
        """
        Convert back to a regular tensor.

        Returns:
            Tensor with dequantized values
        """
        # MLX dequantize signature: dequantize(w, scales, biases, group_size, bits)
        arr = mx.dequantize(
            self._data,
            self._scales,
            self._biases,
            self._group_size,
            self._bits,
        )
        return Tensor._from_mlx_array(arr.astype(self._dtype._mlx_dtype))

    def to(self, dtype: Optional["DType"] = None, device: Optional[str] = None) -> "QuantizedTensor":
        """
        Convert quantized tensor to different dtype or device.

        Note: The quantized data stays in its integer format.
        Only the target dtype for dequantization changes.
        """
        if dtype is None:
            dtype = self._dtype
        return QuantizedTensor(
            self._data,
            self._scales,
            self._biases,
            self._shape,
            dtype,
            self._bits,
            self._group_size,
        )

    def __repr__(self) -> str:
        return (
            f"QuantizedTensor(shape={self._shape}, dtype={self._dtype}, "
            f"bits={self._bits}, group_size={self._group_size})"
        )

    # Arithmetic operations that dequantize first
    def __add__(self, other):
        return self.dequantize() + other

    def __radd__(self, other):
        return other + self.dequantize()

    def __sub__(self, other):
        return self.dequantize() - other

    def __rsub__(self, other):
        return other - self.dequantize()

    def __mul__(self, other):
        return self.dequantize() * other

    def __rmul__(self, other):
        return other * self.dequantize()

    def __matmul__(self, other):
        """
        Matrix multiplication with quantized tensor.

        Uses MLX's quantized_matmul for efficient computation when possible.
        """
        if isinstance(other, Tensor):
            # Use quantized matmul: quantized_matmul(x, w, scales, biases, ...)
            # This computes x @ dequantize(w) efficiently
            result = mx.quantized_matmul(
                other._mlx_array,
                self._data,
                self._scales,
                self._biases,
                transpose=True,
                group_size=self._group_size,
                bits=self._bits,
            )
            return Tensor._from_mlx_array(result)
        return self.dequantize() @ other

    def __rmatmul__(self, other):
        """Matrix multiplication with quantized tensor on right side."""
        if isinstance(other, Tensor):
            # x @ quantized_weight
            result = mx.quantized_matmul(
                other._mlx_array,
                self._data,
                self._scales,
                self._biases,
                transpose=False,
                group_size=self._group_size,
                bits=self._bits,
            )
            return Tensor._from_mlx_array(result)
        return other @ self.dequantize()
