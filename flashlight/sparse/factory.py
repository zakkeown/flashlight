"""
Sparse Tensor Factory Functions

Provides PyTorch-compatible functions for creating sparse tensors.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from ..dtype import DType, float32, int64
from .tensor import SparseCOOTensor, SparseCSRTensor


def sparse_coo_tensor(
    indices: Tensor,
    values: Tensor,
    size: Optional[Tuple[int, ...]] = None,
    dtype: Optional[DType] = None,
    device: Optional[str] = None,
    requires_grad: bool = False,
    is_coalesced: bool = False,
) -> SparseCOOTensor:
    """
    Create a sparse COO tensor.

    Args:
        indices: [sparse_dim, nnz] tensor of coordinates
        values: [nnz, ...] tensor of values
        size: Optional shape tuple. If None, inferred from max indices + 1.
        dtype: Optional data type for values
        device: Ignored (MLX uses unified memory)
        requires_grad: Whether to track gradients (not yet supported for sparse)
        is_coalesced: Whether indices are already sorted and unique

    Returns:
        SparseCOOTensor

    Example:
        >>> indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]])
        >>> values = flashlight.tensor([1.0, 2.0, 3.0])
        >>> sparse = flashlight.sparse_coo_tensor(indices, values, (3, 3))
    """
    if requires_grad:
        import warnings
        warnings.warn(
            "requires_grad for sparse tensors is not yet supported",
            UserWarning,
        )

    # Cast values to specified dtype if needed
    if dtype is not None:
        values = Tensor._from_mlx_array(
            values._mlx_array.astype(dtype._mlx_dtype)
        )

    # Infer size from indices if not provided
    if size is None:
        indices_arr = indices._mlx_array
        sparse_dim = indices_arr.shape[0]
        max_indices = [int(mx.max(indices_arr[d])) + 1 for d in range(sparse_dim)]
        size = tuple(max_indices)

    return SparseCOOTensor(indices, values, size, is_coalesced=is_coalesced)


def sparse_csr_tensor(
    crow_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    size: Optional[Tuple[int, int]] = None,
    dtype: Optional[DType] = None,
    device: Optional[str] = None,
    requires_grad: bool = False,
) -> SparseCSRTensor:
    """
    Create a sparse CSR tensor.

    Args:
        crow_indices: [nrows + 1] row pointer tensor
        col_indices: [nnz] column index tensor
        values: [nnz] value tensor
        size: Optional (nrows, ncols) shape. If None, inferred from indices.
        dtype: Optional data type for values
        device: Ignored (MLX uses unified memory)
        requires_grad: Whether to track gradients (not yet supported for sparse)

    Returns:
        SparseCSRTensor

    Example:
        >>> crow_indices = flashlight.tensor([0, 2, 3, 5])
        >>> col_indices = flashlight.tensor([0, 2, 1, 0, 2])
        >>> values = flashlight.tensor([1., 2., 3., 4., 5.])
        >>> sparse = flashlight.sparse_csr_tensor(crow_indices, col_indices, values, (3, 3))
    """
    if requires_grad:
        import warnings
        warnings.warn(
            "requires_grad for sparse tensors is not yet supported",
            UserWarning,
        )

    # Cast values to specified dtype if needed
    if dtype is not None:
        values = Tensor._from_mlx_array(
            values._mlx_array.astype(dtype._mlx_dtype)
        )

    # Infer size if not provided
    if size is None:
        nrows = crow_indices.shape[0] - 1
        ncols = int(mx.max(col_indices._mlx_array)) + 1
        size = (nrows, ncols)

    return SparseCSRTensor(crow_indices, col_indices, values, size)


def _dense_to_sparse_coo(tensor: Tensor, layout: str = "coo") -> SparseCOOTensor:
    """
    Convert a dense tensor to sparse COO format.

    Args:
        tensor: Dense tensor to convert
        layout: Sparse layout (only "coo" supported)

    Returns:
        SparseCOOTensor with non-zero elements
    """
    arr = tensor._mlx_array
    shape = tensor.shape

    # Find non-zero indices - MLX doesn't have single-arg where()
    # So we iterate through the tensor in Python
    if len(shape) == 2:
        # 2D case - optimized
        arr_list = arr.tolist()
        row_indices = []
        col_indices = []
        values_list = []

        for i, row in enumerate(arr_list):
            for j, val in enumerate(row):
                if val != 0:
                    row_indices.append(i)
                    col_indices.append(j)
                    values_list.append(val)

        if len(values_list) == 0:
            # Handle empty case
            indices = mx.zeros((2, 0), dtype=mx.int64)
            values = mx.array([], dtype=arr.dtype)
        else:
            indices = mx.array([row_indices, col_indices], dtype=mx.int64)
            values = mx.array(values_list, dtype=arr.dtype)
    else:
        # General N-D case
        flat = arr.flatten().tolist()
        flat_indices = [i for i, v in enumerate(flat) if v != 0]
        values_list = [flat[i] for i in flat_indices]

        if len(flat_indices) == 0:
            indices = mx.zeros((len(shape), 0), dtype=mx.int64)
            values = mx.array([], dtype=arr.dtype)
        else:
            # Convert flat indices to N-D indices
            nd_indices = []
            for d in range(len(shape)):
                dim_indices = []
                for flat_idx in flat_indices:
                    # Compute index for this dimension
                    remainder = flat_idx
                    for dim_after in range(d + 1, len(shape)):
                        stride = 1
                        for s in range(dim_after, len(shape)):
                            stride *= shape[s]
                        remainder = remainder % (stride * shape[dim_after - 1] if dim_after > 0 else flat_idx + 1)

                    stride = 1
                    for s in range(d + 1, len(shape)):
                        stride *= shape[s]
                    dim_indices.append(flat_idx // stride % shape[d])
                nd_indices.append(dim_indices)

            indices = mx.array(nd_indices, dtype=mx.int64)
            values = mx.array(values_list, dtype=arr.dtype)

    return SparseCOOTensor(
        Tensor._from_mlx_array(indices),
        Tensor._from_mlx_array(values),
        shape,
        is_coalesced=True,  # from dense is always sorted
    )
