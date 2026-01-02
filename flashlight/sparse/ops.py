"""
Sparse Tensor Operations

Provides operations for sparse tensors including SpMM (sparse-dense multiplication),
sparse addition, and sparse reduction operations.
"""

from typing import Union

import mlx.core as mx

from ..tensor import Tensor
from .tensor import SparseCOOTensor, SparseCSRTensor, SparseTensor


def sparse_mm(
    sparse: SparseTensor,
    dense: Tensor,
) -> Tensor:
    """
    Sparse matrix-dense matrix multiplication.

    Computes sparse @ dense where sparse is a 2D sparse tensor
    and dense is a 2D dense tensor.

    Args:
        sparse: 2D sparse tensor (COO or CSR format)
        dense: 2D dense tensor

    Returns:
        Dense tensor result of sparse @ dense

    Example:
        >>> # Create sparse identity matrix
        >>> indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]])
        >>> values = flashlight.tensor([1.0, 1.0, 1.0])
        >>> sparse = flashlight.sparse_coo_tensor(indices, values, (3, 3))
        >>> dense = flashlight.randn(3, 4)
        >>> result = flashlight.sparse_mm(sparse, dense)  # Same as dense
    """
    if isinstance(sparse, SparseCSRTensor):
        return _spmm_csr(sparse, dense)
    elif isinstance(sparse, SparseCOOTensor):
        # Convert to CSR for efficient multiplication
        return _spmm_csr(sparse.to_sparse_csr(), dense)
    else:
        raise TypeError(f"Unsupported sparse type: {type(sparse)}")


def _spmm_csr(sparse: SparseCSRTensor, dense: Tensor) -> Tensor:
    """
    CSR sparse matrix-dense matrix multiplication.

    This is an optimized implementation for CSR format.
    For very large matrices, a Metal kernel would be faster.
    """
    nrows, ncols_a = sparse.shape
    ncols_b = dense.shape[1]

    crow = sparse._crow_indices._mlx_array.tolist()
    cols = sparse._col_indices._mlx_array.tolist()
    vals = sparse._values._mlx_array.tolist()
    dense_arr = dense._mlx_array

    # Output matrix
    result = [[0.0] * ncols_b for _ in range(nrows)]

    for row in range(nrows):
        start, end = int(crow[row]), int(crow[row + 1])
        for idx in range(start, end):
            col = int(cols[idx])
            val = vals[idx]
            # Add contribution: result[row, :] += val * dense[col, :]
            for j in range(ncols_b):
                result[row][j] += val * float(dense_arr[col, j])

    return Tensor._from_mlx_array(
        mx.array(result, dtype=dense._mlx_array.dtype)
    )


def sparse_mv(sparse: SparseTensor, vector: Tensor) -> Tensor:
    """
    Sparse matrix-vector multiplication.

    Computes sparse @ vector where sparse is a 2D sparse tensor
    and vector is a 1D tensor.

    Args:
        sparse: 2D sparse tensor
        vector: 1D tensor

    Returns:
        1D tensor result
    """
    # Reshape vector to 2D, multiply, reshape back
    result = sparse_mm(sparse, vector.unsqueeze(-1))
    return result.squeeze(-1)


def sparse_add(
    a: SparseTensor,
    b: SparseTensor,
) -> SparseCOOTensor:
    """
    Element-wise addition of two sparse tensors.

    Args:
        a: First sparse tensor
        b: Second sparse tensor (same shape as a)

    Returns:
        SparseCOOTensor with the sum

    Note:
        Both tensors must have the same shape.
        The result is in COO format and may need coalescing.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shapes must match for sparse addition: {a.shape} vs {b.shape}"
        )

    # Convert both to COO
    a_coo = a if isinstance(a, SparseCOOTensor) else a.to_sparse_coo()
    b_coo = b if isinstance(b, SparseCOOTensor) else b.to_sparse_coo()

    # Concatenate indices and values
    new_indices = Tensor._from_mlx_array(
        mx.concatenate([
            a_coo._indices._mlx_array,
            b_coo._indices._mlx_array,
        ], axis=1)
    )

    new_values = Tensor._from_mlx_array(
        mx.concatenate([
            a_coo._values._mlx_array,
            b_coo._values._mlx_array,
        ], axis=0)
    )

    # Return uncoalesced tensor (user can coalesce if needed)
    return SparseCOOTensor(
        new_indices, new_values, a.shape, is_coalesced=False
    )


def sparse_sum(sparse: SparseTensor, dim: int = None) -> Union[Tensor, "SparseTensor"]:
    """
    Sum over sparse tensor.

    Args:
        sparse: Sparse tensor to sum
        dim: Dimension to sum over. If None, sum all elements.

    Returns:
        Summed result (dense Tensor if dim is None)
    """
    if dim is None:
        # Sum all elements
        if isinstance(sparse, SparseCOOTensor):
            coalesced = sparse.coalesce()
            total = mx.sum(coalesced._values._mlx_array)
            return Tensor._from_mlx_array(total)
        else:
            # CSR
            total = mx.sum(sparse._values._mlx_array)
            return Tensor._from_mlx_array(total)

    # Sum along a dimension
    # For now, convert to dense and sum
    dense = sparse.to_dense()
    return dense.sum(dim=dim)


def sparse_t(sparse: SparseTensor) -> SparseTensor:
    """
    Transpose a 2D sparse tensor.

    Args:
        sparse: 2D sparse tensor

    Returns:
        Transposed sparse tensor
    """
    if len(sparse.shape) != 2:
        raise ValueError("Transpose only supported for 2D sparse tensors")

    if isinstance(sparse, SparseCOOTensor):
        # Swap row and column indices
        indices = sparse._indices._mlx_array
        new_indices = mx.stack([indices[1], indices[0]], axis=0)

        return SparseCOOTensor(
            Tensor._from_mlx_array(new_indices),
            sparse._values,
            (sparse.shape[1], sparse.shape[0]),
            is_coalesced=False,  # May not be sorted after transpose
        )

    elif isinstance(sparse, SparseCSRTensor):
        # Convert to COO, transpose, convert back to CSR
        coo = sparse.to_sparse_coo()
        transposed_coo = sparse_t(coo)
        return transposed_coo.to_sparse_csr()

    else:
        raise TypeError(f"Unsupported sparse type: {type(sparse)}")


def sparse_to_dense(sparse: SparseTensor) -> Tensor:
    """
    Convert a sparse tensor to dense.

    This is just an alias for sparse.to_dense().

    Args:
        sparse: Sparse tensor

    Returns:
        Dense tensor
    """
    return sparse.to_dense()


def dense_to_sparse(
    tensor: Tensor,
    layout: str = "coo",
) -> SparseTensor:
    """
    Convert a dense tensor to sparse format.

    Args:
        tensor: Dense tensor
        layout: Sparse format ("coo" or "csr")

    Returns:
        Sparse tensor containing only non-zero elements
    """
    from .factory import _dense_to_sparse_coo

    coo = _dense_to_sparse_coo(tensor)

    if layout == "coo":
        return coo
    elif layout == "csr":
        if len(tensor.shape) != 2:
            raise ValueError("CSR format only supports 2D tensors")
        return coo.to_sparse_csr()
    else:
        raise ValueError(f"Unknown sparse layout: {layout}")
