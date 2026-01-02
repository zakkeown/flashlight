"""
Sparse Tensor Classes

Implements COO and CSR sparse tensor formats with PyTorch-compatible API.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from ..dtype import DType, float32, int64


class SparseTensor:
    """
    Base class for sparse tensors.

    Sparse tensors store only non-zero elements, making them efficient
    for matrices with many zeros (e.g., graph adjacency matrices).
    """

    @property
    def is_sparse(self) -> bool:
        """Return True for sparse tensors."""
        return True

    @property
    def nnz(self) -> int:
        """Return the number of non-zero elements."""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the dense shape of the tensor."""
        raise NotImplementedError

    @property
    def dtype(self) -> DType:
        """Return the data type of the values."""
        raise NotImplementedError

    def to_dense(self) -> Tensor:
        """Convert to a dense tensor."""
        raise NotImplementedError

    def coalesce(self) -> "SparseTensor":
        """Sum duplicate indices and sort."""
        raise NotImplementedError

    def is_coalesced(self) -> bool:
        """Return True if the tensor is coalesced."""
        raise NotImplementedError


class SparseCOOTensor(SparseTensor):
    """
    Coordinate format (COO) sparse tensor.

    Storage format:
    - indices: [sparse_dim, nnz] int64 tensor of coordinates
    - values: [nnz, *dense_dims] tensor of non-zero values
    - size: tuple representing the full dense shape

    This is the most flexible format for construction and modification.

    Example:
        >>> indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]])  # 3 non-zeros
        >>> values = flashlight.tensor([1.0, 2.0, 3.0])
        >>> sparse = SparseCOOTensor(indices, values, size=(3, 3))
    """

    def __init__(
        self,
        indices: Tensor,
        values: Tensor,
        size: Tuple[int, ...],
        is_coalesced: bool = False,
    ):
        """
        Create a COO sparse tensor.

        Args:
            indices: [sparse_dim, nnz] tensor of coordinates
            values: [nnz, ...] tensor of values at those coordinates
            size: The full dense shape
            is_coalesced: Whether indices are sorted and unique
        """
        self._indices = indices
        self._values = values
        self._size = size
        self._coalesced = is_coalesced

    @property
    def indices(self) -> Tensor:
        """
        Return the indices tensor.

        For uncoalesced tensors, this may contain duplicates.
        """
        return self._indices

    @property
    def values(self) -> Tensor:
        """Return the values tensor."""
        return self._values

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the dense shape."""
        return self._size

    @property
    def size(self) -> Tuple[int, ...]:
        """Return the dense shape (alias for shape)."""
        return self._size

    @property
    def nnz(self) -> int:
        """Return the number of stored elements (may include duplicates if uncoalesced)."""
        return self._values.shape[0]

    @property
    def dtype(self) -> DType:
        """Return the data type of values."""
        from ..dtype import get_dtype
        return get_dtype(self._values._mlx_array.dtype)

    @property
    def sparse_dim(self) -> int:
        """Return the number of sparse dimensions."""
        return self._indices.shape[0]

    @property
    def dense_dim(self) -> int:
        """Return the number of dense dimensions in values."""
        return self._values.ndim - 1

    def is_coalesced(self) -> bool:
        """Return True if indices are sorted and unique."""
        return self._coalesced

    def coalesce(self) -> "SparseCOOTensor":
        """
        Sum duplicate indices and sort.

        Returns a new coalesced tensor where indices are sorted in
        lexicographic order and duplicates are summed.
        """
        if self._coalesced:
            return self

        indices_arr = self._indices._mlx_array
        values_arr = self._values._mlx_array

        # Compute linear indices for sorting
        # For 2D: linear_idx = row * num_cols + col
        if len(self._size) == 2:
            linear_idx = indices_arr[0] * self._size[1] + indices_arr[1]
        else:
            # General case: compute linear index
            linear_idx = mx.zeros((self.nnz,), dtype=mx.int64)
            stride = 1
            for dim in reversed(range(len(self._size))):
                linear_idx = linear_idx + indices_arr[dim] * stride
                stride *= self._size[dim]

        # Sort by linear index
        sort_order = mx.argsort(linear_idx)
        sorted_indices = indices_arr[:, sort_order]
        sorted_values = values_arr[sort_order]
        sorted_linear = linear_idx[sort_order]

        # Find unique indices and sum duplicates
        # Create mask for first occurrence of each unique index
        n = self.nnz
        if n == 0:
            return SparseCOOTensor(
                self._indices, self._values, self._size, is_coalesced=True
            )

        # Detect boundaries where linear index changes
        is_first = mx.concatenate([
            mx.array([True]),
            sorted_linear[1:] != sorted_linear[:-1]
        ])

        # Create segment IDs using cumsum
        segment_ids = mx.cumsum(is_first.astype(mx.int32)) - 1

        # Count unique elements
        unique_count = int(mx.sum(is_first.astype(mx.int32)))

        # Get indices of first occurrences by finding where is_first is True
        # MLX doesn't have single-arg where(), so we build indices manually
        is_first_list = is_first.tolist()
        first_indices_list = [i for i, x in enumerate(is_first_list) if x]
        first_indices = mx.array(first_indices_list, dtype=mx.int64)

        # Extract unique indices
        new_indices = sorted_indices[:, first_indices]

        # Sum values for duplicates using segment sum
        # Simple Python loop for segment sum (could be optimized with Metal kernel)
        sorted_values_list = sorted_values.tolist()
        segment_ids_list = segment_ids.tolist()
        new_values_list = [0.0] * unique_count

        for i in range(n):
            seg_id = int(segment_ids_list[i])
            new_values_list[seg_id] += sorted_values_list[i]

        new_values = mx.array(new_values_list, dtype=sorted_values.dtype)

        return SparseCOOTensor(
            Tensor._from_mlx_array(new_indices),
            Tensor._from_mlx_array(new_values),
            self._size,
            is_coalesced=True,
        )

    def to_dense(self) -> Tensor:
        """Convert to a dense tensor."""
        # Create zero tensor
        dense = mx.zeros(self._size, dtype=self._values._mlx_array.dtype)

        # Handle empty tensor case
        if self.nnz == 0:
            return Tensor._from_mlx_array(dense)

        # Scatter values to indices
        indices_arr = self._indices._mlx_array
        values_arr = self._values._mlx_array

        # For 2D case
        if len(self._size) == 2:
            rows = indices_arr[0].tolist()
            cols = indices_arr[1].tolist()
            vals = values_arr.tolist()

            dense_list = dense.tolist()
            for i in range(self.nnz):
                r, c = int(rows[i]), int(cols[i])
                dense_list[r][c] += vals[i]

            dense = mx.array(dense_list, dtype=self._values._mlx_array.dtype)
        else:
            # General N-D case
            dense_flat = dense.flatten().tolist()
            indices_list = [indices_arr[d].tolist() for d in range(len(self._size))]
            vals = values_arr.tolist()

            for i in range(self.nnz):
                # Compute flat index
                flat_idx = 0
                stride = 1
                for d in reversed(range(len(self._size))):
                    flat_idx += int(indices_list[d][i]) * stride
                    stride *= self._size[d]
                dense_flat[flat_idx] += vals[i]

            dense = mx.array(dense_flat, dtype=self._values._mlx_array.dtype).reshape(self._size)

        return Tensor._from_mlx_array(dense)

    def to_sparse_csr(self) -> "SparseCSRTensor":
        """Convert to CSR format (2D only)."""
        if len(self._size) != 2:
            raise ValueError("CSR format only supports 2D tensors")

        coalesced = self.coalesce()

        indices_arr = coalesced._indices._mlx_array
        values_arr = coalesced._values._mlx_array

        rows = indices_arr[0]
        cols = indices_arr[1]

        nrows = self._size[0]

        # Build crow_indices (row pointers) using Python list
        # MLX doesn't support int64 scatter, so we build in Python
        row_counts = [0] * (nrows + 1)
        for r in rows.tolist():
            row_counts[int(r) + 1] += 1

        # Cumulative sum
        for i in range(1, nrows + 1):
            row_counts[i] += row_counts[i - 1]

        crow_indices = mx.array(row_counts, dtype=mx.int64)

        return SparseCSRTensor(
            Tensor._from_mlx_array(crow_indices),
            Tensor._from_mlx_array(cols.astype(mx.int64)),
            coalesced._values,
            self._size,
        )

    def __repr__(self) -> str:
        return (
            f"SparseCOOTensor(indices={self._indices}, values={self._values}, "
            f"size={self._size}, nnz={self.nnz}, coalesced={self._coalesced})"
        )


class SparseCSRTensor(SparseTensor):
    """
    Compressed Sparse Row (CSR) format tensor.

    Storage format:
    - crow_indices: [nrows + 1] row pointers
    - col_indices: [nnz] column indices
    - values: [nnz] non-zero values
    - size: (nrows, ncols)

    CSR is efficient for row-slicing and matrix-vector multiplication.

    Example:
        >>> crow_indices = flashlight.tensor([0, 2, 3, 5])  # 3 rows
        >>> col_indices = flashlight.tensor([0, 2, 1, 0, 2])  # 5 non-zeros
        >>> values = flashlight.tensor([1., 2., 3., 4., 5.])
        >>> sparse = SparseCSRTensor(crow_indices, col_indices, values, (3, 3))
    """

    def __init__(
        self,
        crow_indices: Tensor,
        col_indices: Tensor,
        values: Tensor,
        size: Tuple[int, int],
    ):
        """
        Create a CSR sparse tensor.

        Args:
            crow_indices: [nrows + 1] row pointer tensor
            col_indices: [nnz] column index tensor
            values: [nnz] value tensor
            size: (nrows, ncols) tuple
        """
        self._crow_indices = crow_indices
        self._col_indices = col_indices
        self._values = values
        self._size = size

    @property
    def crow_indices(self) -> Tensor:
        """Return the row pointer tensor."""
        return self._crow_indices

    @property
    def col_indices(self) -> Tensor:
        """Return the column indices tensor."""
        return self._col_indices

    @property
    def values(self) -> Tensor:
        """Return the values tensor."""
        return self._values

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the dense shape."""
        return self._size

    @property
    def size(self) -> Tuple[int, int]:
        """Return the dense shape (alias for shape)."""
        return self._size

    @property
    def nnz(self) -> int:
        """Return the number of non-zero elements."""
        return self._values.shape[0]

    @property
    def dtype(self) -> DType:
        """Return the data type of values."""
        from ..dtype import get_dtype
        return get_dtype(self._values._mlx_array.dtype)

    def is_coalesced(self) -> bool:
        """CSR format is always coalesced."""
        return True

    def coalesce(self) -> "SparseCSRTensor":
        """CSR is already coalesced."""
        return self

    def to_dense(self) -> Tensor:
        """Convert to a dense tensor."""
        nrows, ncols = self._size
        dense = mx.zeros(self._size, dtype=self._values._mlx_array.dtype)

        crow = self._crow_indices._mlx_array.tolist()
        cols = self._col_indices._mlx_array.tolist()
        vals = self._values._mlx_array.tolist()

        dense_list = [[0.0] * ncols for _ in range(nrows)]

        for row in range(nrows):
            start, end = int(crow[row]), int(crow[row + 1])
            for idx in range(start, end):
                col = int(cols[idx])
                dense_list[row][col] += vals[idx]

        dense = mx.array(dense_list, dtype=self._values._mlx_array.dtype)
        return Tensor._from_mlx_array(dense)

    def to_sparse_coo(self) -> SparseCOOTensor:
        """Convert to COO format."""
        nrows = self._size[0]
        crow = self._crow_indices._mlx_array.tolist()
        cols = self._col_indices._mlx_array

        # Build row indices from crow_indices
        row_indices = []
        for row in range(nrows):
            start, end = int(crow[row]), int(crow[row + 1])
            row_indices.extend([row] * (end - start))

        row_indices = mx.array(row_indices, dtype=mx.int64)
        indices = mx.stack([row_indices, cols.astype(mx.int64)], axis=0)

        return SparseCOOTensor(
            Tensor._from_mlx_array(indices),
            self._values,
            self._size,
            is_coalesced=True,
        )

    def __repr__(self) -> str:
        return (
            f"SparseCSRTensor(crow_indices={self._crow_indices}, "
            f"col_indices={self._col_indices}, values={self._values}, "
            f"size={self._size}, nnz={self.nnz})"
        )
