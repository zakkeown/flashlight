"""
Sparse Tensor Module

Provides PyTorch-compatible sparse tensor support for MLX.
Supports COO (Coordinate) and CSR (Compressed Sparse Row) formats.
"""

from .tensor import SparseCOOTensor, SparseCSRTensor, SparseTensor
from .factory import sparse_coo_tensor, sparse_csr_tensor
from .ops import (
    sparse_mm,
    sparse_mv,
    sparse_add,
    sparse_sum,
    sparse_t,
    sparse_to_dense,
    dense_to_sparse,
)

__all__ = [
    # Tensor classes
    "SparseTensor",
    "SparseCOOTensor",
    "SparseCSRTensor",
    # Factory functions
    "sparse_coo_tensor",
    "sparse_csr_tensor",
    # Operations
    "sparse_mm",
    "sparse_mv",
    "sparse_add",
    "sparse_sum",
    "sparse_t",
    "sparse_to_dense",
    "dense_to_sparse",
]
