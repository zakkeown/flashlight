"""
Tests for Sparse Tensor operations.

Tests cover:
- SparseCOOTensor creation and operations
- SparseCSRTensor creation and operations
- Sparse-dense multiplication (SpMM)
- Sparse tensor addition
- Conversion between formats
"""

import pytest

import flashlight
from flashlight.sparse import (
    SparseCOOTensor,
    SparseCSRTensor,
    sparse_coo_tensor,
    sparse_csr_tensor,
    sparse_mm,
    sparse_add,
    sparse_sum,
    sparse_t,
    sparse_to_dense,
    dense_to_sparse,
)


class TestSparseCOOTensor:
    """Tests for COO format sparse tensors."""

    def test_coo_creation_basic(self):
        """Test basic COO tensor creation."""
        indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_coo_tensor(indices, values, size=(3, 3))

        assert isinstance(sparse, SparseCOOTensor)
        assert sparse.shape == (3, 3)
        assert sparse.nnz == 3

    def test_coo_to_dense(self):
        """Test COO to dense conversion."""
        indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_coo_tensor(indices, values, size=(3, 3))

        dense = sparse.to_dense()
        expected = flashlight.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])

        assert (dense == expected).all().item()

    def test_coo_with_duplicates(self):
        """Test COO tensor with duplicate indices (sum them on coalesce)."""
        indices = flashlight.tensor([[0, 0, 1], [0, 0, 1]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])  # Two values at (0,0)
        sparse = sparse_coo_tensor(indices, values, size=(2, 2), is_coalesced=False)

        coalesced = sparse.coalesce()
        assert coalesced.nnz == 2  # Two unique positions

        dense = coalesced.to_dense()
        # (0,0) should have 1.0 + 2.0 = 3.0
        assert dense[0, 0].item() == pytest.approx(3.0)
        assert dense[1, 1].item() == pytest.approx(3.0)

    def test_coo_coalesce_already_coalesced(self):
        """Test coalesce on already coalesced tensor."""
        indices = flashlight.tensor([[0, 1], [0, 1]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0])
        sparse = sparse_coo_tensor(indices, values, size=(2, 2), is_coalesced=True)

        coalesced = sparse.coalesce()
        assert coalesced is sparse  # Should return same object

    def test_coo_sparse_properties(self):
        """Test COO sparse tensor properties."""
        indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_coo_tensor(indices, values, size=(3, 3))

        assert sparse.is_sparse
        assert sparse.sparse_dim == 2
        assert sparse.dense_dim == 0
        assert sparse.dtype == flashlight.float32


class TestSparseCSRTensor:
    """Tests for CSR format sparse tensors."""

    def test_csr_creation_basic(self):
        """Test basic CSR tensor creation."""
        # 3x3 identity matrix
        crow_indices = flashlight.tensor([0, 1, 2, 3], dtype=flashlight.int64)
        col_indices = flashlight.tensor([0, 1, 2], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 1.0, 1.0])
        sparse = sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))

        assert isinstance(sparse, SparseCSRTensor)
        assert sparse.shape == (3, 3)
        assert sparse.nnz == 3

    def test_csr_to_dense(self):
        """Test CSR to dense conversion."""
        # 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        crow_indices = flashlight.tensor([0, 2, 3], dtype=flashlight.int64)
        col_indices = flashlight.tensor([0, 2, 1], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 3))

        dense = sparse.to_dense()
        expected = flashlight.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ])

        assert (dense == expected).all().item()

    def test_csr_to_coo_conversion(self):
        """Test CSR to COO conversion."""
        crow_indices = flashlight.tensor([0, 1, 2, 3], dtype=flashlight.int64)
        col_indices = flashlight.tensor([0, 1, 2], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        csr = sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))

        coo = csr.to_sparse_coo()
        assert isinstance(coo, SparseCOOTensor)
        assert coo.nnz == 3

        # Convert back to dense and compare
        assert (coo.to_dense() == csr.to_dense()).all().item()

    def test_csr_is_always_coalesced(self):
        """Test that CSR is always coalesced."""
        crow_indices = flashlight.tensor([0, 2, 3], dtype=flashlight.int64)
        col_indices = flashlight.tensor([0, 1, 0], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 2))

        assert sparse.is_coalesced()


class TestCOOtoCSRConversion:
    """Tests for COO to CSR conversion."""

    def test_coo_to_csr_basic(self):
        """Test COO to CSR conversion."""
        indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        coo = sparse_coo_tensor(indices, values, size=(3, 3))

        csr = coo.to_sparse_csr()
        assert isinstance(csr, SparseCSRTensor)
        assert csr.nnz == 3

        # Compare dense representations
        assert (coo.to_dense() == csr.to_dense()).all().item()

    def test_coo_to_csr_with_duplicates(self):
        """Test COO to CSR with duplicate indices."""
        indices = flashlight.tensor([[0, 0, 1], [0, 0, 1]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        coo = sparse_coo_tensor(indices, values, size=(2, 2), is_coalesced=False)

        csr = coo.to_sparse_csr()
        # Should have coalesced during conversion
        assert csr.nnz == 2

        dense = csr.to_dense()
        assert dense[0, 0].item() == pytest.approx(3.0)


class TestSparseOperations:
    """Tests for sparse tensor operations."""

    def test_sparse_mm_identity(self):
        """Test SpMM with identity matrix."""
        # Create sparse identity
        indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 1.0, 1.0])
        sparse = sparse_coo_tensor(indices, values, size=(3, 3))

        # Dense matrix
        dense = flashlight.randn(3, 4)

        # sparse @ dense should equal dense
        result = sparse_mm(sparse, dense)
        assert result.shape == (3, 4)

        # Check numerical equality
        max_diff = (result - dense).abs().max().item()
        assert max_diff < 1e-5

    def test_sparse_mm_csr(self):
        """Test SpMM with CSR format."""
        # 2x3 sparse matrix
        crow_indices = flashlight.tensor([0, 2, 3], dtype=flashlight.int64)
        col_indices = flashlight.tensor([0, 2, 1], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 3))

        # 3x4 dense matrix
        dense = flashlight.ones(3, 4)

        result = sparse_mm(sparse, dense)
        assert result.shape == (2, 4)

        # Row 0: 1*1 + 0*1 + 2*1 = 3
        # Row 1: 0*1 + 3*1 + 0*1 = 3
        assert result[0, 0].item() == pytest.approx(3.0)
        assert result[1, 0].item() == pytest.approx(3.0)

    def test_sparse_add_basic(self):
        """Test sparse tensor addition."""
        # Create two sparse tensors
        indices1 = flashlight.tensor([[0, 1], [0, 1]], dtype=flashlight.int64)
        values1 = flashlight.tensor([1.0, 2.0])
        sparse1 = sparse_coo_tensor(indices1, values1, size=(2, 2))

        indices2 = flashlight.tensor([[0, 1], [1, 0]], dtype=flashlight.int64)
        values2 = flashlight.tensor([3.0, 4.0])
        sparse2 = sparse_coo_tensor(indices2, values2, size=(2, 2))

        result = sparse_add(sparse1, sparse2)
        assert isinstance(result, SparseCOOTensor)

        dense = result.to_dense()
        expected = flashlight.tensor([
            [1.0, 3.0],
            [4.0, 2.0],
        ])
        assert (dense == expected).all().item()

    def test_sparse_add_overlapping(self):
        """Test sparse addition with overlapping indices."""
        indices1 = flashlight.tensor([[0], [0]], dtype=flashlight.int64)
        values1 = flashlight.tensor([1.0])
        sparse1 = sparse_coo_tensor(indices1, values1, size=(2, 2))

        indices2 = flashlight.tensor([[0], [0]], dtype=flashlight.int64)
        values2 = flashlight.tensor([2.0])
        sparse2 = sparse_coo_tensor(indices2, values2, size=(2, 2))

        result = sparse_add(sparse1, sparse2)
        coalesced = result.coalesce()

        # After coalescing, overlapping values should be summed
        assert coalesced.to_dense()[0, 0].item() == pytest.approx(3.0)

    def test_sparse_sum_all(self):
        """Test summing all elements of sparse tensor."""
        indices = flashlight.tensor([[0, 1, 2], [0, 1, 2]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_coo_tensor(indices, values, size=(3, 3))

        total = sparse_sum(sparse)
        assert total.item() == pytest.approx(6.0)

    def test_sparse_transpose(self):
        """Test sparse tensor transpose."""
        # [[1, 2], [0, 3]] -> [[1, 0], [2, 3]]
        indices = flashlight.tensor([[0, 0, 1], [0, 1, 1]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_coo_tensor(indices, values, size=(2, 2))

        transposed = sparse_t(sparse)
        assert transposed.shape == (2, 2)

        dense_t = transposed.to_dense()
        expected = flashlight.tensor([
            [1.0, 0.0],
            [2.0, 3.0],
        ])
        assert (dense_t == expected).all().item()

    def test_sparse_transpose_csr(self):
        """Test CSR tensor transpose (converts via COO)."""
        crow_indices = flashlight.tensor([0, 1, 2, 3], dtype=flashlight.int64)
        col_indices = flashlight.tensor([0, 1, 2], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0])
        sparse = sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))

        transposed = sparse_t(sparse)
        assert isinstance(transposed, SparseCSRTensor)

        # Identity matrix transpose is same as original
        assert (transposed.to_dense() == sparse.to_dense()).all().item()


class TestDenseToSparse:
    """Tests for dense to sparse conversion."""

    def test_dense_to_sparse_coo(self):
        """Test converting dense tensor to COO format."""
        dense = flashlight.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ])

        sparse = dense_to_sparse(dense, layout="coo")
        assert isinstance(sparse, SparseCOOTensor)
        assert sparse.nnz == 3  # 3 non-zero elements

        # Convert back and compare
        recovered = sparse.to_dense()
        assert (recovered == dense).all().item()

    def test_dense_to_sparse_csr(self):
        """Test converting dense tensor to CSR format."""
        dense = flashlight.tensor([
            [1.0, 0.0],
            [0.0, 2.0],
        ])

        sparse = dense_to_sparse(dense, layout="csr")
        assert isinstance(sparse, SparseCSRTensor)
        assert sparse.nnz == 2

        recovered = sparse.to_dense()
        assert (recovered == dense).all().item()

    def test_sparse_to_dense_function(self):
        """Test sparse_to_dense utility function."""
        indices = flashlight.tensor([[0, 1], [0, 1]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0])
        sparse = sparse_coo_tensor(indices, values, size=(2, 2))

        dense = sparse_to_dense(sparse)
        expected = flashlight.tensor([
            [1.0, 0.0],
            [0.0, 2.0],
        ])
        assert (dense == expected).all().item()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_sparse_tensor(self):
        """Test sparse tensor with no non-zeros."""
        indices = flashlight.tensor([[], []], dtype=flashlight.int64).reshape(2, 0)
        values = flashlight.tensor([], dtype=flashlight.float32)
        sparse = sparse_coo_tensor(indices, values, size=(3, 3))

        assert sparse.nnz == 0

        dense = sparse.to_dense()
        assert (dense == flashlight.zeros(3, 3)).all().item()

    def test_single_element_sparse(self):
        """Test sparse tensor with single element."""
        indices = flashlight.tensor([[1], [2]], dtype=flashlight.int64)
        values = flashlight.tensor([5.0])
        sparse = sparse_coo_tensor(indices, values, size=(3, 4))

        assert sparse.nnz == 1

        dense = sparse.to_dense()
        assert dense[1, 2].item() == 5.0
        assert dense.sum().item() == 5.0

    def test_full_sparse_tensor(self):
        """Test sparse tensor where all elements are non-zero."""
        # 2x2 matrix with all non-zeros
        indices = flashlight.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=flashlight.int64)
        values = flashlight.tensor([1.0, 2.0, 3.0, 4.0])
        sparse = sparse_coo_tensor(indices, values, size=(2, 2))

        assert sparse.nnz == 4

        dense = sparse.to_dense()
        expected = flashlight.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        assert (dense == expected).all().item()
