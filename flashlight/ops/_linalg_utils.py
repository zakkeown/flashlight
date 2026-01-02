"""
Linear Algebra Utilities for LOBPCG

PyTorch-compatible helper functions ported from torch/_linalg_utils.py.
These provide building blocks for the LOBPCG eigenvalue solver.
"""

from typing import Optional, Tuple, Union

import mlx.core as mx

from ..tensor import Tensor
from ..dtype import DType, float16, float32, float64


def is_sparse(A) -> bool:
    """
    Check if tensor A is a sparse tensor.

    All sparse storage formats (COO, CSR, etc.) return True.
    Dense tensors return False.

    Args:
        A: Input tensor or sparse tensor

    Returns:
        True if A is sparse, False otherwise
    """
    # Import here to avoid circular imports
    try:
        from ..sparse import SparseTensor
        return isinstance(A, SparseTensor)
    except ImportError:
        return False


def get_floating_dtype(A: Tensor) -> DType:
    """
    Return the floating point dtype of tensor A.

    Integer types map to float32 (matching PyTorch behavior).

    Args:
        A: Input tensor

    Returns:
        Floating point dtype (float16, float32, or float64)
    """
    dtype = A.dtype
    if dtype in (float16, float32):
        return dtype
    # MLX doesn't support float64, default to float32
    return float32


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor:
    """
    Multiply two matrices, handling None and sparse cases.

    If A is None, return B unchanged.
    If A is sparse, use sparse matrix multiplication.
    Otherwise, use standard matrix multiplication.

    Args:
        A: First matrix (can be None or sparse)
        B: Second matrix (always dense)

    Returns:
        Result of A @ B, or B if A is None
    """
    if A is None:
        return B
    if is_sparse(A):
        from ..sparse import sparse_mm
        return sparse_mm(A, B)
    return A @ B


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor:
    """
    Compute bilinear form: X^T @ A @ Y

    If A is None, computes X^T @ Y.

    Args:
        X: Left matrix [n, m]
        A: Middle matrix [n, n] (can be None)
        Y: Right matrix [n, k]

    Returns:
        Bilinear form result [m, k]
    """
    return matmul(X.t(), matmul(A, Y))


def qform(A: Optional[Tensor], S: Tensor) -> Tensor:
    """
    Compute quadratic form: S^T @ A @ S

    This is a special case of bilinear form where X = Y = S.

    Args:
        A: Matrix [n, n] (can be None, treated as identity)
        S: Matrix [n, m]

    Returns:
        Quadratic form result [m, m]
    """
    return bform(S, A, S)


def basis(A: Tensor) -> Tensor:
    """
    Return orthogonal basis of A's column space via QR decomposition.

    Args:
        A: Input matrix [n, m]

    Returns:
        Orthonormal basis Q [n, min(n, m)]
    """
    from ..linalg import qr
    return qr(A).Q


def symeig(A: Tensor, largest: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
    """
    Compute eigenvalues and eigenvectors of symmetric matrix.

    Uses the upper triangular part of A (UPLO='U').

    Args:
        A: Symmetric matrix [n, n]
        largest: If True, return eigenvalues in descending order
                 If False (default), return in ascending order

    Returns:
        Tuple of (eigenvalues [n], eigenvectors [n, n])
    """
    from ..linalg import eigh
    from . import flip

    if largest is None:
        largest = False

    E, Z = eigh(A, UPLO="U")

    # PyTorch returns eigenvalues in ascending order by default
    # Flip if largest eigenvalues are requested first
    if largest:
        E = flip(E, dims=(-1,))
        Z = flip(Z, dims=(-1,))

    return E, Z


# Deprecated function stubs for PyTorch compatibility
# These match PyTorch's deprecated functions that raise RuntimeError

def matrix_rank(input, tol=None, symmetric=False, *, out=None) -> Tensor:
    """Deprecated: Use torch.linalg.matrix_rank instead."""
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed.\n"
        "Please use the `flashlight.linalg.matrix_rank` function instead. "
        "The parameter 'symmetric' was renamed in `flashlight.linalg.matrix_rank()` to 'hermitian'."
    )


def solve(input: Tensor, A: Tensor, *, out=None) -> Tuple[Tensor, Tensor]:
    """Deprecated: Use torch.linalg.solve instead."""
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`flashlight.solve` is deprecated in favor of `flashlight.linalg.solve`. "
        "`flashlight.linalg.solve` has its arguments reversed and does not return the LU factorization.\n\n"
        "To get the LU factorization see `flashlight.lu`, which can be used with `flashlight.lu_solve` or `flashlight.lu_unpack`.\n"
        "X = flashlight.solve(B, A).solution "
        "should be replaced with:\n"
        "X = flashlight.linalg.solve(A, B)"
    )


def lstsq(input: Tensor, A: Tensor, *, out=None) -> Tuple[Tensor, Tensor]:
    """Deprecated: Use torch.linalg.lstsq instead."""
    raise RuntimeError(
        "This function was deprecated since version 1.9 and is now removed. "
        "`flashlight.lstsq` is deprecated in favor of `flashlight.linalg.lstsq`.\n"
        "`flashlight.linalg.lstsq` has reversed arguments and does not return the QR decomposition in "
        "the returned tuple (although it returns other information about the problem).\n\n"
        "To get the QR decomposition consider using `flashlight.linalg.qr`.\n\n"
        "The returned solution in `flashlight.lstsq` stored the residuals of the solution in the "
        "last m - n columns of the returned value whenever m > n. In flashlight.linalg.lstsq, "
        "the residuals are in the field 'residuals' of the returned named tuple.\n\n"
        "The unpacking of the solution, as in\n"
        "X, _ = flashlight.lstsq(B, A).solution[:A.size(1)]\n"
        "should be replaced with:\n"
        "X = flashlight.linalg.lstsq(A, B).solution"
    )
