"""
torch.linalg namespace

Implements PyTorch-compatible linear algebra functions with MLX backend.
"""

from typing import Literal, Optional, Tuple, Union

import mlx.core as mx

from .autograd.context import is_grad_enabled
from .tensor import Tensor


def norm(
    input: Tensor,
    ord: Optional[Union[int, float, str]] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """
    Computes a vector or matrix norm.

    Args:
        input: Input tensor
        ord: Order of norm. Can be:
            - None: Frobenius norm for matrices, 2-norm for vectors
            - 'fro': Frobenius norm
            - 'nuc': Nuclear norm (not supported)
            - inf: Max norm
            - -inf: Min norm
            - int/float: p-norm
        dim: Dimensions to compute norm over
        keepdim: Whether to keep reduced dimensions
        dtype: Data type for computation (ignored, uses input dtype)

    Returns:
        Norm tensor

    Example:
        >>> x = flashlight.tensor([[1., 2.], [3., 4.]])
        >>> flashlight.linalg.norm(x)  # Frobenius norm
        tensor(5.4772)
    """
    mlx_array = input._mlx_array

    # Convert dim to axis
    axis = dim

    # Determine norm type
    if ord is None:
        # Default: Frobenius for matrices (dim=None), 2-norm otherwise
        if axis is None and mlx_array.ndim == 2:
            ord = "fro"
        else:
            ord = 2

    if ord == "fro":
        # Frobenius norm: sqrt(sum of squares)
        result_array = mx.sqrt(mx.sum(mx.square(mlx_array), axis=axis, keepdims=keepdim))
    elif ord == "nuc":
        # Nuclear norm is the sum of singular values: ||A||_* = sum(svd(A).S)
        # SVD requires CPU stream in MLX
        _, S, _ = mx.linalg.svd(mlx_array, stream=mx.cpu)
        result_array = mx.sum(S, axis=-1, keepdims=keepdim)
    elif ord == float("inf"):
        # Max absolute value
        result_array = mx.max(mx.abs(mlx_array), axis=axis, keepdims=keepdim)
    elif ord == float("-inf"):
        # Min absolute value
        result_array = mx.min(mx.abs(mlx_array), axis=axis, keepdims=keepdim)
    elif isinstance(ord, (int, float)):
        if ord == 0:
            # Count non-zero elements
            result_array = mx.sum(mlx_array != 0, axis=axis, keepdims=keepdim).astype(
                mlx_array.dtype
            )
        elif ord == 1:
            result_array = mx.sum(mx.abs(mlx_array), axis=axis, keepdims=keepdim)
        elif ord == 2:
            result_array = mx.sqrt(mx.sum(mx.square(mlx_array), axis=axis, keepdims=keepdim))
        else:
            # General p-norm: (sum(|x|^p))^(1/p)
            result_array = mx.power(
                mx.sum(mx.power(mx.abs(mlx_array), ord), axis=axis, keepdims=keepdim), 1.0 / ord
            )
    else:
        raise ValueError(f"Invalid norm order: {ord}")

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def vector_norm(
    input: Tensor = None,
    ord: Union[int, float] = 2,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    dtype: Optional[str] = None,
    *,
    x: Tensor = None,  # Alias for input for compatibility
) -> Tensor:
    """
    Computes a vector norm.

    Args:
        input: Input tensor (also accepts 'x' as alias)
        ord: Order of norm (default: 2)
        dim: Dimensions to compute norm over
        keepdim: Whether to keep reduced dimensions
        dtype: Data type for computation
        x: Alias for input (for compatibility)

    Returns:
        Norm tensor
    """
    # Handle x alias for input
    if x is not None:
        if input is not None:
            raise ValueError("Cannot specify both 'input' and 'x' arguments")
        input = x

    if input is None:
        raise ValueError("Must specify 'input' (or 'x') argument")

    return norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)


def matrix_norm(
    input: Tensor,
    ord: Union[int, float, str] = "fro",
    dim: Tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    dtype: Optional[str] = None,
) -> Tensor:
    """
    Computes a matrix norm.

    Args:
        input: Input tensor (at least 2D)
        ord: Order of norm ('fro' or numeric)
        dim: Two dimensions to compute norm over
        keepdim: Whether to keep reduced dimensions
        dtype: Data type for computation

    Returns:
        Norm tensor
    """
    return norm(input, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)


def svd(input: Tensor, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes singular value decomposition.

    Args:
        input: Input tensor (..., m, n)
        full_matrices: If True, compute full-sized U and Vh

    Returns:
        Tuple of (U, S, Vh)

    Example:
        >>> a = flashlight.randn(3, 4)
        >>> U, S, Vh = flashlight.linalg.svd(a)
    """
    # SVD requires CPU stream in MLX
    # MLX svd doesn't have full_matrices param - always computes reduced SVD
    U, S, Vh = mx.linalg.svd(input._mlx_array, stream=mx.cpu)

    U_tensor = Tensor._from_mlx_array(U)
    S_tensor = Tensor._from_mlx_array(S)
    Vh_tensor = Tensor._from_mlx_array(Vh)

    if is_grad_enabled() and input.requires_grad:
        U_tensor.requires_grad = True
        S_tensor.requires_grad = True
        Vh_tensor.requires_grad = True

    return U_tensor, S_tensor, Vh_tensor


def svdvals(input: Tensor) -> Tensor:
    """
    Computes singular values of a matrix.

    Args:
        input: Input tensor

    Returns:
        Singular values tensor
    """
    _, S, _ = svd(input, full_matrices=False)
    return S


def qr(input: Tensor, mode: str = "reduced") -> Tuple[Tensor, Tensor]:
    """
    Computes QR decomposition.

    Args:
        input: Input tensor (..., m, n)
        mode: 'reduced' or 'complete' (MLX supports 'reduced')

    Returns:
        Tuple of (Q, R)
    """
    # QR requires CPU stream in MLX
    Q, R = mx.linalg.qr(input._mlx_array, stream=mx.cpu)

    Q_tensor = Tensor._from_mlx_array(Q)
    R_tensor = Tensor._from_mlx_array(R)

    if is_grad_enabled() and input.requires_grad:
        Q_tensor.requires_grad = True
        R_tensor.requires_grad = True

    return Q_tensor, R_tensor


def cholesky(input: Tensor, upper: bool = False) -> Tensor:
    """
    Computes Cholesky decomposition of positive-definite matrix.

    Args:
        input: Input tensor (positive-definite)
        upper: If True, return upper triangular; else lower

    Returns:
        Cholesky factor

    Example:
        >>> a = flashlight.tensor([[4., 2.], [2., 5.]])
        >>> L = flashlight.linalg.cholesky(a)
    """
    # Some MLX linalg ops require CPU stream
    L = mx.linalg.cholesky(input._mlx_array, upper=upper, stream=mx.cpu)
    result = Tensor._from_mlx_array(L)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def inv(input: Tensor) -> Tensor:
    """
    Computes inverse of a square matrix.

    Args:
        input: Input tensor (square matrix)

    Returns:
        Inverse matrix
    """
    # inv requires CPU stream in MLX
    result_array = mx.linalg.inv(input._mlx_array, stream=mx.cpu)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def solve(A: Tensor, B: Tensor) -> Tensor:
    """
    Solves the linear system AX = B.

    Args:
        A: Coefficient matrix
        B: Right-hand side matrix/vector

    Returns:
        Solution X
    """
    # solve requires CPU stream in MLX
    result_array = mx.linalg.solve(A._mlx_array, B._mlx_array, stream=mx.cpu)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (A.requires_grad or B.requires_grad):
        result.requires_grad = True

    return result


def eig(input: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes eigenvalues and eigenvectors of a square matrix.

    Args:
        input: Input tensor (square matrix)

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # eig requires CPU stream in MLX
    eigenvalues, eigenvectors = mx.linalg.eig(input._mlx_array, stream=mx.cpu)

    eigenvalues_tensor = Tensor._from_mlx_array(eigenvalues)
    eigenvectors_tensor = Tensor._from_mlx_array(eigenvectors)

    if is_grad_enabled() and input.requires_grad:
        eigenvalues_tensor.requires_grad = True
        eigenvectors_tensor.requires_grad = True

    return eigenvalues_tensor, eigenvectors_tensor


def eigh(input: Tensor, UPLO: str = "L") -> Tuple[Tensor, Tensor]:
    """
    Computes eigenvalues and eigenvectors of a symmetric/Hermitian matrix.

    Args:
        input: Input tensor (symmetric matrix)
        UPLO: 'L' for lower triangle, 'U' for upper

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # eigh requires CPU stream in MLX
    eigenvalues, eigenvectors = mx.linalg.eigh(input._mlx_array, UPLO=UPLO, stream=mx.cpu)

    eigenvalues_tensor = Tensor._from_mlx_array(eigenvalues)
    eigenvectors_tensor = Tensor._from_mlx_array(eigenvectors)

    if is_grad_enabled() and input.requires_grad:
        eigenvalues_tensor.requires_grad = True
        eigenvectors_tensor.requires_grad = True

    return eigenvalues_tensor, eigenvectors_tensor


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    """
    Computes eigenvalues of a symmetric/Hermitian matrix.

    Args:
        input: Input tensor (symmetric matrix)
        UPLO: 'L' for lower triangle, 'U' for upper

    Returns:
        Eigenvalues tensor
    """
    eigenvalues, _ = eigh(input, UPLO=UPLO)
    return eigenvalues


def matrix_rank(input: Tensor, tol: Optional[float] = None) -> Tensor:
    """
    Computes numerical rank of a matrix.

    Args:
        input: Input tensor
        tol: Tolerance for singular values

    Returns:
        Rank (integer tensor)
    """
    S = svdvals(input)

    if tol is None:
        # Default tolerance
        tol = max(input.shape[-2:]) * S._mlx_array[..., 0] * 1e-6

    rank = mx.sum(S._mlx_array > tol, axis=-1)
    return Tensor._from_mlx_array(rank)


def pinv(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """
    Computes Moore-Penrose pseudoinverse.

    Args:
        input: Input tensor of shape (..., m, n)
        rcond: Cutoff for small singular values

    Returns:
        Pseudoinverse tensor of shape (..., n, m)
    """
    # Pure MLX implementation using SVD
    arr = input._mlx_array
    m, n = arr.shape[-2], arr.shape[-1]
    k = min(m, n)

    # Compute SVD on CPU (MLX SVD is CPU-only currently)
    # MLX returns full matrices, so we need to slice to get reduced form
    U_full, S, Vt_full = mx.linalg.svd(arr, stream=mx.cpu)
    mx.eval(U_full, S, Vt_full)

    # Extract reduced matrices: U is (..., m, k), Vt is (..., k, n)
    U = U_full[..., :k]  # (..., m, k)
    Vt = Vt_full[..., :k, :]  # (..., k, n)

    # Invert singular values above threshold
    max_s = mx.max(S, axis=-1, keepdims=True)
    threshold = rcond * max_s
    # S_inv: invert values above threshold, zero out small values
    S_inv = mx.where(S > threshold, 1.0 / S, mx.zeros_like(S))

    # Construct pseudoinverse: V @ diag(S_inv) @ U^T = Vt^T @ diag(S_inv) @ U^T
    # V is Vt transposed: (..., n, k)
    V = mx.swapaxes(Vt, -2, -1)  # (..., n, k)

    # Multiply V by S_inv: (..., n, k) * (..., k) -> (..., n, k)
    VS_inv = V * mx.expand_dims(S_inv, axis=-2)

    # U transposed: (..., k, m)
    Ut = mx.swapaxes(U, -2, -1)

    # Result: (..., n, k) @ (..., k, m) = (..., n, m)
    result_array = mx.matmul(VS_inv, Ut)

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def det(input: Tensor) -> Tensor:
    """
    Computes determinant of a square matrix.

    Note: MLX doesn't have det directly, we compute via LU decomposition.

    Args:
        input: Input tensor (square matrix)

    Returns:
        Determinant
    """
    # Use property: det(A) = product of diagonal of U in LU decomposition
    # For simplicity, use eigenvalues: det = product of eigenvalues
    eigenvalues, _ = eig(input)

    # Product of eigenvalues (handle complex eigenvalues for real matrices)
    # For real symmetric matrices, eigenvalues are real
    result_array = mx.prod(eigenvalues._mlx_array, axis=-1)

    # Take real part (determinant of real matrix is real)
    if result_array.dtype in [mx.complex64]:
        result_array = mx.real(result_array)

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def slogdet(input: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes sign and log absolute value of determinant.

    Args:
        input: Input tensor (square matrix)

    Returns:
        Tuple of (sign, logabsdet)
    """
    d = det(input)
    sign = mx.sign(d._mlx_array)
    logabsdet = mx.log(mx.abs(d._mlx_array))

    return Tensor._from_mlx_array(sign), Tensor._from_mlx_array(logabsdet)


def cross(input: Tensor, other: Tensor, dim: int = -1) -> Tensor:
    """
    Computes cross product of two 3D vectors.

    Args:
        input: First tensor
        other: Second tensor
        dim: Dimension of the 3-element vectors

    Returns:
        Cross product tensor
    """
    # Cross product: [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
    a = input._mlx_array
    b = other._mlx_array

    ndim = a.ndim
    dim = dim if dim >= 0 else ndim + dim

    # Build index slices
    def get_slice(idx):
        slices = [slice(None)] * ndim
        slices[dim] = idx
        return tuple(slices)

    a1 = a[get_slice(0)]
    a2 = a[get_slice(1)]
    a3 = a[get_slice(2)]
    b1 = b[get_slice(0)]
    b2 = b[get_slice(1)]
    b3 = b[get_slice(2)]

    c1 = a2 * b3 - a3 * b2
    c2 = a3 * b1 - a1 * b3
    c3 = a1 * b2 - a2 * b1

    result_array = mx.stack([c1, c2, c3], axis=dim)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True

    return result


__all__ = [
    "norm",
    "vector_norm",
    "matrix_norm",
    "svd",
    "svdvals",
    "qr",
    "cholesky",
    "inv",
    "solve",
    "eig",
    "eigh",
    "eigvalsh",
    "matrix_rank",
    "pinv",
    "det",
    "slogdet",
    "cross",
]
