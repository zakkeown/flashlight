"""
Linear Algebra Operations

Implements PyTorch-compatible linear algebra operations with MLX backend.
These are the torch.* level ops (einsum, tensordot, diag, etc.)
"""

import re
from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx

from ..autograd.context import is_grad_enabled
from ..tensor import Tensor


def _parse_transpose_pattern(equation: str) -> Optional[List[int]]:
    """
    Check if einsum equation is a simple transpose pattern.

    Args:
        equation: Einsum equation string (e.g., "ij->ji", "abc->cba")

    Returns:
        Permutation list if it's a transpose, None otherwise
    """
    # Match patterns like "ij->ji", "ijk->kji", "abcd->dcba"
    match = re.match(r"^([a-zA-Z]+)->([a-zA-Z]+)$", equation.replace(" ", ""))
    if not match:
        return None

    input_indices, output_indices = match.groups()

    # Must have same set of indices (pure permutation)
    if set(input_indices) != set(output_indices):
        return None

    # No repeated indices (that would be trace/contraction)
    if len(input_indices) != len(set(input_indices)):
        return None

    # Build permutation
    perm = [input_indices.index(c) for c in output_indices]
    return perm


def _sublist_to_equation(args) -> Tuple[str, Tuple[Tensor, ...]]:
    """
    Convert sublist format to equation string format.

    Sublist format: einsum(op1, sublist1, op2, sublist2, ..., [output_sublist])
    where sublists are lists of integers representing subscript indices.
    Ellipsis (...) can be used for broadcasting dimensions.

    Args:
        args: Tuple of operands and sublists interleaved

    Returns:
        Tuple of (equation_string, operands_tuple)
    """
    # Parse alternating operands and sublists
    operands = []
    subscript_lists = []

    i = 0
    while i < len(args):
        if isinstance(args[i], Tensor):
            operands.append(args[i])
            i += 1
            # Next should be a subscript list
            if i < len(args) and isinstance(args[i], (list, tuple)):
                subscript_lists.append(args[i])
                i += 1
            else:
                raise ValueError(f"Expected subscript list after operand at position {i-1}")
        elif isinstance(args[i], (list, tuple)):
            # This might be the output subscript list
            subscript_lists.append(args[i])
            i += 1
        else:
            raise ValueError(f"Unexpected argument type at position {i}: {type(args[i])}")

    # Number of operand subscripts should be len(operands)
    # If there's one extra subscript list, it's the output
    if len(subscript_lists) == len(operands):
        output_subscripts = None
    elif len(subscript_lists) == len(operands) + 1:
        output_subscripts = subscript_lists[-1]
        subscript_lists = subscript_lists[:-1]
    else:
        raise ValueError(f"Got {len(subscript_lists)} subscript lists for {len(operands)} operands")

    # Convert integer subscripts to letters
    # Integers 0-25 map to a-z, 26-51 map to A-Z
    def int_to_letter(n: int) -> str:
        if n < 0 or n >= 52:
            raise ValueError(f"Subscript index {n} out of range [0, 52)")
        if n < 26:
            return chr(ord("a") + n)
        else:
            return chr(ord("A") + n - 26)

    def subscripts_to_str(subscripts) -> str:
        """Convert a subscript list to a string, handling ellipsis."""
        result = []
        for s in subscripts:
            if s is Ellipsis or s is ...:
                result.append("...")
            elif isinstance(s, int):
                result.append(int_to_letter(s))
            else:
                raise ValueError(f"Invalid subscript element: {s}")
        return "".join(result)

    # Build equation string
    input_parts = [subscripts_to_str(sl) for sl in subscript_lists]
    equation = ",".join(input_parts)

    if output_subscripts is not None:
        equation += "->" + subscripts_to_str(output_subscripts)

    return equation, tuple(operands)


def einsum(*args) -> Tensor:
    """
    Evaluates the Einstein summation convention on the operands.

    Args:
        *args: Either (equation, *operands) for string format, or
               (op1, sublist1, op2, sublist2, ..., [output_sublist]) for sublist format

    Returns:
        The calculated tensor

    Example:
        >>> a = flashlight.randn(3, 4)
        >>> b = flashlight.randn(4, 5)
        >>> # String format
        >>> c = flashlight.einsum('ij,jk->ik', a, b)  # matrix multiplication
        >>> # Sublist format
        >>> c = flashlight.einsum(a, [0, 1], b, [1, 2], [0, 2])  # same operation
    """
    # Parse args - first arg is either equation or a tensor
    if len(args) == 0:
        raise TypeError("einsum() requires at least 1 argument")

    # Check if first arg is equation string
    if isinstance(args[0], str):
        equation = args[0]
        operands = args[1:]
    else:
        # Sublist format: operands with subscripts embedded
        equation, operands = _sublist_to_equation(args)

    # Optimization: detect simple transpose patterns and use mx.transpose()
    # This avoids the overhead of the full einsum contraction engine
    if len(operands) == 1:
        perm = _parse_transpose_pattern(equation)
        if perm is not None:
            result_array = mx.transpose(operands[0]._mlx_array, perm)
            result = Tensor._from_mlx_array(result_array)
            if is_grad_enabled() and operands[0].requires_grad:
                result.requires_grad = True
            return result

    # General case: use MLX einsum
    mlx_arrays = [op._mlx_array for op in operands]
    result_array = mx.einsum(equation, *mlx_arrays)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and any(op.requires_grad for op in operands):
        result.requires_grad = True

    return result


def tensordot(
    a: Tensor,
    b: Tensor,
    dims: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes a generalized contraction between tensors.

    Args:
        a: First tensor
        b: Second tensor
        dims: Number of dimensions to contract, or explicit axes to contract
        out: Output tensor (ignored, for PyTorch compatibility)

    Returns:
        Contracted tensor

    Example:
        >>> a = flashlight.randn(3, 4, 5)
        >>> b = flashlight.randn(4, 5, 6)
        >>> c = flashlight.tensordot(a, b, dims=2)  # contracts last 2 dims of a with first 2 of b
    """
    if isinstance(dims, int):
        axes = dims
    else:
        axes = dims

    result_array = mx.tensordot(a._mlx_array, b._mlx_array, axes=axes)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (a.requires_grad or b.requires_grad):
        result.requires_grad = True

    # Note: 'out' parameter is ignored in MLX (no in-place output support)
    return result


def diag(input: Tensor, diagonal: int = 0) -> Tensor:
    """
    If input is a vector, returns a 2D tensor with input as the diagonal.
    If input is a matrix, returns a 1D tensor with the diagonal elements.

    Args:
        input: Input tensor (1D or 2D)
        diagonal: Which diagonal (0 = main, positive = above, negative = below)

    Returns:
        Diagonal tensor

    Example:
        >>> v = flashlight.tensor([1, 2, 3])
        >>> flashlight.diag(v)
        tensor([[1, 0, 0],
                [0, 2, 0],
                [0, 0, 3]])
    """
    result_array = mx.diag(input._mlx_array, k=diagonal)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def diagonal(input: Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
    """
    Returns the diagonal of the tensor.

    Args:
        input: Input tensor (at least 2D)
        offset: Which diagonal (0 = main, positive = above, negative = below)
        dim1: First dimension for 2D sub-tensor
        dim2: Second dimension for 2D sub-tensor

    Returns:
        Tensor containing diagonal elements
    """
    mlx_array = input._mlx_array

    # MLX diagonal doesn't support dim1/dim2 directly, need to handle
    if dim1 != 0 or dim2 != 1:
        # Move dimensions to positions 0 and 1
        ndim = len(mlx_array.shape)
        dim1 = dim1 if dim1 >= 0 else ndim + dim1
        dim2 = dim2 if dim2 >= 0 else ndim + dim2

        # Build permutation
        perm = list(range(ndim))
        perm.remove(dim1)
        perm.remove(dim2)
        perm = [dim1, dim2] + perm

        mlx_array = mx.transpose(mlx_array, perm)

    result_array = mx.diagonal(mlx_array, offset=offset)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    """
    Returns upper triangular part of tensor.

    Args:
        input: Input tensor (at least 2D)
        diagonal: Diagonal offset (0 = main, positive = above, negative = below)

    Returns:
        Upper triangular tensor
    """
    result_array = mx.triu(input._mlx_array, k=diagonal)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    """
    Returns lower triangular part of tensor.

    Args:
        input: Input tensor (at least 2D)
        diagonal: Diagonal offset (0 = main, positive = above, negative = below)

    Returns:
        Lower triangular tensor
    """
    result_array = mx.tril(input._mlx_array, k=diagonal)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def trace(input: Tensor) -> Tensor:
    """
    Returns the sum of diagonal elements.

    Args:
        input: Input tensor (2D)

    Returns:
        Scalar tensor with trace value
    """
    result_array = mx.trace(input._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True

    return result


def outer(input: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes outer product of two 1D vectors.

    Args:
        input: First 1D tensor
        vec2: Second 1D tensor

    Returns:
        2D tensor with outer product

    Example:
        >>> a = flashlight.tensor([1, 2, 3])
        >>> b = flashlight.tensor([4, 5])
        >>> flashlight.outer(a, b)
        tensor([[ 4,  5],
                [ 8, 10],
                [12, 15]])
    """
    result_array = mx.outer(input._mlx_array, vec2._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or vec2.requires_grad):
        result.requires_grad = True

    return result


def inner(input: Tensor, other: Tensor) -> Tensor:
    """
    Computes inner product of two tensors.

    For 1D tensors, this is the dot product.
    For higher dimensions, it's a sum product over the last axis.

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Inner product tensor
    """
    result_array = mx.inner(input._mlx_array, other._mlx_array)
    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True

    return result


def dot(input: Tensor, other: Tensor) -> Tensor:
    """
    Computes dot product of two 1D tensors.

    Args:
        input: First 1D tensor
        other: Second 1D tensor

    Returns:
        Scalar tensor with dot product
    """
    # inner and dot are the same for 1D
    return inner(input, other)


def vdot(input: Tensor, other: Tensor) -> Tensor:
    """
    Computes dot product of two 1D tensors (with complex conjugation).

    For real tensors, equivalent to dot.

    Args:
        input: First 1D tensor
        other: Second 1D tensor

    Returns:
        Scalar tensor
    """
    # For real tensors, vdot is same as dot
    # MLX doesn't have complex support anyway
    return dot(input, other)


def kron(input: Tensor, other: Tensor) -> Tensor:
    """
    Computes Kronecker product of two tensors.

    For matrices A (m x n) and B (p x q), the Kronecker product is:
    kron(A, B)[i*p + k, j*q + l] = A[i,j] * B[k,l]

    Args:
        input: First tensor
        other: Second tensor

    Returns:
        Kronecker product tensor
    """
    a = input._mlx_array
    b = other._mlx_array

    # Handle 1D inputs by converting to row vectors
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    m, n = a.shape
    p, q = b.shape

    # Expand and multiply: result shape is (m, n, p, q)
    # a[:, :, None, None] broadcasts to (m, n, 1, 1)
    # b[None, None, :, :] broadcasts to (1, 1, p, q)
    product = a[:, :, None, None] * b[None, None, :, :]

    # Transpose to (m, p, n, q) to interleave dimensions correctly
    # Then reshape to (m*p, n*q)
    result_array = mx.transpose(product, (0, 2, 1, 3))
    result_array = mx.reshape(result_array, (m * p, n * q))

    result = Tensor._from_mlx_array(result_array)

    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True

    return result


__all__ = [
    "einsum",
    "tensordot",
    "diag",
    "diagonal",
    "triu",
    "tril",
    "trace",
    "outer",
    "inner",
    "dot",
    "vdot",
    "kron",
]
