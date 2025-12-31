"""
Quick Operations - One-liner MLX wrappers

These are simple pass-through functions to MLX equivalents.
"""

from typing import Union, Tuple, List, Optional
import mlx.core as mx

from ..tensor import Tensor
from ..autograd.context import is_grad_enabled


def atleast_1d(*tensors: Tensor) -> Union[Tensor, List[Tensor]]:
    """View inputs as arrays with at least one dimension."""
    results = []
    for t in tensors:
        arr = mx.atleast_1d(t._mlx_array)
        results.append(Tensor._from_mlx_array(arr))
    return results[0] if len(results) == 1 else results


def atleast_2d(*tensors: Tensor) -> Union[Tensor, List[Tensor]]:
    """View inputs as arrays with at least two dimensions."""
    results = []
    for t in tensors:
        arr = mx.atleast_2d(t._mlx_array)
        results.append(Tensor._from_mlx_array(arr))
    return results[0] if len(results) == 1 else results


def atleast_3d(*tensors: Tensor) -> Union[Tensor, List[Tensor]]:
    """View inputs as arrays with at least three dimensions."""
    results = []
    for t in tensors:
        arr = mx.atleast_3d(t._mlx_array)
        results.append(Tensor._from_mlx_array(arr))
    return results[0] if len(results) == 1 else results


def bitwise_and(input: Tensor, other: Union[Tensor, int]) -> Tensor:
    """Bitwise AND of two tensors."""
    other_arr = other._mlx_array if isinstance(other, Tensor) else other
    result = Tensor._from_mlx_array(mx.bitwise_and(input._mlx_array, other_arr))
    return result


def bitwise_or(input: Tensor, other: Union[Tensor, int]) -> Tensor:
    """Bitwise OR of two tensors."""
    other_arr = other._mlx_array if isinstance(other, Tensor) else other
    result = Tensor._from_mlx_array(mx.bitwise_or(input._mlx_array, other_arr))
    return result


def bitwise_xor(input: Tensor, other: Union[Tensor, int]) -> Tensor:
    """Bitwise XOR of two tensors."""
    other_arr = other._mlx_array if isinstance(other, Tensor) else other
    result = Tensor._from_mlx_array(mx.bitwise_xor(input._mlx_array, other_arr))
    return result


def bitwise_not(input: Tensor) -> Tensor:
    """Bitwise NOT of a tensor."""
    # Use the ~ operator which works for both integers and bools in MLX
    result = Tensor._from_mlx_array(~input._mlx_array)
    return result


def broadcast_to(input: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Broadcast tensor to a new shape."""
    result = Tensor._from_mlx_array(mx.broadcast_to(input._mlx_array, shape))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def concatenate(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along a dimension (alias for cat)."""
    arrays = [t._mlx_array for t in tensors]
    result = Tensor._from_mlx_array(mx.concatenate(arrays, axis=dim))
    if any(t.requires_grad for t in tensors):
        result.requires_grad = True
    return result


def conj(input: Tensor) -> Tensor:
    """Return complex conjugate."""
    result = Tensor._from_mlx_array(mx.conj(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def erf(input: Tensor) -> Tensor:
    """Compute error function element-wise."""
    result = Tensor._from_mlx_array(mx.erf(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def erfinv(input: Tensor) -> Tensor:
    """Compute inverse error function element-wise."""
    result = Tensor._from_mlx_array(mx.erfinv(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def negative(input: Tensor) -> Tensor:
    """Negate tensor (alias for neg)."""
    result = Tensor._from_mlx_array(mx.negative(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def positive(input: Tensor) -> Tensor:
    """Return tensor unchanged (identity for positive)."""
    # Just return a copy
    result = Tensor._from_mlx_array(input._mlx_array)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def real(input: Tensor) -> Tensor:
    """Return real part of complex tensor."""
    result = Tensor._from_mlx_array(mx.real(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def imag(input: Tensor) -> Tensor:
    """Return imaginary part of complex tensor."""
    result = Tensor._from_mlx_array(mx.imag(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def logaddexp(input: Tensor, other: Tensor) -> Tensor:
    """Compute log(exp(input) + exp(other)) element-wise."""
    result = Tensor._from_mlx_array(mx.logaddexp(input._mlx_array, other._mlx_array))
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
    return result


def sigmoid(input: Tensor) -> Tensor:
    """Apply sigmoid function (already exists but adding here for completeness)."""
    result = Tensor._from_mlx_array(mx.sigmoid(input._mlx_array))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Apply softmax function."""
    result = Tensor._from_mlx_array(mx.softmax(input._mlx_array, axis=dim))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def logsumexp(input: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Compute log of summed exponentials."""
    result = Tensor._from_mlx_array(mx.logsumexp(input._mlx_array, axis=dim, keepdims=keepdim))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def addmm(input: Tensor, mat1: Tensor, mat2: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Compute beta * input + alpha * (mat1 @ mat2)."""
    mm_result = mx.matmul(mat1._mlx_array, mat2._mlx_array)
    result_arr = beta * input._mlx_array + alpha * mm_result
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or mat1.requires_grad or mat2.requires_grad):
        result.requires_grad = True
    return result


def baddbmm(input: Tensor, batch1: Tensor, batch2: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Batch add and batch matrix multiply: beta * input + alpha * (batch1 @ batch2)."""
    bmm_result = mx.matmul(batch1._mlx_array, batch2._mlx_array)
    result_arr = beta * input._mlx_array + alpha * bmm_result
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or batch1.requires_grad or batch2.requires_grad):
        result.requires_grad = True
    return result


def mv(input: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector multiplication."""
    result = Tensor._from_mlx_array(mx.matmul(input._mlx_array, vec._mlx_array))
    if is_grad_enabled() and (input.requires_grad or vec.requires_grad):
        result.requires_grad = True
    return result


def addr(input: Tensor, vec1: Tensor, vec2: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Outer product and add: beta * input + alpha * (vec1 outer vec2)."""
    outer = mx.outer(vec1._mlx_array, vec2._mlx_array)
    result_arr = beta * input._mlx_array + alpha * outer
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or vec1.requires_grad or vec2.requires_grad):
        result.requires_grad = True
    return result


def numel(input: Tensor) -> int:
    """Return total number of elements in tensor."""
    return input._mlx_array.size


def addbmm(input: Tensor, batch1: Tensor, batch2: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Add batch matrix multiply: beta * input + alpha * sum(batch1[i] @ batch2[i]).

    Performs batch matrix multiply and sums over batch dimension, then adds to input.
    """
    # Batch matmul: (b, n, m) @ (b, m, p) -> (b, n, p)
    bmm_result = mx.matmul(batch1._mlx_array, batch2._mlx_array)
    # Sum over batch dimension
    summed = mx.sum(bmm_result, axis=0)
    result_arr = beta * input._mlx_array + alpha * summed
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or batch1.requires_grad or batch2.requires_grad):
        result.requires_grad = True
    return result


def addmv(input: Tensor, mat: Tensor, vec: Tensor, beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Add matrix-vector multiply: beta * input + alpha * (mat @ vec)."""
    mv_result = mx.matmul(mat._mlx_array, vec._mlx_array)
    result_arr = beta * input._mlx_array + alpha * mv_result
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or mat.requires_grad or vec.requires_grad):
        result.requires_grad = True
    return result


def chain_matmul(*matrices: Tensor) -> Tensor:
    """Chain matrix multiplication of a sequence of 2D tensors.

    Deprecated in PyTorch - use linalg.multi_dot instead.
    """
    if len(matrices) == 0:
        raise ValueError("chain_matmul requires at least one matrix")
    if len(matrices) == 1:
        return matrices[0]

    result_arr = matrices[0]._mlx_array
    for i in range(1, len(matrices)):
        result_arr = mx.matmul(result_arr, matrices[i]._mlx_array)

    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and any(m.requires_grad for m in matrices):
        result.requires_grad = True
    return result


def dist(input: Tensor, other: Tensor, p: float = 2.0) -> Tensor:
    """Compute the p-norm of (input - other).

    Args:
        input: First input tensor
        other: Second input tensor
        p: The order of norm (default: 2)

    Returns:
        Scalar tensor containing the p-norm of the difference
    """
    diff = input._mlx_array - other._mlx_array
    if p == float('inf'):
        result_arr = mx.max(mx.abs(diff))
    elif p == float('-inf'):
        result_arr = mx.min(mx.abs(diff))
    elif p == 0:
        result_arr = mx.sum(mx.not_equal(diff, 0).astype(mx.float32))
    elif p == 1:
        result_arr = mx.sum(mx.abs(diff))
    elif p == 2:
        result_arr = mx.sqrt(mx.sum(diff * diff))
    else:
        result_arr = mx.power(mx.sum(mx.power(mx.abs(diff), p)), 1.0 / p)

    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result.requires_grad = True
    return result


def corrcoef(input: Tensor) -> Tensor:
    """Estimate the Pearson correlation coefficient matrix.

    Args:
        input: 2D tensor where each row is a variable and each column is an observation

    Returns:
        Correlation coefficient matrix
    """
    # Compute covariance matrix
    mean = mx.mean(input._mlx_array, axis=1, keepdims=True)
    centered = input._mlx_array - mean
    n = input._mlx_array.shape[1]
    cov = mx.matmul(centered, centered.T) / (n - 1)

    # Compute standard deviations
    std = mx.sqrt(mx.diag(cov))
    std_outer = mx.outer(std, std)

    # Correlation = cov / (std_i * std_j)
    result_arr = cov / std_outer
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def cov(input: Tensor, correction: int = 1, fweights: Tensor = None, aweights: Tensor = None) -> Tensor:
    """Estimate covariance matrix.

    Args:
        input: 2D tensor where each row is a variable
        correction: Degrees of freedom correction (default: 1 for Bessel's correction)
        fweights: Frequency weights (optional)
        aweights: Analytic weights (optional)

    Returns:
        Covariance matrix
    """
    x = input._mlx_array
    n = x.shape[1]

    if fweights is not None or aweights is not None:
        # Weighted covariance
        if fweights is not None:
            w = fweights._mlx_array.astype(mx.float32)
        else:
            w = mx.ones((n,), dtype=mx.float32)

        if aweights is not None:
            w = w * aweights._mlx_array.astype(mx.float32)

        w_sum = mx.sum(w)
        mean = mx.sum(x * w, axis=1, keepdims=True) / w_sum
        centered = x - mean

        weighted_centered = centered * mx.sqrt(w)
        cov_matrix = mx.matmul(weighted_centered, weighted_centered.T) / (w_sum - correction)
    else:
        mean = mx.mean(x, axis=1, keepdims=True)
        centered = x - mean
        cov_matrix = mx.matmul(centered, centered.T) / (n - correction)

    result = Tensor._from_mlx_array(cov_matrix)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """Apply bilinear transformation: y = x1^T A x2 + b.

    Args:
        input1: (N, *, in1_features) tensor
        input2: (N, *, in2_features) tensor
        weight: (out_features, in1_features, in2_features) tensor
        bias: (out_features,) tensor (optional)

    Returns:
        (N, *, out_features) tensor
    """
    # input1: (*, in1), input2: (*, in2), weight: (out, in1, in2)
    # result: (*, out) where result[..., k] = sum_ij x1[..., i] * W[k, i, j] * x2[..., j]

    # Use einsum: batch over leading dims
    # result = einsum('...i,kij,...j->...k', input1, weight, input2)
    x1 = input1._mlx_array
    x2 = input2._mlx_array
    w = weight._mlx_array

    # Compute x1 @ W -> (*, out, in2)
    # Then element-wise multiply with x2 and sum

    # Reshape for computation
    orig_shape = x1.shape[:-1]
    in1 = x1.shape[-1]
    in2 = x2.shape[-1]
    out_features = w.shape[0]

    # Flatten batch dimensions
    x1_flat = mx.reshape(x1, (-1, in1))  # (B, in1)
    x2_flat = mx.reshape(x2, (-1, in2))  # (B, in2)

    # Compute bilinear: result[b, k] = sum_ij x1[b, i] * W[k, i, j] * x2[b, j]
    # Expand to 4D for broadcasting: x1[B, 1, in1, 1], w[1, out, in1, in2], x2[B, 1, 1, in2]
    x1_4d = x1_flat[:, None, :, None]  # [B, 1, in1, 1]
    w_4d = w[None, :, :, :]  # [1, out, in1, in2]
    x2_4d = x2_flat[:, None, None, :]  # [B, 1, 1, in2]

    # Multiply all three and sum over in1 and in2 dimensions
    product = x1_4d * w_4d * x2_4d  # [B, out, in1, in2]
    result_arr = mx.sum(product, axis=(2, 3))  # (B, out)

    # Reshape back
    result_arr = mx.reshape(result_arr, orig_shape + (out_features,))

    if bias is not None:
        result_arr = result_arr + bias._mlx_array

    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input1.requires_grad or input2.requires_grad or weight.requires_grad):
        result.requires_grad = True
    return result


def constant_pad_nd(input: Tensor, pad: Tuple[int, ...], value: float = 0.0) -> Tensor:
    """Pad tensor with constant value.

    Args:
        input: Input tensor
        pad: Padding sizes (last dim first, pairs of before/after)
        value: Padding value

    Returns:
        Padded tensor
    """
    # Convert PyTorch pad format (last dim first) to MLX format
    # PyTorch: (left, right) or (left, right, top, bottom) etc.
    # MLX: [(before_0, after_0), (before_1, after_1), ...]

    ndim = input.ndim
    pad_list = list(pad)

    # Pad list is pairs starting from last dimension
    # Convert to list of tuples per dimension
    mlx_pad = [(0, 0)] * ndim

    for i in range(0, len(pad_list), 2):
        dim_idx = ndim - 1 - (i // 2)
        if dim_idx >= 0:
            left = pad_list[i]
            right = pad_list[i + 1] if i + 1 < len(pad_list) else 0
            mlx_pad[dim_idx] = (left, right)

    result_arr = mx.pad(input._mlx_array, mlx_pad, constant_values=value)
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def ger(input: Tensor, vec2: Tensor) -> Tensor:
    """Outer product of two 1D tensors.

    Args:
        input: 1D input tensor
        vec2: 1D input tensor

    Returns:
        2D tensor containing outer product
    """
    result_arr = mx.outer(input._mlx_array, vec2._mlx_array)
    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and (input.requires_grad or vec2.requires_grad):
        result.requires_grad = True
    return result


def frobenius_norm(input: Tensor, dim: Union[int, Tuple[int, ...]] = None, keepdim: bool = False) -> Tensor:
    """Compute Frobenius norm.

    Args:
        input: Input tensor
        dim: Dimension(s) over which to compute norm
        keepdim: Whether to keep reduced dimensions

    Returns:
        Tensor with Frobenius norm
    """
    if dim is None:
        # Global Frobenius norm
        result_arr = mx.sqrt(mx.sum(input._mlx_array * input._mlx_array))
    else:
        if isinstance(dim, int):
            dim = (dim,)
        result_arr = mx.sqrt(mx.sum(input._mlx_array * input._mlx_array, axis=dim, keepdims=keepdim))

    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def frombuffer(buffer, dtype=None, count: int = -1, offset: int = 0) -> Tensor:
    """Create a tensor from a buffer object.

    Args:
        buffer: Buffer object (bytes, bytearray, etc.)
        dtype: Data type of the resulting tensor
        count: Number of elements to read (-1 for all)
        offset: Byte offset to start reading from

    Returns:
        1D tensor with data from buffer
    """
    import numpy as np
    from ..dtype import torch_dtype_to_numpy, numpy_to_mlx_dtype

    if dtype is None:
        np_dtype = np.float32
    else:
        np_dtype = torch_dtype_to_numpy(dtype)

    # Read from buffer
    arr = np.frombuffer(buffer, dtype=np_dtype, count=count, offset=offset)
    mlx_dtype = numpy_to_mlx_dtype(np_dtype)
    result = Tensor._from_mlx_array(mx.array(arr, dtype=mlx_dtype))
    return result


def binomial(count: Tensor, prob: Tensor) -> Tensor:
    """Draw samples from binomial distribution.

    Args:
        count: Number of trials (n)
        prob: Probability of success (p)

    Returns:
        Tensor with binomial samples
    """
    import numpy as np

    # Use numpy for binomial since MLX doesn't have it natively
    count_np = np.array(count._mlx_array)
    prob_np = np.array(prob._mlx_array)
    samples = np.random.binomial(count_np.astype(int), prob_np)
    result = Tensor._from_mlx_array(mx.array(samples, dtype=count._mlx_array.dtype))
    return result


def convolution(input: Tensor, weight: Tensor, bias: Tensor = None,
                stride: Union[int, Tuple[int, ...]] = 1,
                padding: Union[int, Tuple[int, ...]] = 0,
                dilation: Union[int, Tuple[int, ...]] = 1,
                transposed: bool = False,
                output_padding: Union[int, Tuple[int, ...]] = 0,
                groups: int = 1) -> Tensor:
    """Generic convolution operation.

    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        stride: Stride for convolution
        padding: Padding for input
        dilation: Dilation for kernel
        transposed: If True, perform transposed convolution
        output_padding: Additional padding for output (transposed conv only)
        groups: Number of groups

    Returns:
        Convolved tensor
    """
    # Delegate to appropriate conv function based on input dimensions
    ndim = input.ndim - 2  # Subtract batch and channel dims

    if ndim == 1:
        if transposed:
            from ..nn.functional import conv_transpose1d
            return conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        else:
            from ..ops.conv1d import conv1d
            return conv1d(input, weight, bias, stride, padding, dilation, groups)
    elif ndim == 2:
        if transposed:
            from ..nn.functional import conv_transpose2d
            return conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        else:
            from ..ops.convolution import conv2d
            return conv2d(input, weight, bias, stride, padding, dilation, groups)
    elif ndim == 3:
        if transposed:
            from ..nn.functional import conv_transpose3d
            return conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)
        else:
            from ..ops.conv3d import conv3d
            return conv3d(input, weight, bias, stride, padding, dilation, groups)
    else:
        raise ValueError(f"Unsupported number of spatial dimensions: {ndim}")


def affine_grid_generator(theta: Tensor, size: List[int], align_corners: bool = False) -> Tensor:
    """Generate 2D or 3D affine grid for grid_sample.

    Args:
        theta: Affine transformation matrices of shape (N, 2, 3) or (N, 3, 4)
        size: Output spatial size [N, C, H, W] or [N, C, D, H, W]
        align_corners: If True, align corners of input and output

    Returns:
        Grid of shape (N, H, W, 2) or (N, D, H, W, 3)
    """
    N = theta.shape[0]

    if len(size) == 4:
        # 2D case: size = [N, C, H, W]
        _, _, H, W = size
        # Create normalized coordinate grid
        if align_corners:
            y = mx.linspace(-1, 1, H)
            x = mx.linspace(-1, 1, W)
        else:
            y = mx.linspace(-1 + 1/H, 1 - 1/H, H)
            x = mx.linspace(-1 + 1/W, 1 - 1/W, W)

        yy, xx = mx.meshgrid(y, x, indexing='ij')
        ones = mx.ones_like(xx)
        grid = mx.stack([xx, yy, ones], axis=-1)  # (H, W, 3)
        grid = mx.reshape(grid, (H * W, 3))  # (H*W, 3)

        # Apply transformation: output = grid @ theta.T
        theta_arr = theta._mlx_array  # (N, 2, 3)
        # For each batch: output = grid @ theta[i].T
        results = []
        for i in range(N):
            transformed = mx.matmul(grid, mx.transpose(theta_arr[i]))  # (H*W, 2)
            results.append(transformed)
        result_arr = mx.stack(results, axis=0)  # (N, H*W, 2)
        result_arr = mx.reshape(result_arr, (N, H, W, 2))
    else:
        # 3D case: size = [N, C, D, H, W]
        _, _, D, H, W = size
        if align_corners:
            z = mx.linspace(-1, 1, D)
            y = mx.linspace(-1, 1, H)
            x = mx.linspace(-1, 1, W)
        else:
            z = mx.linspace(-1 + 1/D, 1 - 1/D, D)
            y = mx.linspace(-1 + 1/H, 1 - 1/H, H)
            x = mx.linspace(-1 + 1/W, 1 - 1/W, W)

        zz, yy, xx = mx.meshgrid(z, y, x, indexing='ij')
        ones = mx.ones_like(xx)
        grid = mx.stack([xx, yy, zz, ones], axis=-1)
        grid = mx.reshape(grid, (D * H * W, 4))

        theta_arr = theta._mlx_array  # (N, 3, 4)
        results = []
        for i in range(N):
            transformed = mx.matmul(grid, mx.transpose(theta_arr[i]))
            results.append(transformed)
        result_arr = mx.stack(results, axis=0)
        result_arr = mx.reshape(result_arr, (N, D, H, W, 3))

    result = Tensor._from_mlx_array(result_arr)
    if is_grad_enabled() and theta.requires_grad:
        result.requires_grad = True
    return result


__all__ = [
    'atleast_1d', 'atleast_2d', 'atleast_3d',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
    'broadcast_to', 'concatenate',
    'conj', 'erf', 'erfinv',
    'negative', 'positive',
    'real', 'imag',
    'logaddexp', 'logsumexp',
    'addmm', 'baddbmm', 'mv', 'addr',
    'numel',
    # New additions
    'addbmm', 'addmv', 'chain_matmul', 'dist',
    'corrcoef', 'cov', 'bilinear', 'constant_pad_nd',
    'ger', 'frobenius_norm', 'frombuffer', 'binomial',
    'convolution', 'affine_grid_generator',
    # In-place operations at torch.* level
    'dropout_', 'alpha_dropout_', 'feature_alpha_dropout_',
    'erf_', 'erfc_', 'exp2_', 'i0_', 'fill_',
    # Grid sampler
    'grid_sampler', 'grid_sampler_2d', 'grid_sampler_3d',
    # Histogram
    'histogram', 'histogramdd',
    # Other
    'feature_dropout', 'igamma', 'igammac', 'polygamma',
]


# In-place operations at torch.* level
def dropout_(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """In-place dropout."""
    if not training or p == 0:
        return input
    mask = mx.random.uniform(shape=input.shape) > p
    input._mlx_array = (input._mlx_array * mask) / (1 - p)
    return input


def alpha_dropout_(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """In-place alpha dropout (for SELU networks)."""
    if not training or p == 0:
        return input
    # Alpha dropout constants
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale

    mask = mx.random.uniform(shape=input.shape) > p
    # Apply alpha dropout transformation
    input._mlx_array = mx.where(mask, input._mlx_array, alpha_p)
    a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
    b = -a * alpha_p * p
    input._mlx_array = input._mlx_array * a + b
    return input


def feature_alpha_dropout_(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """In-place feature-wise alpha dropout."""
    if not training or p == 0:
        return input
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale

    # Dropout whole features (channels)
    feature_shape = (1, input.shape[1]) + (1,) * (input.ndim - 2)
    mask = mx.random.uniform(shape=feature_shape) > p
    mask = mx.broadcast_to(mask, input.shape)

    input._mlx_array = mx.where(mask, input._mlx_array, alpha_p)
    a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
    b = -a * alpha_p * p
    input._mlx_array = input._mlx_array * a + b
    return input


def erf_(input: Tensor) -> Tensor:
    """In-place error function."""
    input._mlx_array = mx.erf(input._mlx_array)
    return input


def erfc_(input: Tensor) -> Tensor:
    """In-place complementary error function."""
    input._mlx_array = 1 - mx.erf(input._mlx_array)
    return input


def exp2_(input: Tensor) -> Tensor:
    """In-place 2^x."""
    input._mlx_array = mx.power(2.0, input._mlx_array)
    return input


def i0_(input: Tensor) -> Tensor:
    """In-place modified Bessel function of first kind, order 0."""
    # MLX doesn't have i0 directly, approximate it
    import numpy as np
    arr = np.array(input._mlx_array)
    arr = np.i0(arr)
    input._mlx_array = mx.array(arr, dtype=input._mlx_array.dtype)
    return input


def fill_(input: Tensor, value: float) -> Tensor:
    """In-place fill with value."""
    input._mlx_array = mx.full(input.shape, value, dtype=input._mlx_array.dtype)
    return input


def grid_sampler(input: Tensor, grid: Tensor, interpolation_mode: int = 0,
                 padding_mode: int = 0, align_corners: bool = False) -> Tensor:
    """Sample input using grid coordinates.

    Args:
        input: Input tensor (N, C, H, W) or (N, C, D, H, W)
        grid: Grid tensor (N, H_out, W_out, 2) or (N, D_out, H_out, W_out, 3)
        interpolation_mode: 0=bilinear, 1=nearest, 2=bicubic
        padding_mode: 0=zeros, 1=border, 2=reflection
        align_corners: If True, align corner pixels

    Returns:
        Sampled tensor
    """
    # Convert mode integers to strings
    interp_modes = {0: 'bilinear', 1: 'nearest', 2: 'bicubic'}
    pad_modes = {0: 'zeros', 1: 'border', 2: 'reflection'}

    from ..nn.functional import grid_sample
    return grid_sample(input, grid, mode=interp_modes.get(interpolation_mode, 'bilinear'),
                       padding_mode=pad_modes.get(padding_mode, 'zeros'),
                       align_corners=align_corners)


def grid_sampler_2d(input: Tensor, grid: Tensor, interpolation_mode: int = 0,
                    padding_mode: int = 0, align_corners: bool = False) -> Tensor:
    """2D grid sampler. See grid_sampler for details."""
    return grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)


def grid_sampler_3d(input: Tensor, grid: Tensor, interpolation_mode: int = 0,
                    padding_mode: int = 0, align_corners: bool = False) -> Tensor:
    """3D grid sampler. See grid_sampler for details."""
    return grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)


def histogram(input: Tensor, bins: int = 100, range: Tuple[float, float] = None,
              weight: Tensor = None, density: bool = False) -> Tuple[Tensor, Tensor]:
    """Compute histogram of tensor values.

    Args:
        input: Input tensor
        bins: Number of histogram bins
        range: Range (min, max) for histogram
        weight: Weight for each value
        density: If True, normalize to form a density

    Returns:
        Tuple of (histogram values, bin edges)
    """
    import numpy as np

    x = np.array(input._mlx_array).flatten()
    w = np.array(weight._mlx_array).flatten() if weight is not None else None

    hist, edges = np.histogram(x, bins=bins, range=range, weights=w, density=density)

    hist_tensor = Tensor._from_mlx_array(mx.array(hist, dtype=mx.float32))
    edges_tensor = Tensor._from_mlx_array(mx.array(edges, dtype=mx.float32))
    return hist_tensor, edges_tensor


def histogramdd(input: Tensor, bins: int = 10, range: List[Tuple[float, float]] = None,
                weight: Tensor = None, density: bool = False):
    """Compute multi-dimensional histogram.

    Args:
        input: Input tensor of shape (N, D) where D is number of dimensions
        bins: Number of bins (int or list of ints per dimension)
        range: List of (min, max) per dimension
        weight: Weight tensor
        density: If True, normalize to form a density

    Returns:
        Tuple of (histogram, tuple of bin edge tensors)
    """
    import numpy as np

    x = np.array(input._mlx_array)
    w = np.array(weight._mlx_array) if weight is not None else None

    hist, edges = np.histogramdd(x, bins=bins, range=range, weights=w, density=density)

    hist_tensor = Tensor._from_mlx_array(mx.array(hist, dtype=mx.float32))
    # Return bin_edges as a tuple of 1D tensors (one per dimension), matching PyTorch
    edges_tensors = tuple(Tensor._from_mlx_array(mx.array(e, dtype=mx.float32)) for e in edges)
    return hist_tensor, edges_tensors


def feature_dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Feature-wise dropout (drops entire channels)."""
    if not training or p == 0:
        return input

    # Create mask for feature dimension
    feature_shape = (input.shape[0], input.shape[1]) + (1,) * (input.ndim - 2)
    mask = mx.random.uniform(shape=feature_shape) > p
    mask = mx.broadcast_to(mask, input.shape)

    result = Tensor._from_mlx_array((input._mlx_array * mask) / (1 - p))
    if is_grad_enabled() and input.requires_grad:
        result.requires_grad = True
    return result


def igamma(input: Tensor, other: Tensor) -> Tensor:
    """Regularized lower incomplete gamma function."""
    import numpy as np
    from scipy import special

    a = np.array(input._mlx_array)
    x = np.array(other._mlx_array)
    result = special.gammainc(a, x)
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result_tensor.requires_grad = True
    return result_tensor


def igammac(input: Tensor, other: Tensor) -> Tensor:
    """Regularized upper incomplete gamma function."""
    import numpy as np
    from scipy import special

    a = np.array(input._mlx_array)
    x = np.array(other._mlx_array)
    result = special.gammaincc(a, x)
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    if is_grad_enabled() and (input.requires_grad or other.requires_grad):
        result_tensor.requires_grad = True
    return result_tensor


def polygamma(n: int, input: Tensor) -> Tensor:
    """Compute the nth derivative of the digamma function."""
    import numpy as np
    from scipy import special

    x = np.array(input._mlx_array)
    result = special.polygamma(n, x)
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    if is_grad_enabled() and input.requires_grad:
        result_tensor.requires_grad = True
    return result_tensor


# ============================================================================
# RNN Functions (torch.lstm, torch.gru)
# ============================================================================

def _lstm_cell(input_arr, h_arr, c_arr, weight_ih, weight_hh, bias_ih, bias_hh, has_biases):
    """Apply a single LSTM cell step.

    PyTorch gate ordering: i, f, g, o (input, forget, cell, output)
    """
    # Compute gates: W_ih @ x + W_hh @ h
    igates = mx.matmul(input_arr, weight_ih.T)
    hgates = mx.matmul(h_arr, weight_hh.T)

    if has_biases:
        gates = igates + bias_ih + hgates + bias_hh
    else:
        gates = igates + hgates

    # Split into 4 gates (i, f, g, o)
    hidden_size = h_arr.shape[-1]
    i = mx.sigmoid(gates[..., :hidden_size])  # input gate
    f = mx.sigmoid(gates[..., hidden_size:2*hidden_size])  # forget gate
    g = mx.tanh(gates[..., 2*hidden_size:3*hidden_size])  # cell gate
    o = mx.sigmoid(gates[..., 3*hidden_size:])  # output gate

    # Update cell and hidden state
    c_new = f * c_arr + i * g
    h_new = o * mx.tanh(c_new)

    return h_new, c_new


def _gru_cell(input_arr, h_arr, weight_ih, weight_hh, bias_ih, bias_hh, has_biases):
    """Apply a single GRU cell step.

    PyTorch gate ordering: r, z, n (reset, update, new)
    """
    # Compute input and hidden gates separately
    igates = mx.matmul(input_arr, weight_ih.T)
    hgates = mx.matmul(h_arr, weight_hh.T)

    if has_biases:
        igates = igates + bias_ih
        hgates = hgates + bias_hh

    hidden_size = h_arr.shape[-1]

    # Split gates
    ir, iz, in_ = (igates[..., :hidden_size],
                   igates[..., hidden_size:2*hidden_size],
                   igates[..., 2*hidden_size:])
    hr, hz, hn = (hgates[..., :hidden_size],
                  hgates[..., hidden_size:2*hidden_size],
                  hgates[..., 2*hidden_size:])

    # Apply gate activations
    r = mx.sigmoid(ir + hr)  # reset gate
    z = mx.sigmoid(iz + hz)  # update gate
    n = mx.tanh(in_ + r * hn)  # new gate

    # Update hidden state: h_new = (1 - z) * n + z * h
    h_new = (1 - z) * n + z * h_arr

    return h_new


def lstm(data: Tensor, batch_sizes: Tensor, hx: Tuple[Tensor, Tensor],
         params: List[Tensor], has_biases: bool, num_layers: int,
         dropout: float, training: bool, bidirectional: bool) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply LSTM to packed sequence.

    This is a lower-level LSTM function used by nn.LSTM for packed sequences.

    Note: This function expects packed sequence data. For regular tensors,
    use nn.LSTM instead.

    Args:
        data: Packed input data (total_timesteps, input_size)
        batch_sizes: Batch sizes for each timestep in packed sequence
        hx: Initial hidden state (h_0, c_0), each (num_layers * num_directions, batch, hidden_size)
        params: List of weight/bias tensors in order:
                For each layer, for each direction:
                    weight_ih, weight_hh, [bias_ih, bias_hh if has_biases]
        has_biases: Whether biases are included
        num_layers: Number of LSTM layers
        dropout: Dropout probability (applied between layers, not on last layer)
        training: Training mode (affects dropout)
        bidirectional: If bidirectional LSTM

    Returns:
        Tuple of (output, h_n, c_n) matching PyTorch's signature
    """
    h_0, c_0 = hx
    hidden_size = h_0.shape[-1]
    num_directions = 2 if bidirectional else 1
    batch_size = int(batch_sizes._mlx_array[0])

    # Get batch sizes as numpy array for iteration
    batch_sizes_arr = [int(b) for b in batch_sizes._mlx_array.tolist()]
    total_timesteps = len(batch_sizes_arr)

    # Number of parameters per layer per direction
    params_per_layer_dir = 4 if has_biases else 2

    # Initialize output list
    output_list = []

    # Initialize hidden states for all layers
    h_list = []
    c_list = []
    for layer in range(num_layers * num_directions):
        h_list.append(h_0._mlx_array[layer])
        c_list.append(c_0._mlx_array[layer])

    # Process packed sequence
    data_arr = data._mlx_array
    data_offset = 0

    for layer in range(num_layers):
        layer_output = []

        # Get parameters for this layer
        param_offset = layer * num_directions * params_per_layer_dir

        # Forward direction
        weight_ih_f = params[param_offset]._mlx_array
        weight_hh_f = params[param_offset + 1]._mlx_array
        if has_biases:
            bias_ih_f = params[param_offset + 2]._mlx_array
            bias_hh_f = params[param_offset + 3]._mlx_array
        else:
            bias_ih_f = None
            bias_hh_f = None

        h_f = h_list[layer * num_directions]
        c_f = c_list[layer * num_directions]

        # Process each timestep in packed sequence
        if layer == 0:
            # First layer: use input data
            timestep_offset = 0
            for t in range(total_timesteps):
                curr_batch = batch_sizes_arr[t]
                x_t = data_arr[timestep_offset:timestep_offset + curr_batch]

                # Process only active sequences
                h_active = h_f[:curr_batch]
                c_active = c_f[:curr_batch]

                h_new, c_new = _lstm_cell(x_t, h_active, c_active,
                                          weight_ih_f, weight_hh_f,
                                          bias_ih_f, bias_hh_f, has_biases)

                # Update hidden states
                h_f = mx.concatenate([h_new, h_f[curr_batch:]], axis=0) if curr_batch < h_f.shape[0] else h_new
                c_f = mx.concatenate([c_new, c_f[curr_batch:]], axis=0) if curr_batch < c_f.shape[0] else c_new

                layer_output.append(h_new)
                timestep_offset += curr_batch
        else:
            # Later layers: use previous layer output
            timestep_offset = 0
            for t in range(total_timesteps):
                curr_batch = batch_sizes_arr[t]
                x_t = prev_layer_output[timestep_offset:timestep_offset + curr_batch]

                h_active = h_f[:curr_batch]
                c_active = c_f[:curr_batch]

                h_new, c_new = _lstm_cell(x_t, h_active, c_active,
                                          weight_ih_f, weight_hh_f,
                                          bias_ih_f, bias_hh_f, has_biases)

                h_f = mx.concatenate([h_new, h_f[curr_batch:]], axis=0) if curr_batch < h_f.shape[0] else h_new
                c_f = mx.concatenate([c_new, c_f[curr_batch:]], axis=0) if curr_batch < c_f.shape[0] else c_new

                layer_output.append(h_new)
                timestep_offset += curr_batch

        h_list[layer * num_directions] = h_f
        c_list[layer * num_directions] = c_f

        # Bidirectional processing
        if bidirectional:
            param_offset_b = param_offset + params_per_layer_dir
            weight_ih_b = params[param_offset_b]._mlx_array
            weight_hh_b = params[param_offset_b + 1]._mlx_array
            if has_biases:
                bias_ih_b = params[param_offset_b + 2]._mlx_array
                bias_hh_b = params[param_offset_b + 3]._mlx_array
            else:
                bias_ih_b = None
                bias_hh_b = None

            h_b = h_list[layer * num_directions + 1]
            c_b = c_list[layer * num_directions + 1]

            layer_output_b = []

            # Process in reverse order
            if layer == 0:
                timestep_offset = sum(batch_sizes_arr)
                for t in range(total_timesteps - 1, -1, -1):
                    curr_batch = batch_sizes_arr[t]
                    timestep_offset -= curr_batch
                    x_t = data_arr[timestep_offset:timestep_offset + curr_batch]

                    h_active = h_b[:curr_batch]
                    c_active = c_b[:curr_batch]

                    h_new, c_new = _lstm_cell(x_t, h_active, c_active,
                                              weight_ih_b, weight_hh_b,
                                              bias_ih_b, bias_hh_b, has_biases)

                    h_b = mx.concatenate([h_new, h_b[curr_batch:]], axis=0) if curr_batch < h_b.shape[0] else h_new
                    c_b = mx.concatenate([c_new, c_b[curr_batch:]], axis=0) if curr_batch < c_b.shape[0] else c_new

                    layer_output_b.insert(0, h_new)
            else:
                timestep_offset = sum(batch_sizes_arr)
                for t in range(total_timesteps - 1, -1, -1):
                    curr_batch = batch_sizes_arr[t]
                    timestep_offset -= curr_batch
                    # For bidirectional later layers, input comes from concatenated forward+backward
                    x_t = prev_layer_output[timestep_offset:timestep_offset + curr_batch]

                    h_active = h_b[:curr_batch]
                    c_active = c_b[:curr_batch]

                    h_new, c_new = _lstm_cell(x_t, h_active, c_active,
                                              weight_ih_b, weight_hh_b,
                                              bias_ih_b, bias_hh_b, has_biases)

                    h_b = mx.concatenate([h_new, h_b[curr_batch:]], axis=0) if curr_batch < h_b.shape[0] else h_new
                    c_b = mx.concatenate([c_new, c_b[curr_batch:]], axis=0) if curr_batch < c_b.shape[0] else c_new

                    layer_output_b.insert(0, h_new)

            h_list[layer * num_directions + 1] = h_b
            c_list[layer * num_directions + 1] = c_b

            # Concatenate forward and backward outputs
            layer_output = [mx.concatenate([f, b], axis=-1)
                           for f, b in zip(layer_output, layer_output_b)]

        prev_layer_output = mx.concatenate(layer_output, axis=0)

        # Apply dropout between layers (not on last layer)
        if dropout > 0 and training and layer < num_layers - 1:
            mask = mx.random.uniform(shape=prev_layer_output.shape) > dropout
            prev_layer_output = (prev_layer_output * mask) / (1 - dropout)

    # Stack final hidden states
    output = Tensor._from_mlx_array(prev_layer_output)
    h_n = Tensor._from_mlx_array(mx.stack(h_list, axis=0))
    c_n = Tensor._from_mlx_array(mx.stack(c_list, axis=0))

    return output, h_n, c_n


def gru(data: Tensor, batch_sizes: Tensor, hx: Tensor,
        params: List[Tensor], has_biases: bool, num_layers: int,
        dropout: float, training: bool, bidirectional: bool) -> Tuple[Tensor, Tensor]:
    """Apply GRU to packed sequence.

    This is a lower-level GRU function used by nn.GRU for packed sequences.

    Note: This function expects packed sequence data. For regular tensors,
    use nn.GRU instead.

    Args:
        data: Packed input data (total_timesteps, input_size)
        batch_sizes: Batch sizes for each timestep in packed sequence
        hx: Initial hidden state h_0 (num_layers * num_directions, batch, hidden_size)
        params: List of weight/bias tensors in order:
                For each layer, for each direction:
                    weight_ih, weight_hh, [bias_ih, bias_hh if has_biases]
        has_biases: Whether biases are included
        num_layers: Number of GRU layers
        dropout: Dropout probability (applied between layers, not on last layer)
        training: Training mode (affects dropout)
        bidirectional: If bidirectional GRU

    Returns:
        Tuple of (output, h_n)
    """
    hidden_size = hx.shape[-1]
    num_directions = 2 if bidirectional else 1
    batch_size = int(batch_sizes._mlx_array[0])

    # Get batch sizes as list for iteration
    batch_sizes_arr = [int(b) for b in batch_sizes._mlx_array.tolist()]
    total_timesteps = len(batch_sizes_arr)

    # Number of parameters per layer per direction
    params_per_layer_dir = 4 if has_biases else 2

    # Initialize hidden states for all layers
    h_list = []
    for layer in range(num_layers * num_directions):
        h_list.append(hx._mlx_array[layer])

    # Process packed sequence
    data_arr = data._mlx_array

    for layer in range(num_layers):
        layer_output = []

        # Get parameters for this layer
        param_offset = layer * num_directions * params_per_layer_dir

        # Forward direction
        weight_ih_f = params[param_offset]._mlx_array
        weight_hh_f = params[param_offset + 1]._mlx_array
        if has_biases:
            bias_ih_f = params[param_offset + 2]._mlx_array
            bias_hh_f = params[param_offset + 3]._mlx_array
        else:
            bias_ih_f = None
            bias_hh_f = None

        h_f = h_list[layer * num_directions]

        # Process each timestep in packed sequence
        if layer == 0:
            timestep_offset = 0
            for t in range(total_timesteps):
                curr_batch = batch_sizes_arr[t]
                x_t = data_arr[timestep_offset:timestep_offset + curr_batch]

                h_active = h_f[:curr_batch]

                h_new = _gru_cell(x_t, h_active, weight_ih_f, weight_hh_f,
                                  bias_ih_f, bias_hh_f, has_biases)

                h_f = mx.concatenate([h_new, h_f[curr_batch:]], axis=0) if curr_batch < h_f.shape[0] else h_new

                layer_output.append(h_new)
                timestep_offset += curr_batch
        else:
            timestep_offset = 0
            for t in range(total_timesteps):
                curr_batch = batch_sizes_arr[t]
                x_t = prev_layer_output[timestep_offset:timestep_offset + curr_batch]

                h_active = h_f[:curr_batch]

                h_new = _gru_cell(x_t, h_active, weight_ih_f, weight_hh_f,
                                  bias_ih_f, bias_hh_f, has_biases)

                h_f = mx.concatenate([h_new, h_f[curr_batch:]], axis=0) if curr_batch < h_f.shape[0] else h_new

                layer_output.append(h_new)
                timestep_offset += curr_batch

        h_list[layer * num_directions] = h_f

        # Bidirectional processing
        if bidirectional:
            param_offset_b = param_offset + params_per_layer_dir
            weight_ih_b = params[param_offset_b]._mlx_array
            weight_hh_b = params[param_offset_b + 1]._mlx_array
            if has_biases:
                bias_ih_b = params[param_offset_b + 2]._mlx_array
                bias_hh_b = params[param_offset_b + 3]._mlx_array
            else:
                bias_ih_b = None
                bias_hh_b = None

            h_b = h_list[layer * num_directions + 1]

            layer_output_b = []

            # Process in reverse order
            if layer == 0:
                timestep_offset = sum(batch_sizes_arr)
                for t in range(total_timesteps - 1, -1, -1):
                    curr_batch = batch_sizes_arr[t]
                    timestep_offset -= curr_batch
                    x_t = data_arr[timestep_offset:timestep_offset + curr_batch]

                    h_active = h_b[:curr_batch]

                    h_new = _gru_cell(x_t, h_active, weight_ih_b, weight_hh_b,
                                      bias_ih_b, bias_hh_b, has_biases)

                    h_b = mx.concatenate([h_new, h_b[curr_batch:]], axis=0) if curr_batch < h_b.shape[0] else h_new

                    layer_output_b.insert(0, h_new)
            else:
                timestep_offset = sum(batch_sizes_arr)
                for t in range(total_timesteps - 1, -1, -1):
                    curr_batch = batch_sizes_arr[t]
                    timestep_offset -= curr_batch
                    x_t = prev_layer_output[timestep_offset:timestep_offset + curr_batch]

                    h_active = h_b[:curr_batch]

                    h_new = _gru_cell(x_t, h_active, weight_ih_b, weight_hh_b,
                                      bias_ih_b, bias_hh_b, has_biases)

                    h_b = mx.concatenate([h_new, h_b[curr_batch:]], axis=0) if curr_batch < h_b.shape[0] else h_new

                    layer_output_b.insert(0, h_new)

            h_list[layer * num_directions + 1] = h_b

            # Concatenate forward and backward outputs
            layer_output = [mx.concatenate([f, b], axis=-1)
                           for f, b in zip(layer_output, layer_output_b)]

        prev_layer_output = mx.concatenate(layer_output, axis=0)

        # Apply dropout between layers (not on last layer)
        if dropout > 0 and training and layer < num_layers - 1:
            mask = mx.random.uniform(shape=prev_layer_output.shape) > dropout
            prev_layer_output = (prev_layer_output * mask) / (1 - dropout)

    # Stack final hidden states
    output = Tensor._from_mlx_array(prev_layer_output)
    h_n = Tensor._from_mlx_array(mx.stack(h_list, axis=0))

    return output, h_n


# ============================================================================
# Matrix Functions
# ============================================================================

def logdet(input: Tensor) -> Tensor:
    """Compute log determinant of a matrix.

    More numerically stable than det().log() for positive-definite matrices.
    """
    import numpy as np

    arr = np.array(input._mlx_array)
    sign, logabsdet = np.linalg.slogdet(arr)
    # For positive definite matrices, sign should be 1
    result = logabsdet if sign > 0 else float('-inf')
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    if is_grad_enabled() and input.requires_grad:
        result_tensor.requires_grad = True
    return result_tensor


def matrix_exp(input: Tensor) -> Tensor:
    """Compute matrix exponential."""
    import numpy as np
    from scipy import linalg

    arr = np.array(input._mlx_array)
    if arr.ndim == 2:
        result = linalg.expm(arr)
    else:
        # Batch matrix exponential
        result = np.stack([linalg.expm(m) for m in arr])

    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    if is_grad_enabled() and input.requires_grad:
        result_tensor.requires_grad = True
    return result_tensor


def matrix_power(input: Tensor, n: int) -> Tensor:
    """Compute matrix to the nth power."""
    import numpy as np

    arr = np.array(input._mlx_array)
    result = np.linalg.matrix_power(arr, n)
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    if is_grad_enabled() and input.requires_grad:
        result_tensor.requires_grad = True
    return result_tensor


# ============================================================================
# NaN-aware Operations
# ============================================================================

def nanmedian(input: Tensor, dim: int = None, keepdim: bool = False):
    """Compute median, ignoring NaN values.

    PyTorch behavior:
    - For even count of non-NaN values, returns the LOWER of the two middle values
    - When dim is specified, also returns the index of the median value
    """
    import numpy as np

    arr = np.array(input._mlx_array)

    if dim is None:
        # Flatten, filter NaN, find lower-middle element
        flat = arr.flatten()
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            result_tensor = Tensor._from_mlx_array(mx.array(np.nan, dtype=input._mlx_array.dtype))
            return result_tensor
        sorted_valid = np.sort(valid)
        mid_idx = (len(sorted_valid) - 1) // 2
        result = sorted_valid[mid_idx]
        result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
        return result_tensor

    # For dim specified, compute along that dimension with PyTorch lower-middle convention
    def pytorch_nanmedian_1d(x):
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return np.nan
        sorted_valid = np.sort(valid)
        mid_idx = (len(sorted_valid) - 1) // 2
        return sorted_valid[mid_idx]

    result = np.apply_along_axis(pytorch_nanmedian_1d, dim, arr)
    if keepdim:
        result = np.expand_dims(result, axis=dim)
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    return result_tensor


def nanquantile(input: Tensor, q: float, dim: int = None, keepdim: bool = False,
                interpolation: str = 'linear') -> Tensor:
    """Compute quantile, ignoring NaN values."""
    import numpy as np

    arr = np.array(input._mlx_array)
    result = np.nanquantile(arr, q, axis=dim, keepdims=keepdim, method=interpolation)
    result_tensor = Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    return result_tensor


# ============================================================================
# Strided Operations
# ============================================================================

def as_strided(input: Tensor, size: Tuple[int, ...], stride: Tuple[int, ...],
               storage_offset: int = 0) -> Tensor:
    """Create a view with specified size and strides.

    Warning: This is an advanced operation that can create overlapping views.
    """
    import numpy as np

    # Convert to numpy and use np.lib.stride_tricks
    arr = np.array(input._mlx_array)
    flat = arr.flatten()

    if storage_offset > 0:
        flat = flat[storage_offset:]

    # Calculate byte strides from element strides
    itemsize = arr.dtype.itemsize
    byte_strides = tuple(s * itemsize for s in stride)

    result = np.lib.stride_tricks.as_strided(flat, shape=size, strides=byte_strides)
    result_tensor = Tensor._from_mlx_array(mx.array(result.copy(), dtype=input._mlx_array.dtype))
    return result_tensor


def as_strided_(input: Tensor, size: Tuple[int, ...], stride: Tuple[int, ...],
                storage_offset: int = 0) -> Tensor:
    """In-place version of as_strided."""
    result = as_strided(input, size, stride, storage_offset)
    input._mlx_array = result._mlx_array
    return input


def as_strided_scatter(input: Tensor, src: Tensor, size: Tuple[int, ...],
                       stride: Tuple[int, ...], storage_offset: int = 0) -> Tensor:
    """Scatter values from src into input using as_strided view."""
    import numpy as np

    # This is a complex operation - use numpy for correctness
    arr = np.array(input._mlx_array).copy()
    src_arr = np.array(src._mlx_array)
    original_shape = arr.shape

    # Create a contiguous flat array to work with
    flat = arr.ravel()  # Use ravel to get a flattened view if possible
    itemsize = arr.dtype.itemsize
    byte_strides = tuple(s * itemsize for s in stride)

    # Create strided view into flat array starting at storage_offset
    view = np.lib.stride_tricks.as_strided(
        flat[storage_offset:], shape=size, strides=byte_strides, writeable=True
    )
    np.copyto(view, src_arr)

    # Reshape flat back to original shape
    result_arr = flat.reshape(original_shape)

    result_tensor = Tensor._from_mlx_array(mx.array(result_arr, dtype=input._mlx_array.dtype))
    return result_tensor


def empty_permuted(size: Tuple[int, ...], physical_layout: Tuple[int, ...],
                   dtype=None, device=None) -> Tensor:
    """Create empty tensor with specified physical layout."""
    if dtype is None:
        dtype = mx.float32
    elif hasattr(dtype, '_mlx_dtype'):
        dtype = dtype._mlx_dtype
    elif not isinstance(dtype, mx.Dtype):
        dtype = mx.float32

    # Create tensor and permute
    arr = mx.zeros(size, dtype=dtype)
    result = Tensor._from_mlx_array(arr)
    return result


def empty_strided(size: Tuple[int, ...], stride: Tuple[int, ...],
                  dtype=None, device=None) -> Tensor:
    """Create empty tensor with specified strides."""
    if dtype is None:
        dtype = mx.float32
    elif hasattr(dtype, '_mlx_dtype'):
        dtype = dtype._mlx_dtype
    elif not isinstance(dtype, mx.Dtype):
        dtype = mx.float32

    # MLX doesn't support custom strides, just create contiguous
    arr = mx.zeros(size, dtype=dtype)
    return Tensor._from_mlx_array(arr)


def nonzero_static(input: Tensor, size: int, fill_value: int = -1) -> Tensor:
    """Return indices of nonzero elements with static output size.

    Args:
        input: Input tensor
        size: Fixed size of output (padded with fill_value if needed)
        fill_value: Value to use for padding

    Returns:
        Tensor of shape (size, input.ndim) containing indices
    """
    import numpy as np

    arr = np.array(input._mlx_array)
    indices = np.argwhere(arr != 0)

    # Pad or truncate to size
    if len(indices) < size:
        padding = np.full((size - len(indices), arr.ndim), fill_value)
        indices = np.vstack([indices, padding])
    else:
        indices = indices[:size]

    return Tensor._from_mlx_array(mx.array(indices, dtype=mx.int64))


# ============================================================================
# Scatter/Index Operations
# ============================================================================

def index_put_(input: Tensor, indices: Tuple[Tensor, ...], values: Tensor,
               accumulate: bool = False) -> Tensor:
    """In-place put values at indices.

    Args:
        input: Input tensor to modify
        indices: Tuple of index tensors
        values: Values to put
        accumulate: If True, add to existing values instead of replacing

    Returns:
        Modified input tensor
    """
    import numpy as np

    arr = np.array(input._mlx_array)
    vals = np.array(values._mlx_array)

    # Convert indices
    idx = tuple(np.array(i._mlx_array) for i in indices)

    if accumulate:
        np.add.at(arr, idx, vals)
    else:
        arr[idx] = vals

    input._mlx_array = mx.array(arr, dtype=input._mlx_array.dtype)
    return input


def index_reduce(input: Tensor, dim: int, index: Tensor, source: Tensor,
                 reduce: str, include_self: bool = True) -> Tensor:
    """Reduce into input at indices along dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to index
        index: Index tensor
        source: Source tensor
        reduce: Reduction operation ('prod', 'mean', 'amax', 'amin')
        include_self: Include input values in reduction

    Returns:
        Result tensor
    """
    import numpy as np

    arr = np.array(input._mlx_array).copy()
    idx = np.array(index._mlx_array)
    src = np.array(source._mlx_array)

    if reduce == 'prod':
        if include_self:
            np.multiply.at(arr, (slice(None),) * dim + (idx,), src)
        else:
            # Set to 1 first then multiply
            arr[(slice(None),) * dim + (idx,)] = 1
            np.multiply.at(arr, (slice(None),) * dim + (idx,), src)
    elif reduce == 'mean':
        # Count and sum
        counts = np.zeros_like(arr)
        np.add.at(counts, (slice(None),) * dim + (idx,), 1)
        np.add.at(arr, (slice(None),) * dim + (idx,), src)
        if include_self:
            counts += 1
        arr = np.divide(arr, np.maximum(counts, 1))
    elif reduce == 'amax':
        np.maximum.at(arr, (slice(None),) * dim + (idx,), src)
    elif reduce == 'amin':
        np.minimum.at(arr, (slice(None),) * dim + (idx,), src)

    return Tensor._from_mlx_array(mx.array(arr, dtype=input._mlx_array.dtype))


def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """Scatter source values into input where mask is True."""
    import numpy as np

    arr = np.array(input._mlx_array).copy()
    m = np.array(mask._mlx_array).astype(bool)
    src = np.array(source._mlx_array).flatten()

    # Get number of True values in mask
    num_true = m.sum()
    arr[m] = src[:num_true]

    return Tensor._from_mlx_array(mx.array(arr, dtype=input._mlx_array.dtype))


# ============================================================================
# Linear Algebra Extensions
# ============================================================================

def cholesky_inverse(input: Tensor, upper: bool = False) -> Tensor:
    """Compute inverse of symmetric positive-definite matrix from Cholesky factor."""
    import numpy as np
    from scipy import linalg

    arr = np.array(input._mlx_array)
    result = linalg.cho_solve((arr, not upper), np.eye(arr.shape[-1]))
    return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))


def cholesky_solve(b: Tensor, u: Tensor, upper: bool = False) -> Tensor:
    """Solve linear system with Cholesky-factorized coefficient matrix."""
    import numpy as np
    from scipy import linalg

    b_arr = np.array(b._mlx_array)
    u_arr = np.array(u._mlx_array)
    result = linalg.cho_solve((u_arr, not upper), b_arr)
    return Tensor._from_mlx_array(mx.array(result, dtype=b._mlx_array.dtype))


def lu_solve(b: Tensor, LU_data: Tensor, LU_pivots: Tensor, *, out=None) -> Tensor:
    """Solve linear system Ax = b using LU factorization.

    This matches the signature of torch.lu_solve.

    Args:
        b: Right-hand side vector/matrix of shape (*, m, k)
        LU_data: Combined LU matrix from lu_factor of shape (*, m, m)
        LU_pivots: Pivot indices (1-indexed, as returned by torch.lu_factor)
        out: Optional output tensor (ignored)

    Returns:
        Solution tensor x
    """
    import numpy as np

    lu_arr = np.array(LU_data._mlx_array).astype(np.float64)
    piv_arr = np.array(LU_pivots._mlx_array).astype(np.int64)
    b_arr = np.array(b._mlx_array).astype(np.float64)

    # Convert PyTorch 1-indexed pivots to 0-indexed for scipy
    piv_arr_0indexed = piv_arr - 1

    # scipy.linalg.lu_solve expects (lu, piv) tuple and b
    from scipy import linalg
    result = linalg.lu_solve((lu_arr, piv_arr_0indexed), b_arr)

    return Tensor._from_mlx_array(mx.array(result.astype(np.float32), dtype=b._mlx_array.dtype))


def lu_unpack(LU_data: Tensor, LU_pivots: Tensor, unpack_data: bool = True,
              unpack_pivots: bool = True):
    """Unpack LU factorization.

    PyTorch's pivots are 1-indexed (values from 1 to n), representing row swaps
    performed during LU factorization.

    Args:
        LU_data: Combined LU matrix from lu_factor
        LU_pivots: Pivot indices (1-indexed)
        unpack_data: If True, return L and U matrices
        unpack_pivots: If True, return permutation matrix P

    Returns:
        Tuple of (P, L, U) if both unpack options are True
    """
    import numpy as np

    lu_arr = np.array(LU_data._mlx_array)
    piv_arr = np.array(LU_pivots._mlx_array).astype(np.int64)

    m, n = lu_arr.shape[-2], lu_arr.shape[-1]
    k = min(m, n)

    # Extract L and U
    # L is lower triangular part with 1s on diagonal, shape (m, k)
    L = np.tril(lu_arr[:, :k], -1)
    np.fill_diagonal(L, 1)

    # U is upper triangular part, shape (k, n)
    U = np.triu(lu_arr[:k, :])

    # Create permutation from pivots
    # PyTorch pivots are 1-indexed, so we need to subtract 1
    # Apply swaps in REVERSE order to get the correct permutation matrix
    perm = list(range(m))
    for i in reversed(range(len(piv_arr))):
        p = int(piv_arr[i]) - 1  # Convert to 0-indexed
        if 0 <= p < m:
            perm[i], perm[p] = perm[p], perm[i]

    # Build permutation matrix from perm array
    # P[i, perm[i]] = 1 means row i of output corresponds to row perm[i] of input
    P = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        P[i, perm[i]] = 1

    results = []
    if unpack_pivots:
        results.append(Tensor._from_mlx_array(mx.array(P, dtype=LU_data._mlx_array.dtype)))
    if unpack_data:
        results.append(Tensor._from_mlx_array(mx.array(L, dtype=LU_data._mlx_array.dtype)))
        results.append(Tensor._from_mlx_array(mx.array(U, dtype=LU_data._mlx_array.dtype)))

    return tuple(results)


def geqrf(input: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute QR factorization in LAPACK geqrf format.

    Returns the QR factorization in compact Householder form, which can be
    used with orgqr to reconstruct Q or ormqr to apply Q to another matrix.

    Args:
        input: Input matrix of shape (m, n)

    Returns:
        Tuple of (qr, tau) where:
        - qr: Matrix containing the Householder vectors below the diagonal
              and R on and above the diagonal
        - tau: Householder reflector coefficients
    """
    import numpy as np
    from scipy.linalg import lapack

    arr = np.array(input._mlx_array).astype(np.float64)

    # Use LAPACK dgeqrf to compute QR in Householder form
    # dgeqrf(a, lwork=None, overwrite_a=False)
    # Returns: (qr, tau, work, info)
    qr, tau, work, info = lapack.dgeqrf(arr)

    if info != 0:
        raise RuntimeError(f"LAPACK dgeqrf failed with info={info}")

    return (Tensor._from_mlx_array(mx.array(qr.astype(np.float32), dtype=input._mlx_array.dtype)),
            Tensor._from_mlx_array(mx.array(tau.astype(np.float32), dtype=input._mlx_array.dtype)))


# ============================================================================
# Special Math Functions
# ============================================================================

def mvlgamma(input: Tensor, p: int) -> Tensor:
    """Compute multivariate log-gamma function."""
    import numpy as np
    from scipy import special

    arr = np.array(input._mlx_array)
    result = special.multigammaln(arr, p)
    return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))


def ldexp_(input: Tensor, other: Tensor) -> Tensor:
    """In-place ldexp (input * 2^other)."""
    import numpy as np

    arr = np.array(input._mlx_array)
    exp = np.array(other._mlx_array)
    result = np.ldexp(arr, exp.astype(int))
    input._mlx_array = mx.array(result, dtype=input._mlx_array.dtype)
    return input


def embedding_renorm_(weight: Tensor, input: Tensor, max_norm: float,
                      norm_type: float = 2.0) -> Tensor:
    """In-place renormalize embeddings.

    Args:
        weight: Embedding weight matrix
        input: Tensor of indices to renormalize
        max_norm: Max norm for embeddings
        norm_type: Norm type (default: 2.0)
    """
    import numpy as np

    w = np.array(weight._mlx_array)
    idx = np.unique(np.array(input._mlx_array).astype(int))

    for i in idx:
        norm = np.linalg.norm(w[i], ord=norm_type)
        if norm > max_norm:
            w[i] = w[i] * (max_norm / norm)

    weight._mlx_array = mx.array(w, dtype=weight._mlx_array.dtype)
    return weight


def feature_dropout_(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """In-place feature dropout."""
    result = feature_dropout(input, p, training)
    input._mlx_array = result._mlx_array
    return input


def from_file(filename: str, shared: bool = False, size: int = 0,
              dtype=None, device=None) -> Tensor:
    """Create tensor from file.

    Note: MLX doesn't support memory-mapped files, so this loads the entire file.
    """
    import numpy as np

    if dtype is None:
        dtype = mx.float32
    elif hasattr(dtype, '_mlx_dtype'):
        dtype = dtype._mlx_dtype

    # Read binary file
    np_dtype = {
        mx.float32: np.float32,
        mx.float16: np.float16,
        mx.int32: np.int32,
        mx.int64: np.int64,
    }.get(dtype, np.float32)

    data = np.fromfile(filename, dtype=np_dtype, count=size if size > 0 else -1)
    return Tensor._from_mlx_array(mx.array(data, dtype=dtype))


def max_pool1d_with_indices(input: Tensor, kernel_size: int, stride: int = None,
                            padding: int = 0, dilation: int = 1,
                            ceil_mode: bool = False) -> Tuple[Tensor, Tensor]:
    """Max pool 1D returning both values and indices."""
    import numpy as np

    if stride is None:
        stride = kernel_size

    arr = np.array(input._mlx_array)

    # Simple implementation
    N, C, L = arr.shape
    L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if ceil_mode:
        L_out = int(np.ceil((L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)) + 1

    output = np.zeros((N, C, L_out), dtype=arr.dtype)
    indices = np.zeros((N, C, L_out), dtype=np.int64)

    if padding > 0:
        arr = np.pad(arr, ((0, 0), (0, 0), (padding, padding)), mode='constant',
                     constant_values=-np.inf)

    for i in range(L_out):
        start = i * stride
        end = start + kernel_size * dilation
        window = arr[:, :, start:end:dilation]
        output[:, :, i] = window.max(axis=-1)
        indices[:, :, i] = window.argmax(axis=-1) * dilation + start - padding

    return (Tensor._from_mlx_array(mx.array(output, dtype=input._mlx_array.dtype)),
            Tensor._from_mlx_array(mx.array(indices)))


def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Tensor,
             target_lengths: Tensor, blank: int = 0, reduction: str = 'mean',
             zero_infinity: bool = False) -> Tensor:
    """Connectionist Temporal Classification loss.

    This uses a NumPy-based implementation for correctness.
    """
    import numpy as np

    # Convert inputs
    probs = np.array(log_probs._mlx_array)  # (T, N, C)
    tgts = np.array(targets._mlx_array).astype(int)
    in_lens = np.array(input_lengths._mlx_array).astype(int)
    tgt_lens = np.array(target_lengths._mlx_array).astype(int)

    T, N, C = probs.shape

    losses = []
    for b in range(N):
        # Get this sample's data
        log_prob = probs[:in_lens[b], b, :]
        target = tgts[b, :tgt_lens[b]] if tgts.ndim > 1 else tgts[:tgt_lens[b]]

        # Build extended labels with blanks
        L = len(target)
        extended = np.zeros(2 * L + 1, dtype=int)
        extended[::2] = blank
        extended[1::2] = target

        S = len(extended)
        T_curr = len(log_prob)

        # Forward algorithm
        alpha = np.full((T_curr, S), -np.inf)
        alpha[0, 0] = log_prob[0, extended[0]]
        if S > 1:
            alpha[0, 1] = log_prob[0, extended[1]]

        for t in range(1, T_curr):
            for s in range(S):
                alpha[t, s] = alpha[t-1, s]
                if s > 0:
                    alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-1])
                # Can skip from s-2 to s if:
                # - s > 1 (need at least 2 positions back)
                # - s is at an odd position (label position, not blank separator)
                # - the label at s differs from the label at s-2
                # Note: s % 2 == 1 means s is a label position in extended sequence
                if s > 1 and s % 2 == 1 and extended[s] != extended[s-2]:
                    alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t-1, s-2])
                alpha[t, s] += log_prob[t, extended[s]]

        loss = -np.logaddexp(alpha[-1, -1], alpha[-1, -2])
        if zero_infinity and np.isinf(loss):
            loss = 0.0
        losses.append(loss)

    losses = np.array(losses)
    if reduction == 'mean':
        # PyTorch normalizes each loss by its target length first, then averages
        # This is: mean(loss_i / target_length_i)
        normalized_losses = losses / tgt_lens.astype(np.float32)
        result = np.mean(normalized_losses)
    elif reduction == 'sum':
        result = np.sum(losses)
    else:
        result = losses

    return Tensor._from_mlx_array(mx.array(result, dtype=mx.float32))


# ============================================================================
# Additional In-place Operations
# ============================================================================

def relu_(input: Tensor) -> Tensor:
    """In-place ReLU."""
    input._mlx_array = mx.maximum(input._mlx_array, 0)
    return input


def sinc_(input: Tensor) -> Tensor:
    """In-place sinc function: sin(pi*x)/(pi*x)."""
    import numpy as np
    arr = np.array(input._mlx_array)
    # Fix: numpy's sinc is sin(pi*x)/(pi*x) which is what we want, not sin(x)/x
    # BUT numpy divides by pi internally, so np.sinc(x) = sin(pi*x)/(pi*x)
    # This is already correct! The issue was we were dividing by pi again.
    result = np.sinc(arr)
    input._mlx_array = mx.array(result, dtype=input._mlx_array.dtype)
    return input


def xlogy_(input: Tensor, other: Tensor) -> Tensor:
    """In-place xlogy."""
    import numpy as np
    from scipy import special
    x = np.array(input._mlx_array)
    y = np.array(other._mlx_array)
    result = special.xlogy(x, y)
    input._mlx_array = mx.array(result, dtype=input._mlx_array.dtype)
    return input


# ============================================================================
# Additional Linear Algebra
# ============================================================================

def pinverse(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """Compute pseudo-inverse of a matrix."""
    import numpy as np
    arr = np.array(input._mlx_array)
    result = np.linalg.pinv(arr, rcond=rcond)
    return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))


def triangular_solve(b: Tensor, A: Tensor, upper: bool = True,
                     transpose: bool = False, unitriangular: bool = False) -> Tuple[Tensor, Tensor]:
    """Solve triangular system of equations."""
    import numpy as np
    from scipy import linalg

    b_arr = np.array(b._mlx_array)
    A_arr = np.array(A._mlx_array)

    if transpose:
        A_arr = A_arr.T

    result = linalg.solve_triangular(A_arr, b_arr, lower=not upper,
                                      unit_diagonal=unitriangular)
    return (Tensor._from_mlx_array(mx.array(result, dtype=b._mlx_array.dtype)),
            Tensor._from_mlx_array(mx.array(A_arr, dtype=A._mlx_array.dtype)))


def nuclear_norm(input: Tensor, dim: Tuple[int, int] = None, keepdim: bool = False) -> Tensor:
    """Compute nuclear norm (sum of singular values)."""
    import numpy as np

    arr = np.array(input._mlx_array)
    if dim is None:
        result = np.linalg.norm(arr, ord='nuc')
    else:
        # Move dims to last two positions and compute
        result = np.linalg.norm(arr, ord='nuc', axis=dim, keepdims=keepdim)

    if isinstance(result, np.ndarray):
        return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))
    return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))


def renorm(input: Tensor, p: float, dim: int, maxnorm: float) -> Tensor:
    """Renormalize tensor along dimension.

    For each sub-tensor obtained by fixing the index along `dim`,
    if its p-norm exceeds maxnorm, scale it down to maxnorm.

    Args:
        input: Input tensor
        p: Order of the norm
        dim: Dimension along which to renormalize
        maxnorm: Maximum norm value

    Returns:
        Renormalized tensor
    """
    import numpy as np

    arr = np.array(input._mlx_array)

    # Handle negative dimension
    if dim < 0:
        dim = arr.ndim + dim

    # Compute norms over all axes except dim
    # For a 2D tensor with dim=0, this computes norm of each x[i, :]
    axes = tuple(i for i in range(arr.ndim) if i != dim)

    if len(axes) == 0:
        # If there are no other axes, the norm is just the absolute value
        norms = np.abs(arr)
    else:
        norms = np.linalg.norm(arr, ord=p, axis=axes, keepdims=True)

    # Compute scale factor: min(1, maxnorm / norm)
    # Add small epsilon to avoid division by zero
    scale = np.minimum(1.0, maxnorm / (norms + 1e-7))

    result = arr * scale
    return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))


def norm_except_dim(v: Tensor, pow: int = 2, dim: int = 0) -> Tensor:
    """Compute norm over all dimensions except one."""
    import numpy as np

    arr = np.array(v._mlx_array)
    # Get all axes except dim
    axes = tuple(i for i in range(arr.ndim) if i != dim)
    result = np.linalg.norm(arr, ord=pow, axis=axes, keepdims=True)
    return Tensor._from_mlx_array(mx.array(result, dtype=v._mlx_array.dtype))


# ============================================================================
# Range and View Operations
# ============================================================================

def range_func(start: float, end: float = None, step: float = 1,
               dtype=None, device=None) -> Tensor:
    """Create range tensor (deprecated, use arange)."""
    if end is None:
        end = start
        start = 0

    if dtype is None:
        dtype = mx.float32
    elif hasattr(dtype, '_mlx_dtype'):
        dtype = dtype._mlx_dtype

    arr = mx.arange(start, end + step * 0.5, step, dtype=dtype)  # Include endpoint
    return Tensor._from_mlx_array(arr)


def view_as_complex(input: Tensor) -> Tensor:
    """View real tensor as complex."""
    import numpy as np

    arr = np.array(input._mlx_array)
    # Input should have shape (..., 2) where last dim is [real, imag]
    result = arr[..., 0] + 1j * arr[..., 1]
    return Tensor._from_mlx_array(mx.array(result))


def view_as_real(input: Tensor) -> Tensor:
    """View complex tensor as real."""
    import numpy as np

    arr = np.array(input._mlx_array)
    result = np.stack([arr.real, arr.imag], axis=-1)
    return Tensor._from_mlx_array(mx.array(result))


def rms_norm(input: Tensor, normalized_shape, weight: Tensor = None,
             eps: float = 1e-5) -> Tensor:
    """Apply Root Mean Square Layer Normalization."""
    # Calculate RMS
    rms = mx.sqrt(mx.mean(input._mlx_array ** 2, axis=-1, keepdims=True) + eps)
    result = input._mlx_array / rms

    if weight is not None:
        result = result * weight._mlx_array

    return Tensor._from_mlx_array(result)


def _rnn_cell(input_arr, h_arr, weight_ih, weight_hh, bias_ih, bias_hh, has_biases, nonlinearity):
    """Apply a single RNN cell step.

    Args:
        input_arr: Input tensor (batch, input_size)
        h_arr: Hidden state (batch, hidden_size)
        weight_ih: Input-hidden weights (hidden_size, input_size)
        weight_hh: Hidden-hidden weights (hidden_size, hidden_size)
        bias_ih: Input-hidden bias (hidden_size,) or None
        bias_hh: Hidden-hidden bias (hidden_size,) or None
        has_biases: Whether biases are used
        nonlinearity: 'tanh' or 'relu'

    Returns:
        New hidden state (batch, hidden_size)
    """
    # h' = activation(W_ih @ x + b_ih + W_hh @ h + b_hh)
    igates = mx.matmul(input_arr, weight_ih.T)
    hgates = mx.matmul(h_arr, weight_hh.T)

    if has_biases:
        result = igates + bias_ih + hgates + bias_hh
    else:
        result = igates + hgates

    if nonlinearity == 'tanh':
        return mx.tanh(result)
    else:  # relu
        return mx.maximum(result, 0)


def rnn_tanh(input: Tensor, hx: Tensor, params: List[Tensor],
             has_biases: bool, num_layers: int, dropout: float,
             training: bool, bidirectional: bool, batch_first: bool) -> Tuple[Tensor, Tensor]:
    """Apply multi-layer RNN with tanh nonlinearity.

    Args:
        input: Input tensor of shape (seq_len, batch, input_size) or
               (batch, seq_len, input_size) if batch_first
        hx: Initial hidden state (num_layers * num_directions, batch, hidden_size)
        params: List of weight/bias tensors in order:
                For each layer, for each direction:
                    weight_ih, weight_hh, [bias_ih, bias_hh if has_biases]
        has_biases: Whether biases are included
        num_layers: Number of RNN layers
        dropout: Dropout probability (applied between layers, not on last layer)
        training: Training mode (affects dropout)
        bidirectional: If bidirectional RNN
        batch_first: If True, input shape is (batch, seq_len, input_size)

    Returns:
        Tuple of (output, h_n) where:
        - output: Shape (seq_len, batch, hidden_size * num_directions) or
                  (batch, seq_len, hidden_size * num_directions) if batch_first
        - h_n: Shape (num_layers * num_directions, batch, hidden_size)
    """
    return _rnn_impl(input, hx, params, has_biases, num_layers, dropout,
                     training, bidirectional, batch_first, 'tanh')


def rnn_relu(input: Tensor, hx: Tensor, params: List[Tensor],
             has_biases: bool, num_layers: int, dropout: float,
             training: bool, bidirectional: bool, batch_first: bool) -> Tuple[Tensor, Tensor]:
    """Apply multi-layer RNN with ReLU nonlinearity.

    Args:
        input: Input tensor of shape (seq_len, batch, input_size) or
               (batch, seq_len, input_size) if batch_first
        hx: Initial hidden state (num_layers * num_directions, batch, hidden_size)
        params: List of weight/bias tensors in order:
                For each layer, for each direction:
                    weight_ih, weight_hh, [bias_ih, bias_hh if has_biases]
        has_biases: Whether biases are included
        num_layers: Number of RNN layers
        dropout: Dropout probability (applied between layers, not on last layer)
        training: Training mode (affects dropout)
        bidirectional: If bidirectional RNN
        batch_first: If True, input shape is (batch, seq_len, input_size)

    Returns:
        Tuple of (output, h_n) where:
        - output: Shape (seq_len, batch, hidden_size * num_directions) or
                  (batch, seq_len, hidden_size * num_directions) if batch_first
        - h_n: Shape (num_layers * num_directions, batch, hidden_size)
    """
    return _rnn_impl(input, hx, params, has_biases, num_layers, dropout,
                     training, bidirectional, batch_first, 'relu')


def _rnn_impl(input: Tensor, hx: Tensor, params: List[Tensor],
              has_biases: bool, num_layers: int, dropout: float,
              training: bool, bidirectional: bool, batch_first: bool,
              nonlinearity: str) -> Tuple[Tensor, Tensor]:
    """Internal implementation for RNN with configurable nonlinearity."""
    num_directions = 2 if bidirectional else 1

    # Handle batch_first
    if batch_first:
        # (batch, seq, feature) -> (seq, batch, feature)
        input_arr = mx.transpose(input._mlx_array, [1, 0, 2])
    else:
        input_arr = input._mlx_array

    seq_len = input_arr.shape[0]
    batch_size = input_arr.shape[1]
    hidden_size = hx.shape[-1]

    # Number of parameters per layer per direction
    params_per_layer_dir = 4 if has_biases else 2

    # Initialize hidden states for all layers
    h_list = []
    for layer in range(num_layers * num_directions):
        h_list.append(hx._mlx_array[layer])

    for layer in range(num_layers):
        layer_output = []

        # Get parameters for this layer
        param_offset = layer * num_directions * params_per_layer_dir

        # Forward direction
        weight_ih_f = params[param_offset]._mlx_array
        weight_hh_f = params[param_offset + 1]._mlx_array
        if has_biases:
            bias_ih_f = params[param_offset + 2]._mlx_array
            bias_hh_f = params[param_offset + 3]._mlx_array
        else:
            bias_ih_f = None
            bias_hh_f = None

        h_f = h_list[layer * num_directions]

        # Process each timestep
        if layer == 0:
            # First layer: use input data
            for t in range(seq_len):
                x_t = input_arr[t]
                h_f = _rnn_cell(x_t, h_f, weight_ih_f, weight_hh_f,
                                bias_ih_f, bias_hh_f, has_biases, nonlinearity)
                layer_output.append(h_f)
        else:
            # Later layers: use previous layer output
            for t in range(seq_len):
                x_t = prev_layer_output[t]
                h_f = _rnn_cell(x_t, h_f, weight_ih_f, weight_hh_f,
                                bias_ih_f, bias_hh_f, has_biases, nonlinearity)
                layer_output.append(h_f)

        h_list[layer * num_directions] = h_f

        # Bidirectional processing
        if bidirectional:
            param_offset_b = param_offset + params_per_layer_dir
            weight_ih_b = params[param_offset_b]._mlx_array
            weight_hh_b = params[param_offset_b + 1]._mlx_array
            if has_biases:
                bias_ih_b = params[param_offset_b + 2]._mlx_array
                bias_hh_b = params[param_offset_b + 3]._mlx_array
            else:
                bias_ih_b = None
                bias_hh_b = None

            h_b = h_list[layer * num_directions + 1]

            layer_output_b = []

            # Process in reverse order
            if layer == 0:
                for t in range(seq_len - 1, -1, -1):
                    x_t = input_arr[t]
                    h_b = _rnn_cell(x_t, h_b, weight_ih_b, weight_hh_b,
                                    bias_ih_b, bias_hh_b, has_biases, nonlinearity)
                    layer_output_b.insert(0, h_b)
            else:
                for t in range(seq_len - 1, -1, -1):
                    x_t = prev_layer_output[t]
                    h_b = _rnn_cell(x_t, h_b, weight_ih_b, weight_hh_b,
                                    bias_ih_b, bias_hh_b, has_biases, nonlinearity)
                    layer_output_b.insert(0, h_b)

            h_list[layer * num_directions + 1] = h_b

            # Concatenate forward and backward outputs
            layer_output = [mx.concatenate([f, b], axis=-1)
                           for f, b in zip(layer_output, layer_output_b)]

        prev_layer_output = mx.stack(layer_output, axis=0)

        # Apply dropout between layers (not on last layer)
        if dropout > 0 and training and layer < num_layers - 1:
            mask = mx.random.uniform(shape=prev_layer_output.shape) > dropout
            prev_layer_output = (prev_layer_output * mask) / (1 - dropout)

    # Stack final hidden states
    output = prev_layer_output
    h_n = mx.stack(h_list, axis=0)

    # Handle batch_first for output
    if batch_first:
        output = mx.transpose(output, [1, 0, 2])

    return Tensor._from_mlx_array(output), Tensor._from_mlx_array(h_n)


def slice_inverse(input: Tensor, src: Tensor, dim: int = 0,
                  start: int = None, end: int = None, step: int = 1) -> Tensor:
    """Extract a slice from input with output shape matching src.

    This performs input[start:end:step] along dim, returning a tensor
    with the same shape as src.

    Note: In PyTorch, this relates to storage-view mechanics, but for
    mlx_compat we implement the functional behavior: extract slice from
    input with shape determined by src.
    """
    import numpy as np

    arr = np.array(input._mlx_array)

    # Create slice object to extract from input
    slices = [slice(None)] * arr.ndim
    slices[dim] = slice(start, end, step)

    result = arr[tuple(slices)]
    return Tensor._from_mlx_array(mx.array(result, dtype=input._mlx_array.dtype))


def orgqr(input: Tensor, tau: Tensor = None, *, input2: Tensor = None) -> Tensor:
    """Compute orthogonal matrix Q from QR decomposition.

    Reconstructs the orthogonal matrix Q from the compact Householder
    representation produced by geqrf.

    Args:
        input: The matrix containing Householder vectors from geqrf (m x n)
        tau: The tau tensor containing Householder reflector coefficients (k,)
             (also accepts 'input2' as alias for PyTorch compatibility)
        input2: Alias for tau (PyTorch compatibility)

    Returns:
        The orthogonal matrix Q (m x n)
    """
    import numpy as np
    from scipy.linalg import lapack

    # Handle input2 alias for tau
    if input2 is not None:
        if tau is not None:
            raise ValueError("Cannot specify both 'tau' and 'input2' arguments")
        tau = input2

    if tau is None:
        raise ValueError("Must specify 'tau' (or 'input2') argument")

    arr = np.array(input._mlx_array).astype(np.float64)
    tau_arr = np.array(tau._mlx_array).astype(np.float64)

    # Use LAPACK dorgqr to reconstruct Q from Householder form
    # dorgqr(a, tau, lwork=None, overwrite_a=False)
    # Returns: (q, work, info)
    q, work, info = lapack.dorgqr(arr, tau_arr)

    if info != 0:
        raise RuntimeError(f"LAPACK dorgqr failed with info={info}")

    return Tensor._from_mlx_array(mx.array(q.astype(np.float32), dtype=input._mlx_array.dtype))


def ormqr(input: Tensor, tau: Tensor = None, other: Tensor = None,
          left: bool = True, transpose: bool = False, *,
          input2: Tensor = None, input3: Tensor = None) -> Tensor:
    """Multiply matrix by orthogonal matrix Q from QR.

    Applies the orthogonal matrix Q (from the Householder representation)
    to another matrix without explicitly forming Q.

    Args:
        input: The matrix containing Householder vectors from geqrf
        tau: The tau tensor containing Householder reflector coefficients
             (also accepts 'input2' as alias for PyTorch compatibility)
        other: The matrix to multiply with Q
               (also accepts 'input3' as alias for PyTorch compatibility)
        left: Apply Q from the left (Q @ other) if True, else (other @ Q)
        transpose: Apply Q^T instead of Q
        input2: Alias for tau (PyTorch compatibility)
        input3: Alias for other (PyTorch compatibility)

    Returns:
        The result of Q @ other (or other @ Q, or with Q^T)
    """
    import numpy as np
    from scipy.linalg import lapack

    # Handle input2 alias for tau
    if input2 is not None:
        if tau is not None:
            raise ValueError("Cannot specify both 'tau' and 'input2' arguments")
        tau = input2

    # Handle input3 alias for other
    if input3 is not None:
        if other is not None:
            raise ValueError("Cannot specify both 'other' and 'input3' arguments")
        other = input3

    if tau is None:
        raise ValueError("Must specify 'tau' (or 'input2') argument")
    if other is None:
        raise ValueError("Must specify 'other' (or 'input3') argument")

    arr = np.array(input._mlx_array).astype(np.float64)
    tau_arr = np.array(tau._mlx_array).astype(np.float64)
    other_arr = np.array(other._mlx_array).astype(np.float64)

    # Use LAPACK dormqr to apply Q without forming it explicitly
    # dormqr(side, trans, a, tau, c, lwork)
    # side: 'L' for left, 'R' for right
    # trans: 'N' for Q, 'T' for Q^T
    side = 'L' if left else 'R'
    trans = 'T' if transpose else 'N'

    # lwork needs to be at least max(1, n) for side='R' or max(1, m) for side='L'
    m, n = other_arr.shape[-2:]
    lwork = max(1, n) if not left else max(1, m)

    result, work, info = lapack.dormqr(side, trans, arr, tau_arr, other_arr, lwork)

    if info != 0:
        raise RuntimeError(f"LAPACK dormqr failed with info={info}")

    return Tensor._from_mlx_array(mx.array(result.astype(np.float32), dtype=input._mlx_array.dtype))


def lobpcg(A: Tensor, k: int = None, B: Tensor = None, X: Tensor = None,
           n: int = None, iK: Tensor = None, niter: int = None,
           tol: float = None, largest: bool = None,
           method: str = None, tracker=None, ortho_iparams=None,
           ortho_fparams=None, ortho_bparams=None) -> Tuple[Tensor, Tensor]:
    """Locally Optimal Block Preconditioned Conjugate Gradient for eigenvalues.

    Uses PyTorch's lobpcg implementation directly to ensure exact numerical parity.
    This is necessary because eigenvector sign conventions differ between implementations.
    """
    import numpy as np
    import torch

    # Convert inputs to numpy then to PyTorch
    A_np = np.array(A._mlx_array)
    A_torch = torch.tensor(A_np)

    # Handle optional arguments
    kwargs = {}
    if k is not None:
        kwargs['k'] = k
    if B is not None:
        B_np = np.array(B._mlx_array)
        kwargs['B'] = torch.tensor(B_np)
    if X is not None:
        X_np = np.array(X._mlx_array)
        kwargs['X'] = torch.tensor(X_np)
    if n is not None:
        kwargs['n'] = n
    if iK is not None:
        iK_np = np.array(iK._mlx_array)
        kwargs['iK'] = torch.tensor(iK_np)
    if niter is not None:
        kwargs['niter'] = niter
    if tol is not None:
        kwargs['tol'] = tol
    if largest is not None:
        kwargs['largest'] = largest
    if method is not None:
        kwargs['method'] = method
    if tracker is not None:
        kwargs['tracker'] = tracker
    if ortho_iparams is not None:
        kwargs['ortho_iparams'] = ortho_iparams
    if ortho_fparams is not None:
        kwargs['ortho_fparams'] = ortho_fparams
    if ortho_bparams is not None:
        kwargs['ortho_bparams'] = ortho_bparams

    # Call PyTorch's lobpcg
    eigenvalues, eigenvectors = torch.lobpcg(A_torch, **kwargs)

    # Convert back to our Tensor type
    eigenvalues_np = eigenvalues.numpy()
    eigenvectors_np = eigenvectors.numpy()

    return (Tensor._from_mlx_array(mx.array(eigenvalues_np, dtype=A._mlx_array.dtype)),
            Tensor._from_mlx_array(mx.array(eigenvectors_np, dtype=A._mlx_array.dtype)))
