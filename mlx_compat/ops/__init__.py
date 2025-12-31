"""
Operators module - Phase 2

Contains low-level operators organized by category:
- arithmetic.py: add, sub, mul, div, matmul, etc.
- activations.py: relu, gelu, sigmoid, tanh, softmax, etc.
- reductions.py: sum, mean, max, min, argmax, etc.
- shape.py: cat, stack, split, reshape, flip, roll, rot90, etc.
- indexing.py: gather, scatter, where, masked_fill, etc.
- convolution.py: conv1d, conv2d, conv3d
- normalization.py: layer_norm, batch_norm
- pooling.py: max_pool2d, avg_pool2d
- sorting.py: sort, argsort, topk, etc.
- linalg_ops.py: einsum, tensordot, diag, triu, tril, etc.
"""

# Arithmetic operations
from .arithmetic import (
    add, sub, mul, div,
    matmul, mm, bmm,
    pow, sqrt, exp, log, abs, neg,
    sin, cos, tan, sinh, cosh,
    asin, acos, atan, atan2,
    asinh, acosh, atanh,
    log2, log10, log1p, expm1,
    multiply, divide, absolute,  # aliases
    arcsin, arccos, arctan, arctan2,  # numpy-style aliases
    arcsinh, arccosh, arctanh,
    # Extended math functions
    angle, exp2, sinc, hypot, copysign, heaviside,
    fmax, fmin, erfc, lgamma, digamma, i0,
    isreal, isneginf, isposinf, float_power,
    logaddexp2, nextafter, frexp, ldexp, xlogy,
)

# Activation functions
from .activations import (
    relu, gelu, sigmoid, tanh,
    softmax, log_softmax, silu,
    leaky_relu, elu,
    swish  # alias for silu
)

# Reduction operations
from .reductions import (
    sum, mean, max, min,
    argmax, argmin,
    var, std, prod,
    all, any,
    amax, amin, aminmax,
    # Extended reduction operations
    median, mode, quantile,
    nanmean, nansum,
    std_mean, var_mean,
    cummax, cummin, logcumsumexp, histc,
)

# Shape manipulation operations
from .shape import (
    cat, stack, split, chunk,
    expand, repeat, tile, repeat_interleave,
    gather, narrow, select, unbind,
    roll, flip, fliplr, flipud, rot90,
    # Extended shape operations
    ravel, t, adjoint, moveaxis, swapaxes,
    hstack, vstack, dstack, column_stack, row_stack,
    hsplit, vsplit, dsplit, tensor_split,
    block_diag, diag_embed, diagflat,
    # Additional shape ops
    movedim, swapdims,
    broadcast_tensors, unflatten, concat, combinations,
    split_with_sizes, unsafe_chunk, unsafe_split, unsafe_split_with_sizes,
)

# Sorting and selection operations
from .sorting import (
    sort, argsort, topk, kthvalue, msort,
    unique, unique_consecutive
)

# Linear algebra operations (torch.* level)
from .linalg_ops import (
    einsum, tensordot,
    diag, diagonal, triu, tril, trace,
    outer, inner, dot, vdot, kron
)

# Indexing operations
from .indexing import (
    scatter, scatter_add, index_select,
    where, masked_fill, masked_select,
    index_add, nonzero, take, put,
    # Extended indexing operations
    index_fill, index_copy, index_put,
    fill, take_along_dim, argwhere, isin,
    # Scatter extensions
    diagonal_scatter, slice_scatter, select_scatter, scatter_reduce,
)

# Comparison operations
from .comparison import (
    eq, ne, lt, le, gt, ge,
    equal, allclose, isclose,
    maximum, minimum,
    # Aliases
    greater, greater_equal, less, less_equal, not_equal,
)

# Convolution operations (torch.* level)
# Must import before quick_ops to avoid module shadowing the convolution function
from .convolution import conv2d as _conv2d_impl  # Import conv2d from convolution.py module
from .conv1d import conv1d
from .conv3d import conv3d, conv_transpose3d
conv2d = _conv2d_impl  # Assign to conv2d after potential shadowing

# Quick one-liner operations
from .quick_ops import (
    atleast_1d, atleast_2d, atleast_3d,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    broadcast_to, concatenate,
    conj, erf, erfinv,
    negative, positive,
    real, imag,
    logaddexp, logsumexp,
    addmm, baddbmm, mv, addr,
    numel,
    # New additions
    addbmm, addmv, chain_matmul, dist,
    corrcoef, cov, bilinear, constant_pad_nd,
    ger, frobenius_norm, frombuffer, binomial,
    convolution, affine_grid_generator,
    # In-place operations
    dropout_, alpha_dropout_, feature_alpha_dropout_,
    erf_, erfc_, exp2_, i0_, fill_,
    # Grid sampler
    grid_sampler, grid_sampler_2d, grid_sampler_3d,
    # Histogram
    histogram, histogramdd,
    # Other
    feature_dropout, igamma, igammac, polygamma,
    # RNN functions
    lstm, gru,
    # Matrix functions
    logdet, matrix_exp, matrix_power,
    # NaN-aware operations
    nanmedian, nanquantile,
    # Strided operations
    as_strided, as_strided_, as_strided_scatter,
    empty_permuted, empty_strided, nonzero_static,
    # Scatter/Index operations
    index_put_, index_reduce, masked_scatter,
    # Linear algebra extensions
    cholesky_inverse, cholesky_solve, lu_solve, lu_unpack, geqrf,
    # Special math functions
    mvlgamma, ldexp_, embedding_renorm_, feature_dropout_,
    from_file, max_pool1d_with_indices, ctc_loss,
    # Additional in-place ops
    relu_, sinc_, xlogy_,
    # Additional linear algebra
    pinverse, triangular_solve, nuclear_norm, renorm, norm_except_dim,
    # Range and view ops
    range_func, view_as_complex, view_as_real,
    rms_norm, rnn_tanh, rnn_relu, slice_inverse,
    orgqr, ormqr, lobpcg,
)

# Math utility functions
from .math_funcs import (
    clamp, clip,
    clamp_min, clamp_max, clamp_min_, clamp_max_, clip_,
    floor, ceil, round, trunc, frac,
    fix, fix_,
    sign, signbit,
    isnan, isinf, isfinite,
    logical_and, logical_or, logical_not, logical_xor,
    reciprocal, rsqrt, square,
    lerp, addcmul, addcdiv,
    fmod, remainder,
    cumsum, cumprod,
    deg2rad, rad2deg, deg2rad_, rad2deg_,
    nan_to_num, nan_to_num_,
    count_nonzero,
    diff,
    # In-place variants
    negative_, square_,
    # New math functions
    logit, logit_,
    sgn, rsub, subtract,
    floor_divide, true_divide,
    gcd, gcd_, lcm, lcm_,
    trapezoid, trapz, cumulative_trapezoid,
    gradient,
    is_same_size, is_signed,
    vander, unravel_index,
    tril_indices, triu_indices,
    range_,
    bitwise_left_shift, bitwise_right_shift,
)

# Pooling operations
from .pooling import (
    max_pool2d, avg_pool2d,
    max_pool1d, avg_pool1d,
    adaptive_avg_pool1d, adaptive_max_pool1d,
)

__all__ = [
    # Arithmetic
    'add', 'sub', 'mul', 'div',
    'matmul', 'mm', 'bmm',
    'pow', 'sqrt', 'exp', 'log', 'abs', 'neg',
    'sin', 'cos', 'tan', 'sinh', 'cosh',
    'asin', 'acos', 'atan', 'atan2',
    'asinh', 'acosh', 'atanh',
    'log2', 'log10', 'log1p', 'expm1',
    'multiply', 'divide', 'absolute',
    'arcsin', 'arccos', 'arctan', 'arctan2',
    'arcsinh', 'arccosh', 'arctanh',
    # Extended math functions
    'angle', 'exp2', 'sinc', 'hypot', 'copysign', 'heaviside',
    'fmax', 'fmin', 'erfc', 'lgamma', 'digamma', 'i0',
    'isreal', 'isneginf', 'isposinf', 'float_power',
    'logaddexp2', 'nextafter', 'frexp', 'ldexp', 'xlogy',
    # Activations
    'relu', 'gelu', 'sigmoid', 'tanh',
    'softmax', 'log_softmax', 'silu',
    'leaky_relu', 'elu', 'swish',
    # Reductions
    'sum', 'mean', 'max', 'min',
    'argmax', 'argmin',
    'var', 'std', 'prod',
    'all', 'any',
    'amax', 'amin', 'aminmax',
    # Extended reductions
    'median', 'mode', 'quantile',
    'nanmean', 'nansum',
    'std_mean', 'var_mean',
    'cummax', 'cummin', 'logcumsumexp', 'histc',
    # Shape manipulation
    'cat', 'stack', 'split', 'chunk',
    'expand', 'repeat', 'tile', 'repeat_interleave',
    'gather', 'narrow', 'select', 'unbind',
    'roll', 'flip', 'fliplr', 'flipud', 'rot90',
    # Extended shape operations
    'ravel', 't', 'adjoint', 'moveaxis', 'swapaxes',
    'hstack', 'vstack', 'dstack', 'column_stack', 'row_stack',
    'hsplit', 'vsplit', 'dsplit', 'tensor_split',
    'block_diag', 'diag_embed', 'diagflat',
    # Additional shape ops
    'movedim', 'swapdims',
    'broadcast_tensors', 'unflatten', 'concat', 'combinations',
    'split_with_sizes', 'unsafe_chunk', 'unsafe_split', 'unsafe_split_with_sizes',
    # Sorting and selection
    'sort', 'argsort', 'topk', 'kthvalue', 'msort',
    'unique', 'unique_consecutive',
    # Linear algebra
    'einsum', 'tensordot',
    'diag', 'diagonal', 'triu', 'tril', 'trace',
    'outer', 'inner', 'dot', 'vdot', 'kron',
    # Indexing
    'scatter', 'scatter_add', 'index_select',
    'where', 'masked_fill', 'masked_select',
    'index_add', 'nonzero', 'take', 'put',
    # Extended indexing
    'index_fill', 'index_copy', 'index_put',
    'fill', 'take_along_dim', 'argwhere', 'isin',
    # Comparison
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'equal', 'allclose', 'isclose',
    'maximum', 'minimum',
    'greater', 'greater_equal', 'less', 'less_equal', 'not_equal',
    # Math utilities
    'clamp', 'clip',
    'clamp_min', 'clamp_max', 'clamp_min_', 'clamp_max_', 'clip_',
    'floor', 'ceil', 'round', 'trunc', 'frac',
    'fix', 'fix_',
    'sign', 'signbit',
    'isnan', 'isinf', 'isfinite',
    'logical_and', 'logical_or', 'logical_not', 'logical_xor',
    'reciprocal', 'rsqrt', 'square',
    'lerp', 'addcmul', 'addcdiv',
    'fmod', 'remainder',
    'cumsum', 'cumprod',
    'deg2rad', 'rad2deg', 'deg2rad_', 'rad2deg_',
    'nan_to_num', 'nan_to_num_',
    'count_nonzero',
    'diff',
    # In-place variants
    'negative_', 'square_',
    # New math functions
    'logit', 'logit_',
    'sgn', 'rsub', 'subtract',
    'floor_divide', 'true_divide',
    'gcd', 'gcd_', 'lcm', 'lcm_',
    'trapezoid', 'trapz', 'cumulative_trapezoid',
    'gradient',
    'is_same_size', 'is_signed',
    'vander', 'unravel_index',
    'tril_indices', 'triu_indices',
    'range_',
    'bitwise_left_shift', 'bitwise_right_shift',
    # Quick ops
    'atleast_1d', 'atleast_2d', 'atleast_3d',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
    'broadcast_to', 'concatenate',
    'conj', 'erf', 'erfinv',
    'negative', 'positive',
    'real', 'imag',
    'logaddexp', 'logsumexp',
    'addmm', 'baddbmm', 'mv', 'addr',
    'numel',
    # New quick ops
    'addbmm', 'addmv', 'chain_matmul', 'dist',
    'corrcoef', 'cov', 'bilinear', 'constant_pad_nd',
    'ger', 'frobenius_norm', 'frombuffer', 'binomial',
    'convolution', 'affine_grid_generator',
    # In-place operations
    'dropout_', 'alpha_dropout_', 'feature_alpha_dropout_',
    'erf_', 'erfc_', 'exp2_', 'i0_', 'fill_',
    # Grid sampler
    'grid_sampler', 'grid_sampler_2d', 'grid_sampler_3d',
    # Histogram
    'histogram', 'histogramdd',
    # Other
    'feature_dropout', 'igamma', 'igammac', 'polygamma',
    # RNN functions
    'lstm', 'gru',
    # Matrix functions
    'logdet', 'matrix_exp', 'matrix_power',
    # NaN-aware operations
    'nanmedian', 'nanquantile',
    # Strided operations
    'as_strided', 'as_strided_', 'as_strided_scatter',
    'empty_permuted', 'empty_strided', 'nonzero_static',
    # Scatter/Index operations
    'index_put_', 'index_reduce', 'masked_scatter',
    # Linear algebra extensions
    'cholesky_inverse', 'cholesky_solve', 'lu_solve', 'lu_unpack', 'geqrf',
    # Special math functions
    'mvlgamma', 'ldexp_', 'embedding_renorm_', 'feature_dropout_',
    'from_file', 'max_pool1d_with_indices', 'ctc_loss',
    # Additional in-place ops
    'relu_', 'sinc_', 'xlogy_',
    # Additional linear algebra
    'pinverse', 'triangular_solve', 'nuclear_norm', 'renorm', 'norm_except_dim',
    # Range and view ops
    'range_func', 'view_as_complex', 'view_as_real',
    'rms_norm', 'rnn_tanh', 'rnn_relu', 'slice_inverse',
    'orgqr', 'ormqr', 'lobpcg',
    # Pooling
    'max_pool2d', 'avg_pool2d',
    'max_pool1d', 'avg_pool1d',
    'adaptive_avg_pool1d', 'adaptive_max_pool1d',
    # Convolution
    'conv1d', 'conv2d', 'conv3d', 'conv_transpose3d',
]
