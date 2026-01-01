"""
MLX Compat: PyTorch-compatible API layer for Apple MLX
========================================================

A PyTorch-compatible interface for Apple's MLX machine learning framework,
enabling PyTorch code to run on Apple Silicon with minimal modifications.

Example usage:
    >>> import flashlight
    >>> x = flashlight.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = x * 2 + 1
    >>> y.sum().backward()
    >>> print(x.grad)

The library is organized into several submodules:
    - flashlight.nn: Neural network layers and modules
    - flashlight.optim: Optimizers (SGD, Adam, etc.)
    - flashlight.autograd: Automatic differentiation
    - flashlight.ops: Low-level operators

Version: 0.1.0 (Alpha)
Status: Phase 0 - Project scaffolding complete
"""

__version__ = "0.1.0"
__author__ = "Flashlight Contributors"

from .device import (
    Device,
    current_device,
    device_count,
    get_default_device,
    is_available,
    set_default_device,
    synchronize,
)
from .dtype import (  # Floating point types; Integer types; Boolean; Complex; Functions
    bfloat16,
    bool,
    byte,
    complex64,
    complex128,
    double,
    float,
    float16,
    float32,
    float64,
    get_default_dtype,
    get_dtype,
    half,
    int,
    int8,
    int16,
    int32,
    int64,
    long,
    set_default_dtype,
    short,
    uint8,
    uint16,
    uint32,
    uint64,
)

# Phase 1: Tensor creation functions
from .factories import (  # Constant fill; -like variants; Sequences; Identity; Random; From data; Grid operations
    arange,
    as_tensor,
    bernoulli,
    cartesian_prod,
    clone,
    empty,
    empty_like,
    eye,
    from_numpy,
    full,
    full_like,
    linspace,
    logspace,
    meshgrid,
    multinomial,
    normal,
    ones,
    ones_like,
    poisson,
    rand,
    rand_like,
    randint,
    randint_like,
    randn,
    randn_like,
    randperm,
    scalar_tensor,
    tensor,
    zeros,
    zeros_like,
)

# Phase 2: Operators
from .ops import (  # Arithmetic; Trigonometric; Extended math functions; Logarithms; Activations; Reductions; Extended reductions; Shape manipulation; Extended shape operations; Additional shape ops; Sorting and selection; Linear algebra (torch.* level); Indexing; Extended indexing; Scatter extensions; Comparison; Math utilities; In-place variants; New math functions; Quick ops; New quick ops; In-place operations; Grid sampler; Histogram; Other; RNN functions; Matrix functions; NaN-aware operations; Strided operations; Scatter/Index operations; Linear algebra extensions; Special math functions; Additional in-place ops; Additional linear algebra; Range and view ops; Pooling; Convolution
    abs,
    absolute,
    acos,
    acosh,
    adaptive_avg_pool1d,
    adaptive_max_pool1d,
    add,
    addbmm,
    addcdiv,
    addcmul,
    addmm,
    addmv,
    addr,
    adjoint,
    affine_grid_generator,
    all,
    allclose,
    alpha_dropout_,
    amax,
    amin,
    aminmax,
    angle,
    any,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    argmax,
    argmin,
    argsort,
    argwhere,
    as_strided,
    as_strided_,
    as_strided_scatter,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    avg_pool1d,
    avg_pool2d,
    baddbmm,
    bilinear,
    binomial,
    bitwise_and,
    bitwise_left_shift,
    bitwise_not,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    block_diag,
    bmm,
    broadcast_tensors,
    broadcast_to,
    cat,
    ceil,
    chain_matmul,
    cholesky_inverse,
    cholesky_solve,
    chunk,
    clamp,
    clamp_max,
    clamp_max_,
    clamp_min,
    clamp_min_,
    clip,
    clip_,
    column_stack,
    combinations,
    concat,
    concatenate,
    conj,
    constant_pad_nd,
    conv1d,
    conv2d,
    conv3d,
    convolution,
    copysign,
    corrcoef,
    cos,
    cosh,
    count_nonzero,
    cov,
    cummax,
    cummin,
    cumprod,
    cumsum,
    cumulative_trapezoid,
    deg2rad,
    deg2rad_,
    diag,
    diag_embed,
    diagflat,
    diagonal,
    diagonal_scatter,
    diff,
    digamma,
    dist,
    div,
    divide,
    dot,
    dropout_,
    dsplit,
    dstack,
    einsum,
    elu,
    embedding_renorm_,
    empty_permuted,
    empty_strided,
    eq,
    equal,
    erf,
    erf_,
    erfc,
    erfc_,
    erfinv,
    exp,
    exp2,
    exp2_,
    expand,
    expm1,
    feature_alpha_dropout_,
    feature_dropout,
    feature_dropout_,
    fill,
    fill_,
    fix,
    fix_,
    flip,
    fliplr,
    flipud,
    float_power,
    floor,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frac,
    frexp,
    frobenius_norm,
    from_file,
    frombuffer,
    gather,
    gcd,
    gcd_,
    ge,
    gelu,
    geqrf,
    ger,
    gradient,
    greater,
    greater_equal,
    grid_sampler,
    grid_sampler_2d,
    grid_sampler_3d,
    gru,
    gt,
    heaviside,
    histc,
    histogram,
    histogramdd,
    hsplit,
    hstack,
    hypot,
    i0,
    i0_,
    igamma,
    igammac,
    imag,
    index_add,
    index_copy,
    index_fill,
    index_put,
    index_put_,
    index_reduce,
    index_select,
    inner,
    is_same_size,
    is_signed,
    isclose,
    isfinite,
    isin,
    isinf,
    isnan,
    isneginf,
    isposinf,
    isreal,
    kron,
    kthvalue,
    lcm,
    lcm_,
    ldexp,
    ldexp_,
    le,
    leaky_relu,
    lerp,
    less,
    less_equal,
    lgamma,
    lobpcg,
    log,
    log1p,
    log2,
    log10,
    log_softmax,
    logaddexp,
    logaddexp2,
    logcumsumexp,
    logdet,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    logit,
    logit_,
    logsumexp,
    lstm,
    lt,
    lu_solve,
    lu_unpack,
    masked_fill,
    masked_scatter,
    masked_select,
    matmul,
    matrix_exp,
    matrix_power,
    max,
    max_pool1d,
    max_pool1d_with_indices,
    max_pool2d,
    maximum,
    mean,
    median,
    min,
    minimum,
    mm,
    mode,
    moveaxis,
    movedim,
    msort,
    mul,
    multiply,
    mv,
    mvlgamma,
    nan_to_num,
    nan_to_num_,
    nanmean,
    nanmedian,
    nanquantile,
    nansum,
    narrow,
    ne,
    neg,
    negative,
    negative_,
    nextafter,
    nonzero,
    nonzero_static,
    norm_except_dim,
    not_equal,
    nuclear_norm,
    numel,
    orgqr,
    ormqr,
    outer,
    pinverse,
    polygamma,
    positive,
    pow,
    prod,
    put,
    quantile,
    rad2deg,
    rad2deg_,
    range_,
    range_func,
    ravel,
    real,
    reciprocal,
    relu,
    relu_,
    remainder,
    renorm,
    repeat,
    repeat_interleave,
    rms_norm,
    rnn_relu,
    rnn_tanh,
    roll,
    rot90,
    round,
    row_stack,
    rsqrt,
    rsub,
    scatter,
    scatter_add,
    scatter_reduce,
    select,
    select_scatter,
    sgn,
    sigmoid,
    sign,
    signbit,
    silu,
    sin,
    sinc,
    sinc_,
    sinh,
    slice_inverse,
    slice_scatter,
    softmax,
    sort,
    split,
    split_with_sizes,
    sqrt,
    square,
    square_,
    stack,
    std,
    std_mean,
    sub,
    subtract,
    sum,
    swapaxes,
    swapdims,
    swish,
    t,
    take,
    take_along_dim,
    tan,
    tanh,
    tensor_split,
    tensordot,
    tile,
    topk,
    trace,
    trapezoid,
    trapz,
    triangular_solve,
    tril,
    tril_indices,
    triu,
    triu_indices,
    true_divide,
    trunc,
    unbind,
    unflatten,
    unique,
    unique_consecutive,
    unravel_index,
    unsafe_chunk,
    unsafe_split,
    unsafe_split_with_sizes,
    vander,
    var,
    var_mean,
    vdot,
    view_as_complex,
    view_as_real,
    vsplit,
    vstack,
    where,
    xlogy,
    xlogy_,
)

# Phase 1: Tensor core
from .tensor import Tensor

# Phase 1: View operations
from .view_ops import (
    contiguous,
    flatten,
    permute,
    reshape,
    squeeze,
    transpose,
    unsqueeze,
    view,
)

# Create alias for range (since range_func avoids builtin conflict)
range = range_func

# torch.random namespace
# torch.amp namespace (automatic mixed precision)
# torch.fft namespace
# torch.special namespace
# torch.linalg namespace
from . import amp, fft, linalg, random, special

# Expose linalg functions at torch.* level
from .linalg import (
    cholesky,
    cross,
    det,
)
from .linalg import inv as inverse
from .linalg import norm as linalg_norm
from .linalg import (
    pinv,
    qr,
    slogdet,
    svd,
)

# Transposed convolutions (from nn.functional)
from .nn.functional import (
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
)


def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    """
    Compute the matrix or vector norm.

    This is the torch.norm function (deprecated). For new code, use torch.linalg.norm.

    Args:
        input: Input tensor
        p: Norm order (default: 'fro' for Frobenius norm)
        dim: Dimensions to compute norm over
        keepdim: Keep reduced dimensions
        out: Output tensor (ignored)
        dtype: Output dtype

    Returns:
        Norm tensor
    """
    # torch.norm uses 'p' while torch.linalg.norm uses 'ord'
    return linalg_norm(input, ord=p, dim=dim, keepdim=keepdim, dtype=dtype)


# Expose embedding functions at torch.* level
# Note: torch.embedding(weight, indices) has different signature than
# torch.nn.functional.embedding(input, weight), so we need a wrapper
# Expose additional activation functions at torch.* level
# Expose dropout and normalization functions at torch.* level
from .nn.functional import (
    alpha_dropout,
    batch_norm,
    celu,
    celu_,
    dropout,
)
from .nn.functional import embedding as _embedding_functional
from .nn.functional import embedding_bag as _embedding_bag_functional
from .nn.functional import (
    feature_alpha_dropout,
    glu,
    group_norm,
    hardshrink,
    hardsigmoid,
    hardswish,
    hardtanh,
    hardtanh_,
    instance_norm,
    layer_norm,
    logsigmoid,
    mish,
    prelu,
    relu6,
    rrelu,
    rrelu_,
    selu,
    selu_,
    softmin,
    softplus,
    softshrink,
    softsign,
    tanhshrink,
    threshold,
    threshold_,
)


def embedding(
    input=None,
    weight=None,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    *,
    indices=None,  # Alias for input for compatibility
):
    """
    Embedding lookup (torch.embedding signature).

    Args:
        input: Tensor of indices (also accepts 'indices' as alias)
        weight: Embedding weight matrix of shape (num_embeddings, embedding_dim)
        padding_idx: Index to pad (ignored in forward)
        max_norm: Max norm for embeddings
        norm_type: Norm type for max_norm
        scale_grad_by_freq: Scale gradients by frequency
        sparse: Use sparse gradients (not supported)
        indices: Alias for input (for compatibility)

    Returns:
        Embedded tensor
    """
    # Handle indices alias
    if indices is not None:
        if input is not None:
            raise ValueError("Cannot specify both 'input' and 'indices' arguments")
        input = indices

    if input is None:
        raise ValueError("Must specify 'input' (or 'indices') argument")
    if weight is None:
        raise ValueError("Must specify 'weight' argument")

    return _embedding_functional(
        input=input,
        weight=weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )


def embedding_bag(
    weight,
    input,
    offsets=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    mode="sum",
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=None,
):
    """
    Embedding bag lookup (torch.embedding_bag signature).

    Note: torch.embedding_bag has signature (weight, input, offsets, ...)
    while torch.nn.functional.embedding_bag has (input, weight, offsets, ...).

    This wrapper matches the torch.embedding_bag signature.

    Args:
        weight: Embedding weight matrix of shape (num_embeddings, embedding_dim)
        input: Tensor of indices
        offsets: Starting positions for each bag (when input is 1D)
        max_norm: Max norm for embeddings
        norm_type: Norm type for max_norm
        scale_grad_by_freq: Scale gradients by frequency
        mode: Aggregation mode ('mean', 'sum', 'max') or int (0=sum, 1=mean, 2=max)
        sparse: Use sparse gradients (not supported)
        per_sample_weights: Weights for each sample
        include_last_offset: Whether offsets includes final offset
        padding_idx: Index to ignore

    Returns:
        Tuple of (output, offset2bag, bag_size, max_indices)

        PyTorch behavior for return values:
        - mode='sum' (0): offset2bag is empty (0,), max_indices shape is (num_bags,)
        - mode='mean' (1): offset2bag shape is (num_indices,), max_indices shape is (num_bags,)
        - mode='max' (2): offset2bag shape is (num_indices,), max_indices shape is (num_bags, embedding_dim)
    """
    import builtins

    import mlx.core as mx

    # Normalize mode to string for the functional call
    # PyTorch C++ binding uses int: 0=sum, 1=mean, 2=max
    mode_int_to_str = {0: "sum", 1: "mean", 2: "max"}
    if isinstance(mode, builtins.int):
        mode_str = mode_int_to_str.get(mode, "sum")
        mode_is_sum = mode == 0
        mode_is_max = mode == 2
    else:
        mode_str = mode
        mode_is_sum = mode == "sum"
        mode_is_max = mode == "max"

    output = _embedding_bag_functional(
        input=input,
        weight=weight,
        offsets=offsets,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode_str,
        sparse=sparse,
        per_sample_weights=per_sample_weights,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
    )

    # PyTorch's torch.embedding_bag returns a tuple of 4 elements:
    # (output, offset2bag, bag_size, max_indices)
    #
    # Return value shapes depend on mode:
    # - mode='sum': offset2bag is empty (0,), max_indices shape is (num_bags,)
    # - mode='mean': offset2bag shape is (num_indices,), max_indices shape is (num_bags,)
    # - mode='max': offset2bag shape is (num_indices,), max_indices shape is (num_bags, embedding_dim)

    num_bags = output.shape[0]
    embedding_dim = output.shape[1] if output.ndim > 1 else 1

    # Get input as MLX array
    indices = input._mlx_array if hasattr(input, "_mlx_array") else mx.array(input)
    num_indices = indices.size

    # Compute offset2bag and bag_size based on mode
    # For mode='sum', PyTorch returns empty offset2bag and zeros bag_size (optimization - not needed for backward)
    if mode_is_sum:
        # Return empty offset2bag and zeros bag_size for sum mode (matches PyTorch behavior)
        offset2bag_data = mx.zeros((0,), dtype=mx.int64)
        bag_size_data = mx.zeros((num_bags,), dtype=mx.int64)
    else:
        # For mean and max modes, compute full offset2bag
        if indices.ndim == 2:
            batch_size, bag_len = indices.shape
            # Each element in row i belongs to bag i
            offset2bag_data = mx.repeat(mx.arange(batch_size), bag_len)
            bag_size_data = mx.full((batch_size,), bag_len, dtype=mx.int64)
        elif offsets is not None:
            offsets_data = (
                offsets._mlx_array if hasattr(offsets, "_mlx_array") else mx.array(offsets)
            )
            offsets_data = offsets_data.astype(mx.int64)

            if include_last_offset:
                bag_boundaries = offsets_data
                num_bags_computed = len(offsets_data) - 1
            else:
                bag_boundaries = mx.concatenate(
                    [offsets_data, mx.array([num_indices], dtype=mx.int64)]
                )
                num_bags_computed = len(offsets_data)

            # Compute offset2bag by creating a mapping from each index to its bag
            import builtins

            import numpy as np

            offsets_np = np.array(bag_boundaries)
            offset2bag_np = np.zeros(num_indices, dtype=np.int64)
            bag_size_np = np.zeros(num_bags_computed, dtype=np.int64)

            for i in builtins.range(num_bags_computed):
                start = builtins.int(offsets_np[i])
                end = builtins.int(offsets_np[i + 1])
                offset2bag_np[start:end] = i
                bag_size_np[i] = end - start

            offset2bag_data = mx.array(offset2bag_np)
            bag_size_data = mx.array(bag_size_np)
        else:
            # No offsets - treat as single bag
            offset2bag_data = mx.zeros(num_indices, dtype=mx.int64)
            bag_size_data = mx.array([num_indices], dtype=mx.int64)

    # Create max_indices based on mode
    # - mode='max': shape is (num_bags, embedding_dim)
    # - mode='sum' or 'mean': shape is (num_bags,)
    if mode_is_max:
        max_indices_data = mx.zeros((num_bags, embedding_dim), dtype=mx.int64)
    else:
        max_indices_data = mx.zeros((num_bags,), dtype=mx.int64)

    offset2bag = Tensor._from_mlx_array(offset2bag_data)
    bag_size = Tensor._from_mlx_array(bag_size_data)
    max_indices = Tensor._from_mlx_array(max_indices_data)

    return (output, offset2bag, bag_size, max_indices)


# Expose loss functions at torch.* level
# Import the functional versions first
from .nn import functional as _F

# Simple re-exports (these don't have the reduction parameter issue)
# Expose pixel operations at torch.* level
# Expose distance functions at torch.* level
# Expose RNN cell functions at torch.* level
from .nn.functional import (
    binary_cross_entropy,
    cdist,
    channel_shuffle,
    cosine_similarity,
    cross_entropy,
    gaussian_nll_loss,
    gru_cell,
    huber_loss,
    l1_loss,
    lstm_cell,
    mse_loss,
    multi_margin_loss,
    multilabel_soft_margin_loss,
    nll_loss,
    pairwise_distance,
    pdist,
    pixel_shuffle,
    pixel_unshuffle,
    poisson_nll_loss,
    rnn_relu_cell,
    rnn_tanh_cell,
    smooth_l1_loss,
    soft_margin_loss,
)

# =============================================================================
# Reduction Parameter Wrappers
# =============================================================================
# PyTorch's torch.* level loss functions (C++ bindings) use int reduction:
#   0 = 'none', 1 = 'mean', 2 = 'sum'
# But torch.nn.functional versions use string reduction.
# We need wrappers to convert int -> string for parity.
# =============================================================================


def _reduction_int_to_str(reduction):
    """Convert PyTorch's C++ reduction enum (int) to string."""
    if isinstance(reduction, str):
        return reduction
    if reduction == 0:
        return "none"
    elif reduction == 1:
        return "mean"
    elif reduction == 2:
        return "sum"
    else:
        raise ValueError(f"Invalid reduction value: {reduction}")


def binary_cross_entropy_with_logits(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction=1,  # PyTorch default is 1 (mean)
    pos_weight=None,
):
    """Binary cross entropy with logits (torch.* level API).

    This wrapper converts the int reduction parameter to string for nn.functional.

    Args:
        input: Logits
        target: Binary targets
        weight: Manual rescaling weight
        size_average: Deprecated
        reduce: Deprecated
        reduction: int (0=none, 1=mean, 2=sum) or str
        pos_weight: Weight for positive class

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.binary_cross_entropy_with_logits(
        input,
        target,
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction_str,
        pos_weight=pos_weight,
    )


def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2,
    eps=1e-6,
    swap=False,
    size_average=None,
    reduce=None,
    reduction=1,  # PyTorch default is 1 (mean)
):
    """Triplet margin loss (torch.* level API).

    Args:
        anchor: Anchor samples
        positive: Positive samples
        negative: Negative samples
        margin: Margin value
        p: Norm degree
        eps: Small constant
        swap: Use distance swap
        size_average: Deprecated
        reduce: Deprecated
        reduction: int (0=none, 1=mean, 2=sum) or str

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.triplet_margin_loss(
        anchor,
        positive,
        negative,
        margin=margin,
        p=p,
        eps=eps,
        swap=swap,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction_str,
    )


def margin_ranking_loss(
    input1,
    input2,
    target,
    margin=0,
    size_average=None,
    reduce=None,
    reduction=1,  # PyTorch default is 1 (mean)
):
    """Margin ranking loss (torch.* level API).

    Args:
        input1: First input
        input2: Second input
        target: Target with values 1 or -1
        margin: Margin value
        size_average: Deprecated
        reduce: Deprecated
        reduction: int (0=none, 1=mean, 2=sum) or str

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.margin_ranking_loss(
        input1,
        input2,
        target,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction_str,
    )


def hinge_embedding_loss(
    input,
    target,
    margin=1.0,
    size_average=None,
    reduce=None,
    reduction=1,  # PyTorch default is 1 (mean)
):
    """Hinge embedding loss (torch.* level API).

    Args:
        input: Input tensor
        target: Target with values 1 or -1
        margin: Margin for negative samples
        size_average: Deprecated
        reduce: Deprecated
        reduction: int (0=none, 1=mean, 2=sum) or str

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.hinge_embedding_loss(
        input,
        target,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction_str,
    )


def kl_div(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction=1,  # PyTorch default is 1 (mean)
    log_target=False,
):
    """KL divergence loss (torch.* level API).

    Args:
        input: Log-probabilities
        target: Target probabilities
        size_average: Deprecated
        reduce: Deprecated
        reduction: int (0=none, 1=mean, 2=sum) or str
        log_target: If True, target is in log-space

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.kl_div(
        input,
        target,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction_str,
        log_target=log_target,
    )


def cosine_embedding_loss(
    input1,
    input2,
    target,
    margin=0,
    size_average=None,
    reduce=None,
    reduction=1,  # PyTorch default is 1 (mean)
):
    """Cosine embedding loss (torch.* level API).

    Args:
        input1: First input
        input2: Second input
        target: Target with values 1 or -1
        margin: Margin for negative pairs
        size_average: Deprecated
        reduce: Deprecated
        reduction: int (0=none, 1=mean, 2=sum) or str

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.cosine_embedding_loss(
        input1,
        input2,
        target,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction_str,
    )


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction=1,  # PyTorch default is 1 (mean)
    zero_infinity=False,
):
    """CTC loss (torch.* level API).

    Args:
        log_probs: Log probabilities (T, N, C)
        targets: Target sequences
        input_lengths: Length of each input sequence
        target_lengths: Length of each target sequence
        blank: Blank label index
        reduction: int (0=none, 1=mean, 2=sum) or str
        zero_infinity: Replace inf losses with 0

    Returns:
        Loss tensor
    """
    reduction_str = _reduction_int_to_str(reduction)
    return _F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction_str,
        zero_infinity=zero_infinity,
    )


# Phase 7: Data loading
# Phase 5: Optimizers
# Phase 4: Neural networks
from . import data, nn, optim

# Phase 3: Autograd
from .autograd import (
    enable_grad,
    is_grad_enabled,
    no_grad,
    set_grad_enabled,
)

# Layout optimization for NHWC-native mode
from .layout import (
    Layout,
    convert_layout,
    ensure_nchw,
    ensure_nhwc,
    is_nhwc_mode,
    nchw_mode,
    nhwc_mode,
)

# Expose more pooling functions at torch.* level
# Expose other nn.functional at torch.* level
from .nn.functional import (
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool3d,
    avg_pool3d,
    dropout1d,
    dropout2d,
    dropout3d,
    interpolate,
    linear,
    max_pool3d,
    normalize,
    one_hot,
    pad,
)

# Serialization
from .serialization import load, save

# =============================================================================
# Utility Functions (torch-level)
# =============================================================================


def is_tensor(obj) -> bool:
    """Check if an object is a Tensor."""
    return isinstance(obj, Tensor)


def is_floating_point(input: Tensor) -> bool:
    """Check if tensor has floating point dtype."""
    import mlx.core as mx

    return input._mlx_array.dtype in (mx.float16, mx.float32, mx.bfloat16, mx.float64)


def is_complex(input: Tensor) -> bool:
    """Check if tensor has complex dtype."""
    import mlx.core as mx

    return input._mlx_array.dtype in (mx.complex64,)


def numel(input: Tensor) -> int:
    """Returns the total number of elements in the tensor."""
    return input._mlx_array.size


def manual_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    import mlx.core as mx

    mx.random.seed(seed)


def get_num_threads() -> int:
    """Get number of threads (always 1 for MLX)."""
    return 1


def set_num_threads(num: int) -> None:
    """Set number of threads (no-op for MLX)."""
    pass


# =============================================================================
# Extended Utility Functions (Sprint 6)
# =============================================================================


def result_type(*tensors_or_dtypes) -> "dtype":
    """
    Determine the result dtype from input tensors/dtypes.

    Args:
        *tensors_or_dtypes: Tensors or dtypes to determine result type

    Returns:
        Result dtype
    """
    import mlx.core as mx

    from .dtype import get_dtype

    dtypes = []
    for t in tensors_or_dtypes:
        if isinstance(t, Tensor):
            dtypes.append(t._mlx_array.dtype)
        elif hasattr(t, "_mlx_dtype"):
            dtypes.append(t._mlx_dtype)
        else:
            dtypes.append(get_dtype(t)._mlx_dtype)

    # MLX dtype promotion - simplified
    if any(d == mx.float64 for d in dtypes):
        return float64
    if any(d == mx.float32 for d in dtypes):
        return float32
    if any(d == mx.float16 for d in dtypes):
        return float16
    if any(d == mx.int64 for d in dtypes):
        return int64
    if any(d == mx.int32 for d in dtypes):
        return int32
    return float32


def promote_types(type1, type2) -> "dtype":
    """
    Promote two dtypes to a common type.

    Args:
        type1: First dtype
        type2: Second dtype

    Returns:
        Promoted dtype
    """
    return result_type(type1, type2)


def detach(input: Tensor) -> Tensor:
    """
    Return a new tensor detached from the computation graph.

    Args:
        input: Input tensor

    Returns:
        Detached tensor
    """
    return input.detach()


def detach_(input: Tensor) -> Tensor:
    """
    Detach tensor from computation graph in-place.

    Args:
        input: Input tensor

    Returns:
        Same tensor, detached
    """
    input.requires_grad = False
    input._grad_fn = None
    return input


def asarray(data, *, dtype=None, device=None, copy=None, requires_grad=False) -> Tensor:
    """
    Convert input data to a tensor.

    Args:
        data: Input data (array-like, tensor, or numpy array)
        dtype: Optional dtype
        device: Optional device (ignored, MLX uses unified memory)
        copy: Whether to copy data
        requires_grad: Whether to require gradients

    Returns:
        Tensor
    """
    import mlx.core as mx
    import numpy as np

    from .dtype import get_dtype

    if isinstance(data, Tensor):
        arr = data._mlx_array
        if copy:
            arr = mx.array(arr)
    elif isinstance(data, mx.array):
        arr = data if not copy else mx.array(data)
    elif isinstance(data, np.ndarray):
        arr = mx.array(data)
    else:
        arr = mx.array(data)

    if dtype is not None:
        dtype_obj = get_dtype(dtype)
        arr = arr.astype(dtype_obj._mlx_dtype)

    result = Tensor._from_mlx_array(arr)
    result.requires_grad = requires_grad
    return result


def complex(real: Tensor, imag: Tensor) -> Tensor:
    """
    Create a complex tensor from real and imaginary parts.

    Args:
        real: Real part tensor
        imag: Imaginary part tensor

    Returns:
        Complex tensor

    Note:
        MLX has limited complex support.
    """
    import mlx.core as mx

    # MLX complex number construction
    result_array = real._mlx_array.astype(mx.float32) + 1j * imag._mlx_array.astype(mx.float32)
    return Tensor._from_mlx_array(result_array)


def polar(abs_val: Tensor, angle: Tensor) -> Tensor:
    """
    Create a complex tensor from polar coordinates (magnitude and phase).

    Args:
        abs_val: Magnitude tensor
        angle: Phase tensor (in radians)

    Returns:
        Complex tensor
    """
    import mlx.core as mx

    real = abs_val._mlx_array * mx.cos(angle._mlx_array)
    imag = abs_val._mlx_array * mx.sin(angle._mlx_array)
    result_array = real.astype(mx.float32) + 1j * imag.astype(mx.float32)
    return Tensor._from_mlx_array(result_array)


def conj_physical(input: Tensor) -> Tensor:
    """
    Compute the element-wise conjugate of a tensor.

    Args:
        input: Input tensor

    Returns:
        Conjugated tensor
    """
    import mlx.core as mx

    result_array = mx.conj(input._mlx_array)
    return Tensor._from_mlx_array(result_array)


def conj_physical_(input: Tensor) -> Tensor:
    """
    In-place conjugate (returns new tensor in MLX).
    """
    return conj_physical(input)


def resolve_conj(input: Tensor) -> Tensor:
    """
    Return conjugated copy if tensor has conjugate bit set.

    For MLX, this is just a pass-through.
    """
    return input


def resolve_neg(input: Tensor) -> Tensor:
    """
    Return negated copy if tensor has negative bit set.

    For MLX, this is just a pass-through.
    """
    return input


def is_conj(input: Tensor) -> bool:
    """
    Check if tensor has conjugate bit set.

    Always returns False for MLX.
    """
    return False


def is_neg(input: Tensor) -> bool:
    """
    Check if tensor has negative bit set.

    Always returns False for MLX.
    """
    return False


# =============================================================================
# Window Functions
# =============================================================================


def bartlett_window(
    window_length, *, periodic=True, dtype=None, device=None, requires_grad=False
) -> Tensor:
    """
    Compute the Bartlett window.

    Args:
        window_length: Size of the window
        periodic: If True, return periodic window
        dtype: Output dtype
        device: Output device (ignored)
        requires_grad: Whether to require gradients

    Returns:
        Bartlett window tensor
    """
    import math

    import mlx.core as mx

    from .dtype import get_dtype

    if window_length <= 0:
        return Tensor._from_mlx_array(mx.array([]))

    if window_length == 1:
        result = Tensor._from_mlx_array(mx.array([1.0]))
        result.requires_grad = requires_grad
        return result

    # Fix: Match PyTorch's linspace-based implementation
    # periodic=True corresponds to sym=False in PyTorch signal.windows
    start = -1.0
    constant = 2.0 / (window_length if periodic else window_length - 1)
    k = mx.arange(window_length, dtype=mx.float32) * constant + start
    window = 1.0 - mx.abs(k)

    if dtype is not None:
        from .dtype import get_dtype

        dtype_obj = get_dtype(dtype)
        window = window.astype(dtype_obj._mlx_dtype)

    result = Tensor._from_mlx_array(window)
    result.requires_grad = requires_grad
    return result


def blackman_window(
    window_length, *, periodic=True, dtype=None, device=None, requires_grad=False
) -> Tensor:
    """
    Compute the Blackman window.

    Args:
        window_length: Size of the window
        periodic: If True, return periodic window
        dtype: Output dtype
        device: Output device (ignored)
        requires_grad: Whether to require gradients

    Returns:
        Blackman window tensor
    """
    import math

    import mlx.core as mx

    from .dtype import get_dtype

    if window_length <= 0:
        return Tensor._from_mlx_array(mx.array([]))

    if window_length == 1:
        result = Tensor._from_mlx_array(mx.array([1.0]))
        result.requires_grad = requires_grad
        return result

    N = window_length if periodic else window_length - 1
    n = mx.arange(window_length, dtype=mx.float32)

    # Blackman coefficients
    a0, a1, a2 = 0.42, 0.5, 0.08
    window = a0 - a1 * mx.cos(2.0 * math.pi * n / N) + a2 * mx.cos(4.0 * math.pi * n / N)

    if dtype is not None:
        from .dtype import get_dtype

        dtype_obj = get_dtype(dtype)
        window = window.astype(dtype_obj._mlx_dtype)

    result = Tensor._from_mlx_array(window)
    result.requires_grad = requires_grad
    return result


def hamming_window(
    window_length,
    *,
    periodic=True,
    alpha=0.54,
    beta=0.46,
    dtype=None,
    device=None,
    requires_grad=False,
) -> Tensor:
    """
    Compute the Hamming window.

    Args:
        window_length: Size of the window
        periodic: If True, return periodic window
        alpha: Window coefficient alpha (default: 0.54)
        beta: Window coefficient beta (default: 0.46)
        dtype: Output dtype
        device: Output device (ignored)
        requires_grad: Whether to require gradients

    Returns:
        Hamming window tensor
    """
    import math

    import mlx.core as mx

    from .dtype import get_dtype

    if window_length <= 0:
        return Tensor._from_mlx_array(mx.array([]))

    if window_length == 1:
        result = Tensor._from_mlx_array(mx.array([1.0]))
        result.requires_grad = requires_grad
        return result

    N = window_length if periodic else window_length - 1
    n = mx.arange(window_length, dtype=mx.float32)

    window = alpha - beta * mx.cos(2.0 * math.pi * n / N)

    if dtype is not None:
        from .dtype import get_dtype

        dtype_obj = get_dtype(dtype)
        window = window.astype(dtype_obj._mlx_dtype)

    result = Tensor._from_mlx_array(window)
    result.requires_grad = requires_grad
    return result


def hann_window(
    window_length, *, periodic=True, dtype=None, device=None, requires_grad=False
) -> Tensor:
    """
    Compute the Hann window.

    Args:
        window_length: Size of the window
        periodic: If True, return periodic window
        dtype: Output dtype
        device: Output device (ignored)
        requires_grad: Whether to require gradients

    Returns:
        Hann window tensor
    """
    import math

    import mlx.core as mx

    from .dtype import get_dtype

    if window_length <= 0:
        return Tensor._from_mlx_array(mx.array([]))

    if window_length == 1:
        result = Tensor._from_mlx_array(mx.array([1.0]))
        result.requires_grad = requires_grad
        return result

    N = window_length if periodic else window_length - 1
    n = mx.arange(window_length, dtype=mx.float32)

    window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / N))

    if dtype is not None:
        from .dtype import get_dtype

        dtype_obj = get_dtype(dtype)
        window = window.astype(dtype_obj._mlx_dtype)

    result = Tensor._from_mlx_array(window)
    result.requires_grad = requires_grad
    return result


def kaiser_window(
    window_length, *, periodic=True, beta=12.0, dtype=None, device=None, requires_grad=False
) -> Tensor:
    """
    Compute the Kaiser window.

    Args:
        window_length: Size of the window
        periodic: If True, return periodic window
        beta: Shape parameter (default: 12.0)
        dtype: Output dtype
        device: Output device (ignored)
        requires_grad: Whether to require gradients

    Returns:
        Kaiser window tensor
    """
    import mlx.core as mx
    import numpy as np

    from .dtype import get_dtype

    if window_length <= 0:
        return Tensor._from_mlx_array(mx.array([]))

    if window_length == 1:
        result = Tensor._from_mlx_array(mx.array([1.0]))
        result.requires_grad = requires_grad
        return result

    # Use numpy for Kaiser window calculation (involves Bessel function)
    window_np = np.kaiser(window_length, beta)

    if not periodic:
        # Numpy returns non-periodic by default
        pass
    else:
        # For periodic, extend and truncate
        window_np = np.kaiser(window_length + 1, beta)[:-1]

    window = mx.array(window_np, dtype=mx.float32)

    if dtype is not None:
        from .dtype import get_dtype

        dtype_obj = get_dtype(dtype)
        window = window.astype(dtype_obj._mlx_dtype)

    result = Tensor._from_mlx_array(window)
    result.requires_grad = requires_grad
    return result


def searchsorted(
    sorted_sequence: Tensor,
    values: Tensor,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str = None,
) -> Tensor:
    """
    Find indices where elements should be inserted to maintain order.

    Args:
        sorted_sequence: 1D sorted tensor
        values: Values to search for
        out_int32: If True, return int32 indices
        right: If True, return rightmost index
        side: 'left' or 'right' (overrides `right` parameter)

    Returns:
        Tensor of indices
    """
    import mlx.core as mx
    import numpy as np

    sorted_np = np.array(sorted_sequence._mlx_array)
    values_np = np.array(values._mlx_array)

    if side is not None:
        side_arg = side
    else:
        side_arg = "right" if right else "left"

    indices_np = np.searchsorted(sorted_np.flatten(), values_np, side=side_arg)

    dtype = mx.int32 if out_int32 else mx.int64
    result = Tensor._from_mlx_array(mx.array(indices_np, dtype=dtype))
    return result


def bucketize(
    input: Tensor, boundaries: Tensor, *, out_int32: bool = False, right: bool = False
) -> Tensor:
    """
    Return bucket indices for input values given boundaries.

    Args:
        input: Input tensor
        boundaries: Sorted 1D tensor of boundaries
        out_int32: If True, return int32 indices
        right: If True, use right-side buckets

    Returns:
        Tensor of bucket indices
    """
    return searchsorted(boundaries, input, out_int32=out_int32, right=right)


def bincount(input: Tensor, weights: Tensor = None, minlength: int = 0) -> Tensor:
    """
    Count occurrences of each value in a non-negative int tensor.

    Args:
        input: 1D tensor of non-negative integers
        weights: Optional weights for each value
        minlength: Minimum length of output

    Returns:
        1D tensor of counts
    """
    import builtins

    import mlx.core as mx
    import numpy as np

    input_np = np.array(input._mlx_array).astype(builtins.int)
    weights_np = np.array(weights._mlx_array) if weights is not None else None

    counts = np.bincount(input_np, weights=weights_np, minlength=minlength)

    dtype = mx.float32 if weights is not None else mx.int64
    result = Tensor._from_mlx_array(mx.array(counts, dtype=dtype))
    return result


def can_cast(from_dtype, to_dtype, casting: str = "safe") -> bool:
    """
    Check if a dtype can be cast to another.

    Args:
        from_dtype: Source dtype
        to_dtype: Target dtype
        casting: Casting rule ('no', 'equiv', 'safe', 'same_kind', 'unsafe')

    Returns:
        True if cast is allowed
    """
    import numpy as np

    from .dtype import get_dtype

    if isinstance(from_dtype, Tensor):
        from_np = np.dtype(str(from_dtype._mlx_array.dtype).replace("mlx.core.", ""))
    else:
        from_np = np.dtype(str(get_dtype(from_dtype)._mlx_dtype).replace("mlx.core.", ""))

    if isinstance(to_dtype, Tensor):
        to_np = np.dtype(str(to_dtype._mlx_array.dtype).replace("mlx.core.", ""))
    else:
        to_np = np.dtype(str(get_dtype(to_dtype)._mlx_dtype).replace("mlx.core.", ""))

    return np.can_cast(from_np, to_np, casting=casting)


def is_nonzero(input: Tensor) -> bool:
    """
    Check if a single-element tensor is non-zero.

    Args:
        input: Single-element tensor

    Returns:
        True if tensor is non-zero
    """
    if input.numel() != 1:
        raise RuntimeError(f"Boolean value of Tensor with more than one value is ambiguous")
    return bool(input.item() != 0)


def is_inference_mode_enabled() -> bool:
    """Check if inference mode is enabled (always False for MLX-compat)."""
    return False


def are_deterministic_algorithms_enabled() -> bool:
    """Check if deterministic algorithms are enabled."""
    return False


def set_deterministic_debug_mode(debug_mode: int) -> None:
    """Set deterministic debug mode (no-op for MLX)."""
    pass


def get_deterministic_debug_mode() -> int:
    """Get deterministic debug mode."""
    return 0


# =============================================================================
# Low-level normalization wrappers (torch.* level API)
# =============================================================================
# PyTorch has both high-level (nn.functional.*) and low-level (torch.*) APIs
# for normalization. The signatures differ:
# - nn.functional.batch_norm(input, running_mean, running_var, weight, bias, ...)
# - torch.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)

# Import the functional versions to avoid name collision
from .nn import functional as _F


def _batch_norm_impl(
    input: Tensor,
    weight: "Optional[Tensor]",
    bias: "Optional[Tensor]",
    running_mean: "Optional[Tensor]",
    running_var: "Optional[Tensor]",
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> Tensor:
    """
    Low-level batch normalization (torch.batch_norm signature).

    This wraps nn.functional.batch_norm with the low-level signature.

    Args:
        input: Input tensor (N, C, ...)
        weight: Scale parameter (gamma)
        bias: Shift parameter (beta)
        running_mean: Running mean
        running_var: Running variance
        training: Training mode flag
        momentum: Momentum for running stats update
        eps: Epsilon for numerical stability
        cudnn_enabled: Ignored (for PyTorch compatibility)

    Returns:
        Normalized tensor
    """
    return _F.batch_norm(
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )


def _instance_norm_impl(
    input: Tensor,
    weight: "Optional[Tensor]",
    bias: "Optional[Tensor]",
    running_mean: "Optional[Tensor]",
    running_var: "Optional[Tensor]",
    use_input_stats: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> Tensor:
    """
    Low-level instance normalization (torch.instance_norm signature).

    This wraps nn.functional.instance_norm with the low-level signature.

    Args:
        input: Input tensor (N, C, ...)
        weight: Scale parameter
        bias: Shift parameter
        running_mean: Running mean (usually None for instance norm)
        running_var: Running variance (usually None for instance norm)
        use_input_stats: Whether to use input statistics
        momentum: Momentum for running stats
        eps: Epsilon for numerical stability
        cudnn_enabled: Ignored (for PyTorch compatibility)

    Returns:
        Normalized tensor
    """
    return _F.instance_norm(
        input=input,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )


# Override the functional imports with low-level wrappers for torch.* level API
batch_norm = _batch_norm_impl
instance_norm = _instance_norm_impl


__all__ = [
    # Core
    "__version__",
    "Tensor",
    # Dtypes
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "float",
    "half",
    "double",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "short",
    "int",
    "long",
    "byte",
    "bool",
    "complex64",
    "complex128",
    "get_dtype",
    "set_default_dtype",
    "get_default_dtype",
    # Device
    "Device",
    "get_default_device",
    "set_default_device",
    "current_device",
    "device_count",
    "is_available",
    "synchronize",
    # Factory functions
    "zeros",
    "ones",
    "full",
    "empty",
    "zeros_like",
    "ones_like",
    "full_like",
    "empty_like",
    "rand_like",
    "randn_like",
    "randint_like",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "randn",
    "rand",
    "randint",
    "randperm",
    "normal",
    "bernoulli",
    "multinomial",
    "poisson",
    "tensor",
    "from_numpy",
    "clone",
    "as_tensor",
    "scalar_tensor",
    "meshgrid",
    "cartesian_prod",
    # View operations
    "reshape",
    "view",
    "transpose",
    "permute",
    "squeeze",
    "unsqueeze",
    "flatten",
    "contiguous",
    # Operators (Phase 2)
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "mm",
    "bmm",
    "pow",
    "sqrt",
    "exp",
    "log",
    "abs",
    "neg",
    "multiply",
    "divide",
    # Trigonometric functions
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "atan2",
    "asinh",
    "acosh",
    "atanh",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "arcsinh",
    "arccosh",
    "arctanh",
    "absolute",
    # Extended math functions
    "angle",
    "exp2",
    "sinc",
    "hypot",
    "copysign",
    "heaviside",
    "fmax",
    "fmin",
    "erfc",
    "lgamma",
    "digamma",
    "i0",
    "isreal",
    "isneginf",
    "isposinf",
    "float_power",
    "logaddexp2",
    "nextafter",
    "frexp",
    "ldexp",
    "xlogy",
    # Logarithms
    "log2",
    "log10",
    "log1p",
    "expm1",
    # Activations
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "silu",
    "leaky_relu",
    "elu",
    "swish",
    # Reductions
    "sum",
    "mean",
    "max",
    "min",
    "argmax",
    "argmin",
    "var",
    "std",
    "prod",
    "all",
    "any",
    "amax",
    "amin",
    "aminmax",
    # Extended reductions
    "median",
    "mode",
    "quantile",
    "nanmean",
    "nansum",
    "std_mean",
    "var_mean",
    "cummax",
    "cummin",
    "logcumsumexp",
    "histc",
    # Shape manipulation
    "cat",
    "stack",
    "split",
    "chunk",
    "expand",
    "repeat",
    "tile",
    "repeat_interleave",
    "gather",
    "narrow",
    "select",
    "unbind",
    "roll",
    "flip",
    "fliplr",
    "flipud",
    "rot90",
    # Extended shape operations
    "ravel",
    "t",
    "adjoint",
    "moveaxis",
    "swapaxes",
    "hstack",
    "vstack",
    "dstack",
    "column_stack",
    "row_stack",
    "hsplit",
    "vsplit",
    "dsplit",
    "tensor_split",
    "block_diag",
    "diag_embed",
    "diagflat",
    # Additional shape ops
    "movedim",
    "swapdims",
    "broadcast_tensors",
    "unflatten",
    "concat",
    "combinations",
    "split_with_sizes",
    "unsafe_chunk",
    "unsafe_split",
    "unsafe_split_with_sizes",
    # Sorting and selection
    "sort",
    "argsort",
    "topk",
    "kthvalue",
    "msort",
    "unique",
    "unique_consecutive",
    # Linear algebra (torch.* level)
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
    # Indexing
    "scatter",
    "scatter_add",
    "index_select",
    "where",
    "masked_fill",
    "masked_select",
    "index_add",
    "nonzero",
    "take",
    "put",
    # Extended indexing
    "index_fill",
    "index_copy",
    "index_put",
    "fill",
    "take_along_dim",
    "argwhere",
    "isin",
    # Scatter extensions
    "diagonal_scatter",
    "slice_scatter",
    "select_scatter",
    "scatter_reduce",
    # Comparison
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "equal",
    "allclose",
    "isclose",
    "maximum",
    "minimum",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    # Math utilities
    "clamp",
    "clip",
    "clamp_min",
    "clamp_max",
    "clamp_min_",
    "clamp_max_",
    "clip_",
    "floor",
    "ceil",
    "round",
    "trunc",
    "frac",
    "fix",
    "fix_",
    "sign",
    "signbit",
    "isnan",
    "isinf",
    "isfinite",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "reciprocal",
    "rsqrt",
    "square",
    "lerp",
    "addcmul",
    "addcdiv",
    "fmod",
    "remainder",
    "cumsum",
    "cumprod",
    "deg2rad",
    "rad2deg",
    "deg2rad_",
    "rad2deg_",
    "nan_to_num",
    "nan_to_num_",
    "count_nonzero",
    "diff",
    # In-place variants
    "negative_",
    "square_",
    # New math functions
    "logit",
    "logit_",
    "sgn",
    "rsub",
    "subtract",
    "floor_divide",
    "true_divide",
    "gcd",
    "gcd_",
    "lcm",
    "lcm_",
    "trapezoid",
    "trapz",
    "cumulative_trapezoid",
    "gradient",
    "is_same_size",
    "is_signed",
    "vander",
    "unravel_index",
    "tril_indices",
    "triu_indices",
    "range_",
    "bitwise_left_shift",
    "bitwise_right_shift",
    # Quick ops
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "broadcast_to",
    "concatenate",
    "conj",
    "erf",
    "erfinv",
    "negative",
    "positive",
    "real",
    "imag",
    "logaddexp",
    "logsumexp",
    "addmm",
    "baddbmm",
    "mv",
    "addr",
    "numel",
    # New quick ops
    "addbmm",
    "addmv",
    "chain_matmul",
    "dist",
    "corrcoef",
    "cov",
    "bilinear",
    "constant_pad_nd",
    "ger",
    "frobenius_norm",
    "frombuffer",
    "binomial",
    "convolution",
    "affine_grid_generator",
    # In-place operations
    "dropout_",
    "alpha_dropout_",
    "feature_alpha_dropout_",
    "erf_",
    "erfc_",
    "exp2_",
    "i0_",
    "fill_",
    # Grid sampler
    "grid_sampler",
    "grid_sampler_2d",
    "grid_sampler_3d",
    # Histogram
    "histogram",
    "histogramdd",
    # Other
    "feature_dropout",
    "igamma",
    "igammac",
    "polygamma",
    # RNN functions
    "lstm",
    "gru",
    # Matrix functions
    "logdet",
    "matrix_exp",
    "matrix_power",
    # NaN-aware operations
    "nanmedian",
    "nanquantile",
    # Strided operations
    "as_strided",
    "as_strided_",
    "as_strided_scatter",
    "empty_permuted",
    "empty_strided",
    "nonzero_static",
    # Scatter/Index operations
    "index_put_",
    "index_reduce",
    "masked_scatter",
    # Linear algebra extensions
    "cholesky_inverse",
    "cholesky_solve",
    "lu_solve",
    "lu_unpack",
    "geqrf",
    # Special math functions
    "mvlgamma",
    "ldexp_",
    "embedding_renorm_",
    "feature_dropout_",
    "from_file",
    "max_pool1d_with_indices",
    "ctc_loss",
    # Additional in-place ops
    "relu_",
    "sinc_",
    "xlogy_",
    # Additional linear algebra
    "pinverse",
    "triangular_solve",
    "nuclear_norm",
    "renorm",
    "norm_except_dim",
    # Range and view ops
    "range",
    "view_as_complex",
    "view_as_real",
    "rms_norm",
    "rnn_tanh",
    "rnn_relu",
    "slice_inverse",
    "orgqr",
    "ormqr",
    "lobpcg",
    # Pooling
    "max_pool2d",
    "avg_pool2d",
    "max_pool1d",
    "avg_pool1d",
    "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    # Convolution
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    # Linear algebra (from linalg)
    "norm",
    "det",
    "pinv",
    "cross",
    "svd",
    "qr",
    "cholesky",
    "slogdet",
    "inverse",
    # Dropout and normalization (from nn.functional)
    "dropout",
    "alpha_dropout",
    "feature_alpha_dropout",
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    # Additional activations (from nn.functional)
    "celu",
    "celu_",
    "selu",
    "selu_",
    "hardtanh",
    "hardtanh_",
    "hardshrink",
    "softshrink",
    "tanhshrink",
    "threshold",
    "threshold_",
    "glu",
    "logsigmoid",
    "prelu",
    "softmin",
    "rrelu",
    "rrelu_",
    "relu6",
    "hardswish",
    "hardsigmoid",
    "softplus",
    "softsign",
    "mish",
    # Embedding
    "embedding",
    "embedding_bag",
    # RNN cell functions
    "rnn_tanh_cell",
    "rnn_relu_cell",
    "lstm_cell",
    "gru_cell",
    # Distance functions
    "cosine_similarity",
    "pairwise_distance",
    "pdist",
    "cdist",
    # Pixel operations
    "pixel_shuffle",
    "pixel_unshuffle",
    "channel_shuffle",
    # Loss functions
    "mse_loss",
    "l1_loss",
    "cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "smooth_l1_loss",
    "triplet_margin_loss",
    "margin_ranking_loss",
    "hinge_embedding_loss",
    "huber_loss",
    "kl_div",
    "soft_margin_loss",
    "cosine_embedding_loss",
    "gaussian_nll_loss",
    "poisson_nll_loss",
    "multi_margin_loss",
    "multilabel_soft_margin_loss",
    # Other nn.functional
    "linear",
    "pad",
    "normalize",
    "one_hot",
    "interpolate",
    # Additional pooling
    "max_pool3d",
    "avg_pool3d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool3d",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    # Autograd (Phase 3)
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    # Utility functions
    "is_tensor",
    "is_floating_point",
    "is_complex",
    "manual_seed",
    "get_num_threads",
    "set_num_threads",
    # Extended utility functions
    "result_type",
    "promote_types",
    "searchsorted",
    "bucketize",
    "bincount",
    "can_cast",
    "is_nonzero",
    "is_inference_mode_enabled",
    "are_deterministic_algorithms_enabled",
    "set_deterministic_debug_mode",
    "get_deterministic_debug_mode",
    # Detach and other functions
    "detach",
    "detach_",
    "asarray",
    # Complex number operations
    "complex",
    "polar",
    "conj_physical",
    "conj_physical_",
    "resolve_conj",
    "resolve_neg",
    "is_conj",
    "is_neg",
    # Window functions
    "bartlett_window",
    "blackman_window",
    "hamming_window",
    "hann_window",
    "kaiser_window",
    # Namespaces
    "linalg",
    "special",
    "fft",
    "amp",
    "random",
    # Neural networks (Phase 4)
    "nn",
    # Optimizers (Phase 5)
    "optim",
    # Data loading (Phase 7)
    "data",
    # Serialization
    "save",
    "load",
]


# Development status indicator
def _get_implementation_status():
    """Return current implementation status."""
    return {
        "phase_0_scaffolding": "✓ Complete",
        "phase_1_tensor_core": "✓ Complete",
        "phase_2_operators": "○ Not started",
        "phase_3_autograd": "○ Not started",
        "phase_4_nn_modules": "○ Not started",
        "phase_5_training": "○ Not started",
        "phase_6_validation": "○ Not started",
    }


def show_status():
    """Display implementation status."""
    print(f"MLX Compat v{__version__}")
    print("=" * 50)
    status = _get_implementation_status()
    for phase, state in status.items():
        print(f"{phase}: {state}")
    print("=" * 50)


# Check MLX availability
try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False
    import warnings

    warnings.warn(
        "MLX not found. Please install MLX: pip install mlx",
        ImportWarning,
    )
