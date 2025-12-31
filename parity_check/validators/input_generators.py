"""
Input generators for numerical parity testing.

Provides test input generation for different API types including tensor operations,
nn.Module classes, optimizers, and more.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all frameworks."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import mlx.core as mx
        mx.random.seed(seed)
    except ImportError:
        pass


# ============================================================================
# Basic Tensor Input Generators
# ============================================================================


def unary_tensor_input(shape: Tuple[int, ...] = (4, 8)) -> Dict[str, Any]:
    """Generate input for unary operations like relu, sigmoid, tanh."""
    return {"input": np.random.randn(*shape).astype(np.float32)}


def unary_positive_input(shape: Tuple[int, ...] = (4, 8)) -> Dict[str, Any]:
    """Generate positive input for ops like sqrt, log."""
    return {"input": np.abs(np.random.randn(*shape).astype(np.float32)) + 0.1}


def unary_bounded_input(
    shape: Tuple[int, ...] = (4, 8), low: float = -0.99, high: float = 0.99
) -> Dict[str, Any]:
    """Generate bounded input for ops like acos, asin, atanh."""
    return {"input": np.random.uniform(low, high, shape).astype(np.float32)}


def unary_ge_one_input(shape: Tuple[int, ...] = (4, 8)) -> Dict[str, Any]:
    """Generate input >= 1 for ops like acosh."""
    return {"input": np.abs(np.random.randn(*shape).astype(np.float32)) + 1.0}


def binary_tensor_input(shape: Tuple[int, ...] = (4, 8)) -> Dict[str, Any]:
    """Generate inputs for binary operations like add, mul, sub."""
    return {
        "input": np.random.randn(*shape).astype(np.float32),
        "other": np.random.randn(*shape).astype(np.float32),
    }


def binary_positive_input(shape: Tuple[int, ...] = (4, 8)) -> Dict[str, Any]:
    """Generate positive inputs for ops like div (avoid div by zero)."""
    return {
        "input": np.random.randn(*shape).astype(np.float32),
        "other": np.abs(np.random.randn(*shape).astype(np.float32)) + 0.1,
    }


def matmul_input(m: int = 4, k: int = 8, n: int = 6) -> Dict[str, Any]:
    """Generate inputs for matrix multiplication."""
    return {
        "input": np.random.randn(m, k).astype(np.float32),
        "other": np.random.randn(k, n).astype(np.float32),
    }


def batch_matmul_input(
    batch: int = 2, m: int = 4, k: int = 8, n: int = 6
) -> Dict[str, Any]:
    """Generate inputs for batched matrix multiplication."""
    return {
        "input": np.random.randn(batch, m, k).astype(np.float32),
        "other": np.random.randn(batch, k, n).astype(np.float32),
    }


def reduction_input(shape: Tuple[int, ...] = (4, 8, 6)) -> Dict[str, Any]:
    """Generate input for reduction operations like sum, mean, max."""
    return {"input": np.random.randn(*shape).astype(np.float32)}


def _make_positive_definite(n: int) -> np.ndarray:
    """Generate a positive definite matrix for Cholesky, etc."""
    A = np.random.randn(n, n).astype(np.float32)
    return A @ A.T + np.eye(n).astype(np.float32) * n


def _make_symmetric(n: int) -> np.ndarray:
    """Generate a symmetric matrix."""
    A = np.random.randn(n, n).astype(np.float32)
    return (A + A.T) / 2


def softmax_input(shape: Tuple[int, ...] = (4, 10)) -> Dict[str, Any]:
    """Generate input for softmax and related functions."""
    return {
        "input": np.random.randn(*shape).astype(np.float32),
        "dim": -1,
    }


def index_select_input(shape: Tuple[int, ...] = (8, 6)) -> Dict[str, Any]:
    """Generate input for index selection operations."""
    return {
        "input": np.random.randn(*shape).astype(np.float32),
        "dim": 0,
        "index": np.array([0, 2, 4, 6], dtype=np.int64),
    }


# ============================================================================
# nn.Module Specifications
# ============================================================================


@dataclass
class ModuleSpec:
    """Specification for testing an nn.Module class."""

    init_args: Dict[str, Any] = field(default_factory=dict)
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    input_shape: Tuple[int, ...] = (4, 8)
    # For layers that take multiple inputs (like loss functions)
    extra_inputs: Optional[Dict[str, Any]] = None
    # Weight name remapping if needed
    weight_map: Optional[Dict[str, str]] = None
    # Whether to run in eval mode
    eval_mode: bool = False


# Module specifications for common nn.Module classes
NN_MODULE_SPECS: Dict[str, ModuleSpec] = {
    # Linear layers
    "Linear": ModuleSpec(
        init_kwargs={"in_features": 10, "out_features": 5},
        input_shape=(3, 10),
    ),
    "Bilinear": ModuleSpec(
        init_kwargs={"in1_features": 8, "in2_features": 6, "out_features": 4},
        input_shape=(3, 8),
        extra_inputs={"input2_shape": (3, 6)},
    ),
    "LazyLinear": ModuleSpec(
        init_kwargs={"out_features": 5},
        input_shape=(3, 10),
    ),
    # Convolution layers
    "Conv1d": ModuleSpec(
        init_kwargs={"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1},
        input_shape=(2, 3, 16),
    ),
    "Conv2d": ModuleSpec(
        init_kwargs={"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1},
        input_shape=(2, 3, 16, 16),
    ),
    "Conv3d": ModuleSpec(
        init_kwargs={"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1},
        input_shape=(2, 3, 8, 8, 8),
    ),
    "ConvTranspose1d": ModuleSpec(
        init_kwargs={"in_channels": 8, "out_channels": 3, "kernel_size": 3, "padding": 1},
        input_shape=(2, 8, 16),
    ),
    "ConvTranspose2d": ModuleSpec(
        init_kwargs={"in_channels": 16, "out_channels": 3, "kernel_size": 3, "padding": 1},
        input_shape=(2, 16, 16, 16),
    ),
    "ConvTranspose3d": ModuleSpec(
        init_kwargs={"in_channels": 8, "out_channels": 3, "kernel_size": 3, "padding": 1},
        input_shape=(2, 8, 8, 8, 8),
    ),
    # Pooling layers
    "MaxPool1d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 16),
    ),
    "MaxPool2d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 16, 16),
    ),
    "MaxPool3d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 8, 8, 8),
    ),
    "AvgPool1d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 16),
    ),
    "AvgPool2d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 16, 16),
    ),
    "AvgPool3d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 8, 8, 8),
    ),
    "AdaptiveAvgPool1d": ModuleSpec(
        init_kwargs={"output_size": 4},
        input_shape=(2, 3, 16),
    ),
    "AdaptiveAvgPool2d": ModuleSpec(
        init_kwargs={"output_size": (4, 4)},
        input_shape=(2, 3, 16, 16),
    ),
    "AdaptiveAvgPool3d": ModuleSpec(
        init_kwargs={"output_size": (4, 4, 4)},
        input_shape=(2, 3, 8, 8, 8),
    ),
    "AdaptiveMaxPool1d": ModuleSpec(
        init_kwargs={"output_size": 4},
        input_shape=(2, 3, 16),
    ),
    "AdaptiveMaxPool2d": ModuleSpec(
        init_kwargs={"output_size": (4, 4)},
        input_shape=(2, 3, 16, 16),
    ),
    "AdaptiveMaxPool3d": ModuleSpec(
        init_kwargs={"output_size": (4, 4, 4)},
        input_shape=(2, 3, 8, 8, 8),
    ),
    # Normalization layers
    "BatchNorm1d": ModuleSpec(
        init_kwargs={"num_features": 8},
        input_shape=(4, 8, 16),
        eval_mode=True,  # Use eval mode for deterministic behavior
    ),
    "BatchNorm2d": ModuleSpec(
        init_kwargs={"num_features": 16},
        input_shape=(2, 16, 8, 8),
        eval_mode=True,
    ),
    "BatchNorm3d": ModuleSpec(
        init_kwargs={"num_features": 8},
        input_shape=(2, 8, 4, 4, 4),
        eval_mode=True,
    ),
    "LayerNorm": ModuleSpec(
        init_kwargs={"normalized_shape": [8]},
        input_shape=(4, 6, 8),
    ),
    "GroupNorm": ModuleSpec(
        init_kwargs={"num_groups": 4, "num_channels": 8},
        input_shape=(4, 8, 6, 6),
    ),
    "InstanceNorm1d": ModuleSpec(
        init_kwargs={"num_features": 8},
        input_shape=(4, 8, 16),
        eval_mode=True,
    ),
    "InstanceNorm2d": ModuleSpec(
        init_kwargs={"num_features": 8},
        input_shape=(4, 8, 6, 6),
        eval_mode=True,
    ),
    "InstanceNorm3d": ModuleSpec(
        init_kwargs={"num_features": 8},
        input_shape=(4, 8, 4, 4, 4),
        eval_mode=True,
    ),
    # Activation layers
    "ReLU": ModuleSpec(input_shape=(4, 8)),
    "ReLU6": ModuleSpec(input_shape=(4, 8)),
    "LeakyReLU": ModuleSpec(input_shape=(4, 8)),
    "PReLU": ModuleSpec(input_shape=(4, 8)),
    "ELU": ModuleSpec(input_shape=(4, 8)),
    "SELU": ModuleSpec(input_shape=(4, 8)),
    "CELU": ModuleSpec(input_shape=(4, 8)),
    "GELU": ModuleSpec(input_shape=(4, 8)),
    "SiLU": ModuleSpec(input_shape=(4, 8)),
    "Mish": ModuleSpec(input_shape=(4, 8)),
    "Hardswish": ModuleSpec(input_shape=(4, 8)),
    "Hardsigmoid": ModuleSpec(input_shape=(4, 8)),
    "Hardtanh": ModuleSpec(input_shape=(4, 8)),
    "Tanh": ModuleSpec(input_shape=(4, 8)),
    "Sigmoid": ModuleSpec(input_shape=(4, 8)),
    "Softplus": ModuleSpec(input_shape=(4, 8)),
    "Softshrink": ModuleSpec(input_shape=(4, 8)),
    "Softsign": ModuleSpec(input_shape=(4, 8)),
    "Tanhshrink": ModuleSpec(input_shape=(4, 8)),
    "Threshold": ModuleSpec(
        init_kwargs={"threshold": 0.5, "value": 0.0},
        input_shape=(4, 8),
    ),
    "Softmax": ModuleSpec(
        init_kwargs={"dim": -1},
        input_shape=(4, 10),
    ),
    "Softmax2d": ModuleSpec(
        input_shape=(2, 3, 4, 4),
    ),
    "LogSoftmax": ModuleSpec(
        init_kwargs={"dim": -1},
        input_shape=(4, 10),
    ),
    "GLU": ModuleSpec(
        init_kwargs={"dim": -1},
        input_shape=(4, 16),  # Must be even for GLU
    ),
    # Embedding layers
    "Embedding": ModuleSpec(
        init_kwargs={"num_embeddings": 100, "embedding_dim": 32},
        input_shape=(4, 8),
        extra_inputs={"input_dtype": np.int64},
    ),
    "EmbeddingBag": ModuleSpec(
        init_kwargs={"num_embeddings": 100, "embedding_dim": 32, "mode": "mean"},
        input_shape=(4, 8),
        extra_inputs={"input_dtype": np.int64},
    ),
    # Recurrent layers
    "RNN": ModuleSpec(
        init_kwargs={"input_size": 10, "hidden_size": 20, "batch_first": True},
        input_shape=(4, 8, 10),  # (batch, seq, input)
    ),
    "LSTM": ModuleSpec(
        init_kwargs={"input_size": 10, "hidden_size": 20, "batch_first": True},
        input_shape=(4, 8, 10),
    ),
    "GRU": ModuleSpec(
        init_kwargs={"input_size": 10, "hidden_size": 20, "batch_first": True},
        input_shape=(4, 8, 10),
    ),
    "RNNCell": ModuleSpec(
        init_kwargs={"input_size": 10, "hidden_size": 20},
        input_shape=(4, 10),  # Single timestep
        extra_inputs={"hidden_shape": (4, 20)},
    ),
    "LSTMCell": ModuleSpec(
        init_kwargs={"input_size": 10, "hidden_size": 20},
        input_shape=(4, 10),
        extra_inputs={"hidden_shape": (4, 20), "cell_shape": (4, 20)},
    ),
    "GRUCell": ModuleSpec(
        init_kwargs={"input_size": 10, "hidden_size": 20},
        input_shape=(4, 10),
        extra_inputs={"hidden_shape": (4, 20)},
    ),
    # Transformer layers
    "TransformerEncoderLayer": ModuleSpec(
        init_kwargs={"d_model": 32, "nhead": 4, "batch_first": True},
        input_shape=(4, 8, 32),  # (batch, seq, d_model)
        eval_mode=True,  # Disable dropout for deterministic testing
    ),
    "TransformerDecoderLayer": ModuleSpec(
        init_kwargs={"d_model": 32, "nhead": 4, "batch_first": True},
        input_shape=(4, 8, 32),
        extra_inputs={"memory_shape": (4, 10, 32)},
        eval_mode=True,  # Disable dropout for deterministic testing
    ),
    "MultiheadAttention": ModuleSpec(
        init_kwargs={"embed_dim": 32, "num_heads": 4, "batch_first": True},
        input_shape=(4, 8, 32),  # query
        extra_inputs={"key_shape": (4, 10, 32), "value_shape": (4, 10, 32)},
        eval_mode=True,  # Disable dropout for deterministic testing
    ),
    # Loss functions (need targets)
    "MSELoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10)},
    ),
    "L1Loss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10)},
    ),
    "SmoothL1Loss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10)},
    ),
    "HuberLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10)},
    ),
    "CrossEntropyLoss": ModuleSpec(
        input_shape=(4, 10),  # logits
        extra_inputs={"target_shape": (4,), "target_dtype": np.int64, "target_max": 10},
    ),
    "NLLLoss": ModuleSpec(
        input_shape=(4, 10),  # log probs
        extra_inputs={"target_shape": (4,), "target_dtype": np.int64, "target_max": 10},
    ),
    "BCELoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "input_transform": "sigmoid", "target_transform": "binary"},
    ),
    "BCEWithLogitsLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "target_transform": "binary"},
    ),
    "KLDivLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "input_transform": "log_softmax", "target_transform": "softmax"},
    ),
    "PoissonNLLLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "input_transform": "positive", "target_transform": "positive_int"},
    ),
    "CosineEmbeddingLoss": ModuleSpec(
        input_shape=(4, 32),
        extra_inputs={"input2_shape": (4, 32), "target_shape": (4,), "target_values": [-1, 1]},
    ),
    "MarginRankingLoss": ModuleSpec(
        input_shape=(4,),
        extra_inputs={"input2_shape": (4,), "target_shape": (4,), "target_values": [-1, 1]},
    ),
    "TripletMarginLoss": ModuleSpec(
        input_shape=(4, 32),  # anchor
        extra_inputs={"positive_shape": (4, 32), "negative_shape": (4, 32)},
    ),
    "CTCLoss": ModuleSpec(
        init_kwargs={"blank": 0},
        input_shape=(50, 4, 20),  # (T, N, C) - log probs
        extra_inputs={
            "target_shape": (4, 30),  # (N, S)
            "target_dtype": np.int64,
            "target_max": 19,
            "input_lengths": [50, 50, 50, 50],
            "target_lengths": [30, 25, 20, 28],
        },
    ),
    # Flatten and reshape
    "Flatten": ModuleSpec(
        input_shape=(2, 3, 4, 5),
    ),
    "Unflatten": ModuleSpec(
        init_kwargs={"dim": 1, "unflattened_size": (3, 4)},
        input_shape=(2, 12, 5),
    ),
    # Identity
    "Identity": ModuleSpec(
        input_shape=(4, 8),
    ),
    # Additional activation layers
    "Hardshrink": ModuleSpec(input_shape=(4, 8)),
    "LogSigmoid": ModuleSpec(input_shape=(4, 8)),
    "Softmin": ModuleSpec(
        init_kwargs={"dim": -1},
        input_shape=(4, 10),
    ),
    "Softmax2d": ModuleSpec(
        input_shape=(2, 3, 4, 4),
    ),
    # Pixel shuffle/unshuffle
    "PixelShuffle": ModuleSpec(
        init_kwargs={"upscale_factor": 2},
        input_shape=(2, 8, 4, 4),  # C must be divisible by upscale_factor^2
    ),
    "PixelUnshuffle": ModuleSpec(
        init_kwargs={"downscale_factor": 2},
        input_shape=(2, 2, 8, 8),
    ),
    # Upsample layers
    "Upsample": ModuleSpec(
        init_kwargs={"scale_factor": 2, "mode": "nearest"},
        input_shape=(2, 3, 8, 8),
    ),
    "UpsamplingNearest2d": ModuleSpec(
        init_kwargs={"scale_factor": 2},
        input_shape=(2, 3, 8, 8),
    ),
    "UpsamplingBilinear2d": ModuleSpec(
        init_kwargs={"scale_factor": 2},
        input_shape=(2, 3, 8, 8),
    ),
    # Padding layers - 1D
    "ReflectionPad1d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 16),
    ),
    "ReplicationPad1d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 16),
    ),
    "ConstantPad1d": ModuleSpec(
        init_kwargs={"padding": 2, "value": 0.0},
        input_shape=(2, 3, 16),
    ),
    # Padding layers - 2D
    "ZeroPad2d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 8, 8),
    ),
    "ReflectionPad2d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 8, 8),
    ),
    "ReplicationPad2d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 8, 8),
    ),
    "ConstantPad2d": ModuleSpec(
        init_kwargs={"padding": 2, "value": 0.0},
        input_shape=(2, 3, 8, 8),
    ),
    "CircularPad2d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 8, 8),
    ),
    # Padding layers - 3D
    "ZeroPad3d": ModuleSpec(
        init_kwargs={"padding": 1},
        input_shape=(2, 3, 4, 4, 4),
    ),
    "ReflectionPad3d": ModuleSpec(
        init_kwargs={"padding": 1},
        input_shape=(2, 3, 4, 4, 4),
    ),
    "ReplicationPad3d": ModuleSpec(
        init_kwargs={"padding": 1},
        input_shape=(2, 3, 4, 4, 4),
    ),
    "ConstantPad3d": ModuleSpec(
        init_kwargs={"padding": 1, "value": 0.0},
        input_shape=(2, 3, 4, 4, 4),
    ),
    # Channel shuffle
    "ChannelShuffle": ModuleSpec(
        init_kwargs={"groups": 2},
        input_shape=(2, 8, 4, 4),  # C must be divisible by groups
    ),
    # Similarity/distance modules
    "CosineSimilarity": ModuleSpec(
        init_kwargs={"dim": 1},
        input_shape=(4, 32),
        extra_inputs={"input2_shape": (4, 32)},
    ),
    "PairwiseDistance": ModuleSpec(
        init_kwargs={"p": 2.0},
        input_shape=(4, 32),
        extra_inputs={"input2_shape": (4, 32)},
    ),
    # Fold/Unfold
    # Fold input shape: (N, C*kH*kW, L) where L = number of sliding blocks
    # For output_size=(8,8), kernel_size=(2,2), stride=(1,1): L = 7*7 = 49
    "Fold": ModuleSpec(
        init_kwargs={"output_size": (8, 8), "kernel_size": (2, 2)},
        input_shape=(2, 12, 49),  # 12 = 3*2*2, 49 = 7*7 sliding blocks
    ),
    "Unfold": ModuleSpec(
        init_kwargs={"kernel_size": (2, 2)},
        input_shape=(2, 3, 8, 8),
    ),
    # LocalResponseNorm
    "LocalResponseNorm": ModuleSpec(
        init_kwargs={"size": 5},
        input_shape=(2, 8, 16, 16),
    ),
    # RMSNorm (if available in mlx_compat)
    "RMSNorm": ModuleSpec(
        init_kwargs={"normalized_shape": 8},
        input_shape=(4, 6, 8),
    ),
    # Additional loss functions
    "GaussianNLLLoss": ModuleSpec(
        input_shape=(4, 10),  # mean
        extra_inputs={
            "target_shape": (4, 10),
            "var_shape": (4, 10),
            "var_transform": "positive",
        },
    ),
    "HingeEmbeddingLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "target_values": [-1, 1]},
    ),
    "MultiLabelMarginLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "target_dtype": np.int64, "target_transform": "label_margin"},
    ),
    "MultiLabelSoftMarginLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "target_transform": "binary"},
    ),
    "MultiMarginLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4,), "target_dtype": np.int64, "target_max": 10},
    ),
    "SoftMarginLoss": ModuleSpec(
        input_shape=(4, 10),
        extra_inputs={"target_shape": (4, 10), "target_values": [-1, 1]},
    ),
    "TripletMarginWithDistanceLoss": ModuleSpec(
        input_shape=(4, 32),  # anchor
        extra_inputs={"positive_shape": (4, 32), "negative_shape": (4, 32)},
    ),
    # Fractional pooling
    "FractionalMaxPool2d": ModuleSpec(
        init_kwargs={"kernel_size": 2, "output_size": (4, 4)},
        input_shape=(2, 3, 8, 8),
    ),
    "FractionalMaxPool3d": ModuleSpec(
        init_kwargs={"kernel_size": 2, "output_size": (2, 2, 2)},
        input_shape=(2, 3, 4, 4, 4),
    ),
    # LP pooling
    "LPPool1d": ModuleSpec(
        init_kwargs={"norm_type": 2, "kernel_size": 2},
        input_shape=(2, 3, 16),
    ),
    "LPPool2d": ModuleSpec(
        init_kwargs={"norm_type": 2, "kernel_size": 2},
        input_shape=(2, 3, 8, 8),
    ),
    "LPPool3d": ModuleSpec(
        init_kwargs={"norm_type": 2, "kernel_size": 2},
        input_shape=(2, 3, 4, 4, 4),
    ),
    # MaxUnpool (needs indices from max_pool - uses _maxunpool_input special handling)
    "MaxUnpool1d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 16),  # Original input shape before max_pool
        extra_inputs={"_maxunpool_input": "1d", "kernel_size": 2},
    ),
    "MaxUnpool2d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 8, 8),  # Original input shape before max_pool
        extra_inputs={"_maxunpool_input": "2d", "kernel_size": 2},
    ),
    "MaxUnpool3d": ModuleSpec(
        init_kwargs={"kernel_size": 2},
        input_shape=(2, 3, 4, 4, 4),  # Original input shape before max_pool
        extra_inputs={"_maxunpool_input": "3d", "kernel_size": 2},
    ),
    # SyncBatchNorm
    "SyncBatchNorm": ModuleSpec(
        init_kwargs={"num_features": 8},
        input_shape=(4, 8, 16),
        eval_mode=True,
    ),
    # Lazy layers
    "LazyBatchNorm1d": ModuleSpec(
        input_shape=(4, 8, 16),
        eval_mode=True,
    ),
    "LazyBatchNorm2d": ModuleSpec(
        input_shape=(2, 8, 4, 4),
        eval_mode=True,
    ),
    "LazyBatchNorm3d": ModuleSpec(
        input_shape=(2, 8, 2, 2, 2),
        eval_mode=True,
    ),
    "LazyInstanceNorm1d": ModuleSpec(
        input_shape=(4, 8, 16),
        eval_mode=True,
    ),
    "LazyInstanceNorm2d": ModuleSpec(
        input_shape=(4, 8, 6, 6),
        eval_mode=True,
    ),
    "LazyInstanceNorm3d": ModuleSpec(
        input_shape=(4, 8, 4, 4, 4),
        eval_mode=True,
    ),
    "LazyConv1d": ModuleSpec(
        init_kwargs={"out_channels": 8, "kernel_size": 3, "padding": 1},
        input_shape=(2, 3, 16),
    ),
    "LazyConv2d": ModuleSpec(
        init_kwargs={"out_channels": 16, "kernel_size": 3, "padding": 1},
        input_shape=(2, 3, 16, 16),
    ),
    "LazyConv3d": ModuleSpec(
        init_kwargs={"out_channels": 8, "kernel_size": 3, "padding": 1},
        input_shape=(2, 3, 8, 8, 8),
    ),
    "LazyConvTranspose1d": ModuleSpec(
        init_kwargs={"out_channels": 3, "kernel_size": 3, "padding": 1},
        input_shape=(2, 8, 16),
    ),
    "LazyConvTranspose2d": ModuleSpec(
        init_kwargs={"out_channels": 3, "kernel_size": 3, "padding": 1},
        input_shape=(2, 16, 16, 16),
    ),
    "LazyConvTranspose3d": ModuleSpec(
        init_kwargs={"out_channels": 3, "kernel_size": 3, "padding": 1},
        input_shape=(2, 8, 8, 8, 8),
    ),
    # Additional padding layers
    "CircularPad1d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 16),
    ),
    "CircularPad3d": ModuleSpec(
        init_kwargs={"padding": 1},
        input_shape=(2, 3, 4, 4, 4),
    ),
    # Additional activation layers
    "RReLU": ModuleSpec(
        init_kwargs={"lower": 0.125, "upper": 0.3333},
        input_shape=(4, 8),
        eval_mode=True,  # Use eval mode for deterministic behavior
    ),
    # Additional normalization
    "CrossMapLRN2d": ModuleSpec(
        init_kwargs={"size": 5},
        input_shape=(2, 8, 16, 16),
    ),
    # Deprecated loss
    "NLLLoss2d": ModuleSpec(
        input_shape=(4, 10, 8, 8),  # log probs (N, C, H, W)
        extra_inputs={"target_shape": (4, 8, 8), "target_dtype": np.int64, "target_max": 10},
    ),
    # AdaptiveLogSoftmaxWithLoss (complex - needs special handling)
    "AdaptiveLogSoftmaxWithLoss": ModuleSpec(
        init_kwargs={"in_features": 32, "n_classes": 100, "cutoffs": [10, 50]},
        input_shape=(4, 32),
        extra_inputs={"target_shape": (4,), "target_dtype": np.int64, "target_max": 100},
    ),
    # Transformer modules
    "Transformer": ModuleSpec(
        init_kwargs={"d_model": 32, "nhead": 4, "batch_first": True},
        input_shape=(4, 8, 32),  # src: (batch, seq, d_model)
        extra_inputs={"tgt_shape": (4, 10, 32)},  # tgt: (batch, seq, d_model)
        eval_mode=True,  # Disable dropout for deterministic testing
    ),
    "TransformerEncoder": ModuleSpec(
        init_kwargs={
            "encoder_layer_kwargs": {"d_model": 32, "nhead": 4, "batch_first": True},
            "num_layers": 2,
        },
        input_shape=(4, 8, 32),
        extra_inputs={"needs_encoder_layer": True},
        eval_mode=True,  # Disable dropout for deterministic testing
    ),
    "TransformerDecoder": ModuleSpec(
        init_kwargs={
            "decoder_layer_kwargs": {"d_model": 32, "nhead": 4, "batch_first": True},
            "num_layers": 2,
        },
        input_shape=(4, 8, 32),
        extra_inputs={"memory_shape": (4, 10, 32), "needs_decoder_layer": True},
        eval_mode=True,  # Disable dropout for deterministic testing
    ),
    # Additional padding
    "ZeroPad1d": ModuleSpec(
        init_kwargs={"padding": 2},
        input_shape=(2, 3, 16),
    ),
}


# ============================================================================
# Optimizer Specifications
# ============================================================================


@dataclass
class OptimizerSpec:
    """Specification for testing an optimizer."""

    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    param_shape: Tuple[int, ...] = (5, 3)
    num_steps: int = 1


OPTIMIZER_SPECS: Dict[str, OptimizerSpec] = {
    "SGD": OptimizerSpec(
        init_kwargs={"lr": 0.01},
    ),
    "Adam": OptimizerSpec(
        init_kwargs={"lr": 0.001},
    ),
    "AdamW": OptimizerSpec(
        init_kwargs={"lr": 0.001, "weight_decay": 0.01},
    ),
    "Adamax": OptimizerSpec(
        init_kwargs={"lr": 0.002},
    ),
    "Adadelta": OptimizerSpec(
        init_kwargs={"lr": 1.0},
    ),
    "Adagrad": OptimizerSpec(
        init_kwargs={"lr": 0.01},
    ),
    "RMSprop": OptimizerSpec(
        init_kwargs={"lr": 0.01},
    ),
    "Rprop": OptimizerSpec(
        init_kwargs={"lr": 0.01},
    ),
    "ASGD": OptimizerSpec(
        init_kwargs={"lr": 0.01},
    ),
    "NAdam": OptimizerSpec(
        init_kwargs={"lr": 0.002},
    ),
    "RAdam": OptimizerSpec(
        init_kwargs={"lr": 0.001},
    ),
}


# ============================================================================
# LR Scheduler Specifications
# ============================================================================


@dataclass
class LRSchedulerSpec:
    """Specification for testing an LR scheduler."""

    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_steps: int = 5  # Number of scheduler steps to compare
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"lr": 0.1})
    # For ReduceLROnPlateau which needs metrics
    step_arg_generator: Optional[Callable[[], Any]] = None


# LR Scheduler specifications for common schedulers
LR_SCHEDULER_SPECS: Dict[str, LRSchedulerSpec] = {
    "StepLR": LRSchedulerSpec(
        init_kwargs={"step_size": 2, "gamma": 0.5},
    ),
    "MultiStepLR": LRSchedulerSpec(
        init_kwargs={"milestones": [2, 4], "gamma": 0.5},
    ),
    "ExponentialLR": LRSchedulerSpec(
        init_kwargs={"gamma": 0.9},
    ),
    "CosineAnnealingLR": LRSchedulerSpec(
        init_kwargs={"T_max": 10},
    ),
    "ConstantLR": LRSchedulerSpec(
        init_kwargs={"factor": 0.5, "total_iters": 3},
    ),
    "LinearLR": LRSchedulerSpec(
        init_kwargs={"start_factor": 0.1, "end_factor": 1.0, "total_iters": 5},
    ),
    "PolynomialLR": LRSchedulerSpec(
        init_kwargs={"total_iters": 5, "power": 2.0},
    ),
    "CosineAnnealingWarmRestarts": LRSchedulerSpec(
        init_kwargs={"T_0": 5, "T_mult": 1},
    ),
    "CyclicLR": LRSchedulerSpec(
        init_kwargs={"base_lr": 0.01, "max_lr": 0.1, "step_size_up": 2},
        num_steps=10,
    ),
    "OneCycleLR": LRSchedulerSpec(
        init_kwargs={"max_lr": 0.1, "total_steps": 10},
        num_steps=10,
    ),
    "ReduceLROnPlateau": LRSchedulerSpec(
        init_kwargs={"mode": "min", "factor": 0.5, "patience": 2},
        num_steps=10,
        # Generate decreasing then flat metrics to trigger reduction
        step_arg_generator=lambda: iter([1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
    ),
    # These schedulers require other schedulers or lambdas, need special handling
    "LambdaLR": LRSchedulerSpec(
        init_kwargs={"lr_lambda": lambda epoch: 0.95 ** epoch},
    ),
    "MultiplicativeLR": LRSchedulerSpec(
        init_kwargs={"lr_lambda": lambda epoch: 0.95},
    ),
}


# ============================================================================
# NN Utils Specifications
# ============================================================================


@dataclass
class NNUtilsSpec:
    """Specification for testing torch.nn.utils functions.

    These functions require special setup with modules, parameters, or gradients.
    """

    # Type of test: "grad_clip", "param_vector", "module_norm", "fusion"
    test_type: str
    # Shape for parameters/tensors
    param_shape: Tuple[int, ...] = (5, 3)
    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)


# NN Utils specifications for torch.nn.utils functions
NN_UTILS_SPECS: Dict[str, NNUtilsSpec] = {
    # Gradient clipping functions - require parameters with gradients
    "clip_grad_norm_": NNUtilsSpec(
        test_type="grad_clip",
        param_shape=(5, 3),
        config={"max_norm": 1.0, "norm_type": 2.0},
    ),
    "clip_grad_norm": NNUtilsSpec(
        test_type="grad_clip",
        param_shape=(5, 3),
        config={"max_norm": 1.0, "norm_type": 2.0},
    ),
    "clip_grad_value_": NNUtilsSpec(
        test_type="grad_value_clip",
        param_shape=(5, 3),
        config={"clip_value": 0.5},
    ),
    "clip_grads_with_norm_": NNUtilsSpec(
        test_type="grad_clip_with_norm",
        param_shape=(5, 3),
        config={"max_norm": 1.0},
    ),
    "get_total_norm": NNUtilsSpec(
        test_type="total_norm",
        param_shape=(5, 3),
        config={"norm_type": 2.0},
    ),

    # Parameter vector functions
    "parameters_to_vector": NNUtilsSpec(
        test_type="param_to_vec",
        param_shape=(5, 3),
    ),
    "vector_to_parameters": NNUtilsSpec(
        test_type="vec_to_param",
        param_shape=(5, 3),
    ),

    # Module normalization functions (stubs in MLX)
    "weight_norm": NNUtilsSpec(
        test_type="module_norm",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),
    "remove_weight_norm": NNUtilsSpec(
        test_type="module_norm_remove",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),
    "spectral_norm": NNUtilsSpec(
        test_type="module_norm",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),
    "remove_spectral_norm": NNUtilsSpec(
        test_type="module_norm_remove",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),

    # Fusion functions (stubs in MLX)
    "fuse_conv_bn_eval": NNUtilsSpec(
        test_type="fusion_conv_bn",
        config={"in_channels": 3, "out_channels": 16, "kernel_size": 3},
    ),
    "fuse_conv_bn_weights": NNUtilsSpec(
        test_type="fusion_conv_bn_weights",
        config={"in_channels": 3, "out_channels": 16, "kernel_size": 3},
    ),
    "fuse_linear_bn_eval": NNUtilsSpec(
        test_type="fusion_linear_bn",
        config={"in_features": 10, "out_features": 5},
    ),
    "fuse_linear_bn_weights": NNUtilsSpec(
        test_type="fusion_linear_bn_weights",
        config={"in_features": 10, "out_features": 5},
    ),

    # Memory format conversion (no-ops in MLX)
    "convert_conv2d_weight_memory_format": NNUtilsSpec(
        test_type="memory_format",
        config={"module_type": "Conv2d", "in_channels": 3, "out_channels": 16, "kernel_size": 3},
    ),
    "convert_conv3d_weight_memory_format": NNUtilsSpec(
        test_type="memory_format",
        config={"module_type": "Conv3d", "in_channels": 3, "out_channels": 8, "kernel_size": 3},
    ),

    # Skip init (create module without init)
    "skip_init": NNUtilsSpec(
        test_type="skip_init",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),

    # RNN utility functions (torch.nn.utils.rnn)
    # Use full module path as key since they're in a submodule
    # Note: pack/unpack functions are stubs in MLX - they just verify the API works
    "torch.nn.utils.rnn.PackedSequence": NNUtilsSpec(
        test_type="rnn_packed_sequence_class",
        config={"feature_size": 16, "seq_len": 10},
    ),
    "torch.nn.utils.rnn.pack_padded_sequence": NNUtilsSpec(
        test_type="rnn_pack_padded",
        config={"seq_len": 10, "batch_size": 4, "feature_size": 16, "is_stub": True},
    ),
    "torch.nn.utils.rnn.pad_packed_sequence": NNUtilsSpec(
        test_type="rnn_pad_packed",
        config={"seq_len": 10, "batch_size": 4, "feature_size": 16, "is_stub": True},
    ),
    "torch.nn.utils.rnn.pad_sequence": NNUtilsSpec(
        test_type="rnn_pad_sequence",
        config={"lengths": [10, 8, 6], "feature_size": 16},
    ),
    "torch.nn.utils.rnn.unpad_sequence": NNUtilsSpec(
        test_type="rnn_unpad_sequence",
        config={"lengths": [10, 8, 6], "feature_size": 16},
    ),
    "torch.nn.utils.rnn.pack_sequence": NNUtilsSpec(
        test_type="rnn_pack_sequence",
        config={"lengths": [10, 8, 6], "feature_size": 16, "is_stub": True},
    ),
    "torch.nn.utils.rnn.unpack_sequence": NNUtilsSpec(
        test_type="rnn_unpack_sequence",
        config={"lengths": [10, 8, 6], "feature_size": 16, "is_stub": True},
    ),

    # ==========================================================================
    # torch.nn.utils.parametrizations functions
    # ==========================================================================
    "torch.nn.utils.parametrizations.orthogonal": NNUtilsSpec(
        test_type="parametrization",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),
    "torch.nn.utils.parametrizations.spectral_norm": NNUtilsSpec(
        test_type="parametrization",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),
    "torch.nn.utils.parametrizations.weight_norm": NNUtilsSpec(
        test_type="parametrization",
        config={"module_type": "Linear", "in_features": 10, "out_features": 5},
    ),

    # ==========================================================================
    # torch.nn.utils.parametrize functions
    # ==========================================================================
    "torch.nn.utils.parametrize.is_parametrized": NNUtilsSpec(
        test_type="is_parametrized",
        config={},
    ),
    "torch.nn.utils.parametrize.register_parametrization": NNUtilsSpec(
        test_type="register_parametrization",
        config={},
    ),
    "torch.nn.utils.parametrize.remove_parametrizations": NNUtilsSpec(
        test_type="remove_parametrizations",
        config={},
    ),
    "torch.nn.utils.parametrize.type_before_parametrizations": NNUtilsSpec(
        test_type="type_before_parametrizations",
        config={},
    ),
    "torch.nn.utils.parametrize.transfer_parametrizations_and_params": NNUtilsSpec(
        test_type="transfer_parametrizations",
        config={},
    ),

    # ==========================================================================
    # torch.nn.utils.stateless functions
    # ==========================================================================
    "torch.nn.utils.stateless.functional_call": NNUtilsSpec(
        test_type="functional_call",
        config={"in_features": 10, "out_features": 5, "batch_size": 4},
    ),
}


# ============================================================================
# Input Generator Registry
# ============================================================================


class InputGeneratorRegistry:
    """
    Registry mapping API patterns to input generators.

    Provides default test inputs for various PyTorch/mlx_compat operations.
    """

    def __init__(self):
        self._generators: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self._pattern_generators: List[Tuple[re.Pattern, Callable]] = []
        self._register_defaults()

    def _register_defaults(self):
        """Register default input generators for common operations."""
        # Unary tensor operations
        unary_ops = [
            "abs", "neg", "sign", "ceil", "floor", "round", "trunc",
            "exp", "exp2", "expm1", "log1p",
            "sin", "cos", "tan", "sinh", "cosh", "tanh",
            "relu", "sigmoid", "tanh", "gelu", "silu", "mish",
            "softplus", "softsign", "hardsigmoid", "hardswish", "hardtanh",
            "elu", "selu", "celu", "leaky_relu", "rrelu", "relu6",
            "erf", "erfc", "erfinv",
            "reciprocal", "rsqrt", "square",
            "logical_not", "bitwise_not",
            "isnan", "isinf", "isfinite", "isneginf", "isposinf", "isreal",
            "real", "imag", "conj", "angle",
            "deg2rad", "rad2deg",
            "nan_to_num",
        ]
        for op in unary_ops:
            self._generators[f"torch.{op}"] = unary_tensor_input
            self._generators[f"torch.nn.functional.{op}"] = unary_tensor_input

        # Positive-only unary ops
        positive_ops = ["sqrt", "log", "log2", "log10", "lgamma", "digamma"]
        for op in positive_ops:
            self._generators[f"torch.{op}"] = unary_positive_input

        # Bounded unary ops (-1, 1) for asin, acos, atanh
        bounded_ops = ["asin", "acos", "atanh"]
        for op in bounded_ops:
            self._generators[f"torch.{op}"] = lambda: unary_bounded_input()

        # acosh needs input >= 1
        self._generators["torch.acosh"] = unary_ge_one_input

        # Binary tensor operations
        binary_ops = [
            "add", "sub", "mul", "fmod", "remainder",
            "maximum", "minimum", "fmax", "fmin",
            "eq", "ne", "lt", "le", "gt", "ge",
            "logical_and", "logical_or", "logical_xor",
            "atan2", "hypot", "copysign", "nextafter",
            "xlogy", "special.xlogy",
        ]
        for op in binary_ops:
            self._generators[f"torch.{op}"] = binary_tensor_input

        # pow needs special handling - exponent as second positional arg
        self._generators["torch.pow"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "exponent": np.random.uniform(0.5, 2.0, (4, 8)).astype(np.float32),
        }

        # Bitwise ops need integer inputs
        def bitwise_input():
            return {
                "input": np.random.randint(0, 256, (4, 8)).astype(np.int32),
                "other": np.random.randint(0, 256, (4, 8)).astype(np.int32),
            }
        for op in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
            self._generators[f"torch.{op}"] = bitwise_input

        # bitwise_not - let it fail to show the implementation difference
        self._generators["torch.bitwise_not"] = lambda: {
            "input": np.random.randint(0, 100, (4, 8)).astype(np.int32),
        }

        # Division (avoid div by zero) - use positional name
        div_ops = ["div", "true_divide", "floor_divide"]
        for op in div_ops:
            self._generators[f"torch.{op}"] = binary_positive_input

        # frac is unary
        self._generators["torch.frac"] = unary_tensor_input

        # Matrix operations - use correct argument names
        self._generators["torch.matmul"] = matmul_input
        # mm and bmm use mat1/mat2 not input/other
        self._generators["torch.mm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "mat2": np.random.randn(8, 6).astype(np.float32),
        }
        self._generators["torch.bmm"] = lambda: {
            "input": np.random.randn(2, 4, 8).astype(np.float32),
            "mat2": np.random.randn(2, 8, 6).astype(np.float32),
        }

        # Reduction operations
        reduction_ops = ["sum", "mean", "prod", "std", "var",
                        "amax", "amin", "argmax", "argmin", "norm"]
        for op in reduction_ops:
            self._generators[f"torch.{op}"] = reduction_input

        # all/any return single bool, max/min return values+indices
        self._generators["torch.all"] = reduction_input
        self._generators["torch.any"] = reduction_input
        self._generators["torch.max"] = reduction_input
        self._generators["torch.min"] = reduction_input

        # logsumexp needs dim argument
        self._generators["torch.logsumexp"] = lambda: {
            "input": np.random.randn(4, 8, 6).astype(np.float32),
            "dim": 1,
        }

        # Softmax family
        softmax_ops = ["softmax", "log_softmax", "softmin"]
        for op in softmax_ops:
            self._generators[f"torch.{op}"] = softmax_input
            self._generators[f"torch.nn.functional.{op}"] = softmax_input

        # log1p needs values > -1
        self._generators["torch.log1p"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32))
        }

        # erfinv needs values in (-1, 1)
        self._generators["torch.erfinv"] = lambda: unary_bounded_input()

        # =====================================================================
        # Additional torch.* functions
        # =====================================================================

        # Tensor creation functions
        self._generators["torch.zeros"] = lambda: {"size": (4, 8)}
        self._generators["torch.ones"] = lambda: {"size": (4, 8)}
        self._generators["torch.empty"] = lambda: {"size": (4, 8)}
        self._generators["torch.full"] = lambda: {"size": (4, 8), "fill_value": 3.14}
        self._generators["torch.arange"] = lambda: {"start": 0, "end": 10, "step": 1}
        self._generators["torch.linspace"] = lambda: {"start": 0.0, "end": 1.0, "steps": 10}
        self._generators["torch.logspace"] = lambda: {"start": 0.0, "end": 2.0, "steps": 10}
        self._generators["torch.eye"] = lambda: {"n": 4}
        self._generators["torch.zeros_like"] = unary_tensor_input
        self._generators["torch.ones_like"] = unary_tensor_input
        self._generators["torch.empty_like"] = unary_tensor_input
        self._generators["torch.full_like"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "fill_value": 2.5,
        }

        # Shape manipulation - these need _tensor_list marker
        self._generators["torch.cat"] = lambda: {
            "_tensor_list": [
                np.random.randn(2, 8).astype(np.float32),
                np.random.randn(3, 8).astype(np.float32),
            ],
            "dim": 0,
        }
        self._generators["torch.stack"] = lambda: {
            "_tensor_list": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(4, 8).astype(np.float32),
            ],
            "dim": 0,
        }
        # chunk and split - use tensor argument name
        self._generators["torch.chunk"] = lambda: {
            "tensor": np.random.randn(12, 8).astype(np.float32),
            "chunks": 3,
            "dim": 0,
        }
        self._generators["torch.split"] = lambda: {
            "tensor": np.random.randn(10, 8).astype(np.float32),
            "split_size_or_sections": 2,
            "dim": 0,
        }
        self._generators["torch.tensor_split"] = lambda: {
            "input": np.random.randn(10, 8).astype(np.float32),
            "indices_or_sections": 2,
            "dim": 0,
        }
        self._generators["torch.hsplit"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "indices_or_sections": 2,
        }
        self._generators["torch.vsplit"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "indices_or_sections": 2,
        }
        self._generators["torch.dsplit"] = lambda: {
            "input": np.random.randn(4, 4, 8).astype(np.float32),
            "indices_or_sections": 2,
        }

        # More concatenation variants
        self._generators["torch.hstack"] = lambda: {
            "_tensor_list": [
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4, 5).astype(np.float32),
            ],
        }
        self._generators["torch.vstack"] = lambda: {
            "_tensor_list": [
                np.random.randn(2, 8).astype(np.float32),
                np.random.randn(3, 8).astype(np.float32),
            ],
        }
        self._generators["torch.dstack"] = lambda: {
            "_tensor_list": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(4, 8).astype(np.float32),
            ],
        }
        self._generators["torch.column_stack"] = lambda: {
            "_tensor_list": [
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
            ],
        }
        self._generators["torch.row_stack"] = lambda: {
            "_tensor_list": [
                np.random.randn(4).astype(np.float32),
                np.random.randn(4).astype(np.float32),
            ],
        }

        # atleast functions
        self._generators["torch.atleast_1d"] = lambda: {
            "input": np.array(5.0, dtype=np.float32),
        }
        self._generators["torch.atleast_2d"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
        }
        self._generators["torch.atleast_3d"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # Matrix operations
        self._generators["torch.addmm"] = lambda: {
            "input": np.random.randn(4, 6).astype(np.float32),
            "mat1": np.random.randn(4, 8).astype(np.float32),
            "mat2": np.random.randn(8, 6).astype(np.float32),
        }
        self._generators["torch.addmv"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
            "mat": np.random.randn(4, 8).astype(np.float32),
            "vec": np.random.randn(8).astype(np.float32),
        }
        self._generators["torch.addbmm"] = lambda: {
            "input": np.random.randn(4, 6).astype(np.float32),
            "batch1": np.random.randn(3, 4, 8).astype(np.float32),
            "batch2": np.random.randn(3, 8, 6).astype(np.float32),
        }
        self._generators["torch.addr"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "vec1": np.random.randn(4).astype(np.float32),
            "vec2": np.random.randn(8).astype(np.float32),
        }
        self._generators["torch.baddbmm"] = lambda: {
            "input": np.random.randn(3, 4, 6).astype(np.float32),
            "batch1": np.random.randn(3, 4, 8).astype(np.float32),
            "batch2": np.random.randn(3, 8, 6).astype(np.float32),
        }

        # Aliases
        self._generators["torch.absolute"] = unary_tensor_input
        self._generators["torch.clip_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "min": -1.0,
            "max": 1.0,
        }
        self._generators["torch.concat"] = lambda: {
            "_tensor_list": [
                np.random.randn(2, 8).astype(np.float32),
                np.random.randn(3, 8).astype(np.float32),
            ],
            "dim": 0,
        }
        self._generators["torch.concatenate"] = lambda: {
            "_tensor_list": [
                np.random.randn(2, 8).astype(np.float32),
                np.random.randn(3, 8).astype(np.float32),
            ],
            "dim": 0,
        }

        # Fix arcsinh/arctan - unary functions
        self._generators["torch.arctan"] = unary_tensor_input
        self._generators["torch.arcsinh"] = unary_tensor_input

        # aminmax returns two tensors
        self._generators["torch.aminmax"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # argwhere
        self._generators["torch.argwhere"] = lambda: {
            "input": (np.random.rand(4, 8) > 0.5).astype(np.float32),
        }

        # narrow variants
        self._generators["torch.narrow"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
            "dim": 0,
            "start": 2,
            "length": 3,
        }
        self._generators["torch.narrow_copy"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
            "dim": 0,
            "start": 2,
            "length": 3,
        }

        # movedim / moveaxis / swapaxes / swapdims
        self._generators["torch.movedim"] = lambda: {
            "input": np.random.randn(2, 4, 8).astype(np.float32),
            "source": 0,
            "destination": 2,
        }
        self._generators["torch.moveaxis"] = lambda: {
            "input": np.random.randn(2, 4, 8).astype(np.float32),
            "source": 0,
            "destination": 2,
        }
        self._generators["torch.swapaxes"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "axis0": 0,
            "axis1": 1,
        }
        self._generators["torch.swapdims"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim0": 0,
            "dim1": 1,
        }

        # adjoint / conj_physical
        self._generators["torch.adjoint"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.conj_physical"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # view_as_real / view_as_complex - need complex input
        # These are complex-specific and may not work

        # count_nonzero
        self._generators["torch.count_nonzero"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # diff
        self._generators["torch.diff"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # gradient
        self._generators["torch.gradient"] = lambda: {
            "input": np.random.randn(8).astype(np.float32),
        }

        # histc / histogram
        self._generators["torch.histc"] = lambda: {
            "input": np.random.randn(100).astype(np.float32),
            "bins": 10,
        }
        self._generators["torch.histogram"] = lambda: {
            "input": np.random.randn(100).astype(np.float32),
            "bins": 10,
        }

        # bincount - let it fail to show the dtype handling bug
        self._generators["torch.bincount"] = lambda: {
            "input": np.random.randint(0, 10, (32,)).astype(np.int64),
        }

        # bucketize
        self._generators["torch.bucketize"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "boundaries": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        }

        # searchsorted
        self._generators["torch.searchsorted"] = lambda: {
            "sorted_sequence": np.sort(np.random.randn(10).astype(np.float32)),
            "values": np.random.randn(5).astype(np.float32),
        }

        # tensordot
        self._generators["torch.tensordot"] = lambda: {
            "a": np.random.randn(4, 8).astype(np.float32),
            "b": np.random.randn(8, 6).astype(np.float32),
            "dims": 1,
        }

        # vdot
        self._generators["torch.vdot"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
            "other": np.random.randn(16).astype(np.float32),
        }

        # cdist
        self._generators["torch.cdist"] = lambda: {
            "x1": np.random.randn(5, 8).astype(np.float32),
            "x2": np.random.randn(6, 8).astype(np.float32),
        }

        # pdist
        self._generators["torch.pdist"] = lambda: {
            "input": np.random.randn(5, 8).astype(np.float32),
        }

        # cosine_similarity
        self._generators["torch.cosine_similarity"] = lambda: {
            "x1": np.random.randn(4, 8).astype(np.float32),
            "x2": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }

        # Variance/covariance
        self._generators["torch.var_mean"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.std_mean"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.cov"] = lambda: {
            "input": np.random.randn(4, 100).astype(np.float32),
        }
        self._generators["torch.corrcoef"] = lambda: {
            "input": np.random.randn(4, 100).astype(np.float32),
        }

        # Quantile / percentile
        self._generators["torch.quantile"] = lambda: {
            "input": np.random.randn(100).astype(np.float32),
            "q": 0.5,
        }
        self._generators["torch.nanquantile"] = lambda: {
            "input": np.random.randn(100).astype(np.float32),
            "q": 0.5,
        }
        self._generators["torch.median"] = lambda: {
            "input": np.random.randn(100).astype(np.float32),
        }
        self._generators["torch.nanmedian"] = lambda: {
            "input": np.random.randn(100).astype(np.float32),
        }
        self._generators["torch.mode"] = lambda: {
            "input": np.random.randint(0, 10, (100,)).astype(np.float32),
        }

        # NaN-aware functions
        self._generators["torch.nanmean"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nansum"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # special reductions
        self._generators["torch.count_nonzero"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # tensor ops
        self._generators["torch.numel"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # logcumsumexp
        self._generators["torch.logcumsumexp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }

        # logaddexp variants
        self._generators["torch.logaddexp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.logaddexp2"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # scatter variants
        self._generators["torch.scatter_add"] = lambda: {
            "input": np.zeros((4, 8)).astype(np.float32),
            "dim": 1,
            "index": np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]]).astype(np.int64),
            "src": np.random.randn(4, 3).astype(np.float32),
        }
        self._generators["torch.scatter_reduce"] = lambda: {
            "input": np.zeros((4, 8)).astype(np.float32),
            "dim": 1,
            "index": np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]]).astype(np.int64),
            "src": np.random.randn(4, 3).astype(np.float32),
            "reduce": "sum",
        }

        # select / select_scatter
        self._generators["torch.select"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 0,
            "index": 2,
        }
        self._generators["torch.select_scatter"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "src": np.random.randn(8).astype(np.float32),
            "dim": 0,
            "index": 2,
        }
        self._generators["torch.slice_scatter"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "src": np.random.randn(2, 8).astype(np.float32),
            "dim": 0,
            "start": 1,
            "end": 3,
        }
        self._generators["torch.diagonal_scatter"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
            "src": np.random.randn(4).astype(np.float32),
        }

        # index put/copy
        self._generators["torch.index_add"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
            "dim": 0,
            "index": np.array([0, 2, 4]).astype(np.int64),
            "source": np.random.randn(3, 8).astype(np.float32),
        }
        self._generators["torch.index_copy"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
            "dim": 0,
            "index": np.array([0, 2, 4]).astype(np.int64),
            "source": np.random.randn(3, 8).astype(np.float32),
        }
        self._generators["torch.index_fill"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 0,
            "index": np.array([0, 2]).astype(np.int64),
            "value": 0.0,
        }

        # masked operations
        self._generators["torch.masked_fill"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "mask": np.random.rand(4, 8) > 0.5,
            "value": 0.0,
        }
        self._generators["torch.masked_scatter"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "mask": np.random.rand(4, 8) > 0.5,
            "source": np.random.randn(100).astype(np.float32),
        }

        # take / put
        self._generators["torch.take"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "index": np.array([0, 5, 10, 15]).astype(np.int64),
        }
        self._generators["torch.take_along_dim"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "indices": np.random.randint(0, 8, (4, 3)).astype(np.int64),
            "dim": 1,
        }

        # ravel / flatten_to_1d
        self._generators["torch.ravel"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # matrix diagonal ops
        self._generators["torch.diag_embed"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
        }
        self._generators["torch.diagflat"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
        }

        # block_diag
        self._generators["torch.block_diag"] = lambda: {
            "_tensor_list": [
                np.random.randn(2, 2).astype(np.float32),
                np.random.randn(3, 3).astype(np.float32),
            ],
        }

        # cartesian_prod
        self._generators["torch.cartesian_prod"] = lambda: {
            "_tensor_list": [
                np.array([1, 2, 3]).astype(np.float32),
                np.array([4, 5]).astype(np.float32),
            ],
        }

        # combinations / permutations
        self._generators["torch.combinations"] = lambda: {
            "input": np.array([1, 2, 3, 4]).astype(np.float32),
            "r": 2,
        }

        # kron
        self._generators["torch.kron"] = lambda: {
            "input": np.random.randn(2, 2).astype(np.float32),
            "other": np.random.randn(3, 3).astype(np.float32),
        }

        # tensordot
        self._generators["torch.tensordot"] = lambda: {
            "a": np.random.randn(4, 8).astype(np.float32),
            "b": np.random.randn(8, 6).astype(np.float32),
            "dims": 1,
        }

        # unfold - uses int args, not tensor
        # left out as it requires non-standard invocation

        # =====================================================================
        # Additional torch.* functions (batch 2)
        # =====================================================================

        # Pooling functions
        self._generators["torch.adaptive_avg_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "output_size": 4,
        }
        self._generators["torch.adaptive_max_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "output_size": 4,
        }
        self._generators["torch.avg_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.max_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.max_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.max_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "kernel_size": 2,
        }

        # Convolution functions
        # Note: conv1d, conv3d, conv_transpose3d, convolution are in numerical_exclusions.yaml
        # (MLX limitations) so they will be skipped. Keep proper generators for the rest.
        self._generators["torch.conv1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3).astype(np.float32),
        }
        self._generators["torch.conv2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3).astype(np.float32),
        }
        self._generators["torch.conv3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3, 3).astype(np.float32),
        }
        self._generators["torch.conv_transpose1d"] = lambda: {
            "input": np.random.randn(2, 8, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3).astype(np.float32),
        }
        self._generators["torch.conv_transpose2d"] = lambda: {
            "input": np.random.randn(2, 8, 16, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3).astype(np.float32),
        }
        self._generators["torch.conv_transpose3d"] = lambda: {
            "input": np.random.randn(2, 8, 8, 8, 8).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3, 3).astype(np.float32),
        }
        # torch.convolution is in numerical_exclusions.yaml (low-level interface)
        self._generators["torch.convolution"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3).astype(np.float32),
        }

        # Normalization functions
        self._generators["torch.batch_norm"] = lambda: {
            "input": np.random.randn(4, 8, 6, 6).astype(np.float32),
            "running_mean": np.zeros(8).astype(np.float32),
            "running_var": np.ones(8).astype(np.float32),
            "weight": np.ones(8).astype(np.float32),
            "bias": np.zeros(8).astype(np.float32),
            "training": False,
            "momentum": 0.1,
            "eps": 1e-5,
            "cudnn_enabled": False,  # Required by PyTorch
        }
        self._generators["torch.group_norm"] = lambda: {
            "input": np.random.randn(4, 8, 6, 6).astype(np.float32),
            "num_groups": 4,
        }
        self._generators["torch.layer_norm"] = lambda: {
            "input": np.random.randn(4, 8, 16).astype(np.float32),
            "normalized_shape": [16],
        }
        self._generators["torch.instance_norm"] = lambda: {
            "input": np.random.randn(4, 8, 6, 6).astype(np.float32),
            "running_mean": np.zeros(8).astype(np.float32),
            "running_var": np.ones(8).astype(np.float32),
            "weight": np.ones(8).astype(np.float32),
            "bias": np.zeros(8).astype(np.float32),
            "use_input_stats": True,
            "momentum": 0.1,
            "eps": 1e-5,
            "cudnn_enabled": False,  # Required by PyTorch
        }

        # Clamp variants
        self._generators["torch.clamp_max"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "max": 1.0,
        }
        self._generators["torch.clamp_min"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "min": -1.0,
        }
        self._generators["torch.clamp_max_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "max": 1.0,
        }
        self._generators["torch.clamp_min_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "min": -1.0,
        }
        self._generators["torch.clamp_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "min": -1.0,
            "max": 1.0,
        }

        # Clone / copy
        self._generators["torch.clone"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # Window functions
        self._generators["torch.bartlett_window"] = lambda: {"window_length": 16}
        self._generators["torch.blackman_window"] = lambda: {"window_length": 16}
        self._generators["torch.hamming_window"] = lambda: {"window_length": 16}
        self._generators["torch.hann_window"] = lambda: {"window_length": 16}
        # kaiser_window - PyTorch requires positional args (window_length, periodic, beta)
        self._generators["torch.kaiser_window"] = lambda: {
            "_positional": [16],  # window_length
            "periodic": True,
            "beta": 12.0,
        }

        # Bitwise shift operations
        self._generators["torch.bitwise_left_shift"] = lambda: {
            "input": np.random.randint(1, 100, (4, 8)).astype(np.int32),
            "other": np.random.randint(0, 4, (4, 8)).astype(np.int32),
        }
        self._generators["torch.bitwise_right_shift"] = lambda: {
            "input": np.random.randint(1, 1000, (4, 8)).astype(np.int32),
            "other": np.random.randint(0, 4, (4, 8)).astype(np.int32),
        }

        # Chain matmul
        self._generators["torch.chain_matmul"] = lambda: {
            "_tensor_list": [
                np.random.randn(4, 6).astype(np.float32),
                np.random.randn(6, 8).astype(np.float32),
                np.random.randn(8, 3).astype(np.float32),
            ],
        }
        self._generators["torch.linalg.multi_dot"] = lambda: {
            "tensors": [
                np.random.randn(4, 6).astype(np.float32),
                np.random.randn(6, 8).astype(np.float32),
                np.random.randn(8, 3).astype(np.float32),
            ],
        }

        # channel_shuffle
        self._generators["torch.channel_shuffle"] = lambda: {
            "input": np.random.randn(2, 8, 4, 4).astype(np.float32),
            "groups": 2,
        }

        # Cholesky and matrix decompositions
        self._generators["torch.cholesky"] = lambda: {
            "input": _make_positive_definite(4),
        }
        self._generators["torch.cholesky_inverse"] = lambda: {
            "input": np.linalg.cholesky(_make_positive_definite(4)).astype(np.float32),
        }
        # cholesky_solve - use positional args (PyTorch uses B/L, mlx_compat uses b/u)
        self._generators["torch.cholesky_solve"] = lambda: {
            "_positional": [
                np.random.randn(4, 2).astype(np.float32),
                np.linalg.cholesky(_make_positive_definite(4)).astype(np.float32),
            ],
        }

        # Constant padding
        self._generators["torch.constant_pad_nd"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "pad": (1, 1, 1, 1),
            "value": 0.0,
        }

        # Bilinear
        self._generators["torch.bilinear"] = lambda: {
            "input1": np.random.randn(4, 8).astype(np.float32),
            "input2": np.random.randn(4, 6).astype(np.float32),
            "weight": np.random.randn(10, 8, 6).astype(np.float32),
        }

        # Binary cross entropy with logits - torch.* version (uses int: 0=none, 1=mean, 2=sum)
        self._generators["torch.binary_cross_entropy_with_logits"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": (np.random.rand(4, 8) > 0.5).astype(np.float32),
            "reduction": 0,  # 0=none for torch.* C++ bindings
        }

        # Celu in-place
        self._generators["torch.celu_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # Complex tensor creation
        self._generators["torch.complex"] = lambda: {
            "real": np.random.randn(4, 8).astype(np.float32),
            "imag": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.polar"] = lambda: {
            "abs": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "angle": np.random.randn(4, 8).astype(np.float32),
        }

        # conj_physical in-place
        self._generators["torch.conj_physical_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # copysign
        self._generators["torch.copysign"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # corrcoef - already added
        # cov - already added

        # cross (with dim specified to avoid deprecation warning)
        self._generators["torch.cross"] = lambda: {
            "input": np.random.randn(4, 3).astype(np.float32),
            "other": np.random.randn(4, 3).astype(np.float32),
            "dim": 1,
        }

        # det
        self._generators["torch.det"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.logdet"] = lambda: {
            "input": _make_positive_definite(4),
        }
        self._generators["torch.slogdet"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # dist
        self._generators["torch.dist"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # divide (alias for div)
        self._generators["torch.divide"] = binary_positive_input

        # dot
        self._generators["torch.dot"] = lambda: {
            "input": np.random.randn(8).astype(np.float32),
            "other": np.random.randn(8).astype(np.float32),
        }

        # dsplit already added

        # dstack already added

        # einsum equation and operands
        self._generators["torch.einsum"] = lambda: {
            "equation": "ij,jk->ik",
            "_operands": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(8, 6).astype(np.float32),
            ],
        }

        # eq_, ne_, lt_, le_, gt_, ge_ (in-place comparison)
        for op in ["eq_", "ne_", "lt_", "le_", "gt_", "ge_"]:
            self._generators[f"torch.{op}"] = binary_tensor_input

        # equal (returns single bool)
        self._generators["torch.equal"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # exp_, log_, sqrt_ etc (in-place)
        for op in ["exp_", "exp2_", "expm1_", "log_", "log2_", "log10_", "log1p_",
                   "sqrt_", "rsqrt_", "reciprocal_", "neg_", "abs_", "sign_",
                   "ceil_", "floor_", "round_", "trunc_", "frac_"]:
            if "log" in op or "sqrt" in op or "rsqrt" in op or "reciprocal" in op:
                self._generators[f"torch.{op}"] = unary_positive_input
            else:
                self._generators[f"torch.{op}"] = unary_tensor_input

        # sin_, cos_, tan_, etc
        for op in ["sin_", "cos_", "tan_", "sinh_", "cosh_", "tanh_",
                   "asin_", "acos_", "atan_", "asinh_", "atanh_"]:
            if op in ["asin_", "acos_", "atanh_"]:
                self._generators[f"torch.{op}"] = lambda: unary_bounded_input()
            else:
                self._generators[f"torch.{op}"] = unary_tensor_input

        self._generators["torch.acosh_"] = unary_ge_one_input

        # eye
        self._generators["torch.eye"] = lambda: {"n": 4, "m": 6}

        # fake_quantize functions
        self._generators["torch.fake_quantize_per_channel_affine"] = lambda: {
            "input": np.random.randn(2, 4, 8, 8).astype(np.float32),
            "scale": np.ones(4).astype(np.float32),
            "zero_point": np.zeros(4).astype(np.int32),
            "axis": 1,
            "quant_min": 0,
            "quant_max": 255,
        }
        self._generators["torch.fake_quantize_per_tensor_affine"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "scale": 1.0,
            "zero_point": 0,
            "quant_min": 0,
            "quant_max": 255,
        }

        # fill_
        self._generators["torch.fill_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "value": 3.14,
        }

        # finfo / iinfo - type info (not tensor ops)

        # fix (alias for trunc)
        self._generators["torch.fix"] = unary_tensor_input
        self._generators["torch.fix_"] = unary_tensor_input

        # float_power
        self._generators["torch.float_power"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "exponent": np.random.uniform(0.5, 2.0, (4, 8)).astype(np.float32),
        }

        # floor_divide
        self._generators["torch.floor_divide"] = binary_positive_input

        # fmax, fmin (NaN-safe max/min)
        self._generators["torch.fmax"] = binary_tensor_input
        self._generators["torch.fmin"] = binary_tensor_input

        # frexp
        self._generators["torch.frexp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # full
        self._generators["torch.full"] = lambda: {"size": (4, 8), "fill_value": 3.14}

        # gcd, lcm
        self._generators["torch.gcd"] = lambda: {
            "input": np.random.randint(1, 100, (4, 8)).astype(np.int64),
            "other": np.random.randint(1, 100, (4, 8)).astype(np.int64),
        }
        self._generators["torch.lcm"] = lambda: {
            "input": np.random.randint(1, 20, (4, 8)).astype(np.int64),
            "other": np.random.randint(1, 20, (4, 8)).astype(np.int64),
        }

        # ger (outer product)
        self._generators["torch.ger"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
            "vec2": np.random.randn(8).astype(np.float32),
        }

        # heaviside
        self._generators["torch.heaviside"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "values": np.ones((4, 8)).astype(np.float32) * 0.5,
        }

        # hsplit, vsplit already added

        # hypot
        self._generators["torch.hypot"] = binary_tensor_input

        # i0 (Bessel function)
        self._generators["torch.i0"] = unary_tensor_input
        self._generators["torch.i0_"] = unary_tensor_input

        # igamma, igammac
        self._generators["torch.igamma"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.igammac"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }

        # index_reduce
        self._generators["torch.index_reduce"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
            "dim": 0,
            "index": np.array([0, 1, 0, 1, 2, 2, 3, 3]).astype(np.int64),
            "source": np.random.randn(8, 8).astype(np.float32),
            "reduce": "mean",
        }

        # inner
        self._generators["torch.inner"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # inverse (deprecated, use linalg.inv)
        self._generators["torch.inverse"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2,
        }

        # isclose
        self._generators["torch.isclose"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # isnan, isinf, isfinite, isneginf, isposinf, isreal
        for op in ["isnan", "isinf", "isfinite", "isneginf", "isposinf", "isreal"]:
            self._generators[f"torch.{op}"] = unary_tensor_input

        # istft (inverse STFT)
        # Complex and requires matching stft output - skip for now

        # kthvalue already added

        # ldexp
        self._generators["torch.ldexp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randint(-3, 4, (4, 8)).astype(np.int32),
        }

        # lerp already added

        # linspace already added

        # log_softmax - already registered via softmax_ops

        # logaddexp, logaddexp2 already added

        # logcumsumexp already added

        # logical_and, logical_or, logical_xor, logical_not already added

        # logit
        self._generators["torch.logit"] = lambda: unary_bounded_input(low=0.01, high=0.99)

        # logspace already added

        # logsumexp already added

        # lstsq (deprecated)
        self._generators["torch.lstsq"] = lambda: {
            "input": np.random.randn(6, 2).astype(np.float32),
            "A": np.random.randn(6, 4).astype(np.float32),
        }

        # lu, lu_solve, lu_unpack
        self._generators["torch.lu"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.lu_solve"] = lambda: {
            "input": np.random.randn(4, 2).astype(np.float32),
            "LU_data": np.random.randn(4, 4).astype(np.float32),
            "LU_pivots": np.array([0, 1, 2, 3], dtype=np.int32),
        }
        self._generators["torch.lu_unpack"] = lambda: {
            "LU_data": np.random.randn(4, 4).astype(np.float32),
            "LU_pivots": np.array([0, 1, 2, 3], dtype=np.int32),
        }

        # margin_ranking_loss (uses int: 0=none, 1=mean, 2=sum for torch.* C++ binding)
        self._generators["torch.margin_ranking_loss"] = lambda: {
            "input1": np.random.randn(4, 8).astype(np.float32),
            "input2": np.random.randn(4, 8).astype(np.float32),
            "target": np.random.choice([-1, 1], (4, 8)).astype(np.float32),
            "reduction": 0,  # 0=none
        }

        # masked_fill, masked_scatter already added

        # matrix_exp
        self._generators["torch.matrix_exp"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32) * 0.1,
        }

        # matrix_power
        self._generators["torch.matrix_power"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
            "n": 2,
        }

        # matrix_rank
        self._generators["torch.matrix_rank"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # max, min already added

        # maximum, minimum already added

        # mean already added

        # meshgrid already handled with _tensor_list

        # min, max already added

        # mm already added

        # mode already added

        # moveaxis, movedim already added

        # msort
        self._generators["torch.msort"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # mul_, div_, add_, sub_ (in-place arithmetic)
        for op in ["mul_", "div_", "add_", "sub_"]:
            self._generators[f"torch.{op}"] = binary_tensor_input

        # multinomial - random, skip

        # multiply (alias for mul)
        self._generators["torch.multiply"] = binary_tensor_input

        # mv
        self._generators["torch.mv"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "vec": np.random.randn(8).astype(np.float32),
        }

        # mvlgamma
        self._generators["torch.mvlgamma"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 2.0,
            "p": 2,
        }

        # nan_to_num
        self._generators["torch.nan_to_num"] = unary_tensor_input
        self._generators["torch.nan_to_num_"] = unary_tensor_input

        # nanmean, nansum already added

        # narrow, narrow_copy already added

        # native_layer_norm
        self._generators["torch.native_layer_norm"] = lambda: {
            "input": np.random.randn(4, 8, 16).astype(np.float32),
            "normalized_shape": [16],
            "weight": np.ones(16).astype(np.float32),
            "bias": np.zeros(16).astype(np.float32),
            "eps": 1e-5,
        }

        # native_batch_norm
        self._generators["torch.native_batch_norm"] = lambda: {
            "input": np.random.randn(4, 8, 16).astype(np.float32),
            "weight": np.ones(8).astype(np.float32),
            "bias": np.zeros(8).astype(np.float32),
            "running_mean": np.zeros(8).astype(np.float32),
            "running_var": np.ones(8).astype(np.float32),
            "training": False,
            "momentum": 0.1,
            "eps": 1e-5,
        }

        # negative (alias for neg)
        self._generators["torch.negative"] = unary_tensor_input
        self._generators["torch.negative_"] = unary_tensor_input

        # nextafter
        self._generators["torch.nextafter"] = binary_tensor_input

        # nonzero already added

        # normal - random, skip

        # not_equal (alias for ne)
        self._generators["torch.not_equal"] = binary_tensor_input

        # nuclear_norm
        self._generators["torch.nuclear_norm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # numel already added

        # ones, zeros, empty already added

        # orgqr
        self._generators["torch.orgqr"] = lambda: {
            "input": np.random.randn(4, 3).astype(np.float32),
            "input2": np.random.randn(3).astype(np.float32),
        }

        # ormqr
        self._generators["torch.ormqr"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
            "input2": np.random.randn(4).astype(np.float32),
            "input3": np.random.randn(4, 2).astype(np.float32),
        }

        # outer already added

        # pairwise_distance
        self._generators["torch.pairwise_distance"] = lambda: {
            "x1": np.random.randn(4, 8).astype(np.float32),
            "x2": np.random.randn(4, 8).astype(np.float32),
        }

        # pdist already added

        # permute already added

        # pinverse
        self._generators["torch.pinverse"] = lambda: {
            "input": np.random.randn(4, 6).astype(np.float32),
        }

        # pixel_shuffle, pixel_unshuffle
        self._generators["torch.pixel_shuffle"] = lambda: {
            "input": np.random.randn(2, 8, 4, 4).astype(np.float32),
            "upscale_factor": 2,
        }
        self._generators["torch.pixel_unshuffle"] = lambda: {
            "input": np.random.randn(2, 2, 8, 8).astype(np.float32),
            "downscale_factor": 2,
        }

        # poisson - random, skip

        # polygamma
        self._generators["torch.polygamma"] = lambda: {
            "n": 0,
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }

        # positive
        self._generators["torch.positive"] = unary_tensor_input

        # pow already added

        # pow_ (in-place)
        self._generators["torch.pow_"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "exponent": 2.0,
        }

        # prelu
        self._generators["torch.prelu"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "weight": np.random.randn(1).astype(np.float32),
        }

        # prod already added

        # promote_types - not tensor op

        # put
        self._generators["torch.put"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "index": np.array([0, 5, 10, 15]).astype(np.int64),
            "source": np.random.randn(4).astype(np.float32),
        }

        # qr
        self._generators["torch.qr"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # quantile, nanquantile already added

        # rad2deg, deg2rad
        self._generators["torch.rad2deg"] = unary_tensor_input
        self._generators["torch.deg2rad"] = unary_tensor_input

        # ravel already added

        # real
        self._generators["torch.real"] = unary_tensor_input

        # reciprocal already registered

        # remainder already registered

        # renorm
        self._generators["torch.renorm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "p": 2,
            "dim": 0,
            "maxnorm": 1.0,
        }

        # repeat_interleave already added

        # reshape already added

        # resolve_conj, resolve_neg
        self._generators["torch.resolve_conj"] = unary_tensor_input
        self._generators["torch.resolve_neg"] = unary_tensor_input

        # result_type - not tensor op

        # roll already added

        # rot90 already added

        # round already registered

        # row_stack already added

        # rrelu, rrelu_ - random

        # rsqrt already registered

        # scatter, scatter_add, scatter_reduce already added

        # searchsorted already added

        # select, select_scatter already added

        # sigmoid already registered

        # sign, signbit
        self._generators["torch.sign"] = unary_tensor_input
        self._generators["torch.signbit"] = unary_tensor_input

        # sin, cos, tan, etc already registered

        # sinc
        self._generators["torch.sinc"] = unary_tensor_input

        # slice_scatter already added

        # slogdet already added

        # smm (sparse mm) - skip sparse

        # softmax, log_softmax, softmin already registered

        # solve (deprecated)
        self._generators["torch.solve"] = lambda: {
            "input": np.random.randn(4, 2).astype(np.float32),
            "A": np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2,
        }

        # sort already added

        # split already added

        # sqrt already registered

        # square
        self._generators["torch.square"] = unary_tensor_input
        self._generators["torch.square_"] = unary_tensor_input

        # squeeze, unsqueeze already added

        # sspaddmm - sparse, skip

        # stack already added with _tensor_list

        # std, var already added

        # std_mean, var_mean already added

        # stft
        self._generators["torch.stft"] = lambda: {
            "input": np.random.randn(1024).astype(np.float32),
            "n_fft": 256,
            "return_complex": True,
        }

        # sub already registered

        # subtract (alias)
        self._generators["torch.subtract"] = binary_tensor_input

        # sum already added

        # svd
        self._generators["torch.svd"] = lambda: {
            "input": np.random.randn(4, 6).astype(np.float32),
        }

        # svd_lowrank
        self._generators["torch.svd_lowrank"] = lambda: {
            "A": np.random.randn(10, 8).astype(np.float32),
        }

        # swapaxes, swapdims already added

        # symeig (deprecated)
        self._generators["torch.symeig"] = lambda: {
            "input": _make_symmetric(4),
        }

        # t (transpose for 2D)
        self._generators["torch.t"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # take, take_along_dim already added

        # tan already registered

        # tanh already registered

        # tensor - can't test directly

        # tensordot already added

        # tile already added

        # topk already added

        # trace already added

        # transpose already added

        # trapezoid, trapz (integration)
        self._generators["torch.trapezoid"] = lambda: {
            "y": np.random.randn(10).astype(np.float32),
        }
        self._generators["torch.trapz"] = lambda: {
            "y": np.random.randn(10).astype(np.float32),
        }

        # triangular_solve
        self._generators["torch.triangular_solve"] = lambda: {
            "input": np.random.randn(4, 2).astype(np.float32),
            "A": np.triu(np.random.randn(4, 4).astype(np.float32)) + np.eye(4).astype(np.float32),
        }

        # tril, triu already added

        # tril_, triu_ (in-place)
        self._generators["torch.tril_"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.triu_"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # triplet_margin_loss (uses int: 0=none, 1=mean, 2=sum for torch.* C++ binding)
        self._generators["torch.triplet_margin_loss"] = lambda: {
            "anchor": np.random.randn(4, 8).astype(np.float32),
            "positive": np.random.randn(4, 8).astype(np.float32),
            "negative": np.random.randn(4, 8).astype(np.float32),
            "reduction": 0,  # 0=none
        }

        # true_divide already registered

        # trunc already registered

        # unbind
        self._generators["torch.unbind"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 0,
        }

        # unflatten
        self._generators["torch.unflatten"] = lambda: {
            "input": np.random.randn(2, 12, 5).astype(np.float32),
            "dim": 1,
            "sizes": (3, 4),
        }

        # unique, unique_consecutive already added

        # unsqueeze already added

        # vander
        self._generators["torch.vander"] = lambda: {
            "x": np.random.randn(5).astype(np.float32),
        }

        # var, std already registered

        # vdot already added

        # view_as_complex, view_as_real - complex specific

        # vstack already added

        # where already added

        # xlogy
        self._generators["torch.xlogy"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.xlogy_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }

        # zeros_like, ones_like, empty_like, full_like already added

        self._generators["torch.squeeze"] = lambda: {
            "input": np.random.randn(1, 4, 1, 8).astype(np.float32),
        }
        self._generators["torch.unsqueeze"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 0,
        }
        self._generators["torch.reshape"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "shape": (8, 4),
        }
        self._generators["torch.flatten"] = lambda: {
            "input": np.random.randn(2, 4, 8).astype(np.float32),
        }
        self._generators["torch.transpose"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim0": 0,
            "dim1": 1,
        }
        self._generators["torch.permute"] = lambda: {
            "input": np.random.randn(2, 4, 8).astype(np.float32),
            "dims": (2, 0, 1),
        }
        self._generators["torch.flip"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dims": [0],
        }
        self._generators["torch.roll"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "shifts": 2,
            "dims": 0,
        }
        self._generators["torch.rot90"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "k": 1,
            "dims": [0, 1],
        }

        # Indexing and slicing
        self._generators["torch.gather"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
            "index": np.random.randint(0, 8, (4, 3)).astype(np.int64),
        }
        self._generators["torch.scatter"] = lambda: {
            "input": np.zeros((4, 8)).astype(np.float32),
            "dim": 1,
            "index": np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]]).astype(np.int64),
            "src": np.random.randn(4, 3).astype(np.float32),
        }
        self._generators["torch.index_select"] = index_select_input
        self._generators["torch.masked_select"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "mask": np.random.rand(4, 8) > 0.5,
        }
        self._generators["torch.where"] = lambda: {
            "condition": np.random.rand(4, 8) > 0.5,
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nonzero"] = lambda: {
            "input": (np.random.rand(4, 8) > 0.5).astype(np.float32),
        }

        # Clamp and clip
        self._generators["torch.clamp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "min": -1.0,
            "max": 1.0,
        }
        self._generators["torch.clip"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "min": -1.0,
            "max": 1.0,
        }

        # Sorting and comparison
        self._generators["torch.sort"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.argsort"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.topk"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "k": 3,
        }
        self._generators["torch.kthvalue"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "k": 3,
        }

        # Cumulative operations
        self._generators["torch.cumsum"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }
        self._generators["torch.cumprod"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "dim": 1,
        }
        self._generators["torch.cummax"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }
        self._generators["torch.cummin"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }

        # Linear algebra
        self._generators["torch.dot"] = lambda: {
            "input": np.random.randn(8).astype(np.float32),
            "other": np.random.randn(8).astype(np.float32),
        }
        self._generators["torch.mv"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "vec": np.random.randn(8).astype(np.float32),
        }
        self._generators["torch.outer"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
            "vec2": np.random.randn(8).astype(np.float32),
        }
        self._generators["torch.inner"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.cross"] = lambda: {
            "input": np.random.randn(4, 3).astype(np.float32),
            "other": np.random.randn(4, 3).astype(np.float32),
        }
        self._generators["torch.trace"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.diag"] = lambda: {
            "input": np.random.randn(4).astype(np.float32),
        }
        self._generators["torch.diagonal"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.tril"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.triu"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # Additional math operations
        self._generators["torch.lerp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "end": np.random.randn(4, 8).astype(np.float32),
            "weight": 0.5,
        }
        self._generators["torch.addcdiv"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "tensor1": np.random.randn(4, 8).astype(np.float32),
            "tensor2": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.addcmul"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "tensor1": np.random.randn(4, 8).astype(np.float32),
            "tensor2": np.random.randn(4, 8).astype(np.float32),
        }

        # Aliases (arc* = a*)
        for alias, orig in [("arcsin", "asin"), ("arccos", "acos"), ("arctan", "atan"),
                            ("arcsinh", "asinh"), ("arccosh", "acosh"), ("arctanh", "atanh")]:
            if f"torch.{orig}" in self._generators:
                self._generators[f"torch.{alias}"] = self._generators[f"torch.{orig}"]

        self._generators["torch.arctan2"] = binary_tensor_input
        self._generators["torch.atan"] = unary_tensor_input
        self._generators["torch.asinh"] = unary_tensor_input

        # Trig functions not in unary list
        self._generators["torch.sinc"] = unary_tensor_input
        self._generators["torch.hypot"] = binary_tensor_input

        # allclose returns bool
        self._generators["torch.allclose"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # einsum - use _operands marker for tensor list
        self._generators["torch.einsum"] = lambda: {
            "equation": "ij,jk->ik",
            "_operands": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(8, 6).astype(np.float32),
            ],
        }

        # repeat/tile
        self._generators["torch.repeat_interleave"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "repeats": 2,
            "dim": 0,
        }
        self._generators["torch.tile"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dims": (2, 1),
        }

        # Unique operations
        self._generators["torch.unique"] = lambda: {
            "input": np.array([1, 2, 2, 3, 3, 3, 4]).astype(np.float32),
        }

        # Expand/broadcast - use positional args (PyTorch uses "size", mlx_compat uses "shape")
        self._generators["torch.broadcast_to"] = lambda: {
            "_positional": [
                np.random.randn(1, 8).astype(np.float32),
                (4, 8),
            ],
        }

        # Mesh grid - use _tensor_list marker
        self._generators["torch.meshgrid"] = lambda: {
            "_tensor_list": [
                np.arange(4).astype(np.float32),
                np.arange(5).astype(np.float32),
            ],
            "indexing": "ij",
        }

        # =====================================================================
        # torch.nn.functional operations
        # =====================================================================

        # Activation functions
        for fn in ["relu", "relu6", "elu", "selu", "celu", "leaky_relu", "prelu",
                   "rrelu", "gelu", "logsigmoid", "hardshrink", "tanhshrink",
                   "softsign", "softplus", "softmin", "softshrink", "gumbel_softmax",
                   "log_softmax", "softmax", "tanh", "sigmoid", "hardsigmoid",
                   "hardtanh", "hardswish", "silu", "mish", "threshold"]:
            self._generators[f"torch.nn.functional.{fn}"] = unary_tensor_input

        # threshold needs extra args
        self._generators["torch.nn.functional.threshold"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "threshold": 0.5,
            "value": 0.0,
        }

        # Normalization
        self._generators["torch.nn.functional.normalize"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "p": 2.0,
            "dim": 1,
        }
        self._generators["torch.nn.functional.layer_norm"] = lambda: {
            "input": np.random.randn(4, 8, 16).astype(np.float32),
            "normalized_shape": [16],
        }
        self._generators["torch.nn.functional.group_norm"] = lambda: {
            "input": np.random.randn(4, 8, 6, 6).astype(np.float32),
            "num_groups": 4,
        }

        # Padding
        self._generators["torch.nn.functional.pad"] = lambda: {
            "input": np.random.randn(2, 3, 4, 4).astype(np.float32),
            "pad": (1, 1, 1, 1),
        }

        # Convolution
        # Note: conv1d is excluded via the torch.nn.Conv1d exclusion, but provide proper generator
        self._generators["torch.nn.functional.conv1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3).astype(np.float32),
        }
        self._generators["torch.nn.functional.conv2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3).astype(np.float32),
        }

        # Pooling
        self._generators["torch.nn.functional.max_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.max_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.avg_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.avg_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.adaptive_avg_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "output_size": 4,
        }
        self._generators["torch.nn.functional.adaptive_avg_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "output_size": (4, 4),
        }

        # Linear
        self._generators["torch.nn.functional.linear"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "weight": np.random.randn(16, 8).astype(np.float32),
        }
        self._generators["torch.nn.functional.bilinear"] = lambda: {
            "input1": np.random.randn(4, 8).astype(np.float32),
            "input2": np.random.randn(4, 6).astype(np.float32),
            "weight": np.random.randn(10, 8, 6).astype(np.float32),
        }

        # Loss functions
        self._generators["torch.nn.functional.mse_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nn.functional.l1_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nn.functional.smooth_l1_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nn.functional.huber_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nn.functional.cross_entropy"] = lambda: {
            "input": np.random.randn(4, 10).astype(np.float32),
            "target": np.random.randint(0, 10, (4,)).astype(np.int64),
        }
        self._generators["torch.nn.functional.nll_loss"] = lambda: {
            "input": np.random.randn(4, 10).astype(np.float32),
            "target": np.random.randint(0, 10, (4,)).astype(np.int64),
        }
        self._generators["torch.nn.functional.binary_cross_entropy"] = lambda: {
            "input": 1 / (1 + np.exp(-np.random.randn(4, 8).astype(np.float32))),  # sigmoid
            "target": (np.random.rand(4, 8) > 0.5).astype(np.float32),
        }
        self._generators["torch.nn.functional.binary_cross_entropy_with_logits"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": (np.random.rand(4, 8) > 0.5).astype(np.float32),
        }

        # Embedding
        self._generators["torch.nn.functional.embedding"] = lambda: {
            "input": np.random.randint(0, 100, (4, 8)).astype(np.int64),
            "weight": np.random.randn(100, 32).astype(np.float32),
        }

        # One-hot
        self._generators["torch.nn.functional.one_hot"] = lambda: {
            "tensor": np.random.randint(0, 10, (4, 8)).astype(np.int64),
            "num_classes": 10,
        }

        # Interpolate / upsample
        self._generators["torch.nn.functional.interpolate"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "scale_factor": 2.0,
            "mode": "nearest",
        }

        # =====================================================================
        # Additional torch.nn.functional operations
        # =====================================================================

        # 3D pooling
        self._generators["torch.nn.functional.max_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.avg_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.adaptive_avg_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "output_size": (4, 4, 4),
        }
        self._generators["torch.nn.functional.adaptive_max_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "output_size": 4,
        }
        self._generators["torch.nn.functional.adaptive_max_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "output_size": (4, 4),
        }
        self._generators["torch.nn.functional.adaptive_max_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "output_size": (4, 4, 4),
        }
        # with_indices versions
        self._generators["torch.nn.functional.adaptive_max_pool1d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "output_size": 4,
        }
        self._generators["torch.nn.functional.adaptive_max_pool2d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "output_size": (4, 4),
        }
        self._generators["torch.nn.functional.adaptive_max_pool3d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "output_size": (4, 4, 4),
        }
        self._generators["torch.nn.functional.max_pool1d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.max_pool2d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.max_pool3d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "kernel_size": 2,
        }

        # Fractional max pooling
        self._generators["torch.nn.functional.fractional_max_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "kernel_size": 2,
            "output_size": (4, 4),
        }
        self._generators["torch.nn.functional.fractional_max_pool2d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "kernel_size": 2,
            "output_size": (4, 4),
        }
        self._generators["torch.nn.functional.fractional_max_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "kernel_size": 2,
            "output_size": (2, 2, 2),
        }
        self._generators["torch.nn.functional.fractional_max_pool3d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "kernel_size": 2,
            "output_size": (2, 2, 2),
        }

        # LP pooling - now excluded in numerical_exclusions.yaml, provide proper generators
        self._generators["torch.nn.functional.lp_pool1d"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "norm_type": 2.0,
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.lp_pool2d"] = lambda: {
            "input": np.random.randn(2, 3, 16, 16).astype(np.float32),
            "norm_type": 2.0,
            "kernel_size": 2,
        }
        self._generators["torch.nn.functional.lp_pool3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "norm_type": 2.0,
            "kernel_size": 2,
        }

        # 3D convolution - now excluded in numerical_exclusions.yaml, provide proper generators
        self._generators["torch.nn.functional.conv3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3, 3).astype(np.float32),
        }
        self._generators["torch.nn.functional.conv_transpose1d"] = lambda: {
            "input": np.random.randn(2, 8, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3).astype(np.float32),
        }
        self._generators["torch.nn.functional.conv_transpose2d"] = lambda: {
            "input": np.random.randn(2, 8, 16, 16).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3).astype(np.float32),
        }
        self._generators["torch.nn.functional.conv_transpose3d"] = lambda: {
            "input": np.random.randn(2, 8, 8, 8, 8).astype(np.float32),
            "weight": np.random.randn(8, 3, 3, 3, 3).astype(np.float32),
        }

        # Batch normalization
        self._generators["torch.nn.functional.batch_norm"] = lambda: {
            "input": np.random.randn(4, 8, 16).astype(np.float32),
            "running_mean": np.zeros(8).astype(np.float32),
            "running_var": np.ones(8).astype(np.float32),
            "training": False,
        }
        self._generators["torch.nn.functional.instance_norm"] = lambda: {
            "input": np.random.randn(4, 8, 16).astype(np.float32),
        }
        self._generators["torch.nn.functional.local_response_norm"] = lambda: {
            "input": np.random.randn(2, 8, 16, 16).astype(np.float32),
            "size": 5,
        }

        # Activations - in-place variants
        self._generators["torch.nn.functional.relu_"] = unary_tensor_input
        self._generators["torch.nn.functional.elu_"] = unary_tensor_input
        self._generators["torch.nn.functional.celu_"] = unary_tensor_input
        self._generators["torch.nn.functional.leaky_relu_"] = unary_tensor_input
        self._generators["torch.nn.functional.selu_"] = unary_tensor_input
        self._generators["torch.nn.functional.hardtanh_"] = unary_tensor_input
        self._generators["torch.nn.functional.rrelu_"] = unary_tensor_input

        # GLU
        self._generators["torch.nn.functional.glu"] = lambda: {
            "input": np.random.randn(4, 16).astype(np.float32),  # Must be even in last dim
            "dim": -1,
        }

        # Pixel shuffle/unshuffle
        self._generators["torch.nn.functional.pixel_shuffle"] = lambda: {
            "input": np.random.randn(2, 8, 4, 4).astype(np.float32),
            "upscale_factor": 2,
        }
        self._generators["torch.nn.functional.pixel_unshuffle"] = lambda: {
            "input": np.random.randn(2, 2, 8, 8).astype(np.float32),
            "downscale_factor": 2,
        }
        self._generators["torch.nn.functional.channel_shuffle"] = lambda: {
            "input": np.random.randn(2, 8, 4, 4).astype(np.float32),
            "groups": 2,
        }

        # Upsample variants
        self._generators["torch.nn.functional.upsample"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "scale_factor": 2,
            "mode": "nearest",
        }
        self._generators["torch.nn.functional.upsample_nearest"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "scale_factor": 2,
        }
        self._generators["torch.nn.functional.upsample_bilinear"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "scale_factor": 2,
        }

        # Affine grid / grid sample - now excluded in numerical_exclusions.yaml
        self._generators["torch.nn.functional.affine_grid"] = lambda: {
            "theta": np.random.randn(2, 2, 3).astype(np.float32),
            "size": [2, 3, 8, 8],
        }
        self._generators["torch.nn.functional.grid_sample"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "grid": np.random.randn(2, 4, 4, 2).astype(np.float32),
        }

        # Fold / unfold
        # fold input shape: (N, C*kH*kW, L) where L = output_blocks
        # For output_size=(8,8), kernel_size=(2,2), stride=(1,1): L = 7*7 = 49
        self._generators["torch.nn.functional.fold"] = lambda: {
            "input": np.random.randn(2, 12, 49).astype(np.float32),  # 12 = 3*2*2
            "output_size": (8, 8),
            "kernel_size": (2, 2),
        }
        self._generators["torch.nn.functional.unfold"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "kernel_size": (2, 2),
        }

        # Additional loss functions
        self._generators["torch.nn.functional.kl_div"] = lambda: {
            "input": np.log(np.abs(np.random.randn(4, 10).astype(np.float32)) + 0.01),
            "target": np.abs(np.random.randn(4, 10).astype(np.float32)),
        }
        self._generators["torch.nn.functional.poisson_nll_loss"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "target": np.abs(np.random.randn(4, 8).astype(np.float32)).astype(np.int64),
        }
        self._generators["torch.nn.functional.gaussian_nll_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.random.randn(4, 8).astype(np.float32),
            "var": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.nn.functional.hinge_embedding_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.where(np.random.rand(4, 8) > 0.5, 1.0, -1.0).astype(np.float32),
        }
        self._generators["torch.nn.functional.margin_ranking_loss"] = lambda: {
            "input1": np.random.randn(4).astype(np.float32),
            "input2": np.random.randn(4).astype(np.float32),
            "target": np.array([1, -1, 1, -1]).astype(np.float32),
        }
        self._generators["torch.nn.functional.cosine_embedding_loss"] = lambda: {
            "input1": np.random.randn(4, 32).astype(np.float32),
            "input2": np.random.randn(4, 32).astype(np.float32),
            "target": np.array([1, -1, 1, -1]).astype(np.float32),
        }
        self._generators["torch.nn.functional.triplet_margin_loss"] = lambda: {
            "anchor": np.random.randn(4, 32).astype(np.float32),
            "positive": np.random.randn(4, 32).astype(np.float32),
            "negative": np.random.randn(4, 32).astype(np.float32),
        }
        self._generators["torch.nn.functional.triplet_margin_with_distance_loss"] = lambda: {
            "anchor": np.random.randn(4, 32).astype(np.float32),
            "positive": np.random.randn(4, 32).astype(np.float32),
            "negative": np.random.randn(4, 32).astype(np.float32),
        }
        self._generators["torch.nn.functional.multi_margin_loss"] = lambda: {
            "input": np.random.randn(4, 10).astype(np.float32),
            "target": np.random.randint(0, 10, (4,)).astype(np.int64),
        }
        self._generators["torch.nn.functional.multilabel_margin_loss"] = lambda: {
            "input": np.random.randn(4, 10).astype(np.float32),
            "target": np.random.randint(-1, 10, (4, 10)).astype(np.int64),
        }
        self._generators["torch.nn.functional.multilabel_soft_margin_loss"] = lambda: {
            "input": np.random.randn(4, 10).astype(np.float32),
            "target": (np.random.rand(4, 10) > 0.5).astype(np.float32),
        }
        self._generators["torch.nn.functional.soft_margin_loss"] = lambda: {
            "input": np.random.randn(4, 10).astype(np.float32),
            "target": np.where(np.random.rand(4, 10) > 0.5, 1.0, -1.0).astype(np.float32),
        }
        self._generators["torch.nn.functional.ctc_loss"] = lambda: {
            "log_probs": np.random.randn(50, 4, 20).astype(np.float32),  # (T, N, C)
            "targets": np.random.randint(1, 20, (4, 30)).astype(np.int64),  # (N, S)
            "input_lengths": np.array([50, 50, 50, 50], dtype=np.int64),
            "target_lengths": np.array([30, 25, 20, 28], dtype=np.int64),
        }

        # Cosine similarity
        self._generators["torch.nn.functional.cosine_similarity"] = lambda: {
            "x1": np.random.randn(4, 8).astype(np.float32),
            "x2": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }
        self._generators["torch.nn.functional.pairwise_distance"] = lambda: {
            "x1": np.random.randn(4, 8).astype(np.float32),
            "x2": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.nn.functional.pdist"] = lambda: {
            "input": np.random.randn(5, 8).astype(np.float32),
        }

        # Embedding bag
        self._generators["torch.nn.functional.embedding_bag"] = lambda: {
            "input": np.random.randint(0, 100, (4, 8)).astype(np.int64),
            "weight": np.random.randn(100, 32).astype(np.float32),
        }

        # Multi-head attention
        self._generators["torch.nn.functional.multi_head_attention_forward"] = lambda: {
            "query": np.random.randn(8, 4, 32).astype(np.float32),
            "key": np.random.randn(10, 4, 32).astype(np.float32),
            "value": np.random.randn(10, 4, 32).astype(np.float32),
            "embed_dim_to_check": 32,
            "num_heads": 4,
            "in_proj_weight": np.random.randn(96, 32).astype(np.float32),
            "in_proj_bias": np.random.randn(96).astype(np.float32),
            "bias_k": None,
            "bias_v": None,
            "add_zero_attn": False,
            "dropout_p": 0.0,
            "out_proj_weight": np.random.randn(32, 32).astype(np.float32),
            "out_proj_bias": np.random.randn(32).astype(np.float32),
        }
        self._generators["torch.nn.functional.scaled_dot_product_attention"] = lambda: {
            "query": np.random.randn(4, 4, 8, 16).astype(np.float32),
            "key": np.random.randn(4, 4, 10, 16).astype(np.float32),
            "value": np.random.randn(4, 4, 10, 16).astype(np.float32),
        }

        # =====================================================================
        # torch.linalg operations
        # =====================================================================

        # Matrix operations
        self._generators["torch.linalg.matrix_norm"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.vector_norm"] = lambda: {
            "x": np.random.randn(8).astype(np.float32),
        }
        self._generators["torch.linalg.norm"] = lambda: {
            "A": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.linalg.det"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.slogdet"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.inv"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2,
        }
        self._generators["torch.linalg.pinv"] = lambda: {
            "A": np.random.randn(4, 6).astype(np.float32),
        }
        self._generators["torch.linalg.matrix_rank"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.matrix_power"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
            "n": 2,
        }
        self._generators["torch.linalg.multi_dot"] = lambda: {
            "tensors": [
                np.random.randn(4, 6).astype(np.float32),
                np.random.randn(6, 8).astype(np.float32),
                np.random.randn(8, 3).astype(np.float32),
            ],
        }
        self._generators["torch.linalg.cholesky"] = lambda: {
            "A": _make_positive_definite(4),
        }
        self._generators["torch.linalg.qr"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.lu"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.lu_factor"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.svd"] = lambda: {
            "A": np.random.randn(4, 6).astype(np.float32),
        }
        self._generators["torch.linalg.svdvals"] = lambda: {
            "A": np.random.randn(4, 6).astype(np.float32),
        }
        self._generators["torch.linalg.eig"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.eigh"] = lambda: {
            "A": _make_symmetric(4),
        }
        self._generators["torch.linalg.eigvals"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.eigvalsh"] = lambda: {
            "A": _make_symmetric(4),
        }
        self._generators["torch.linalg.solve"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2,
            "B": np.random.randn(4, 2).astype(np.float32),
        }
        self._generators["torch.linalg.lstsq"] = lambda: {
            "A": np.random.randn(6, 4).astype(np.float32),
            "B": np.random.randn(6, 2).astype(np.float32),
        }
        self._generators["torch.linalg.cond"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.cross"] = lambda: {
            "x": np.random.randn(4, 3).astype(np.float32),
            "y": np.random.randn(4, 3).astype(np.float32),
        }
        self._generators["torch.linalg.diagonal"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32),
        }
        self._generators["torch.linalg.tensorinv"] = lambda: {
            "A": np.random.randn(4, 6, 8, 3).astype(np.float32),
            "ind": 2,
        }
        self._generators["torch.linalg.tensorsolve"] = lambda: {
            "A": np.random.randn(2, 3, 6).astype(np.float32),
            "B": np.random.randn(6,).astype(np.float32),
        }
        self._generators["torch.linalg.matrix_exp"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32) * 0.1,  # Scale down for stability
        }
        self._generators["torch.linalg.householder_product"] = lambda: {
            "A": np.random.randn(4, 3).astype(np.float32),
            "tau": np.random.randn(3).astype(np.float32),
        }
        self._generators["torch.linalg.ldl_factor"] = lambda: {
            "A": _make_symmetric(4),
        }
        self._generators["torch.linalg.cholesky_ex"] = lambda: {
            "A": _make_positive_definite(4),
        }
        self._generators["torch.linalg.inv_ex"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2,
        }
        self._generators["torch.linalg.solve_ex"] = lambda: {
            "A": np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2,
            "B": np.random.randn(4, 2).astype(np.float32),
        }
        self._generators["torch.linalg.lu_solve"] = lambda: {
            "LU": np.random.randn(4, 4).astype(np.float32),
            "pivots": np.array([0, 1, 2, 3], dtype=np.int64),
            "B": np.random.randn(4, 2).astype(np.float32),
        }
        self._generators["torch.linalg.solve_triangular"] = lambda: {
            "A": np.triu(np.random.randn(4, 4).astype(np.float32)) + np.eye(4).astype(np.float32),
            "B": np.random.randn(4, 2).astype(np.float32),
            "upper": True,
        }
        self._generators["torch.linalg.vander"] = lambda: {
            "x": np.random.randn(5).astype(np.float32),
        }

        # =====================================================================
        # torch.fft operations
        # =====================================================================

        self._generators["torch.fft.fft"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
        }
        self._generators["torch.fft.ifft"] = lambda: {
            "input": np.random.randn(16).astype(np.float32) + 1j * np.random.randn(16).astype(np.float32),
        }
        self._generators["torch.fft.fft2"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.ifft2"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32) + 1j * np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.fftn"] = lambda: {
            "input": np.random.randn(4, 4, 4).astype(np.float32),
        }
        self._generators["torch.fft.ifftn"] = lambda: {
            "input": np.random.randn(4, 4, 4).astype(np.float32) + 1j * np.random.randn(4, 4, 4).astype(np.float32),
        }
        self._generators["torch.fft.rfft"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
        }
        self._generators["torch.fft.irfft"] = lambda: {
            "input": np.random.randn(9).astype(np.float32) + 1j * np.random.randn(9).astype(np.float32),
        }
        self._generators["torch.fft.rfft2"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.irfft2"] = lambda: {
            "input": np.random.randn(8, 5).astype(np.float32) + 1j * np.random.randn(8, 5).astype(np.float32),
        }
        self._generators["torch.fft.rfftn"] = lambda: {
            "input": np.random.randn(4, 4, 4).astype(np.float32),
        }
        self._generators["torch.fft.irfftn"] = lambda: {
            "input": np.random.randn(4, 4, 3).astype(np.float32) + 1j * np.random.randn(4, 4, 3).astype(np.float32),
        }
        self._generators["torch.fft.hfft"] = lambda: {
            "input": np.random.randn(16).astype(np.float32) + 1j * np.random.randn(16).astype(np.float32),
        }
        self._generators["torch.fft.ihfft"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
        }
        self._generators["torch.fft.hfft2"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32) + 1j * np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.ihfft2"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.hfftn"] = lambda: {
            "input": np.random.randn(4, 4, 4).astype(np.float32) + 1j * np.random.randn(4, 4, 4).astype(np.float32),
        }
        self._generators["torch.fft.ihfftn"] = lambda: {
            "input": np.random.randn(4, 4, 4).astype(np.float32),
        }
        self._generators["torch.fft.fftshift"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.ifftshift"] = lambda: {
            "input": np.random.randn(8, 8).astype(np.float32),
        }
        self._generators["torch.fft.fftfreq"] = lambda: {
            "n": 16,
        }
        self._generators["torch.fft.rfftfreq"] = lambda: {
            "n": 16,
        }

        # =====================================================================
        # torch.special operations
        # =====================================================================

        self._generators["torch.special.expit"] = unary_tensor_input
        self._generators["torch.special.logit"] = lambda: unary_bounded_input(low=0.01, high=0.99)
        self._generators["torch.special.erf"] = unary_tensor_input
        self._generators["torch.special.erfc"] = unary_tensor_input
        self._generators["torch.special.erfinv"] = lambda: unary_bounded_input()
        self._generators["torch.special.erfcx"] = unary_tensor_input
        self._generators["torch.special.ndtr"] = unary_tensor_input
        self._generators["torch.special.ndtri"] = lambda: unary_bounded_input(low=0.01, high=0.99)
        self._generators["torch.special.log_ndtr"] = unary_tensor_input
        self._generators["torch.special.expm1"] = unary_tensor_input
        self._generators["torch.special.exp2"] = unary_tensor_input
        self._generators["torch.special.log1p"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32))
        }
        self._generators["torch.special.sinc"] = unary_tensor_input
        self._generators["torch.special.round"] = unary_tensor_input
        self._generators["torch.special.softmax"] = softmax_input
        self._generators["torch.special.log_softmax"] = softmax_input
        self._generators["torch.special.entr"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32))
        }
        self._generators["torch.special.i0"] = unary_tensor_input
        self._generators["torch.special.i0e"] = unary_tensor_input
        self._generators["torch.special.i1"] = unary_tensor_input
        self._generators["torch.special.i1e"] = unary_tensor_input
        self._generators["torch.special.polygamma"] = lambda: {
            "n": 0,
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.digamma"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.gammaln"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.gammainc"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.gammaincc"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.multigammaln"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 2.0,
            "p": 2,
        }
        self._generators["torch.special.zeta"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 1.1,
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.bessel_j0"] = unary_tensor_input
        self._generators["torch.special.bessel_j1"] = unary_tensor_input
        self._generators["torch.special.bessel_y0"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.bessel_y1"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.modified_bessel_i0"] = unary_tensor_input
        self._generators["torch.special.modified_bessel_i1"] = unary_tensor_input
        self._generators["torch.special.modified_bessel_k0"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.modified_bessel_k1"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.scaled_modified_bessel_k0"] = lambda: {
            "x": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.scaled_modified_bessel_k1"] = lambda: {
            "x": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.spherical_bessel_j0"] = lambda: {
            "x": np.random.randn(4, 8).astype(np.float32),
        }

        # =====================================================================
        # torch.signal.windows operations
        # =====================================================================

        window_funcs = [
            "bartlett", "blackman", "cosine", "exponential", "gaussian",
            "general_cosine", "general_hamming", "hamming", "hann",
            "kaiser", "nuttall", "triang"
        ]
        for w in window_funcs:
            self._generators[f"torch.signal.windows.{w}"] = lambda: {"M": 16}

        # Special cases for window functions with extra params
        self._generators["torch.signal.windows.exponential"] = lambda: {
            "M": 16,
            "tau": 3.0,
        }
        self._generators["torch.signal.windows.gaussian"] = lambda: {
            "M": 16,
            "std": 1.0,
        }
        self._generators["torch.signal.windows.kaiser"] = lambda: {
            "M": 16,
            "beta": 12.0,
        }
        self._generators["torch.signal.windows.general_cosine"] = lambda: {
            "M": 16,
            "a": [0.5, 0.5],
        }
        self._generators["torch.signal.windows.general_hamming"] = lambda: {
            "M": 16,
            "alpha": 0.54,
        }

        # ======================================================================
        # Functions that need no inputs or special handling
        # ======================================================================

        # Functions with no arguments
        self._generators["torch.are_deterministic_algorithms_enabled"] = lambda: {}
        self._generators["torch.get_default_device"] = lambda: {}
        self._generators["torch.get_deterministic_debug_mode"] = lambda: {}
        self._generators["torch.get_num_threads"] = lambda: {}
        self._generators["torch.is_inference_mode_enabled"] = lambda: {}

        # Functions that set state (skip or test with safe values)
        self._generators["torch.set_deterministic_debug_mode"] = lambda: {"mode": 0}
        self._generators["torch.set_num_threads"] = lambda: {"num": 1}

        # unravel_index
        self._generators["torch.unravel_index"] = lambda: {
            "indices": np.array([0, 1, 5, 10], dtype=np.int64),
            "shape": (4, 4),
        }

        # Functions using positional args instead of 'input'
        self._generators["torch.atleast_1d"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
        }
        self._generators["torch.atleast_2d"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
        }
        self._generators["torch.atleast_3d"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
        }
        self._generators["torch.block_diag"] = lambda: {
            "_positional": [
                np.random.randn(2, 2).astype(np.float32),
                np.random.randn(3, 3).astype(np.float32),
            ],
        }
        self._generators["torch.cartesian_prod"] = lambda: {
            "_positional": [
                np.arange(3).astype(np.float32),
                np.arange(4).astype(np.float32),
            ],
        }
        self._generators["torch.chain_matmul"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(8, 6).astype(np.float32),
                np.random.randn(6, 4).astype(np.float32),
            ],
        }
        self._generators["torch.meshgrid"] = lambda: {
            "_positional": [
                np.arange(3).astype(np.float32),
                np.arange(4).astype(np.float32),
            ],
        }

        # chunk needs specific format
        self._generators["torch.chunk"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32)],
            "chunks": 3,
        }

        # split needs specific format
        self._generators["torch.split"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32)],
            "split_size_or_sections": 4,
        }
        self._generators["torch.split_with_sizes"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32)],
            "split_sizes": [3, 4, 5],
        }

        # dsplit, hsplit, vsplit - use positional args for sections
        self._generators["torch.dsplit"] = lambda: {
            "_positional": [np.random.randn(4, 4, 6).astype(np.float32), 2],
        }
        self._generators["torch.hsplit"] = lambda: {
            "_positional": [np.random.randn(4, 6).astype(np.float32), 2],
        }
        self._generators["torch.vsplit"] = lambda: {
            "_positional": [np.random.randn(6, 4).astype(np.float32), 2],
        }
        self._generators["torch.tensor_split"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32), 3],
        }

        # Functions with 'size' instead of 'input'
        self._generators["torch.empty"] = lambda: {
            "_positional": [(4, 8)],
        }
        self._generators["torch.ones"] = lambda: {
            "_positional": [(4, 8)],
        }
        self._generators["torch.zeros"] = lambda: {
            "_positional": [(4, 8)],
        }

        # einsum
        self._generators["torch.einsum"] = lambda: {
            "_positional": ["ij,jk->ik"],
            "_operands": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(8, 6).astype(np.float32),
            ],
        }

        # dot
        self._generators["torch.dot"] = lambda: {
            "_positional": [
                np.random.randn(8).astype(np.float32),
                np.random.randn(8).astype(np.float32),
            ],
        }

        # narrow
        self._generators["torch.narrow"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
            "dim": 1,
            "start": 2,
            "length": 4,
        }

        # permute - use positional args since mlx_compat doesn't accept dims=
        self._generators["torch.permute"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                (2, 0, 1),
            ],
        }

        # select
        self._generators["torch.select"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
            "dim": 0,
            "index": 2,
        }

        # swapdims/swapaxes - use positional args since mlx_compat doesn't accept dim0=/dim1=
        self._generators["torch.swapdims"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                0,
                2,
            ],
        }
        self._generators["torch.swapaxes"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                0,
                2,
            ],
        }

        # tile
        self._generators["torch.tile"] = lambda: {
            "_positional": [np.random.randn(2, 4).astype(np.float32)],
            "dims": (2, 3),
        }

        # unbind
        self._generators["torch.unbind"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
            "dim": 0,
        }

        # repeat_interleave
        self._generators["torch.repeat_interleave"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
            "repeats": 2,
        }

        # where (with 3 args)
        self._generators["torch.where"] = lambda: {
            "_positional": [
                (np.random.randn(4, 8) > 0).astype(bool),
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(4, 8).astype(np.float32),
            ],
        }

        # searchsorted
        self._generators["torch.searchsorted"] = lambda: {
            "_positional": [
                np.sort(np.random.randn(10).astype(np.float32)),
                np.random.randn(5).astype(np.float32),
            ],
        }

        # ======================================================================
        # torch.linalg functions - use positional args since builtin functions
        # don't always accept kwargs properly
        # ======================================================================

        self._generators["torch.linalg.det"] = lambda: {
            "_positional": [np.random.randn(4, 4).astype(np.float32)],
        }
        self._generators["torch.linalg.inv"] = lambda: {
            "_positional": [np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2],
        }
        self._generators["torch.linalg.qr"] = lambda: {
            "_positional": [np.random.randn(4, 4).astype(np.float32)],
        }
        self._generators["torch.linalg.svd"] = lambda: {
            "_positional": [np.random.randn(4, 6).astype(np.float32)],
        }
        self._generators["torch.linalg.svdvals"] = lambda: {
            "_positional": [np.random.randn(4, 6).astype(np.float32)],
        }
        self._generators["torch.linalg.slogdet"] = lambda: {
            "_positional": [np.random.randn(4, 4).astype(np.float32) + np.eye(4).astype(np.float32) * 2],
        }
        self._generators["torch.linalg.cholesky"] = lambda: {
            "_positional": [_make_positive_definite(4)],
        }
        self._generators["torch.linalg.eig"] = lambda: {
            "_positional": [np.random.randn(4, 4).astype(np.float32)],
        }
        self._generators["torch.linalg.eigh"] = lambda: {
            "_positional": [_make_symmetric(4)],
        }
        self._generators["torch.linalg.eigvalsh"] = lambda: {
            "_positional": [_make_symmetric(4)],
        }
        self._generators["torch.linalg.norm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.linalg.matrix_norm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        # vector_norm - let it fail to show the parameter name mismatch
        self._generators["torch.linalg.vector_norm"] = lambda: {
            "x": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.linalg.cross"] = lambda: {
            "input": np.random.randn(4, 3).astype(np.float32),
            "other": np.random.randn(4, 3).astype(np.float32),
        }
        # pinv - let it fail to show the matmul shape bug
        self._generators["torch.linalg.pinv"] = lambda: {
            "_positional": [np.random.randn(4, 6).astype(np.float32)],
        }
        self._generators["torch.linalg.matrix_rank"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # ======================================================================
        # Additional torch.special functions
        # ======================================================================

        self._generators["torch.special.logsumexp"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 1,
        }
        self._generators["torch.special.psi"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.xlog1py"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }
        self._generators["torch.special.xlogy"] = lambda: {
            "input": np.abs(np.random.randn(4, 8).astype(np.float32)),
            "other": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
        }

        # Chebyshev polynomials need x and n - use positional args
        for poly_name in [
            "chebyshev_polynomial_t", "chebyshev_polynomial_u",
            "chebyshev_polynomial_v", "chebyshev_polynomial_w",
            "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w",
        ]:
            self._generators[f"torch.special.{poly_name}"] = lambda: {
                "_positional": [
                    np.random.uniform(-1, 1, (4, 8)).astype(np.float32),
                    np.random.randint(0, 5, (4, 8)).astype(np.int64),
                ],
            }

        # Hermite polynomials - use positional args
        self._generators["torch.special.hermite_polynomial_h"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randint(0, 5, (4, 8)).astype(np.int64),
            ],
        }
        self._generators["torch.special.hermite_polynomial_he"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randint(0, 5, (4, 8)).astype(np.int64),
            ],
        }
        self._generators["torch.special.laguerre_polynomial_l"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randint(0, 5, (4, 8)).astype(np.int64),
            ],
        }
        self._generators["torch.special.legendre_polynomial_p"] = lambda: {
            "_positional": [
                np.random.uniform(-1, 1, (4, 8)).astype(np.float32),
                np.random.randint(0, 5, (4, 8)).astype(np.int64),
            ],
        }
        self._generators["torch.special.airy_ai"] = lambda: {
            "x": np.random.randn(4, 8).astype(np.float32),
        }

        # ======================================================================
        # More torch functions
        # ======================================================================

        # as_strided
        self._generators["torch.as_strided"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
            "size": (4, 4),
            "stride": (4, 1),
        }
        self._generators["torch.as_strided_"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
            "size": (4, 4),
            "stride": (4, 1),
        }

        # as_strided_scatter - let it fail to show the implementation difference
        self._generators["torch.as_strided_scatter"] = lambda: {
            "input": np.random.randn(16).astype(np.float32),
            "src": np.random.randn(4).astype(np.float32),
            "size": (4,),
            "stride": (4,),
        }

        # from_numpy
        self._generators["torch.from_numpy"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
        }

        # as_tensor / asarray - use positional since mlx_compat doesn't accept obj=
        self._generators["torch.as_tensor"] = lambda: {
            "data": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.asarray"] = lambda: {
            "_positional": [np.random.randn(4, 8).astype(np.float32)],
        }

        # tensor / scalar_tensor - use positional since mlx_compat doesn't accept s=
        self._generators["torch.tensor"] = lambda: {
            "data": np.random.randn(4, 8).astype(np.float32).tolist(),
        }
        self._generators["torch.scalar_tensor"] = lambda: {
            "_positional": [3.14],
        }

        # detach
        self._generators["torch.detach"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.detach_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # fill
        self._generators["torch.fill"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "value": 1.0,
        }

        # fliplr, flipud
        self._generators["torch.fliplr"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.flipud"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # frobenius_norm - requires dim argument
        self._generators["torch.frobenius_norm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "dim": 0,
        }

        # greater/less
        self._generators["torch.greater"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.greater_equal"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.less"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.less_equal"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # unique_consecutive
        self._generators["torch.unique_consecutive"] = lambda: {
            "input": np.array([1, 1, 2, 2, 3, 1, 1]).astype(np.float32),
        }

        # rsub
        self._generators["torch.rsub"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }

        # sgn
        self._generators["torch.sgn"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # lobpcg (eigenvalue computation)
        # Include explicit X to ensure deterministic behavior across frameworks
        # (otherwise each call generates its own random X from current torch state)
        def _lobpcg_inputs():
            A = _make_positive_definite(8).astype(np.float32)
            k = 2
            n = A.shape[0]
            X = np.random.randn(n, k).astype(np.float32)
            return {"A": A, "k": k, "X": X}
        self._generators["torch.lobpcg"] = _lobpcg_inputs

        # view_as_real/complex
        self._generators["torch.view_as_real"] = lambda: {
            "input": (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64),
        }
        self._generators["torch.view_as_complex"] = lambda: {
            "input": np.random.randn(4, 8, 2).astype(np.float32),
        }

        # tril_indices, triu_indices
        self._generators["torch.tril_indices"] = lambda: {
            "row": 4,
            "col": 4,
        }
        self._generators["torch.triu_indices"] = lambda: {
            "row": 4,
            "col": 4,
        }

        # range (deprecated but still in API)
        self._generators["torch.range"] = lambda: {
            "start": 0,
            "end": 10,
        }

        # is_* predicates
        self._generators["torch.is_conj"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.is_neg"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.is_same_size"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "other": np.random.randn(4, 8).astype(np.float32),
        }
        self._generators["torch.is_signed"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
        }

        # isin
        self._generators["torch.isin"] = lambda: {
            "elements": np.random.randint(0, 10, (4, 8)).astype(np.int64),
            "test_elements": np.arange(5).astype(np.int64),
        }

        # promote_types / result_type / can_cast
        self._generators["torch.promote_types"] = lambda: {
            "_dtype_pair": True,
        }
        self._generators["torch.result_type"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                np.random.randn(4, 8).astype(np.float32),
            ],
        }
        self._generators["torch.can_cast"] = lambda: {
            "_dtype_pair": True,
        }

        # histogram functions
        self._generators["torch.histogramdd"] = lambda: {
            "input": np.random.randn(100, 3).astype(np.float32),
            "bins": [10, 10, 10],
        }

        # unsafe_* variants
        self._generators["torch.unsafe_chunk"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32)],
            "chunks": 3,
        }
        self._generators["torch.unsafe_split"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32)],
            "split_size": 4,
        }
        self._generators["torch.unsafe_split_with_sizes"] = lambda: {
            "_positional": [np.random.randn(12, 8).astype(np.float32)],
            "split_sizes": [3, 4, 5],
        }

        # ======================================================================
        # torch.nn.functional additions
        # ======================================================================

        # one_hot
        self._generators["torch.nn.functional.one_hot"] = lambda: {
            "_positional": [np.random.randint(0, 5, (4,)).astype(np.int64)],
            "num_classes": 5,
        }

        # prelu
        self._generators["torch.nn.functional.prelu"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "weight": np.random.randn(8).astype(np.float32) * 0.25,
        }

        # gumbel_softmax
        self._generators["torch.nn.functional.gumbel_softmax"] = lambda: {
            "logits": np.random.randn(4, 8).astype(np.float32),
            "tau": 1.0,
        }

        # bilinear - let it fail to show the broadcast shape bug
        self._generators["torch.nn.functional.bilinear"] = lambda: {
            "input1": np.random.randn(4, 8).astype(np.float32),
            "input2": np.random.randn(4, 6).astype(np.float32),
            "weight": np.random.randn(10, 8, 6).astype(np.float32),
        }

        # threshold / threshold_
        self._generators["torch.threshold"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "threshold": 0.5,
            "value": 0.0,
        }
        # threshold_ - let it fail to show the TypeError bug
        self._generators["torch.threshold_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "threshold": 0.5,
            "value": 0.0,
        }
        self._generators["torch.nn.functional.threshold"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "threshold": 0.5,
            "value": 0.0,
        }
        # threshold_ - let it fail to show the TypeError bug
        self._generators["torch.nn.functional.threshold_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "threshold": 0.5,
            "value": 0.0,
        }

        # hardshrink
        self._generators["torch.hardshrink"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "lambd": 0.5,
        }
        self._generators["torch.nn.functional.hardshrink"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "lambd": 0.5,
        }

        # RNN functions are defined later in the file with proper generators

        # dropout variants - use positional args (PyTorch uses 'train', mlx_compat uses 'training')
        self._generators["torch.dropout"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                0.0,  # Use 0 for deterministic testing
                False,
            ],
        }
        self._generators["torch.dropout_"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.alpha_dropout"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.alpha_dropout_"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.feature_dropout"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.feature_dropout_"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.feature_alpha_dropout"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.feature_alpha_dropout_"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8).astype(np.float32),
                0.0,
                False,
            ],
        }
        self._generators["torch.nn.functional.alpha_dropout"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "p": 0.0,
            "training": False,
        }
        self._generators["torch.nn.functional.feature_alpha_dropout"] = lambda: {
            "input": np.random.randn(2, 4, 8).astype(np.float32),
            "p": 0.0,
            "training": False,
        }

        # embedding functions - let them fail to show the signature mismatch
        # torch.embedding(weight, indices) vs F.embedding(input, weight)
        self._generators["torch.embedding"] = lambda: {
            "weight": np.random.randn(100, 16).astype(np.float32),
            "indices": np.random.randint(0, 100, (4, 8)).astype(np.int64),
        }
        self._generators["torch.nn.functional.embedding"] = lambda: {
            "input": np.random.randint(0, 100, (4, 8)).astype(np.int64),
            "weight": np.random.randn(100, 16).astype(np.float32),
        }
        # embedding_bag is now excluded in numerical_exclusions.yaml (returns tuple)
        self._generators["torch.embedding_bag"] = lambda: {
            "weight": np.random.randn(100, 16).astype(np.float32),
            "indices": np.random.randint(0, 100, (8,)).astype(np.int64),
            "offsets": np.array([0, 3, 6]).astype(np.int64),
        }
        self._generators["torch.nn.functional.embedding_bag"] = lambda: {
            "input": np.random.randint(0, 100, (8,)).astype(np.int64),
            "weight": np.random.randn(100, 16).astype(np.float32),
            "offsets": np.array([0, 3, 6]).astype(np.int64),
        }

        # embedding_renorm_(weight, indices, max_norm, norm_type) - uses positional args
        self._generators["torch.embedding_renorm_"] = lambda: {
            "_positional": [
                np.random.randn(100, 16).astype(np.float32),  # weight
                np.random.randint(0, 100, (8,)).astype(np.int64),  # indices
                1.0,  # max_norm
                2.0,  # norm_type
            ],
        }

        # ctc_loss (uses int: 0=none, 1=mean, 2=sum for torch.* C++ binding)
        self._generators["torch.ctc_loss"] = lambda: {
            "log_probs": np.random.randn(50, 4, 20).astype(np.float32),
            "targets": np.random.randint(1, 20, (4, 10)).astype(np.int64),
            "input_lengths": np.array([50, 50, 50, 50]).astype(np.int64),
            "target_lengths": np.array([10, 10, 10, 10]).astype(np.int64),
            "reduction": 0,  # 0=none
        }
        self._generators["torch.nn.functional.ctc_loss"] = lambda: {
            "log_probs": np.random.randn(50, 4, 20).astype(np.float32),
            "targets": np.random.randint(1, 20, (4, 10)).astype(np.int64),
            "input_lengths": np.array([50, 50, 50, 50]).astype(np.int64),
            "target_lengths": np.array([10, 10, 10, 10]).astype(np.int64),
        }

        # cosine_embedding_loss (uses int: 0=none, 1=mean, 2=sum for torch.* C++ binding)
        self._generators["torch.cosine_embedding_loss"] = lambda: {
            "input1": np.random.randn(4, 8).astype(np.float32),
            "input2": np.random.randn(4, 8).astype(np.float32),
            "target": np.sign(np.random.randn(4)).astype(np.float32),
            "reduction": 0,  # 0=none
        }
        self._generators["torch.nn.functional.cosine_embedding_loss"] = lambda: {
            "input1": np.random.randn(4, 8).astype(np.float32),
            "input2": np.random.randn(4, 8).astype(np.float32),
            "target": np.sign(np.random.randn(4)).astype(np.float32),
            "reduction": "none",
        }

        # hinge_embedding_loss (uses int: 0=none, 1=mean, 2=sum for torch.* C++ binding)
        self._generators["torch.hinge_embedding_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.sign(np.random.randn(4, 8)).astype(np.float32),
            "reduction": 0,  # 0=none
        }
        self._generators["torch.nn.functional.hinge_embedding_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.sign(np.random.randn(4, 8)).astype(np.float32),
            "reduction": "none",
        }

        # kl_div (uses int: 0=none, 1=mean, 2=sum for torch.* C++ binding)
        self._generators["torch.kl_div"] = lambda: {
            "input": np.log(np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1),
            "target": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "reduction": 0,  # 0=none
        }
        self._generators["torch.nn.functional.kl_div"] = lambda: {
            "input": np.log(np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1),
            "target": np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1,
            "reduction": "none",
        }

        # poisson_nll_loss - torch.poisson_nll_loss is a C++ binding requiring positional args
        # Args: input, target, log_input, full, eps, reduction (reduction is an int: 0=none, 1=mean, 2=sum)
        self._generators["torch.poisson_nll_loss"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),  # input
                np.abs(np.random.randn(4, 8).astype(np.float32)),  # target
                True,   # log_input
                False,  # full
                1e-8,   # eps
                1,      # reduction (1 = mean)
            ],
        }
        self._generators["torch.nn.functional.poisson_nll_loss"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "target": np.abs(np.random.randn(4, 8).astype(np.float32)),
            "log_input": True,
        }

        # index_put variants
        self._generators["torch.index_put"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "indices": (np.array([0, 1, 2]).astype(np.int64),),
            "values": np.random.randn(3, 8).astype(np.float32),
        }
        self._generators["torch.index_put_"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "indices": (np.array([0, 1, 2]).astype(np.int64),),
            "values": np.random.randn(3, 8).astype(np.float32),
        }

        # In-place math ops
        for op in ["deg2rad_", "rad2deg_", "gcd_", "lcm_", "ldexp_", "logit_", "relu_",
                   "selu_", "erf_", "erfc_", "rrelu_", "sinc_"]:
            if "gcd" in op or "lcm" in op:
                self._generators[f"torch.{op}"] = lambda: {
                    "input": np.random.randint(1, 100, (4, 8)).astype(np.int64),
                    "other": np.random.randint(1, 100, (4, 8)).astype(np.int64),
                }
            elif "ldexp" in op:
                self._generators[f"torch.{op}"] = lambda: {
                    "input": np.random.randn(4, 8).astype(np.float32),
                    "other": np.random.randint(-5, 5, (4, 8)).astype(np.int32),
                }
            elif "logit" in op:
                self._generators[f"torch.{op}"] = lambda: {
                    "input": np.random.uniform(0.01, 0.99, (4, 8)).astype(np.float32),
                }
            elif "erf" in op:
                self._generators[f"torch.{op}"] = lambda: {
                    "input": np.random.randn(4, 8).astype(np.float32),
                }
            elif "rrelu" in op:
                self._generators[f"torch.{op}"] = lambda: {
                    "input": np.random.randn(4, 8).astype(np.float32),
                    "lower": 0.125,
                    "upper": 0.3333,
                    "training": False,
                }
            else:
                self._generators[f"torch.{op}"] = unary_tensor_input

        # cumulative_trapezoid
        self._generators["torch.cumulative_trapezoid"] = lambda: {
            "y": np.random.randn(4, 8).astype(np.float32),
        }

        # rms_norm
        self._generators["torch.rms_norm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "normalized_shape": [8],
        }
        self._generators["torch.nn.functional.rms_norm"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "normalized_shape": [8],
        }

        # max_pool1d_with_indices
        self._generators["torch.max_pool1d_with_indices"] = lambda: {
            "input": np.random.randn(2, 3, 16).astype(np.float32),
            "kernel_size": 2,
        }

        # norm_except_dim
        self._generators["torch.norm_except_dim"] = lambda: {
            "v": np.random.randn(4, 8).astype(np.float32),
            "pow": 2,
        }

        # nonzero_static
        self._generators["torch.nonzero_static"] = lambda: {
            "input": (np.random.randn(4, 8) > 0).astype(bool),
            "size": 10,
        }

        # slice_inverse - let it fail to show the shape bug
        self._generators["torch.slice_inverse"] = lambda: {
            "input": np.random.randn(4, 8).astype(np.float32),
            "src": np.random.randn(2, 8).astype(np.float32),
            "dim": 0,
            "start": 0,
            "end": 2,
        }

        # affine_grid_generator
        self._generators["torch.affine_grid_generator"] = lambda: {
            "theta": np.random.randn(4, 2, 3).astype(np.float32),
            "size": [4, 1, 8, 8],
            "align_corners": True,
        }
        self._generators["torch.nn.functional.affine_grid"] = lambda: {
            "theta": np.random.randn(4, 2, 3).astype(np.float32),
            "size": [4, 1, 8, 8],
            "align_corners": True,
        }

        # grid samplers - now excluded in numerical_exclusions.yaml
        self._generators["torch.grid_sampler"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "grid": np.random.randn(2, 4, 4, 2).astype(np.float32),
            "interpolation_mode": 0,
            "padding_mode": 0,
            "align_corners": True,
        }
        self._generators["torch.grid_sampler_2d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8).astype(np.float32),
            "grid": np.random.randn(2, 4, 4, 2).astype(np.float32),
            "interpolation_mode": 0,
            "padding_mode": 0,
            "align_corners": True,
        }
        self._generators["torch.grid_sampler_3d"] = lambda: {
            "input": np.random.randn(2, 3, 8, 8, 8).astype(np.float32),
            "grid": np.random.randn(2, 4, 4, 4, 3).astype(np.float32),
            "interpolation_mode": 0,
            "padding_mode": 0,
            "align_corners": True,
        }

        # max_unpool functions
        # For max_unpool, indices must be valid (non-overlapping) indices that would come from max_pool
        # Each pooled position should map to a unique position in the output
        def _gen_max_unpool1d():
            # Input: (N, C, L_pooled), kernel_size=2, stride=2
            # Output will be (N, C, L_pooled * 2)
            # Each input position i can map to output position 2*i or 2*i+1
            N, C, L_pooled = 2, 3, 4
            input_data = np.random.randn(N, C, L_pooled).astype(np.float32)
            # Generate valid indices: for position i, choose either 2*i or 2*i+1
            indices = np.zeros((N, C, L_pooled), dtype=np.int64)
            for n in range(N):
                for c in range(C):
                    for l in range(L_pooled):
                        indices[n, c, l] = 2 * l + np.random.randint(0, 2)
            return {"input": input_data, "indices": indices, "kernel_size": 2}
        self._generators["torch.nn.functional.max_unpool1d"] = _gen_max_unpool1d

        def _gen_max_unpool2d():
            # Input: (N, C, H_pooled, W_pooled), kernel_size=2, stride=2
            # Each position (h, w) can map to (2*h + dh, 2*w + dw) where dh, dw in {0, 1}
            N, C, H_pooled, W_pooled = 2, 3, 2, 2
            H_out, W_out = H_pooled * 2, W_pooled * 2
            input_data = np.random.randn(N, C, H_pooled, W_pooled).astype(np.float32)
            indices = np.zeros((N, C, H_pooled, W_pooled), dtype=np.int64)
            for n in range(N):
                for c in range(C):
                    for h in range(H_pooled):
                        for w in range(W_pooled):
                            dh = np.random.randint(0, 2)
                            dw = np.random.randint(0, 2)
                            out_h = 2 * h + dh
                            out_w = 2 * w + dw
                            # Flatten index: out_h * W_out + out_w
                            indices[n, c, h, w] = out_h * W_out + out_w
            return {"input": input_data, "indices": indices, "kernel_size": 2}
        self._generators["torch.nn.functional.max_unpool2d"] = _gen_max_unpool2d

        def _gen_max_unpool3d():
            # Input: (N, C, D_pooled, H_pooled, W_pooled), kernel_size=2, stride=2
            N, C, D_pooled, H_pooled, W_pooled = 2, 3, 1, 1, 1
            D_out, H_out, W_out = D_pooled * 2, H_pooled * 2, W_pooled * 2
            input_data = np.random.randn(N, C, D_pooled, H_pooled, W_pooled).astype(np.float32)
            indices = np.zeros((N, C, D_pooled, H_pooled, W_pooled), dtype=np.int64)
            for n in range(N):
                for c in range(C):
                    for d in range(D_pooled):
                        for h in range(H_pooled):
                            for w in range(W_pooled):
                                dd = np.random.randint(0, 2)
                                dh = np.random.randint(0, 2)
                                dw = np.random.randint(0, 2)
                                out_d = 2 * d + dd
                                out_h = 2 * h + dh
                                out_w = 2 * w + dw
                                # Flatten index: out_d * H_out * W_out + out_h * W_out + out_w
                                indices[n, c, d, h, w] = out_d * H_out * W_out + out_h * W_out + out_w
            return {"input": input_data, "indices": indices, "kernel_size": 2}
        self._generators["torch.nn.functional.max_unpool3d"] = _gen_max_unpool3d

        # channel_shuffle
        self._generators["torch.nn.functional.native_channel_shuffle"] = lambda: {
            "input": np.random.randn(2, 6, 4, 4).astype(np.float32),
            "groups": 2,
        }

        # conv_tbc - now excluded in numerical_exclusions.yaml
        self._generators["torch.nn.functional.conv_tbc"] = lambda: {
            "input": np.random.randn(16, 2, 8).astype(np.float32),  # T, B, C
            "weight": np.random.randn(3, 8, 16).astype(np.float32),  # kW, C_in, C_out
            "bias": np.random.randn(16).astype(np.float32),
        }

        # broadcast_tensors
        self._generators["torch.broadcast_tensors"] = lambda: {
            "_positional": [
                np.random.randn(4, 1).astype(np.float32),
                np.random.randn(1, 8).astype(np.float32),
            ],
        }

        # polar
        self._generators["torch.polar"] = lambda: {
            "_positional": [
                np.abs(np.random.randn(4, 8).astype(np.float32)),
                np.random.randn(4, 8).astype(np.float32),
            ],
        }

        # ======================================================================
        # orgqr/ormqr linalg helpers - let them fail to show param name mismatch
        # ======================================================================

        # orgqr - let it fail to show parameter name mismatch
        self._generators["torch.orgqr"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
            "input2": np.random.randn(4).astype(np.float32),
        }
        # ormqr - let it fail to show parameter name mismatch
        self._generators["torch.ormqr"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
            "input2": np.random.randn(4).astype(np.float32),
            "input3": np.random.randn(4, 4).astype(np.float32),
        }

        # geqrf
        self._generators["torch.geqrf"] = lambda: {
            "input": np.random.randn(4, 4).astype(np.float32),
        }

        # triangular_solve - returns tuple (solution, A_clone)
        self._generators["torch.triangular_solve"] = lambda: {
            "_positional": [
                np.random.randn(4, 4).astype(np.float32),  # input (b)
                np.tril(np.random.randn(4, 4).astype(np.float32)) + np.eye(4).astype(np.float32),  # A
            ],
        }

        # frombuffer
        self._generators["torch.frombuffer"] = lambda: {
            "buffer": np.random.randn(32).astype(np.float32).tobytes(),
            "dtype": "float32",
        }

        # empty_strided / empty_permuted
        self._generators["torch.empty_strided"] = lambda: {
            "size": (4, 8),
            "stride": (8, 1),
        }
        self._generators["torch.empty_permuted"] = lambda: {
            "size": (4, 8),
            "physical_layout": (1, 0),
        }

        # nn.Module classes - now excluded in numerical_exclusions.yaml (MLX limitation)
        # Provide proper init_kwargs for reference (not used since excluded)
        self._generators["torch.nn.ConvTranspose3d"] = lambda: {
            "in_channels": 3,
            "out_channels": 8,
            "kernel_size": 3,
        }

        # binomial
        self._generators["torch.binomial"] = lambda: {
            "count": np.ones((4, 8)).astype(np.float32) * 10,
            "prob": np.random.uniform(0.1, 0.9, (4, 8)).astype(np.float32),
        }

        # ======================================================================
        # torch.nn.utils functions - tested via NNUtilsSpec in _test_nn_utils_function
        # These are handled specially by the validator, no input generators needed
        # ======================================================================

        # ======================================================================
        # torch.nn.grad functions - gradient computation
        # ======================================================================

        self._generators["torch.nn.grad.conv1d_input"] = lambda: {
            "_positional": [
                (4, 8, 16),  # input_size
                np.random.randn(16, 8, 3).astype(np.float32),  # weight
                np.random.randn(4, 16, 14).astype(np.float32),  # grad_output
            ],
        }
        self._generators["torch.nn.grad.conv1d_weight"] = lambda: {
            "_positional": [
                np.random.randn(4, 8, 16).astype(np.float32),  # input
                (16, 8, 3),  # weight_size
                np.random.randn(4, 16, 14).astype(np.float32),  # grad_output
            ],
        }
        self._generators["torch.nn.grad.conv2d_input"] = lambda: {
            "_positional": [
                (4, 8, 16, 16),  # input_size
                np.random.randn(16, 8, 3, 3).astype(np.float32),  # weight
                np.random.randn(4, 16, 14, 14).astype(np.float32),  # grad_output
            ],
        }
        self._generators["torch.nn.grad.conv2d_weight"] = lambda: {
            "_positional": [
                np.random.randn(4, 8, 16, 16).astype(np.float32),  # input
                (16, 8, 3, 3),  # weight_size
                np.random.randn(4, 16, 14, 14).astype(np.float32),  # grad_output
            ],
        }
        self._generators["torch.nn.grad.conv3d_input"] = lambda: {
            "_positional": [
                (2, 4, 8, 8, 8),  # input_size
                np.random.randn(8, 4, 3, 3, 3).astype(np.float32),  # weight
                np.random.randn(2, 8, 6, 6, 6).astype(np.float32),  # grad_output
            ],
        }
        self._generators["torch.nn.grad.conv3d_weight"] = lambda: {
            "_positional": [
                np.random.randn(2, 4, 8, 8, 8).astype(np.float32),  # input
                (8, 4, 3, 3, 3),  # weight_size
                np.random.randn(2, 8, 6, 6, 6).astype(np.float32),  # grad_output
            ],
        }

        # ======================================================================
        # torch.nn.utils.rnn functions
        # ======================================================================

        self._generators["torch.nn.utils.rnn.pack_padded_sequence"] = lambda: {
            "_positional": [
                np.random.randn(10, 4, 16).astype(np.float32),  # input (T, B, *)
                np.array([10, 8, 6, 4]),  # lengths
            ],
            "batch_first": False,
            "enforce_sorted": True,
        }
        self._generators["torch.nn.utils.rnn.pad_packed_sequence"] = lambda: {
            "_skip": "Requires PackedSequence from pack_padded_sequence",
        }
        self._generators["torch.nn.utils.rnn.pad_sequence"] = lambda: {
            "_positional": [
                [
                    np.random.randn(10, 16).astype(np.float32),
                    np.random.randn(8, 16).astype(np.float32),
                    np.random.randn(6, 16).astype(np.float32),
                ],
            ],
            "batch_first": False,
        }
        self._generators["torch.nn.utils.rnn.unpad_sequence"] = lambda: {
            "_skip": "Requires padded sequence and lengths",
        }
        self._generators["torch.nn.utils.rnn.pack_sequence"] = lambda: {
            "_positional": [
                [
                    np.random.randn(10, 16).astype(np.float32),
                    np.random.randn(8, 16).astype(np.float32),
                    np.random.randn(6, 16).astype(np.float32),
                ],
            ],
            "enforce_sorted": True,
        }
        self._generators["torch.nn.utils.rnn.unpack_sequence"] = lambda: {
            "_skip": "Requires PackedSequence from pack_sequence",
        }
        self._generators["torch.nn.utils.rnn.invert_permutation"] = lambda: {
            "_positional": [
                np.array([2, 0, 3, 1]),  # permutation
            ],
        }

        # ======================================================================
        # torch.nn.utils.parametrizations
        # Now handled via NNUtilsSpec with test_type="parametrization"
        # ======================================================================

        # ======================================================================
        # torch.nn.utils.parametrize
        # Most functions now handled via NNUtilsSpec
        # ======================================================================

        self._generators["torch.nn.utils.parametrize.cached"] = lambda: {
            "_skip": "Context manager - not numerical",
        }

        # ======================================================================
        # torch.nn.utils.stateless
        # Now handled via NNUtilsSpec with test_type="functional_call"
        # ======================================================================

        # ======================================================================
        # torch.nn.attention
        # ======================================================================

        self._generators["torch.nn.attention.sdpa_kernel"] = lambda: {
            "_skip": "Context manager - not numerical",
        }

        # ======================================================================
        # Complex number operations - require complex dtype input
        # ======================================================================

        self._generators["torch.imag"] = lambda: {
            "_positional": [
                np.array([1+2j, 3+4j, 5+6j]).astype(np.complex64),
            ],
        }
        self._generators["torch.angle"] = lambda: {
            "_positional": [
                np.array([1+2j, 3+4j, 5+6j]).astype(np.complex64),
            ],
        }
        self._generators["torch.conj"] = lambda: {
            "_positional": [
                np.array([1+2j, 3+4j, 5+6j]).astype(np.complex64),
            ],
        }
        self._generators["torch.view_as_real"] = lambda: {
            "_positional": [
                np.array([1+2j, 3+4j, 5+6j]).astype(np.complex64),
            ],
        }
        self._generators["torch.view_as_complex"] = lambda: {
            "_positional": [
                np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32),
            ],
        }

        # ======================================================================
        # RNN functions - require weight matrices
        # ======================================================================

        # RNN cell functions (single timestep)
        # Signature: cell(input, hx, w_ih, w_hh, b_ih, b_hh)
        def _rnn_cell_input():
            batch = 4
            input_size = 10
            hidden_size = 20
            return {
                "_positional": [
                    np.random.randn(batch, input_size).astype(np.float32),  # input
                    np.random.randn(batch, hidden_size).astype(np.float32),  # hx
                    np.random.randn(hidden_size, input_size).astype(np.float32),  # w_ih
                    np.random.randn(hidden_size, hidden_size).astype(np.float32),  # w_hh
                    np.random.randn(hidden_size).astype(np.float32),  # b_ih
                    np.random.randn(hidden_size).astype(np.float32),  # b_hh
                ],
            }

        self._generators["torch.rnn_tanh_cell"] = _rnn_cell_input
        self._generators["torch.rnn_relu_cell"] = _rnn_cell_input

        # GRU cell (3x hidden_size for gates)
        def _gru_cell_input():
            batch = 4
            input_size = 10
            hidden_size = 20
            return {
                "_positional": [
                    np.random.randn(batch, input_size).astype(np.float32),  # input
                    np.random.randn(batch, hidden_size).astype(np.float32),  # hx
                    np.random.randn(3 * hidden_size, input_size).astype(np.float32),  # w_ih
                    np.random.randn(3 * hidden_size, hidden_size).astype(np.float32),  # w_hh
                    np.random.randn(3 * hidden_size).astype(np.float32),  # b_ih
                    np.random.randn(3 * hidden_size).astype(np.float32),  # b_hh
                ],
            }

        self._generators["torch.gru_cell"] = _gru_cell_input

        # LSTM cell (4x hidden_size for gates, hx is tuple (h, c))
        def _lstm_cell_input():
            batch = 4
            input_size = 10
            hidden_size = 20
            return {
                "_positional": [
                    np.random.randn(batch, input_size).astype(np.float32),  # input
                    (  # hx tuple (h, c)
                        np.random.randn(batch, hidden_size).astype(np.float32),
                        np.random.randn(batch, hidden_size).astype(np.float32),
                    ),
                    np.random.randn(4 * hidden_size, input_size).astype(np.float32),  # w_ih
                    np.random.randn(4 * hidden_size, hidden_size).astype(np.float32),  # w_hh
                    np.random.randn(4 * hidden_size).astype(np.float32),  # b_ih
                    np.random.randn(4 * hidden_size).astype(np.float32),  # b_hh
                ],
            }

        self._generators["torch.lstm_cell"] = _lstm_cell_input

        # Multi-layer RNN functions (rnn_tanh, rnn_relu)
        # Signature: rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)
        def _rnn_multi_layer_input():
            batch = 4
            seq_len = 8
            input_size = 10
            hidden_size = 20
            num_layers = 1
            has_biases = True

            # params is a tuple of weight/bias tensors
            params = [
                np.random.randn(hidden_size, input_size).astype(np.float32),  # weight_ih
                np.random.randn(hidden_size, hidden_size).astype(np.float32),  # weight_hh
                np.random.randn(hidden_size).astype(np.float32),  # bias_ih
                np.random.randn(hidden_size).astype(np.float32),  # bias_hh
            ]

            return {
                "_positional": [
                    np.random.randn(seq_len, batch, input_size).astype(np.float32),  # input
                    np.random.randn(num_layers, batch, hidden_size).astype(np.float32),  # hx
                    tuple(params),  # params (must be tuple for PyTorch)
                    has_biases,  # has_biases
                    num_layers,  # num_layers
                    0.0,  # dropout
                    False,  # train
                    False,  # bidirectional
                    False,  # batch_first
                ],
            }

        self._generators["torch.rnn_tanh"] = _rnn_multi_layer_input
        self._generators["torch.rnn_relu"] = _rnn_multi_layer_input

        # GRU for packed sequences
        # Signature: gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional)
        def _gru_packed_input():
            batch = 4
            seq_len = 8
            input_size = 10
            hidden_size = 20
            num_layers = 1
            has_biases = True

            # params for GRU (3x hidden_size for gates)
            params = [
                np.random.randn(3 * hidden_size, input_size).astype(np.float32),  # weight_ih
                np.random.randn(3 * hidden_size, hidden_size).astype(np.float32),  # weight_hh
                np.random.randn(3 * hidden_size).astype(np.float32),  # bias_ih
                np.random.randn(3 * hidden_size).astype(np.float32),  # bias_hh
            ]

            return {
                "_positional": [
                    np.random.randn(seq_len * batch, input_size).astype(np.float32),  # data (packed)
                    np.array([batch] * seq_len, dtype=np.int64),  # batch_sizes
                    np.random.randn(num_layers, batch, hidden_size).astype(np.float32),  # hx
                    tuple(params),  # params
                    has_biases,  # has_biases
                    num_layers,  # num_layers
                    0.0,  # dropout
                    False,  # train
                    False,  # bidirectional
                ],
            }

        self._generators["torch.gru"] = _gru_packed_input

        # LSTM for packed sequences
        # Signature: lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional)
        def _lstm_packed_input():
            batch = 4
            seq_len = 8
            input_size = 10
            hidden_size = 20
            num_layers = 1
            has_biases = True

            # params for LSTM (4x hidden_size for gates)
            params = [
                np.random.randn(4 * hidden_size, input_size).astype(np.float32),  # weight_ih
                np.random.randn(4 * hidden_size, hidden_size).astype(np.float32),  # weight_hh
                np.random.randn(4 * hidden_size).astype(np.float32),  # bias_ih
                np.random.randn(4 * hidden_size).astype(np.float32),  # bias_hh
            ]

            return {
                "_positional": [
                    np.random.randn(seq_len * batch, input_size).astype(np.float32),  # data (packed)
                    np.array([batch] * seq_len, dtype=np.int64),  # batch_sizes
                    (  # hx tuple (h0, c0)
                        np.random.randn(num_layers, batch, hidden_size).astype(np.float32),
                        np.random.randn(num_layers, batch, hidden_size).astype(np.float32),
                    ),
                    tuple(params),  # params
                    has_biases,  # has_biases
                    num_layers,  # num_layers
                    0.0,  # dropout
                    False,  # train
                    False,  # bidirectional
                ],
            }

        self._generators["torch.lstm"] = _lstm_packed_input

        # ======================================================================
        # Functions requiring special input from other functions
        # ======================================================================

        def _lu_solve_input():
            """Generate proper lu_solve inputs using numpy's LU decomposition."""
            # Create a well-conditioned square matrix
            A = np.random.randn(4, 4).astype(np.float32)
            A = A @ A.T + 4 * np.eye(4, dtype=np.float32)  # Make positive definite
            b = np.random.randn(4, 2).astype(np.float32)

            # Compute LU factorization using scipy
            try:
                from scipy import linalg
                lu, piv = linalg.lu_factor(A)
                # Convert pivots to 0-indexed
                pivots = np.arange(4, dtype=np.int64)
                for i, p in enumerate(piv):
                    pivots[i], pivots[p] = pivots[p], pivots[i]
            except ImportError:
                # Fallback - use a simple manual decomposition
                lu = A.copy()
                pivots = np.arange(4, dtype=np.int64)

            return {
                "_positional": [
                    b,     # B matrix (right-hand side)
                    lu,    # LU_data
                    pivots.astype(np.int32) + 1,  # LU_pivots (1-indexed for PyTorch)
                ],
            }

        self._generators["torch.lu_solve"] = _lu_solve_input

        def _lu_unpack_input():
            """Generate proper lu_unpack inputs using numpy's LU decomposition."""
            # Create a well-conditioned square matrix
            A = np.random.randn(4, 4).astype(np.float32)
            A = A @ A.T + 4 * np.eye(4, dtype=np.float32)  # Make positive definite

            # Compute LU factorization using scipy
            try:
                from scipy import linalg
                lu, piv = linalg.lu_factor(A)
                # PyTorch expects 1-indexed pivots
                pivots = (piv + 1).astype(np.int32)
            except ImportError:
                # Fallback
                lu = A.copy()
                pivots = np.arange(1, 5, dtype=np.int32)

            return {
                "_positional": [
                    lu,      # LU_data
                    pivots,  # LU_pivots (1-indexed for PyTorch)
                ],
            }

        self._generators["torch.lu_unpack"] = _lu_unpack_input

        # ======================================================================
        # Fix from_numpy - needs actual numpy array, not tensor
        # ======================================================================

        self._generators["torch.from_numpy"] = lambda: {
            "_positional": [
                np.random.randn(4, 8).astype(np.float32),
            ],
            "_raw_numpy": True,  # Signal to not convert to tensor
        }

        # ======================================================================
        # Fix frombuffer - needs proper bytes buffer
        # ======================================================================

        self._generators["torch.frombuffer"] = lambda: {
            "_positional": [
                np.random.randn(16).astype(np.float32).tobytes(),  # raw bytes buffer
            ],
            "dtype": np.float32,  # Use numpy dtype, validator converts to framework dtype
            "_raw_buffer": True,  # Signal to handle buffer specially
        }

        # ======================================================================
        # convolution - low-level interface with many required args
        # ======================================================================

        self._generators["torch.convolution"] = lambda: {
            "_positional": [
                np.random.randn(2, 3, 16, 16).astype(np.float32),  # input
                np.random.randn(8, 3, 3, 3).astype(np.float32),  # weight
                np.random.randn(8).astype(np.float32),  # bias
                [1, 1],  # stride
                [1, 1],  # padding
                [1, 1],  # dilation
                False,  # transposed
                [0, 0],  # output_padding
                1,  # groups
            ],
        }

        # ======================================================================
        # embedding_bag - fix parameter name and ensure proper dtypes
        # ======================================================================

        self._generators["torch.embedding_bag"] = lambda: {
            "_positional": [
                np.random.randn(100, 16).astype(np.float32),  # weight (num_embeddings, embedding_dim) - must be 2D
                np.array([0, 5, 10, 15, 20], dtype=np.int64),  # input (indices) - must be 1D int64
                np.array([0, 2, 4], dtype=np.int64),  # offsets - must be 1D int64
            ],
        }

    def register(self, api_name: str, generator: Callable[[], Dict[str, Any]]):
        """Register a generator for a specific API."""
        self._generators[api_name] = generator

    def register_pattern(self, pattern: str, generator: Callable[[], Dict[str, Any]]):
        """Register a generator for APIs matching a regex pattern."""
        self._pattern_generators.append((re.compile(pattern), generator))

    def get_generator(
        self, module: str, api_name: str
    ) -> Optional[Callable[[], Dict[str, Any]]]:
        """Get the input generator for an API."""
        full_name = f"{module}.{api_name}"

        # Check exact match first
        if full_name in self._generators:
            return self._generators[full_name]

        # Check pattern matches
        for pattern, generator in self._pattern_generators:
            if pattern.match(full_name):
                return generator

        # Check if it's an nn.Module
        if api_name in NN_MODULE_SPECS:
            return lambda: {"_module_spec": NN_MODULE_SPECS[api_name]}

        # Check if it's an optimizer
        if api_name in OPTIMIZER_SPECS:
            return lambda: {"_optimizer_spec": OPTIMIZER_SPECS[api_name]}

        return None

    def get_module_spec(self, api_name: str) -> Optional[ModuleSpec]:
        """Get the ModuleSpec for an nn.Module class."""
        return NN_MODULE_SPECS.get(api_name)

    def get_optimizer_spec(self, api_name: str) -> Optional[OptimizerSpec]:
        """Get the OptimizerSpec for an optimizer class."""
        return OPTIMIZER_SPECS.get(api_name)

    def get_lr_scheduler_spec(self, api_name: str) -> Optional[LRSchedulerSpec]:
        """Get the LRSchedulerSpec for an LR scheduler class."""
        return LR_SCHEDULER_SPECS.get(api_name)

    def get_nn_utils_spec(self, api_name: str) -> Optional[NNUtilsSpec]:
        """Get the NNUtilsSpec for a torch.nn.utils function."""
        return NN_UTILS_SPECS.get(api_name)


# Global registry instance
_REGISTRY: Optional[InputGeneratorRegistry] = None


def get_input_registry() -> InputGeneratorRegistry:
    """Get the global input generator registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = InputGeneratorRegistry()
    return _REGISTRY
