"""
Neural Networks module

Implements nn.Module system and layers for building neural networks.
"""

# Submodules that should be accessible as nn.xxx
from . import functional
from . import functional as F  # PyTorch-style alias
from . import init
from . import parameter
from . import modules
from . import common_types
from . import grad
from . import attention
from . import qat
from . import quantizable
from . import utils

# Core classes
from .module import Module
from .parameter import Parameter, Buffer, UninitializedParameter, UninitializedBuffer

# Container modules
from .containers import Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict, Container

# Linear layers
from .layers.linear import Linear

# Dropout layers
from .layers.dropout import Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout, FeatureAlphaDropout

# Convolution layers
from .layers.conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

# Pooling layers
from .layers.pooling import (
    MaxPool1d, MaxPool2d, MaxPool3d,
    AvgPool1d, AvgPool2d, AvgPool3d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d,
    AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d,
    LPPool1d, LPPool2d, LPPool3d,
    FractionalMaxPool2d, FractionalMaxPool3d,
    MaxUnpool1d, MaxUnpool2d, MaxUnpool3d,
)

# Normalization layers
from .layers.normalization import (
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    LayerNorm, GroupNorm,
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
    RMSNorm, LocalResponseNorm, SyncBatchNorm,
    CrossMapLRN2d,
)

# Activation layers
from .layers.activation import (
    ReLU, LeakyReLU, ELU,
    Sigmoid, Tanh,
    GELU, SiLU,
    Softmax, LogSoftmax,
    ReLU6, SELU, CELU,
    Hardtanh, Hardsigmoid, Hardswish, Hardshrink,
    Softplus, Softshrink, Softsign, Tanhshrink,
    LogSigmoid, Softmin, Mish, GLU, PReLU, Threshold,
    RReLU, Softmax2d, ChannelShuffle, Bilinear,
)

# Loss functions
from .losses import (
    MSELoss, L1Loss,
    NLLLoss, NLLLoss2d, CrossEntropyLoss,
    BCELoss, BCEWithLogitsLoss,
    SmoothL1Loss, HuberLoss,
    KLDivLoss, MarginRankingLoss, HingeEmbeddingLoss,
    CosineEmbeddingLoss, SoftMarginLoss,
    TripletMarginLoss, PoissonNLLLoss,
    GaussianNLLLoss, MultiLabelMarginLoss, MultiMarginLoss,
    MultiLabelSoftMarginLoss, TripletMarginWithDistanceLoss,
    CTCLoss, AdaptiveLogSoftmaxWithLoss,
)

# Shape layers
from .layers.shape import Flatten, Unflatten, Identity

# Padding layers
from .layers.padding import (
    ZeroPad1d, ZeroPad2d, ZeroPad3d,
    ConstantPad1d, ConstantPad2d, ConstantPad3d,
    ReflectionPad1d, ReflectionPad2d, ReflectionPad3d,
    ReplicationPad1d, ReplicationPad2d, ReplicationPad3d,
    CircularPad1d, CircularPad2d, CircularPad3d,
)

# RNN layers
from .layers.rnn import (
    RNNCellBase, RNNBase,
    RNNCell, LSTMCell, GRUCell,
    RNN, LSTM, GRU,
)

# Embedding layers
from .layers.embedding import Embedding, EmbeddingBag

# Attention layers
from .layers.attention import MultiheadAttention, scaled_dot_product_attention

# Upsampling layers
from .layers.upsample import (
    Upsample, UpsamplingNearest2d, UpsamplingBilinear2d,
    PixelShuffle, PixelUnshuffle,
)

# Distance layers
from .layers.distance import CosineSimilarity, PairwiseDistance

# Fold/Unfold
from .layers.fold import Fold, Unfold

# Transformer layers
from .layers.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
)

# Utility function
def factory_kwargs(kwargs):
    """
    Filter out device and dtype from kwargs.

    This is a compatibility shim - MLX uses unified memory so device is ignored.
    """
    device = kwargs.pop('device', None)
    dtype = kwargs.pop('dtype', None)
    return {'device': device, 'dtype': dtype}


__all__ = [
    # Submodules
    'functional', 'F', 'init', 'parameter', 'modules', 'common_types', 'grad',
    'attention', 'qat', 'quantizable',
    # Utility
    'factory_kwargs',
    # Core
    'Module',
    'Parameter', 'Buffer', 'UninitializedParameter', 'UninitializedBuffer',
    # Containers
    'Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict', 'Container',
    # Linear
    'Linear',
    # Dropout
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout',
    # Convolution
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    # Pooling
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
    'LPPool1d', 'LPPool2d', 'LPPool3d',
    'FractionalMaxPool2d', 'FractionalMaxPool3d',
    'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d',
    # Normalization
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'LayerNorm', 'GroupNorm',
    'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'RMSNorm', 'LocalResponseNorm', 'SyncBatchNorm',
    'CrossMapLRN2d',
    # Activations
    'ReLU', 'LeakyReLU', 'ELU',
    'Sigmoid', 'Tanh',
    'GELU', 'SiLU',
    'Softmax', 'LogSoftmax',
    'ReLU6', 'SELU', 'CELU',
    'Hardtanh', 'Hardsigmoid', 'Hardswish', 'Hardshrink',
    'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink',
    'LogSigmoid', 'Softmin', 'Mish', 'GLU', 'PReLU', 'Threshold',
    'RReLU', 'Softmax2d', 'ChannelShuffle', 'Bilinear',
    # Losses
    'MSELoss', 'L1Loss',
    'NLLLoss', 'NLLLoss2d', 'CrossEntropyLoss',
    'BCELoss', 'BCEWithLogitsLoss',
    'SmoothL1Loss', 'HuberLoss',
    'KLDivLoss', 'MarginRankingLoss', 'HingeEmbeddingLoss',
    'CosineEmbeddingLoss', 'SoftMarginLoss',
    'TripletMarginLoss', 'PoissonNLLLoss',
    'GaussianNLLLoss', 'MultiLabelMarginLoss', 'MultiMarginLoss',
    'MultiLabelSoftMarginLoss', 'TripletMarginWithDistanceLoss',
    'CTCLoss', 'AdaptiveLogSoftmaxWithLoss',
    # Shape
    'Flatten', 'Unflatten', 'Identity',
    # Padding
    'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
    'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d',
    'CircularPad1d', 'CircularPad2d', 'CircularPad3d',
    # RNN
    'RNNCellBase', 'RNNBase',
    'RNNCell', 'LSTMCell', 'GRUCell',
    'RNN', 'LSTM', 'GRU',
    # Embedding
    'Embedding', 'EmbeddingBag',
    # Upsampling
    'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d',
    'PixelShuffle', 'PixelUnshuffle',
    # Distance
    'CosineSimilarity', 'PairwiseDistance',
    # Fold/Unfold
    'Fold', 'Unfold',
    # Attention & Transformer
    'MultiheadAttention', 'scaled_dot_product_attention',
    'TransformerEncoderLayer', 'TransformerDecoderLayer',
    'TransformerEncoder', 'TransformerDecoder',
    'Transformer',
]
