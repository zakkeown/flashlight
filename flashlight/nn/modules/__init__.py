"""
Neural Network Modules

PyTorch-compatible torch.nn.modules subpackage that re-exports all layer classes.
This mirrors the torch.nn.modules structure for compatibility.
"""

# Container - deprecated but still exported
from ..containers import Container, ModuleDict, ModuleList, ParameterDict, ParameterList, Sequential

# Activation layers
from ..layers.activation import (
    CELU,
    ELU,
    GELU,
    GLU,
    SELU,
    Bilinear,
    ChannelShuffle,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    Sigmoid,
    SiLU,
    Softmax,
    Softmax2d,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
)

# Attention and Transformer layers
from ..layers.attention import MultiheadAttention, scaled_dot_product_attention

# Convolution layers
from ..layers.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)

# Distance layers
from ..layers.distance import CosineSimilarity, PairwiseDistance

# Dropout layers
from ..layers.dropout import (
    AlphaDropout,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
)

# Embedding layers
from ..layers.embedding import Embedding, EmbeddingBag

# Fold/Unfold
from ..layers.fold import Fold, Unfold

# Lazy modules
from ..layers.lazy import (
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d,
    LazyInstanceNorm1d,
    LazyInstanceNorm2d,
    LazyInstanceNorm3d,
    LazyLinear,
)

# Linear layers
from ..layers.linear import Linear

# Normalization layers
from ..layers.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    CrossMapLRN2d,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
    LocalResponseNorm,
    RMSNorm,
    SyncBatchNorm,
)

# Padding layers
from ..layers.padding import (
    CircularPad1d,
    CircularPad2d,
    CircularPad3d,
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    ReflectionPad1d,
    ReflectionPad2d,
    ReflectionPad3d,
    ReplicationPad1d,
    ReplicationPad2d,
    ReplicationPad3d,
    ZeroPad1d,
    ZeroPad2d,
    ZeroPad3d,
)

# Pooling layers
from ..layers.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    FractionalMaxPool2d,
    FractionalMaxPool3d,
    LPPool1d,
    LPPool2d,
    LPPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MaxUnpool1d,
    MaxUnpool2d,
    MaxUnpool3d,
)

# RNN layers
from ..layers.rnn import (
    GRU,
    LSTM,
    RNN,
    GRUCell,
    LSTMCell,
    RNNBase,
    RNNCell,
    RNNCellBase,
)

# Shape layers
from ..layers.shape import Flatten, Identity, Unflatten
from ..layers.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

# Upsampling layers
from ..layers.upsample import (
    PixelShuffle,
    PixelUnshuffle,
    Upsample,
    UpsamplingBilinear2d,
    UpsamplingNearest2d,
)

# Loss functions (also available as layers)
from ..losses import (
    AdaptiveLogSoftmaxWithLoss,
    BCELoss,
    BCEWithLogitsLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    CTCLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    NLLLoss2d,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)

# Re-export everything from nn layers
from ..module import Module
from ..parameter import Buffer, Parameter, UninitializedBuffer, UninitializedParameter

__all__ = [
    # Core
    "Module",
    "Parameter",
    "Buffer",
    "UninitializedParameter",
    "UninitializedBuffer",
    # Containers
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
    "Container",
    # Linear
    "Linear",
    # Convolution
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # Pooling
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    # Normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "RMSNorm",
    "LocalResponseNorm",
    "SyncBatchNorm",
    "CrossMapLRN2d",
    # Activation
    "ReLU",
    "LeakyReLU",
    "ELU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "SiLU",
    "Softmax",
    "LogSoftmax",
    "ReLU6",
    "SELU",
    "CELU",
    "Hardtanh",
    "Hardsigmoid",
    "Hardswish",
    "Hardshrink",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanhshrink",
    "LogSigmoid",
    "Softmin",
    "Mish",
    "GLU",
    "PReLU",
    "Threshold",
    "RReLU",
    "Softmax2d",
    "ChannelShuffle",
    "Bilinear",
    # Dropout
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
    # Padding
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    # Shape
    "Flatten",
    "Unflatten",
    "Identity",
    # RNN
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    "LSTM",
    "GRU",
    "RNNBase",
    "RNNCellBase",
    # Embedding
    "Embedding",
    "EmbeddingBag",
    # Attention/Transformer
    "MultiheadAttention",
    "scaled_dot_product_attention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
    # Upsampling
    "Upsample",
    "UpsamplingNearest2d",
    "UpsamplingBilinear2d",
    "PixelShuffle",
    "PixelUnshuffle",
    # Distance
    "CosineSimilarity",
    "PairwiseDistance",
    # Fold
    "Fold",
    "Unfold",
    # Lazy modules
    "LazyLinear",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    # Losses
    "MSELoss",
    "L1Loss",
    "NLLLoss",
    "NLLLoss2d",
    "CrossEntropyLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "KLDivLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "CosineEmbeddingLoss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "MultiLabelMarginLoss",
    "MultiMarginLoss",
    "MultiLabelSoftMarginLoss",
    "TripletMarginWithDistanceLoss",
    "CTCLoss",
    "AdaptiveLogSoftmaxWithLoss",
]
